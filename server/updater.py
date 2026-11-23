"""Extension live-reload build hashing and safe git-based auto-update management."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable

from server.schemas import UpdateApplyResponse, UpdateStatusResponse

logger = logging.getLogger(__name__)

_SKIP_FILENAMES = {".DS_Store", "Thumbs.db"}
_SKIP_SUFFIXES = (".tmp", ".bak")
_LOCAL_GIT_TIMEOUT_SEC = 5.0
_NETWORK_GIT_TIMEOUT_SEC = 15.0
_RESTART_DELAY_SEC = 1.5
RESTART_EXIT_CODE = 42


def _should_skip_extension_path(rel_path: Path) -> bool:
    if "__pycache__" in rel_path.parts:
        return True
    name = rel_path.name
    if name in _SKIP_FILENAMES:
        return True
    return name.endswith(_SKIP_SUFFIXES)


def compute_extension_build_id(ext_dir: Path) -> str:
    """Compute a deterministic 16-char lowercase hex SHA-256 hash over extension files."""
    if not ext_dir.exists() or not ext_dir.is_dir():
        return ""

    entries: list[tuple[str, Path]] = []
    for path in ext_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(ext_dir)
        if _should_skip_extension_path(rel_path):
            continue
        entries.append((rel_path.as_posix(), path))

    entries.sort(key=lambda item: item[0])

    digest = hashlib.sha256()
    for rel_posix, file_path in entries:
        data = file_path.read_bytes()
        digest.update(rel_posix.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(data)
    return digest.hexdigest()[:16]


def read_extension_version(ext_dir: Path) -> str:
    """Read extension version from extension/manifest.json with a safe fallback."""
    manifest_path = ext_dir / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            version = raw.get("version")
            if isinstance(version, str) and version.strip():
                return version.strip()
    except Exception:
        pass
    return "1.0.0"


@dataclass
class _GitSnapshot:
    is_git_repo: bool = False
    has_remote: bool = False
    remote_name: str = ""
    remote_url: str = ""
    branch: str = ""
    current_commit: str = ""
    working_tree_clean: bool = True


class UpdateManager:
    """Thread-safe manager for extension content hashing and git fast-forward updates."""

    def __init__(
        self,
        repo_root: Path,
        extension_dir: Path | None = None,
        auto_update_enabled: bool = True,
        check_interval_sec: int = 1800,
        on_restart_requested: Callable[[], None] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.extension_dir = (
            Path(extension_dir).resolve()
            if extension_dir is not None
            else (self.repo_root / "extension")
        )
        self.auto_update_enabled = bool(auto_update_enabled)
        self.check_interval_sec = max(1, int(check_interval_sec))
        self._on_restart_requested = on_restart_requested

        self._op_lock = threading.RLock()
        self._state_lock = threading.Lock()

        self._remote_commit: str | None = None
        self._commits_behind: int = 0
        self._update_available: bool = False
        self._last_checked: float | None = None
        self._last_error: str | None = None
        self._updating: bool = False
        self._restart_required: bool = False
        self._restart_scheduled: bool = False
        self._git_snap_cache: tuple[float, _GitSnapshot] | None = None

    def _run_git(self, args: list[str], timeout: float) -> tuple[int, str, str]:
        if not (self.repo_root / ".git").exists():
            return (128, "", "Not a git repository")

        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(self.repo_root),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                creationflags=creationflags,
            )
            return (proc.returncode, proc.stdout.strip(), proc.stderr.strip())
        except FileNotFoundError:
            return (127, "", "git executable not found")
        except subprocess.TimeoutExpired:
            return (124, "", f"git {' '.join(args[:2])} timed out after {timeout:.0f}s")
        except OSError as exc:
            return (1, "", str(exc))

    def _inspect_git_local(self) -> _GitSnapshot:
        if not (self.repo_root / ".git").exists():
            return _GitSnapshot()

        rc_remotes, out_remotes, _ = self._run_git(["remote"], timeout=_LOCAL_GIT_TIMEOUT_SEC)
        if rc_remotes == 127:
            return _GitSnapshot()

        remotes = [line.strip() for line in out_remotes.splitlines() if line.strip()] if rc_remotes == 0 else []
        remote_name = "origin" if "origin" in remotes else (remotes[0] if remotes else "")
        remote_url = ""
        if remote_name:
            rc_url, out_url, _ = self._run_git(
                ["remote", "get-url", remote_name],
                timeout=_LOCAL_GIT_TIMEOUT_SEC,
            )
            if rc_url == 0:
                remote_url = out_url

        rc_branch, out_branch, _ = self._run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            timeout=_LOCAL_GIT_TIMEOUT_SEC,
        )
        branch = out_branch if rc_branch == 0 else ""

        rc_commit, out_commit, _ = self._run_git(
            ["rev-parse", "--short", "HEAD"],
            timeout=_LOCAL_GIT_TIMEOUT_SEC,
        )
        current_commit = out_commit if rc_commit == 0 else ""

        rc_status, out_status, _ = self._run_git(
            ["status", "--porcelain"],
            timeout=_LOCAL_GIT_TIMEOUT_SEC,
        )
        working_tree_clean = (rc_status == 0 and out_status == "")

        return _GitSnapshot(
            is_git_repo=True,
            has_remote=bool(remote_name),
            remote_name=remote_name,
            remote_url=remote_url,
            branch=branch,
            current_commit=current_commit,
            working_tree_clean=working_tree_clean,
        )

    def _resolve_upstream_ref(self, remote_name: str, branch: str) -> str | None:
        rc, out, _ = self._run_git(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            timeout=_LOCAL_GIT_TIMEOUT_SEC,
        )
        if rc == 0 and out:
            return out

        if remote_name and branch and branch != "HEAD":
            candidate = f"{remote_name}/{branch}"
            rc_verify, _, _ = self._run_git(
                ["rev-parse", "--verify", candidate],
                timeout=_LOCAL_GIT_TIMEOUT_SEC,
            )
            if rc_verify == 0:
                return candidate
        return None

    def _get_cached_git_snapshot(self, max_age_sec: float = 5.0) -> _GitSnapshot:
        now = time.monotonic()
        with self._state_lock:
            if self._git_snap_cache is not None and (now - self._git_snap_cache[0]) < max_age_sec:
                return self._git_snap_cache[1]
        snap = self._inspect_git_local()
        with self._state_lock:
            self._git_snap_cache = (time.monotonic(), snap)
        return snap

    def _invalidate_git_snapshot(self) -> None:
        with self._state_lock:
            self._git_snap_cache = None

    def detect_has_remote(self) -> bool:
        """Return True if repo_root is a git repository with a configured remote."""
        return self._get_cached_git_snapshot().has_remote

    def get_health_fields(self) -> dict[str, object]:
        """Return extension version/build ID and cached remote update flags for /health."""
        snap = self._get_cached_git_snapshot()
        ext_version = read_extension_version(self.extension_dir)
        ext_build_id = compute_extension_build_id(self.extension_dir)
        with self._state_lock:
            update_available = self._update_available if snap.has_remote else False
        return {
            "extension_version": ext_version,
            "extension_build_id": ext_build_id,
            "has_remote": snap.has_remote,
            "update_available": update_available,
        }

    def get_status(self) -> UpdateStatusResponse:
        """Return live extension build ID, local git metadata, and cached remote update state."""
        self._invalidate_git_snapshot()
        snap = self._inspect_git_local()
        ext_version = read_extension_version(self.extension_dir)
        ext_build_id = compute_extension_build_id(self.extension_dir)

        with self._state_lock:
            remote_commit = self._remote_commit if snap.has_remote else None
            commits_behind = self._commits_behind if snap.has_remote else 0
            update_available = self._update_available if snap.has_remote else False
            last_checked = self._last_checked
            last_error = self._last_error
            updating = self._updating
            restart_required = self._restart_required

        return UpdateStatusResponse(
            extension_version=ext_version,
            extension_build_id=ext_build_id,
            has_remote=snap.has_remote,
            remote_url=snap.remote_url,
            branch=snap.branch,
            current_commit=snap.current_commit,
            remote_commit=remote_commit,
            commits_behind=commits_behind,
            update_available=update_available,
            auto_update_enabled=self.auto_update_enabled,
            working_tree_clean=snap.working_tree_clean,
            last_checked=last_checked,
            last_error=last_error,
            updating=updating,
            restart_required=restart_required,
        )

    def check_for_updates(self, auto_apply: bool = False) -> UpdateStatusResponse:
        """Fetch remote commits if configured and optionally apply fast-forward updates."""
        with self._op_lock:
            self._invalidate_git_snapshot()
            snap = self._inspect_git_local()
            if not snap.has_remote:
                with self._state_lock:
                    self._remote_commit = None
                    self._commits_behind = 0
                    self._update_available = False
                    self._last_checked = time.time()
                    self._last_error = None
                return self.get_status()

            rc_fetch, _, err_fetch = self._run_git(
                ["fetch", snap.remote_name],
                timeout=_NETWORK_GIT_TIMEOUT_SEC,
            )
            if rc_fetch != 0:
                with self._state_lock:
                    self._last_checked = time.time()
                    self._last_error = err_fetch or f"Failed to fetch from {snap.remote_name}"
                return self.get_status()

            upstream = self._resolve_upstream_ref(snap.remote_name, snap.branch)
            if upstream is None:
                with self._state_lock:
                    self._remote_commit = None
                    self._commits_behind = 0
                    self._update_available = False
                    self._last_checked = time.time()
                    self._last_error = f"No upstream tracking branch found for {snap.branch or 'HEAD'}"
                return self.get_status()

            rc_remote_sha, remote_short, err_remote_sha = self._run_git(
                ["rev-parse", "--short", upstream],
                timeout=_LOCAL_GIT_TIMEOUT_SEC,
            )
            rc_behind, behind_out, err_behind = self._run_git(
                ["rev-list", "--count", f"HEAD..{upstream}"],
                timeout=_LOCAL_GIT_TIMEOUT_SEC,
            )
            if rc_remote_sha != 0 or rc_behind != 0:
                with self._state_lock:
                    self._last_checked = time.time()
                    self._last_error = err_remote_sha or err_behind or "Failed to compare commits"
                return self.get_status()

            try:
                commits_behind = max(0, int(behind_out))
            except ValueError:
                commits_behind = 0

            update_available = commits_behind > 0
            with self._state_lock:
                self._remote_commit = remote_short
                self._commits_behind = commits_behind
                self._update_available = update_available
                self._last_checked = time.time()
                self._last_error = None

            if auto_apply and update_available and snap.working_tree_clean:
                self.apply_update()

            return self.get_status()

    def _schedule_restart_once(self) -> bool:
        if os.environ.get("MLT_AUTO_RESTART") != "1":
            return False

        with self._state_lock:
            if self._restart_scheduled:
                return True
            self._restart_scheduled = True

        def _do_restart() -> None:
            if self._on_restart_requested is not None:
                try:
                    self._on_restart_requested()
                except Exception:
                    logger.exception("Error during pre-restart cleanup")
            os._exit(RESTART_EXIT_CODE)

        timer = threading.Timer(_RESTART_DELAY_SEC, _do_restart)
        timer.daemon = True
        timer.start()
        return True

    def apply_update(self) -> UpdateApplyResponse:
        """Apply a safe fast-forward git pull when a remote exists and tree is clean."""
        with self._op_lock:
            self._invalidate_git_snapshot()
            snap = self._inspect_git_local()
            if not snap.has_remote:
                return UpdateApplyResponse(
                    ok=False,
                    updated=False,
                    message="No git remote is configured for automatic updates.",
                    extension_changed=False,
                    server_changed=False,
                    restart_scheduled=False,
                    status=self.get_status(),
                )

            if not snap.working_tree_clean:
                return UpdateApplyResponse(
                    ok=False,
                    updated=False,
                    message="Local repository has uncommitted changes; commit or stash them before updating.",
                    extension_changed=False,
                    server_changed=False,
                    restart_scheduled=False,
                    status=self.get_status(),
                )

            with self._state_lock:
                self._updating = True

            try:
                rc_old, old_head, err_old = self._run_git(
                    ["rev-parse", "HEAD"],
                    timeout=_LOCAL_GIT_TIMEOUT_SEC,
                )
                if rc_old != 0 or not old_head:
                    with self._state_lock:
                        self._last_error = err_old or "Unable to resolve current HEAD commit"
                    return UpdateApplyResponse(
                        ok=False,
                        updated=False,
                        message=f"Could not read current HEAD commit: {err_old or 'unknown error'}",
                        extension_changed=False,
                        server_changed=False,
                        restart_scheduled=False,
                        status=self.get_status(),
                    )

                rc_upstream, _, _ = self._run_git(
                    ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
                    timeout=_LOCAL_GIT_TIMEOUT_SEC,
                )
                if rc_upstream == 0:
                    pull_args = ["pull", "--ff-only"]
                elif snap.branch and snap.branch != "HEAD":
                    pull_args = ["pull", "--ff-only", snap.remote_name, snap.branch]
                else:
                    pull_args = ["pull", "--ff-only"]

                rc_pull, _, err_pull = self._run_git(
                    pull_args,
                    timeout=_NETWORK_GIT_TIMEOUT_SEC,
                )
                if rc_pull != 0:
                    with self._state_lock:
                        self._last_checked = time.time()
                        self._last_error = err_pull or "git pull --ff-only failed"
                    return UpdateApplyResponse(
                        ok=False,
                        updated=False,
                        message=f"Fast-forward update failed: {err_pull or 'diverged history or network error'}",
                        extension_changed=False,
                        server_changed=False,
                        restart_scheduled=False,
                        status=self.get_status(),
                    )

                rc_new, new_head, _ = self._run_git(
                    ["rev-parse", "HEAD"],
                    timeout=_LOCAL_GIT_TIMEOUT_SEC,
                )
                rc_short, new_short, _ = self._run_git(
                    ["rev-parse", "--short", "HEAD"],
                    timeout=_LOCAL_GIT_TIMEOUT_SEC,
                )
                short_sha = new_short if rc_short == 0 else new_head[:7]

                if not new_head or new_head == old_head:
                    with self._state_lock:
                        self._remote_commit = short_sha or self._remote_commit
                        self._commits_behind = 0
                        self._update_available = False
                        self._last_checked = time.time()
                        self._last_error = None
                    return UpdateApplyResponse(
                        ok=True,
                        updated=False,
                        message="Already up to date.",
                        extension_changed=False,
                        server_changed=False,
                        restart_scheduled=False,
                        status=self.get_status(),
                    )

                rc_diff, diff_out, _ = self._run_git(
                    ["diff", "--name-only", old_head, new_head],
                    timeout=_LOCAL_GIT_TIMEOUT_SEC,
                )
                changed_files = (
                    [line.strip().replace("\\", "/") for line in diff_out.splitlines() if line.strip()]
                    if rc_diff == 0
                    else []
                )
                extension_changed = any(
                    p == "extension" or p.startswith("extension/") for p in changed_files
                )
                server_changed = any(
                    p.startswith("server/") or p in ("requirements.txt", "config.json")
                    for p in changed_files
                )

                with self._state_lock:
                    self._remote_commit = short_sha
                    self._commits_behind = 0
                    self._update_available = False
                    self._last_checked = time.time()
                    self._last_error = None
                    if server_changed:
                        self._restart_required = True

                restart_scheduled = self._schedule_restart_once() if server_changed else False
                msg = (
                    "Update applied. Server restart scheduled to load updated backend code."
                    if restart_scheduled
                    else (
                        "Update applied. Please restart the server to load backend changes."
                        if server_changed
                        else "Update applied successfully."
                    )
                )

                return UpdateApplyResponse(
                    ok=True,
                    updated=True,
                    message=msg,
                    extension_changed=extension_changed,
                    server_changed=server_changed,
                    restart_scheduled=restart_scheduled,
                    status=self.get_status(),
                )
            finally:
                with self._state_lock:
                    self._updating = False

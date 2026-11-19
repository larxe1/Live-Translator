"""Silent, detached supervisor for the Manga Live Translator local server.

Mirrors `Start Manga Translator.bat` without any console window: pins all
caches to the repo, optionally fast-forwards a clean git tree, runs
`python -m server.app`, and restarts only on exit code 42 (update restart).

Never writes to stdout on its own behalf except through the log file the
parent redirected it to.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

RESTART_EXIT_CODE = 42
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

REPO_ROOT = Path(__file__).resolve().parent.parent


def cache_dir(root: Path = REPO_ROOT) -> Path:
    return Path(root) / "cache"


def pid_file(root: Path = REPO_ROOT) -> Path:
    return cache_dir(root) / "server.pid"


def load_config(root: Path = REPO_ROOT) -> dict:
    cfg = {"host": DEFAULT_HOST, "port": DEFAULT_PORT}
    try:
        data = json.loads((Path(root) / "config.json").read_text(encoding="utf-8"))
        if isinstance(data, dict):
            host = data.get("host")
            port = data.get("port")
            if isinstance(host, str) and host:
                cfg["host"] = host
            if isinstance(port, int) and 0 < port < 65536:
                cfg["port"] = port
    except Exception:
        pass
    return cfg


def health_host(cfg: dict) -> str:
    """Address to use for local HTTP calls (0.0.0.0 is not connectable)."""
    host = cfg.get("host", DEFAULT_HOST)
    return DEFAULT_HOST if host in ("0.0.0.0", "", "::") else host


def base_url(cfg: dict) -> str:
    return f"http://{health_host(cfg)}:{cfg.get('port', DEFAULT_PORT)}"


def fetch_health(cfg: dict, timeout: float = 1.5) -> Optional[dict]:
    """Return the parsed /health JSON, {} if reachable but not JSON, None if down."""
    try:
        with urllib.request.urlopen(base_url(cfg) + "/health", timeout=timeout) as resp:
            if resp.status != 200:
                return None
            try:
                data = json.loads(resp.read().decode("utf-8", "replace"))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    """Cross-platform liveness check. Never uses os.kill on Windows."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def read_pid(root: Path = REPO_ROOT) -> Optional[int]:
    try:
        text = pid_file(root).read_text(encoding="utf-8").strip()
        return int(text) if text else None
    except Exception:
        return None


def live_supervisor_pid(root: Path = REPO_ROOT) -> Optional[int]:
    pid = read_pid(root)
    if pid is not None and pid_alive(pid):
        return pid
    return None


def remove_pid_file(root: Path = REPO_ROOT, only_if_pid: Optional[int] = None) -> None:
    try:
        if only_if_pid is not None and read_pid(root) != only_if_pid:
            return
        pid_file(root).unlink()
    except Exception:
        pass


def acquire_pid_file(root: Path = REPO_ROOT, own_pid: Optional[int] = None) -> bool:
    """Atomically claim cache/server.pid. False if another live supervisor owns it."""
    own_pid = own_pid or os.getpid()
    path = pid_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = read_pid(root)
            if existing == own_pid:
                return True
            if existing is not None and pid_alive(existing):
                return False
            remove_pid_file(root)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(own_pid))
        return True
    return False


def build_env(root: Path = REPO_ROOT, base: Optional[dict] = None) -> dict:
    """Environment matching Start Manga Translator.bat (all caches repo-local)."""
    env = dict(os.environ if base is None else base)
    cache = cache_dir(root)
    mapping = {
        "TEMP": cache / "tmp",
        "TMP": cache / "tmp",
        "PIP_CACHE_DIR": cache / "pip",
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "EASYOCR_MODULE_PATH": cache / "easyocr",
        "XDG_CACHE_HOME": cache / "xdg",
        "PLAYWRIGHT_BROWSERS_PATH": cache / "playwright",
    }
    for key, path in mapping.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        env[key] = str(path)
    env["MLT_AUTO_RESTART"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _run_git(args: list, root: Path, env: dict, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_creation_flags(),
    )


def try_git_pull(root: Path = REPO_ROOT, env: Optional[dict] = None, timeout: float = 10.0) -> bool:
    """Fast-forward pull on a clean tree with a remote. Fail-open: never raises."""
    try:
        if not (Path(root) / ".git").exists():
            return False
        env = env if env is not None else build_env(root)
        remotes = _run_git(["remote"], root, env, timeout)
        if remotes.returncode != 0 or not remotes.stdout.strip():
            return False
        status = _run_git(["status", "--porcelain"], root, env, timeout)
        if status.returncode != 0 or status.stdout.strip():
            return False
        pull = _run_git(
            ["-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=8", "pull", "--ff-only"],
            root,
            env,
            timeout,
        )
        return pull.returncode == 0
    except Exception:
        return False


def venv_python(root: Path = REPO_ROOT) -> Path:
    exe = Path(root) / ".venv" / "Scripts" / "python.exe"
    if exe.exists() or os.name == "nt":
        return exe
    return Path(sys.executable)


def run_server_once(root: Path, env: dict, log_path: Path) -> int:
    """Run `python -m server.app` once and return its exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            [str(venv_python(root)), "-m", "server.app"],
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            creationflags=_creation_flags(),
        )
        return proc.wait()


def supervise(
    root: Path = REPO_ROOT,
    runner: Callable[[Path, dict, Path], int] = run_server_once,
    max_restarts: int = 50,
) -> int:
    """Run the server, looping only on exit code 42. Returns the last exit code."""
    env = build_env(root)
    log_path = cache_dir(root) / "server.log"
    restarts = 0
    while True:
        code = runner(root, env, log_path)
        if code == RESTART_EXIT_CODE and restarts < max_restarts:
            restarts += 1
            time.sleep(0.5)
            continue
        return code


def main() -> int:
    root = REPO_ROOT
    cfg = load_config(root)
    if fetch_health(cfg) is not None:
        return 0
    if not acquire_pid_file(root):
        return 0
    own = os.getpid()
    try:
        os.chdir(str(root))
        try_git_pull(root, build_env(root))
        return supervise(root)
    finally:
        remove_pid_file(root, only_if_pid=own)


if __name__ == "__main__":
    sys.exit(main())

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
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

RESTART_EXIT_CODE = 42
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
CREATED_AT_TOLERANCE_100NS = 10_000_000

REPO_ROOT = Path(__file__).resolve().parent.parent

_KERNEL32 = None
_PYTHON_EXE_WIN_RE = re.compile(r"^pythonw?(?:\d+(?:\.\d+)*)?\.exe$")
_PYTHON_EXE_POSIX_RE = re.compile(r"^pythonw?(?:\d+(?:\.\d+)*)?$")


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


def _win_kernel32():
    """Return a private kernel32 WinDLL instance with 64-bit-safe prototypes."""
    global _KERNEL32
    if _KERNEL32 is not None:
        return _KERNEL32
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE

    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL

    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetExitCodeProcess.restype = wintypes.BOOL

    k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    k32.QueryFullProcessImageNameW.restype = wintypes.BOOL

    k32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    k32.GetProcessTimes.restype = wintypes.BOOL

    _KERNEL32 = k32
    return k32


def pid_alive(pid: int) -> bool:
    """Cross-platform liveness check. Never uses os.kill on Windows."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            k32 = _win_kernel32()
            handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                code = wintypes.DWORD()
                if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                k32.CloseHandle(handle)
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


def process_identity(pid: int) -> Optional[tuple[str, int]]:
    """Return (exe_path_lower, creation_time_100ns) for a live process, or None."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            ERROR_INSUFFICIENT_BUFFER = 122

            k32 = _win_kernel32()
            handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return None
            try:
                code = wintypes.DWORD()
                if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return None
                if code.value != STILL_ACTIVE:
                    return None

                exe_path: Optional[str] = None
                for buf_len in (1024, 32768):
                    buf = ctypes.create_unicode_buffer(buf_len)
                    size = wintypes.DWORD(buf_len)
                    if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                        exe_path = buf.value.lower()
                        break
                    if ctypes.get_last_error() != ERROR_INSUFFICIENT_BUFFER:
                        return None
                if not exe_path:
                    return None

                creation = wintypes.FILETIME()
                exit_ft = wintypes.FILETIME()
                kernel_ft = wintypes.FILETIME()
                user_ft = wintypes.FILETIME()
                if not k32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_ft),
                    ctypes.byref(kernel_ft),
                    ctypes.byref(user_ft),
                ):
                    return None
                created_at = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                return (exe_path, created_at)
            finally:
                k32.CloseHandle(handle)
        except Exception:
            return None

    # POSIX fallback: verify liveness and inspect /proc when available.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    except Exception:
        return None

    proc_dir = Path(f"/proc/{pid}")
    try:
        exe_path = os.path.realpath(str(proc_dir / "exe")).lower()
        if not exe_path:
            return None
        start_time_100ns = 0
        try:
            stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
            rparen = stat_text.rfind(")")
            if rparen != -1:
                fields = stat_text[rparen + 2 :].split()
                if len(fields) > 19:
                    ticks = int(fields[19])
                    clk_tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
                    start_time_100ns = ticks * (10_000_000 // max(1, int(clk_tck)))
        except Exception:
            pass
        return (exe_path, start_time_100ns)
    except Exception:
        return None


_DEFAULT_PID_ALIVE = pid_alive
_DEFAULT_PROCESS_IDENTITY = process_identity


def _is_python_executable(exe_path: str) -> bool:
    name = os.path.basename(exe_path.replace("\\", "/")).lower()
    if not name:
        return False
    if os.name == "nt" or name.endswith(".exe"):
        return bool(_PYTHON_EXE_WIN_RE.match(name))
    return bool(_PYTHON_EXE_POSIX_RE.match(name))


def _norm_path_str(p: str | Path) -> str:
    text = str(p)
    if os.name == "nt" or (len(text) >= 2 and text[1] == ":"):
        return text.replace("/", "\\").rstrip("\\").lower()
    return text.rstrip("/").lower()


def _parent_dir_str(norm_path: str) -> str:
    sep = "\\" if "\\" in norm_path else "/"
    idx = norm_path.rfind(sep)
    return norm_path[:idx] if idx > 0 else ""


def _allowed_legacy_python_dirs(root: Path) -> set[str]:
    """Directories whose Python binary may legitimately own a legacy plain-int server.pid."""
    dirs: set[str] = set()
    root_path = Path(root)
    venv_dir = root_path / ".venv"

    def _add(p: Path) -> None:
        try:
            dirs.add(_norm_path_str(p))
        except Exception:
            pass
        try:
            dirs.add(_norm_path_str(p.resolve()))
        except Exception:
            pass

    if venv_dir.is_dir():
        cfg = venv_dir / "pyvenv.cfg"
        if cfg.is_file():
            try:
                for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip().lower()
                    val = val.strip()
                    if not val:
                        continue
                    if key == "home":
                        _add(Path(val))
                    elif key == "executable":
                        _add(Path(val).parent)
            except Exception:
                pass
        try:
            if _norm_path_str(Path(sys.prefix).resolve()) == _norm_path_str(venv_dir.resolve()):
                _add(Path(sys.base_prefix))
                _add(Path(sys.executable).parent)
        except Exception:
            pass
    else:
        _add(Path(sys.base_prefix))
        _add(Path(sys.executable).parent)
        if os.name != "nt":
            _add(Path(sys.base_prefix) / "bin")
    dirs.discard("")
    return dirs


def _is_repo_python_path(exe_norm: str, root: Path) -> bool:
    """For legacy plain-int server.pid (no created_at), verify Python belongs to this repo/venv."""
    exe_clean = _norm_path_str(exe_norm)
    sep = "\\" if (os.name == "nt" or "\\" in exe_clean) else "/"
    try:
        root_norm = _norm_path_str(Path(root).resolve())
        if root_norm and exe_clean.startswith(root_norm + sep):
            return True
    except Exception:
        pass
    try:
        root_raw = _norm_path_str(Path(root))
        if root_raw and exe_clean.startswith(root_raw + sep):
            return True
    except Exception:
        pass
    return _parent_dir_str(exe_clean) in _allowed_legacy_python_dirs(root)


def is_supervisor_process(
    pid: int,
    expected_created_at: Optional[int] = None,
    root: Path = REPO_ROOT,
) -> bool:
    """Return True iff `pid` is a live Python process matching this supervisor record."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False

    pid_alive_patched = pid_alive is not _DEFAULT_PID_ALIVE
    identity_patched = process_identity is not _DEFAULT_PROCESS_IDENTITY

    if pid_alive_patched or not identity_patched:
        if not pid_alive(pid):
            return False

    identity = process_identity(pid)
    if identity is None:
        if pid_alive_patched and not identity_patched:
            return True
        if os.name != "nt" and not identity_patched:
            return True
        return False

    exe_lower, actual_created_at = identity
    if not isinstance(exe_lower, str) or not exe_lower:
        return False
    if not _is_python_executable(exe_lower):
        return False

    if expected_created_at is not None:
        if not isinstance(actual_created_at, int) or isinstance(actual_created_at, bool):
            return False
        return abs(actual_created_at - expected_created_at) <= CREATED_AT_TOLERANCE_100NS

    return _is_repo_python_path(exe_lower, root)


def read_pid_record(root: Path = REPO_ROOT) -> Optional[tuple[int, Optional[int]]]:
    """Read cache/server.pid as (pid, created_at_or_None). Supports JSON and legacy plain int."""
    try:
        text = pid_file(root).read_text(encoding="utf-8").strip()
        if not text:
            return None
        if text.startswith("{"):
            data = json.loads(text)
            if not isinstance(data, dict):
                return None
            pid = data.get("pid")
            if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
                return None
            created_at = data.get("created_at")
            if created_at is None:
                return (pid, None)
            if isinstance(created_at, int) and not isinstance(created_at, bool) and created_at >= 0:
                return (pid, created_at)
            return None
        pid = int(text)
        return (pid, None) if pid > 0 else None
    except Exception:
        return None


def read_pid(root: Path = REPO_ROOT) -> Optional[int]:
    rec = read_pid_record(root)
    return rec[0] if rec is not None else None


def live_supervisor_pid(root: Path = REPO_ROOT) -> Optional[int]:
    rec = read_pid_record(root)
    if rec is None:
        return None
    pid, created_at = rec
    if is_supervisor_process(pid, expected_created_at=created_at, root=root):
        return pid
    remove_pid_file(root, only_if_pid=pid)
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
            rec = read_pid_record(root)
            if rec is not None:
                existing_pid, existing_created_at = rec
                if existing_pid == own_pid:
                    return True
                if is_supervisor_process(
                    existing_pid,
                    expected_created_at=existing_created_at,
                    root=root,
                ):
                    return False
                remove_pid_file(root, only_if_pid=existing_pid)
            else:
                remove_pid_file(root)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            try:
                identity = process_identity(own_pid)
            except Exception:
                identity = None
            created_at = identity[1] if identity is not None else None
            payload = (
                json.dumps({"pid": own_pid, "created_at": created_at}, separators=(",", ":"))
                if created_at is not None
                else str(own_pid)
            )
            fh.write(payload)
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

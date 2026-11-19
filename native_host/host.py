"""Edge/Chrome Native Messaging host for Manga Live Translator.

Protocol: 4-byte little-endian uint32 length + UTF-8 JSON on binary stdio.
Actions: ping, status, start_server, stop_server.

IMPORTANT: nothing but framed responses may ever be written to stdout.
Internal errors go to cache/native-host.log only.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import supervisor  # noqa: E402

VERSION = "1.0.0"
HOST_NAME = "com.mangalivetranslator.host"
REPO_ROOT = HERE.parent
MAX_INCOMING = 1024 * 1024
MAX_OUTGOING = 1024 * 1024

log = logging.getLogger("mlt.native_host")


# ---------------------------------------------------------------- logging

def setup_logging(root: Path = REPO_ROOT) -> None:
    try:
        cache = root / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(cache / "native-host.log", encoding="utf-8")
    except Exception:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False


# ---------------------------------------------------------------- framing

def read_exact(stream, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def read_message(stream) -> Optional[Any]:
    """Return the next decoded JSON message, or None on clean EOF."""
    header = read_exact(stream, 4)
    if header is None:
        return None
    (length,) = struct.unpack("<I", header)
    if length > MAX_INCOMING:
        raise ValueError(f"incoming message too large: {length}")
    body = read_exact(stream, length) if length else b""
    if body is None:
        raise ValueError("truncated message body")
    return json.loads(body.decode("utf-8"))


def encode_message(obj: Any) -> bytes:
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_OUTGOING:
        body = json.dumps({"ok": False, "error": "response too large"}).encode("utf-8")
    return struct.pack("<I", len(body)) + body


def write_message(stream, obj: Any) -> None:
    stream.write(encode_message(obj))
    stream.flush()


# ---------------------------------------------------------------- helpers

def _flags(*names: str) -> int:
    value = 0
    for name in names:
        value |= getattr(subprocess, name, 0)
    return value if os.name == "nt" else 0


def _server_config(root: Path) -> dict:
    return supervisor.load_config(root)


def _status_snapshot(root: Path) -> dict:
    cfg = _server_config(root)
    health = supervisor.fetch_health(cfg)
    sup_pid = supervisor.live_supervisor_pid(root)
    running = health is not None
    ready = bool(health.get("ready")) if isinstance(health, dict) else False
    return {
        "running": running,
        "ready": ready,
        "pid": sup_pid,
        "supervisor_alive": sup_pid is not None,
    }


def _supervisor_python(root: Path) -> Path:
    scripts = root / ".venv" / "Scripts"
    for name in ("pythonw.exe", "python.exe"):
        candidate = scripts / name
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def spawn_supervisor(root: Path = REPO_ROOT) -> int:
    """Start supervisor.py fully detached; never inherits Edge's stdio pipes."""
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cmd = [str(_supervisor_python(root)), str(root / "native_host" / "supervisor.py")]
    env = supervisor.build_env(root)
    base_flags = _flags("CREATE_NO_WINDOW", "DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP")
    attempts = [base_flags]
    if os.name == "nt":
        breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
        attempts.insert(0, base_flags | breakaway)
    last_err: Optional[Exception] = None
    with open(cache / "server.log", "ab") as logfile:
        for flags in attempts:
            kwargs: dict = dict(
                cwd=str(root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            if os.name == "nt":
                kwargs["creationflags"] = flags
            else:
                kwargs["start_new_session"] = True
            try:
                proc = subprocess.Popen(cmd, **kwargs)
                return proc.pid
            except OSError as exc:  # e.g. job object forbids breakaway
                last_err = exc
                log.warning("spawn with flags %s failed: %s", hex(flags), exc)
    raise last_err or OSError("failed to spawn supervisor")


def find_repo_pids(root: Path = REPO_ROOT) -> list:
    """PIDs of this repo's supervisor, server.app, and llama-server.exe only."""
    if os.name != "nt":
        return []
    script = (
        "$ic = [System.StringComparison]::OrdinalIgnoreCase;"
        "$root = $env:MLT_REPO_ROOT;"
        "$out = @();"
        "Get-CimInstance Win32_Process | ForEach-Object {"
        "  $n = $_.Name; $cl = $_.CommandLine; $ep = $_.ExecutablePath;"
        "  $hit = $false;"
        "  if ($n -eq 'llama-server.exe' -and $ep -and $ep.StartsWith($root + '\\bin\\llama-cpp', $ic)) { $hit = $true }"
        "  elseif (($n -eq 'python.exe' -or $n -eq 'pythonw.exe') -and $cl -and $cl.IndexOf($root, $ic) -ge 0 -and"
        "          ($cl.IndexOf('supervisor.py', $ic) -ge 0 -or $cl.IndexOf('server.app', $ic) -ge 0)) { $hit = $true }"
        "  elseif (($n -eq 'python.exe' -or $n -eq 'pythonw.exe') -and $cl -and $cl.IndexOf('server.app', $ic) -ge 0 -and"
        "          $ep -and $ep.StartsWith($root, $ic)) { $hit = $true }"
        "  if ($hit) { $out += [int]$_.ProcessId }"
        "};"
        "ConvertTo-Json -InputObject @($out) -Compress"
    )
    env = dict(os.environ)
    env["MLT_REPO_ROOT"] = str(root).rstrip("\\/")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
            stdin=subprocess.DEVNULL,
            creationflags=_flags("CREATE_NO_WINDOW"),
        )
        data = json.loads(result.stdout.strip() or "[]")
        if isinstance(data, int):
            data = [data]
        return [int(p) for p in data if int(p) != os.getpid()]
    except Exception as exc:
        log.warning("find_repo_pids failed: %s", exc)
        return []


def kill_pids(pids: list) -> None:
    own = os.getpid()
    for pid in pids:
        if pid == own:
            continue
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
                creationflags=_flags("CREATE_NO_WINDOW"),
            )
        except Exception as exc:
            log.warning("taskkill %s failed: %s", pid, exc)


# ---------------------------------------------------------------- actions

def action_ping(root: Path) -> dict:
    return {"ok": True, "version": VERSION}


def action_status(root: Path) -> dict:
    snap = _status_snapshot(root)
    return {"ok": True, "running": snap["running"], "ready": snap["ready"], "pid": snap["pid"]}


def action_start_server(root: Path) -> dict:
    snap = _status_snapshot(root)
    if snap["running"] or snap["supervisor_alive"]:
        return {"ok": True, "started": False, "already_running": True, "pid": snap["pid"]}
    pid = spawn_supervisor(root)
    return {"ok": True, "started": True, "already_running": False, "pid": pid}


def action_stop_server(root: Path) -> dict:
    cfg = _server_config(root)
    try:
        req = urllib.request.Request(supervisor.base_url(cfg) + "/shutdown", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=4).close()
    except Exception:
        pass
    deadline = time.time() + 3.0
    while time.time() < deadline and supervisor.fetch_health(cfg, timeout=0.5) is not None:
        time.sleep(0.25)
    time.sleep(0.5)
    kill_pids(find_repo_pids(root))
    supervisor.remove_pid_file(root)
    return {"ok": True, "stopped": True}


ACTIONS = {
    "ping": action_ping,
    "status": action_status,
    "start_server": action_start_server,
    "stop_server": action_stop_server,
}


def handle_message(msg: Any, root: Path = REPO_ROOT) -> dict:
    try:
        action = msg.get("action") if isinstance(msg, dict) else None
        fn = ACTIONS.get(action)
        if fn is None:
            return {"ok": False, "error": f"unknown action: {action!r}"}
        return fn(root)
    except Exception as exc:
        log.exception("action failed")
        return {"ok": False, "error": str(exc) or exc.__class__.__name__}


# ---------------------------------------------------------------- main

def main() -> int:
    setup_logging(REPO_ROOT)
    if os.name == "nt":
        import msvcrt

        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    # Any stray print() must not corrupt the framed channel.
    try:
        sys.stdout = open(REPO_ROOT / "cache" / "native-host.log", "a", encoding="utf-8")
    except Exception:
        sys.stdout = open(os.devnull, "w")
    sys.stderr = sys.stdout

    while True:
        try:
            msg = read_message(stdin)
        except Exception as exc:
            log.error("read error: %s", exc)
            try:
                write_message(stdout, {"ok": False, "error": f"bad message: {exc}"})
            except Exception:
                pass
            return 1
        if msg is None:
            return 0
        response = handle_message(msg)
        try:
            write_message(stdout, response)
        except Exception as exc:
            log.error("write error: %s", exc)
            return 1


if __name__ == "__main__":
    sys.exit(main())

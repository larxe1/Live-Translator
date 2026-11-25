import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NATIVE_DIR = REPO_ROOT / "native_host"
sys.path.insert(0, str(NATIVE_DIR))

import host  # noqa: E402
import register_host  # noqa: E402
import supervisor  # noqa: E402


def frame(obj) -> bytes:
    body = json.dumps(obj).encode("utf-8")
    return struct.pack("<I", len(body)) + body


def parse_frames(raw: bytes) -> list:
    out = []
    pos = 0
    while pos < len(raw):
        (length,) = struct.unpack("<I", raw[pos:pos + 4])
        pos += 4
        out.append(json.loads(raw[pos:pos + length].decode("utf-8")))
        pos += length
    assert pos == len(raw), "stray bytes after framed responses"
    return out


# ------------------------------------------------------------ framing

def test_host_binary_framing_roundtrip_subprocess():
    payload = frame({"action": "ping"}) + frame({"action": "status"}) + frame({"action": "bogus"})
    proc = subprocess.run(
        [sys.executable, "-u", str(NATIVE_DIR / "host.py")],
        input=payload,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0
    ping, status, bogus = parse_frames(proc.stdout)
    assert ping == {"ok": True, "version": "1.0.0"}
    assert status["ok"] is True
    assert isinstance(status["running"], bool)
    assert isinstance(status["ready"], bool)
    assert status["pid"] is None or isinstance(status["pid"], int)
    assert bogus["ok"] is False and "error" in bogus


def test_encode_and_read_message_roundtrip():
    import io

    stream = io.BytesIO(host.encode_message({"hello": "wörld"}))
    assert host.read_message(stream) == {"hello": "wörld"}
    assert host.read_message(stream) is None


def test_host_cmd_has_no_stray_output():
    text = (NATIVE_DIR / "host.cmd").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    assert lines[0].lower() == "@echo off"
    assert "host.py" in lines[1] and ".venv" in lines[1] and "-u" in lines[1]
    assert not any(ln.lower().startswith("echo ") for ln in lines)


# ------------------------------------------------------------ start_server

def test_start_server_already_running_when_health_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(host.supervisor, "fetch_health", lambda cfg, timeout=1.5: {"ready": True})
    spawned = []
    monkeypatch.setattr(host, "spawn_supervisor", lambda root: spawned.append(root) or 1)
    resp = host.handle_message({"action": "start_server"}, tmp_path)
    assert resp == {"ok": True, "started": False, "already_running": True, "pid": None}
    assert spawned == []


def test_start_server_already_running_when_supervisor_pid_alive(tmp_path, monkeypatch):
    monkeypatch.setattr(host.supervisor, "fetch_health", lambda cfg, timeout=1.5: None)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "server.pid").write_text(str(os.getpid()))
    resp = host.handle_message({"action": "start_server"}, tmp_path)
    assert resp["ok"] is True and resp["already_running"] is True
    assert resp["pid"] == os.getpid()


def test_spawn_supervisor_uses_detached_stdio_and_flags(tmp_path, monkeypatch):
    captured = {}

    class FakeProc:
        pid = 4321

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(host.subprocess, "Popen", fake_popen)
    pid = host.spawn_supervisor(tmp_path)
    kw = captured["kwargs"]
    assert pid == 4321
    assert kw["stdin"] == subprocess.DEVNULL
    assert kw["stderr"] == subprocess.STDOUT
    assert kw["close_fds"] is True
    assert kw["stdout"] is not subprocess.PIPE and hasattr(kw["stdout"], "write")
    assert captured["cmd"][1].endswith("supervisor.py")
    assert kw["env"]["MLT_AUTO_RESTART"] == "1"
    if os.name == "nt":
        flags = kw["creationflags"]
        for name in ("CREATE_NO_WINDOW", "DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
            assert flags & getattr(subprocess, name)


def test_spawn_supervisor_does_not_hold_stdout_pipe(tmp_path):
    """A parent whose stdout is a pipe must see EOF while the child still runs."""
    (tmp_path / "native_host").mkdir()
    marker = tmp_path / "child_done.txt"
    (tmp_path / "native_host" / "supervisor.py").write_text(
        "import time, pathlib\n"
        "time.sleep(3)\n"
        f"pathlib.Path(r'{marker}').write_text('done')\n",
        encoding="utf-8",
    )
    code = (
        "import sys\n"
        f"sys.path.insert(0, r'{NATIVE_DIR}')\n"
        "import host\n"
        f"from pathlib import Path\n"
        f"print(host.spawn_supervisor(Path(r'{tmp_path}')))\n"
    )
    start = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        timeout=20,
    )
    elapsed = time.time() - start
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout.strip()) > 0
    assert elapsed < 2.8, f"stdout pipe was inherited by detached child ({elapsed:.1f}s)"
    deadline = time.time() + 15
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.2)
    assert marker.exists()


# ------------------------------------------------------------ registration

EXT_ID = "abcdefghijklmnopabcdefghijklmnop"
OTHER_ID = "ponmlkjihgfedcbaponmlkjihgfedcba"


def _write_prefs(profile: Path, name: str, settings: dict) -> None:
    profile.mkdir(parents=True, exist_ok=True)
    (profile / name).write_text(json.dumps({"extensions": {"settings": settings}}), encoding="utf-8")


def test_discover_extension_ids_and_build_manifest(tmp_path):
    repo = tmp_path / "repo"
    (repo / "extension").mkdir(parents=True)
    user_data = tmp_path / "Edge" / "User Data"
    _write_prefs(
        user_data / "Default",
        "Secure Preferences",
        {
            EXT_ID: {"location": 4, "path": str(repo / "extension")},
            OTHER_ID: {"location": 4, "path": str(tmp_path / "somewhere" / "else")},
            "notanid": {"path": str(repo / "extension")},
        },
    )
    _write_prefs(
        user_data / "Profile 1",
        "Preferences",
        {"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": {"path": str(repo / "extension").upper()}},
    )
    ids = register_host.discover_extension_ids(repo, [user_data])
    assert EXT_ID in ids and OTHER_ID not in ids and "notanid" not in ids
    assert "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" in ids or os.name != "nt"

    manifest = register_host.build_host_manifest(repo, ids)
    assert manifest["name"] == "com.mangalivetranslator.host"
    assert manifest["type"] == "stdio"
    assert manifest["path"] == str(repo.resolve() / "native_host" / "host.cmd")
    assert f"chrome-extension://{EXT_ID}/" in manifest["allowed_origins"]


def test_build_manifest_falls_back_to_default_id(tmp_path):
    assert register_host.discover_extension_ids(tmp_path, [tmp_path / "missing"]) == []
    manifest = register_host.build_host_manifest(tmp_path, [])
    assert manifest["allowed_origins"] == ["chrome-extension://gkljnbldlglhdijoanaeenhpfkhcahch/"]


class FakeWinreg:
    HKEY_CURRENT_USER = "HKCU"
    KEY_SET_VALUE = 2
    REG_SZ = 1

    def __init__(self):
        self.values = {}

    def CreateKeyEx(self, root, sub, reserved, access):
        return (root, sub)

    def SetValueEx(self, key, name, reserved, typ, value):
        self.values[key] = value

    def CloseKey(self, key):
        pass


def test_register_native_host_writes_manifest_and_registry(tmp_path):
    repo = tmp_path / "repo"
    (repo / "extension").mkdir(parents=True)
    fake = FakeWinreg()
    result = register_host.register_native_host(
        repo, extension_ids=[EXT_ID], user_data_roots=[], winreg_module=fake
    )
    manifest_file = repo.resolve() / "cache" / "com.mangalivetranslator.host.json"
    assert Path(result["manifest_path"]) == manifest_file
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["allowed_origins"] == [f"chrome-extension://{EXT_ID}/"]
    assert len(fake.values) == 2
    subs = sorted(sub for (_root, sub) in fake.values)
    assert subs[0].startswith("Software\\Google\\Chrome\\NativeMessagingHosts\\")
    assert subs[1].startswith("Software\\Microsoft\\Edge\\NativeMessagingHosts\\")
    assert set(fake.values.values()) == {str(manifest_file)}


# ------------------------------------------------------------ supervisor

def test_supervise_restarts_only_on_exit_code_42(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor.time, "sleep", lambda s: None)
    codes = iter([42, 42, 0])
    calls = []

    def runner(root, env, log_path):
        calls.append(1)
        assert env["MLT_AUTO_RESTART"] == "1"
        return next(codes)

    assert supervisor.supervise(tmp_path, runner=runner) == 0
    assert len(calls) == 3


@pytest.mark.parametrize("code", [0, 1, 3])
def test_supervise_exits_on_other_codes(tmp_path, monkeypatch, code):
    monkeypatch.setattr(supervisor.time, "sleep", lambda s: None)
    calls = []

    def runner(root, env, log_path):
        calls.append(1)
        return code

    assert supervisor.supervise(tmp_path, runner=runner) == code
    assert len(calls) == 1


def test_run_server_once_uses_mocked_popen(tmp_path, monkeypatch):
    seen = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            seen.append((cmd, kwargs))

        def wait(self):
            return 42

    monkeypatch.setattr(supervisor.subprocess, "Popen", FakePopen)
    code = supervisor.run_server_once(tmp_path, supervisor.build_env(tmp_path), tmp_path / "cache" / "server.log")
    assert code == 42
    cmd, kw = seen[0]
    assert cmd[1:] == ["-m", "server.app"]
    assert kw["stdin"] == subprocess.DEVNULL and kw["stderr"] == subprocess.STDOUT
    assert kw["close_fds"] is True


def test_build_env_pins_caches_to_repo(tmp_path):
    env = supervisor.build_env(tmp_path, base={})
    cache = tmp_path / "cache"
    assert env["TEMP"] == env["TMP"] == str(cache / "tmp")
    assert env["HF_HOME"] == str(cache / "huggingface")
    assert env["TORCH_HOME"] == str(cache / "torch")
    assert env["EASYOCR_MODULE_PATH"] == str(cache / "easyocr")
    assert env["XDG_CACHE_HOME"] == str(cache / "xdg")
    assert env["PLAYWRIGHT_BROWSERS_PATH"] == str(cache / "playwright")
    assert env["MLT_AUTO_RESTART"] == "1"


def test_load_config_defaults_and_values(tmp_path):
    assert supervisor.load_config(tmp_path) == {"host": "127.0.0.1", "port": 8765}
    (tmp_path / "config.json").write_text(json.dumps({"host": "127.0.0.1", "port": 9001}))
    assert supervisor.load_config(tmp_path)["port"] == 9001


def test_pid_file_claim_and_stale_cleanup(tmp_path):
    assert supervisor.acquire_pid_file(tmp_path, own_pid=os.getpid())
    assert supervisor.read_pid(tmp_path) == os.getpid()
    # a different caller must not steal a live supervisor's pid file
    assert supervisor.acquire_pid_file(tmp_path, own_pid=os.getpid() + 100000) is False
    supervisor.remove_pid_file(tmp_path)
    (tmp_path / "cache" / "server.pid").write_text("999999999")
    assert supervisor.acquire_pid_file(tmp_path, own_pid=os.getpid())
    supervisor.remove_pid_file(tmp_path, only_if_pid=os.getpid())
    assert not (tmp_path / "cache" / "server.pid").exists()


# ------------------------------------------------------------ PID reuse protection

def test_legacy_pid_reused_by_conhost_is_rejected_and_cleaned(tmp_path, monkeypatch):
    """Stale plain-int server.pid reused by Windows conhost.exe must not block start_server."""
    stale_pid = 18216
    pid_path = tmp_path / "cache" / "server.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(stale_pid), encoding="utf-8")

    monkeypatch.setattr(supervisor, "pid_alive", lambda pid: pid in (stale_pid, os.getpid()))
    monkeypatch.setattr(
        supervisor,
        "process_identity",
        lambda pid: (
            (r"c:\windows\system32\conhost.exe", 134400000000000000)
            if pid == stale_pid
            else (str(tmp_path / ".venv" / "Scripts" / "pythonw.exe").lower(), 134400500000000000)
        ),
    )

    assert supervisor.live_supervisor_pid(tmp_path) is None
    assert not pid_path.exists()

    # Re-create stale file and verify acquire_pid_file replaces it cleanly.
    pid_path.write_text(str(stale_pid), encoding="utf-8")
    assert supervisor.acquire_pid_file(tmp_path, own_pid=os.getpid()) is True
    assert supervisor.read_pid_record(tmp_path) == (os.getpid(), 134400500000000000)

    # Re-create stale file and verify host.action_start_server spawns a new supervisor.
    pid_path.write_text(str(stale_pid), encoding="utf-8")
    monkeypatch.setattr(host.supervisor, "fetch_health", lambda cfg, timeout=1.5: None)
    spawned = []
    monkeypatch.setattr(host, "spawn_supervisor", lambda root: spawned.append(root) or 55555)
    resp = host.handle_message({"action": "start_server"}, tmp_path)
    assert resp == {"ok": True, "started": True, "already_running": False, "pid": 55555}
    assert spawned == [tmp_path]
    assert not pid_path.exists()


def test_legacy_pid_unrelated_system_python_rejected_but_repo_venv_accepted(tmp_path, monkeypatch):
    """Legacy plain-int server.pid must reject unrelated system python.exe but accept repo venv."""
    pid_path = tmp_path / "cache" / "server.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)

    other_pid = 22222
    pid_path.write_text(str(other_pid), encoding="utf-8")
    monkeypatch.setattr(supervisor, "pid_alive", lambda pid: pid == other_pid)
    monkeypatch.setattr(
        supervisor,
        "process_identity",
        lambda pid: (r"c:\someotherpython\python.exe", 134400100000000000) if pid == other_pid else None,
    )
    assert supervisor.live_supervisor_pid(tmp_path) is None
    assert not pid_path.exists()

    repo_exe = str(tmp_path / ".venv" / "Scripts" / "pythonw.exe").lower()
    pid_path.write_text(str(other_pid), encoding="utf-8")
    monkeypatch.setattr(
        supervisor,
        "process_identity",
        lambda pid: (repo_exe, 134400100000000000) if pid == other_pid else None,
    )
    assert supervisor.live_supervisor_pid(tmp_path) == other_pid
    assert pid_path.exists()


def test_json_pid_reused_by_another_python_with_mismatched_created_at(tmp_path, monkeypatch):
    """JSON server.pid whose PID was reused by a newer python.exe must be treated as stale."""
    reused_pid = 24680
    old_created_at = 134400000000000000
    new_created_at = old_created_at + 600_000_000  # 60 seconds later (> 1s tolerance)
    pid_path = tmp_path / "cache" / "server.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(json.dumps({"pid": reused_pid, "created_at": old_created_at}), encoding="utf-8")

    monkeypatch.setattr(supervisor, "pid_alive", lambda pid: pid in (reused_pid, os.getpid()))
    monkeypatch.setattr(
        supervisor,
        "process_identity",
        lambda pid: (
            (r"d:\code\manga-live-translator\.venv\scripts\python.exe", new_created_at)
            if pid == reused_pid
            else (r"d:\code\manga-live-translator\.venv\scripts\pythonw.exe", new_created_at + 50_000_000)
        ),
    )

    assert supervisor.live_supervisor_pid(tmp_path) is None
    assert not pid_path.exists()

    # acquire_pid_file should also reclaim a stale JSON record when creation time differs.
    pid_path.write_text(json.dumps({"pid": reused_pid, "created_at": old_created_at}), encoding="utf-8")
    assert supervisor.acquire_pid_file(tmp_path, own_pid=os.getpid()) is True
    assert supervisor.read_pid(tmp_path) == os.getpid()


@pytest.mark.parametrize(
    "exe_path,delta_100ns",
    [
        (r"d:\code\manga-live-translator\.venv\scripts\pythonw.exe", 0),
        (
            r"c:\program files\windowsapps\pythonsoftwarefoundation.python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\pythonw3.12.exe",
            5_000_000,  # 0.5s within 1s tolerance
        ),
    ],
)
def test_json_pid_matching_pythonw_and_created_at_is_recognized_live(tmp_path, monkeypatch, exe_path, delta_100ns):
    """JSON server.pid with matching pythonw executable and created_at must be treated as live."""
    live_pid = 31337
    created_at = 134400380158222135
    pid_path = tmp_path / "cache" / "server.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(json.dumps({"pid": live_pid, "created_at": created_at}), encoding="utf-8")

    monkeypatch.setattr(supervisor, "pid_alive", lambda pid: pid == live_pid)
    monkeypatch.setattr(
        supervisor,
        "process_identity",
        lambda pid: (exe_path, created_at + delta_100ns) if pid == live_pid else None,
    )

    assert supervisor.live_supervisor_pid(tmp_path) == live_pid
    assert pid_path.exists()
    assert supervisor.acquire_pid_file(tmp_path, own_pid=os.getpid()) is False

    monkeypatch.setattr(host.supervisor, "fetch_health", lambda cfg, timeout=1.5: None)
    spawned = []
    monkeypatch.setattr(host, "spawn_supervisor", lambda root: spawned.append(root) or 99999)
    resp = host.handle_message({"action": "start_server"}, tmp_path)
    assert resp == {"ok": True, "started": False, "already_running": True, "pid": live_pid}
    assert spawned == []


def test_read_pid_record_parses_json_legacy_and_rejects_malformed(tmp_path):
    pid_path = tmp_path / "cache" / "server.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    pid_path.write_text("12345\n", encoding="utf-8")
    assert supervisor.read_pid_record(tmp_path) == (12345, None)
    assert supervisor.read_pid(tmp_path) == 12345

    pid_path.write_text(json.dumps({"pid": 12345, "created_at": 134400000000000000}), encoding="utf-8")
    assert supervisor.read_pid_record(tmp_path) == (12345, 134400000000000000)
    assert supervisor.read_pid(tmp_path) == 12345

    for bad in ("", "not-a-pid", "-5", "0", "{bad json", '{"pid": true}', '{"pid": 123, "created_at": "bad"}'):
        pid_path.write_text(bad, encoding="utf-8")
        assert supervisor.read_pid_record(tmp_path) is None
        assert supervisor.read_pid(tmp_path) is None


def test_process_identity_and_acquire_pid_file_write_json_on_windows(tmp_path):
    ident = supervisor.process_identity(os.getpid())
    if os.name == "nt":
        assert ident is not None
        exe_lower, created_at = ident
        assert exe_lower == exe_lower.lower()
        assert supervisor._is_python_executable(exe_lower)
        assert isinstance(created_at, int) and created_at > 0
        assert supervisor.acquire_pid_file(tmp_path, own_pid=os.getpid()) is True
        raw = json.loads((tmp_path / "cache" / "server.pid").read_text(encoding="utf-8"))
        assert raw["pid"] == os.getpid()
        assert raw["created_at"] == created_at
        assert supervisor.live_supervisor_pid(tmp_path) == os.getpid()


def test_monkeypatched_pid_alive_backward_compatibility(tmp_path, monkeypatch):
    """Existing-style unit tests that monkeypatch only supervisor.pid_alive keep working."""
    fake_pid = 88888888
    pid_path = tmp_path / "cache" / "server.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(fake_pid), encoding="utf-8")

    monkeypatch.setattr(supervisor, "pid_alive", lambda pid: pid == fake_pid)
    assert supervisor.live_supervisor_pid(tmp_path) == fake_pid

    monkeypatch.setattr(supervisor, "pid_alive", lambda pid: False)
    assert supervisor.live_supervisor_pid(tmp_path) is None
    assert not pid_path.exists()

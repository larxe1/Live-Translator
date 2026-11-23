from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading

from fastapi.testclient import TestClient
import pytest

from server.app import create_app
from server.cache import TranslationCache
from server.config import AppConfig
from server.translator.memory import ChapterMemoryStore
from server.updater import (
    RESTART_EXIT_CODE,
    UpdateManager,
    compute_extension_build_id,
    read_extension_version,
)


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "Test User",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test User",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_repo_with_extension(repo_dir: Path) -> None:
    ext_dir = repo_dir / "extension"
    srv_dir = repo_dir / "server"
    ext_dir.mkdir(parents=True, exist_ok=True)
    srv_dir.mkdir(parents=True, exist_ok=True)
    (ext_dir / "manifest.json").write_text(
        json.dumps({"manifest_version": 3, "name": "Test", "version": "1.1.0"}),
        encoding="utf-8",
    )
    (ext_dir / "background.js").write_text("console.log('v1');\n", encoding="utf-8")
    (srv_dir / "dummy.py").write_text("VALUE = 1\n", encoding="utf-8")


def _setup_bare_and_clones(tmp_path: Path) -> tuple[Path, Path, Path]:
    bare = tmp_path / "remote.git"
    bare.mkdir()
    _git(bare, "init", "--bare", "-b", "main")

    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(bare), str(seed))
    _git(seed, "checkout", "-b", "main")
    _init_repo_with_extension(seed)
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "initial commit")
    _git(seed, "push", "-u", "origin", "main")

    local = tmp_path / "local"
    _git(tmp_path, "clone", "-b", "main", str(bare), str(local))
    return bare, seed, local


def test_compute_extension_build_id_deterministic_and_ignores_junk(tmp_path: Path) -> None:
    ext_dir = tmp_path / "extension"
    ext_dir.mkdir()
    (ext_dir / "manifest.json").write_text('{"version": "1.1.0"}', encoding="utf-8")
    (ext_dir / "content.js").write_text("const x = 1;", encoding="utf-8")

    build_1 = compute_extension_build_id(ext_dir)
    build_2 = compute_extension_build_id(ext_dir)
    assert len(build_1) == 16
    assert build_1 == build_1.lower()
    assert build_1 == build_2

    # Ignored files should not alter build ID
    pycache = ext_dir / "__pycache__"
    pycache.mkdir()
    (pycache / "mod.pyc").write_bytes(b"\x00\x01\x02")
    (ext_dir / ".DS_Store").write_bytes(b"junk")
    (ext_dir / "Thumbs.db").write_bytes(b"junk")
    (ext_dir / "draft.tmp").write_text("tmp", encoding="utf-8")
    (ext_dir / "old.bak").write_text("bak", encoding="utf-8")
    assert compute_extension_build_id(ext_dir) == build_1

    # Editing an extension file changes the build ID immediately
    (ext_dir / "content.js").write_text("const x = 2;", encoding="utf-8")
    build_3 = compute_extension_build_id(ext_dir)
    assert build_3 != build_1
    assert len(build_3) == 16

    assert read_extension_version(ext_dir) == "1.1.0"
    assert read_extension_version(tmp_path / "missing") == "1.0.0"


def test_update_manager_detects_refuses_dirty_and_applies_clean_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLT_AUTO_RESTART", raising=False)
    _bare, seed, local = _setup_bare_and_clones(tmp_path)

    mgr = UpdateManager(repo_root=local, extension_dir=local / "extension")
    initial_status = mgr.get_status()
    assert initial_status.has_remote is True
    assert initial_status.branch == "main"
    assert initial_status.working_tree_clean is True
    assert initial_status.update_available is False
    initial_build_id = initial_status.extension_build_id
    assert len(initial_build_id) == 16

    # Push an extension-only commit from seed
    (seed / "extension" / "background.js").write_text("console.log('v2');\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "update extension")
    _git(seed, "push")

    checked = mgr.check_for_updates(auto_apply=False)
    assert checked.commits_behind == 1
    assert checked.update_available is True
    assert checked.remote_commit is not None
    assert checked.last_error is None

    # Dirty working tree must refuse cleanly without touching local edit
    local_bg = local / "extension" / "background.js"
    local_bg.write_text("console.log('local edit');\n", encoding="utf-8")
    dirty_res = mgr.apply_update()
    assert dirty_res.ok is False
    assert dirty_res.updated is False
    assert "uncommitted" in dirty_res.message.lower()
    assert local_bg.read_text(encoding="utf-8") == "console.log('local edit');\n"

    # Restore clean working tree and apply extension-only update
    _git(local, "checkout", "--", "extension/background.js")
    applied_ext = mgr.apply_update()
    assert applied_ext.ok is True
    assert applied_ext.updated is True
    assert applied_ext.extension_changed is True
    assert applied_ext.server_changed is False
    assert applied_ext.restart_scheduled is False
    assert applied_ext.status.commits_behind == 0
    assert applied_ext.status.update_available is False
    assert applied_ext.status.extension_build_id != initial_build_id

    # Push a server-side commit and verify server_changed + restart_required without MLT_AUTO_RESTART
    (seed / "server" / "dummy.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "update server")
    _git(seed, "push")

    applied_srv = mgr.apply_update()
    assert applied_srv.ok is True
    assert applied_srv.updated is True
    assert applied_srv.extension_changed is False
    assert applied_srv.server_changed is True
    assert applied_srv.restart_scheduled is False
    assert applied_srv.status.restart_required is True


def test_update_manager_schedules_restart_only_when_env_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bare, seed, local = _setup_bare_and_clones(tmp_path)

    cleanup_called = False
    exit_codes: list[int] = []
    scheduled_callbacks: list[object] = []

    def _on_restart() -> None:
        nonlocal cleanup_called
        cleanup_called = True

    class _FakeTimer:
        def __init__(self, interval: float, fn) -> None:  # type: ignore[no-untyped-def]
            self.interval = interval
            self.fn = fn
            self.daemon = False

        def start(self) -> None:
            scheduled_callbacks.append(self.fn)

    monkeypatch.setenv("MLT_AUTO_RESTART", "1")
    monkeypatch.setattr(threading, "Timer", _FakeTimer)
    monkeypatch.setattr(os, "_exit", lambda code: exit_codes.append(code))

    (seed / "server" / "dummy.py").write_text("VALUE = 99\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "server hotfix")
    _git(seed, "push")

    mgr = UpdateManager(
        repo_root=local,
        extension_dir=local / "extension",
        on_restart_requested=_on_restart,
    )
    res = mgr.apply_update()
    assert res.ok is True
    assert res.updated is True
    assert res.server_changed is True
    assert res.restart_scheduled is True
    assert len(scheduled_callbacks) == 1

    # Invoke the scheduled restart callback manually to verify cleanup + exit code 42
    callback = scheduled_callbacks[0]
    assert callable(callback)
    callback()
    assert cleanup_called is True
    assert exit_codes == [RESTART_EXIT_CODE]


def test_update_manager_no_remote_graceful(tmp_path: Path) -> None:
    repo = tmp_path / "local_only"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _init_repo_with_extension(repo)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "local init")

    mgr = UpdateManager(repo_root=repo, extension_dir=repo / "extension")
    checked = mgr.check_for_updates(auto_apply=True)
    assert checked.has_remote is False
    assert checked.update_available is False
    assert checked.commits_behind == 0
    assert checked.last_error is None
    assert len(checked.extension_build_id) == 16

    applied = mgr.apply_update()
    assert applied.ok is False
    assert applied.updated is False
    assert applied.restart_scheduled is False


def test_fastapi_update_endpoints_and_health_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLT_AUTO_RESTART", raising=False)
    _bare, seed, local = _setup_bare_and_clones(tmp_path)

    cfg = AppConfig(
        project_root=local,
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        llama_bin_dir=tmp_path / "bin" / "llama-cpp",
        auto_update_enabled=True,
        auto_update_check_interval_sec=1800,
    )
    app = create_app(
        cfg=cfg,
        cache=TranslationCache(tmp_path / "cache" / "translations.db"),
        memory=ChapterMemoryStore(),
        auto_load_models=False,
    )

    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert set(health_data.keys()) >= {
            "status",
            "ready",
            "gpu_name",
            "vram_used_mb",
            "vram_total_mb",
            "llm_model",
            "llm_ready",
            "loading_stage",
            "extension_version",
            "extension_build_id",
            "has_remote",
            "update_available",
        }
        assert health_data["extension_version"] == "1.1.0"
        assert len(health_data["extension_build_id"]) == 16
        assert health_data["has_remote"] is True
        assert health_data["update_available"] is False

        status_resp = client.get("/update/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert set(status_data.keys()) == {
            "extension_version",
            "extension_build_id",
            "has_remote",
            "remote_url",
            "branch",
            "current_commit",
            "remote_commit",
            "commits_behind",
            "update_available",
            "auto_update_enabled",
            "working_tree_clean",
            "last_checked",
            "last_error",
            "updating",
            "restart_required",
        }
        old_build_id = status_data["extension_build_id"]

        # Push an upstream extension change
        (seed / "extension" / "background.js").write_text("console.log('v3');\n", encoding="utf-8")
        _git(seed, "add", ".")
        _git(seed, "commit", "-m", "v3")
        _git(seed, "push")

        # POST /update/check with empty body defaults to auto_apply=False
        check_resp = client.post("/update/check")
        assert check_resp.status_code == 200
        check_data = check_resp.json()
        assert check_data["commits_behind"] == 1
        assert check_data["update_available"] is True

        # GET /health now reflects update_available=True
        assert client.get("/health").json()["update_available"] is True

        # POST /update/apply pulls the update and updates extension_build_id
        apply_resp = client.post("/update/apply")
        assert apply_resp.status_code == 200
        apply_data = apply_resp.json()
        assert set(apply_data.keys()) == {
            "ok",
            "updated",
            "message",
            "extension_changed",
            "server_changed",
            "restart_scheduled",
            "status",
        }
        assert apply_data["ok"] is True
        assert apply_data["updated"] is True
        assert apply_data["extension_changed"] is True
        assert apply_data["server_changed"] is False
        assert apply_data["restart_scheduled"] is False
        assert apply_data["status"]["extension_build_id"] != old_build_id
        assert apply_data["status"]["update_available"] is False

        # Push another commit and test POST /update/check {"auto_apply": true}
        (seed / "extension" / "background.js").write_text("console.log('v4');\n", encoding="utf-8")
        _git(seed, "add", ".")
        _git(seed, "commit", "-m", "v4")
        _git(seed, "push")

        auto_check_resp = client.post("/update/check", json={"auto_apply": True})
        assert auto_check_resp.status_code == 200
        auto_data = auto_check_resp.json()
        assert auto_data["update_available"] is False
        assert auto_data["commits_behind"] == 0
        assert auto_data["extension_build_id"] != apply_data["status"]["extension_build_id"]


def test_update_manager_caches_git_snapshot_during_health_polls(tmp_path: Path) -> None:
    _bare, _seed, local = _setup_bare_and_clones(tmp_path)
    mgr = UpdateManager(repo_root=local, extension_dir=local / "extension")

    calls = 0
    orig_inspect = mgr._inspect_git_local

    def _counting_inspect():  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return orig_inspect()

    mgr._inspect_git_local = _counting_inspect  # type: ignore[method-assign]

    h1 = mgr.get_health_fields()
    h2 = mgr.get_health_fields()
    h3 = mgr.get_health_fields()
    assert h1["has_remote"] is True
    assert h1["extension_build_id"] == h2["extension_build_id"] == h3["extension_build_id"]
    assert calls == 1

    mgr.get_status()
    assert calls == 2

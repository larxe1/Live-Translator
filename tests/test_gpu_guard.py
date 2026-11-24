"""Unit tests for server.gpu_guard telemetry, VRAM headroom, thermal guard, and /health integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys

from fastapi.testclient import TestClient
import pytest

from server import gpu_guard as gg
from server.app import create_app
from server.cache import TranslationCache
from server.config import AppConfig
from server.translator.memory import ChapterMemoryStore


def _make_snap(
    *,
    available: bool = True,
    name: str = "NVIDIA GeForce RTX 3070",
    vram_used_mb: int = 2100,
    vram_total_mb: int = 8192,
    vram_free_mb: int = 5800,
    temp_c: int = 52,
    utilization_pct: int = 12,
    power_w: float = 45.0,
    power_limit_w: float = 240.0,
) -> gg.GpuSnapshot:
    return gg.GpuSnapshot(
        available=available,
        name=name,
        vram_used_mb=vram_used_mb,
        vram_total_mb=vram_total_mb,
        vram_free_mb=vram_free_mb,
        temp_c=temp_c,
        utilization_pct=utilization_pct,
        power_w=power_w,
        power_limit_w=power_limit_w,
        sampled_at=100.0,
    )


def test_get_gpu_snapshot_parses_nvidia_smi_and_caches(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3070, 2173, 8192, 5846, 47, 15, 55.93, 240.00\n",
        )

    monkeypatch.setattr(gg, "_snapshot_cache", None)
    monkeypatch.setattr(gg.subprocess, "run", fake_run)

    s1 = gg.get_gpu_snapshot(force_refresh=True)
    s2 = gg.get_gpu_snapshot()
    assert calls["n"] == 1
    assert s1.available is True
    assert s1.name == "NVIDIA GeForce RTX 3070"
    assert s1.vram_used_mb == 2173
    assert s1.vram_total_mb == 8192
    assert s1.vram_free_mb == 5846
    assert s1.temp_c == 47
    assert s1.utilization_pct == 15
    assert s1.power_w == pytest.approx(55.93)
    assert s1.power_limit_w == pytest.approx(240.0)
    assert s2 == s1

    gg.get_gpu_snapshot(force_refresh=True)
    assert calls["n"] == 2


def test_get_gpu_snapshot_never_raises_on_failure(monkeypatch) -> None:
    def failing_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi missing")

    monkeypatch.setattr(gg, "_snapshot_cache", None)
    monkeypatch.setattr(gg.subprocess, "run", failing_run)

    snap = gg.get_gpu_snapshot(force_refresh=True)
    assert snap.available is False
    assert snap.name == "Unavailable"
    assert snap.vram_free_mb == 0


def test_has_safe_vram_headroom_prefers_nvidia_smi_over_torch(monkeypatch) -> None:
    # Even if torch reports 7000 MiB free (inflated WDDM), nvidia-smi 900 MiB free must deny 1024.
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (7000 * 1024 * 1024, 8192 * 1024 * 1024),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(vram_free_mb=900))
    assert gg.has_safe_vram_headroom(1024) is False

    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(vram_free_mb=1500))
    assert gg.has_safe_vram_headroom(1024) is True


def test_has_safe_vram_headroom_falls_back_to_torch_when_nvidia_smi_unavailable(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (1536 * 1024 * 1024, 8192 * 1024 * 1024),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(available=False, vram_free_mb=0)
    )
    assert gg.has_safe_vram_headroom(1024) is True
    assert gg.has_safe_vram_headroom(2048) is False


def test_gpu_safety_guard_status_states(tmp_path: Path, monkeypatch) -> None:
    cfg = AppConfig(project_root=tmp_path)
    guard = gg.GpuSafetyGuard(cfg)

    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(available=False))
    assert guard.status()["state"] == "unavailable"

    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(temp_c=60))
    st_ok = guard.status()
    assert st_ok["state"] == "ok"
    assert st_ok["temp_c"] == 60
    assert st_ok["vram_free_mb"] == 5800

    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(temp_c=79))
    assert guard.status()["state"] == "throttled"

    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(temp_c=85))
    assert guard.status()["state"] == "paused"


def test_gpu_safety_guard_wait_if_hot_transitions(tmp_path: Path, monkeypatch) -> None:
    cfg = AppConfig(
        project_root=tmp_path,
        gpu_temp_soft_c=78,
        gpu_temp_hard_c=84,
        gpu_temp_resume_c=76,
        gpu_cooldown_seconds=0.01,
    )
    guard = gg.GpuSafetyGuard(cfg)

    # Normal temperature -> immediate "ok"
    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(temp_c=65))
    assert asyncio.run(guard.wait_if_hot()) == "ok"

    # Soft threshold -> "throttled"
    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(temp_c=80))
    assert asyncio.run(guard.wait_if_hot()) == "throttled"

    # Hard threshold -> polls until temp <= resume_temp_c -> "paused_cooled"
    seq = iter([_make_snap(temp_c=86), _make_snap(temp_c=80), _make_snap(temp_c=75)])
    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: next(seq))
    assert asyncio.run(guard.wait_if_hot(max_wait_s=2.0)) == "paused_cooled"

    # Hard threshold never cools -> "paused_timeout"
    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(temp_c=88))
    assert asyncio.run(guard.wait_if_hot(max_wait_s=0.06)) == "paused_timeout"

    # Cancelled during hard pause -> "cancelled"
    async def _run_cancel() -> str:
        ev = asyncio.Event()
        task = asyncio.create_task(guard.wait_if_hot(cancel_event=ev, max_wait_s=5.0))
        await asyncio.sleep(0.02)
        ev.set()
        return await task

    assert asyncio.run(_run_cancel()) == "cancelled"


def test_choose_safe_gpu_layers_full_offload_and_scaledown(tmp_path: Path, monkeypatch) -> None:
    cfg = AppConfig(project_root=tmp_path, gpu_llama_headroom_mb=600)
    guard = gg.GpuSafetyGuard(cfg)

    # Enough VRAM (4850 + 600 = 5450 <= 5600) -> keep full 99
    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(vram_free_mb=5600))
    assert guard.choose_safe_gpu_layers(99) == 99

    # Tight VRAM (3600 free -> 3000 budget -> int(37 * 3000 / 4850) = 22)
    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(vram_free_mb=3600))
    scaled = guard.choose_safe_gpu_layers(99)
    assert 0 < scaled < 37
    assert scaled == 22

    # Critical VRAM (<= headroom) -> 0
    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(vram_free_mb=500))
    assert guard.choose_safe_gpu_layers(99) == 0

    # Unavailable nvidia-smi -> keep requested
    monkeypatch.setattr(gg, "get_gpu_snapshot", lambda force_refresh=False: _make_snap(available=False))
    assert guard.choose_safe_gpu_layers(99) == 99


def test_health_endpoint_includes_gpu_guard_telemetry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        gg,
        "get_gpu_snapshot",
        lambda force_refresh=False: _make_snap(
            name="RTX 3070",
            vram_used_mb=2400,
            vram_total_mb=8192,
            vram_free_mb=5600,
            temp_c=64,
        ),
    )
    cfg = AppConfig(project_root=tmp_path)
    app = create_app(
        cfg=cfg,
        cache=TranslationCache(tmp_path / "t.db"),
        memory=ChapterMemoryStore(),
        auto_load_models=False,
    )
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gpu_name"] == "RTX 3070"
        assert data["vram_used_mb"] == 2400
        assert data["vram_total_mb"] == 8192
        assert data["gpu_guard"]["state"] == "ok"
        assert data["gpu_guard"]["temp_c"] == 64
        assert data["gpu_guard"]["vram_free_mb"] == 5600


def test_detector_uses_gpu_guard_headroom(monkeypatch) -> None:
    import torch
    from server.vision.detector import BubbleDetector

    seen_thresholds: list[int] = []

    def fake_headroom(min_free_mb: int) -> bool:
        seen_thresholds.append(min_free_mb)
        return False

    class FakeModel:
        def eval(self):
            return self

        def to(self, dev):
            return self

    monkeypatch.setattr(gg, "has_safe_vram_headroom", fake_headroom)
    monkeypatch.setattr(
        "transformers.AutoImageProcessor.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForObjectDetection.from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )

    det = BubbleDetector(device="cuda")
    assert det.min_free_vram_mb == 650
    det.load()
    assert seen_thresholds == [650]
    assert det._runtime_device == "cpu"
    assert det._use_fp16 is False
    assert torch is not None


def test_headroom_and_layer_selection_bypass_stale_2s_cache(tmp_path: Path, monkeypatch) -> None:
    # Seed the 2s TTL cache with pre-load high free VRAM (6000 MiB)
    outputs = iter([
        SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3070, 2000, 8192, 6000, 50, 5, 40.0, 240.0\n",
        ),
        SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3070, 7200, 8192, 800, 62, 70, 120.0, 240.0\n",
        ),
        SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3070, 2200, 8192, 5800, 48, 5, 38.0, 240.0\n",
        ),
    ])
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return next(outputs)

    monkeypatch.setattr(gg, "_snapshot_cache", None)
    monkeypatch.setattr(gg.subprocess, "run", fake_run)

    # Cached read (e.g. from /health) populates 6000 MiB free
    assert gg.get_gpu_snapshot().vram_free_mb == 6000
    assert calls["n"] == 1

    # Allocation check immediately afterward (<2s) MUST bypass stale 6000 MiB cache and see 800 MiB free
    assert gg.has_safe_vram_headroom(1024) is False
    assert calls["n"] == 2

    # Next layer selection immediately afterward (<2s) MUST bypass stale 800 MiB cache and see 5800 MiB free
    guard = gg.GpuSafetyGuard(AppConfig(project_root=tmp_path, gpu_llama_headroom_mb=600))
    assert guard.choose_safe_gpu_layers(99) == 99
    assert calls["n"] == 3


def test_health_and_settings_endpoints_manage_env_without_leaking_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import json

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        gg,
        "get_gpu_snapshot",
        lambda force_refresh=False: _make_snap(name="RTX 3070"),
    )

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "shisa-v2-qwen2.5-7b.i1-Q4_K_M.gguf").write_bytes(b"shisa")

    cfg = AppConfig(project_root=tmp_path)
    app = create_app(
        cfg=cfg,
        cache=TranslationCache(tmp_path / "t.db"),
        memory=ChapterMemoryStore(),
        auto_load_models=False,
    )

    with TestClient(app) as client:
        r0 = client.get("/health")
        assert r0.status_code == 200
        d0 = r0.json()
        assert d0["translation_mode"] == "hybrid"
        assert d0["gemini_configured"] is False
        assert d0["gemini_key_hint"] == ""
        assert d0["active_local_model"] == ""
        assert d0["available_local_models"]["ja"] is True
        assert d0["available_local_models"]["ko"] is False

        # Save API key + switch mode via POST /settings
        secret = "FAKE_TEST_GEMINI_TOKEN_9876"
        r1 = client.post(
            "/settings",
            json={"translation_mode": "gemini", "gemini_api_key": secret},
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1 == {
            "ok": True,
            "translation_mode": "gemini",
            "gemini_configured": True,
            "gemini_key_hint": "FAKE...9876",
        }
        assert secret not in r1.text
        assert app.state.cfg.translation_mode == "gemini"
        assert app.state.llm_lazy is True

        # Verify key is ONLY in .env, never in config.json
        env_text = (tmp_path / ".env").read_text(encoding="utf-8")
        assert f"GEMINI_API_KEY={secret}" in env_text
        cfg_json = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert cfg_json == {"translation_mode": "gemini"}
        assert secret not in (tmp_path / "config.json").read_text(encoding="utf-8")

        # /health reflects configured hint without returning raw key
        r2 = client.get("/health")
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["translation_mode"] == "gemini"
        assert d2["gemini_configured"] is True
        assert d2["gemini_key_hint"] == "FAKE...9876"
        assert secret not in r2.text

        # Invalid mode rejected cleanly
        r_bad = client.post("/settings", json={"translation_mode": "cloud_only"})
        assert r_bad.status_code == 422

        # Clear key via POST /settings
        r3 = client.post(
            "/settings",
            json={"translation_mode": "local", "clear_gemini_api_key": True},
        )
        assert r3.status_code == 200
        assert r3.json() == {
            "ok": True,
            "translation_mode": "local",
            "gemini_configured": False,
            "gemini_key_hint": "",
        }
        assert app.state.llm_lazy is False


def test_hybrid_mode_skips_eager_llama_startup_when_gemini_key_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import server.app as app_mod
    import server.ocr.engine as ocr_mod
    import server.vision.detector as det_mod

    start_calls: list[str] = []

    class FakeLlamaMgr:
        def __init__(self, cfg, gpu_guard=None):
            self.cfg = cfg
            self.base_url = "http://127.0.0.1:8766"
            self.model_name = "shisa-v2-qwen2.5-7b.i1-Q4_K_M.gguf"
            self.active_model_filename = ""
            self.active_model_role = ""

        def start(self, *args, **kwargs):
            start_calls.append("start")
            self.active_model_filename = "shisa-v2-qwen2.5-7b.i1-Q4_K_M.gguf"
            self.active_model_role = "ja"

        def stop(self):
            pass

        def is_healthy(self):
            return bool(self.active_model_filename)

        def available_local_models(self):
            return {"ja": False, "ko": False}

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def load(self):
            pass

    class FakeOcr:
        def __init__(self, **kwargs):
            pass

        def load_ja(self):
            pass

    monkeypatch.setattr("server.translator.llama_manager.LlamaServerManager", FakeLlamaMgr)
    monkeypatch.setattr(det_mod, "BubbleDetector", FakeDetector)
    monkeypatch.setattr(ocr_mod, "OcrEngine", FakeOcr)

    # 1. Hybrid mode WITH Gemini key -> llama-server NOT eagerly started!
    monkeypatch.setenv("GEMINI_API_KEY", "TEST_HYBRID_KEY_123456")
    cfg_hybrid = AppConfig(project_root=tmp_path, translation_mode="hybrid")
    app_hybrid = create_app(
        cfg=cfg_hybrid,
        cache=TranslationCache(tmp_path / "h.db"),
        memory=ChapterMemoryStore(),
        auto_load_models=False,
    )
    asyncio.run(app_mod._load_production_models(app_hybrid))
    assert start_calls == []
    assert app_hybrid.state.ready is True
    assert app_hybrid.state.loading_stage == "ready"
    assert app_hybrid.state.llm_lazy is True

    # _ensure_llm_ready must also keep llama-server lazy in hybrid mode
    asyncio.run(app_mod._ensure_llm_ready(app_hybrid))
    assert start_calls == []

    # 2. Local mode -> llama-server IS eagerly started!
    cfg_local = AppConfig(project_root=tmp_path, translation_mode="local")
    app_local = create_app(
        cfg=cfg_local,
        cache=TranslationCache(tmp_path / "l.db"),
        memory=ChapterMemoryStore(),
        auto_load_models=False,
    )
    asyncio.run(app_mod._load_production_models(app_local))
    assert start_calls == ["start"]
    assert app_local.state.ready is True
    assert app_local.state.llm_lazy is False


def test_lazy_hybrid_idle_watchdog_and_health_state_reset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import server.app as app_mod

    monkeypatch.setenv("GEMINI_API_KEY", "TEST_HYBRID_KEY_123456")
    monkeypatch.setattr(
        gg,
        "get_gpu_snapshot",
        lambda force_refresh=False: _make_snap(name="RTX 3070"),
    )

    class FakeProc:
        def __init__(self, alive: bool = True) -> None:
            self.alive = alive

        def poll(self):
            return None if self.alive else 0

    class FakeLazyLlamaMgr:
        def __init__(self) -> None:
            self._process: FakeProc | None = None
            self.active_model_filename = ""
            self.active_model_role = ""
            self.model_name = "shisa-v2-qwen2.5-7b.i1-Q4_K_M.gguf"
            self.stop_calls = 0

        def simulate_ensure_model(self, filename: str = "shisa-v2-qwen2.5-7b.i1-Q4_K_M.gguf") -> None:
            self._process = FakeProc(alive=True)
            self.active_model_filename = filename
            self.active_model_role = "ja"

        def is_healthy(self) -> bool:
            return self._process is not None and self._process.poll() is None

        def stop(self) -> None:
            self.stop_calls += 1
            self._process = None
            self.active_model_filename = ""
            self.active_model_role = ""

        def available_local_models(self) -> dict[str, bool]:
            return {"ja": True}

    mgr = FakeLazyLlamaMgr()
    cfg = AppConfig(project_root=tmp_path, translation_mode="hybrid")
    app = create_app(
        cfg=cfg,
        cache=TranslationCache(tmp_path / "watchdog.db"),
        memory=ChapterMemoryStore(),
        llama_manager=mgr,  # type: ignore[arg-type]
        auto_load_models=False,
    )
    app.state.ready = True
    app.state.loading_stage = "ready"
    app.state.llm_lazy = True

    # Simulate lazy load from explicit page #1, then idle watchdog unloading it
    mgr.simulate_ensure_model()
    app.state.last_activity = 0.0

    async def _run_watchdog_once() -> None:
        orig_sleep = asyncio.sleep
        ticks = {"n": 0}

        async def fast_sleep(_delay: float) -> None:
            ticks["n"] += 1
            if ticks["n"] > 1:
                raise asyncio.CancelledError
            await orig_sleep(0)

        monkeypatch.setattr(app_mod.asyncio, "sleep", fast_sleep)
        await app_mod._idle_watchdog_loop(app, idle_unload_minutes=1)

    asyncio.run(_run_watchdog_once())
    assert mgr.stop_calls == 1
    assert app.state.llm_idle_unloaded is False
    assert app.state.loading_stage == "ready"

    # Simulate explicit page #2 restarting via ensure_model(), plus stale llm_idle_unloaded=True
    app.state.llm_idle_unloaded = True
    mgr.simulate_ensure_model("Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf")
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["llm_ready"] is True
        assert d["active_local_model"] == "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf"
        assert app.state.llm_idle_unloaded is False

    # And POST /settings switching local (after idle_unload) -> hybrid clears idle state
    app.state.llm_idle_unloaded = True
    app.state.loading_stage = "idle_unloaded"
    with TestClient(app) as client:
        r_set = client.post("/settings", json={"translation_mode": "hybrid"})
        assert r_set.status_code == 200
        assert app.state.llm_lazy is True
        assert app.state.llm_idle_unloaded is False
        assert app.state.loading_stage == "ready"



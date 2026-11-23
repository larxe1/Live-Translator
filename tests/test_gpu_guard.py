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



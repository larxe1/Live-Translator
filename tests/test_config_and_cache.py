"""Unit tests for configuration, D:-drive cache enforcement, schemas, and LRU cache."""

from __future__ import annotations

import json
import os
from pathlib import Path

from server.cache import TranslationCache
from server.config import AppConfig, enforce_d_drive_env, load_config
from server.schemas import Box, BubbleResult, TranslateRequest, TranslateResponse
from server.translator.llama_manager import LlamaServerManager


def _sample_response(text: str = "Ranshin...?") -> TranslateResponse:
    return TranslateResponse(
        lang="ja",
        width=800,
        height=1200,
        elapsed_ms=950,
        cached=False,
        bubbles=[
            BubbleResult(
                id=0,
                bbox=Box(x1=10, y1=10, x2=100, y2=200),
                safe_box=Box(x1=16, y1=18, x2=94, y2=192),
                bg_color="#FFFFFF",
                text_color="#111111",
                is_dark=False,
                original="藍星……？",
                translated=text,
            )
        ],
    )


def test_enforce_d_drive_env_sets_all_caches_under_project_root(tmp_path: Path, monkeypatch) -> None:
    env_vars = (
        "HF_HOME",
        "TORCH_HOME",
        "EASYOCR_MODULE_PATH",
        "PIP_CACHE_DIR",
        "XDG_CACHE_HOME",
        "PLAYWRIGHT_BROWSERS_PATH",
    )
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

    cfg = load_config(project_root=tmp_path)
    env = enforce_d_drive_env(cfg)

    for var in env_vars:
        assert Path(env[var]).is_relative_to(tmp_path / "cache")
        assert Path(env[var]).is_dir()
        assert os.environ[var] == env[var]

    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8765
    assert cfg.llama_port == 8766
    assert cfg.llm_repo_id == "richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF"
    assert cfg.llm_filename == "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf"
    assert cfg.llm_filename_pattern == "*Qwen2.5-7B*Q4_K_M*.gguf"
    assert cfg.llm_gpu_layers == 99
    assert cfg.llm_ctx_size == 2048
    assert cfg.llm_kv_cache_type == "q8_0"
    assert cfg.llm_batch_size == 256
    assert cfg.llm_chunk_size == 12
    assert cfg.use_story_context is True
    assert cfg.max_context_pairs == 6
    assert cfg.gpu_min_free_vram_mb == 650
    assert cfg.gpu_llama_headroom_mb == 550
    assert cfg.gpu_temp_soft_c == 78
    assert cfg.gpu_temp_hard_c == 84
    assert cfg.gpu_temp_resume_c == 76
    assert cfg.gpu_cooldown_seconds == 1.5
    assert cfg.allow_lan is False
    assert cfg.bind_host == "127.0.0.1"
    assert cfg.max_image_bytes == 25_000_000
    assert cfg.max_image_width == 2400
    assert cfg.max_queue_size == 12
    assert cfg.idle_unload_minutes == 3


def test_repo_config_json_has_qwen25_defaults() -> None:
    cfg = load_config()
    assert cfg.llm_repo_id == "richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF"
    assert cfg.llm_filename == "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf"
    assert cfg.llm_filename_pattern == "*Qwen2.5-7B*Q4_K_M*.gguf"
    assert cfg.llm_gpu_layers == 99
    assert cfg.llm_ctx_size == 2048
    assert cfg.llm_kv_cache_type == "q8_0"
    assert cfg.llm_batch_size == 256
    assert cfg.llm_chunk_size == 12
    assert cfg.use_story_context is True
    assert cfg.max_context_pairs == 6
    assert cfg.gpu_min_free_vram_mb == 650
    assert cfg.gpu_llama_headroom_mb == 550
    assert cfg.idle_unload_minutes == 3
    assert TranslateRequest(image_base64="abc").include_floating is False


def test_load_config_reads_custom_json_overrides(tmp_path: Path) -> None:
    custom_cfg = tmp_path / "custom.json"
    custom_cfg.write_text(
        json.dumps(
            {
                "port": 9001,
                "llama_port": 9002,
                "llm_ctx_size": 8192,
                "max_cache_entries": 500,
                "allow_lan": True,
                "max_image_bytes": 10_000_000,
                "max_image_width": 1800,
                "max_queue_size": 6,
                "idle_unload_minutes": 10,
                "llm_model_path": "models/custom.gguf",
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(path=custom_cfg, project_root=tmp_path)
    assert cfg.port == 9001
    assert cfg.llama_port == 9002
    assert cfg.llm_ctx_size == 8192
    assert cfg.max_cache_entries == 500
    assert cfg.allow_lan is True
    assert cfg.bind_host == "0.0.0.0"
    assert cfg.max_image_bytes == 10_000_000
    assert cfg.max_image_width == 1800
    assert cfg.max_queue_size == 6
    assert cfg.idle_unload_minutes == 10
    assert cfg.llm_model_path == (tmp_path / "models" / "custom.gguf").resolve()


def test_cache_roundtrip_and_glossary_invalidation(tmp_path: Path) -> None:
    cache = TranslationCache(tmp_path / "translations.db", max_entries=5)
    resp = _sample_response("Ranshin...?")

    k1 = cache.make_key(b"fake-jpg", "ja", {"藍星": "Ranshin"})
    k2 = cache.make_key(b"fake-jpg", "ja", {"藍星": "Aiboshi"})
    k3 = cache.make_key(b"fake-jpg", "ko", {"藍星": "Ranshin"})
    assert len({k1, k2, k3}) == 3

    cache.put(k1, resp)
    hit = cache.get(k1)
    assert hit is not None
    assert hit.cached is True
    assert hit.bubbles[0].translated == "Ranshin...?"
    assert cache.get(k2) is None


def test_cache_lru_eviction_and_clear(tmp_path: Path) -> None:
    cache = TranslationCache(tmp_path / "lru.db", max_entries=2)
    k1 = cache.make_key(b"page-1", "ja")
    k2 = cache.make_key(b"page-2", "ja")
    k3 = cache.make_key(b"page-3", "ja")

    cache.put(k1, _sample_response("Page 1"))
    cache.put(k2, _sample_response("Page 2"))
    # Access k1 so k2 becomes the least-recently-used item
    assert cache.get(k1) is not None

    cache.put(k3, _sample_response("Page 3"))
    assert cache.get(k1) is not None
    assert cache.get(k2) is None
    assert cache.get(k3) is not None

    cache.clear()
    assert cache.get(k1) is None
    assert cache.get(k3) is None


def test_llama_manager_build_command_flags_and_ctx_cap(tmp_path: Path, monkeypatch) -> None:
    from server import gpu_guard

    monkeypatch.setattr(
        gpu_guard,
        "get_gpu_snapshot",
        lambda force_refresh=False: gpu_guard.GpuSnapshot(
            available=True,
            name="RTX 3070",
            vram_used_mb=2000,
            vram_total_mb=8192,
            vram_free_mb=6000,
            temp_c=50,
            utilization_pct=0,
            power_w=30.0,
            power_limit_w=240.0,
            sampled_at=0.0,
        ),
    )

    exe = tmp_path / "bin" / "llama-cpp" / "llama-server.exe"
    gguf = tmp_path / "models" / "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf"

    cfg_default = AppConfig(project_root=tmp_path)
    cmd_default = LlamaServerManager(cfg_default).build_command(exe, gguf)
    assert cmd_default[:2] == [str(exe), "--model"]
    assert cmd_default[2] == str(gguf)
    for flag, val in [
        ("--host", "127.0.0.1"),
        ("--port", "8766"),
        ("-ngl", "99"),
        ("--ctx-size", "2048"),
        ("--batch-size", "256"),
        ("--ubatch-size", "256"),
        ("-ctk", "q8_0"),
        ("-ctv", "q8_0"),
        ("--parallel", "1"),
        ("--flash-attn", "on"),
    ]:
        idx = cmd_default.index(flag)
        assert cmd_default[idx + 1] == val
    assert "--cache-prompt" in cmd_default
    assert "--reasoning" not in cmd_default

    cfg_qwen3 = AppConfig(
        project_root=tmp_path,
        llm_filename="mlabonne_Qwen3-8B-abliterated-Q4_K_M.gguf",
    )
    gguf_qwen3 = tmp_path / "models" / "mlabonne_Qwen3-8B-abliterated-Q4_K_M.gguf"
    cmd_qwen3 = LlamaServerManager(cfg_qwen3).build_command(exe, gguf_qwen3)
    assert cmd_qwen3[cmd_qwen3.index("--reasoning") + 1] == "off"

    cfg_4096 = AppConfig(project_root=tmp_path, llm_ctx_size=4096)
    cmd_4096 = LlamaServerManager(cfg_4096).build_command(exe, gguf)
    assert cmd_4096[cmd_4096.index("--ctx-size") + 1] == "4096"

    cfg_8192 = AppConfig(project_root=tmp_path, llm_ctx_size=8192, llm_kv_cache_type="")
    cmd_8192 = LlamaServerManager(cfg_8192).build_command(exe, gguf)
    assert cmd_8192[cmd_8192.index("--ctx-size") + 1] == "4096"
    assert "-ctk" not in cmd_8192
    assert "-ctv" not in cmd_8192


def test_llama_manager_estimates_vram_and_layers_from_gguf(tmp_path: Path, monkeypatch) -> None:
    seen: list[tuple[int, int, int]] = []

    class FakeGuard:
        def choose_safe_gpu_layers(
            self,
            requested_layers: int,
            total_layers: int = 37,
            estimated_model_mb: int = 4850,
        ) -> int:
            seen.append((requested_layers, total_layers, estimated_model_mb))
            return requested_layers

    exe = tmp_path / "bin" / "llama-cpp" / "llama-server.exe"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    missing_qwen25 = models_dir / "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf"
    mgr = LlamaServerManager(AppConfig(project_root=tmp_path), gpu_guard=FakeGuard())
    mgr.build_command(exe, missing_qwen25)
    assert seen[-1] == (99, 29, 4500)

    existing_qwen3 = models_dir / "mlabonne_Qwen3-8B-abliterated-Q4_K_M.gguf"
    existing_qwen3.write_bytes(b"\x00" * (10 * 1024 * 1024))
    mgr_qwen3 = LlamaServerManager(
        AppConfig(project_root=tmp_path, llm_filename=existing_qwen3.name),
        gpu_guard=FakeGuard(),
    )
    mgr_qwen3.build_command(exe, existing_qwen3)
    assert seen[-1] == (99, 37, 210)


def test_llama_manager_resolve_model_path_pattern_priority_and_fallback(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    qwen3 = models_dir / "mlabonne_Qwen3-8B-abliterated-Q4_K_M.gguf"
    qwen3.write_bytes(b"qwen3")

    cfg = AppConfig(project_root=tmp_path)
    mgr = LlamaServerManager(cfg)
    # Graceful fallback to ggufs[0] when default Qwen2.5 file is absent
    assert mgr.resolve_model_path() == qwen3

    qwen25 = models_dir / "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf"
    qwen25.write_bytes(b"qwen25")
    # Default Qwen2.5 match takes priority over Qwen3 regardless of sort order
    assert mgr.resolve_model_path() == qwen25

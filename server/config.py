"""Application configuration and D:-drive cache path enforcement."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:
    """Runtime configuration for the local manga translation server."""

    project_root: Path
    host: str = "127.0.0.1"
    port: int = 8765
    llama_port: int = 8766
    detector_model_id: str = "ogkalu/comic-text-and-bubble-detector"
    llm_repo_id: str = "richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF"
    llm_filename_pattern: str = "*Q4_K_M*.gguf"
    llm_model_path: Path | None = None
    llm_ctx_size: int = 4096
    llm_gpu_layers: int = 23
    max_cache_entries: int = 2000
    allow_lan: bool = False
    max_image_bytes: int = 25_000_000
    max_image_width: int = 2400
    max_queue_size: int = 12
    idle_unload_minutes: int = 20
    cache_dir: Path | None = None
    models_dir: Path | None = None
    llama_bin_dir: Path | None = None

    @property
    def bind_host(self) -> str:
        """Bind 0.0.0.0 only when allow_lan=True; otherwise strictly 127.0.0.1."""
        return "0.0.0.0" if self.allow_lan else self.host

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        if self.cache_dir is None:
            self.cache_dir = self.project_root / "cache"
        else:
            self.cache_dir = _resolve_path(self.cache_dir, self.project_root)

        if self.models_dir is None:
            self.models_dir = self.project_root / "models"
        else:
            self.models_dir = _resolve_path(self.models_dir, self.project_root)

        if self.llama_bin_dir is None:
            self.llama_bin_dir = self.project_root / "bin" / "llama-cpp"
        else:
            self.llama_bin_dir = _resolve_path(self.llama_bin_dir, self.project_root)

        if self.llm_model_path is not None:
            self.llm_model_path = _resolve_path(self.llm_model_path, self.project_root)


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(
    path: Path | None = None,
    project_root: Path | None = None,
) -> AppConfig:
    """Load AppConfig from JSON, defaulting directories under project_root."""
    resolved_root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    cfg_path = Path(path).resolve() if path is not None else (resolved_root / "config.json")

    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    return AppConfig(
        project_root=resolved_root,
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 8765)),
        llama_port=int(raw.get("llama_port", 8766)),
        detector_model_id=str(
            raw.get("detector_model_id", "ogkalu/comic-text-and-bubble-detector")
        ),
        llm_repo_id=str(
            raw.get("llm_repo_id", "richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF")
        ),
        llm_filename_pattern=str(raw.get("llm_filename_pattern", "*Q4_K_M*.gguf")),
        llm_model_path=raw.get("llm_model_path"),
        llm_ctx_size=int(raw.get("llm_ctx_size", 4096)),
        llm_gpu_layers=int(raw.get("llm_gpu_layers", 23)),
        max_cache_entries=int(raw.get("max_cache_entries", 2000)),
        allow_lan=bool(raw.get("allow_lan", False)),
        max_image_bytes=int(raw.get("max_image_bytes", 25_000_000)),
        max_image_width=int(raw.get("max_image_width", 2400)),
        max_queue_size=int(raw.get("max_queue_size", 12)),
        idle_unload_minutes=int(raw.get("idle_unload_minutes", 20)),
        cache_dir=raw.get("cache_dir"),
        models_dir=raw.get("models_dir"),
        llama_bin_dir=raw.get("llama_bin_dir"),
    )


def enforce_d_drive_env(cfg: AppConfig) -> dict[str, str]:
    """Create D:-drive cache directories and pin all ML/pip cache env vars to them."""
    assert cfg.cache_dir is not None
    assert cfg.models_dir is not None
    assert cfg.llama_bin_dir is not None

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.llama_bin_dir.mkdir(parents=True, exist_ok=True)

    env_paths = {
        "HF_HOME": cfg.cache_dir / "huggingface",
        "TORCH_HOME": cfg.cache_dir / "torch",
        "EASYOCR_MODULE_PATH": cfg.cache_dir / "easyocr",
        "PIP_CACHE_DIR": cfg.cache_dir / "pip",
        "XDG_CACHE_HOME": cfg.cache_dir / "xdg",
        "PLAYWRIGHT_BROWSERS_PATH": cfg.cache_dir / "playwright",
    }

    applied: dict[str, str] = {}
    for var, dir_path in env_paths.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        str_path = str(dir_path)
        os.environ[var] = str_path
        applied[var] = str_path

    return applied

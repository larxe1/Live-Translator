"""Application configuration and D:-drive cache path enforcement."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any

DEFAULT_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-4-hydrogen",
)

DEFAULT_LOCAL_MODELS: dict[str, dict[str, str]] = {
    "ja": {
        "repo_id": "mradermacher/shisa-v2-qwen2.5-7b-i1-GGUF",
        "filename": "shisa-v2-qwen2.5-7b.i1-Q4_K_M.gguf",
    },
    "ja_uncensored": {
        "repo_id": "richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF",
        "filename": "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf",
    },
    "ko": {
        "repo_id": "mradermacher/EXAONE-3.5-7.8B-Instruct-abliterated-i1-GGUF",
        "filename": "EXAONE-3.5-7.8B-Instruct-abliterated.i1-IQ4_XS.gguf",
    },
    "default": {
        "repo_id": "richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF",
        "filename": "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf",
    },
}

VALID_TRANSLATION_MODES: tuple[str, ...] = ("hybrid", "local", "gemini")
_SECRET_CONFIG_KEYS: frozenset[str] = frozenset({"gemini_api_key", "GEMINI_API_KEY", "api_key"})


def _default_gemini_models() -> list[str]:
    return list(DEFAULT_GEMINI_MODELS)


def _default_local_models() -> dict[str, dict[str, str]]:
    return copy.deepcopy(DEFAULT_LOCAL_MODELS)


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_env_path(env_path: Path | None = None) -> Path:
    if env_path is not None:
        return Path(env_path).resolve()
    return (_default_project_root() / ".env").resolve()


def _parse_dotenv_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not path.is_file():
        return parsed
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        parsed[key] = value
    return parsed


def load_dotenv(env_path: Path | None = None) -> dict[str, str]:
    """Load key=value pairs from .env into os.environ without overwriting explicit env vars."""
    target = _resolve_env_path(env_path)
    parsed = _parse_dotenv_file(target)
    for key, value in parsed.items():
        os.environ.setdefault(key, value)
    return parsed


def get_gemini_api_key(env_path: Path | None = None) -> str:
    """Return GEMINI_API_KEY from process environment or .env, or empty string if unset."""
    if env_path is None:
        env_val = os.environ.get("GEMINI_API_KEY", "").strip()
        if env_val:
            return env_val
        parsed = load_dotenv()
        return parsed.get("GEMINI_API_KEY", "").strip()

    parsed = _parse_dotenv_file(_resolve_env_path(env_path))
    file_val = parsed.get("GEMINI_API_KEY", "").strip()
    if file_val:
        return file_val
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _atomic_write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_gemini_api_key(api_key: str, env_path: Path | None = None) -> None:
    """Atomically save or clear GEMINI_API_KEY in .env and os.environ (never touches config.json)."""
    cleaned = (api_key or "").strip()
    target = _resolve_env_path(env_path)

    existing_lines: list[str] = []
    if target.is_file():
        existing_lines = target.read_text(encoding="utf-8").splitlines()

    new_lines: list[str] = []
    replaced = False
    for line in existing_lines:
        stripped = line.strip()
        candidate = stripped[7:].strip() if stripped.startswith("export ") else stripped
        if candidate.startswith("GEMINI_API_KEY") and "=" in candidate and candidate.split("=", 1)[0].strip() == "GEMINI_API_KEY":
            if cleaned and not replaced:
                new_lines.append(f"GEMINI_API_KEY={cleaned}")
                replaced = True
            continue
        new_lines.append(line)

    if cleaned and not replaced:
        new_lines.append(f"GEMINI_API_KEY={cleaned}")

    body = "\n".join(new_lines) + ("\n" if new_lines else "")
    _atomic_write_text(target, body)

    if cleaned:
        os.environ["GEMINI_API_KEY"] = cleaned
    else:
        os.environ.pop("GEMINI_API_KEY", None)


def mask_api_key(api_key: str) -> str:
    """Return a masked display hint for an API key without exposing the secret."""
    k = (api_key or "").strip()
    if not k:
        return ""
    if len(k) >= 10:
        return f"{k[:4]}...{k[-4:]}"
    return "***"


@dataclass
class AppConfig:
    """Runtime configuration for the local manga translation server."""

    project_root: Path
    host: str = "127.0.0.1"
    port: int = 8765
    llama_port: int = 8766
    detector_model_id: str = "ogkalu/comic-text-and-bubble-detector"
    llm_repo_id: str = DEFAULT_LOCAL_MODELS["ja"]["repo_id"]
    llm_filename: str = DEFAULT_LOCAL_MODELS["ja"]["filename"]
    llm_filename_pattern: str = "*shisa-v2*Q4_K_M*.gguf"
    llm_model_path: Path | None = None
    translation_mode: str = "hybrid"
    gemini_models: list[str] = field(default_factory=_default_gemini_models)
    gemini_use_vision: bool = True
    gemini_chunk_size: int = 20
    gemini_timeout_s: float = 25.0
    gemini_cooldown_s: float = 60.0
    local_models: dict[str, dict[str, str]] = field(default_factory=_default_local_models)
    llm_ctx_size: int = 2048
    llm_gpu_layers: int = 99
    llm_kv_cache_type: str = "q8_0"
    llm_batch_size: int = 256
    llm_chunk_size: int = 12
    use_story_context: bool = True
    max_context_pairs: int = 6
    gpu_min_free_vram_mb: int = 650
    gpu_llama_headroom_mb: int = 550
    gpu_temp_soft_c: int = 78
    gpu_temp_hard_c: int = 84
    gpu_temp_resume_c: int = 76
    gpu_cooldown_seconds: float = 1.5
    max_cache_entries: int = 2000
    allow_lan: bool = False
    max_image_bytes: int = 25_000_000
    max_image_width: int = 2400
    max_queue_size: int = 12
    idle_unload_minutes: int = 3
    auto_update_enabled: bool = True
    auto_update_check_interval_sec: int = 1800
    cache_dir: Path | None = None
    models_dir: Path | None = None
    llama_bin_dir: Path | None = None

    @property
    def bind_host(self) -> str:
        """Bind 0.0.0.0 only when allow_lan=True; otherwise strictly 127.0.0.1."""
        return "0.0.0.0" if self.allow_lan else self.host

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        mode = str(self.translation_mode or "hybrid").strip().lower()
        self.translation_mode = mode if mode in VALID_TRANSLATION_MODES else "hybrid"
        if not self.gemini_models:
            self.gemini_models = _default_gemini_models()
        if not self.local_models:
            self.local_models = _default_local_models()

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


def _normalize_local_models(raw_models: Any) -> dict[str, dict[str, str]]:
    merged = _default_local_models()
    if not isinstance(raw_models, dict):
        return merged
    for role, entry in raw_models.items():
        if not isinstance(entry, dict):
            continue
        repo_id = str(entry.get("repo_id", "") or "").strip()
        filename = str(entry.get("filename", "") or "").strip()
        if repo_id and filename:
            merged[str(role).strip()] = {"repo_id": repo_id, "filename": filename}
    return merged


def resolve_local_model(
    cfg: AppConfig,
    lang: str,
    uncensored: bool = False,
) -> tuple[str, str, str]:
    """Resolve (role_key, repo_id, filename) for a target language and uncensored fallback."""
    models = getattr(cfg, "local_models", None) or DEFAULT_LOCAL_MODELS
    norm_lang = (lang or "").strip().lower()

    role_key = "default"
    if uncensored:
        uncensored_key = f"{norm_lang}_uncensored" if norm_lang else ""
        if uncensored_key and uncensored_key in models:
            role_key = uncensored_key
        elif norm_lang in ("", "auto", "ja") and "ja_uncensored" in models:
            role_key = "ja_uncensored"
        elif norm_lang in models:
            role_key = norm_lang
        elif "ja_uncensored" in models:
            role_key = "ja_uncensored"
    else:
        if norm_lang in models:
            role_key = norm_lang
        elif "default" in models:
            role_key = "default"

    entry = models.get(role_key) or models.get("default")
    if isinstance(entry, dict):
        repo_id = str(entry.get("repo_id") or cfg.llm_repo_id)
        filename = str(entry.get("filename") or cfg.llm_filename)
        return (role_key, repo_id, filename)
    return (role_key, cfg.llm_repo_id, cfg.llm_filename)


def load_config(
    path: Path | None = None,
    project_root: Path | None = None,
) -> AppConfig:
    """Load AppConfig from JSON and .env, defaulting directories under project_root."""
    resolved_root = (
        Path(project_root).resolve()
        if project_root is not None
        else _default_project_root()
    )
    load_dotenv(resolved_root / ".env")
    cfg_path = Path(path).resolve() if path is not None else (resolved_root / "config.json")

    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    raw_filename = raw.get("llm_filename")
    raw_pattern = raw.get("llm_filename_pattern")
    if raw_filename is not None:
        llm_filename = str(raw_filename)
        llm_filename_pattern = str(raw_pattern if raw_pattern is not None else raw_filename)
    elif raw_pattern is not None:
        llm_filename_pattern = str(raw_pattern)
        llm_filename = llm_filename_pattern
    else:
        llm_filename = DEFAULT_LOCAL_MODELS["ja"]["filename"]
        llm_filename_pattern = "*shisa-v2*Q4_K_M*.gguf"

    raw_gemini_models = raw.get("gemini_models")
    if isinstance(raw_gemini_models, list) and raw_gemini_models:
        gemini_models = [str(m).strip() for m in raw_gemini_models if str(m).strip()]
    else:
        gemini_models = _default_gemini_models()

    return AppConfig(
        project_root=resolved_root,
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 8765)),
        llama_port=int(raw.get("llama_port", 8766)),
        detector_model_id=str(
            raw.get("detector_model_id", "ogkalu/comic-text-and-bubble-detector")
        ),
        llm_repo_id=str(
            raw.get("llm_repo_id", DEFAULT_LOCAL_MODELS["ja"]["repo_id"])
        ),
        llm_filename=llm_filename,
        llm_filename_pattern=llm_filename_pattern,
        llm_model_path=raw.get("llm_model_path"),
        translation_mode=str(raw.get("translation_mode", "hybrid") or "hybrid"),
        gemini_models=gemini_models,
        gemini_use_vision=bool(raw.get("gemini_use_vision", True)),
        gemini_chunk_size=int(raw.get("gemini_chunk_size", 20)),
        gemini_timeout_s=float(raw.get("gemini_timeout_s", 25.0)),
        gemini_cooldown_s=float(raw.get("gemini_cooldown_s", 60.0)),
        local_models=_normalize_local_models(raw.get("local_models")),
        llm_ctx_size=int(raw.get("llm_ctx_size", 2048)),
        llm_gpu_layers=int(raw.get("llm_gpu_layers", 99)),
        llm_kv_cache_type=str(
            raw.get("llm_kv_cache_type", "q8_0")
            if raw.get("llm_kv_cache_type") is not None
            else "q8_0"
        ).strip(),
        llm_batch_size=int(raw.get("llm_batch_size", 256)),
        llm_chunk_size=int(raw.get("llm_chunk_size", 12)),
        use_story_context=bool(raw.get("use_story_context", True)),
        max_context_pairs=int(raw.get("max_context_pairs", 6)),
        gpu_min_free_vram_mb=int(raw.get("gpu_min_free_vram_mb", 650)),
        gpu_llama_headroom_mb=int(raw.get("gpu_llama_headroom_mb", 550)),
        gpu_temp_soft_c=int(raw.get("gpu_temp_soft_c", 78)),
        gpu_temp_hard_c=int(raw.get("gpu_temp_hard_c", 84)),
        gpu_temp_resume_c=int(raw.get("gpu_temp_resume_c", 76)),
        gpu_cooldown_seconds=float(raw.get("gpu_cooldown_seconds", 1.5)),
        max_cache_entries=int(raw.get("max_cache_entries", 2000)),
        allow_lan=bool(raw.get("allow_lan", False)),
        max_image_bytes=int(raw.get("max_image_bytes", 25_000_000)),
        max_image_width=int(raw.get("max_image_width", 2400)),
        max_queue_size=int(raw.get("max_queue_size", 12)),
        idle_unload_minutes=int(raw.get("idle_unload_minutes", 3)),
        auto_update_enabled=bool(raw.get("auto_update_enabled", True)),
        auto_update_check_interval_sec=int(raw.get("auto_update_check_interval_sec", 1800)),
        cache_dir=raw.get("cache_dir"),
        models_dir=raw.get("models_dir"),
        llama_bin_dir=raw.get("llama_bin_dir"),
    )


def save_config_updates(updates: dict[str, Any], path: Path | None = None) -> AppConfig:
    """Persist non-secret configuration updates atomically to config.json and reload AppConfig."""
    cfg_path = (
        Path(path).resolve()
        if path is not None
        else (_default_project_root() / "config.json").resolve()
    )
    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    for key, value in updates.items():
        if key in _SECRET_CONFIG_KEYS:
            continue
        if key == "translation_mode":
            mode = str(value or "").strip().lower()
            if mode not in VALID_TRANSLATION_MODES:
                raise ValueError(f"Invalid translation_mode: {value!r}")
            raw[key] = mode
        else:
            raw[key] = value

    _atomic_write_text(cfg_path, json.dumps(raw, indent=2) + "\n")
    return load_config(path=cfg_path, project_root=cfg_path.parent)


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

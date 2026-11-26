from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import gc
import inspect
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
import uvicorn

from server import gpu_guard
from server.cache import TranslationCache
from server.config import (
    AppConfig,
    VALID_TRANSLATION_MODES,
    enforce_d_drive_env,
    get_gemini_api_key,
    load_config,
    mask_api_key,
    save_config_updates,
    save_gemini_api_key,
)
from server.gpu_guard import GpuSafetyGuard
from server.pipeline import PayloadTooLargeError, TranslationPipeline
from server.schemas import (
    GlossaryLookupRequest,
    GlossaryLookupResponse,
    GlossaryUpdateRequest,
    HealthResponse,
    MemoryClearRequest,
    SettingsUpdateRequest,
    SettingsUpdateResponse,
    TranslateRequest,
    TranslateResponse,
    UpdateApplyResponse,
    UpdateCheckRequest,
    UpdateStatusResponse,
)
from server.translator.memory import ChapterMemoryStore
from server.translator.wiki_glossary import (
    GlossaryLookupError,
    InvalidLookupQueryError,
    lookup_character_glossary,
)
from server.updater import UpdateManager

if TYPE_CHECKING:
    from server.translator.llama_manager import LlamaServerManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class CancelRequest(BaseModel):
    request_id: str


def _query_gpu_stats_sync() -> tuple[str, int, int]:
    """Query live GPU name and VRAM usage via nvidia-smi safely (cached briefly)."""
    snap = gpu_guard.get_gpu_snapshot()
    if not snap.available:
        return ("Unavailable", 0, 0)
    return (snap.name, snap.vram_used_mb, snap.vram_total_mb)


def _accepts_kwarg(fn: Any, kwarg_name: str) -> bool:
    """Return True when fn accepts kwarg_name explicitly or via **kwargs."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return kwarg_name in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _configure_server_logging(cfg: AppConfig) -> None:
    """Ensure server.* INFO logs emit to stderr (captured in cache/server.log) cleanly."""
    server_logger = logging.getLogger("server")
    server_logger.setLevel(logging.INFO)
    if not server_logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        server_logger.addHandler(stream_handler)
        try:
            is_tty = sys.stderr is None or sys.stderr.isatty()
        except Exception:
            is_tty = False
        if is_tty and cfg.cache_dir is not None:
            cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(cfg.cache_dir / "server.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            server_logger.addHandler(file_handler)
        server_logger.propagate = False


def _decode_base64_image(raw_b64: str, max_image_bytes: int) -> bytes:
    """Decode raw base64 or data-URL image string to raw bytes, enforcing max_image_bytes."""
    payload = raw_b64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    if not payload:
        raise ValueError("Empty image_base64 payload")
    # Guard against multi-hundred-MB base64 strings before decoding in memory.
    max_b64_chars = int(max_image_bytes * 1.4) + 1024
    if len(payload) > max_b64_chars:
        raise PayloadTooLargeError(f"Base64 image exceeds max_image_bytes ({max_image_bytes})")
    decoded = base64.b64decode(payload, validate=False)
    if len(decoded) > max_image_bytes:
        raise PayloadTooLargeError(
            f"Decoded image ({len(decoded)} bytes) exceeds max_image_bytes ({max_image_bytes})"
        )
    return decoded


def _parse_bool_form(raw: Any, default: bool = True) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if not s:
        return default
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    raise HTTPException(status_code=422, detail=f"Invalid boolean value: {raw!r}")


def _scan_available_local_models(cfg: AppConfig) -> dict[str, bool]:
    """Return {role: is_downloaded} by checking configured local_models in cfg.models_dir."""
    models_dir = cfg.models_dir or (cfg.project_root / "models")
    present_names = (
        {p.name.lower() for p in models_dir.rglob("*.gguf") if p.is_file()}
        if models_dir.is_dir()
        else set()
    )
    local_models = getattr(cfg, "local_models", None) or {}
    result: dict[str, bool] = {}
    for role, entry in local_models.items():
        if isinstance(entry, dict):
            fname = str(entry.get("filename", "") or "").strip().lower()
            result[str(role)] = bool(fname and fname in present_names)
    return result


def _normalize_optional_mode(raw: Any) -> str | None:
    if raw is None:
        return None
    mode = str(raw).strip().lower()
    if not mode:
        return None
    if mode not in VALID_TRANSLATION_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"translation_mode must be one of {VALID_TRANSLATION_MODES}",
        )
    return mode


async def _parse_translate_request(
    request: Request,
    max_image_bytes: int,
) -> tuple[bytes, Literal["auto", "ja", "ko"], str, int, dict[str, str], str | None, bool, str | None, str | None]:
    """Parse multipart or JSON translation request and enforce payload limits."""
    content_length_hdr = request.headers.get("content-length")
    if content_length_hdr:
        try:
            # Allow generous envelope overhead for JSON base64 / multipart headers.
            if int(content_length_hdr) > int(max_image_bytes * 1.5) + 65536:
                raise HTTPException(
                    status_code=413,
                    detail=f"Payload exceeds max_image_bytes ({max_image_bytes})",
                )
        except ValueError:
            pass

    content_type = request.headers.get("content-type", "").lower()
    lang: Literal["auto", "ja", "ko"] = "auto"
    chapter_id = "default"
    priority = 0
    glossary: dict[str, str] = {}
    request_id: str | None = None
    include_floating = True
    series_id: str | None = None
    translation_mode: str | None = None
    image_bytes: bytes = b""

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="Missing 'file' upload in multipart form")
        image_bytes = await upload.read()
        if len(image_bytes) > max_image_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded image ({len(image_bytes)} bytes) exceeds limit ({max_image_bytes})",
            )
        lang_raw = str(form.get("lang") or "auto")
        if lang_raw not in ("auto", "ja", "ko"):
            raise HTTPException(status_code=422, detail="lang must be 'auto', 'ja', or 'ko'")
        lang = lang_raw  # type: ignore[assignment]
        chapter_id = str(form.get("chapter_id") or "default")
        try:
            priority = int(form.get("priority") or 0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="priority must be an integer") from exc
        req_id_raw = form.get("request_id")
        request_id = str(req_id_raw) if req_id_raw else None
        include_floating = _parse_bool_form(form.get("include_floating"), default=True)
        series_raw = form.get("series_id")
        series_id = str(series_raw).strip() or None if series_raw is not None else None
        translation_mode = _normalize_optional_mode(form.get("translation_mode"))
        glossary_raw = form.get("glossary")
        if glossary_raw:
            try:
                parsed = json.loads(str(glossary_raw))
                if not isinstance(parsed, dict):
                    raise ValueError("glossary must be a JSON object")
                glossary = {str(k): str(v) for k, v in parsed.items()}
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Invalid glossary JSON: {exc}") from exc
    else:
        try:
            body = await request.json()
            req = TranslateRequest.model_validate(body)
        except (ValidationError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid translation request: {exc}") from exc

        try:
            image_bytes = _decode_base64_image(req.image_base64, max_image_bytes=max_image_bytes)
        except PayloadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc

        lang = req.lang
        chapter_id = req.chapter_id
        priority = req.priority
        glossary = req.glossary
        request_id = req.request_id
        include_floating = req.include_floating
        series_id = req.series_id.strip() or None if req.series_id is not None else None
        translation_mode = _normalize_optional_mode(req.translation_mode)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload")

    return (
        image_bytes,
        lang,
        chapter_id,
        priority,
        glossary,
        request_id,
        include_floating,
        series_id,
        translation_mode,
    )


def _detect_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


_LOCAL_PROGRESS_STEPS: dict[str, tuple[int, int, float, str]] = {
    "initializing": (0, 5, 2.0, "Initializing engine"),
    "starting_llm": (5, 35, 15.0, "Starting local LLM"),
    "loading_detector": (35, 38, 1.0, "Loading bubble detector"),
    "importing_transformers": (38, 46, 4.0, "Importing vision libraries"),
    "loading_processor": (46, 52, 2.0, "Loading detector processor"),
    "loading_weights": (52, 63, 5.0, "Loading detector weights"),
    "moving_to_gpu": (63, 68, 2.0, "Moving detector to GPU"),
    "loading_ocr": (68, 70, 1.0, "Loading OCR engine"),
    "loading_ocr_model": (70, 86, 8.0, "Loading Japanese OCR"),
    "warming_ocr_gpu": (86, 93, 3.0, "Warming up OCR on GPU"),
    "initializing_pipeline": (93, 99, 2.0, "Initializing translation pipeline"),
}

_LAZY_PROGRESS_STEPS: dict[str, tuple[int, int, float, str]] = {
    "initializing": (0, 8, 2.0, "Initializing engine"),
    "loading_detector": (8, 12, 1.0, "Loading bubble detector"),
    "importing_transformers": (12, 24, 4.0, "Importing vision libraries"),
    "loading_processor": (24, 33, 2.0, "Loading detector processor"),
    "loading_weights": (33, 48, 5.0, "Loading detector weights"),
    "moving_to_gpu": (48, 55, 2.0, "Moving detector to GPU"),
    "loading_ocr": (55, 58, 1.0, "Loading OCR engine"),
    "loading_ocr_model": (58, 80, 8.0, "Loading Japanese OCR"),
    "warming_ocr_gpu": (80, 90, 3.0, "Warming up OCR on GPU"),
    "initializing_pipeline": (90, 99, 2.0, "Initializing translation pipeline"),
}

_RESTART_PROGRESS_STEP: tuple[int, int, float, str] = (
    5,
    100,
    12.0,
    "Restarting local LLM",
)

_SUBSTEP_TO_TOP_STAGE: dict[str, str] = {
    "importing_transformers": "loading_detector",
    "loading_processor": "loading_detector",
    "loading_weights": "loading_detector",
    "moving_to_gpu": "loading_detector",
    "loading_ocr_model": "loading_ocr",
    "warming_ocr_gpu": "loading_ocr",
}


class _StartupProgressTracker:
    """Thread-safe monotonic startup progress tracker with asymptotic intra-step interpolation."""

    def __init__(
        self,
        *,
        lazy_llm: bool = False,
        stage: str = "unloaded",
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = threading.Lock()
        self._lazy_llm = bool(lazy_llm)
        self._now_fn = now_fn
        self._step = stage
        self._step_started_at = float(now_fn())
        self._high_water = 100 if stage == "ready" else 0
        self._frozen_progress: int | None = None
        if stage == "ready":
            self._detail = "Ready"
        elif stage == "unloaded":
            self._detail = ""
        else:
            _, _, _, default_detail = self._lookup_step_unlocked(stage)
            self._detail = default_detail

    def _lookup_step_unlocked(self, step: str) -> tuple[int, int, float, str]:
        if step == "restarting_llm":
            return _RESTART_PROGRESS_STEP
        table = _LAZY_PROGRESS_STEPS if self._lazy_llm else _LOCAL_PROGRESS_STEPS
        if step in table:
            return table[step]
        return (self._high_water, min(99, self._high_water + 5), 5.0, step or "")

    def _compute_unlocked(self) -> int:
        if self._step == "unloaded":
            return 0
        if self._step == "ready":
            return 100
        if self._step == "error":
            val = self._frozen_progress if self._frozen_progress is not None else self._high_water
            return max(0, min(100, int(val)))
        base, ceiling, expected_s, _ = self._lookup_step_unlocked(self._step)
        elapsed = max(0.0, float(self._now_fn()) - self._step_started_at)
        span = max(0, ceiling - base)
        cap = max(base, min(99, ceiling - 1))
        raw = base + round(span * (1.0 - math.exp(-elapsed / max(expected_s, 0.1))))
        clamped = max(0, min(cap, int(raw)))
        self._high_water = max(self._high_water, clamped)
        return max(0, min(99, self._high_water))

    @property
    def top_stage(self) -> str:
        with self._lock:
            return _SUBSTEP_TO_TOP_STAGE.get(self._step, self._step)

    def set_lazy_llm(self, lazy_llm: bool) -> None:
        with self._lock:
            self._lazy_llm = bool(lazy_llm)

    def set_step(self, step: str, detail: str | None = None) -> None:
        with self._lock:
            if step == "unloaded":
                self._step = "unloaded"
                self._high_water = 0
                self._frozen_progress = None
                self._detail = "" if detail is None else detail
                self._step_started_at = float(self._now_fn())
                return
            if step == "restarting_llm" and self._step != "restarting_llm":
                self._high_water = 0
            elif step == "initializing" and self._step == "unloaded":
                self._high_water = 0

            base, _, _, default_detail = self._lookup_step_unlocked(step)
            self._step = step
            self._step_started_at = float(self._now_fn())
            self._frozen_progress = None
            self._detail = default_detail if detail is None else detail
            self._high_water = max(self._high_water, max(0, min(99, base)))

    def mark_ready(self, detail: str = "Ready") -> None:
        with self._lock:
            self._step = "ready"
            self._high_water = 100
            self._frozen_progress = None
            self._detail = detail
            self._step_started_at = float(self._now_fn())

    def mark_error(self, detail: str = "") -> None:
        with self._lock:
            if self._step != "error":
                self._frozen_progress = self._compute_unlocked()
            self._step = "error"
            self._detail = detail

    def get_progress(self) -> tuple[int, str]:
        with self._lock:
            return self._compute_unlocked(), self._detail


def _get_loading_state(app: FastAPI) -> tuple[str, int, str]:
    """Return (loading_stage, loading_progress, loading_detail) safely."""
    stage = str(getattr(app.state, "loading_stage", "") or "")
    ready = bool(getattr(app.state, "ready", False))
    loading_error = getattr(app.state, "loading_error", None)
    tracker: _StartupProgressTracker | None = getattr(app.state, "progress_tracker", None)

    if loading_error or stage.startswith("error"):
        err_detail = str(loading_error or stage)
        if tracker is not None:
            tracker.mark_error(err_detail)
            progress, detail = tracker.get_progress()
            return (stage or "error", max(0, min(100, progress)), detail)
        return (stage or "error", 0, err_detail)

    if stage == "unloaded" and not ready:
        return ("unloaded", 0, "")

    if ready and stage != "restarting_llm":
        if tracker is not None and stage == "ready":
            tracker.mark_ready()
        detail = "Local LLM idle-unloaded" if stage == "idle_unloaded" else ("Ready" if stage == "ready" else "")
        return (stage or "ready", 100, detail)

    if tracker is not None:
        if stage and stage != tracker.top_stage and stage in (*_LOCAL_PROGRESS_STEPS, "restarting_llm"):
            tracker.set_step(stage)
        progress, detail = tracker.get_progress()
        return (stage, max(0, min(99, progress)), detail)

    return (stage, 0, "")


def _sanitize_header_value(value: str) -> str:
    return " ".join(str(value or "").splitlines()).strip()


def _raise_not_ready_503(app: FastAPI) -> NoReturn:
    stage, progress, detail = _get_loading_state(app)
    progress_clamped = max(0, min(99, int(progress)))
    raise HTTPException(
        status_code=503,
        detail=f"Models are not ready yet ({stage}, {progress_clamped}%)",
        headers={
            "X-MLT-Loading-Stage": _sanitize_header_value(stage),
            "X-MLT-Loading-Progress": str(progress_clamped),
            "X-MLT-Loading-Detail": _sanitize_header_value(detail),
        },
    )


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


async def _ensure_llm_ready(app: FastAPI) -> None:
    """Transparently start or restart llama-server when local inference is active."""
    app.state.last_activity = time.monotonic()
    mgr: LlamaServerManager | None = app.state.llama_manager
    if mgr is None or getattr(app.state, "llm_lazy", False):
        return
    if not app.state.llm_idle_unloaded and await asyncio.to_thread(mgr.is_healthy):
        return

    lock: asyncio.Lock = app.state.llm_restart_lock
    async with lock:
        if app.state.llama_manager is None or getattr(app.state, "llm_lazy", False):
            return
        if app.state.llm_idle_unloaded or not await asyncio.to_thread(app.state.llama_manager.is_healthy):
            tracker: _StartupProgressTracker | None = getattr(app.state, "progress_tracker", None)
            app.state.loading_stage = "restarting_llm"
            if tracker is not None:
                tracker.set_step("restarting_llm", "Restarting local LLM")
            await asyncio.to_thread(app.state.llama_manager.start)
            app.state.llm_idle_unloaded = False
            app.state.loading_stage = "ready"
            if tracker is not None:
                tracker.mark_ready()
            app.state.last_activity = time.monotonic()


def _is_llama_proc_alive(mgr: LlamaServerManager) -> bool:
    """Return True when llama-server manager holds a live child process."""
    proc = getattr(mgr, "_process", None)
    return proc is not None and proc.poll() is None


async def _idle_watchdog_loop(app: FastAPI, idle_unload_minutes: int) -> None:
    """Stop llama-server after prolonged inactivity to free GPU VRAM until next request."""
    idle_seconds = max(1, int(idle_unload_minutes)) * 60.0
    check_interval = min(30.0, idle_seconds / 2.0)
    try:
        while True:
            await asyncio.sleep(check_interval)
            mgr: LlamaServerManager | None = app.state.llama_manager
            if mgr is not None and _is_llama_proc_alive(mgr):
                app.state.llm_idle_unloaded = False
            if (
                mgr is None
                or not app.state.ready
                or app.state.llm_idle_unloaded
                or (getattr(mgr, "_process", None) is None and not await asyncio.to_thread(mgr.is_healthy))
            ):
                continue
            elapsed_idle = time.monotonic() - float(app.state.last_activity)
            if elapsed_idle < idle_seconds:
                continue

            lock: asyncio.Lock = app.state.llm_restart_lock
            async with lock:
                if app.state.llama_manager is not None and _is_llama_proc_alive(app.state.llama_manager):
                    app.state.llm_idle_unloaded = False
                if (
                    app.state.llama_manager is not None
                    and not app.state.llm_idle_unloaded
                    and (time.monotonic() - float(app.state.last_activity)) >= idle_seconds
                    and (
                        getattr(app.state.llama_manager, "_process", None) is not None
                        or await asyncio.to_thread(app.state.llama_manager.is_healthy)
                    )
                ):
                    logger.info("Unloading llama-server after %.1f minutes idle", elapsed_idle / 60.0)
                    await asyncio.to_thread(app.state.llama_manager.stop)
                    await asyncio.to_thread(_release_torch_cache)
                    if not getattr(app.state, "llm_lazy", False):
                        app.state.llm_idle_unloaded = True
                        app.state.loading_stage = "idle_unloaded"
                    else:
                        app.state.llm_idle_unloaded = False
    except asyncio.CancelledError:
        return


async def _update_check_loop(updater: UpdateManager, interval_sec: int) -> None:
    """Periodically check remote repository for updates when auto-update is enabled."""
    sleep_seconds = max(10.0, float(interval_sec))
    try:
        while True:
            await asyncio.sleep(sleep_seconds)
            try:
                await asyncio.to_thread(updater.check_for_updates, True)
            except Exception:
                logger.exception("Periodic update check failed")
    except asyncio.CancelledError:
        return


async def _load_production_models(app: FastAPI) -> None:
    """Load llama-server, detector, and OCR models in the background during startup."""
    from server.ocr.engine import OcrEngine
    from server.translator.llama_manager import LlamaServerManager
    from server.translator.translator import DialogueTranslator
    from server.vision.detector import BubbleDetector

    cfg: AppConfig = app.state.cfg
    memory: ChapterMemoryStore = app.state.memory
    cache: TranslationCache = app.state.cache

    max_image_bytes = int(getattr(cfg, "max_image_bytes", 25_000_000))
    max_image_width = int(getattr(cfg, "max_image_width", 2400))
    max_queue_size = int(getattr(cfg, "max_queue_size", 12))

    tracker: _StartupProgressTracker | None = getattr(app.state, "progress_tracker", None)
    if tracker is None:
        tracker = _StartupProgressTracker(stage="initializing")
        app.state.progress_tracker = tracker

    try:
        app.state.loading_stage = "initializing"
        tracker.set_step("initializing", "Detecting compute device")
        device = await asyncio.to_thread(_detect_torch_device)
        gpu_guard: GpuSafetyGuard = getattr(app.state, "gpu_guard", None) or GpuSafetyGuard(cfg)
        app.state.gpu_guard = gpu_guard
        min_free_vram_mb = int(getattr(cfg, "gpu_min_free_vram_mb", 1024))

        llama_mgr = app.state.llama_manager or LlamaServerManager(cfg, gpu_guard=gpu_guard)
        app.state.llama_manager = llama_mgr

        mode = str(getattr(cfg, "translation_mode", "hybrid") or "hybrid").strip().lower()
        has_gemini_key = bool(get_gemini_api_key(env_path=cfg.project_root / ".env"))
        lazy_llm = mode in ("hybrid", "gemini") and has_gemini_key
        app.state.llm_lazy = lazy_llm
        tracker.set_lazy_llm(lazy_llm)

        if not lazy_llm:
            app.state.loading_stage = "starting_llm"
            tracker.set_step("starting_llm", "Starting local LLM")
            await asyncio.to_thread(llama_mgr.start)
            app.state.llm_idle_unloaded = False
        else:
            logger.info("Deferring local llama-server startup in %s mode (Gemini key configured)", mode)
            app.state.llm_idle_unloaded = False

        app.state.loading_stage = "loading_detector"
        tracker.set_step("loading_detector", "Loading bubble detector")
        detector = BubbleDetector(
            model_id=cfg.detector_model_id,
            device=device,
            min_free_vram_mb=min_free_vram_mb,
        )
        if _accepts_kwarg(detector.load, "progress_cb"):
            await asyncio.to_thread(detector.load, progress_cb=tracker.set_step)
        else:
            await asyncio.to_thread(detector.load)

        app.state.loading_stage = "loading_ocr"
        tracker.set_step("loading_ocr", "Loading OCR engine")
        ocr = OcrEngine(
            device=device,
            cache_dir=cfg.cache_dir,
            min_free_vram_mb=min_free_vram_mb,
        )
        try:
            if _accepts_kwarg(ocr.load_ja, "progress_cb"):
                await asyncio.to_thread(ocr.load_ja, progress_cb=tracker.set_step)
            else:
                await asyncio.to_thread(ocr.load_ja)
        except Exception as exc:
            logger.warning("Preloading Japanese OCR deferred to first request: %s", exc)

        app.state.loading_stage = "initializing_pipeline"
        tracker.set_step("initializing_pipeline", "Initializing translation pipeline")
        gemini_translator: Any = None
        try:
            from server.translator.gemini_translator import GeminiVisionTranslator

            if _accepts_kwarg(GeminiVisionTranslator.__init__, "cfg"):
                gemini_translator = GeminiVisionTranslator(cfg=cfg)
            else:
                gemini_translator = GeminiVisionTranslator()
        except ImportError:
            logger.info("GeminiVisionTranslator unavailable; local translator active")
        except Exception as exc:
            logger.warning("GeminiVisionTranslator initialization skipped: %s", exc)

        translator_kwargs: dict[str, Any] = {
            "llama_base_url": llama_mgr.base_url,
            "memory": memory,
        }
        if _accepts_kwarg(DialogueTranslator.__init__, "cfg"):
            translator_kwargs["cfg"] = cfg
        if _accepts_kwarg(DialogueTranslator.__init__, "llama_mgr"):
            translator_kwargs["llama_mgr"] = llama_mgr
        if gemini_translator is not None and _accepts_kwarg(
            DialogueTranslator.__init__, "gemini_translator"
        ):
            translator_kwargs["gemini_translator"] = gemini_translator
        translator = DialogueTranslator(**translator_kwargs)

        pipeline_kwargs: dict[str, Any] = {
            "detector": detector,
            "ocr": ocr,
            "translator": translator,
            "memory": memory,
            "cache": cache,
            "max_image_bytes": max_image_bytes,
            "max_image_width": max_image_width,
            "max_queue_size": max_queue_size,
            "ensure_llm_ready": lambda: _ensure_llm_ready(app),
        }
        if _accepts_kwarg(TranslationPipeline.__init__, "gpu_guard"):
            pipeline_kwargs["gpu_guard"] = gpu_guard
        if _accepts_kwarg(TranslationPipeline.__init__, "cfg"):
            pipeline_kwargs["cfg"] = cfg
        pipeline = TranslationPipeline(**pipeline_kwargs)
        if getattr(pipeline, "gpu_guard", None) is None:
            pipeline.gpu_guard = gpu_guard
        await pipeline.start()

        app.state.pipeline = pipeline
        app.state.ready = True
        app.state.loading_stage = "ready"
        app.state.loading_error = None
        tracker.mark_ready()
        app.state.last_activity = time.monotonic()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Failed to initialize GPU models")
        app.state.ready = False
        app.state.loading_error = str(exc)
        app.state.loading_stage = f"error: {exc}"
        tracker.mark_error(str(exc))


def create_app(
    cfg: AppConfig | None = None,
    pipeline: TranslationPipeline | None = None,
    cache: TranslationCache | None = None,
    memory: ChapterMemoryStore | None = None,
    llama_manager: LlamaServerManager | None = None,
    auto_load_models: bool = True,
) -> FastAPI:
    """Build and configure the local Manga Live Translator FastAPI application."""
    resolved_cfg = cfg if cfg is not None else load_config()
    resolved_memory = memory if memory is not None else ChapterMemoryStore()
    if cache is not None:
        resolved_cache = cache
    else:
        assert resolved_cfg.cache_dir is not None
        resolved_cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        resolved_cache = TranslationCache(
            resolved_cfg.cache_dir / "translations.db",
            max_entries=resolved_cfg.max_cache_entries,
        )

    max_image_bytes = int(getattr(resolved_cfg, "max_image_bytes", 25_000_000))
    idle_unload_minutes = int(getattr(resolved_cfg, "idle_unload_minutes", 3))
    extension_dir = resolved_cfg.project_root / "extension"

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        if auto_load_models and "PYTEST_CURRENT_TEST" not in os.environ:
            _configure_server_logging(app_instance.state.cfg)
        if auto_load_models and app_instance.state.pipeline is None:
            enforce_d_drive_env(app_instance.state.cfg)
            app_instance.state.load_task = asyncio.create_task(
                _load_production_models(app_instance),
                name="model-loader",
            )
            if idle_unload_minutes > 0:
                app_instance.state.watchdog_task = asyncio.create_task(
                    _idle_watchdog_loop(app_instance, idle_unload_minutes),
                    name="idle-watchdog",
                )
        elif app_instance.state.pipeline is not None:
            await app_instance.state.pipeline.start()

        app_updater: UpdateManager = app_instance.state.updater
        if app_updater.auto_update_enabled and await asyncio.to_thread(app_updater.detect_has_remote):
            app_instance.state.update_task = asyncio.create_task(
                _update_check_loop(app_updater, app_updater.check_interval_sec),
                name="auto-updater",
            )

        try:
            yield
        finally:
            for attr in ("update_task", "watchdog_task", "load_task"):
                task: asyncio.Task[None] | None = getattr(app_instance.state, attr, None)
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            active_pipeline: TranslationPipeline | None = app_instance.state.pipeline
            if active_pipeline is not None:
                await active_pipeline.stop()

            active_llama: LlamaServerManager | None = app_instance.state.llama_manager
            if active_llama is not None and auto_load_models:
                await asyncio.to_thread(active_llama.stop)

    app = FastAPI(title="Live Manga & Manhwa Translator", lifespan=lifespan)

    def _stop_llama_for_restart() -> None:
        mgr: LlamaServerManager | None = app.state.llama_manager
        if mgr is not None:
            try:
                mgr.stop()
            except Exception:
                logger.exception("Failed to stop llama-server before restart")

    updater = UpdateManager(
        repo_root=resolved_cfg.project_root,
        extension_dir=extension_dir,
        auto_update_enabled=bool(getattr(resolved_cfg, "auto_update_enabled", True)),
        check_interval_sec=int(getattr(resolved_cfg, "auto_update_check_interval_sec", 1800)),
        on_restart_requested=_stop_llama_for_restart,
    )

    app.state.cfg = resolved_cfg
    app.state.gpu_guard = GpuSafetyGuard(resolved_cfg)
    app.state.memory = resolved_memory
    app.state.cache = resolved_cache
    app.state.llama_manager = llama_manager
    app.state.pipeline = pipeline
    app.state.updater = updater
    app.state.ready = pipeline is not None
    app.state.loading_stage = "ready" if pipeline is not None else ("initializing" if auto_load_models else "unloaded")
    app.state.loading_error = None
    app.state.load_task = None
    app.state.watchdog_task = None
    app.state.update_task = None
    app.state.last_activity = time.monotonic()
    app.state.llm_idle_unloaded = False
    app.state.llm_lazy = (
        str(getattr(resolved_cfg, "translation_mode", "hybrid") or "hybrid").strip().lower()
        in ("hybrid", "gemini")
        and bool(get_gemini_api_key(env_path=resolved_cfg.project_root / ".env"))
    )
    app.state.progress_tracker = _StartupProgressTracker(
        lazy_llm=app.state.llm_lazy,
        stage=app.state.loading_stage,
    )
    app.state.llm_restart_lock = asyncio.Lock()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-MLT-Loading-Stage",
            "X-MLT-Loading-Progress",
            "X-MLT-Loading-Detail",
        ],
    )

    assets_dir = resolved_cfg.project_root / "assets"
    app.mount("/assets", StaticFiles(directory=str(assets_dir), check_dir=False), name="assets")
    app.mount("/ext", StaticFiles(directory=str(extension_dir), check_dir=False), name="extension")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        live_cfg: AppConfig = app.state.cfg
        gpu_guard_obj: GpuSafetyGuard = app.state.gpu_guard
        gpu_name, vram_used, vram_total = await asyncio.to_thread(_query_gpu_stats_sync)
        gpu_guard_status = await asyncio.to_thread(gpu_guard_obj.status)
        mgr: LlamaServerManager | None = app.state.llama_manager
        if mgr is not None:
            proc_alive = _is_llama_proc_alive(mgr)
            if proc_alive:
                app.state.llm_idle_unloaded = False
            skip_llm_probe = bool(app.state.llm_idle_unloaded) or (
                hasattr(mgr, "_process") and not proc_alive and bool(app.state.llm_lazy)
            )
            llm_ready = False if skip_llm_probe else await asyncio.to_thread(mgr.is_healthy)
            active_local_model = str(getattr(mgr, "active_model_filename", "") or "") if llm_ready else ""
            llm_model = active_local_model or getattr(mgr, "model_name", live_cfg.llm_repo_id)
            available_local_models = await asyncio.to_thread(mgr.available_local_models)
        else:
            llm_model = live_cfg.llm_repo_id
            llm_ready = bool(app.state.ready)
            active_local_model = ""
            available_local_models = await asyncio.to_thread(_scan_available_local_models, live_cfg)

        gemini_key = get_gemini_api_key(env_path=live_cfg.project_root / ".env")
        health_update = await asyncio.to_thread(updater.get_health_fields)
        loading_stage, loading_progress, loading_detail = _get_loading_state(app)
        status = "ok" if app.state.ready else ("error" if app.state.loading_error else "loading")
        return HealthResponse(
            status=status,
            ready=bool(app.state.ready),
            gpu_name=gpu_name,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            llm_model=llm_model,
            llm_ready=bool(llm_ready),
            loading_stage=loading_stage,
            loading_progress=loading_progress,
            loading_detail=loading_detail,
            extension_version=str(health_update["extension_version"]),
            extension_build_id=str(health_update["extension_build_id"]),
            has_remote=bool(health_update["has_remote"]),
            update_available=bool(health_update["update_available"]),
            translation_mode=str(getattr(live_cfg, "translation_mode", "hybrid") or "hybrid"),
            gemini_configured=bool(gemini_key),
            gemini_key_hint=mask_api_key(gemini_key),
            active_local_model=active_local_model,
            available_local_models=available_local_models,
            gpu_guard=gpu_guard_status,
        )

    @app.post("/settings", response_model=SettingsUpdateResponse)
    async def update_settings(req: SettingsUpdateRequest) -> SettingsUpdateResponse:
        live_cfg: AppConfig = app.state.cfg
        env_path = live_cfg.project_root / ".env"
        cfg_path = live_cfg.project_root / "config.json"

        if req.translation_mode is not None:
            mode = req.translation_mode.strip().lower()
            if mode not in VALID_TRANSLATION_MODES:
                raise HTTPException(
                    status_code=422,
                    detail=f"translation_mode must be one of {VALID_TRANSLATION_MODES}",
                )
            await asyncio.to_thread(
                save_config_updates,
                {"translation_mode": mode},
                cfg_path,
            )
            live_cfg.translation_mode = mode

        if req.clear_gemini_api_key:
            await asyncio.to_thread(save_gemini_api_key, "", env_path)
        elif req.gemini_api_key is not None:
            await asyncio.to_thread(save_gemini_api_key, req.gemini_api_key, env_path)

        current_key = get_gemini_api_key(env_path=env_path)
        active_mode = str(getattr(live_cfg, "translation_mode", "hybrid") or "hybrid").strip().lower()
        app.state.llm_lazy = active_mode in ("hybrid", "gemini") and bool(current_key)
        if app.state.llm_lazy:
            app.state.llm_idle_unloaded = False
            if app.state.loading_stage == "idle_unloaded":
                app.state.loading_stage = "ready"
        return SettingsUpdateResponse(
            ok=True,
            translation_mode=active_mode,
            gemini_configured=bool(current_key),
            gemini_key_hint=mask_api_key(current_key),
        )

    @app.get("/update/status", response_model=UpdateStatusResponse)
    async def update_status() -> UpdateStatusResponse:
        return await asyncio.to_thread(updater.get_status)

    @app.post("/update/check", response_model=UpdateStatusResponse)
    async def update_check(request: Request) -> UpdateStatusResponse:
        raw_body = await request.body()
        if raw_body.strip():
            try:
                req = UpdateCheckRequest.model_validate_json(raw_body)
            except (ValidationError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid update check request: {exc}",
                ) from exc
        else:
            req = UpdateCheckRequest(auto_apply=False)
        return await asyncio.to_thread(updater.check_for_updates, req.auto_apply)

    @app.post("/update/apply", response_model=UpdateApplyResponse)
    async def update_apply() -> UpdateApplyResponse:
        return await asyncio.to_thread(updater.apply_update)

    @app.post("/translate", response_model=TranslateResponse)
    async def translate(request: Request) -> TranslateResponse:
        active_pipeline: TranslationPipeline | None = app.state.pipeline
        if active_pipeline is None or not app.state.ready:
            _raise_not_ready_503(app)

        (
            image_bytes,
            lang,
            chapter_id,
            priority,
            glossary,
            request_id,
            include_floating,
            series_id,
            translation_mode,
        ) = await _parse_translate_request(
            request, max_image_bytes=max_image_bytes
        )
        app.state.last_activity = time.monotonic()

        translate_kwargs: dict[str, Any] = {
            "image_bytes": image_bytes,
            "lang": lang,
            "chapter_id": chapter_id,
            "priority": priority,
            "glossary": glossary,
            "request_id": request_id,
            "include_floating": include_floating,
            "series_id": series_id,
        }
        if translation_mode is not None and _accepts_kwarg(
            active_pipeline.translate_image, "translation_mode"
        ):
            translate_kwargs["translation_mode"] = translation_mode

        try:
            return await active_pipeline.translate_image(**translate_kwargs)
        except PayloadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except asyncio.CancelledError as exc:
            raise HTTPException(status_code=499, detail="Translation request cancelled") from exc

    @app.post("/translate/stream")
    async def translate_stream(request: Request) -> StreamingResponse:
        active_pipeline: TranslationPipeline | None = app.state.pipeline
        if active_pipeline is None or not app.state.ready:
            _raise_not_ready_503(app)

        (
            image_bytes,
            lang,
            chapter_id,
            priority,
            glossary,
            request_id,
            include_floating,
            series_id,
            translation_mode,
        ) = await _parse_translate_request(
            request, max_image_bytes=max_image_bytes
        )
        if len(image_bytes) > active_pipeline.max_image_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Image payload ({len(image_bytes)} bytes) exceeds limit "
                    f"({active_pipeline.max_image_bytes} bytes)"
                ),
            )
        app.state.last_activity = time.monotonic()

        stream_kwargs: dict[str, Any] = {
            "image_bytes": image_bytes,
            "lang": lang,
            "chapter_id": chapter_id,
            "priority": priority,
            "glossary": glossary,
            "request_id": request_id,
            "include_floating": include_floating,
            "series_id": series_id,
        }
        if translation_mode is not None and _accepts_kwarg(
            active_pipeline.translate_image_stream, "translation_mode"
        ):
            stream_kwargs["translation_mode"] = translation_mode

        async def _event_stream() -> AsyncIterator[bytes]:
            stream_iter = active_pipeline.translate_image_stream(**stream_kwargs)
            try:
                async for event in stream_iter:
                    if await request.is_disconnected():
                        if request_id:
                            active_pipeline.cancel(request_id)
                        break
                    app.state.last_activity = time.monotonic()
                    yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            except PayloadTooLargeError as exc:
                yield (json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n").encode("utf-8")
            except asyncio.CancelledError:
                if request_id:
                    active_pipeline.cancel(request_id)
                raise
            finally:
                await stream_iter.aclose()

        return StreamingResponse(
            _event_stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/translate/cancel")
    async def translate_cancel(req: CancelRequest) -> dict[str, str]:
        active_pipeline: TranslationPipeline | None = app.state.pipeline
        if active_pipeline is not None:
            active_pipeline.cancel(req.request_id)
        return {"status": "cancelled", "request_id": req.request_id}

    @app.get("/glossary")
    async def get_glossary(
        chapter_id: str = Query(default="default"),
        series_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        return {
            "chapter_id": chapter_id,
            "series_id": series_id,
            "entries": app.state.memory.get_glossary(chapter_id, series_id=series_id),
        }

    @app.post("/glossary")
    async def update_glossary(req: GlossaryUpdateRequest) -> dict[str, Any]:
        app.state.memory.set_glossary(req.chapter_id, req.entries, series_id=req.series_id)
        return {
            "status": "ok",
            "chapter_id": req.chapter_id,
            "series_id": req.series_id,
            "entries": app.state.memory.get_glossary(req.chapter_id, series_id=req.series_id),
        }

    @app.post("/glossary/lookup", response_model=GlossaryLookupResponse)
    async def lookup_glossary(req: GlossaryLookupRequest) -> GlossaryLookupResponse:
        query = req.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Lookup query must not be empty")
        try:
            result = await lookup_character_glossary(query=query, source=req.source)
        except (InvalidLookupQueryError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GlossaryLookupError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected glossary lookup error")
            raise HTTPException(status_code=502, detail=f"Glossary lookup failed: {exc}") from exc
        return GlossaryLookupResponse.model_validate(result)

    @app.post("/memory/clear")
    async def clear_memory(req: MemoryClearRequest) -> dict[str, Any]:
        app.state.memory.clear(req.chapter_id, series_id=req.series_id)
        return {"status": "ok", "chapter_id": req.chapter_id, "series_id": req.series_id}

    @app.post("/cache/clear")
    async def clear_cache() -> dict[str, str]:
        app.state.cache.clear()
        return {"status": "ok"}

    @app.post("/shutdown")
    async def shutdown() -> JSONResponse:
        async def _graceful_exit() -> None:
            await asyncio.sleep(0.15)
            active_pipeline: TranslationPipeline | None = app.state.pipeline
            if active_pipeline is not None:
                try:
                    await active_pipeline.stop()
                except Exception:
                    pass
            mgr: LlamaServerManager | None = app.state.llama_manager
            if mgr is not None:
                try:
                    await asyncio.to_thread(mgr.stop)
                except Exception:
                    pass

            def _terminate() -> None:
                os._exit(0)

            threading.Timer(0.1, _terminate).start()

        asyncio.create_task(_graceful_exit(), name="graceful-shutdown")
        return JSONResponse({"status": "shutting_down"})

    @app.get("/", response_class=HTMLResponse)
    async def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/test", response_class=HTMLResponse)
    async def test_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "test.html", headers={"Cache-Control": "no-store"})

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        icon_path = extension_dir / "icons" / "icon48.png"
        if icon_path.exists():
            return FileResponse(icon_path, media_type="image/png")
        return Response(status_code=204)

    return app


def main() -> None:
    cfg = load_config()
    enforce_d_drive_env(cfg)
    _configure_server_logging(cfg)
    app = create_app(cfg=cfg, auto_load_models=True)
    bind_host = "0.0.0.0" if getattr(cfg, "allow_lan", False) else cfg.host
    uvicorn.run(app, host=bind_host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()

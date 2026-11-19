from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
import uvicorn

from server.cache import TranslationCache
from server.config import AppConfig, enforce_d_drive_env, load_config
from server.pipeline import PayloadTooLargeError, TranslationPipeline
from server.schemas import (
    GlossaryUpdateRequest,
    HealthResponse,
    MemoryClearRequest,
    TranslateRequest,
    TranslateResponse,
    UpdateApplyResponse,
    UpdateCheckRequest,
    UpdateStatusResponse,
)
from server.translator.memory import ChapterMemoryStore
from server.updater import UpdateManager

if TYPE_CHECKING:
    from server.translator.llama_manager import LlamaServerManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class CancelRequest(BaseModel):
    request_id: str


def _query_gpu_stats_sync() -> tuple[str, int, int]:
    """Query live GPU name and VRAM usage via nvidia-smi safely."""
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
            creationflags=creationflags,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return ("Unavailable", 0, 0)
        first_line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 3:
            return ("Unavailable", 0, 0)
        name = parts[0] or "Unknown GPU"
        used = int(float(parts[1]))
        total = int(float(parts[2]))
        return (name, used, total)
    except Exception:
        return ("Unavailable", 0, 0)


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


async def _parse_translate_request(
    request: Request,
    max_image_bytes: int,
) -> tuple[bytes, Literal["auto", "ja", "ko"], str, int, dict[str, str], str | None]:
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

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload")

    return image_bytes, lang, chapter_id, priority, glossary, request_id


def _detect_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


async def _ensure_llm_ready(app: FastAPI) -> None:
    """Transparently restart llama-server if it was unloaded due to inactivity."""
    app.state.last_activity = time.monotonic()
    mgr: LlamaServerManager | None = app.state.llama_manager
    if mgr is None or not app.state.llm_idle_unloaded:
        return

    lock: asyncio.Lock = app.state.llm_restart_lock
    async with lock:
        if app.state.llm_idle_unloaded and app.state.llama_manager is not None:
            app.state.loading_stage = "restarting_llm"
            await asyncio.to_thread(app.state.llama_manager.start)
            app.state.llm_idle_unloaded = False
            app.state.loading_stage = "ready"
            app.state.last_activity = time.monotonic()


async def _idle_watchdog_loop(app: FastAPI, idle_unload_minutes: int) -> None:
    """Stop llama-server after prolonged inactivity to free GPU VRAM until next request."""
    idle_seconds = max(1, int(idle_unload_minutes)) * 60.0
    check_interval = min(30.0, idle_seconds / 2.0)
    try:
        while True:
            await asyncio.sleep(check_interval)
            mgr: LlamaServerManager | None = app.state.llama_manager
            if (
                mgr is None
                or not app.state.ready
                or app.state.llm_idle_unloaded
            ):
                continue
            elapsed_idle = time.monotonic() - float(app.state.last_activity)
            if elapsed_idle < idle_seconds:
                continue

            lock: asyncio.Lock = app.state.llm_restart_lock
            async with lock:
                if (
                    app.state.llama_manager is not None
                    and not app.state.llm_idle_unloaded
                    and (time.monotonic() - float(app.state.last_activity)) >= idle_seconds
                ):
                    logger.info("Unloading llama-server after %.1f minutes idle", elapsed_idle / 60.0)
                    await asyncio.to_thread(app.state.llama_manager.stop)
                    await asyncio.to_thread(_release_torch_cache)
                    app.state.llm_idle_unloaded = True
                    app.state.loading_stage = "idle_unloaded"
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

    try:
        device = _detect_torch_device()
        app.state.loading_stage = "starting_llm"
        llama_mgr = app.state.llama_manager or LlamaServerManager(cfg)
        app.state.llama_manager = llama_mgr
        await asyncio.to_thread(llama_mgr.start)
        app.state.llm_idle_unloaded = False

        app.state.loading_stage = "loading_detector"
        detector = BubbleDetector(model_id=cfg.detector_model_id, device=device)
        await asyncio.to_thread(detector.load)

        app.state.loading_stage = "loading_ocr"
        ocr = OcrEngine(device=device, cache_dir=cfg.cache_dir)

        app.state.loading_stage = "initializing_pipeline"
        translator = DialogueTranslator(llama_base_url=llama_mgr.base_url, memory=memory)
        pipeline = TranslationPipeline(
            detector=detector,
            ocr=ocr,
            translator=translator,
            memory=memory,
            cache=cache,
            max_image_bytes=max_image_bytes,
            max_image_width=max_image_width,
            max_queue_size=max_queue_size,
            ensure_llm_ready=lambda: _ensure_llm_ready(app),
        )
        await pipeline.start()

        app.state.pipeline = pipeline
        app.state.ready = True
        app.state.loading_stage = "ready"
        app.state.loading_error = None
        app.state.last_activity = time.monotonic()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Failed to initialize GPU models")
        app.state.ready = False
        app.state.loading_error = str(exc)
        app.state.loading_stage = f"error: {exc}"


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
    app.state.llm_restart_lock = asyncio.Lock()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    assets_dir = resolved_cfg.project_root / "assets"
    app.mount("/assets", StaticFiles(directory=str(assets_dir), check_dir=False), name="assets")
    app.mount("/ext", StaticFiles(directory=str(extension_dir), check_dir=False), name="extension")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        gpu_name, vram_used, vram_total = await asyncio.to_thread(_query_gpu_stats_sync)
        mgr: LlamaServerManager | None = app.state.llama_manager
        if mgr is not None:
            llm_model = getattr(mgr, "model_name", resolved_cfg.llm_repo_id)
            llm_ready = False if app.state.llm_idle_unloaded else await asyncio.to_thread(mgr.is_healthy)
        else:
            llm_model = resolved_cfg.llm_repo_id
            llm_ready = bool(app.state.ready)

        health_update = await asyncio.to_thread(updater.get_health_fields)
        status = "ok" if app.state.ready else ("error" if app.state.loading_error else "loading")
        return HealthResponse(
            status=status,
            ready=bool(app.state.ready),
            gpu_name=gpu_name,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            llm_model=llm_model,
            llm_ready=bool(llm_ready),
            loading_stage=str(app.state.loading_stage or ""),
            extension_version=str(health_update["extension_version"]),
            extension_build_id=str(health_update["extension_build_id"]),
            has_remote=bool(health_update["has_remote"]),
            update_available=bool(health_update["update_available"]),
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
            raise HTTPException(
                status_code=503,
                detail=f"Models are not ready yet ({app.state.loading_stage})",
            )

        image_bytes, lang, chapter_id, priority, glossary, request_id = await _parse_translate_request(
            request, max_image_bytes=max_image_bytes
        )
        app.state.last_activity = time.monotonic()

        try:
            return await active_pipeline.translate_image(
                image_bytes=image_bytes,
                lang=lang,
                chapter_id=chapter_id,
                priority=priority,
                glossary=glossary,
                request_id=request_id,
            )
        except PayloadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except asyncio.CancelledError as exc:
            raise HTTPException(status_code=499, detail="Translation request cancelled") from exc

    @app.post("/translate/stream")
    async def translate_stream(request: Request) -> StreamingResponse:
        active_pipeline: TranslationPipeline | None = app.state.pipeline
        if active_pipeline is None or not app.state.ready:
            raise HTTPException(
                status_code=503,
                detail=f"Models are not ready yet ({app.state.loading_stage})",
            )

        image_bytes, lang, chapter_id, priority, glossary, request_id = await _parse_translate_request(
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

        async def _event_stream() -> AsyncIterator[bytes]:
            stream_iter = active_pipeline.translate_image_stream(
                image_bytes=image_bytes,
                lang=lang,
                chapter_id=chapter_id,
                priority=priority,
                glossary=glossary,
                request_id=request_id,
            )
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
    async def get_glossary(chapter_id: str = Query(default="default")) -> dict[str, Any]:
        return {
            "chapter_id": chapter_id,
            "entries": app.state.memory.get_glossary(chapter_id),
        }

    @app.post("/glossary")
    async def update_glossary(req: GlossaryUpdateRequest) -> dict[str, Any]:
        app.state.memory.set_glossary(req.chapter_id, req.entries)
        return {
            "status": "ok",
            "chapter_id": req.chapter_id,
            "entries": app.state.memory.get_glossary(req.chapter_id),
        }

    @app.post("/memory/clear")
    async def clear_memory(req: MemoryClearRequest) -> dict[str, str]:
        app.state.memory.clear(req.chapter_id)
        return {"status": "ok", "chapter_id": req.chapter_id}

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
    app = create_app(cfg=cfg, auto_load_models=True)
    bind_host = "0.0.0.0" if getattr(cfg, "allow_lan", False) else cfg.host
    uvicorn.run(app, host=bind_host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()

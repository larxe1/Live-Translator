from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
import gc
import heapq
import inspect
from io import BytesIO
import itertools
import logging
import time
from typing import Any, Awaitable, Callable, Literal, TypeVar

import numpy as np
from PIL import Image

from server.cache import TranslationCache
from server.ocr.engine import is_meaningful_dialogue
from server.schemas import Box, BubbleResult, TranslateResponse
from server.translator.memory import ChapterMemoryStore, contains_non_english_cjk
from server.vision.detector import sort_reading_order
from server.vision.geometry import BubbleStyle, analyze_bubble_appearance
from server.vision.slicer import RawDetection

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PayloadTooLargeError(ValueError):
    """Raised when an incoming image exceeds max_image_bytes."""


@dataclass
class _QueueItem:
    priority: int
    seq: int
    fn: Callable[[], Awaitable[Any]]
    future: asyncio.Future[Any]
    request_id: str | None = None
    cancelled: bool = False


class PriorityGpuQueue:
    """Bounded single-worker priority queue that serializes GPU jobs (max_concurrent=1).

    Lower numeric priority runs first (0 = visible viewport, 1 = look-ahead).
    Ties are broken by a strictly monotonic FIFO counter.
    When full (len >= max_queue_size), evicts the oldest lower-priority look-ahead item.
    """

    def __init__(self, max_queue_size: int = 12) -> None:
        self.max_queue_size = max(1, int(max_queue_size))
        self._heap: list[tuple[int, int, _QueueItem]] = []
        self._wakeup: asyncio.Event | None = None
        self._counter = itertools.count()
        self._worker_task: asyncio.Task[None] | None = None
        self._items_by_id: dict[str, list[_QueueItem]] = defaultdict(list)
        self._stopping = False

    async def start(self) -> None:
        """Start the single GPU consumer coroutine if not already running."""
        if self._wakeup is None:
            self._wakeup = asyncio.Event()
        self._stopping = False
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop(), name="gpu-priority-worker")

    async def stop(self) -> None:
        """Stop the worker coroutine and cancel any pending queued jobs."""
        self._stopping = True
        while self._heap:
            _, _, item = heapq.heappop(self._heap)
            item.cancelled = True
            if not item.future.done():
                item.future.cancel()
        self._items_by_id.clear()
        if self._wakeup is not None:
            self._wakeup.set()

        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None

    def _prune_inactive(self) -> None:
        if not self._heap:
            return
        active = [
            entry
            for entry in self._heap
            if not entry[2].cancelled and not entry[2].future.cancelled() and not entry[2].future.done()
        ]
        if len(active) != len(self._heap):
            heapq.heapify(active)
            self._heap = active

    def _evict_for_capacity(self, incoming_priority: int) -> bool:
        """Make room when queue is at capacity. Returns False if incoming item should be dropped."""
        self._prune_inactive()
        if len(self._heap) < self.max_queue_size:
            return True

        lookahead = [entry for entry in self._heap if entry[2].priority >= 1]
        if lookahead:
            # Evict the oldest lower-priority look-ahead entry (smallest seq).
            victim = min(lookahead, key=lambda e: e[1])
        elif incoming_priority >= 1:
            return False
        else:
            # Queue is full of priority-0 items and a new priority-0 item arrived; evict oldest.
            victim = min(self._heap, key=lambda e: e[1])

        victim_item = victim[2]
        victim_item.cancelled = True
        if not victim_item.future.done():
            victim_item.future.cancel()
        if victim_item.request_id and victim_item.request_id in self._items_by_id:
            rem = [x for x in self._items_by_id[victim_item.request_id] if x is not victim_item]
            if rem:
                self._items_by_id[victim_item.request_id] = rem
            else:
                del self._items_by_id[victim_item.request_id]
        self._prune_inactive()
        return True

    def pending_count(self) -> int:
        """Return count of active queued items waiting for execution."""
        self._prune_inactive()
        return len(self._heap)

    async def enqueue(
        self,
        priority: int,
        fn: Callable[[], Awaitable[T]],
        request_id: str | None = None,
    ) -> T:
        """Enqueue an async job and await its result in priority + FIFO order."""
        await self.start()
        assert self._wakeup is not None

        if not self._evict_for_capacity(priority):
            raise asyncio.CancelledError("GPU look-ahead queue is full")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        seq = next(self._counter)
        item = _QueueItem(
            priority=priority,
            seq=seq,
            fn=fn,
            future=future,
            request_id=request_id,
        )
        if request_id:
            self._items_by_id[request_id].append(item)

        heapq.heappush(self._heap, (priority, seq, item))
        self._wakeup.set()
        try:
            return await future
        except asyncio.CancelledError:
            item.cancelled = True
            raise

    def cancel(self, request_id: str) -> None:
        """Cancel all pending queued jobs matching request_id."""
        items = self._items_by_id.pop(request_id, [])
        for item in items:
            item.cancelled = True
            if not item.future.done():
                item.future.cancel()

    async def _worker_loop(self) -> None:
        assert self._wakeup is not None
        while not self._stopping:
            self._prune_inactive()
            if not self._heap:
                self._wakeup.clear()
                if self._stopping:
                    break
                await self._wakeup.wait()
                continue

            _, _, item = heapq.heappop(self._heap)
            if item.request_id and item.request_id in self._items_by_id:
                remaining = [x for x in self._items_by_id[item.request_id] if x is not item]
                if remaining:
                    self._items_by_id[item.request_id] = remaining
                else:
                    del self._items_by_id[item.request_id]

            if item.cancelled or item.future.cancelled() or item.future.done():
                continue

            try:
                result = await item.fn()
            except asyncio.CancelledError:
                if not item.future.done():
                    item.future.cancel()
                if self._stopping:
                    raise
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
            else:
                if not item.future.done():
                    item.future.set_result(result)


async def _call_maybe_async(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke either an async function or a blocking sync function in a worker thread."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    res = await asyncio.to_thread(fn, *args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


def _fallback_lang(lang: Literal["auto", "ja", "ko"], img_w: int, img_h: int) -> Literal["ja", "ko"]:
    if lang == "ko":
        return "ko"
    if lang == "ja":
        return "ja"
    return "ko" if (img_h / max(img_w, 1)) > 2.2 else "ja"


def _scale_box(box: Box, inv_scale: float, max_w: int, max_h: int) -> Box:
    if inv_scale == 1.0:
        return box
    x1 = max(0, min(max_w - 1, int(round(box.x1 * inv_scale))))
    y1 = max(0, min(max_h - 1, int(round(box.y1 * inv_scale))))
    x2 = max(x1 + 1, min(max_w, int(round(box.x2 * inv_scale))))
    y2 = max(y1 + 1, min(max_h, int(round(box.y2 * inv_scale))))
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def _cleanup_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class TranslationPipeline:
    """Orchestrates cache lookup, priority GPU queueing, detection, OCR, sorting, and translation."""

    def __init__(
        self,
        detector: Any,
        ocr: Any,
        translator: Any,
        memory: ChapterMemoryStore,
        cache: TranslationCache,
        queue: PriorityGpuQueue | None = None,
        max_image_bytes: int = 25_000_000,
        max_image_width: int = 2400,
        max_queue_size: int = 12,
        ensure_llm_ready: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.translator = translator
        self.memory = memory
        self.cache = cache
        self.max_image_bytes = int(max_image_bytes)
        self.max_image_width = int(max_image_width)
        self.ensure_llm_ready = ensure_llm_ready
        self.queue = queue if queue is not None else PriorityGpuQueue(max_queue_size=max_queue_size)

    async def start(self) -> None:
        """Start the underlying priority GPU worker queue."""
        await self.queue.start()

    async def stop(self) -> None:
        """Stop the underlying priority GPU worker queue."""
        await self.queue.stop()

    def cancel(self, request_id: str) -> None:
        """Cancel a pending queued translation request by request_id."""
        self.queue.cancel(request_id)

    async def translate_image(
        self,
        image_bytes: bytes,
        lang: Literal["auto", "ja", "ko"] = "auto",
        chapter_id: str = "default",
        priority: int = 0,
        glossary: dict[str, str] | None = None,
        request_id: str | None = None,
    ) -> TranslateResponse:
        """Translate all dialogue bubbles in an image, checking cache before GPU queueing."""
        if len(image_bytes) > self.max_image_bytes:
            raise PayloadTooLargeError(
                f"Image payload ({len(image_bytes)} bytes) exceeds limit ({self.max_image_bytes} bytes)"
            )

        chapter_glossary = self.memory.get_glossary(chapter_id)
        merged_glossary = {**chapter_glossary, **(glossary or {})}

        cache_key = self.cache.make_key(image_bytes, lang, merged_glossary)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        async def _run_gpu_job() -> TranslateResponse:
            # Re-check cache in case an identical image finished while we waited in queue.
            cached_inner = self.cache.get(cache_key)
            if cached_inner is not None:
                return cached_inner

            if self.ensure_llm_ready is not None:
                await self.ensure_llm_ready()

            t0 = time.perf_counter()
            pil_img: Image.Image | None = None
            infer_img: Image.Image | None = None
            img_rgb: np.ndarray | None = None
            try:
                with Image.open(BytesIO(image_bytes)) as opened:
                    pil_img = opened.convert("RGB")
                orig_w, orig_h = pil_img.size

                if orig_w > self.max_image_width:
                    scale = self.max_image_width / float(orig_w)
                    infer_w = self.max_image_width
                    infer_h = max(1, int(round(orig_h * scale)))
                    infer_img = pil_img.resize((infer_w, infer_h), Image.Resampling.LANCZOS)
                    inv_scale = orig_w / float(infer_w)
                else:
                    infer_img = pil_img
                    infer_w, infer_h = orig_w, orig_h
                    inv_scale = 1.0

                img_rgb = np.asarray(infer_img)

                detections: list[RawDetection] = await _call_maybe_async(self.detector.detect, infer_img)
                if not detections:
                    resolved_empty = _fallback_lang(lang, orig_w, orig_h)
                    empty_resp = TranslateResponse(
                        lang=resolved_empty,
                        width=orig_w,
                        height=orig_h,
                        elapsed_ms=int((time.perf_counter() - t0) * 1000),
                        cached=False,
                        bubbles=[],
                    )
                    self._store_cache(image_bytes, lang, resolved_empty, merged_glossary, cache_key, empty_resp)
                    return empty_resp

                det_styles: list[tuple[RawDetection, BubbleStyle]] = []
                crops_with_dark: list[tuple[Image.Image, bool]] = []
                for det in detections:
                    style = analyze_bubble_appearance(img_rgb, det.box)
                    tight_box = getattr(det, "ocr_box", None)
                    if tight_box is not None:
                        pad = 3
                        x1 = max(0, min(infer_w - 1, tight_box.x1 - pad))
                        y1 = max(0, min(infer_h - 1, tight_box.y1 - pad))
                        x2 = max(x1 + 1, min(infer_w, tight_box.x2 + pad))
                        y2 = max(y1 + 1, min(infer_h, tight_box.y2 + pad))
                    elif det.label.strip().lower() in ("bubble", "balloon"):
                        sb = style.safe_box
                        x1 = max(0, min(infer_w - 1, sb.x1))
                        y1 = max(0, min(infer_h - 1, sb.y1))
                        x2 = max(x1 + 1, min(infer_w, sb.x2))
                        y2 = max(y1 + 1, min(infer_h, sb.y2))
                    else:
                        x1 = max(0, min(infer_w - 1, det.box.x1))
                        y1 = max(0, min(infer_h - 1, det.box.y1))
                        x2 = max(x1 + 1, min(infer_w, det.box.x2))
                        y2 = max(y1 + 1, min(infer_h, det.box.y2))
                    crop = infer_img.crop((x1, y1, x2, y2))
                    det_styles.append((det, style))
                    crops_with_dark.append((crop, style.is_dark))

                resolved_lang, ocr_texts = await _call_maybe_async(
                    self.ocr.read_page_bubbles,
                    crops_with_dark,
                    lang_hint=lang,
                    aspect_ratio=infer_h / max(infer_w, 1),
                )

                meaningful: list[tuple[RawDetection, BubbleStyle, str]] = []
                for (det, style), text in zip(det_styles, ocr_texts):
                    cleaned = (text or "").strip()
                    if is_meaningful_dialogue(cleaned):
                        meaningful.append((det, style, cleaned))

                if not meaningful:
                    empty_resp = TranslateResponse(
                        lang=resolved_lang,
                        width=orig_w,
                        height=orig_h,
                        elapsed_ms=int((time.perf_counter() - t0) * 1000),
                        cached=False,
                        bubbles=[],
                    )
                    self._store_cache(image_bytes, lang, resolved_lang, merged_glossary, cache_key, empty_resp)
                    return empty_resp

                sorted_dets = sort_reading_order(
                    [item[0] for item in meaningful],
                    infer_w,
                    infer_h,
                    resolved_lang,
                )

                by_obj_id = {id(det): (det, style, text) for det, style, text in meaningful}
                by_box: dict[tuple[int, int, int, int], deque[tuple[RawDetection, BubbleStyle, str]]] = defaultdict(deque)
                for det, style, text in meaningful:
                    by_box[(det.box.x1, det.box.y1, det.box.x2, det.box.y2)].append((det, style, text))

                ordered_items: list[tuple[RawDetection, BubbleStyle, str]] = []
                for s_det in sorted_dets:
                    item = by_obj_id.get(id(s_det))
                    if item is None:
                        key_box = (s_det.box.x1, s_det.box.y1, s_det.box.x2, s_det.box.y2)
                        if by_box[key_box]:
                            item = by_box[key_box].popleft()
                    if item is not None:
                        ordered_items.append(item)

                prompt_bubbles = [(idx, item[2]) for idx, item in enumerate(ordered_items)]
                translations: dict[int, str] = await self.translator.translate_page(
                    prompt_bubbles,
                    lang=resolved_lang,
                    chapter_id=chapter_id,
                    extra_glossary=glossary,
                )

                bubbles = [
                    BubbleResult(
                        id=idx,
                        bbox=_scale_box(det.box, inv_scale, orig_w, orig_h),
                        safe_box=_scale_box(style.safe_box, inv_scale, orig_w, orig_h),
                        bg_color=style.bg_color,
                        text_color=style.text_color,
                        is_dark=style.is_dark,
                        original=text,
                        translated=translations.get(idx, text),
                    )
                    for idx, (det, style, text) in enumerate(ordered_items)
                ]

                response = TranslateResponse(
                    lang=resolved_lang,
                    width=orig_w,
                    height=orig_h,
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    cached=False,
                    bubbles=bubbles,
                )
                self._store_cache(image_bytes, lang, resolved_lang, merged_glossary, cache_key, response)
                return response
            finally:
                del img_rgb, infer_img, pil_img
                await asyncio.to_thread(_cleanup_gpu_memory)

        return await self.queue.enqueue(priority=priority, fn=_run_gpu_job, request_id=request_id)

    submit = translate_image

    def _store_cache(
        self,
        image_bytes: bytes,
        requested_lang: str,
        resolved_lang: str,
        merged_glossary: dict[str, str],
        primary_key: str,
        response: TranslateResponse,
    ) -> None:
        bad_bubbles = [b.id for b in response.bubbles if contains_non_english_cjk(b.translated)]
        if bad_bubbles:
            logger.warning(
                "Skipping translation cache write: non-English CJK characters remain in bubble(s) %s",
                bad_bubbles,
            )
            return

        self.cache.put(primary_key, response)
        if requested_lang != resolved_lang:
            resolved_key = self.cache.make_key(image_bytes, resolved_lang, merged_glossary)
            self.cache.put(resolved_key, response)

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from server.app import create_app
from server.cache import TranslationCache
from server.config import AppConfig
from server.pipeline import PayloadTooLargeError, PriorityGpuQueue, TranslationPipeline
from server.schemas import Box
from server.translator.memory import ChapterMemoryStore
from server.vision.slicer import RawDetection


def _make_sample_png_bytes(
    width: int = 600,
    height: int = 800,
    draw_dark_bubble: bool = True,
) -> bytes:
    img = Image.new("RGB", (width, height), (250, 250, 250))
    if draw_dark_bubble:
        draw = ImageDraw.Draw(img)
        # Right bubble (light)
        draw.rectangle([380, 60, 560, 240], fill=(250, 250, 250), outline=(10, 10, 10), width=4)
        # Left dark screaming bubble
        draw.rectangle([40, 260, 200, 480], fill=(12, 12, 12))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeDetector:
    def __init__(self, detections: list[RawDetection] | None = None) -> None:
        self.calls = 0
        self.seen_sizes: list[tuple[int, int]] = []
        self._detections = detections

    def detect(self, image: Image.Image) -> list[RawDetection]:
        self.calls += 1
        self.seen_sizes.append(image.size)
        if self._detections is not None:
            return list(self._detections)
        return [
            RawDetection(Box(x1=40, y1=260, x2=200, y2=480), "bubble", 0.95),
            RawDetection(Box(x1=380, y1=60, x2=560, y2=240), "bubble", 0.93),
            RawDetection(Box(x1=220, y1=80, x2=320, y2=160), "bubble", 0.70),
        ]


class _FakeOcr:
    def __init__(self, texts: list[str] | None = None) -> None:
        self.calls = 0
        self._texts = texts or ["ぎゃああああっ", "藍星……？", "..."]

    def read_page_bubbles(
        self,
        crops: list[tuple[Image.Image, bool]],
        lang_hint: str = "auto",
        aspect_ratio: float = 1.0,
    ) -> tuple[str, list[str]]:
        self.calls += 1
        resolved = "ko" if lang_hint == "ko" else "ja"
        return resolved, self._texts[: len(crops)]


class _FakeTranslator:
    def __init__(self, memory: ChapterMemoryStore) -> None:
        self.calls = 0
        self.memory = memory

    async def translate_page(
        self,
        bubbles: list[tuple[int, str]],
        lang: str,
        chapter_id: str,
        extra_glossary: dict[str, str] | None = None,
    ) -> dict[int, str]:
        self.calls += 1
        glossary = {**self.memory.get_glossary(chapter_id), **(extra_glossary or {})}
        out: dict[int, str] = {}
        for idx, text in bubbles:
            if "藍星" in text:
                name = glossary.get("藍星", "Aiboshi")
                out[idx] = f"{name}...?"
            elif "ぎゃ" in text:
                out[idx] = "GYAAAAAH!"
            else:
                out[idx] = f"EN:{text}"
        return out


@pytest.mark.asyncio
async def test_priority_queue_runs_visible_page_before_lookahead_and_serializes_gpu() -> None:
    active = 0
    max_concurrent = 0
    order: list[str] = []

    async def worker(tag: str) -> str:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.03)
        order.append(tag)
        active -= 1
        return tag

    q = PriorityGpuQueue()
    await q.start()
    t_running = asyncio.create_task(q.enqueue(priority=1, fn=lambda: worker("first")))
    await asyncio.sleep(0.005)
    t_low = asyncio.create_task(q.enqueue(priority=1, fn=lambda: worker("lookahead")))
    t_high = asyncio.create_task(q.enqueue(priority=0, fn=lambda: worker("visible")))
    results = await asyncio.gather(t_running, t_low, t_high)
    await q.stop()

    assert max_concurrent == 1
    assert order == ["first", "visible", "lookahead"]
    assert results == ["first", "lookahead", "visible"]


@pytest.mark.asyncio
async def test_priority_queue_cancel_skips_queued_item() -> None:
    ran: list[str] = []
    gate = asyncio.Event()

    async def blocker() -> str:
        await gate.wait()
        ran.append("blocker")
        return "blocker"

    async def skipped() -> str:
        ran.append("skipped")
        return "skipped"

    q = PriorityGpuQueue()
    await q.start()
    t_block = asyncio.create_task(q.enqueue(priority=0, fn=blocker, request_id="r1"))
    await asyncio.sleep(0.01)
    t_cancel = asyncio.create_task(q.enqueue(priority=1, fn=skipped, request_id="r2"))
    await asyncio.sleep(0.01)
    q.cancel("r2")
    gate.set()

    assert await t_block == "blocker"
    with pytest.raises(asyncio.CancelledError):
        await t_cancel
    await q.stop()
    assert ran == ["blocker"]


@pytest.mark.asyncio
async def test_priority_queue_bounded_evicts_oldest_lookahead_when_full() -> None:
    gate = asyncio.Event()
    executed: list[str] = []

    async def make_job(tag: str):
        if tag == "head":
            await gate.wait()
        executed.append(tag)
        return tag

    q = PriorityGpuQueue(max_queue_size=3)
    await q.start()

    t_head = asyncio.create_task(q.enqueue(priority=0, fn=lambda: make_job("head"), request_id="head"))
    await asyncio.sleep(0.01)
    t_la1 = asyncio.create_task(q.enqueue(priority=1, fn=lambda: make_job("la1"), request_id="la1"))
    t_la2 = asyncio.create_task(q.enqueue(priority=1, fn=lambda: make_job("la2"), request_id="la2"))
    t_la3 = asyncio.create_task(q.enqueue(priority=1, fn=lambda: make_job("la3"), request_id="la3"))
    await asyncio.sleep(0.01)
    # Queue is full (la1, la2, la3). Adding visible priority=0 must evict oldest lookahead (la1).
    t_vis = asyncio.create_task(q.enqueue(priority=0, fn=lambda: make_job("vis"), request_id="vis"))
    await asyncio.sleep(0.01)
    gate.set()

    res_head = await t_head
    res_vis = await t_vis
    res_la2 = await t_la2
    res_la3 = await t_la3
    with pytest.raises(asyncio.CancelledError):
        await t_la1
    await q.stop()

    assert res_head == "head"
    assert res_vis == "vis"
    assert res_la2 == "la2"
    assert res_la3 == "la3"
    assert "la1" not in executed
    assert executed == ["head", "vis", "la2", "la3"]


@pytest.mark.asyncio
async def test_pipeline_orchestrates_stages_filters_noise_sorts_and_caches(tmp_path: Path) -> None:
    memory = ChapterMemoryStore()
    memory.set_glossary("ch1", {"藍星": "Ranshin"})
    cache = TranslationCache(tmp_path / "translations.db", max_entries=50)
    detector = _FakeDetector()
    ocr = _FakeOcr()
    translator = _FakeTranslator(memory)
    pipeline = TranslationPipeline(detector=detector, ocr=ocr, translator=translator, memory=memory, cache=cache)
    await pipeline.start()

    img_bytes = _make_sample_png_bytes()
    resp1 = await pipeline.translate_image(img_bytes, lang="auto", chapter_id="ch1", priority=0)
    assert resp1.lang == "ja"
    assert resp1.cached is False
    # "..." noise bubble filtered out -> 2 bubbles remain, sorted RTL (right bubble id=0, left dark bubble id=1)
    assert len(resp1.bubbles) == 2
    assert resp1.bubbles[0].original == "藍星……？"
    assert resp1.bubbles[0].translated == "Ranshin...?"
    assert resp1.bubbles[0].is_dark is False
    assert resp1.bubbles[1].original == "ぎゃああああっ"
    assert resp1.bubbles[1].translated == "GYAAAAAH!"
    assert resp1.bubbles[1].is_dark is True
    assert resp1.bubbles[1].text_color == "#FFFFFF"

    # Second call with same inputs hits cache immediately (no GPU/detector/OCR/LLM calls)
    resp2 = await pipeline.translate_image(img_bytes, lang="auto", chapter_id="ch1", priority=0)
    assert resp2.cached is True
    assert detector.calls == 1
    assert ocr.calls == 1
    assert translator.calls == 1

    # Different glossary invalidates cache key
    resp3 = await pipeline.translate_image(
        img_bytes,
        lang="auto",
        chapter_id="ch1",
        glossary={"藍星": "Aiboshi"},
    )
    assert resp3.cached is False
    assert resp3.bubbles[0].translated == "Aiboshi...?"
    assert detector.calls == 2

    await pipeline.stop()


@pytest.mark.asyncio
async def test_pipeline_downscales_wide_image_and_scales_boxes_back(tmp_path: Path) -> None:
    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=20)
    # At max_image_width=1000, a 2000x1000 image scales by 0.5 to 1000x500.
    detector = _FakeDetector(
        detections=[RawDetection(Box(x1=100, y1=50, x2=300, y2=200), "bubble", 0.95)]
    )
    ocr = _FakeOcr(texts=["テスト"])
    translator = _FakeTranslator(memory)
    pipeline = TranslationPipeline(
        detector=detector,
        ocr=ocr,
        translator=translator,
        memory=memory,
        cache=cache,
        max_image_width=1000,
    )
    await pipeline.start()

    wide_bytes = _make_sample_png_bytes(width=2000, height=1000, draw_dark_bubble=False)
    resp = await pipeline.translate_image(wide_bytes, lang="ja")
    await pipeline.stop()

    assert detector.seen_sizes == [(1000, 500)]
    assert resp.width == 2000
    assert resp.height == 1000
    assert resp.bubbles[0].bbox == Box(x1=200, y1=100, x2=600, y2=400)


@pytest.mark.asyncio
async def test_pipeline_rejects_oversized_image_bytes(tmp_path: Path) -> None:
    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=20)
    pipeline = TranslationPipeline(
        detector=_FakeDetector(),
        ocr=_FakeOcr(),
        translator=_FakeTranslator(memory),
        memory=memory,
        cache=cache,
        max_image_bytes=128,
    )
    with pytest.raises(PayloadTooLargeError):
        await pipeline.translate_image(b"x" * 256, lang="ja")


def test_api_health_translate_glossary_memory_and_static_pages(tmp_path: Path) -> None:
    cfg = AppConfig(project_root=Path(__file__).resolve().parent.parent, cache_dir=tmp_path / "cache")
    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=50)
    detector = _FakeDetector()
    ocr = _FakeOcr()
    translator = _FakeTranslator(memory)
    pipeline = TranslationPipeline(
        detector=detector,
        ocr=ocr,
        translator=translator,
        memory=memory,
        cache=cache,
        max_image_bytes=500_000,
    )

    app = create_app(cfg=cfg, pipeline=pipeline, cache=cache, memory=memory, auto_load_models=False)
    img_bytes = _make_sample_png_bytes()
    b64_data_url = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")

    with TestClient(app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["ready"] is True

        # Glossary update + read
        r_gpost = client.post("/glossary", json={"chapter_id": "ch42", "entries": {"藍星": "Ranshin"}})
        assert r_gpost.status_code == 200
        r_gget = client.get("/glossary", params={"chapter_id": "ch42"})
        assert r_gget.status_code == 200
        assert r_gget.json()["entries"] == {"藍星": "Ranshin"}

        # JSON /translate (miss then hit)
        r_tr1 = client.post(
            "/translate",
            json={"image_base64": b64_data_url, "lang": "auto", "chapter_id": "ch42", "priority": 0},
        )
        assert r_tr1.status_code == 200
        data1 = r_tr1.json()
        assert data1["cached"] is False
        assert data1["lang"] == "ja"
        assert data1["bubbles"][0]["translated"] == "Ranshin...?"

        r_tr2 = client.post(
            "/translate",
            json={"image_base64": b64_data_url, "lang": "auto", "chapter_id": "ch42", "priority": 0},
        )
        assert r_tr2.status_code == 200
        assert r_tr2.json()["cached"] is True

        # Multipart /translate upload
        r_multi = client.post(
            "/translate",
            files={"file": ("page.png", img_bytes, "image/png")},
            data={"lang": "ja", "chapter_id": "ch42", "priority": "0", "glossary": '{"藍星": "Ranshin"}'},
        )
        assert r_multi.status_code == 200
        assert r_multi.json()["lang"] == "ja"

        # Oversized payload rejected cleanly with 413
        oversized_b64 = base64.b64encode(b"0" * 600_000).decode("ascii")
        r_big = client.post("/translate", json={"image_base64": oversized_b64})
        assert r_big.status_code == 413

        # Cancel, memory clear, cache clear
        assert client.post("/translate/cancel", json={"request_id": "missing-id"}).status_code == 200
        assert client.post("/memory/clear", json={"chapter_id": "ch42"}).status_code == 200
        assert client.post("/cache/clear").status_code == 200

        # Static dashboard & test harness pages
        r_idx = client.get("/")
        assert r_idx.status_code == 200
        assert "Open Live Visual Test Page" in r_idx.text

        r_test = client.get("/test")
        assert r_test.status_code == 200
        assert "/ext/overlay.js" in r_test.text
        assert "jp_centipede.jpg" in r_test.text

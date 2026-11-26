from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
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
from server.translator.translator import DialogueTranslator
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


def test_api_translate_stream_ndjson_miss_and_cache_hit(tmp_path: Path) -> None:
    cfg = AppConfig(project_root=Path(__file__).resolve().parent.parent, cache_dir=tmp_path / "cache")
    memory = ChapterMemoryStore()
    memory.set_glossary("ch-stream", {"藍星": "Ranshin"})
    cache = TranslationCache(tmp_path / "translations.db", max_entries=50)
    detector = _FakeDetector()
    ocr = _FakeOcr()

    async def fake_stream(_messages):
        yield '{"0": "Ranshin...?", '
        yield '"1": "GYAAAAAH!"}'

    translator = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_stream_caller=fake_stream,
    )
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
        with client.stream(
            "POST",
            "/translate/stream",
            json={"image_base64": b64_data_url, "lang": "auto", "chapter_id": "ch-stream", "priority": 0},
        ) as resp1:
            assert resp1.status_code == 200
            assert resp1.headers["content-type"].startswith("application/x-ndjson")
            events1 = [json.loads(line) for line in resp1.iter_lines() if line.strip()]

        ocr_status_events1 = [e for e in events1 if e["type"] == "status" and e.get("stage") == "ocr"]
        assert ocr_status_events1 == [
            {"type": "status", "stage": "ocr", "done": 0, "total": 3},
            {"type": "status", "stage": "ocr", "done": 3, "total": 3},
        ]
        translate_status_events1 = [
            e for e in events1 if e["type"] == "status" and e.get("stage") == "translate"
        ]
        assert translate_status_events1 == [
            {"type": "status", "stage": "translate", "provider": "local:ja"},
        ]
        meta_idx1 = next(i for i, e in enumerate(events1) if e["type"] == "meta")
        assert meta_idx1 == len(ocr_status_events1)
        meta_ev1 = events1[meta_idx1]
        assert meta_ev1["lang"] == "ja"
        assert len(meta_ev1["bubbles"]) == 2
        assert isinstance(meta_ev1["bubbles"][0]["bbox"], list)
        assert len(meta_ev1["bubbles"][0]["bbox"]) == 4
        assert "translated" not in meta_ev1["bubbles"][0]

        bubble_events1 = [e for e in events1 if e["type"] == "bubble"]
        assert [(e["id"], e["translated"], e["final"]) for e in bubble_events1] == [
            (0, "Ranshin...?", False),
            (1, "GYAAAAAH!", False),
        ]

        assert events1[-1]["type"] == "done"
        assert events1[-1]["response"]["cached"] is False
        assert events1[-1]["response"]["bubbles"][0]["translated"] == "Ranshin...?"
        assert events1[-1]["response"]["bubbles"][1]["translated"] == "GYAAAAAH!"
        assert detector.calls == 1
        assert ocr.calls == 1

        # Second call hits cache and streams meta -> bubble (final=True) -> done immediately
        with client.stream(
            "POST",
            "/translate/stream",
            json={"image_base64": b64_data_url, "lang": "auto", "chapter_id": "ch-stream", "priority": 0},
        ) as resp2:
            assert resp2.status_code == 200
            events2 = [json.loads(line) for line in resp2.iter_lines() if line.strip()]

        assert events2[0]["type"] == "meta"
        assert [(e["id"], e["translated"], e["final"]) for e in events2 if e["type"] == "bubble"] == [
            (0, "Ranshin...?", True),
            (1, "GYAAAAAH!", True),
        ]
        assert events2[-1]["type"] == "done"
        assert events2[-1]["response"]["cached"] is True
        assert detector.calls == 1
        assert ocr.calls == 1


@pytest.mark.asyncio
async def test_pipeline_empty_string_fallback_never_returns_raw_japanese(tmp_path: Path) -> None:
    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=20)
    detector = _FakeDetector(
        detections=[RawDetection(Box(x1=40, y1=60, x2=200, y2=240), "bubble", 0.95)]
    )
    ocr = _FakeOcr(texts=["現在は2人が勝手にナナス"])

    class _MissingTranslator:
        async def translate_page(self, bubbles, lang, chapter_id, extra_glossary=None):
            return {}

    pipeline = TranslationPipeline(
        detector=detector,
        ocr=ocr,
        translator=_MissingTranslator(),
        memory=memory,
        cache=cache,
    )
    await pipeline.start()
    resp = await pipeline.translate_image(_make_sample_png_bytes(), lang="ja")
    await pipeline.stop()

    assert len(resp.bubbles) == 1
    assert resp.bubbles[0].original == "現在は2人が勝手にナナス"
    assert resp.bubbles[0].translated == ""


def test_api_translate_stream_returns_503_when_not_ready(tmp_path: Path) -> None:
    cfg = AppConfig(project_root=Path(__file__).resolve().parent.parent, cache_dir=tmp_path / "cache")
    app = create_app(cfg=cfg, pipeline=None, auto_load_models=False)
    img_bytes = _make_sample_png_bytes()
    b64_data_url = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")

    with TestClient(app) as client:
        resp = client.post("/translate/stream", json={"image_base64": b64_data_url})
        assert resp.status_code == 503


class _FloatingAwareDetector:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def detect(self, image: Image.Image, *, include_floating: bool = True) -> list[RawDetection]:
        self.calls.append(include_floating)
        out = [
            RawDetection(Box(x1=380, y1=60, x2=560, y2=240), "bubble", 0.95, region_type="bubble"),
        ]
        if include_floating:
            out.append(
                RawDetection(Box(x1=40, y1=260, x2=200, y2=400), "text_free", 0.85, region_type="floating")
            )
        return out


@pytest.mark.asyncio
async def test_pipeline_floating_toggle_region_type_and_cache_hit_memory_recording(tmp_path: Path) -> None:
    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=50)
    detector = _FloatingAwareDetector()
    ocr = _FakeOcr(texts=["藍星……？", "カサカサ"])
    seen_bubble_types: list[dict[int, str] | None] = []
    seen_series_ids: list[str | None] = []

    class _CapturingTranslator:
        async def translate_page(
            self,
            bubbles,
            lang,
            chapter_id,
            extra_glossary=None,
            series_id=None,
            bubble_types=None,
        ):
            seen_series_ids.append(series_id)
            seen_bubble_types.append(bubble_types)
            out = {bid: ("Ranshin...?" if "藍星" in txt else "*RUSTLE*") for bid, txt in bubbles}
            memory.record_page(
                chapter_id,
                [(txt, out[bid]) for bid, txt in bubbles],
                series_id=series_id,
            )
            return out

    pipeline = TranslationPipeline(
        detector=detector,
        ocr=ocr,
        translator=_CapturingTranslator(),
        memory=memory,
        cache=cache,
    )
    await pipeline.start()

    img_bytes = _make_sample_png_bytes()
    resp_with_floating = await pipeline.translate_image(
        img_bytes,
        lang="ja",
        chapter_id="ch1",
        include_floating=True,
        series_id="my-manga",
    )
    assert resp_with_floating.cached is False
    assert [b.region_type for b in resp_with_floating.bubbles] == ["bubble", "floating"]
    assert detector.calls == [True]
    assert seen_series_ids == ["my-manga"]
    assert seen_bubble_types == [{0: "bubble", 1: "floating"}]

    # Clear memory and hit cache on ch2 -> cached bubbles MUST still feed chapter/series context!
    memory.clear("ch1", series_id="my-manga")
    assert memory.get_context("ch2", series_id="my-manga") == []
    resp_cached = await pipeline.translate_image(
        img_bytes,
        lang="ja",
        chapter_id="ch2",
        include_floating=True,
        series_id="my-manga",
    )
    assert resp_cached.cached is True
    assert memory.get_context("ch2", series_id="my-manga") == [
        ("藍星……？", "Ranshin...?"),
        ("カサカサ", "*RUSTLE*"),
    ]

    # Toggling include_floating=False uses a distinct cache key and omits floating region
    resp_no_floating = await pipeline.translate_image(
        img_bytes,
        lang="ja",
        chapter_id="ch2",
        include_floating=False,
        series_id="my-manga",
    )
    assert resp_no_floating.cached is False
    assert [b.region_type for b in resp_no_floating.bubbles] == ["bubble"]
    assert detector.calls == [True, False]

    # Stream cache hit also records context and includes region_type in meta event
    memory.clear("ch3", series_id="series-stream")
    stream_events = [
        ev
        async for ev in pipeline.translate_image_stream(
            img_bytes,
            lang="ja",
            chapter_id="ch3",
            include_floating=True,
            series_id="series-stream",
        )
    ]
    assert stream_events[0]["type"] == "meta"
    assert [b["region_type"] for b in stream_events[0]["bubbles"]] == ["bubble", "floating"]
    assert memory.get_context("ch3", series_id="series-stream") == [
        ("藍星……？", "Ranshin...?"),
        ("カサカサ", "*RUSTLE*"),
    ]

    await pipeline.stop()


def test_api_region_type_floating_toggle_series_glossary_and_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server.app as app_mod

    cfg = AppConfig(project_root=Path(__file__).resolve().parent.parent, cache_dir=tmp_path / "cache")
    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=50)
    detector = _FloatingAwareDetector()
    ocr = _FakeOcr(texts=["藍星……？", "カサカサ"])

    async def fake_llm(_messages):
        return '{"0": "Ranshin...?", "1": "*RUSTLE*"}'

    async def fake_stream(_messages):
        yield '{"0": "Ranshin...?", "1": "*RUSTLE*"}'

    translator = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=fake_llm,
        llm_stream_caller=fake_stream,
    )
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

    async def fake_lookup(query: str, source: str = "auto"):
        if query == "bad-host":
            raise app_mod.InvalidLookupQueryError("Unsupported host")
        if query == "upstream-down":
            raise app_mod.GlossaryLookupError("Upstream 503")
        return {
            "title": "Blue Box",
            "source": "anilist" if source == "auto" else source,
            "url": "https://anilist.co/manga/132182",
            "entries": {"猪股大喜": "Taiki Inomata", "鹿野千夏": "Chinatsu Kano"},
            "count": 2,
            "warnings": [],
        }

    monkeypatch.setattr(app_mod, "lookup_character_glossary", fake_lookup)

    with TestClient(app) as client:
        # Series-level glossary persists across chapters
        r_set = client.post(
            "/glossary",
            json={"chapter_id": "ch1", "series_id": "series-1", "entries": {"藍星": "Ranshin"}},
        )
        assert r_set.status_code == 200
        assert r_set.json()["series_id"] == "series-1"
        r_get_ch2 = client.get("/glossary", params={"chapter_id": "ch2", "series_id": "series-1"})
        assert r_get_ch2.status_code == 200
        assert r_get_ch2.json()["entries"] == {"藍星": "Ranshin"}

        # /translate/stream includes region_type in live meta and done events
        with client.stream(
            "POST",
            "/translate/stream",
            json={
                "image_base64": b64_data_url,
                "lang": "ja",
                "chapter_id": "ch1",
                "series_id": "series-1",
                "include_floating": True,
            },
        ) as resp:
            assert resp.status_code == 200
            events = [json.loads(line) for line in resp.iter_lines() if line.strip()]
        assert events[0] == {"type": "status", "stage": "ocr", "done": 0, "total": 2}
        meta_ev = next(e for e in events if e["type"] == "meta")
        assert [b["region_type"] for b in meta_ev["bubbles"]] == ["bubble", "floating"]
        assert [b["region_type"] for b in events[-1]["response"]["bubbles"]] == ["bubble", "floating"]

        # Multipart /translate with include_floating=false omits floating regions
        r_multi = client.post(
            "/translate",
            files={"file": ("page.png", img_bytes, "image/png")},
            data={
                "lang": "ja",
                "chapter_id": "ch1",
                "series_id": "series-1",
                "include_floating": "false",
            },
        )
        assert r_multi.status_code == 200
        assert [b["region_type"] for b in r_multi.json()["bubbles"]] == ["bubble"]

        # POST /glossary/lookup success + 400 + 502 error handling
        r_ok = client.post("/glossary/lookup", json={"query": "Blue Box", "source": "auto"})
        assert r_ok.status_code == 200
        payload = r_ok.json()
        assert payload["title"] == "Blue Box"
        assert payload["count"] == 2
        assert payload["entries"]["猪股大喜"] == "Taiki Inomata"

        assert client.post("/glossary/lookup", json={"query": "   "}).status_code == 400
        assert client.post("/glossary/lookup", json={"query": "bad-host"}).status_code == 400
        assert client.post("/glossary/lookup", json={"query": "upstream-down"}).status_code == 502

        # POST /memory/clear with series_id
        r_clear = client.post("/memory/clear", json={"chapter_id": "ch1", "series_id": "series-1"})
        assert r_clear.status_code == 200
        assert r_clear.json()["series_id"] == "series-1"


@pytest.mark.asyncio
async def test_pipeline_cancel_request_aborts_inflight_chunked_translation_and_skips_cache(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=20)
    dets = [
        RawDetection(Box(x1=10 + (i % 5) * 40, y1=10 + (i // 5) * 40, x2=40 + (i % 5) * 40, y2=40 + (i // 5) * 40), "bubble", 0.95)
        for i in range(25)
    ]
    detector = _FakeDetector(detections=dets)
    ocr = _FakeOcr(texts=[f"台詞{i}" for i in range(25)])

    chunk_calls = 0
    started_chunk1 = asyncio.Event()
    proceed_chunk1 = asyncio.Event()

    async def slow_chunk_llm(messages):
        nonlocal chunk_calls
        chunk_calls += 1
        user_msg = messages[1]["content"]
        in_map = json.loads(user_msg[user_msg.rfind("{") :])
        if chunk_calls == 1:
            started_chunk1.set()
            await proceed_chunk1.wait()
        return json.dumps({k: f"Translated {k}" for k in in_map})

    translator = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=slow_chunk_llm,
        cfg=SimpleNamespace(use_story_context=False, llm_chunk_size=10),
    )
    pipeline = TranslationPipeline(
        detector=detector,
        ocr=ocr,
        translator=translator,
        memory=memory,
        cache=cache,
    )
    await pipeline.start()

    img_bytes = _make_sample_png_bytes()
    task = asyncio.create_task(
        pipeline.translate_image(img_bytes, lang="ja", chapter_id="ch-cancel", request_id="req-inflight")
    )
    await asyncio.wait_for(started_chunk1.wait(), timeout=2.0)
    # Cancel WHILE request is actively in-flight in chunk 1
    pipeline.cancel_request("req-inflight")
    proceed_chunk1.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    await pipeline.stop()
    assert chunk_calls == 1
    assert cache.get(cache.make_key(img_bytes, "ja", {})) is None


@pytest.mark.asyncio
async def test_pipeline_wires_gpu_guard_into_stages_and_translator(tmp_path: Path) -> None:
    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=20)
    detector = _FakeDetector()
    ocr = _FakeOcr()

    guard_calls: list[bool] = []
    seen_guard_in_translator: list[object] = []

    class _FakeGpuGuard:
        async def wait_if_hot(self, cancel_event=None, max_wait_s=45.0) -> str:
            guard_calls.append(cancel_event is not None)
            return "ok"

    class _GuardObservingTranslator:
        async def translate_page(
            self,
            bubbles,
            lang,
            chapter_id,
            extra_glossary=None,
            cancel_event=None,
            gpu_guard=None,
        ):
            seen_guard_in_translator.append(gpu_guard)
            return {bid: "Valid English line." for bid, _ in bubbles}

    fake_guard = _FakeGpuGuard()
    pipeline = TranslationPipeline(
        detector=detector,
        ocr=ocr,
        translator=_GuardObservingTranslator(),
        memory=memory,
        cache=cache,
        gpu_guard=fake_guard,
    )
    await pipeline.start()
    resp = await pipeline.translate_image(_make_sample_png_bytes(), lang="ja", chapter_id="ch-guard")
    await pipeline.stop()

    assert len(resp.bubbles) == 2
    assert len(guard_calls) >= 3
    assert all(guard_calls)
    assert seen_guard_in_translator == [fake_guard]


@pytest.mark.asyncio
async def test_pipeline_stream_emits_incremental_ocr_status_events_and_cancels_between_slices(
    tmp_path: Path,
) -> None:
    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=20)
    dets = [
        RawDetection(
            Box(x1=10 + (i % 5) * 40, y1=10 + (i // 5) * 40, x2=40 + (i % 5) * 40, y2=40 + (i // 5) * 40),
            "bubble",
            0.95,
        )
        for i in range(20)
    ]
    detector = _FakeDetector(detections=dets)

    class _SliceCountingOcr:
        def __init__(self) -> None:
            self.slice_sizes: list[int] = []

        def read_page_bubbles(self, crops, lang_hint="auto", aspect_ratio=1.0):
            self.slice_sizes.append(len(crops))
            resolved = "ko" if lang_hint == "ko" else "ja"
            return resolved, [f"台詞{i}" for i in range(len(crops))]

    ocr = _SliceCountingOcr()
    translator = _FakeTranslator(memory)
    pipeline = TranslationPipeline(
        detector=detector,
        ocr=ocr,
        translator=translator,
        memory=memory,
        cache=cache,
    )
    await pipeline.start()

    img_bytes = _make_sample_png_bytes()
    events = [
        ev
        async for ev in pipeline.translate_image_stream(
            img_bytes,
            lang="ja",
            chapter_id="ch-ocr-stream",
        )
    ]
    status_events = [e for e in events if e["type"] == "status"]
    assert status_events == [
        {"type": "status", "stage": "ocr", "done": 0, "total": 20},
        {"type": "status", "stage": "ocr", "done": 8, "total": 20},
        {"type": "status", "stage": "ocr", "done": 16, "total": 20},
        {"type": "status", "stage": "ocr", "done": 20, "total": 20},
    ]
    assert ocr.slice_sizes == [8, 8, 4]
    meta_idx = next(i for i, e in enumerate(events) if e["type"] == "meta")
    assert meta_idx == len(status_events)
    assert events[-1]["type"] == "done"
    assert len(events[-1]["response"]["bubbles"]) == 20

    # Cancelling during OCR slice 1 aborts immediately before remaining OCR slices or translation
    cache.clear()
    ocr_cancel_calls = 0
    slice1_started = asyncio.Event()
    slice1_release = asyncio.Event()

    class _SlowCancelOcr:
        async def read_page_bubbles(self, crops, lang_hint="auto", aspect_ratio=1.0):
            nonlocal ocr_cancel_calls
            ocr_cancel_calls += 1
            if ocr_cancel_calls == 1:
                slice1_started.set()
                await slice1_release.wait()
            return "ja", [f"台詞{i}" for i in range(len(crops))]

    translator2 = _FakeTranslator(memory)
    pipeline2 = TranslationPipeline(
        detector=detector,
        ocr=_SlowCancelOcr(),
        translator=translator2,
        memory=memory,
        cache=cache,
    )
    await pipeline2.start()

    task = asyncio.create_task(
        pipeline2.translate_image(img_bytes, lang="ja", chapter_id="ch-ocr-cancel", request_id="ocr-req-1")
    )
    await asyncio.wait_for(slice1_started.wait(), timeout=2.0)
    pipeline2.cancel_request("ocr-req-1")
    slice1_release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    await pipeline.stop()
    await pipeline2.stop()
    assert ocr_cancel_calls == 1
    assert translator2.calls == 0


def test_startup_progress_tracker_monotonic_asymptotic_and_states() -> None:
    from server.app import _StartupProgressTracker

    now = [0.0]

    def fake_now() -> float:
        return now[0]

    # 1. Local mode (lazy_llm=False)
    tracker = _StartupProgressTracker(lazy_llm=False, stage="unloaded", now_fn=fake_now)
    assert tracker.get_progress() == (0, "")

    tracker.set_step("initializing")
    p0, d0 = tracker.get_progress()
    assert p0 == 0
    assert d0

    tracker.set_step("starting_llm", "Starting local LLM")
    p_llm_start, d_llm = tracker.get_progress()
    assert p_llm_start == 5
    assert d_llm == "Starting local LLM"

    now[0] = 15.0
    p_llm_mid, _ = tracker.get_progress()
    assert p_llm_start < p_llm_mid < 35

    now[0] = 10_000.0
    p_llm_cap, _ = tracker.get_progress()
    assert p_llm_cap == 34  # capped at ceiling - 1

    # Detector sub-steps advance monotonically
    tracker.set_step("importing_transformers", "Importing vision libraries")
    p_imp, _ = tracker.get_progress()
    assert p_imp >= p_llm_cap
    tracker.set_step("loading_weights", "Loading detector weights")
    p_wt, _ = tracker.get_progress()
    assert p_wt > p_imp

    # Error freezes progress even as time advances
    now[0] = 10_005.0
    p_before_err, _ = tracker.get_progress()
    tracker.mark_error("CUDA out of memory")
    now[0] = 20_000.0
    p_err, d_err = tracker.get_progress()
    assert p_err == p_before_err
    assert d_err == "CUDA out of memory"

    # Ready returns 100; restarting_llm resets ramp and returns to 100 on completion
    tracker.mark_ready()
    assert tracker.get_progress()[0] == 100
    tracker.set_step("restarting_llm", "Restarting local LLM")
    p_restart, d_restart = tracker.get_progress()
    assert 0 <= p_restart < 100
    assert d_restart == "Restarting local LLM"
    tracker.mark_ready()
    assert tracker.get_progress()[0] == 100

    # 2. Lazy LLM mode (lazy_llm=True)
    now[0] = 0.0
    lazy_tracker = _StartupProgressTracker(lazy_llm=True, stage="initializing", now_fn=fake_now)
    lazy_tracker.set_step("loading_detector", "Loading bubble detector")
    p_det, _ = lazy_tracker.get_progress()
    lazy_tracker.set_step("loading_ocr_model", "Loading Japanese OCR")
    p_ocr, _ = lazy_tracker.get_progress()
    assert p_ocr > p_det


def test_bubble_detector_load_emits_progress_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace
    from server import gpu_guard
    from server.vision.detector import BubbleDetector

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def half(self):
            return self

        def float(self):
            return self

    fake_transformers = SimpleNamespace(
        AutoImageProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: object()),
        AutoModelForObjectDetection=SimpleNamespace(from_pretrained=lambda *a, **kw: FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(gpu_guard, "has_safe_vram_headroom", lambda min_free_mb: True)

    events: list[tuple[str, str]] = []
    det = BubbleDetector(device="cuda")
    det.load(progress_cb=lambda step, detail: events.append((step, detail)))

    assert events == [
        ("importing_transformers", "Importing vision libraries"),
        ("loading_processor", "Loading detector processor"),
        ("loading_weights", "Loading detector weights"),
        ("moving_to_gpu", "Moving detector to GPU"),
    ]


def test_api_503_progress_headers_health_progress_and_static_pages(tmp_path: Path) -> None:
    cfg = AppConfig(project_root=Path(__file__).resolve().parent.parent, cache_dir=tmp_path / "cache")
    app = create_app(cfg=cfg, pipeline=None, auto_load_models=False)
    img_bytes = _make_sample_png_bytes()
    b64_data_url = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")

    with TestClient(app) as client:
        # Unloaded state -> 0%
        h_unloaded = client.get("/health").json()
        assert h_unloaded["loading_stage"] == "unloaded"
        assert h_unloaded["loading_progress"] == 0
        assert h_unloaded["loading_detail"] == ""

        # Active loading sub-step -> progress & headers on 503
        app.state.loading_stage = "loading_ocr"
        app.state.progress_tracker.set_step("loading_ocr_model", "Loading Japanese OCR")
        h_loading = client.get("/health").json()
        assert h_loading["ready"] is False
        assert h_loading["loading_stage"] == "loading_ocr"
        assert 1 <= h_loading["loading_progress"] <= 99
        assert h_loading["loading_detail"] == "Loading Japanese OCR"

        for endpoint in ("/translate", "/translate/stream"):
            resp = client.post(endpoint, json={"image_base64": b64_data_url})
            assert resp.status_code == 503
            assert f"loading_ocr, {h_loading['loading_progress']}%" in resp.json()["detail"]
            assert resp.headers["x-mlt-loading-stage"] == "loading_ocr"
            assert int(resp.headers["x-mlt-loading-progress"]) == h_loading["loading_progress"]
            assert resp.headers["x-mlt-loading-detail"] == "Loading Japanese OCR"

        # Restarting LLM while ready -> progress < 100 in /health
        app.state.ready = True
        app.state.loading_stage = "restarting_llm"
        app.state.progress_tracker.set_step("restarting_llm", "Restarting local LLM")
        h_restart = client.get("/health").json()
        assert h_restart["ready"] is True
        assert h_restart["loading_stage"] == "restarting_llm"
        assert 0 <= h_restart["loading_progress"] < 100
        assert h_restart["loading_detail"] == "Restarting local LLM"

        # Static UI pages surface loading_progress and loading_detail
        idx_html = client.get("/").text
        test_html = client.get("/test").text
        assert "loading_progress" in idx_html and "loading_detail" in idx_html
        assert "loading_progress" in test_html and "loading_detail" in test_html


@pytest.mark.asyncio
async def test_load_production_models_uses_to_thread_and_wires_progress_cb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server.app as app_mod
    import server.ocr.engine as ocr_mod
    import server.translator.translator as tr_mod
    import server.vision.detector as det_mod

    to_thread_targets: list[str] = []
    orig_to_thread = asyncio.to_thread

    async def tracking_to_thread(func, /, *args, **kwargs):
        to_thread_targets.append(getattr(func, "__name__", repr(func)))
        return await orig_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(app_mod.asyncio, "to_thread", tracking_to_thread)
    monkeypatch.setattr(app_mod, "_detect_torch_device", lambda: "cpu")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    seen_progress_cbs: list[str] = []

    class FakeLlamaMgr:
        base_url = "http://127.0.0.1:8766"

        def __init__(self, cfg, gpu_guard=None):
            self._process = None

        def start(self):
            pass

        def stop(self):
            pass

        def is_healthy(self):
            return True

        def available_local_models(self):
            return {"ja": True, "ko": False}

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def load(self, progress_cb=None):
            assert callable(progress_cb)
            seen_progress_cbs.append("detector")
            progress_cb("loading_weights", "Loading detector weights")

    class FakeOcr:
        def __init__(self, **kwargs):
            pass

        def load_ja(self, progress_cb=None):
            assert callable(progress_cb)
            seen_progress_cbs.append("ocr")
            progress_cb("loading_ocr_model", "Loading Japanese OCR")

    class FakeTranslator:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("server.translator.llama_manager.LlamaServerManager", FakeLlamaMgr)
    monkeypatch.setattr(det_mod, "BubbleDetector", FakeDetector)
    monkeypatch.setattr(ocr_mod, "OcrEngine", FakeOcr)
    monkeypatch.setattr(tr_mod, "DialogueTranslator", FakeTranslator)

    cfg = AppConfig(project_root=tmp_path, cache_dir=tmp_path / "cache", translation_mode="local")
    app = create_app(
        cfg=cfg,
        cache=TranslationCache(tmp_path / "cache" / "t.db"),
        memory=ChapterMemoryStore(),
        auto_load_models=False,
    )
    await app_mod._load_production_models(app)

    assert app.state.ready is True
    assert app.state.loading_stage == "ready"
    assert seen_progress_cbs == ["detector", "ocr"]
    assert "<lambda>" in to_thread_targets or "_detect_torch_device" in to_thread_targets
    assert "load" in to_thread_targets
    assert "load_ja" in to_thread_targets
    await app.state.pipeline.stop()

"""Unit tests for OCR preprocessing, meaningful dialogue filtering, and script routing."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from server import gpu_guard
from server.ocr.engine import (
    OcrEngine,
    classify_script,
    is_meaningful_dialogue,
    preprocess_crop_for_ocr,
)


def test_dark_bubble_crop_is_inverted_for_ocr():
    dark = Image.fromarray(np.full((64, 64, 3), 10, dtype=np.uint8))
    inverted = preprocess_crop_for_ocr(dark, is_dark=True)
    assert np.array(inverted).mean() > 235
    normal = preprocess_crop_for_ocr(dark, is_dark=False)
    assert np.array(normal).mean() < 20


def test_dark_bubble_with_white_glyphs_inverts_to_dark_ink_on_light_bg():
    arr = np.full((64, 64, 3), 15, dtype=np.uint8)
    arr[20:44, 24:40] = 240
    inverted = np.asarray(preprocess_crop_for_ocr(Image.fromarray(arr), is_dark=True))
    assert inverted[5, 5].mean() > 230
    assert inverted[32, 32].mean() < 30


def test_meaningful_dialogue_filters_punctuation_and_keeps_short_screams():
    assert is_meaningful_dialogue("ぎ") is True
    assert is_meaningful_dialogue("蟲") is True
    assert is_meaningful_dialogue("뭐?!") is True
    assert is_meaningful_dialogue("42") is True
    assert is_meaningful_dialogue("...") is False
    assert is_meaningful_dialogue("！？") is False
    assert is_meaningful_dialogue("——") is False
    assert is_meaningful_dialogue("   ") is False
    assert is_meaningful_dialogue("") is False


def test_classify_script_distinguishes_japanese_and_korean():
    assert classify_script(["今！助けます！", "蜈蚣が！"], aspect_ratio=0.7) == "ja"
    assert classify_script(["지금 바로 구해줄게!", "지네야!"], aspect_ratio=0.15) == "ko"
    # Tiebreaker on empty/ambiguous text supports both h/w (>2.2) and w/h (<0.45) tall strips
    assert classify_script(["..."], aspect_ratio=4.75) == "ko"
    assert classify_script(["..."], aspect_ratio=0.21) == "ko"
    assert classify_script(["..."], aspect_ratio=1.4) == "ja"


def test_ocr_engine_auto_probing_routes_tall_korean_strip_and_switches_when_needed():
    engine = OcrEngine(device="cpu")

    class FakeKoReader:
        def __init__(self) -> None:
            self.calls = 0

        def readtext(self, arr, detail=1, paragraph=False):
            self.calls += 1
            return [([[0, 0], [50, 0], [50, 20], [0, 20]], "지금 바로 구해줄게!", 0.98)]

    ja_calls = {"count": 0}

    def fake_ja(img: Image.Image) -> str:
        ja_calls["count"] += 1
        return "..."  # No Japanese glyphs when fed non-Japanese test crop

    fake_ko = FakeKoReader()
    engine._ja_ocr = fake_ja
    engine._ko_ocr = fake_ko

    crops = [
        (Image.new("RGB", (200, 120), (255, 255, 255)), False),
        (Image.new("RGB", (80, 60), (20, 20, 20)), True),
        (Image.new("RGB", (180, 100), (255, 255, 255)), False),
    ]

    # Tall aspect ratio (h/w = 4.75) probes Korean first without ever invoking Japanese OCR
    lang, texts = engine.read_page_bubbles(crops, lang_hint="auto", aspect_ratio=4.75)
    assert lang == "ko"
    assert len(texts) == 3
    assert all("구해줄게" in t for t in texts)
    assert ja_calls["count"] == 0

    # Standard page aspect ratio (1.4) probes Japanese first, sees zero Japanese chars, switches to Korean
    lang2, texts2 = engine.read_page_bubbles(crops, lang_hint="auto", aspect_ratio=1.4)
    assert lang2 == "ko"
    assert len(texts2) == 3
    assert ja_calls["count"] == 2  # probed only the 2 largest crops before switching


def test_load_ja_converts_model_to_fp16_on_cuda_and_preserves_offline_fallback(monkeypatch):
    class FakeTensor:
        def __init__(self, dtype: str = "fp32") -> None:
            self.dtype = dtype

        def half(self) -> FakeTensor:
            return FakeTensor("fp16")

    class FakeModel:
        def __init__(self) -> None:
            self.dtype = "fp32"

        def half(self) -> FakeModel:
            self.dtype = "fp16"
            return self

        def float(self) -> FakeModel:
            self.dtype = "fp32"
            return self

    attempts: list[tuple[bool, str | None]] = []

    class FakeMangaOcr:
        def __init__(self, force_cpu: bool = False) -> None:
            attempts.append((force_cpu, os.environ.get("HF_HUB_OFFLINE")))
            if len(attempts) == 1:
                raise OSError("missing offline snapshot")
            self.model = FakeModel()
            self.last_input_dtype: str | None = None

        def _preprocess(self, img: Image.Image) -> FakeTensor:
            return FakeTensor("fp32")

        def __call__(self, img: Image.Image) -> str:
            t = self._preprocess(img)
            self.last_input_dtype = t.dtype
            return "テスト"

    monkeypatch.setitem(sys.modules, "manga_ocr", SimpleNamespace(MangaOcr=FakeMangaOcr))
    monkeypatch.setattr(gpu_guard, "has_safe_vram_headroom", lambda min_free_mb: True)

    engine = OcrEngine(device="cuda", min_free_vram_mb=1024)
    engine.load_ja()

    assert attempts == [(False, "1"), (False, None)]
    assert engine._ja_fp16 is True
    assert engine._ja_ocr.model.dtype == "fp16"
    assert engine._ja_ocr.last_input_dtype == "fp16"


def test_load_ja_falls_back_to_fp32_when_fp16_fails(monkeypatch):
    class BrokenHalfModel:
        def __init__(self) -> None:
            self.dtype = "fp32"

        def half(self) -> BrokenHalfModel:
            self.dtype = "fp16"
            raise RuntimeError("half unsupported")

        def float(self) -> BrokenHalfModel:
            self.dtype = "fp32"
            return self

    class FakeMangaOcr:
        def __init__(self, force_cpu: bool = False) -> None:
            self.force_cpu = force_cpu
            self.model = BrokenHalfModel()

        def _preprocess(self, img: Image.Image) -> str:
            return "fp32"

        def __call__(self, img: Image.Image) -> str:
            return "ok"

    monkeypatch.setitem(sys.modules, "manga_ocr", SimpleNamespace(MangaOcr=FakeMangaOcr))
    monkeypatch.setattr(gpu_guard, "has_safe_vram_headroom", lambda min_free_mb: True)

    engine = OcrEngine(device="cuda", min_free_vram_mb=1024)
    engine.load_ja()

    assert engine._ja_fp16 is False
    assert engine._ja_ocr.model.dtype == "fp32"


def test_load_ja_uses_cpu_when_vram_headroom_insufficient(monkeypatch):
    seen: list[bool] = []

    class FakeMangaOcr:
        def __init__(self, force_cpu: bool = False) -> None:
            seen.append(force_cpu)
            self.model = None

    monkeypatch.setitem(sys.modules, "manga_ocr", SimpleNamespace(MangaOcr=FakeMangaOcr))
    monkeypatch.setattr(gpu_guard, "has_safe_vram_headroom", lambda min_free_mb: False)

    engine = OcrEngine(device="cuda", min_free_vram_mb=1024)
    engine.load_ja()
    assert seen == [True]
    assert engine._ja_fp16 is False

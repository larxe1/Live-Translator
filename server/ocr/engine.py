"""Specialized Japanese (manga-ocr) & Korean (EasyOCR) OCR engine with auto script routing."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps

# Meaningful dialogue character ranges (excludes standalone punctuation, ellipses, dashes, spaces)
_MEANINGFUL_RE = re.compile(
    r"[\u3040-\u309F"       # Hiragana
    r"\u30A1-\u30FA"        # Katakana letters (excludes middle dot & prolonged sound mark alone)
    r"\uFF66-\uFF9D"        # Halfwidth Katakana letters
    r"\u4E00-\u9FFF"        # CJK Unified Ideographs (Kanji/Hanja)
    r"\u3400-\u4DBF"        # CJK Extension A
    r"\uAC00-\uD7A3"        # Hangul Syllables
    r"\u3131-\u318E"        # Hangul Compatibility Jamo
    r"\u1100-\u11FF"        # Hangul Jamo
    r"A-Za-z0-9"            # ASCII Latin & digits
    r"\uFF10-\uFF19\uFF21-\uFF3A\uFF41-\uFF5A]"  # Fullwidth Latin & digits
)

_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3\u3130-\u318F\u1100-\u11FF]")
_JA_KANA_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\uFF66-\uFF9F]")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")


def preprocess_crop_for_ocr(crop: Image.Image, is_dark: bool) -> Image.Image:
    """Prepare a bubble crop for OCR, inverting dark white-on-black bubbles."""
    rgb = crop.convert("RGB") if crop.mode != "RGB" else crop.copy()
    if not is_dark:
        return rgb

    inverted = ImageOps.invert(rgb)
    arr = np.asarray(inverted)
    # Apply gentle autocontrast only when dynamic range exists (keeps uniform crops intact)
    if arr.size > 0 and int(arr.max()) - int(arr.min()) > 25:
        inverted = ImageOps.autocontrast(inverted, cutoff=1)
    return inverted


def is_meaningful_dialogue(text: str) -> bool:
    """Return True if text contains at least one Japanese, Korean, Latin, or numeric character."""
    if not text or not text.strip():
        return False
    return _MEANINGFUL_RE.search(text) is not None


def _is_tall_webtoon_aspect(aspect_ratio: float) -> bool:
    """Detect tall vertical webtoon strip ratios regardless of h/w vs w/h convention."""
    return aspect_ratio > 2.2 or (0.0 < aspect_ratio < 0.45)


def _count_scripts(texts: list[str]) -> tuple[int, int]:
    """Return (hangul_count, japanese_kana_plus_cjk_count)."""
    joined = "".join(texts)
    ko_count = len(_HANGUL_RE.findall(joined))
    ja_count = len(_JA_KANA_RE.findall(joined)) + len(_CJK_RE.findall(joined))
    return ko_count, ja_count


def classify_script(texts: list[str], aspect_ratio: float) -> Literal["ja", "ko"]:
    """Classify OCR output texts as Japanese ('ja') or Korean ('ko').

    Counts Hangul vs Hiragana/Katakana/CJK characters; uses ``aspect_ratio``
    only as a tiebreaker when counts are equal.
    """
    ko_count, ja_count = _count_scripts(texts)
    if ko_count > ja_count:
        return "ko"
    if ja_count > ko_count:
        return "ja"
    return "ko" if _is_tall_webtoon_aspect(aspect_ratio) else "ja"


def _has_cuda_vram_headroom(min_free_mb: int) -> bool:
    """Return True only when CUDA is available and has at least min_free_mb free VRAM."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return (free_bytes / (1024 * 1024)) >= float(min_free_mb)
    except Exception:
        return False


class OcrEngine:
    """Lazy-loading dual OCR engine for Japanese manga and Korean manhwa."""

    def __init__(self, device: str = "cuda", cache_dir: Path | None = None) -> None:
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._ja_ocr: Any = None
        self._ko_ocr: Any = None

    def load_ja(self) -> None:
        """Lazy-load manga-ocr onto CUDA (when VRAM headroom suffices) or CPU."""
        from manga_ocr import MangaOcr

        use_cuda = self.device.startswith("cuda") and _has_cuda_vram_headroom(min_free_mb=1400)
        self._ja_ocr = MangaOcr(force_cpu=not use_cuda)

    def load_ko(self) -> None:
        """Lazy-load EasyOCR Korean reader onto CUDA (when VRAM headroom suffices) or CPU."""
        import easyocr

        use_cuda = self.device.startswith("cuda") and _has_cuda_vram_headroom(min_free_mb=1600)

        storage_dir: Path | None = None
        if self.cache_dir is not None:
            storage_dir = self.cache_dir / "easyocr"
        elif os.environ.get("EASYOCR_MODULE_PATH"):
            storage_dir = Path(os.environ["EASYOCR_MODULE_PATH"])

        kwargs: dict[str, Any] = {"gpu": use_cuda, "verbose": False}
        if storage_dir is not None:
            storage_dir.mkdir(parents=True, exist_ok=True)
            kwargs["model_storage_directory"] = str(storage_dir)

        self._ko_ocr = easyocr.Reader(["ko"], **kwargs)

    def _maybe_empty_cuda_cache(self) -> None:
        """Release cached PyTorch CUDA allocator blocks when CUDA is active."""
        if not self.device.startswith("cuda"):
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _run_with_inference_mode(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute OCR callable inside torch.inference_mode() when PyTorch is installed."""
        try:
            import torch
        except ImportError:
            return fn(*args, **kwargs)

        try:
            with torch.inference_mode():
                return fn(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

    def read_bubble(
        self,
        crop: Image.Image,
        lang: Literal["ja", "ko"],
        is_dark: bool = False,
    ) -> str:
        """Read text from a single bubble crop using the selected language engine."""
        prepped = preprocess_crop_for_ocr(crop, is_dark=is_dark)

        if lang == "ja":
            if self._ja_ocr is None:
                self.load_ja()
            try:
                raw = self._run_with_inference_mode(self._ja_ocr, prepped)
            except Exception as exc:
                if "out of memory" in str(exc).lower() and self.device.startswith("cuda"):
                    self.device = "cpu"
                    self.load_ja()
                    raw = self._run_with_inference_mode(self._ja_ocr, prepped)
                else:
                    raise
            finally:
                del prepped
            return str(raw).strip()

        if self._ko_ocr is None:
            self.load_ko()
        arr = np.asarray(prepped)
        try:
            results = self._run_with_inference_mode(
                self._ko_ocr.readtext, arr, detail=1, paragraph=False
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower() and self.device.startswith("cuda"):
                self.device = "cpu"
                self.load_ko()
                results = self._run_with_inference_mode(
                    self._ko_ocr.readtext, arr, detail=1, paragraph=False
                )
            else:
                raise
        finally:
            del arr, prepped
        if not results:
            return ""

        # Support both EasyOCR detail=1 tuples (bbox, text, conf) and simple mocked strings
        if isinstance(results[0], str):
            return " ".join(s.strip() for s in results if s and s.strip()).strip()

        def _sort_key(entry: Any) -> tuple[float, float]:
            box = entry[0]
            try:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                return (min(ys), min(xs))
            except Exception:
                return (0.0, 0.0)

        ordered = sorted(results, key=_sort_key)
        parts = [str(entry[1]).strip() for entry in ordered if len(entry) >= 2 and str(entry[1]).strip()]
        return " ".join(parts).strip()

    def read_page_bubbles(
        self,
        crops: list[tuple[Image.Image, bool]],
        lang_hint: Literal["auto", "ja", "ko"],
        aspect_ratio: float,
    ) -> tuple[Literal["ja", "ko"], list[str]]:
        """Read all bubbles on a page, resolving 'auto' script via largest-crop probing."""
        if not crops:
            resolved: Literal["ja", "ko"] = (
                lang_hint if lang_hint in ("ja", "ko") else classify_script([], aspect_ratio)
            )
            return resolved, []

        if lang_hint in ("ja", "ko"):
            texts = [
                self.read_bubble(crop, lang=lang_hint, is_dark=is_dark)
                for crop, is_dark in crops
            ]
            self._maybe_empty_cuda_cache()
            return lang_hint, texts

        # Auto script routing: probe up to 2 largest crops
        indexed_by_area = sorted(
            range(len(crops)),
            key=lambda i: crops[i][0].width * crops[i][0].height,
            reverse=True,
        )
        probe_indices = indexed_by_area[:2]

        first_lang: Literal["ja", "ko"] = "ko" if _is_tall_webtoon_aspect(aspect_ratio) else "ja"
        other_lang: Literal["ja", "ko"] = "ja" if first_lang == "ko" else "ko"

        first_cache: dict[int, str] = {}
        for idx in probe_indices:
            crop, is_dark = crops[idx]
            first_cache[idx] = self.read_bubble(crop, lang=first_lang, is_dark=is_dark)

        first_texts = [first_cache[idx] for idx in probe_indices]
        ko_count, ja_count = _count_scripts(first_texts)

        chosen_lang: Literal["ja", "ko"] = first_lang
        chosen_cache: dict[int, str] = first_cache

        # Check if first probe produced meaningful target-script glyphs; if not, probe other engine
        target_hits = ko_count if first_lang == "ko" else ja_count
        if target_hits == 0:
            other_cache: dict[int, str] = {}
            for idx in probe_indices:
                crop, is_dark = crops[idx]
                other_cache[idx] = self.read_bubble(crop, lang=other_lang, is_dark=is_dark)
            other_texts = [other_cache[idx] for idx in probe_indices]
            other_ko, other_ja = _count_scripts(other_texts)
            other_hits = other_ko if other_lang == "ko" else other_ja
            if other_hits > 0:
                chosen_lang = other_lang
                chosen_cache = other_cache
            else:
                chosen_lang = classify_script(first_texts + other_texts, aspect_ratio)
                chosen_cache = first_cache if chosen_lang == first_lang else other_cache
        else:
            # Verify dominant script if mixed (e.g., Hangul appeared despite non-tall aspect)
            detected = classify_script(first_texts, aspect_ratio)
            if detected != first_lang:
                chosen_lang = detected
                chosen_cache = {}

        final_texts: list[str] = []
        for idx, (crop, is_dark) in enumerate(crops):
            if idx in chosen_cache:
                final_texts.append(chosen_cache[idx])
            else:
                final_texts.append(self.read_bubble(crop, lang=chosen_lang, is_dark=is_dark))

        self._maybe_empty_cuda_cache()
        return chosen_lang, final_texts

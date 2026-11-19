"""OCR stage: Japanese manga-ocr, Korean EasyOCR, dark-bubble inversion, and script routing."""

from server.ocr.engine import (
    OcrEngine,
    classify_script,
    is_meaningful_dialogue,
    preprocess_crop_for_ocr,
)

__all__ = [
    "OcrEngine",
    "classify_script",
    "is_meaningful_dialogue",
    "preprocess_crop_for_ocr",
]

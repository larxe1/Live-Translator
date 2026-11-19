"""Unit tests for tall webtoon slicing, seam deduplication, and bubble appearance geometry."""

from __future__ import annotations

import numpy as np
from PIL import Image

from server.schemas import Box
from server.vision.geometry import analyze_bubble_appearance, is_clean_narration_box
from server.vision.slicer import (
    RawDetection,
    merge_slice_detections,
    slice_tall_image,
)


def test_standard_height_page_is_not_sliced():
    img = Image.new("RGB", (900, 1400), (255, 255, 255))
    slices = slice_tall_image(img, slice_height=1600, overlap=250)
    assert len(slices) == 1
    assert slices[0].y_offset == 0
    assert slices[0].image.size == (900, 1400)


def test_tall_korean_strip_slices_and_deduplicates_seam_bubble():
    img = Image.new("RGB", (800, 3800), (255, 255, 255))
    slices = slice_tall_image(img, slice_height=1600, overlap=250)
    assert len(slices) == 3
    assert slices[0].y_offset == 0
    assert slices[1].y_offset == 1350
    assert slices[2].y_offset == 2700
    assert slices[-1].y_offset + slices[-1].image.height == 3800

    # Same bubble detected in slice 0 and slice 1 near y=1450
    d1 = RawDetection(Box(x1=100, y1=1400, x2=420, y2=1560), "bubble", 0.92)
    d2 = RawDetection(Box(x1=104, y1=1405, x2=418, y2=1558), "bubble", 0.88)
    # Distinct bubble lower down on the strip
    d3 = RawDetection(Box(x1=200, y1=2600, x2=520, y2=2820), "bubble", 0.90)
    merged = merge_slice_detections([d2, d1, d3], iou_threshold=0.45)
    assert len(merged) == 2
    assert merged[0].score == 0.92
    assert {m.box.y1 for m in merged} == {1400, 2600}


def test_merge_slice_detections_suppresses_high_containment_partial_seam_box():
    full_bubble = RawDetection(Box(x1=100, y1=1360, x2=420, y2=1620), "bubble", 0.93)
    clipped_half = RawDetection(Box(x1=102, y1=1362, x2=418, y2=1490), "bubble", 0.78)
    merged = merge_slice_detections([clipped_half, full_bubble], iou_threshold=0.45)
    assert len(merged) == 1
    assert merged[0].score == 0.93
    assert merged[0].box == full_bubble.box


def test_dark_bubble_detected_with_white_text_and_light_bubble_dark_text():
    canvas = np.full((300, 300, 3), 250, dtype=np.uint8)
    # Dark screaming bubble with white glyphs inside
    canvas[20:140, 20:140] = 15
    canvas[50:110, 65:95] = 245
    dark_style = analyze_bubble_appearance(canvas, Box(x1=20, y1=20, x2=140, y2=140))
    assert dark_style.is_dark is True
    assert dark_style.text_color == "#FFFFFF"
    assert dark_style.mean_luminance < 110.0
    assert dark_style.safe_box.x1 > 20 and dark_style.safe_box.x2 < 140
    assert dark_style.safe_box.x2 - dark_style.safe_box.x1 > 8
    assert dark_style.safe_box.y2 - dark_style.safe_box.y1 > 8

    light_style = analyze_bubble_appearance(canvas, Box(x1=160, y1=160, x2=280, y2=280))
    assert light_style.is_dark is False
    assert light_style.text_color == "#111111"
    assert light_style.mean_luminance > 200.0
    assert light_style.uniform_bg_ratio > 0.95


def test_clean_narration_box_rescued_but_artwork_sfx_rejected():
    # Gradient artwork background with stylized SFX text -> low uniform background ratio
    art = np.tile(np.linspace(30, 210, 200, dtype=np.uint8), (200, 1))
    art_rgb = np.stack([art, art // 2, 255 - art], axis=-1)
    assert is_clean_narration_box(art_rgb, Box(x1=20, y1=20, x2=180, y2=180)) is False

    # Solid white rectangular narration box with dark border
    box_rgb = np.full((200, 200, 3), 252, dtype=np.uint8)
    box_rgb[20:180, 20:23] = 10
    box_rgb[20:180, 177:180] = 10
    box_rgb[20:23, 20:180] = 10
    box_rgb[177:180, 20:180] = 10
    assert is_clean_narration_box(box_rgb, Box(x1=20, y1=20, x2=180, y2=180)) is True

    # Borderless mid-gray shaded panel (typical SFX background) -> rejected
    shaded = np.full((200, 200, 3), 135, dtype=np.uint8)
    assert is_clean_narration_box(shaded, Box(x1=20, y1=20, x2=180, y2=180)) is False

    # Enclosed dark narration box with crisp white border -> rescued
    dark_box = np.full((200, 200, 3), 245, dtype=np.uint8)
    dark_box[20:180, 20:180] = 20
    assert is_clean_narration_box(dark_box, Box(x1=20, y1=20, x2=180, y2=180)) is True

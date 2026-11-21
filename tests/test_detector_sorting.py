"""Unit tests for dialogue region filtering, bubble/text_bubble merging, and reading-order sort."""

from __future__ import annotations

import numpy as np

from server.schemas import Box
from server.vision.detector import filter_dialogue_regions, sort_reading_order
from server.vision.slicer import RawDetection


def test_filter_drops_text_free_over_artwork_when_include_floating_false():
    art = np.random.default_rng(42).integers(0, 255, size=(400, 400, 3), dtype=np.uint8)
    dets = [
        RawDetection(Box(x1=10, y1=10, x2=120, y2=150), "bubble", 0.90),
        RawDetection(Box(x1=200, y1=50, x2=300, y2=180), "text_free", 0.85),
        RawDetection(Box(x1=5, y1=5, x2=15, y2=18), "bubble", 0.95),  # tiny noise < 20px
    ]
    kept = filter_dialogue_regions(art, dets, include_floating=False)
    assert len(kept) == 1
    assert kept[0].label == "bubble"
    assert kept[0].region_type == "bubble"
    assert kept[0].box == Box(x1=10, y1=10, x2=120, y2=150)


def test_filter_keeps_floating_text_free_by_default_and_dedupes_overlaps():
    art = np.random.default_rng(42).integers(0, 255, size=(500, 500, 3), dtype=np.uint8)
    dets = [
        RawDetection(Box(x1=10, y1=10, x2=120, y2=150), "bubble", 0.90),
        # Overlaps bubble heavily -> must be skipped
        RawDetection(Box(x1=20, y1=20, x2=110, y2=140), "text_free", 0.92),
        # Standalone floating dialogue / SFX over artwork -> kept as "floating"
        RawDetection(Box(x1=200, y1=50, x2=320, y2=180), "text_free", 0.88),
        # Overlapping lower-score duplicate of the floating region above -> must be skipped
        RawDetection(Box(x1=205, y1=55, x2=315, y2=175), "text_free", 0.72),
        # Second distinct floating region -> kept as "floating"
        RawDetection(Box(x1=350, y1=250, x2=470, y2=360), "text_free", 0.80),
    ]
    kept = filter_dialogue_regions(art, dets)
    assert len(kept) == 3
    by_x1 = {k.box.x1: k for k in kept}
    assert by_x1[10].region_type == "bubble"
    assert by_x1[200].region_type == "floating"
    assert by_x1[200].score == 0.88
    assert by_x1[350].region_type == "floating"


def test_filter_merges_inner_text_bubble_into_enclosing_bubble_without_duplicates():
    canvas = np.full((400, 400, 3), 250, dtype=np.uint8)
    outer_bubble = RawDetection(Box(x1=30, y1=30, x2=170, y2=220), "bubble", 0.88)
    inner_text = RawDetection(Box(x1=55, y1=55, x2=145, y2=195), "text_bubble", 0.94)
    standalone_tb = RawDetection(Box(x1=240, y1=60, x2=350, y2=200), "text_bubble", 0.86)

    kept = filter_dialogue_regions(canvas, [outer_bubble, inner_text, standalone_tb])
    assert len(kept) == 2
    boxes = {k.box.x1: k for k in kept}
    assert 30 in boxes
    assert boxes[30].box == outer_bubble.box  # keeps full bubble bounds
    assert boxes[30].ocr_box == inner_text.box  # preserves tight inner text_bubble for OCR
    assert boxes[30].score == 0.94
    assert boxes[30].region_type == "bubble"
    assert 240 in boxes
    assert boxes[240].region_type == "bubble"


def test_filter_rescues_clean_narration_box_text_free():
    canvas = np.full((300, 300, 3), 252, dtype=np.uint8)
    # Crisp dark rectangular narration border around [40:200, 40:180]
    canvas[40:180, 40:43] = 15
    canvas[40:180, 197:200] = 15
    canvas[40:43, 40:200] = 15
    canvas[177:180, 40:200] = 15

    dets = [RawDetection(Box(x1=40, y1=40, x2=200, y2=180), "text_free", 0.81)]
    kept = filter_dialogue_regions(canvas, dets, include_floating=False)
    assert len(kept) == 1
    assert kept[0].label == "text_free"
    assert kept[0].region_type == "narration"


def test_japanese_rtl_and_two_page_spread_sorting():
    # Wide two-page spread (2000x1400): right page bubbles MUST all precede left page bubbles
    right_top = RawDetection(Box(x1=1600, y1=100, x2=1850, y2=300), "bubble", 0.9)
    right_bottom = RawDetection(Box(x1=1100, y1=900, x2=1350, y2=1100), "bubble", 0.9)
    left_top = RawDetection(Box(x1=600, y1=80, x2=850, y2=280), "bubble", 0.9)
    ordered = sort_reading_order([left_top, right_bottom, right_top], img_w=2000, img_h=1400, lang="ja")
    assert [d.box.x1 for d in ordered] == [1600, 1100, 600]


def test_japanese_single_page_band_rtl_sorting():
    top_right = RawDetection(Box(x1=550, y1=80, x2=740, y2=260), "bubble", 0.9)
    top_left = RawDetection(Box(x1=80, y1=110, x2=270, y2=290), "bubble", 0.9)
    bottom_right = RawDetection(Box(x1=520, y1=600, x2=730, y2=790), "bubble", 0.9)
    bottom_left = RawDetection(Box(x1=90, y1=620, x2=280, y2=810), "bubble", 0.9)

    ordered = sort_reading_order(
        [bottom_left, top_left, bottom_right, top_right],
        img_w=800,
        img_h=1200,
        lang="ja",
    )
    assert [d.box.x1 for d in ordered] == [550, 80, 520, 90]


def test_korean_manhwa_top_to_bottom_left_to_right_sorting():
    top_left = RawDetection(Box(x1=60, y1=100, x2=300, y2=260), "bubble", 0.9)
    top_right = RawDetection(Box(x1=440, y1=120, x2=720, y2=280), "bubble", 0.9)
    mid = RawDetection(Box(x1=200, y1=1500, x2=550, y2=1720), "bubble", 0.9)
    bottom = RawDetection(Box(x1=150, y1=3100, x2=520, y2=3320), "bubble", 0.9)

    ordered = sort_reading_order([bottom, top_right, mid, top_left], img_w=800, img_h=3800, lang="ko")
    assert [d.box.y1 for d in ordered] == [100, 120, 1500, 3100]

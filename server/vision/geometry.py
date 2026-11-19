"""Bubble interior appearance analysis, safe-box geometry, and narration-box detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from server.schemas import Box


@dataclass
class BubbleStyle:
    """Sampled visual style and safe text-layout geometry for a dialogue bubble."""

    safe_box: Box
    bg_color: str
    text_color: str
    is_dark: bool
    mean_luminance: float
    uniform_bg_ratio: float


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize 2D grayscale or RGBA arrays to uint8 HxWx3 RGB."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _clamp_box(bbox: Box, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Clamp box coordinates to valid non-empty image bounds."""
    x1 = max(0, min(bbox.x1, img_w - 1))
    y1 = max(0, min(bbox.y1, img_h - 1))
    x2 = max(x1 + 1, min(bbox.x2, img_w))
    y2 = max(y1 + 1, min(bbox.y2, img_h))
    return x1, y1, x2, y2


def _dominant_color(pixels: np.ndarray) -> tuple[int, int, int]:
    """Find the dominant RGB color via quantized mode cluster median."""
    if pixels.size == 0:
        return (255, 255, 255)

    pts = pixels.reshape(-1, 3).astype(np.int32)
    quantized = pts // 16
    keys = (quantized[:, 0] << 8) | (quantized[:, 1] << 4) | quantized[:, 2]
    unique_keys, counts = np.unique(keys, return_counts=True)
    best_key = unique_keys[int(np.argmax(counts))]
    mode_mask = keys == best_key
    mode_center = np.median(pts[mode_mask], axis=0)

    dists = np.linalg.norm(pts.astype(np.float32) - mode_center.astype(np.float32), axis=1)
    cluster = pts[dists <= 32.0]
    if cluster.shape[0] == 0:
        cluster = pts[mode_mask]
    med = np.median(cluster, axis=0)
    return (int(round(med[0])), int(round(med[1])), int(round(med[2])))


def _sample_interior_ring(roi: np.ndarray) -> np.ndarray:
    """Sample an elliptical annulus inside the bubble avoiding outline and central glyphs."""
    h, w = roi.shape[:2]
    if h < 8 or w < 8:
        return roi.reshape(-1, 3)

    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    ry = max(h / 2.0, 1.0)
    rx = max(w / 2.0, 1.0)
    norm_r2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2

    # r in [0.56, 0.86] stays strictly inside oval ink stroke and outside dense center glyphs
    ring_mask = (norm_r2 >= 0.56**2) & (norm_r2 <= 0.86**2)
    ring_pixels = roi[ring_mask]
    if ring_pixels.shape[0] >= 16:
        return ring_pixels

    # Fallback for tiny/extreme-aspect regions: 10% inset rectangle
    mx = max(1, int(round(w * 0.10)))
    my = max(1, int(round(h * 0.10)))
    sub = roi[my : max(my + 1, h - my), mx : max(mx + 1, w - mx)]
    return sub.reshape(-1, 3) if sub.size > 0 else roi.reshape(-1, 3)


def _compute_safe_box(x1: int, y1: int, x2: int, y2: int) -> Box:
    """Compute an inset safe text box strictly inside the bubble stroke."""
    w = x2 - x1
    h = y2 - y1
    inset_x = int(round(w * 0.06))
    inset_y = int(round(h * 0.06))

    # Clamp inset so safe_box stays strictly wider/taller than 8px whenever possible
    max_inset_x = max(0, (w - 9) // 2)
    max_inset_y = max(0, (h - 9) // 2)
    inset_x = min(inset_x, max_inset_x)
    inset_y = min(inset_y, max_inset_y)

    sx1 = x1 + inset_x
    sx2 = x2 - inset_x
    sy1 = y1 + inset_y
    sy2 = y2 - inset_y

    if sx2 <= sx1 + 8:
        sx1 = x1
        sx2 = max(x2, x1 + 9)
    if sy2 <= sy1 + 8:
        sy1 = y1
        sy2 = max(y2, y1 + 9)

    return Box(x1=int(sx1), y1=int(sy1), x2=int(sx2), y2=int(sy2))


def analyze_bubble_appearance(image_rgb: np.ndarray, bbox: Box) -> BubbleStyle:
    """Analyze bubble interior background color, darkness, uniformity, and safe text bounds."""
    rgb = _ensure_rgb(image_rgb)
    img_h, img_w = rgb.shape[:2]
    x1, y1, x2, y2 = _clamp_box(bbox, img_w, img_h)
    roi = rgb[y1:y2, x1:x2]
    h, w = roi.shape[:2]

    ring_pixels = _sample_interior_ring(roi)
    r, g, b = _dominant_color(ring_pixels)
    bg_vec = np.array([r, g, b], dtype=np.float32)

    mean_luminance = float(0.299 * r + 0.587 * g + 0.114 * b)
    is_dark = mean_luminance < 110.0
    bg_color = f"#{r:02X}{g:02X}{b:02X}"
    text_color = "#FFFFFF" if is_dark else "#111111"

    # Measure uniformity across the inset interior (excluding outer ink border)
    mx = min(max(2, int(round(w * 0.08))), max(0, (w - 2) // 2))
    my = min(max(2, int(round(h * 0.08))), max(0, (h - 2) // 2))
    interior = roi[my : max(my + 1, h - my), mx : max(mx + 1, w - mx)]
    if interior.size == 0:
        interior = roi
    dists = np.linalg.norm(interior.reshape(-1, 3).astype(np.float32) - bg_vec, axis=1)
    uniform_bg_ratio = float(np.mean(dists <= 28.0))

    safe_box = _compute_safe_box(x1, y1, x2, y2)

    return BubbleStyle(
        safe_box=safe_box,
        bg_color=bg_color,
        text_color=text_color,
        is_dark=is_dark,
        mean_luminance=mean_luminance,
        uniform_bg_ratio=uniform_bg_ratio,
    )


def _has_rectangular_border_contrast(
    rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    bg_vec: np.ndarray,
) -> bool:
    """Check whether at least 3 of 4 box sides show a crisp border contrast."""
    img_h, img_w = rgb.shape[:2]
    w = x2 - x1
    h = y2 - y1
    bw = max(2, min(6, int(round(min(w, h) * 0.06))))

    def _side_has_contrast(inner_strip: np.ndarray, outer_strip: np.ndarray | None) -> bool:
        if inner_strip.size > 0:
            d_in = np.linalg.norm(
                inner_strip.reshape(-1, 3).astype(np.float32) - bg_vec, axis=1
            )
            if float(np.percentile(d_in, 75)) >= 45.0:
                return True
        if outer_strip is not None and outer_strip.size > 0:
            d_out = np.linalg.norm(
                outer_strip.reshape(-1, 3).astype(np.float32) - bg_vec, axis=1
            )
            if float(np.median(d_out)) >= 45.0:
                return True
        return False

    sides = [
        (
            rgb[y1 : min(y2, y1 + bw), x1:x2],
            rgb[max(0, y1 - bw) : y1, x1:x2] if y1 >= 2 else None,
        ),
        (
            rgb[max(y1, y2 - bw) : y2, x1:x2],
            rgb[y2 : min(img_h, y2 + bw), x1:x2] if y2 <= img_h - 2 else None,
        ),
        (
            rgb[y1:y2, x1 : min(x2, x1 + bw)],
            rgb[y1:y2, max(0, x1 - bw) : x1] if x1 >= 2 else None,
        ),
        (
            rgb[y1:y2, max(x1, x2 - bw) : x2],
            rgb[y1:y2, x2 : min(img_w, x2 + bw)] if x2 <= img_w - 2 else None,
        ),
    ]
    contrasting = sum(1 for inner, outer in sides if _side_has_contrast(inner, outer))
    return contrasting >= 3


def is_clean_narration_box(image_rgb: np.ndarray, bbox: Box) -> bool:
    """Return True ONLY for clean uniform-fill narration boxes or plain gutters.

    Strictly rejects floating SFX drawn over gradients, screentones, or textured artwork.
    """
    rgb = _ensure_rgb(image_rgb)
    img_h, img_w = rgb.shape[:2]
    x1, y1, x2, y2 = _clamp_box(bbox, img_w, img_h)
    w = x2 - x1
    h = y2 - y1
    if w < 24 or h < 24:
        return False

    style = analyze_bubble_appearance(rgb, Box(x1=x1, y1=y1, x2=x2, y2=y2))
    if style.uniform_bg_ratio < 0.62:
        return False

    # Parse dominant bg RGB and require low color saturation (neutral paper/ink tones)
    r = int(style.bg_color[1:3], 16)
    g = int(style.bg_color[3:5], 16)
    b = int(style.bg_color[5:7], 16)
    if (max(r, g, b) - min(r, g, b)) > 45:
        return False

    bg_vec = np.array([r, g, b], dtype=np.float32)
    roi = rgb[y1:y2, x1:x2]

    # Check interior perimeter consistency across all 4 inner-margin strips (rejects gradients)
    m_out_x = max(2, int(round(w * 0.06)))
    m_in_x = max(m_out_x + 2, int(round(w * 0.16)))
    m_out_y = max(2, int(round(h * 0.06)))
    m_in_y = max(m_out_y + 2, int(round(h * 0.16)))

    strips = [
        roi[m_out_y:m_in_y, m_out_x : max(m_out_x + 1, w - m_out_x)],
        roi[max(0, h - m_in_y) : max(1, h - m_out_y), m_out_x : max(m_out_x + 1, w - m_out_x)],
        roi[m_out_y : max(m_out_y + 1, h - m_out_y), m_out_x:m_in_x],
        roi[m_out_y : max(m_out_y + 1, h - m_out_y), max(0, w - m_in_x) : max(1, w - m_out_x)],
    ]

    consistent_strips = 0
    ring_samples: list[np.ndarray] = []
    for strip in strips:
        if strip.size == 0:
            continue
        pts = strip.reshape(-1, 3).astype(np.float32)
        ring_samples.append(pts)
        med = np.median(pts, axis=0)
        if float(np.linalg.norm(med - bg_vec)) <= 30.0:
            consistent_strips += 1

    if consistent_strips < 3 or not ring_samples:
        return False

    all_ring = np.concatenate(ring_samples, axis=0)
    ring_dists = np.linalg.norm(all_ring - bg_vec, axis=1)
    if float(np.mean(ring_dists <= 30.0)) < 0.65:
        return False

    # Light white/off-white narration boxes & plain white gutters pass cleanly.
    # Darker or mid-tone shaded panels (common behind SFX) require a crisp rectangular border.
    if style.mean_luminance < 175.0:
        if not _has_rectangular_border_contrast(rgb, x1, y1, x2, y2, bg_vec):
            return False

    return True

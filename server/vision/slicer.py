"""Tall webtoon strip slicer and seam-overlap detection deduplicator."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from server.schemas import Box, RegionType


@dataclass
class ImageSlice:
    """Vertical slice of an image with its Y offset in natural coordinates."""

    y_offset: int
    image: Image.Image


@dataclass
class RawDetection:
    """Raw object detection box with class label, confidence score, OCR box, and region type."""

    box: Box
    label: str
    score: float
    ocr_box: Box | None = None
    region_type: RegionType = "bubble"


def box_area(box: Box) -> int:
    """Return non-negative area in square pixels."""
    return max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)


def box_intersection(a: Box, b: Box) -> int:
    """Return intersection area between two axis-aligned boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def compute_iou(a: Box, b: Box) -> float:
    """Compute Intersection-over-Union between two boxes."""
    inter = box_intersection(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def compute_containment(a: Box, b: Box) -> float:
    """Compute fraction of the smaller box covered by the intersection."""
    inter = box_intersection(a, b)
    if inter <= 0:
        return 0.0
    min_area = min(box_area(a), box_area(b))
    return float(inter / min_area) if min_area > 0 else 0.0


def slice_tall_image(
    image: Image.Image,
    slice_height: int = 1600,
    overlap: int = 250,
) -> list[ImageSlice]:
    """Slice tall vertical webtoon strips into overlapping chunks.

    Images with height <= 2000 are returned unmodified as a single slice.
    Taller images slide with step ``slice_height - overlap`` covering full height.
    """
    if slice_height <= 0:
        raise ValueError("slice_height must be positive")
    if overlap < 0 or overlap >= slice_height:
        raise ValueError("overlap must satisfy 0 <= overlap < slice_height")

    width, height = image.size
    if height <= 2000 or height <= slice_height:
        return [ImageSlice(y_offset=0, image=image)]

    step = slice_height - overlap
    slices: list[ImageSlice] = []
    y = 0
    while y < height:
        y_end = min(y + slice_height, height)
        crop = image.crop((0, y, width, y_end))
        slices.append(ImageSlice(y_offset=y, image=crop))
        if y_end >= height:
            break
        y += step

    return slices


def merge_slice_detections(
    detections: list[RawDetection],
    iou_threshold: float = 0.45,
    containment_threshold: float = 0.80,
) -> list[RawDetection]:
    """Deduplicate overlapping detections across slice seams.

    Runs class-aware NMS combined with high-containment suppression, always
    retaining the highest-scoring detection so seam-spanning bubbles never
    produce duplicate overlays.
    """
    if len(detections) <= 1:
        return list(detections)

    ranked = sorted(
        detections,
        key=lambda d: (d.score, box_area(d.box), -d.box.y1, -d.box.x1),
        reverse=True,
    )

    kept: list[RawDetection] = []
    for candidate in ranked:
        cand_label = candidate.label.strip().lower()
        suppressed = False
        for existing in kept:
            if existing.label.strip().lower() != cand_label:
                continue
            iou = compute_iou(candidate.box, existing.box)
            containment = compute_containment(candidate.box, existing.box)
            if iou >= iou_threshold or containment >= containment_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)

    return kept

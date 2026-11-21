"""Comic bubble detector, SFX filter, narration rescue, and reading-order sorting."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PIL import Image

from server.schemas import Box
from server.vision.geometry import is_clean_narration_box
from server.vision.slicer import (
    RawDetection,
    box_area,
    box_intersection,
    compute_containment,
    compute_iou,
    merge_slice_detections,
    slice_tall_image,
)

_BUBBLE_LABELS = {"bubble", "balloon"}
_TEXT_BUBBLE_LABELS = {"text_bubble"}
_MIN_BOX_SIDE = 20


def filter_dialogue_regions(
    image_rgb: np.ndarray,
    detections: list[RawDetection],
    *,
    include_floating: bool = True,
) -> list[RawDetection]:
    """Filter raw comic detections to dialogue bubbles, narration boxes, and optional floating text/SFX.

    - Drops tiny noise detections (width < 20 or height < 20).
    - Keeps ``bubble`` and ``text_bubble`` detections (tagged ``region_type="bubble"``), merging
      inner ``text_bubble`` into its enclosing ``bubble`` so each speech bubble yields one region.
    - Deduplicates ``text_free`` against already-kept bubbles (``iou >= 0.30`` or
      ``containment >= 0.55``) and against other kept ``text_free`` regions (``iou >= 0.45`` or
      ``containment >= 0.80``).
    - Tags clean uniform ``text_free`` regions as ``region_type="narration"``, and when
      ``include_floating=True``, keeps remaining non-overlapping ``text_free`` regions tagged
      ``region_type="floating"`` (borderless dialogue / onomatopoeia SFX).
    """
    valid: list[RawDetection] = []
    for det in detections:
        w = det.box.x2 - det.box.x1
        h = det.box.y2 - det.box.y1
        if w < _MIN_BOX_SIDE or h < _MIN_BOX_SIDE:
            continue
        valid.append(det)

    bubbles: list[RawDetection] = []
    text_bubbles: list[RawDetection] = []
    free_texts: list[RawDetection] = []

    for det in valid:
        lbl = det.label.strip().lower()
        if lbl in _BUBBLE_LABELS:
            bubbles.append(det)
        elif lbl in _TEXT_BUBBLE_LABELS:
            text_bubbles.append(det)
        else:
            free_texts.append(det)

    # Deduplicate overlapping outer bubbles first (keeping highest score)
    kept_bubbles = [
        RawDetection(
            box=b.box,
            label=b.label,
            score=b.score,
            ocr_box=b.ocr_box,
            region_type="bubble",
        )
        for b in merge_slice_detections(bubbles, iou_threshold=0.50, containment_threshold=0.85)
    ]
    kept: list[RawDetection] = list(kept_bubbles)

    # Associate inner text_bubble regions with enclosing bubbles; keep standalone ones
    for tb in sorted(text_bubbles, key=lambda d: d.score, reverse=True):
        tb_area = max(box_area(tb.box), 1)
        tb_cx = (tb.box.x1 + tb.box.x2) / 2.0
        tb_cy = (tb.box.y1 + tb.box.y2) / 2.0

        matched_idx: int | None = None
        best_overlap = 0.0
        for idx, b in enumerate(kept_bubbles):
            iou = compute_iou(tb.box, b.box)
            tb_inside_ratio = box_intersection(tb.box, b.box) / tb_area
            center_inside = (b.box.x1 <= tb_cx <= b.box.x2) and (b.box.y1 <= tb_cy <= b.box.y2)
            if iou >= 0.30 or tb_inside_ratio >= 0.55 or (center_inside and tb_inside_ratio >= 0.40):
                if tb_inside_ratio > best_overlap:
                    best_overlap = tb_inside_ratio
                    matched_idx = idx

        if matched_idx is not None:
            # Prefer full bubble bounds for clean masking & safe_box, while preserving
            # tight inner text_bubble bounds in ocr_box for cleaner OCR crops.
            existing = kept[matched_idx]
            if existing.ocr_box is None:
                merged_ocr_box = tb.box
            else:
                merged_ocr_box = Box(
                    x1=min(existing.ocr_box.x1, tb.box.x1),
                    y1=min(existing.ocr_box.y1, tb.box.y1),
                    x2=max(existing.ocr_box.x2, tb.box.x2),
                    y2=max(existing.ocr_box.y2, tb.box.y2),
                )
            kept[matched_idx] = RawDetection(
                box=existing.box,
                label=existing.label,
                score=max(existing.score, tb.score),
                ocr_box=merged_ocr_box,
                region_type="bubble",
            )
        else:
            # Standalone text_bubble (e.g., borderless speech bubble)
            duplicate = any(
                compute_iou(tb.box, k.box) >= 0.45 or compute_containment(tb.box, k.box) >= 0.80
                for k in kept
            )
            if not duplicate:
                kept.append(
                    RawDetection(
                        box=tb.box,
                        label=tb.label,
                        score=tb.score,
                        ocr_box=tb.ocr_box,
                        region_type="bubble",
                    )
                )

    kept_bubble_regions = list(kept)
    kept_free: list[RawDetection] = []
    for tf in sorted(free_texts, key=lambda d: d.score, reverse=True):
        overlaps_bubble = any(
            compute_iou(tf.box, k.box) >= 0.30 or compute_containment(tf.box, k.box) >= 0.55
            for k in kept_bubble_regions
        )
        if overlaps_bubble:
            continue
        overlaps_free = any(
            compute_iou(tf.box, f.box) >= 0.45 or compute_containment(tf.box, f.box) >= 0.80
            for f in kept_free
        )
        if overlaps_free:
            continue

        if is_clean_narration_box(image_rgb, tf.box):
            tagged = RawDetection(
                box=tf.box,
                label=tf.label,
                score=tf.score,
                ocr_box=tf.ocr_box,
                region_type="narration",
            )
            kept_free.append(tagged)
            kept.append(tagged)
        elif include_floating:
            tagged = RawDetection(
                box=tf.box,
                label=tf.label,
                score=tf.score,
                ocr_box=tf.ocr_box,
                region_type="floating",
            )
            kept_free.append(tagged)
            kept.append(tagged)

    return kept


def _sort_bands(
    items: list[RawDetection],
    rtl: bool,
) -> list[RawDetection]:
    """Group detections into horizontal reading tiers and sort within each tier."""
    if len(items) <= 1:
        return list(items)

    by_top = sorted(items, key=lambda d: (d.box.y1, (d.box.y1 + d.box.y2) / 2.0))
    bands: list[list[RawDetection]] = []
    band_anchors: list[tuple[float, float]] = []  # (anchor_y1, anchor_y2)

    for item in by_top:
        iy1, iy2 = float(item.box.y1), float(item.box.y2)
        ih = max(iy2 - iy1, 1.0)
        icy = (iy1 + iy2) / 2.0

        placed = False
        if bands:
            ay1, ay2 = band_anchors[-1]
            ah = max(ay2 - ay1, 1.0)
            overlap_y = max(0.0, min(iy2, ay2) - max(iy1, ay1))
            min_h = min(ih, ah)
            # Same horizontal tier if substantial vertical overlap and center within band span
            if overlap_y >= 0.30 * min_h and icy <= ay2:
                bands[-1].append(item)
                # Update anchor using median/capped bottom so one huge box doesn't chain the page
                new_y1 = min(ay1, iy1)
                new_y2 = max(ay2, min(iy2, ay1 + max(ah, ih) * 1.35))
                band_anchors[-1] = (new_y1, new_y2)
                placed = True

        if not placed:
            bands.append([item])
            band_anchors.append((iy1, iy2))

    ordered: list[RawDetection] = []
    for band in bands:
        if rtl:
            band.sort(key=lambda d: (-((d.box.x1 + d.box.x2) / 2.0), d.box.y1))
        else:
            band.sort(key=lambda d: (((d.box.x1 + d.box.x2) / 2.0), d.box.y1))
        ordered.extend(band)

    return ordered


def sort_reading_order(
    detections: list[RawDetection],
    img_w: int,
    img_h: int,
    lang: Literal["ja", "ko"],
) -> list[RawDetection]:
    """Sort dialogue detections in natural manga/manhwa reading order.

    - ``ja``: Right-to-left, top-to-bottom. Wide two-page spreads
      (``img_w / max(img_h, 1) > 1.25``) sort the entire right page before the left page.
    - ``ko``: Top-to-bottom, left-to-right.
    """
    if len(detections) <= 1:
        return list(detections)

    if lang == "ko":
        return _sort_bands(detections, rtl=False)

    # Japanese manga reading order
    aspect = img_w / max(img_h, 1)
    if aspect > 1.25:
        mid_x = img_w / 2.0
        right_half = [d for d in detections if ((d.box.x1 + d.box.x2) / 2.0) >= mid_x]
        left_half = [d for d in detections if ((d.box.x1 + d.box.x2) / 2.0) < mid_x]
        return _sort_bands(right_half, rtl=True) + _sort_bands(left_half, rtl=True)

    return _sort_bands(detections, rtl=True)


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


class BubbleDetector:
    """RT-DETR comic speech-bubble detector with tall-strip slicing and SFX filtering."""

    def __init__(
        self,
        model_id: str = "ogkalu/comic-text-and-bubble-detector",
        device: str = "cuda",
        score_threshold: float = 0.30,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.score_threshold = score_threshold
        self._processor: Any = None
        self._model: Any = None
        self._runtime_device: str = "cpu"
        self._use_fp16: bool = False

    def load(self) -> None:
        """Load AutoImageProcessor and AutoModelForObjectDetection (RT-DETR / RT-DETR-v2) onto CUDA/CPU."""
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        use_cuda = self.device.startswith("cuda") and _has_cuda_vram_headroom(min_free_mb=1200)
        self._runtime_device = self.device if use_cuda else "cpu"

        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        model = AutoModelForObjectDetection.from_pretrained(self.model_id)
        model.eval()
        model.to(self._runtime_device)

        if use_cuda:
            try:
                model.half()
                self._use_fp16 = True
            except Exception:
                model.float()
                self._use_fp16 = False
        else:
            self._use_fp16 = False

        self._model = model

    def _infer_slice(self, slice_img: Image.Image) -> dict[str, Any]:
        """Run model forward pass + post-processing on a single slice with fp32 and CPU OOM fallback."""
        import torch

        def _run_and_postprocess() -> dict[str, Any]:
            inputs: dict[str, Any] | None = None
            outputs: Any = None
            target_sizes: Any = None
            try:
                inputs = self._processor(images=slice_img, return_tensors="pt")
                inputs = {k: v.to(self._runtime_device) for k, v in inputs.items()}
                if self._use_fp16 and "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
                with torch.inference_mode():
                    outputs = self._model(**inputs)
                    target_sizes = torch.tensor(
                        [[slice_img.height, slice_img.width]],
                        device=self._runtime_device,
                    )
                    results = self._processor.post_process_object_detection(
                        outputs,
                        target_sizes=target_sizes,
                        threshold=self.score_threshold,
                    )
                res = results[0]
                return {
                    "scores": res["scores"].detach().cpu().tolist(),
                    "labels": res["labels"].detach().cpu().tolist(),
                    "boxes": res["boxes"].detach().cpu().tolist(),
                }
            finally:
                del inputs, outputs, target_sizes

        try:
            return _run_and_postprocess()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Gracefully fall back to CPU on CUDA OOM
            self._runtime_device = "cpu"
            self._use_fp16 = False
            self._model.float().to("cpu")
            return _run_and_postprocess()
        except (RuntimeError, TypeError):
            if not self._use_fp16:
                raise
            # Graceful fallback to fp32 if any RT-DETR-v2 kernel or post-processor rejects fp16
            if self._runtime_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._model.float()
            self._use_fp16 = False
            return _run_and_postprocess()

    def detect(
        self,
        image: Image.Image,
        *,
        include_floating: bool = True,
    ) -> list[RawDetection]:
        """Detect dialogue bubbles on an image (auto-slicing tall strips and filtering regions)."""
        if self._model is None or self._processor is None:
            self.load()

        rgb_image = image.convert("RGB") if image.mode != "RGB" else image
        img_w, img_h = rgb_image.size
        slices = slice_tall_image(rgb_image)
        id2label = getattr(self._model.config, "id2label", {})

        all_raw: list[RawDetection] = []
        for img_slice in slices:
            result = self._infer_slice(img_slice.image)
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            for score, label_id, raw_box in zip(scores, labels, boxes):
                x1 = max(0, min(img_w - 1, int(round(raw_box[0]))))
                y1 = max(0, min(img_h - 1, int(round(raw_box[1])) + img_slice.y_offset))
                x2 = max(x1 + 1, min(img_w, int(round(raw_box[2]))))
                y2 = max(y1 + 1, min(img_h, int(round(raw_box[3])) + img_slice.y_offset))
                label_name = str(id2label.get(int(label_id), id2label.get(str(int(label_id)), "bubble")))
                all_raw.append(
                    RawDetection(
                        box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                        label=label_name,
                        score=float(score),
                    )
                )

        if self._runtime_device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        merged = merge_slice_detections(all_raw)
        img_rgb = np.asarray(rgb_image)
        return filter_dialogue_regions(img_rgb, merged, include_floating=include_floating)

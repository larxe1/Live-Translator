"""Vision pipeline: tall strip slicing, bubble detection, and style geometry."""

from server.vision.detector import (
    BubbleDetector,
    filter_dialogue_regions,
    sort_reading_order,
)
from server.vision.geometry import (
    BubbleStyle,
    analyze_bubble_appearance,
    is_clean_narration_box,
)
from server.vision.slicer import (
    ImageSlice,
    RawDetection,
    merge_slice_detections,
    slice_tall_image,
)

__all__ = [
    "BubbleDetector",
    "BubbleStyle",
    "ImageSlice",
    "RawDetection",
    "analyze_bubble_appearance",
    "filter_dialogue_regions",
    "is_clean_narration_box",
    "merge_slice_detections",
    "slice_tall_image",
    "sort_reading_order",
]

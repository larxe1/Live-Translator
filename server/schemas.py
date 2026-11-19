"""Shared Pydantic v2 API request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Box(BaseModel):
    """Axis-aligned pixel bounding box in natural image coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int


class BubbleResult(BaseModel):
    """Detected dialogue bubble geometry, appearance, and translated text."""

    id: int
    bbox: Box
    safe_box: Box
    bg_color: str
    text_color: str
    is_dark: bool
    original: str
    translated: str


class TranslateRequest(BaseModel):
    """Request payload for translating a single manga or webtoon page."""

    image_base64: str
    lang: Literal["auto", "ja", "ko"] = "auto"
    chapter_id: str = "default"
    priority: int = 0
    glossary: dict[str, str] = Field(default_factory=dict)
    request_id: str | None = None


class TranslateResponse(BaseModel):
    """Translated page response containing ordered dialogue bubbles."""

    lang: Literal["ja", "ko"]
    width: int
    height: int
    elapsed_ms: int
    cached: bool = False
    bubbles: list[BubbleResult]


class HealthResponse(BaseModel):
    """Local GPU server readiness and VRAM telemetry."""

    status: str
    ready: bool
    gpu_name: str
    vram_used_mb: int
    vram_total_mb: int
    llm_model: str
    llm_ready: bool
    loading_stage: str = ""


class GlossaryUpdateRequest(BaseModel):
    """Request payload to update pinned character/term glossary entries."""

    chapter_id: str = "default"
    entries: dict[str, str]


class MemoryClearRequest(BaseModel):
    """Request payload to clear rolling chapter translation context."""

    chapter_id: str = "default"

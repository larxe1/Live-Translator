"""Shared Pydantic v2 API request and response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RegionType = Literal["bubble", "narration", "floating"]


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
    region_type: RegionType = "bubble"


class TranslateRequest(BaseModel):
    """Request payload for translating a single manga or webtoon page."""

    image_base64: str
    lang: Literal["auto", "ja", "ko"] = "auto"
    chapter_id: str = "default"
    priority: int = 0
    glossary: dict[str, str] = Field(default_factory=dict)
    request_id: str | None = None
    include_floating: bool = False
    series_id: str | None = None
    translation_mode: str | None = None


class TranslateResponse(BaseModel):
    """Translated page response containing ordered dialogue bubbles."""

    lang: Literal["ja", "ko"]
    width: int
    height: int
    elapsed_ms: int
    cached: bool = False
    provider: str = ""
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
    extension_version: str = "1.0.0"
    extension_build_id: str = ""
    has_remote: bool = False
    update_available: bool = False
    translation_mode: str = "hybrid"
    gemini_configured: bool = False
    gemini_key_hint: str = ""
    active_local_model: str = ""
    available_local_models: dict[str, bool] = Field(default_factory=dict)
    gpu_guard: dict[str, Any] = Field(default_factory=dict)


class SettingsUpdateRequest(BaseModel):
    """Request payload to update translation mode or Gemini API key safely."""

    translation_mode: str | None = None
    gemini_api_key: str | None = None
    clear_gemini_api_key: bool = False


class SettingsUpdateResponse(BaseModel):
    """Sanitized settings response (never returns raw API keys)."""

    ok: bool = True
    translation_mode: str = "hybrid"
    gemini_configured: bool = False
    gemini_key_hint: str = ""


class UpdateStatusResponse(BaseModel):
    """Current extension build ID and git repository update status."""

    extension_version: str
    extension_build_id: str
    has_remote: bool
    remote_url: str
    branch: str
    current_commit: str
    remote_commit: str | None = None
    commits_behind: int = 0
    update_available: bool = False
    auto_update_enabled: bool = True
    working_tree_clean: bool = True
    last_checked: float | None = None
    last_error: str | None = None
    updating: bool = False
    restart_required: bool = False


class UpdateCheckRequest(BaseModel):
    """Optional request payload for checking and optionally applying updates."""

    auto_apply: bool = False


class UpdateApplyResponse(BaseModel):
    """Result of applying a fast-forward repository update."""

    ok: bool
    updated: bool
    message: str
    extension_changed: bool
    server_changed: bool
    restart_scheduled: bool
    status: UpdateStatusResponse


class GlossaryUpdateRequest(BaseModel):
    """Request payload to update pinned character/term glossary entries."""

    chapter_id: str = "default"
    series_id: str | None = None
    entries: dict[str, str]


class GlossaryLookupRequest(BaseModel):
    """Request payload to look up character names from AniList, Wikipedia, or Fandom."""

    query: str
    source: Literal["auto", "anilist", "wikipedia", "fandom"] = "auto"


class GlossaryLookupResponse(BaseModel):
    """Response payload containing imported character glossary entries."""

    title: str
    source: str
    url: str
    entries: dict[str, str] = Field(default_factory=dict)
    count: int
    warnings: list[str] = Field(default_factory=list)


class MemoryClearRequest(BaseModel):
    """Request payload to clear rolling chapter translation context."""

    chapter_id: str = "default"
    series_id: str | None = None

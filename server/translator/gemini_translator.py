"""Official GA Gemini Vision Interactions API client with smart strip cropping and circuit-breaking."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import logging
import os
import re
import time
from typing import Any, Literal

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1/interactions"
DEFAULT_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-4-hydrogen",
)
DEFAULT_MODEL_BREAKER_S = 120.0
DEFAULT_GEMINI_TIMEOUT_S = 25.0

_SECRET_PATTERN_RE = re.compile(r"(?:AQ\.[A-Za-z0-9_-]{10,}|AIza[A-Za-z0-9_-]{15,})")
_SAFETY_KEYWORDS: tuple[str, ...] = (
    "safety",
    "prohibited",
    "blocked",
    "content_filter",
    "policy",
    "harm_category",
    "recitation",
    "sexually",
    "nsfw",
    "moderation",
    "inappropriate",
    "explicit",
)
_RATE_LIMIT_KEYWORDS: tuple[str, ...] = (
    "resource_exhausted",
    "rate_limit",
    "rate limit",
    "quota",
    "too many requests",
)


class GeminiBlockedError(RuntimeError):
    """Raised when Gemini refuses or blocks an explicit/sensitive panel or returns 0 valid bubbles."""


class GeminiRateLimitError(RuntimeError):
    """Raised when all candidate Gemini cloud models return HTTP 429 / resource_exhausted."""


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini cloud models are unavailable (503/500/network/timeout/missing key)."""


def _get_gemini_api_key(cfg: Any = None) -> str:
    """Dynamically fetch GEMINI_API_KEY via server.config.get_gemini_api_key() or environment."""
    try:
        from server import config as config_mod

        getter = getattr(config_mod, "get_gemini_api_key", None)
        if callable(getter):
            project_root = getattr(cfg, "project_root", None)
            if project_root is not None:
                return str(getter(env_path=project_root / ".env") or "").strip()
            return str(getter() or "").strip()
    except Exception:
        pass
    return str(os.environ.get("GEMINI_API_KEY", "") or "").strip()


def _redact_secret(text: str, secret: str = "") -> str:
    """Ensure API keys never leak into exception strings or logs."""
    out = str(text or "")
    if secret:
        out = out.replace(secret, "[REDACTED]")
    out = _SECRET_PATTERN_RE.sub("[REDACTED]", out)
    return out


def _extract_box_y_bounds(box: Any) -> tuple[int, int] | None:
    """Extract (y1, y2) bounds from Box, dict, or 4-element sequence."""
    if box is None:
        return None
    if hasattr(box, "y1") and hasattr(box, "y2"):
        return int(box.y1), int(box.y2)
    if isinstance(box, dict) and "y1" in box and "y2" in box:
        return int(box["y1"]), int(box["y2"])
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return int(box[1]), int(box[3])
    return None


def prepare_vision_image(
    image: Image.Image,
    bubble_boxes: dict[int, Any] | None = None,
    chunk_ids: list[int] | set[int] | None = None,
    *,
    max_width: int = 1024,
    max_height: int = 2048,
    jpeg_quality: int = 78,
    tall_threshold: int = 2400,
    band_pad: int = 200,
) -> str:
    """Crop tall webtoon strips to the active chunk band, resize cleanly, and base64-encode JPEG."""
    img = image.convert("RGB") if image.mode != "RGB" else image.copy()

    if img.height > tall_threshold and bubble_boxes and chunk_ids:
        y1s: list[int] = []
        y2s: list[int] = []
        for cid in chunk_ids:
            bounds = _extract_box_y_bounds(bubble_boxes.get(cid))
            if bounds is not None:
                y1s.append(bounds[0])
                y2s.append(bounds[1])
        if y1s and y2s:
            min_y = max(0, min(y1s) - band_pad)
            max_y = min(img.height, max(y2s) + band_pad)
            if max_y > min_y:
                img = img.crop((0, min_y, img.width, max_y))

    scale = min(
        1.0,
        max_width / float(max(img.width, 1)),
        max_height / float(max(img.height, 1)),
    )
    if scale < 1.0:
        new_w = max(1, int(round(img.width * scale)))
        new_h = max(1, int(round(img.height * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_steps_text(data: dict[str, Any]) -> str:
    """Extract concatenated model_output text parts from GA v1 /v1/interactions response."""
    steps = data.get("steps")
    if not isinstance(steps, list):
        return ""
    parts: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("type") != "model_output":
            continue
        content = step.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                txt = part.get("text")
                if txt:
                    parts.append(str(txt))
    return "\n".join(parts).strip()


def _extract_error_text(resp: httpx.Response) -> str:
    """Extract human-readable error message from a JSON or plain-text HTTP error response."""
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                code = str(err.get("code") or err.get("status") or "")
                msg = str(err.get("message") or "")
                return f"{code} {msg}".strip()
            if err:
                return str(err)
            return json.dumps(body, ensure_ascii=False)
    except Exception:
        pass
    return (resp.text or "").strip()


def _is_safety_error(text: str) -> bool:
    low = (text or "").lower()
    return any(kw in low for kw in _SAFETY_KEYWORDS)


def _is_rate_limit_error(status_code: int, text: str) -> bool:
    if status_code == 429:
        return True
    low = (text or "").lower()
    return any(kw in low for kw in _RATE_LIMIT_KEYWORDS)


def _compute_max_output_tokens(chunk_len: int) -> int:
    return min(1024, max(256, int(chunk_len) * 64))


class GeminiVisionTranslator:
    """GA Gemini v1 Interactions API vision-context translator with model failover."""

    def __init__(
        self,
        cfg: Any = None,
        *,
        api_key: str | None = None,
        timeout_s: float | None = None,
        model_breaker_s: float = DEFAULT_MODEL_BREAKER_S,
    ) -> None:
        self.cfg = cfg
        self._api_key_override = api_key
        self._timeout_override = timeout_s
        self.model_breaker_s = max(0.0, float(model_breaker_s))
        self._model_unavailable_until: dict[str, float] = {}
        self.last_model: str = ""

    def get_api_key(self) -> str:
        """Read Gemini API key dynamically so live /settings updates take effect immediately."""
        if self._api_key_override is not None:
            return self._api_key_override.strip()
        return _get_gemini_api_key(self.cfg)

    def is_configured(self) -> bool:
        """Return True when a non-empty Gemini API key is available."""
        return bool(self.get_api_key())

    def _get_models(self) -> list[str]:
        raw = getattr(self.cfg, "gemini_models", None)
        if isinstance(raw, (list, tuple)):
            cleaned = [str(m).strip() for m in raw if str(m).strip()]
            if cleaned:
                return cleaned
        return list(DEFAULT_GEMINI_MODELS)

    def _get_timeout_s(self) -> float:
        if self._timeout_override is not None:
            return float(self._timeout_override)
        return float(getattr(self.cfg, "gemini_timeout_s", DEFAULT_GEMINI_TIMEOUT_S))

    def _use_vision_default(self) -> bool:
        return bool(getattr(self.cfg, "gemini_use_vision", True))

    def _mark_model_unavailable(self, model: str) -> None:
        if self.model_breaker_s > 0:
            self._model_unavailable_until[model] = time.monotonic() + self.model_breaker_s

    def _active_models(self) -> list[str]:
        now = time.monotonic()
        return [
            m
            for m in self._get_models()
            if now >= self._model_unavailable_until.get(m, 0.0)
        ]

    @staticmethod
    def build_system_instruction(lang: Literal["ja", "ko"]) -> str:
        """Build a concise, high-craft manga/manhwa localization prompt for Gemini Vision."""
        if lang == "ja":
            source_label = "Japanese manga"
            lang_rules = (
                "- Supply natural omitted English subjects/objects from panel context and dialogue flow "
                "(e.g., '好き' -> 'I like you.'/'I love you.', '俺も' -> 'Me too.').\n"
                "- Translate idioms, slang, and honorific nuance naturally ('ｗ'/'ｗｗ' -> 'haha'/'lol', "
                "NEVER bare 'w'; adapt or preserve '-chan'/'-kun'/'-senpai'/'-sensei' naturally).\n"
                "- Zero Japanese Romaji (NEVER 'Chotto matte', 'Koko ja yada') and zero CJK/kana characters."
            )
        else:
            source_label = "Korean manhwa/webtoon"
            lang_rules = (
                "- Supply natural omitted English subjects/objects from panel context and dialogue flow "
                "(e.g., '좋아해' -> 'I like you.', '나도' -> 'Me too.').\n"
                "- Translate Korean webtoon slang, laughter, and honorific nuance naturally "
                "('ㅋㅋ'/'ㅎㅎ' -> 'haha'/'lol', '오빠'/'형'/'언니'/'누나'/'선배' adapted naturally for English dialogue).\n"
                "- Zero Korean romanization (NEVER 'Jamkkanman', 'Gwaenchana', 'Jebal', 'Hajima') and zero Hangul/CJK characters."
            )

        return (
            f"You are an expert {source_label}-to-English localization translator.\n"
            "Rules:\n"
            "- Use the attached page image ONLY as visual context for speaker gender, facial expressions, tone, and panel flow.\n"
            "- Translate ONLY the exact numbered OCR bubble keys in the JSON below. Return the EXACT same string numeric keys — do NOT invent extra keys, merge keys, or omit keys.\n"
            "- Produce natural, fluent, idiomatic English comic lettering (concise, natural dialogue—not stiff literal translations).\n"
            f"{lang_rules}\n"
            "- Keep ellipses ('…'/'...') as '...' — NEVER spell them out as 'Dot dot'. Preserve stutters, screams, and sighs naturally.\n"
            "- Keep character names consistent with the glossary, established names, and recent dialogue context.\n"
            "- Entries tagged [floating] may be borderless speech/thoughts or SFX: translate speech naturally and SFX concisely (e.g., '*THUMP*', '*RUSTLE*').\n"
            "- Return ONLY a single valid JSON object mapping each string bubble ID to its English translation, with zero markdown or commentary."
        )

    @staticmethod
    def build_user_prompt(
        bubbles: list[tuple[int, str]],
        *,
        glossary: dict[str, str] | None = None,
        context: list[tuple[str, str]] | str = "",
        known_names: list[str] | None = None,
        bubble_types: dict[int, str] | None = None,
        max_context_pairs: int = 6,
    ) -> str:
        from server.translator.translator import DialogueTranslator, _normalize_ocr_for_llm

        sections = DialogueTranslator._format_context_sections(
            glossary or {},
            context,
            known_names=known_names,
            max_context_pairs=max_context_pairs,
        )
        if bubble_types:
            tagged_notes = [
                f'"{bid}": [{bubble_types[bid]}]'
                for bid, _ in bubbles
                if bubble_types.get(bid) in ("floating", "narration")
            ]
            if tagged_notes:
                sections.append(
                    "Region Type Hints (do NOT include bracket tags in your output): "
                    + ", ".join(tagged_notes)
                )
        bubble_map = {str(bid): _normalize_ocr_for_llm(text) for bid, text in bubbles}
        sections.append(
            "Translate ONLY the exact numbered OCR bubble keys below into natural, idiomatic English "
            "(strictly English ONLY; zero Romaji/romanization, zero CJK/Hangul). "
            "Return ONLY a JSON object with these exact string keys:\n"
            + json.dumps(bubble_map, ensure_ascii=False)
        )
        return "\n\n".join(sections)

    async def _post_interaction(
        self,
        payload: dict[str, Any],
        api_key: str,
        timeout_s: float,
        cancel_event: asyncio.Event | None = None,
    ) -> httpx.Response | None:
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            if cancel_event is not None:
                req_task = asyncio.create_task(
                    client.post(GEMINI_INTERACTIONS_URL, headers=headers, json=payload)
                )
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    {req_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                if cancel_task in done:
                    return None
                return req_task.result()
            return await client.post(GEMINI_INTERACTIONS_URL, headers=headers, json=payload)

    async def translate_chunk(
        self,
        bubbles: list[tuple[int, str]],
        lang: Literal["ja", "ko"],
        *,
        glossary: dict[str, str] | None = None,
        context: list[tuple[str, str]] | str = "",
        known_names: list[str] | None = None,
        bubble_types: dict[int, str] | None = None,
        max_context_pairs: int = 6,
        image: Image.Image | None = None,
        bubble_boxes: dict[int, Any] | None = None,
        use_vision: bool | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> tuple[dict[int, str], str]:
        """Translate a chunk of bubbles via GA Gemini v1 Interactions API.

        Returns ``(valid_translations, model_name)`` on success, or ``({}, "")`` if cancelled.
        Raises ``GeminiBlockedError``, ``GeminiRateLimitError``, or ``GeminiUnavailableError``.
        """
        if not bubbles or (cancel_event is not None and cancel_event.is_set()):
            return {}, ""

        api_key = self.get_api_key()
        if not api_key:
            raise GeminiUnavailableError("Gemini API key is not configured")

        from server.translator.translator import (
            _parse_translation_json,
            _sanitize_translation_candidate,
            is_refusal,
        )

        candidate_models = self._active_models()
        if not candidate_models:
            raise GeminiUnavailableError("All Gemini models are temporarily in 503 circuit-breaker cooldown")

        include_image = (
            (self._use_vision_default() if use_vision is None else bool(use_vision))
            and image is not None
        )
        content_parts: list[dict[str, Any]] = []
        if include_image and image is not None:
            chunk_ids = [bid for bid, _ in bubbles]
            b64_jpeg = await asyncio.to_thread(
                prepare_vision_image,
                image,
                bubble_boxes,
                chunk_ids,
            )
            content_parts.append(
                {
                    "type": "image",
                    "data": b64_jpeg,
                    "mime_type": "image/jpeg",
                }
            )

        user_prompt = self.build_user_prompt(
            bubbles,
            glossary=glossary,
            context=context,
            known_names=known_names,
            bubble_types=bubble_types,
            max_context_pairs=max_context_pairs,
        )
        content_parts.append({"type": "text", "text": user_prompt})

        system_instruction = self.build_system_instruction(lang)
        max_output_tokens = _compute_max_output_tokens(len(bubbles))
        timeout_s = self._get_timeout_s()

        rate_limited_count = 0
        unavailable_count = 0
        last_error = ""

        for model in candidate_models:
            if cancel_event is not None and cancel_event.is_set():
                return {}, ""

            payload: dict[str, Any] = {
                "model": model,
                "store": False,
                "system_instruction": system_instruction,
                "generation_config": {
                    "temperature": 0.3,
                    "max_output_tokens": max_output_tokens,
                    "thinking_level": "minimal",
                },
                "input": [
                    {
                        "type": "user_input",
                        "content": content_parts,
                    }
                ],
            }

            try:
                resp = await self._post_interaction(
                    payload,
                    api_key=api_key,
                    timeout_s=timeout_s,
                    cancel_event=cancel_event,
                )
            except (httpx.TimeoutException, httpx.HTTPError, OSError) as exc:
                safe_err = _redact_secret(str(exc), api_key)
                logger.warning("Gemini model %s network/timeout error: %s", model, safe_err)
                self._mark_model_unavailable(model)
                unavailable_count += 1
                last_error = safe_err
                continue

            if resp is None or (cancel_event is not None and cancel_event.is_set()):
                return {}, ""

            status_code = resp.status_code
            if status_code == 200:
                try:
                    data = resp.json()
                except Exception as exc:
                    safe_err = _redact_secret(str(exc), api_key)
                    self._mark_model_unavailable(model)
                    unavailable_count += 1
                    last_error = f"Invalid JSON from {model}: {safe_err}"
                    continue

                if isinstance(data, dict) and data.get("error"):
                    err_str = _redact_secret(str(data["error"]), api_key)
                    if _is_rate_limit_error(200, err_str):
                        logger.warning("Gemini model %s returned resource_exhausted in body", model)
                        rate_limited_count += 1
                        last_error = err_str
                        continue
                    if _is_safety_error(err_str):
                        raise GeminiBlockedError(f"Gemini blocked interaction ({model}): {err_str}")
                    self._mark_model_unavailable(model)
                    unavailable_count += 1
                    last_error = err_str
                    continue

                extracted_text = _extract_steps_text(data if isinstance(data, dict) else {})
                interaction_status = str(data.get("status", "") if isinstance(data, dict) else "").lower()

                if interaction_status in ("failed", "incomplete") and not extracted_text:
                    raise GeminiBlockedError(
                        f"Gemini interaction {interaction_status} with empty output ({model})"
                    )
                if not extracted_text:
                    raise GeminiBlockedError(f"Gemini returned empty output ({model})")

                parsed = _parse_translation_json(extracted_text)
                valid: dict[int, str] = {}
                for bid, orig_text in bubbles:
                    cand = parsed.get(bid, "").strip()
                    if not cand:
                        continue
                    sanitized, ok = _sanitize_translation_candidate(orig_text, cand)
                    if ok:
                        valid[bid] = sanitized

                if not valid:
                    if is_refusal(extracted_text):
                        raise GeminiBlockedError(f"Gemini refused translation ({model})")
                    raise GeminiBlockedError(
                        f"Gemini returned 0 valid translated bubbles ({model})"
                    )

                self.last_model = model
                return valid, model

            err_text = _redact_secret(_extract_error_text(resp), api_key)
            if _is_rate_limit_error(status_code, err_text):
                logger.warning("Gemini model %s rate limited (HTTP %d)", model, status_code)
                rate_limited_count += 1
                last_error = f"HTTP {status_code}: {err_text}"
                continue

            if status_code == 400:
                if _is_safety_error(err_text):
                    raise GeminiBlockedError(
                        f"Gemini blocked request on safety policy (HTTP 400, {model}): {err_text}"
                    )
                logger.warning("Gemini model %s returned HTTP 400: %s", model, err_text)
                self._mark_model_unavailable(model)
                unavailable_count += 1
                last_error = f"HTTP 400: {err_text}"
                continue

            if status_code in (401, 403):
                raise GeminiUnavailableError(
                    f"Gemini API authentication/permission error (HTTP {status_code})"
                )

            if status_code in (404, 500, 502, 503, 504) or status_code >= 500:
                logger.warning("Gemini model %s unavailable (HTTP %d): %s", model, status_code, err_text)
                self._mark_model_unavailable(model)
                unavailable_count += 1
                last_error = f"HTTP {status_code}: {err_text}"
                continue

            logger.warning("Gemini model %s unexpected HTTP %d: %s", model, status_code, err_text)
            self._mark_model_unavailable(model)
            unavailable_count += 1
            last_error = f"HTTP {status_code}: {err_text}"

        if rate_limited_count > 0 and unavailable_count == 0:
            raise GeminiRateLimitError(
                f"All Gemini models rate-limited ({rate_limited_count} model(s)): {last_error}"
            )
        raise GeminiUnavailableError(
            f"All Gemini models unavailable ({unavailable_count} unavailable, {rate_limited_count} rate-limited): {last_error}"
        )

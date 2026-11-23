"""Fast chunked uncensored manga/manhwa page translator with Romaji/CJK guard and cancellation."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

import httpx

from server.translator.memory import (
    ChapterMemoryStore,
    contains_non_english_cjk,
    extract_character_names,
    looks_like_romaji,
)

logger = logging.getLogger(__name__)

LlmCaller = Callable[[list[dict[str, str]]], Awaitable[str]]
LlmStreamCaller = Callable[[list[dict[str, str]]], AsyncIterator[str]]
BubbleCallback = Callable[[int, str, bool], Awaitable[None]]

# Targeted refusal patterns: catch AI safety/policy boilerplate while never false-positiving
# on genuine manga dialogue like "Get the fuck off me!" or "I can't move!" or "I cannot forgive you!".
_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bi\s+(?:cannot|can't|am unable to|will not|won't)\s+"
        r"(?:translate|fulfill|assist|comply|provide|generate|process|help\s+with|output|reproduce)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi(?:'m|\s+am)\s+(?:sorry|apologize|afraid)[,.]?\s+(?:but\s+)?i\s+"
        r"(?:cannot|can't|won't|will not|am unable|must decline)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bas an ai(?:\s+language\s+model)?\b", re.IGNORECASE),
    re.compile(r"\bas a language model\b", re.IGNORECASE),
    re.compile(
        r"\b(?:explicit|violent|adult|nsfw|inappropriate|harmful|offensive|sexually explicit)\s+content\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:violates?|against)\s+(?:my|our|the)?\s*(?:content\s+policy|safety\s+guidelines|usage\s+policies|programming|ethical\s+guidelines)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bunable to translate\b", re.IGNORECASE),
    re.compile(r"\bmust decline\b", re.IGNORECASE),
    re.compile(r"\bcontent policy\b", re.IGNORECASE),
    re.compile(r"\bsafety guidelines\b", re.IGNORECASE),
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_KV_FALLBACK_RE = re.compile(r'"(\d+)"\s*:\s*"((?:\\.|[^"\\])*)"')
_ELLIPSIS_CLUSTER_RE = re.compile(r"(?:[…‥]+|[．・。]{2,}|\.{2,})+")
_TRAILING_EMPTY_BRACES_RE = re.compile(r"(?:\s*\{\s*\})+\s*$")
_ORIG_ELLIPSIS_RE = re.compile(r"(?:[…‥]|[．・。]{2,}|\.{2,})")
_DOT_DOT_HALLUCINATION_RE = re.compile(r"\bdot\s+dot(?:\s+dot)?\b", re.IGNORECASE)
_FULLWIDTH_PUNCT_TABLE = str.maketrans(
    {
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "～": "~",
        "：": ":",
        "；": ";",
        "，": ",",
    }
)

DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9
DEFAULT_REPEAT_PENALTY = 1.05


def _normalize_ocr_for_llm(text: str) -> str:
    """Normalize fullwidth/repeated ellipsis clusters and fullwidth punctuation before LLM translation."""
    if not text:
        return ""
    normalized = _ELLIPSIS_CLUSTER_RE.sub("...", text)
    normalized = normalized.translate(_FULLWIDTH_PUNCT_TABLE)
    return normalized.strip()


def _clean_candidate(candidate: str) -> str:
    """Strip whitespace and stray trailing empty-brace artifacts like 'Nnn...{}'."""
    cleaned = (candidate or "").strip()
    if not cleaned:
        return ""
    cleaned = _TRAILING_EMPTY_BRACES_RE.sub("", cleaned).strip()
    return cleaned


def is_refusal(text: str) -> bool:
    """Return True if text contains AI safety/content-policy refusal boilerplate."""
    if not text or not text.strip():
        return False
    return any(pattern.search(text) is not None for pattern in _REFUSAL_PATTERNS)


def is_valid_english_translation(orig: str, candidate: str) -> bool:
    """Return True if candidate is a non-empty, non-refusal, non-Romaji English translation without CJK scripts."""
    cleaned = _clean_candidate(candidate)
    if not cleaned:
        return False
    if is_refusal(cleaned):
        return False
    if contains_non_english_cjk(cleaned):
        return False
    if contains_non_english_cjk(orig) and cleaned == orig.strip():
        return False
    if looks_like_romaji(orig, cleaned):
        return False
    if _ORIG_ELLIPSIS_RE.search(orig or "") and _DOT_DOT_HALLUCINATION_RE.search(cleaned):
        return False
    return True


def _sanitize_translation_candidate(orig: str, candidate: str) -> tuple[str, bool]:
    """Clean stray artifacts and strictly validate candidate as English without auto-romanizing."""
    cleaned = _clean_candidate(candidate)
    if is_valid_english_translation(orig, cleaned):
        return (cleaned, True)
    return (cleaned, False)


def _chunk_max_tokens(chunk_len: int) -> int:
    """Compute tight per-chunk token cap so small chunks never burn excessive generation budget."""
    return min(640, max(160, int(chunk_len) * 56))


def _extract_int_key(raw_key: Any) -> int | None:
    if isinstance(raw_key, int):
        return raw_key
    s = str(raw_key).strip()
    if s.isdigit():
        return int(s)
    match = re.search(r"(\d+)", s)
    return int(match.group(1)) if match else None


def _parse_translation_json(raw: str) -> dict[int, str]:
    """Robustly extract {bubble_id: translated_text} from LLM JSON output."""
    if not raw or not raw.strip():
        return {}

    text = raw.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    parsed_obj: Any = None
    try:
        parsed_obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed_obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                parsed_obj = None

    result: dict[int, str] = {}
    if isinstance(parsed_obj, dict):
        for k, v in parsed_obj.items():
            idx = _extract_int_key(k)
            if idx is not None and v is not None:
                result[idx] = str(v).strip()
        return result

    # Fallback regex recovery for slightly truncated JSON objects
    for match in _KV_FALLBACK_RE.finditer(text):
        idx = int(match.group(1))
        raw_val = match.group(2)
        try:
            decoded = json.loads(f'"{raw_val}"')
        except json.JSONDecodeError:
            decoded = raw_val
        result[idx] = str(decoded).strip()

    return result


class DialogueTranslator:
    """Chunked streaming manga/manhwa translator backed by local llama-server."""

    def __init__(
        self,
        llama_base_url: str,
        memory: ChapterMemoryStore,
        llm_caller: LlmCaller | None = None,
        timeout_s: float = 45.0,
        llm_stream_caller: LlmStreamCaller | None = None,
        cfg: Any = None,
    ) -> None:
        self.llama_base_url = llama_base_url.rstrip("/")
        self.memory = memory
        self._llm_caller = llm_caller
        self._llm_stream_caller = llm_stream_caller
        self.timeout_s = timeout_s
        self.cfg = cfg

    def _is_qwen3_model(self) -> bool:
        """Return True only when the configured LLM filename/model targets Qwen3."""
        if self.cfg is None:
            return False
        filename = getattr(self.cfg, "llm_filename", None)
        if filename:
            return "qwen3" in str(filename).lower()
        model_path = getattr(self.cfg, "llm_model_path", None)
        if model_path:
            return "qwen3" in str(model_path).lower()
        pattern = getattr(self.cfg, "llm_filename_pattern", None)
        if pattern:
            return "qwen3" in str(pattern).lower()
        return False

    async def _call_llama(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        max_tokens: int = 640,
        json_mode: bool = True,
        cancel_event: asyncio.Event | None = None,
    ) -> str:
        """Send chat completion request to local llama-server OpenAI-compatible API."""
        if cancel_event is not None and cancel_event.is_set():
            return ""
        payload: dict[str, Any] = {
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "max_tokens": max_tokens,
        }
        if self._is_qwen3_model():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            if cancel_event is not None:
                req_task = asyncio.create_task(
                    client.post(f"{self.llama_base_url}/v1/chat/completions", json=payload)
                )
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    {req_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for p in pending:
                    p.cancel()
                    try:
                        await p
                    except (asyncio.CancelledError, Exception):
                        pass
                if cancel_task in done:
                    return ""
                resp = req_task.result()
            else:
                resp = await client.post(
                    f"{self.llama_base_url}/v1/chat/completions",
                    json=payload,
                )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])

    async def _call_llama_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        max_tokens: int = 640,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion content deltas from local llama-server SSE API."""
        if cancel_event is not None and cancel_event.is_set():
            return
        payload: dict[str, Any] = {
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "stream": True,
        }
        if self._is_qwen3_model():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            async with client.stream(
                "POST",
                f"{self.llama_base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                line_iter = resp.aiter_lines().__aiter__()
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    if cancel_event is not None:
                        next_task = asyncio.create_task(line_iter.__anext__())
                        cancel_task = asyncio.create_task(cancel_event.wait())
                        done, pending = await asyncio.wait(
                            {next_task, cancel_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for p in pending:
                            p.cancel()
                            try:
                                await p
                            except (asyncio.CancelledError, StopAsyncIteration):
                                pass
                        if cancel_task in done:
                            break
                        try:
                            line = next_task.result()
                        except StopAsyncIteration:
                            break
                    else:
                        try:
                            line = await line_iter.__anext__()
                        except StopAsyncIteration:
                            break

                    stripped = line.strip()
                    if not stripped or not stripped.startswith("data:"):
                        continue
                    data_str = stripped[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield str(content)

    async def _invoke_llm(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        max_tokens: int = 640,
        json_mode: bool = True,
        cancel_event: asyncio.Event | None = None,
    ) -> str:
        if cancel_event is not None and cancel_event.is_set():
            return ""
        if self._llm_caller is not None:
            return await self._llm_caller(messages)
        return await self._call_llama(
            messages,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
            json_mode=json_mode,
            cancel_event=cancel_event,
        )

    @staticmethod
    def _format_context_sections(
        glossary: dict[str, str],
        context: list[tuple[str, str]] | str,
        known_names: list[str] | None = None,
        max_context_pairs: int = 6,
    ) -> list[str]:
        sections: list[str] = []
        if glossary:
            glossary_lines = "\n".join(f"- {k} = {v}" for k, v in glossary.items())
            sections.append(f"Pinned Character/Term Glossary (MUST follow exactly):\n{glossary_lines}")

        if known_names:
            glossary_vals_lower = {v.lower() for v in glossary.values()}
            extra_names = [n for n in known_names if n.lower() not in glossary_vals_lower]
            if extra_names:
                sections.append(
                    "Established Character Names (keep spelling consistent across kanji/kana): "
                    + ", ".join(extra_names)
                )

        formatted_context = ""
        if isinstance(context, str):
            formatted_context = context.strip()
        elif context:
            window = context[-max_context_pairs:] if max_context_pairs > 0 else context
            formatted_context = "\n".join(
                f"- {_normalize_ocr_for_llm(orig)} -> {trans}" for orig, trans in window
            )

        if formatted_context:
            sections.append(
                f"Recent Dialogue Context (keep subjects, tone, and names consistent):\n{formatted_context}"
            )
        return sections

    def _build_primary_messages(
        self,
        bubbles: list[tuple[int, str]],
        lang: Literal["ja", "ko"],
        glossary: dict[str, str],
        context: list[tuple[str, str]] | str,
        bubble_types: dict[int, str] | None = None,
        known_names: list[str] | None = None,
        max_context_pairs: int = 6,
    ) -> list[dict[str, str]]:
        # Keep system prompt completely static per language so llama.cpp reuses KV-cache prefix across chunks.
        source_label = "Japanese manga" if lang == "ja" else "Korean manhwa"
        if lang == "ja":
            subject_rule = (
                "Supply natural omitted English subjects and objects from dialogue flow "
                "(e.g., '好き'/'好きです' -> 'I like you.'/'I love you.', '俺も' -> 'Me too.', "
                "'片思いしていた' -> 'I had a one-sided crush.')."
            )
            idiom_rule = (
                "Translate idioms and colloquialisms into natural English "
                "(e.g., 'こういうの久々だな…' -> \"It's been a while since we did this...\", "
                "'三軍' -> 'bottom tier', 'ｗ'/'ｗｗ' -> 'lol'/'haha' or natural tone, NEVER bare 'w')."
            )
            sigh_example = "'ちょっ ちょっと待って' -> 'W-Wait a second!', 'んん…' -> 'Mmh...'"
            name_rule = (
                "Keep character names consistent across kanji and kana readings using the glossary "
                "and recent dialogue (e.g., if 'Masaki' is established, keep 'Masaki' for '将暉')."
            )
            anti_romaji = (
                'Translate ALL spoken words into English—never Japanese Romaji '
                '(e.g., "Chotto matte", "Koko ja yada") or CJK characters; '
                "keep only proper character names in Latin letters."
            )
            examples = (
                'Example input: {"0": "夢幻団", "1": "...すき", "2": "俺も", '
                '"3": "ちょっ ちょっと待って", "4": "こういうの久々だな...", '
                '"5": "なわけねーじゃんｗ", "6": "ぎゃああああっ!"}\n'
                'Example output: {"0": "Phantom Troupe", "1": "...I love you.", "2": "Me too.", '
                '"3": "W-Wait a second!", "4": "It\'s been a while since we did this...", '
                '"5": "No way, right? haha", "6": "GYAAAAAH!"}'
            )
        else:
            subject_rule = (
                "Supply natural omitted English subjects and objects from dialogue flow "
                "(e.g., '좋아해' -> 'I like you.', '나도' -> 'Me too.')."
            )
            idiom_rule = (
                "Translate idioms and colloquialisms into natural English "
                "(e.g., '이런 거 오랜만이네...' -> \"It's been a while since we did this...\", "
                "'ㅋㅋ'/'ㅎㅎ' -> 'lol'/'haha' or natural tone)."
            )
            sigh_example = "'잠깐만 기다려' -> 'W-Wait a second!', '으음…' -> 'Mmh...'"
            name_rule = (
                "Keep character names consistent across bubbles and pages using the glossary "
                "and recent dialogue."
            )
            anti_romaji = (
                'Translate ALL spoken words into English—never phonetic Korean '
                '(e.g., "Jamkkanman gidaryeo", "Yeogin silheo") or Hangul/CJK characters; '
                "keep only proper character names in Latin letters."
            )
            examples = (
                'Example input: {"0": "환영여단", "1": "...좋아해", "2": "나도", '
                '"3": "잠깐만 기다려", "4": "이런 거 오랜만이네...", '
                '"5": "말도 안 돼ㅋㅋ", "6": "젠장! 끄아아악!"}\n'
                'Example output: {"0": "Phantom Troupe", "1": "...I like you.", "2": "Me too.", '
                '"3": "W-Wait a second!", "4": "It\'s been a while since we did this...", '
                '"5": "No way, lol", "6": "Damn it! GYAAAAAH!"}'
            )

        system_prompt = (
            f"You are an expert uncensored {source_label}-to-English localization translator.\n"
            "Rules:\n"
            "1. Translate every bubble into natural, fluent, idiomatic English manga dialogue—"
            "concise comic lettering, NOT stiff word-for-word literal translations.\n"
            f"2. Output strictly English ONLY. {anti_romaji}\n"
            f"3. {subject_rule}\n"
            f"4. {idiom_rule}\n"
            "5. Keep ellipses ('…'/'...') as '...'—NEVER spell them out as 'Dot dot'. "
            f"Preserve emotional punch, hesitation, stutters, screams, and sighs naturally ({sigh_example}).\n"
            f"6. {name_rule}\n"
            "7. Entries tagged [floating] may be borderless speech/thoughts or SFX: translate speech "
            "naturally and SFX concisely in comic style (e.g., '*THUMP*', '*RUSTLE*').\n"
            "8. Stay 100% faithful and completely uncensored across all genres (profanity, horror, "
            "violence, mature/NSFW). Never sanitize, lecture, or refuse.\n"
            "9. Output ONLY a single valid JSON object mapping each string bubble ID to its English "
            "translation, with zero commentary or markdown.\n"
            f"{examples}"
        )

        sections = self._format_context_sections(
            glossary,
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
                    "Region Type Hints (do NOT include bracket tags in your translated output): "
                    + ", ".join(tagged_notes)
                )

        bubble_map = {str(bid): _normalize_ocr_for_llm(text) for bid, text in bubbles}
        sections.append(
            "Translate EVERY bubble below into natural, idiomatic English comic dialogue "
            "(English ONLY; never Romaji, never CJK). "
            "Return ONLY a JSON object mapping each string bubble ID to its English translation:\n"
            + json.dumps(bubble_map, ensure_ascii=False)
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(sections)},
        ]

    def _build_retry_messages(
        self,
        failed_bubbles: list[tuple[int, str]],
        lang: Literal["ja", "ko"],
        glossary: dict[str, str],
        reason: str = "missing_or_refusal",
        context: list[tuple[str, str]] | None = None,
        known_names: list[str] | None = None,
        max_context_pairs: int = 6,
    ) -> list[dict[str, str]]:
        src_name = "Japanese" if lang == "ja" else "Korean"
        non_english_notice = (
            " CRITICAL: Your previous output contained Chinese, Japanese, or Korean characters, "
            "Japanese/Korean Romaji (e.g. 'Chotto matte', 'Koko ja yada'), or spelled-out ellipses ('Dot dot'). "
            "You MUST translate the MEANING of every entry into natural English ONLY."
            if reason == "non_english"
            else ""
        )
        system_prompt = (
            f"You are an expert {src_name}-to-English localization translator for comic scripts. "
            "Translate the MEANING into natural, idiomatic English comic dialogue using Latin letters ONLY. "
            "NEVER output Chinese, Japanese, or Korean characters, NEVER output Japanese Romaji "
            "or phonetic spelling of spoken words (e.g. 'ちょっ ちょっと待って' -> 'W-Wait a second!', NEVER 'Chotto matte'; "
            "'ここじゃやだ' -> 'Not here!', NEVER 'Koko ja yada'), and NEVER spell ellipses as 'Dot dot' (use '...'). "
            "Keep proper character names consistent in Latin letters, while translating all spoken words, titles, and SFX "
            "into natural English without omission, commentary, or refusal."
            f"{non_english_notice} "
            "Return ONLY a valid JSON object mapping each string ID to its English translation."
        )
        parts = self._format_context_sections(
            glossary,
            context or [],
            known_names=known_names,
            max_context_pairs=max_context_pairs,
        )
        bubble_map = {str(bid): _normalize_ocr_for_llm(text) for bid, text in failed_bubbles}
        user_header = (
            "Previous output contained non-English text, Romaji, or literalized ellipses and was rejected. "
            "Translate the MEANING of EVERY item below into natural English ONLY (zero CJK, zero Romaji):"
            if reason == "non_english"
            else "Translate EVERY item below into natural, idiomatic English ONLY (zero CJK, zero Romaji):"
        )
        parts.append(f"{user_header}\n" + json.dumps(bubble_map, ensure_ascii=False))
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    async def _translate_single_bubble_english(
        self,
        bid: int,
        orig_text: str,
        lang: Literal["ja", "ko"],
        glossary: dict[str, str],
        cancel_event: asyncio.Event | None = None,
        context: list[tuple[str, str]] | None = None,
        known_names: list[str] | None = None,
        max_context_pairs: int = 6,
    ) -> str | None:
        """Compact single-bubble natural English retry (max_tokens=96, zero Romaji)."""
        if cancel_event is not None and cancel_event.is_set():
            return None
        src_name = "Japanese" if lang == "ja" else "Korean"
        system_prompt = (
            f"You are an expert {src_name}-to-English manga localization translator. "
            "Translate the MEANING of this single comic line into natural, fluent, idiomatic English. "
            "Supply omitted subjects/objects naturally from context, keep ellipses as '...' (NEVER 'Dot dot'), "
            "and preserve stutters/sighs naturally (e.g. 'ちょっ ちょっと待って' -> 'W-Wait a second!'). "
            "NEVER output Chinese, Japanese, or Korean characters, and NEVER output Japanese Romaji "
            "(NEVER 'Chotto matte' or 'Koko ja yada'). "
            "Keep character names consistent in Latin letters, and translate all dialogue, titles, and SFX into English. "
            f'Return ONLY a valid JSON object with key "{bid}".'
        )
        parts = self._format_context_sections(
            glossary,
            context or [],
            known_names=known_names,
            max_context_pairs=max_context_pairs,
        )
        normalized_orig = _normalize_ocr_for_llm(orig_text)
        parts.append(
            "Translate this single item's MEANING into natural, idiomatic English ONLY (zero CJK, zero Romaji). "
            f'Return JSON mapping key "{bid}" to its English translation:\n'
            + json.dumps({str(bid): normalized_orig}, ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        raw = await self._invoke_llm(
            messages,
            temperature=0.2,
            top_p=DEFAULT_TOP_P,
            repeat_penalty=DEFAULT_REPEAT_PENALTY,
            max_tokens=96,
            json_mode=True,
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            return None
        parsed = _parse_translation_json(raw)
        candidate = parsed.get(bid, "").strip()
        if not candidate and len(parsed) == 1:
            candidate = next(iter(parsed.values())).strip()
        sanitized, ok = _sanitize_translation_candidate(orig_text, candidate)
        if ok:
            return sanitized
        logger.warning(
            "Single-bubble English retry failed for bubble %s (orig=%r, raw=%r)",
            bid,
            orig_text,
            raw,
        )
        return None

    async def _run_primary_stream(
        self,
        bubbles: list[tuple[int, str]],
        primary_messages: list[dict[str, str]],
        on_bubble: BubbleCallback,
        *,
        max_tokens: int = 640,
        cancel_event: asyncio.Event | None = None,
    ) -> tuple[dict[int, str], list[tuple[int, str]], bool]:
        """Stream primary JSON response for a chunk and emit each valid bubble immediately."""
        orig_by_id = dict(bubbles)
        resolved: dict[int, str] = {}
        seen_ids: set[int] = set()
        rejected_candidates: dict[int, str] = {}
        buffer = ""
        scan_pos = 0

        stream_iter = (
            self._llm_stream_caller(primary_messages)
            if self._llm_stream_caller is not None
            else self._call_llama_stream(
                primary_messages,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                repeat_penalty=DEFAULT_REPEAT_PENALTY,
                max_tokens=max_tokens,
                cancel_event=cancel_event,
            )
        )
        async for chunk in stream_iter:
            if cancel_event is not None and cancel_event.is_set():
                break
            if not chunk:
                continue
            buffer += chunk
            for match in _KV_FALLBACK_RE.finditer(buffer, scan_pos):
                scan_pos = match.end()
                bid = int(match.group(1))
                if bid not in orig_by_id or bid in seen_ids:
                    continue
                seen_ids.add(bid)
                raw_val = match.group(2)
                try:
                    decoded = str(json.loads(f'"{raw_val}"')).strip()
                except json.JSONDecodeError:
                    decoded = raw_val.strip()
                sanitized, ok = _sanitize_translation_candidate(orig_by_id[bid], decoded)
                if ok:
                    resolved[bid] = sanitized
                    await on_bubble(bid, sanitized, False)
                else:
                    rejected_candidates[bid] = decoded
                    logger.warning(
                        "Primary stream bubble %s rejected (orig=%r, candidate=%r)",
                        bid,
                        orig_by_id[bid],
                        decoded,
                    )

        if cancel_event is not None and cancel_event.is_set():
            return resolved, [], False

        # Sweep complete buffer for any valid pairs missed by incremental regex
        if len(seen_ids) < len(bubbles):
            parsed = _parse_translation_json(buffer)
            for bid, orig_text in bubbles:
                if bid in seen_ids:
                    continue
                candidate = parsed.get(bid, "").strip()
                if not candidate:
                    continue
                seen_ids.add(bid)
                sanitized, ok = _sanitize_translation_candidate(orig_text, candidate)
                if ok:
                    resolved[bid] = sanitized
                    await on_bubble(bid, sanitized, False)
                else:
                    rejected_candidates[bid] = candidate

        failed: list[tuple[int, str]] = []
        any_non_english = False
        for bid, orig_text in bubbles:
            if bid not in resolved:
                cand = rejected_candidates.get(bid, "")
                if cand and not is_refusal(cand):
                    any_non_english = True
                failed.append((bid, orig_text))

        if failed:
            logger.warning(
                "Primary stream pass had %d failed/missing bubble(s) %s; raw=%r",
                len(failed),
                [b for b, _ in failed],
                buffer,
            )

        return resolved, failed, any_non_english

    def _get_memory_glossary(self, chapter_id: str, series_id: str | None) -> dict[str, str]:
        fn = self.memory.get_glossary
        params = inspect.signature(fn).parameters
        if "series_id" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(fn(chapter_id, series_id=series_id))
        return dict(fn(chapter_id))

    def _get_memory_pairs(
        self,
        chapter_id: str,
        series_id: str | None,
    ) -> list[tuple[str, str]]:
        ctx_fn = self.memory.get_context
        params = inspect.signature(ctx_fn).parameters
        if "series_id" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return list(ctx_fn(chapter_id, series_id=series_id))
        return list(ctx_fn(chapter_id))

    def _get_memory_context(
        self,
        chapter_id: str,
        series_id: str | None,
        max_pairs: int = 6,
    ) -> list[tuple[str, str]]:
        if max_pairs <= 0:
            return []
        pairs = self._get_memory_pairs(chapter_id, series_id)
        return pairs[-max_pairs:]

    def _record_memory_page(
        self,
        chapter_id: str,
        pairs: list[tuple[str, str]],
        series_id: str | None,
    ) -> None:
        fn = self.memory.record_page
        params = inspect.signature(fn).parameters
        if "series_id" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            fn(chapter_id, pairs, series_id=series_id)
        else:
            fn(chapter_id, pairs)

    async def translate_page(
        self,
        bubbles: list[tuple[int, str]],
        lang: Literal["ja", "ko"],
        chapter_id: str,
        extra_glossary: dict[str, str] | None = None,
        on_bubble: BubbleCallback | None = None,
        series_id: str | None = None,
        bubble_types: dict[int, str] | None = None,
        cancel_event: asyncio.Event | None = None,
        gpu_guard: Any = None,
    ) -> dict[int, str]:
        """Translate page bubbles in fast sequential chunks with sliding context, intra-page carry-over, and cancellation."""
        if not bubbles:
            return {}

        merged_glossary = self._get_memory_glossary(chapter_id, series_id)
        if extra_glossary:
            for k, v in extra_glossary.items():
                ks = str(k).strip()
                vs = str(v).strip()
                if ks and vs:
                    merged_glossary[ks] = vs

        use_story_context = bool(getattr(self.cfg, "use_story_context", True))
        max_context_pairs = max(0, int(getattr(self.cfg, "max_context_pairs", 6)))
        chunk_size = max(1, int(getattr(self.cfg, "llm_chunk_size", 12)))

        rolling_context: list[tuple[str, str]] = []
        known_names: list[str] = []
        if use_story_context:
            prior_pairs = self._get_memory_pairs(chapter_id, series_id)
            if prior_pairs:
                known_names = extract_character_names(prior_pairs)
                if max_context_pairs > 0:
                    rolling_context = list(prior_pairs[-max_context_pairs:])

        chunks = [bubbles[i : i + chunk_size] for i in range(0, len(bubbles), chunk_size)]
        resolved: dict[int, str] = {}

        for chunk_idx, chunk in enumerate(chunks):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Translation cancelled before chunk %d/%d", chunk_idx + 1, len(chunks))
                break

            if gpu_guard is not None:
                guard_state = await gpu_guard.wait_if_hot(cancel_event=cancel_event)
                if guard_state == "cancelled" or (cancel_event is not None and cancel_event.is_set()):
                    logger.info(
                        "Translation cancelled during thermal guard before chunk %d/%d",
                        chunk_idx + 1,
                        len(chunks),
                    )
                    break

            chunk_max_tokens = _chunk_max_tokens(len(chunk))
            primary_messages = self._build_primary_messages(
                chunk,
                lang,
                merged_glossary,
                rolling_context if use_story_context else [],
                bubble_types=bubble_types,
                known_names=known_names if use_story_context else None,
                max_context_pairs=max_context_pairs,
            )

            chunk_resolved: dict[int, str] = {}
            failed: list[tuple[int, str]] = []
            any_non_english = False

            if on_bubble is not None and (self._llm_stream_caller is not None or self._llm_caller is None):
                chunk_resolved, failed, any_non_english = await self._run_primary_stream(
                    chunk,
                    primary_messages,
                    on_bubble,
                    max_tokens=chunk_max_tokens,
                    cancel_event=cancel_event,
                )
            else:
                raw_primary = await self._invoke_llm(
                    primary_messages,
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P,
                    repeat_penalty=DEFAULT_REPEAT_PENALTY,
                    max_tokens=chunk_max_tokens,
                    json_mode=True,
                    cancel_event=cancel_event,
                )
                parsed = _parse_translation_json(raw_primary)
                for bid, orig_text in chunk:
                    candidate = parsed.get(bid, "").strip()
                    sanitized, ok = _sanitize_translation_candidate(orig_text, candidate)
                    if ok:
                        chunk_resolved[bid] = sanitized
                        if on_bubble is not None and (cancel_event is None or not cancel_event.is_set()):
                            await on_bubble(bid, sanitized, False)
                    else:
                        if candidate and not is_refusal(candidate):
                            any_non_english = True
                        failed.append((bid, orig_text))
                if failed and (cancel_event is None or not cancel_event.is_set()):
                    logger.warning(
                        "Primary chunk %d/%d had %d failed/missing bubble(s) %s; raw=%r",
                        chunk_idx + 1,
                        len(chunks),
                        len(failed),
                        [b for b, _ in failed],
                        raw_primary,
                    )

            resolved.update(chunk_resolved)

            if cancel_event is not None and cancel_event.is_set():
                break

            # Build local rolling context for retries (prior rolling window + valid lines already resolved in this chunk)
            retry_context: list[tuple[str, str]] = []
            retry_names: list[str] = []
            if use_story_context:
                chunk_so_far = [
                    (orig_text, chunk_resolved[bid])
                    for bid, orig_text in chunk
                    if bid in chunk_resolved
                ]
                if max_context_pairs > 0:
                    retry_context = (rolling_context + chunk_so_far)[-max_context_pairs:]
                retry_names = extract_character_names(chunk_so_far, existing=known_names)

            if len(failed) >= 2:
                retry_reason = "non_english" if any_non_english else "missing_or_refusal"
                logger.warning(
                    "Triggering compact batch retry (%s) for chunk %d/%d bubble(s) %s",
                    retry_reason,
                    chunk_idx + 1,
                    len(chunks),
                    [b for b, _ in failed],
                )
                retry_messages = self._build_retry_messages(
                    failed,
                    lang,
                    merged_glossary,
                    reason=retry_reason,
                    context=retry_context if use_story_context else None,
                    known_names=retry_names if use_story_context else None,
                    max_context_pairs=max_context_pairs,
                )
                raw_retry = await self._invoke_llm(
                    retry_messages,
                    temperature=0.2,
                    top_p=DEFAULT_TOP_P,
                    repeat_penalty=DEFAULT_REPEAT_PENALTY,
                    max_tokens=_chunk_max_tokens(len(failed)),
                    json_mode=True,
                    cancel_event=cancel_event,
                )
                if cancel_event is not None and cancel_event.is_set():
                    break
                retry_parsed = _parse_translation_json(raw_retry)
                still_failed: list[tuple[int, str]] = []
                for bid, orig_text in failed:
                    candidate = retry_parsed.get(bid, "").strip()
                    sanitized, ok = _sanitize_translation_candidate(orig_text, candidate)
                    if ok:
                        resolved[bid] = sanitized
                        chunk_resolved[bid] = sanitized
                        if on_bubble is not None:
                            await on_bubble(bid, sanitized, True)
                    else:
                        still_failed.append((bid, orig_text))

                # Fall back to single-bubble retry (max 1 attempt per remaining bubble) so real bubbles are never silently dropped.
                for bid, orig_text in still_failed:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    single_candidate = await self._translate_single_bubble_english(
                        bid,
                        orig_text,
                        lang,
                        merged_glossary,
                        cancel_event=cancel_event,
                        context=retry_context if use_story_context else None,
                        known_names=retry_names if use_story_context else None,
                        max_context_pairs=max_context_pairs,
                    )
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    if single_candidate is not None:
                        resolved[bid] = single_candidate
                        chunk_resolved[bid] = single_candidate
                        if on_bubble is not None:
                            await on_bubble(bid, single_candidate, True)
                    else:
                        logger.warning(
                            "Single-bubble fallback after batch retry failed for bubble %s (orig=%r); emitting empty string",
                            bid,
                            orig_text,
                        )
                        resolved[bid] = ""
                        if on_bubble is not None:
                            await on_bubble(bid, "", True)
            elif len(failed) == 1:
                bid, orig_text = failed[0]
                single_candidate = await self._translate_single_bubble_english(
                    bid,
                    orig_text,
                    lang,
                    merged_glossary,
                    cancel_event=cancel_event,
                    context=retry_context if use_story_context else None,
                    known_names=retry_names if use_story_context else None,
                    max_context_pairs=max_context_pairs,
                )
                if cancel_event is not None and cancel_event.is_set():
                    break
                if single_candidate is not None:
                    resolved[bid] = single_candidate
                    chunk_resolved[bid] = single_candidate
                    if on_bubble is not None:
                        await on_bubble(bid, single_candidate, True)
                else:
                    logger.warning(
                        "Single-bubble retry failed for bubble %s (orig=%r); emitting empty string",
                        bid,
                        orig_text,
                    )
                    resolved[bid] = ""
                    if on_bubble is not None:
                        await on_bubble(bid, "", True)

            if cancel_event is not None and cancel_event.is_set():
                break

            # Intra-page chunk carry-over: feed this chunk's clean pairs and detected names into the next chunk.
            if use_story_context:
                chunk_valid_pairs = [
                    (orig_text, resolved[bid])
                    for bid, orig_text in chunk
                    if is_valid_english_translation(orig_text, resolved.get(bid, ""))
                ]
                if chunk_valid_pairs:
                    if max_context_pairs > 0:
                        rolling_context = (rolling_context + chunk_valid_pairs)[-max_context_pairs:]
                    known_names = extract_character_names(chunk_valid_pairs, existing=known_names)

        recorded_pairs: list[tuple[str, str]] = []
        for bid, orig_text in bubbles:
            trans = resolved.get(bid, "")
            if is_valid_english_translation(orig_text, trans):
                recorded_pairs.append((orig_text, trans))
        if recorded_pairs:
            self._record_memory_page(chapter_id, recorded_pairs, series_id)

        return resolved

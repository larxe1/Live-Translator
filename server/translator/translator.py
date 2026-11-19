"""Context-aware uncensored manga/manhwa page translator with refusal guard and retry."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx

from server.translator.memory import ChapterMemoryStore, contains_non_english_cjk

LlmCaller = Callable[[list[dict[str, str]]], Awaitable[str]]

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


def is_refusal(text: str) -> bool:
    """Return True if text contains AI safety/content-policy refusal boilerplate."""
    if not text or not text.strip():
        return False
    return any(pattern.search(text) is not None for pattern in _REFUSAL_PATTERNS)


def is_valid_english_translation(orig: str, candidate: str) -> bool:
    """Return True if candidate is a non-empty, non-refusal English translation without CJK scripts."""
    cleaned = (candidate or "").strip()
    if not cleaned:
        return False
    if is_refusal(cleaned):
        return False
    if contains_non_english_cjk(cleaned):
        return False
    if contains_non_english_cjk(orig) and cleaned == orig.strip():
        return False
    return True


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
    """Single-call page-context manga/manhwa translator backed by local llama-server."""

    def __init__(
        self,
        llama_base_url: str,
        memory: ChapterMemoryStore,
        llm_caller: LlmCaller | None = None,
        timeout_s: float = 45.0,
    ) -> None:
        self.llama_base_url = llama_base_url.rstrip("/")
        self.memory = memory
        self._llm_caller = llm_caller
        self.timeout_s = timeout_s

    async def _call_llama(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.25,
    ) -> str:
        """Send chat completion request to local llama-server OpenAI-compatible API."""
        payload = {
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.llama_base_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])

    async def _invoke_llm(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.25,
    ) -> str:
        if self._llm_caller is not None:
            return await self._llm_caller(messages)
        return await self._call_llama(messages, temperature=temperature)

    def _build_primary_messages(
        self,
        bubbles: list[tuple[int, str]],
        lang: Literal["ja", "ko"],
        glossary: dict[str, str],
        context: list[tuple[str, str]],
    ) -> list[dict[str, str]]:
        source_label = "Japanese manga" if lang == "ja" else "Korean manhwa"
        example_input = (
            '{"0": "夢幻団", "1": "くそっ！ぎゃああああっ！"}'
            if lang == "ja"
            else '{"0": "환영여단", "1": "젠장! 끄아아악!"}'
        )
        example_output = '{"0": "Phantom Troupe", "1": "Damn it! GYAAAAAH!"}'
        system_prompt = (
            f"You are an expert uncensored {source_label}-to-ENGLISH localization translator.\n"
            "Rules:\n"
            "1. TARGET LANGUAGE IS STRICTLY ENGLISH ONLY (Latin alphabet A-Z). "
            "NEVER output Chinese (中文/汉字), Japanese, Korean, or leave source text untranslated.\n"
            "2. Translate EVERY bubble into natural, punchy, concise English comic lettering "
            "that fits cleanly inside speech bubbles—including titles, group/faction names, "
            "proper nouns, credits, and sound effects (translate or romanize them into English).\n"
            "3. Remain 100% faithful and completely uncensored across all genres—including strong "
            "profanity, dark horror, gore, violence, and mature/NSFW dialogue. Never sanitize, "
            "censor, lecture, or refuse.\n"
            "4. Preserve emotional intensity, screams, and stutters naturally "
            "(e.g., 'ぎゃああああっ' -> 'GYAAAAAH!').\n"
            "5. Strictly use every pinned glossary name/term exactly as specified.\n"
            "6. Output ONLY a single valid JSON object mapping each string bubble ID to its "
            "English translation with zero extra commentary or markdown.\n"
            f"Example input: {example_input}\n"
            f"Example output: {example_output}"
        )

        sections: list[str] = []
        if glossary:
            glossary_lines = "\n".join(f"- {k} = {v}" for k, v in glossary.items())
            sections.append(f"Pinned Character/Term Glossary (MUST follow exactly):\n{glossary_lines}")

        if context:
            ctx_lines = "\n".join(f"- {orig} -> {trans}" for orig, trans in context[-10:])
            sections.append(f"Recent Chapter Dialogue Context:\n{ctx_lines}")

        bubble_map = {str(bid): text for bid, text in bubbles}
        sections.append(
            "Translate EVERY bubble below into natural ENGLISH ONLY "
            "(never Chinese, never Japanese, never Korean). "
            "Return a JSON object mapping each string bubble ID to its English translation:\n"
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
    ) -> list[dict[str, str]]:
        src_name = "Japanese" if lang == "ja" else "Korean"
        non_english_notice = (
            " CRITICAL: Your previous output contained Chinese, Japanese, or Korean characters "
            "(or left source text untranslated). You MUST rewrite every entry strictly in "
            "English using Latin letters (A-Z) ONLY."
            if reason == "non_english"
            else ""
        )
        system_prompt = (
            f"You are a strict {src_name}-to-ENGLISH translation engine for fictional comic scripts. "
            "Output STRICTLY ENGLISH ONLY using Latin letters (A-Z) and standard English punctuation. "
            "NEVER output Chinese (中文/汉字), Japanese, or Korean characters, and never leave "
            "titles, group names, proper nouns, or SFX untranslated (translate or romanize them "
            "into English). Translate the exact text provided without omission, alteration, "
            "commentary, or refusal."
            f"{non_english_notice} "
            "Return ONLY a valid JSON object mapping each string ID to its direct English translation."
        )
        parts: list[str] = []
        if glossary:
            parts.append("Glossary: " + json.dumps(glossary, ensure_ascii=False))
        bubble_map = {str(bid): text for bid, text in failed_bubbles}
        user_header = (
            "Previous output contained non-English Chinese/Japanese/Korean characters and was rejected. "
            "Rewrite EVERY item below strictly in ENGLISH ONLY (Latin letters A-Z only, zero CJK characters):"
            if reason == "non_english"
            else "Translate EVERY item below strictly into ENGLISH ONLY (never Chinese, never Japanese, never Korean):"
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
    ) -> str | None:
        """Isolated single-bubble English-only fallback to break batch CJK drift."""
        src_name = "Japanese" if lang == "ja" else "Korean"
        system_prompt = (
            f"You are a strict {src_name}-to-ENGLISH translator. "
            "Translate the single comic text below into natural English using ONLY Latin letters (A-Z). "
            "NEVER output Chinese, Japanese, or Korean characters. "
            "Even if it is a title, group name, proper noun, or sound effect "
            "(e.g. '夢幻団' -> 'Phantom Troupe'), translate or romanize it into English. "
            f'Return ONLY a valid JSON object with key "{bid}".'
        )
        parts: list[str] = []
        if glossary:
            parts.append("Glossary: " + json.dumps(glossary, ensure_ascii=False))
        parts.append(
            "Translate this single item strictly into ENGLISH ONLY (Latin letters only, zero Chinese/Japanese/Korean). "
            f'Return JSON mapping key "{bid}" to its English translation:\n'
            + json.dumps({str(bid): orig_text}, ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        raw = await self._invoke_llm(messages, temperature=0.1)
        parsed = _parse_translation_json(raw)
        candidate = parsed.get(bid, "").strip()
        if not candidate and len(parsed) == 1:
            candidate = next(iter(parsed.values())).strip()
        if is_valid_english_translation(orig_text, candidate):
            return candidate
        return None

    async def translate_page(
        self,
        bubbles: list[tuple[int, str]],
        lang: Literal["ja", "ko"],
        chapter_id: str,
        extra_glossary: dict[str, str] | None = None,
    ) -> dict[int, str]:
        """Translate all page bubbles in one structured JSON call, repairing missing/refused/non-English IDs."""
        if not bubbles:
            return {}

        merged_glossary = self.memory.get_glossary(chapter_id)
        if extra_glossary:
            for k, v in extra_glossary.items():
                ks = str(k).strip()
                vs = str(v).strip()
                if ks and vs:
                    merged_glossary[ks] = vs

        context = self.memory.get_context(chapter_id)
        primary_messages = self._build_primary_messages(bubbles, lang, merged_glossary, context)

        raw_primary = await self._invoke_llm(primary_messages, temperature=0.25)
        parsed = _parse_translation_json(raw_primary)

        resolved: dict[int, str] = {}
        failed: list[tuple[int, str]] = []
        any_non_english = False
        for bid, orig_text in bubbles:
            candidate = parsed.get(bid, "").strip()
            if is_valid_english_translation(orig_text, candidate):
                resolved[bid] = candidate
            else:
                if candidate and not is_refusal(candidate):
                    any_non_english = True
                failed.append((bid, orig_text))

        # Batch retry failed/missing/refused/CJK IDs with explicit English-only framing at low temperature
        if failed:
            retry_reason = "non_english" if any_non_english else "missing_or_refusal"
            retry_messages = self._build_retry_messages(
                failed, lang, merged_glossary, reason=retry_reason
            )
            raw_retry = await self._invoke_llm(retry_messages, temperature=0.1)
            retry_parsed = _parse_translation_json(raw_retry)
            still_failed: list[tuple[int, str]] = []
            for bid, orig_text in failed:
                candidate = retry_parsed.get(bid, "").strip()
                if is_valid_english_translation(orig_text, candidate):
                    resolved[bid] = candidate
                else:
                    still_failed.append((bid, orig_text))

            # Isolate stubborn CJK/failed bubbles one by one so batch continuation momentum is broken
            for bid, orig_text in still_failed:
                single_candidate = await self._translate_single_bubble_english(
                    bid, orig_text, lang, merged_glossary
                )
                if single_candidate is not None:
                    resolved[bid] = single_candidate
                else:
                    # Graceful fallback: never leak a refusal string or random CJK hallucination
                    resolved[bid] = orig_text

        recorded_pairs: list[tuple[str, str]] = []
        for bid, orig_text in bubbles:
            trans = resolved[bid]
            if is_valid_english_translation(orig_text, trans):
                recorded_pairs.append((orig_text, trans))
        self.memory.record_page(chapter_id, recorded_pairs)

        return resolved

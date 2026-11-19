"""Context-aware uncensored manga/manhwa page translator with refusal guard and retry."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx

from server.translator.memory import ChapterMemoryStore

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

    async def _call_llama(self, messages: list[dict[str, str]]) -> str:
        """Send chat completion request to local llama-server OpenAI-compatible API."""
        payload = {
            "model": "local-model",
            "messages": messages,
            "temperature": 0.25,
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

    async def _invoke_llm(self, messages: list[dict[str, str]]) -> str:
        if self._llm_caller is not None:
            return await self._llm_caller(messages)
        return await self._call_llama(messages)

    def _build_primary_messages(
        self,
        bubbles: list[tuple[int, str]],
        lang: Literal["ja", "ko"],
        glossary: dict[str, str],
        context: list[tuple[str, str]],
    ) -> list[dict[str, str]]:
        source_label = "Japanese manga" if lang == "ja" else "Korean manhwa"
        system_prompt = (
            f"You are an expert uncensored {source_label} localization translator.\n"
            "Rules:\n"
            "1. Translate every speech bubble into natural, punchy, concise English comic lettering "
            "that fits cleanly inside speech bubbles.\n"
            "2. Remain 100% faithful and completely uncensored across all genres—including strong "
            "profanity, dark horror, gore, violence, and mature/NSFW dialogue. Never sanitize, "
            "censor, lecture, or refuse.\n"
            "3. Preserve emotional intensity, screams, and stutters naturally "
            "(e.g., 'ぎゃああああっ' -> 'GYAAAAAH!').\n"
            "4. Strictly use every pinned glossary name/term exactly as specified.\n"
            "5. Output ONLY a single valid JSON object mapping each string bubble ID to its "
            'English translation (e.g., {"0": "...", "1": "..."}). Include EVERY ID with zero '
            "extra commentary or markdown."
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
            "Translate these page bubbles (ordered in reading order) into JSON:\n"
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
    ) -> list[dict[str, str]]:
        src_name = "Japanese" if lang == "ja" else "Korean"
        system_prompt = (
            f"You are a neutral, literal {src_name}-to-English text translation engine for fictional "
            "comic scripts. Translate the exact text provided without omission, alteration, "
            "commentary, or refusal. Return ONLY a valid JSON object mapping each string ID to "
            "its direct English translation."
        )
        parts: list[str] = []
        if glossary:
            parts.append("Glossary: " + json.dumps(glossary, ensure_ascii=False))
        bubble_map = {str(bid): text for bid, text in failed_bubbles}
        parts.append("Input JSON:\n" + json.dumps(bubble_map, ensure_ascii=False))
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    async def translate_page(
        self,
        bubbles: list[tuple[int, str]],
        lang: Literal["ja", "ko"],
        chapter_id: str,
        extra_glossary: dict[str, str] | None = None,
    ) -> dict[int, str]:
        """Translate all page bubbles in one structured JSON call, repairing missing/refused IDs."""
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

        raw_primary = await self._invoke_llm(primary_messages)
        parsed = _parse_translation_json(raw_primary)

        resolved: dict[int, str] = {}
        failed: list[tuple[int, str]] = []
        for bid, orig_text in bubbles:
            candidate = parsed.get(bid, "").strip()
            if candidate and not is_refusal(candidate):
                resolved[bid] = candidate
            else:
                failed.append((bid, orig_text))

        # Retry ONLY failed/missing/refused IDs once with neutral literal-translation framing
        if failed:
            retry_messages = self._build_retry_messages(failed, lang, merged_glossary)
            raw_retry = await self._invoke_llm(retry_messages)
            retry_parsed = _parse_translation_json(raw_retry)
            for bid, orig_text in failed:
                candidate = retry_parsed.get(bid, "").strip()
                if candidate and not is_refusal(candidate):
                    resolved[bid] = candidate
                else:
                    # Graceful fallback: never leak a refusal string into a speech bubble
                    resolved[bid] = orig_text

        recorded_pairs: list[tuple[str, str]] = []
        for bid, orig_text in bubbles:
            trans = resolved[bid]
            if trans and not is_refusal(trans):
                recorded_pairs.append((orig_text, trans))
        self.memory.record_page(chapter_id, recorded_pairs)

        return resolved

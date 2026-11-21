"""Context-aware uncensored manga/manhwa page translator with refusal guard and retry."""

from __future__ import annotations

import inspect
import json
import logging
import re
import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

import httpx

from server.translator.memory import ChapterMemoryStore, contains_non_english_cjk

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
_HANZI_OR_HANGUL_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]"
)
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_KANA_SPAN_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\uff65-\uff9f]+")
_KANA_WORD_RE = re.compile(r"[\u3040-\u309f\u30a1-\u30fa\u30fc-\u30fe]+")

_KANA_DIGRAPHS: dict[str, str] = {
    # Hiragana digraphs
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo", "きぇ": "kye",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "しぇ": "she",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho", "ちぇ": "che",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo", "にぇ": "nye",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo", "ひぇ": "hye",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo", "みぇ": "mye",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo", "りぇ": "rye",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo", "ぎぇ": "gye",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo", "じぇ": "je",
    "ぢゃ": "ja", "ぢゅ": "ju", "ぢょ": "jo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo", "びぇ": "bye",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo", "ぴぇ": "pye",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo", "ふゅ": "fyu",
    "てぃ": "ti", "てゅ": "tyu", "でぃ": "di", "でゅ": "dyu",
    "とぅ": "tu", "どぅ": "du",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo", "ゔゅ": "vyu",
    "つぁ": "tsa", "つぃ": "tsi", "つぇ": "tse", "つぉ": "tso",
    "いぇ": "ye", "くぁ": "kwa", "くぃ": "kwi", "くぇ": "kwe", "くぉ": "kwo", "ぐぁ": "gwa",
    # Katakana digraphs
    "キャ": "kya", "キュ": "kyu", "キョ": "kyo", "キェ": "kye",
    "シャ": "sha", "シュ": "shu", "ショ": "sho", "シェ": "she",
    "チャ": "cha", "チュ": "chu", "チョ": "cho", "チェ": "che",
    "ニャ": "nya", "ニュ": "nyu", "ニョ": "nyo", "ニェ": "nye",
    "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo", "ヒェ": "hye",
    "ミャ": "mya", "ミュ": "myu", "ミョ": "myo", "ミェ": "mye",
    "リャ": "rya", "リュ": "ryu", "リョ": "ryo", "リェ": "rye",
    "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo", "ギェ": "gye",
    "ジャ": "ja", "ジュ": "ju", "ジョ": "jo", "ジェ": "je",
    "ヂャ": "ja", "ヂュ": "ju", "ヂョ": "jo",
    "ビャ": "bya", "ビュ": "byu", "ビョ": "byo", "ビェ": "bye",
    "ピャ": "pya", "ピュ": "pyu", "ピョ": "pyo", "ピェ": "pye",
    "ファ": "fa", "フィ": "fi", "フェ": "fe", "フォ": "fo", "フュ": "fyu",
    "ティ": "ti", "テュ": "tyu", "ディ": "di", "デュ": "dyu",
    "トゥ": "tu", "ドゥ": "du",
    "ウィ": "wi", "ウェ": "we", "ウォ": "wo",
    "ヴァ": "va", "ヴィ": "vi", "ヴェ": "ve", "ヴォ": "vo", "ヴュ": "vyu",
    "ツァ": "tsa", "ツィ": "tsi", "ツェ": "tse", "ツォ": "tso",
    "イェ": "ye", "クァ": "kwa", "クィ": "kwi", "クェ": "kwe", "クォ": "kwo", "グァ": "gwa",
}

_KANA_MONOS: dict[str, str] = {
    # Hiragana
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "wi", "ゑ": "we", "を": "wo", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ゔ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa",
    "ゕ": "ka", "ゖ": "ke",
    # Katakana
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヰ": "wi", "ヱ": "we", "ヲ": "wo", "ン": "n",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "ヴ": "vu",
    "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
    "ャ": "ya", "ュ": "yu", "ョ": "yo", "ヮ": "wa",
    "ヵ": "ka", "ヶ": "ga",
}


def _romanize_kana_word(word: str) -> str:
    """Convert a contiguous Hiragana/Katakana token into capitalized Hepburn romaji."""
    tokens: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if ch in ("っ", "ッ"):
            tokens.append("_GEM_")
            i += 1
            continue
        if ch == "ー":
            tokens.append("_LONG_")
            i += 1
            continue
        if i + 1 < n:
            pair = word[i : i + 2]
            digraph = _KANA_DIGRAPHS.get(pair)
            if digraph is not None:
                tokens.append(digraph)
                i += 2
                continue
        mono = _KANA_MONOS.get(ch)
        if mono is not None:
            tokens.append(mono)
        i += 1

    out: list[str] = []
    for idx, tok in enumerate(tokens):
        if tok == "_GEM_":
            next_tok = ""
            for j in range(idx + 1, len(tokens)):
                if tokens[j] not in ("_GEM_", "_LONG_"):
                    next_tok = tokens[j]
                    break
            if next_tok.startswith("ch"):
                out.append("t")
            elif next_tok and next_tok[0] in "bcdfghjklmnpqrstvwxyz":
                out.append(next_tok[0])
        elif tok == "_LONG_":
            if out:
                last_char = out[-1][-1]
                if last_char in "aeiou":
                    out.append(last_char)
        else:
            out.append(tok)

    joined = "".join(out)
    if not joined:
        return ""
    return joined[0].upper() + joined[1:]


def romanize_residual_kana(text: str) -> str:
    """Convert Hiragana/Katakana spans in text to Hepburn romaji while preserving ASCII text."""
    if not text:
        return text

    def _replace_span(match: re.Match[str]) -> str:
        normalized = unicodedata.normalize("NFKC", match.group(0))
        normalized = normalized.replace("\u30fb", " ").replace("\u30a0", "-").replace("\u30ff", "koto")
        converted = _KANA_WORD_RE.sub(lambda m: _romanize_kana_word(m.group(0)), normalized)
        # Strip any remaining stray kana marks (e.g., standalone dakuten/iteration marks)
        return re.sub(r"[\u3040-\u309f\u30a0-\u30ff]", "", converted)

    return _KANA_SPAN_RE.sub(_replace_span, text)


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


def _sanitize_translation_candidate(orig: str, candidate: str) -> tuple[str, bool]:
    """Validate candidate, salvaging otherwise-English text with residual kana via romanization."""
    cleaned = (candidate or "").strip()
    if is_valid_english_translation(orig, cleaned):
        return (cleaned, True)
    if not cleaned or is_refusal(cleaned):
        return (cleaned, False)
    if _HANZI_OR_HANGUL_RE.search(cleaned) is None and len(_ASCII_LETTER_RE.findall(cleaned)) >= 2:
        romanized = romanize_residual_kana(cleaned).strip()
        if is_valid_english_translation(orig, romanized):
            logger.warning(
                "Sanitized residual kana in translation candidate: %r -> %r",
                cleaned,
                romanized,
            )
            return (romanized, True)
    return (cleaned, False)


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
        llm_stream_caller: LlmStreamCaller | None = None,
    ) -> None:
        self.llama_base_url = llama_base_url.rstrip("/")
        self.memory = memory
        self._llm_caller = llm_caller
        self._llm_stream_caller = llm_stream_caller
        self.timeout_s = timeout_s

    async def _call_llama(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.25,
        json_mode: bool = True,
    ) -> str:
        """Send chat completion request to local llama-server OpenAI-compatible API."""
        payload: dict[str, Any] = {
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1536,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
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
        temperature: float = 0.25,
    ) -> AsyncIterator[str]:
        """Stream chat completion content deltas from local llama-server SSE API."""
        payload: dict[str, Any] = {
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1536,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            async with client.stream(
                "POST",
                f"{self.llama_base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
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
        temperature: float = 0.25,
        json_mode: bool = True,
    ) -> str:
        if self._llm_caller is not None:
            return await self._llm_caller(messages)
        return await self._call_llama(messages, temperature=temperature, json_mode=json_mode)

    def _build_primary_messages(
        self,
        bubbles: list[tuple[int, str]],
        lang: Literal["ja", "ko"],
        glossary: dict[str, str],
        context: list[tuple[str, str]] | str,
        bubble_types: dict[int, str] | None = None,
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
            "2. Write natural, idiomatic spoken English manga/manhwa dialogue—avoid stiff, literal "
            "translation-ese. Keep lines punchy and concise so they fit cleanly inside speech bubbles, "
            "including titles, group/faction names, proper nouns, and credits.\n"
            "3. Treat all bubbles on the page as a connected conversation and use the multi-page "
            "Story Context to resolve dropped subjects (I/you/he/she/they), speaker identity, tone, "
            "and continuity across pages.\n"
            "4. Entries annotated as [floating] may be borderless dialogue/internal thoughts OR "
            "onomatopoeia/SFX: translate borderless dialogue naturally like normal speech or thought, "
            "and translate SFX concisely in comic style (e.g., '*THUMP*', '*RUSTLE*', '*WHOOSH*').\n"
            "5. Remain 100% faithful and completely uncensored across all genres—including strong "
            "profanity, dark horror, gore, violence, and mature/NSFW dialogue. Never sanitize, "
            "censor, lecture, or refuse.\n"
            "6. Preserve emotional intensity, screams, and stutters naturally "
            "(e.g., 'ぎゃああああっ' -> 'GYAAAAAH!').\n"
            "7. Strictly use every pinned glossary name/term exactly as specified.\n"
            "8. Output ONLY a single valid JSON object mapping each string bubble ID to its "
            "English translation with zero extra commentary or markdown.\n"
            f"Example input: {example_input}\n"
            f"Example output: {example_output}"
        )

        sections: list[str] = []
        if glossary:
            glossary_lines = "\n".join(f"- {k} = {v}" for k, v in glossary.items())
            sections.append(f"Pinned Character/Term Glossary (MUST follow exactly):\n{glossary_lines}")

        formatted_context = ""
        if isinstance(context, str):
            formatted_context = context.strip()
        elif context:
            formatted_context = "\n".join(f"- {orig} -> {trans}" for orig, trans in context[-28:])

        if formatted_context:
            sections.append(f"Story Context (Recent Multi-Page Dialogue):\n{formatted_context}")

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
        raw = await self._invoke_llm(messages, temperature=0.1, json_mode=True)
        parsed = _parse_translation_json(raw)
        candidate = parsed.get(bid, "").strip()
        if not candidate and len(parsed) == 1:
            candidate = next(iter(parsed.values())).strip()
        sanitized, ok = _sanitize_translation_candidate(orig_text, candidate)
        if ok:
            return sanitized
        logger.warning(
            "Single-bubble JSON retry failed for bubble %s (orig=%r, raw=%r)",
            bid,
            orig_text,
            raw,
        )
        return None

    async def _translate_single_bubble_plain(
        self,
        bid: int,
        orig_text: str,
        lang: Literal["ja", "ko"],
        glossary: dict[str, str],
    ) -> str | None:
        """4th-stage plain-text single-bubble fallback without JSON grammar."""
        system_prompt = (
            "Translate this manga dialogue line into natural English. "
            "Transliterate all character names (e.g. ナナス -> Nanasu, ヘルメス -> Hermes) into Latin letters. "
            "Output ONLY the English translation with no Japanese, Chinese, or Korean characters."
        )
        parts: list[str] = []
        if glossary:
            parts.append("Glossary: " + json.dumps(glossary, ensure_ascii=False))
        parts.append(orig_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        raw = await self._invoke_llm(messages, temperature=0.1, json_mode=False)
        cleaned = (raw or "").strip()
        fence_match = _FENCE_RE.search(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"' and cleaned.count('"') == 2:
            cleaned = cleaned[1:-1].strip()
        sanitized, ok = _sanitize_translation_candidate(orig_text, cleaned)
        if ok:
            return sanitized
        logger.warning(
            "Single-bubble plain-text fallback failed for bubble %s (orig=%r, raw=%r)",
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
    ) -> tuple[dict[int, str], list[tuple[int, str]], bool]:
        """Stream primary JSON response and emit each bubble as soon as its value closes."""
        orig_by_id = dict(bubbles)
        resolved: dict[int, str] = {}
        seen_ids: set[int] = set()
        rejected_candidates: dict[int, str] = {}
        buffer = ""
        scan_pos = 0

        stream_iter = (
            self._llm_stream_caller(primary_messages)
            if self._llm_stream_caller is not None
            else self._call_llama_stream(primary_messages, temperature=0.25)
        )
        async for chunk in stream_iter:
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

        # Sweep complete buffer for any valid pairs missed by incremental regex (e.g. non-string values)
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

    def _get_memory_context(
        self,
        chapter_id: str,
        series_id: str | None,
    ) -> list[tuple[str, str]] | str:
        formatted_fn = getattr(self.memory, "get_formatted_context", None)
        if callable(formatted_fn):
            params = inspect.signature(formatted_fn).parameters
            if "series_id" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return str(formatted_fn(chapter_id, series_id=series_id, max_pairs=28))
            return str(formatted_fn(chapter_id))
        ctx_fn = self.memory.get_context
        params = inspect.signature(ctx_fn).parameters
        if "series_id" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return list(ctx_fn(chapter_id, series_id=series_id))
        return list(ctx_fn(chapter_id))

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
    ) -> dict[int, str]:
        """Translate all page bubbles in one structured JSON call, repairing missing/refused/non-English IDs."""
        if not bubbles:
            return {}

        merged_glossary = self._get_memory_glossary(chapter_id, series_id)
        if extra_glossary:
            for k, v in extra_glossary.items():
                ks = str(k).strip()
                vs = str(v).strip()
                if ks and vs:
                    merged_glossary[ks] = vs

        context = self._get_memory_context(chapter_id, series_id)
        primary_messages = self._build_primary_messages(
            bubbles,
            lang,
            merged_glossary,
            context,
            bubble_types=bubble_types,
        )

        resolved: dict[int, str] = {}
        failed: list[tuple[int, str]] = []
        any_non_english = False

        if on_bubble is not None and (self._llm_stream_caller is not None or self._llm_caller is None):
            resolved, failed, any_non_english = await self._run_primary_stream(
                bubbles, primary_messages, on_bubble
            )
        else:
            raw_primary = await self._invoke_llm(primary_messages, temperature=0.25, json_mode=True)
            parsed = _parse_translation_json(raw_primary)
            for bid, orig_text in bubbles:
                candidate = parsed.get(bid, "").strip()
                sanitized, ok = _sanitize_translation_candidate(orig_text, candidate)
                if ok:
                    resolved[bid] = sanitized
                    if on_bubble is not None:
                        await on_bubble(bid, sanitized, False)
                else:
                    if candidate and not is_refusal(candidate):
                        any_non_english = True
                    failed.append((bid, orig_text))
            if failed:
                logger.warning(
                    "Primary translation had %d failed/missing bubble(s) %s; raw=%r",
                    len(failed),
                    [b for b, _ in failed],
                    raw_primary,
                )

        # Stage 2: Batch retry failed/missing/refused/CJK IDs with explicit English-only framing at low temperature
        if failed:
            retry_reason = "non_english" if any_non_english else "missing_or_refusal"
            logger.warning(
                "Triggering batch retry (%s) for bubble(s) %s",
                retry_reason,
                [b for b, _ in failed],
            )
            retry_messages = self._build_retry_messages(
                failed, lang, merged_glossary, reason=retry_reason
            )
            raw_retry = await self._invoke_llm(retry_messages, temperature=0.1, json_mode=True)
            retry_parsed = _parse_translation_json(raw_retry)
            still_failed: list[tuple[int, str]] = []
            for bid, orig_text in failed:
                candidate = retry_parsed.get(bid, "").strip()
                sanitized, ok = _sanitize_translation_candidate(orig_text, candidate)
                if ok:
                    resolved[bid] = sanitized
                    if on_bubble is not None:
                        await on_bubble(bid, sanitized, True)
                else:
                    still_failed.append((bid, orig_text))

            if still_failed:
                logger.warning(
                    "Batch retry left %d bubble(s) unresolved %s; raw=%r",
                    len(still_failed),
                    [b for b, _ in still_failed],
                    raw_retry,
                )

            # Stage 3 & 4: Isolate stubborn bubbles via single-bubble JSON, then plain-text fallback
            for bid, orig_text in still_failed:
                single_candidate = await self._translate_single_bubble_english(
                    bid, orig_text, lang, merged_glossary
                )
                if single_candidate is not None:
                    resolved[bid] = single_candidate
                    if on_bubble is not None:
                        await on_bubble(bid, single_candidate, True)
                    continue

                logger.warning(
                    "Triggering 4th-stage plain-text fallback for bubble %s (orig=%r)",
                    bid,
                    orig_text,
                )
                plain_candidate = await self._translate_single_bubble_plain(
                    bid, orig_text, lang, merged_glossary
                )
                if plain_candidate is not None:
                    resolved[bid] = plain_candidate
                    if on_bubble is not None:
                        await on_bubble(bid, plain_candidate, True)
                else:
                    # Graceful fallback: never leak raw Japanese/CJK or refusal text
                    logger.warning(
                        "All translation stages failed for bubble %s (orig=%r); emitting empty string",
                        bid,
                        orig_text,
                    )
                    resolved[bid] = ""
                    if on_bubble is not None:
                        await on_bubble(bid, "", True)

        recorded_pairs: list[tuple[str, str]] = []
        for bid, orig_text in bubbles:
            trans = resolved[bid]
            if is_valid_english_translation(orig_text, trans):
                recorded_pairs.append((orig_text, trans))
        self._record_memory_page(chapter_id, recorded_pairs, series_id)

        return resolved

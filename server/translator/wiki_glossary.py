"""Character name glossary lookup via AniList GraphQL, Wikipedia, and Fandom APIs."""

from __future__ import annotations

import html
import re
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

import httpx

LookupSource = Literal["auto", "anilist", "wikipedia", "fandom"]

ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"
DEFAULT_WIKIPEDIA_HOST = "en.wikipedia.org"
DEFAULT_TIMEOUT = 10.0
USER_AGENT = "MangaLiveTranslator/1.3 (local glossary lookup)"

_VALID_SOURCES = frozenset({"auto", "anilist", "wikipedia", "fandom"})
_ANILIST_PAGE_HOSTS = frozenset({"anilist.co", "www.anilist.co"})
_ALLOWED_OUTBOUND_HOSTS = frozenset({"anilist.co", "www.anilist.co", "graphql.anilist.co"})
_DOMAIN_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_ANILIST_MANGA_PATH_RE = re.compile(r"^/manga/(\d+)(?:/.*)?$")
_WIKI_PATH_RE = re.compile(r"^(?:/([a-z]{2,3}(?:-[a-z]+)?))?/wiki/(.+)$")

_CJK_HANGUL_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u309f\u30a0-\u30ff\uff65-\uff9f"
    r"\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]"
)
_LATIN_RE = re.compile(r"[A-Za-z]")
_SPACE_RE = re.compile(r"[\s\u3000]+")

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_FULL_RE = re.compile(r"<ref\b[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_REF_SELF_RE = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]|\[\[([^\]]+)\]\]")
_HEADING_RE = re.compile(r"^\s*={2,6}\s*(.+?)\s*={2,6}\s*$")
_DEF_HEADING_RE = re.compile(r"^\s*;\s*(.+)$")

_NON_CHAR_SECTION_KEYWORDS = (
    "publication",
    "media",
    "anime",
    "manga",
    "plot",
    "synopsis",
    "premise",
    "production",
    "development",
    "reception",
    "awards",
    "references",
    "external links",
    "see also",
    "notes",
    "bibliography",
    "soundtrack",
    "video game",
    "stage play",
    "spin-off",
)
_ATTRIBUTION_PREFIX_RE = re.compile(
    r"(?:voiced|portrayed|played|acted|written|illustrated|directed|created|published|serialized|adapted)\s+(?:by|in)\s*:?\s*$",
    re.IGNORECASE,
)
_NON_PERSON_ENGLISH_RE = re.compile(
    r"\b(?:manga|anime|manhwa|webtoon|series|magazine|chapter|volume|episode|season|studio|"
    r"company|publishing|publisher|shueisha|kodansha|shogakukan|hakusensha|kadokawa|"
    r"viz media|crunchyroll|netflix|arc|saga|franchise|light novel|adaptation)\b",
    re.IGNORECASE,
)
_NATIVE_LABEL_PREFIX_RE = re.compile(
    r"^(?:japanese|jpn?|kanji|kana|korean|kor?|hangul|hanja|chinese|zh|traditional|simplified|native)\s*:\s*",
    re.IGNORECASE,
)
_ROMANIZATION_KEY_RE = re.compile(
    r"^(?:hepburn|romaji|rōmaji|rr|mr|lit\.?|literally|trans\.?|or)\b",
    re.IGNORECASE,
)
_INLINE_NAME_PAIR_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([A-Z][A-Za-z0-9'.-]*(?:-[A-Za-z0-9'.-]+)*"
    r"(?:\s+(?:[A-Z][A-Za-z0-9'.-]*(?:-[A-Za-z0-9'.-]+)*|of|the|de|da|do|la|le|van|von|al|bin|ibn)){0,4})"
    r"\s*\(([^()]{1,160})\)"
)
_TABLE_CELL_PAIR_RE = re.compile(r"^\|\s*([^|\n]+?)\s*\|\|\s*([^|\n]+)")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[*#:;]+|\d+\.)\s*")


class GlossaryLookupError(RuntimeError):
    """Raised when an upstream glossary source fails or returns invalid data."""


class InvalidLookupQueryError(ValueError):
    """Raised when the lookup query, source, or URL fails validation / SSRF checks."""


def _normalize_space(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _contains_cjk_or_hangul(text: str) -> bool:
    return bool(text and _CJK_HANGUL_RE.search(text))


def _contains_latin(text: str) -> bool:
    return bool(text and _LATIN_RE.search(text))


def _is_valid_english_name(name: str) -> bool:
    cleaned = _normalize_space(name)
    if not cleaned or len(cleaned) > 60:
        return False
    if any(ch in cleaned for ch in "{}[]|"):
        return False
    if _contains_cjk_or_hangul(cleaned) or not _contains_latin(cleaned):
        return False
    words = cleaned.split(" ")
    if len(words) > 6:
        return False
    if _NON_PERSON_ENGLISH_RE.search(cleaned):
        return False
    return True


def _is_valid_native_name(native: str) -> bool:
    cleaned = _normalize_space(native)
    if not cleaned or len(cleaned) > 40:
        return False
    if any(ch in cleaned for ch in "{}[]|"):
        return False
    return _contains_cjk_or_hangul(cleaned)


def _is_valid_hostname_labels(host: str) -> bool:
    labels = host.split(".")
    return bool(labels) and all(_DOMAIN_LABEL_RE.match(label) for label in labels)


def _classify_allowed_host(host: str) -> Literal["anilist", "wikipedia", "fandom"] | None:
    if not host or not _is_valid_hostname_labels(host):
        return None
    if host in _ALLOWED_OUTBOUND_HOSTS:
        return "anilist"
    if host.endswith(".wikipedia.org") and len(host) > len(".wikipedia.org"):
        return "wikipedia"
    if host.endswith(".fandom.com") and len(host) > len(".fandom.com"):
        return "fandom"
    return None


def _validate_and_route_url(url: str) -> tuple[Literal["anilist", "wikipedia", "fandom"], str, str]:
    """Validate HTTPS URL against strict host allowlist and return (source, host, path_or_id)."""
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise InvalidLookupQueryError("Only HTTPS URLs are allowed for glossary lookup.")
    if parsed.username is not None or parsed.password is not None:
        raise InvalidLookupQueryError("URLs containing credentials are not allowed.")
    if parsed.port is not None and parsed.port != 443:
        raise InvalidLookupQueryError("Non-standard URL ports are not allowed.")

    host = (parsed.hostname or "").lower().rstrip(".")
    detected = _classify_allowed_host(host)
    if detected is None:
        raise InvalidLookupQueryError(f"Host '{host or '<empty>'}' is not on the allowed glossary host list.")

    if detected == "anilist":
        if host not in _ANILIST_PAGE_HOSTS:
            raise InvalidLookupQueryError("AniList URLs must point to https://anilist.co/manga/<id>.")
        match = _ANILIST_MANGA_PATH_RE.match(parsed.path or "")
        if not match:
            raise InvalidLookupQueryError("AniList URL must match https://anilist.co/manga/<id>.")
        return "anilist", host, match.group(1)

    if detected == "wikipedia":
        m_suffix = ".m.wikipedia.org"
        api_host = host[: -len(m_suffix)] + ".wikipedia.org" if host.endswith(m_suffix) else host
        page_title = _extract_mediawiki_page_title(parsed.path, parsed.query)
        if not page_title:
            raise InvalidLookupQueryError("Wikipedia URL must include a /wiki/<Page> path or title parameter.")
        return "wikipedia", api_host, page_title

    # fandom
    page_title, lang_prefix = _extract_fandom_page_and_prefix(parsed.path, parsed.query)
    if not page_title:
        raise InvalidLookupQueryError("Fandom URL must include a /wiki/<Page> path or title parameter.")
    api_base = f"https://{host}/{lang_prefix}/api.php" if lang_prefix else f"https://{host}/api.php"
    return "fandom", api_base, page_title


def _extract_mediawiki_page_title(path: str, query: str) -> str:
    wiki_match = _WIKI_PATH_RE.match(path or "")
    if wiki_match:
        return _normalize_space(unquote(wiki_match.group(2)).replace("_", " "))
    if query:
        for part in query.split("&"):
            if part.startswith("title="):
                return _normalize_space(unquote(part[6:]).replace("_", " "))
    return ""


def _extract_fandom_page_and_prefix(path: str, query: str) -> tuple[str, str]:
    wiki_match = _WIKI_PATH_RE.match(path or "")
    if wiki_match:
        lang_prefix = wiki_match.group(1) or ""
        title = _normalize_space(unquote(wiki_match.group(2)).replace("_", " "))
        return title, lang_prefix
    if query:
        for part in query.split("&"):
            if part.startswith("title="):
                return _normalize_space(unquote(part[6:]).replace("_", " ")), ""
    return "", ""


def _clean_wikitext(text: str) -> str:
    """Strip MediaWiki markup, links, refs, and comments into readable plain text."""
    cleaned = _COMMENT_RE.sub("", text)
    cleaned = _REF_FULL_RE.sub("", cleaned)
    cleaned = _REF_SELF_RE.sub("", cleaned)
    cleaned = _BR_RE.sub("\n", cleaned)
    cleaned = re.sub(r"\{\{(?:ruby|furigana)\|([^|{}]+)\|[^|{}]+\}\}", r"\1", cleaned, flags=re.IGNORECASE)
    # Iteratively unwrap simple [[Target|Display]] and [[Link]]
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _WIKILINK_RE.sub(lambda m: m.group(2) if m.group(2) is not None else m.group(3), cleaned)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = cleaned.replace("'''", "").replace("''", "")
    return cleaned


def _split_template_parts(inner: str) -> list[str]:
    """Split MediaWiki template content on top-level pipes, respecting nested {{...}} and [[...]]."""
    parts: list[str] = []
    buf: list[str] = []
    brace_depth = 0
    bracket_depth = 0
    i = 0
    n = len(inner)
    while i < n:
        two = inner[i : i + 2]
        if two == "{{":
            brace_depth += 1
            buf.append(two)
            i += 2
            continue
        if two == "}}" and brace_depth > 0:
            brace_depth -= 1
            buf.append(two)
            i += 2
            continue
        if two == "[[":
            bracket_depth += 1
            buf.append(two)
            i += 2
            continue
        if two == "]]" and bracket_depth > 0:
            bracket_depth -= 1
            buf.append(two)
            i += 2
            continue
        ch = inner[i]
        if ch == "|" and brace_depth == 0 and bracket_depth == 0:
            parts.append("".join(buf))
            buf.clear()
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _iter_templates(wikitext: str) -> list[tuple[int, int, str]]:
    """Return (start, end, inner_content) for templates in wikitext."""
    results: list[tuple[int, int, str]] = []
    stack: list[int] = []
    i = 0
    n = len(wikitext)
    while i < n - 1:
        two = wikitext[i : i + 2]
        if two == "{{":
            stack.append(i)
            i += 2
            continue
        if two == "}}" and stack:
            start = stack.pop()
            results.append((start, i + 2, wikitext[start + 2 : i]))
            i += 2
            continue
        i += 1
    return results


def _parse_template_args(parts: list[str]) -> tuple[list[str], dict[str, str]]:
    positional: list[str] = []
    named: dict[str, str] = {}
    for raw in parts[1:]:
        if "=" in raw:
            k, v = raw.split("=", 1)
            key = k.strip().lower()
            if key.isdigit():
                idx = int(key) - 1
                while len(positional) <= idx:
                    positional.append("")
                positional[idx] = v.strip()
            else:
                named[key] = v.strip()
        else:
            positional.append(raw.strip())
    return positional, named


def _extract_primary_native(raw_inner: str) -> str:
    """Extract the primary CJK/Hangul name from a parenthetical or native field."""
    cleaned = _normalize_space(_clean_wikitext(raw_inner))
    if not cleaned:
        return ""
    cleaned = _NATIVE_LABEL_PREFIX_RE.sub("", cleaned)
    # Split on common separators between native script, reading, and romanization
    for sep_part in re.split(r"[,;／/]", cleaned):
        candidate = _normalize_space(sep_part)
        candidate = _NATIVE_LABEL_PREFIX_RE.sub("", candidate)
        if not candidate or _ROMANIZATION_KEY_RE.match(candidate):
            continue
        # Strip trailing romanized annotation inside sub-parens if any
        candidate = _normalize_space(re.sub(r"\([^()]*\)$", "", candidate))
        if _is_valid_native_name(candidate):
            return candidate
    return ""


def _add_entry(entries: dict[str, str], native: str, english: str) -> None:
    n = _normalize_space(native)
    e = _normalize_space(english)
    if _is_valid_native_name(n) and _is_valid_english_name(e) and n not in entries:
        entries[n] = e


def _extract_anilist_entries(nodes: list[dict[str, Any]]) -> dict[str, str]:
    """Extract clean native->English character pairs plus unambiguous single-token splits."""
    entries: dict[str, str] = {}
    two_part_pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []

    for node in nodes:
        if not isinstance(node, dict):
            continue
        name_obj = node.get("name") or {}
        if not isinstance(name_obj, dict):
            continue
        native = _normalize_space(str(name_obj.get("native") or ""))
        full = _normalize_space(str(name_obj.get("full") or ""))
        if not _is_valid_native_name(native) or not _is_valid_english_name(full):
            continue
        _add_entry(entries, native, full)

        native_parts = native.split(" ")
        full_parts = full.split(" ")
        if len(native_parts) == 2 and len(full_parts) == 2:
            two_part_pairs.append(((native_parts[0], native_parts[1]), (full_parts[0], full_parts[1])))

    # Only add split surname/given-name tokens when shared across >=2 characters
    # disambiguates forward vs. reverse name order without risking swapped names.
    if len(two_part_pairs) >= 2:
        reverse_first: dict[str, set[str]] = {}
        forward_first: dict[str, set[str]] = {}
        forward_second: dict[str, set[str]] = {}
        for (n0, n1), (e0, e1) in two_part_pairs:
            reverse_first.setdefault(n0, set()).add(e1)
            forward_first.setdefault(n0, set()).add(e0)
            forward_second.setdefault(n1, set()).add(e1)

        candidate_splits: dict[str, set[str]] = {}
        for (n0, n1), (e0, e1) in two_part_pairs:
            is_reverse = (
                sum(1 for (m0, _), (_, f1) in two_part_pairs if m0 == n0 and f1 == e1) >= 2
                and len(reverse_first.get(n0, set())) == 1
            )
            is_forward = (
                (
                    sum(1 for (m0, _), (f0, _) in two_part_pairs if m0 == n0 and f0 == e0) >= 2
                    and len(forward_first.get(n0, set())) == 1
                )
                or (
                    sum(1 for (_, m1), (_, f1) in two_part_pairs if m1 == n1 and f1 == e1) >= 2
                    and len(forward_second.get(n1, set())) == 1
                )
            )
            if is_reverse and not is_forward:
                candidate_splits.setdefault(n0, set()).add(e1)
                candidate_splits.setdefault(n1, set()).add(e0)
            elif is_forward and not is_reverse:
                candidate_splits.setdefault(n0, set()).add(e0)
                candidate_splits.setdefault(n1, set()).add(e1)

        for tok_native, eng_set in candidate_splits.items():
            if len(eng_set) == 1 and tok_native not in entries:
                _add_entry(entries, tok_native, next(iter(eng_set)))

    return entries


def _filter_character_sections_wikitext(wikitext: str) -> str:
    """Focus on character sections when present, or strip non-character sections otherwise."""
    lines = wikitext.splitlines()
    char_lines: list[str] = []
    in_char_section = False
    found_char_section = False
    char_level = 99

    for line in lines:
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(line.strip()) - len(line.strip().lstrip("="))
            title_lower = _clean_wikitext(heading_match.group(1)).strip().lower()
            is_char = any(kw in title_lower for kw in ("character", "cast"))
            is_non_char = any(kw in title_lower for kw in _NON_CHAR_SECTION_KEYWORDS)
            if is_char:
                found_char_section = True
                in_char_section = True
                char_level = level
                char_lines.append(line)
                continue
            if in_char_section:
                if level <= char_level or is_non_char:
                    in_char_section = False
                else:
                    char_lines.append(line)
                continue
        elif in_char_section:
            char_lines.append(line)

    if found_char_section and char_lines:
        return "\n".join(char_lines)

    # No explicit Characters/Cast heading (e.g. dedicated "List of ... characters" page):
    # strip clearly non-character sections like Plot/Publication/Reception/References.
    filtered_lines: list[str] = []
    skip_level = 99
    skipping = False
    for line in lines:
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(line.strip()) - len(line.strip().lstrip("="))
            title_lower = _clean_wikitext(heading_match.group(1)).strip().lower()
            is_non_char = any(kw in title_lower for kw in _NON_CHAR_SECTION_KEYWORDS)
            if is_non_char:
                skipping = True
                skip_level = level
                continue
            if skipping and level <= skip_level:
                skipping = False
        if not skipping:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)


def _clean_heading_candidate(raw: str) -> str:
    cleaned = _normalize_space(_clean_wikitext(raw))
    cleaned = _BULLET_PREFIX_RE.sub("", cleaned)
    # Strip trailing parenthetical if any
    cleaned = _normalize_space(re.sub(r"\s*\([^()]*\)$", "", cleaned))
    return cleaned


def _extract_nihongo_pair(
    tpl_name: str,
    parts: list[str],
    preceding_text: str,
    current_heading: str,
) -> tuple[str, str] | None:
    positional, named = _parse_template_args(parts)
    english = ""
    native_raw = ""

    if tpl_name in ("nihongo", "nihongo krt"):
        english = _clean_wikitext(positional[0]) if len(positional) >= 1 else ""
        native_raw = positional[1] if len(positional) >= 2 else ""
    elif tpl_name == "nihongo3":
        # {{nihongo3|English/extra|Kanji|Romaji}}
        english = _clean_wikitext(positional[0]) if len(positional) >= 1 else ""
        native_raw = positional[1] if len(positional) >= 2 else ""
        if not _is_valid_english_name(english) and len(positional) >= 3:
            english = _clean_wikitext(positional[2])
    elif tpl_name == "nihongo2":
        native_raw = positional[0] if positional else ""
    elif tpl_name in ("lang-ja", "lang-ko", "lang-zh", "lang-zh-hans", "lang-zh-hant"):
        native_raw = positional[0] if positional else ""
    elif tpl_name == "lang":
        native_raw = positional[1] if len(positional) >= 2 else ""
    elif tpl_name in ("korean", "ko-hhrm"):
        native_raw = named.get("hangul") or (positional[0] if positional else "")
    elif tpl_name in ("zh", "chinese"):
        native_raw = named.get("s") or named.get("t") or named.get("c") or (positional[0] if positional else "")

    native = _extract_primary_native(native_raw)
    if not native:
        return None

    english = _normalize_space(english)
    if not _is_valid_english_name(english):
        # Try extracting the English name directly before '(' in preceding text
        pre = _clean_wikitext(preceding_text)
        pre = _BULLET_PREFIX_RE.sub("", pre).strip()
        if pre.endswith("("):
            pre = pre[:-1].strip()
        if _ATTRIBUTION_PREFIX_RE.search(pre):
            return None
        # Take trailing phrase after punctuation or start of line
        m = re.search(
            r"([A-Z][A-Za-z0-9'.-]*(?:-[A-Za-z0-9'.-]+)*"
            r"(?:\s+(?:[A-Z][A-Za-z0-9'.-]*(?:-[A-Za-z0-9'.-]+)*|of|the|de|da|do|la|le|van|von|al|bin|ibn)){0,4})$",
            pre,
        )
        if m and not _ATTRIBUTION_PREFIX_RE.search(pre[: m.start()]):
            english = _normalize_space(m.group(1))
        elif not pre and _is_valid_english_name(current_heading):
            english = current_heading

    if _is_valid_english_name(english):
        return native, english
    return None


_SIMPLE_TEMPLATE_NAMES = frozenset(
    {
        "nihongo",
        "nihongo2",
        "nihongo3",
        "nihongo krt",
        "korean",
        "ko-hhrm",
        "zh",
        "chinese",
        "lang-ja",
        "lang-ko",
        "lang-zh",
        "lang-zh-hans",
        "lang-zh-hant",
        "lang",
    }
)


def _strip_all_templates(text: str) -> str:
    """Remove {{...}} templates cleanly from innermost to outermost."""
    out = text
    prev = None
    while prev != out:
        prev = out
        out = re.sub(r"\{\{[^{}]*\}\}", "", out)
    return out


def _extract_from_wikitext(wikitext: str, *, filter_sections: bool = True) -> dict[str, str]:
    entries: dict[str, str] = {}
    cleaned_raw = _COMMENT_RE.sub("", wikitext)
    cleaned_raw = _REF_FULL_RE.sub("", cleaned_raw)
    cleaned_raw = _REF_SELF_RE.sub("", cleaned_raw)
    scoped = _filter_character_sections_wikitext(cleaned_raw) if filter_sections else cleaned_raw

    current_heading = ""
    for line in scoped.splitlines():
        heading_m = _HEADING_RE.match(line)
        if heading_m:
            cand = _clean_heading_candidate(heading_m.group(1))
            if _is_valid_english_name(cand):
                current_heading = cand
            else:
                current_heading = ""
            continue

        def_m = _DEF_HEADING_RE.match(line)
        if def_m:
            cand = _clean_heading_candidate(def_m.group(1).split("(", 1)[0])
            if _is_valid_english_name(cand):
                current_heading = cand

        # 1) Template-based extraction ({{nihongo|...}}, {{Korean|...}}, {{lang-ja|...}}, etc.)
        for start, end, inner in _iter_templates(line):
            parts = _split_template_parts(inner)
            if not parts:
                continue
            tpl_name = parts[0].strip().lower().replace("_", " ")
            if tpl_name in _SIMPLE_TEMPLATE_NAMES:
                pair = _extract_nihongo_pair(tpl_name, parts, line[:start], current_heading)
                if pair is not None:
                    _add_entry(entries, pair[0], pair[1])

        # 2) Wiki table row "| [[Name]] || 漢字"
        table_m = _TABLE_CELL_PAIR_RE.match(line)
        if table_m:
            left = _normalize_space(_clean_wikitext(table_m.group(1)))
            right = _extract_primary_native(table_m.group(2))
            if right and _is_valid_english_name(left):
                _add_entry(entries, right, left)

        # 3) Parenthetical patterns on cleaned line: "English Name (Japanese: 漢字)"
        plain_line = _clean_wikitext(_strip_all_templates(line))
        _extract_inline_pairs_from_line(plain_line, entries, current_heading=current_heading)

    return entries


def _extract_inline_pairs_from_line(
    line: str,
    entries: dict[str, str],
    *,
    current_heading: str = "",
) -> None:
    stripped = _BULLET_PREFIX_RE.sub("", line).strip()
    if not stripped:
        return

    # Line starting directly with "(Japanese: 漢字)" under a character heading
    if stripped.startswith("(") and current_heading and _is_valid_english_name(current_heading):
        close_idx = stripped.find(")")
        if close_idx > 1:
            native = _extract_primary_native(stripped[1:close_idx])
            if native:
                _add_entry(entries, native, current_heading)

    for m in _INLINE_NAME_PAIR_RE.finditer(stripped):
        prefix = stripped[: m.start()]
        if _ATTRIBUTION_PREFIX_RE.search(prefix):
            continue
        english = _normalize_space(m.group(1))
        native = _extract_primary_native(m.group(2))
        if native and _is_valid_english_name(english):
            _add_entry(entries, native, english)


_BLOCK_BREAK_TAG_RE = re.compile(
    r"</?(?:p|div|li|dt|dd|tr|td|th|h[1-6]|blockquote|section|article|br)\b[^>]*>",
    re.IGNORECASE,
)


def _extract_from_plain_or_html(text: str) -> dict[str, str]:
    """Extract parenthetical English (CJK/Hangul) pairs from HTML or plain text extracts."""
    entries: dict[str, str] = {}
    plain = _BLOCK_BREAK_TAG_RE.sub("\n", text)
    plain = _HTML_TAG_RE.sub("", plain)
    plain = html.unescape(plain)
    prev_heading = ""
    for line in plain.splitlines():
        cleaned = _normalize_space(line)
        if not cleaned:
            continue
        _extract_inline_pairs_from_line(cleaned, entries, current_heading=prev_heading)
        if not cleaned.startswith("(") and _is_valid_english_name(cleaned):
            prev_heading = cleaned
        else:
            prev_heading = ""
    return entries


def _extract_fandom_infobox_entries(wikitext: str, page_title: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    cleaned_raw = _COMMENT_RE.sub("", wikitext)
    cleaned_raw = _REF_FULL_RE.sub("", cleaned_raw)
    cleaned_raw = _REF_SELF_RE.sub("", cleaned_raw)

    native_keys = (
        "kanji",
        "japanese",
        "korean",
        "hangul",
        "native",
        "native_name",
        "kanji_name",
        "jp",
        "jpname",
        "kr",
        "krname",
        "hanja",
    )
    english_keys = ("name", "title", "english", "english_name", "romaji")
    default_title = _normalize_space(re.sub(r"\s*\([^()]*\)$", "", page_title))

    for _, _, inner in _iter_templates(cleaned_raw):
        parts = _split_template_parts(inner)
        if len(parts) < 2:
            continue
        _, named = _parse_template_args(parts)
        if not any(k in named for k in native_keys):
            continue

        english = ""
        for ek in english_keys:
            raw_val = named.get(ek, "")
            if not raw_val:
                continue
            cleaned_lines = [ln.strip() for ln in _clean_wikitext(raw_val).splitlines() if ln.strip()]
            for ln in cleaned_lines:
                cand = _normalize_space(ln)
                if _is_valid_english_name(cand):
                    english = cand
                    break
            if english:
                break
        if not english and _is_valid_english_name(default_title):
            english = default_title
        if not english:
            continue

        for nk in native_keys:
            raw_native = named.get(nk, "")
            if not raw_native:
                continue
            # Use first non-empty line before <br> so alternate forms/epithets aren't mislabeled
            cleaned_native_lines = [ln.strip() for ln in _clean_wikitext(raw_native).splitlines() if ln.strip()]
            for ln in cleaned_native_lines:
                native = _extract_primary_native(ln)
                if native:
                    _add_entry(entries, native, english)
                    break

    return entries


async def _get_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    allow_404_json: bool = False,
) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    try:
        response = await client.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=headers,
        )
    except httpx.HTTPError as exc:
        raise GlossaryLookupError(f"Request to {url} failed: {exc}") from exc

    if response.status_code == 404 and allow_404_json:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        except ValueError:
            pass

    if response.status_code >= 400:
        raise GlossaryLookupError(f"Upstream HTTP {response.status_code} from {url}.")

    try:
        data = response.json()
    except ValueError as exc:
        raise GlossaryLookupError(f"Invalid JSON response from {url}.") from exc
    if not isinstance(data, dict):
        raise GlossaryLookupError(f"Unexpected JSON payload type from {url}.")
    return data


async def _lookup_anilist(
    client: httpx.AsyncClient,
    query_or_id: str,
    *,
    by_id: bool,
    original_url: str | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    if by_id:
        gql = """
        query ($id: Int) {
          Media(id: $id, type: MANGA) {
            title { romaji english native }
            siteUrl
            characters(sort: [ROLE, RELEVANCE], perPage: 30) {
              nodes {
                name { full native alternative }
              }
            }
          }
        }
        """
        variables: dict[str, Any] = {"id": int(query_or_id)}
    else:
        gql = """
        query ($search: String) {
          Media(search: $search, type: MANGA) {
            title { romaji english native }
            siteUrl
            characters(sort: [ROLE, RELEVANCE], perPage: 30) {
              nodes {
                name { full native alternative }
              }
            }
          }
        }
        """
        variables = {"search": query_or_id}

    data = await _get_json(
        client,
        "POST",
        ANILIST_GRAPHQL_URL,
        json_body={"query": gql, "variables": variables},
        allow_404_json=True,
    )
    media = (data.get("data") or {}).get("Media")
    if not media:
        errors = data.get("errors") or []
        is_not_found = any(isinstance(e, dict) and e.get("status") == 404 for e in errors)
        if errors and not is_not_found and not by_id:
            msg = errors[0].get("message", "AniList GraphQL error") if isinstance(errors[0], dict) else str(errors[0])
            raise GlossaryLookupError(f"AniList error: {msg}")
        warnings.append("AniList returned no matching manga.")
        return {
            "title": query_or_id,
            "source": "anilist",
            "url": original_url or f"https://anilist.co/manga/{query_or_id}" if by_id else "https://anilist.co",
            "entries": {},
            "count": 0,
            "warnings": warnings,
        }

    title_obj = media.get("title") or {}
    title = str(
        title_obj.get("english")
        or title_obj.get("romaji")
        or title_obj.get("native")
        or query_or_id
    )
    site_url = str(media.get("siteUrl") or original_url or "https://anilist.co")
    nodes = ((media.get("characters") or {}).get("nodes")) or []
    entries = _extract_anilist_entries(nodes)
    if not entries:
        warnings.append(f"AniList entry '{title}' had no CJK/Hangul character names.")

    return {
        "title": title,
        "source": "anilist",
        "url": site_url,
        "entries": entries,
        "count": len(entries),
        "warnings": warnings,
    }


async def _lookup_wikipedia(
    client: httpx.AsyncClient,
    query_or_title: str,
    *,
    host: str = DEFAULT_WIKIPEDIA_HOST,
    is_direct_page: bool = False,
    original_url: str | None = None,
) -> dict[str, Any]:
    api_url = f"https://{host}/w/api.php"
    warnings: list[str] = []
    page_title = query_or_title

    if not is_direct_page:
        search_data = await _get_json(
            client,
            "GET",
            api_url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query_or_title,
                "srlimit": 5,
                "format": "json",
            },
        )
        hits = ((search_data.get("query") or {}).get("search")) or []
        if not hits:
            warnings.append(f"Wikipedia returned no search results for '{query_or_title}'.")
            return {
                "title": query_or_title,
                "source": "wikipedia",
                "url": f"https://{host}/",
                "entries": {},
                "count": 0,
                "warnings": warnings,
            }
        # Prefer dedicated "List of ... characters" page when present in top results
        chosen = hits[0]
        for hit in hits:
            t_lower = str(hit.get("title") or "").lower()
            if "character" in t_lower and "list of" in t_lower:
                chosen = hit
                break
        page_title = str(chosen.get("title") or query_or_title)

    parse_data = await _get_json(
        client,
        "GET",
        api_url,
        params={
            "action": "parse",
            "page": page_title,
            "prop": "wikitext|text",
            "redirects": "1",
            "format": "json",
        },
    )
    if "error" in parse_data:
        info = (parse_data["error"] or {}).get("info", "Wikipedia parse error")
        raise GlossaryLookupError(f"Wikipedia error: {info}")

    parse_obj = parse_data.get("parse") or {}
    resolved_title = str(parse_obj.get("title") or page_title)
    wikitext = ((parse_obj.get("wikitext") or {}).get("*")) or ""
    html_text = ((parse_obj.get("text") or {}).get("*")) or ""

    entries: dict[str, str] = {}
    if wikitext:
        entries = _extract_from_wikitext(wikitext, filter_sections=True)
    if not entries and html_text:
        entries = _extract_from_plain_or_html(html_text)

    if not entries:
        warnings.append(f"No character name pairs found on Wikipedia page '{resolved_title}'.")

    canonical_url = original_url or f"https://{host}/wiki/{resolved_title.replace(' ', '_')}"
    return {
        "title": resolved_title,
        "source": "wikipedia",
        "url": canonical_url,
        "entries": entries,
        "count": len(entries),
        "warnings": warnings,
    }


async def _lookup_fandom(
    client: httpx.AsyncClient,
    api_url: str,
    page_title: str,
    *,
    original_url: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    parse_data = await _get_json(
        client,
        "GET",
        api_url,
        params={
            "action": "parse",
            "page": page_title,
            "prop": "wikitext",
            "redirects": "1",
            "format": "json",
        },
    )
    if "error" in parse_data:
        info = (parse_data["error"] or {}).get("info", "Fandom parse error")
        raise GlossaryLookupError(f"Fandom error: {info}")

    parse_obj = parse_data.get("parse") or {}
    resolved_title = str(parse_obj.get("title") or page_title)
    wikitext = ((parse_obj.get("wikitext") or {}).get("*")) or ""

    entries = _extract_fandom_infobox_entries(wikitext, resolved_title)
    for k, v in _extract_from_wikitext(wikitext, filter_sections=False).items():
        if k not in entries:
            entries[k] = v

    if not entries:
        warnings.append(f"No character name pairs found on Fandom page '{resolved_title}'.")

    return {
        "title": resolved_title,
        "source": "fandom",
        "url": original_url,
        "entries": entries,
        "count": len(entries),
        "warnings": warnings,
    }


async def _execute_lookup(
    client: httpx.AsyncClient,
    cleaned_query: str,
    source: LookupSource,
) -> dict[str, Any]:
    is_url = bool(_URL_SCHEME_RE.match(cleaned_query))
    if is_url:
        detected_source, host_or_api, target = _validate_and_route_url(cleaned_query)
        if source != "auto" and source != detected_source:
            raise InvalidLookupQueryError(
                f"URL belongs to '{detected_source}', which does not match source='{source}'."
            )
        if detected_source == "anilist":
            return await _lookup_anilist(client, target, by_id=True, original_url=cleaned_query)
        if detected_source == "wikipedia":
            return await _lookup_wikipedia(
                client,
                target,
                host=host_or_api,
                is_direct_page=True,
                original_url=cleaned_query,
            )
        return await _lookup_fandom(client, host_or_api, target, original_url=cleaned_query)

    # Plain title query
    if source == "fandom":
        raise InvalidLookupQueryError("Fandom lookup requires a full https://<wiki>.fandom.com/wiki/... URL.")
    if source == "anilist":
        return await _lookup_anilist(client, cleaned_query, by_id=False)
    if source == "wikipedia":
        return await _lookup_wikipedia(client, cleaned_query, host=DEFAULT_WIKIPEDIA_HOST, is_direct_page=False)

    # source == "auto": try AniList first, fall back to English Wikipedia if 0 hits or error
    warnings: list[str] = []
    try:
        anilist_result = await _lookup_anilist(client, cleaned_query, by_id=False)
        if anilist_result["count"] >= 1:
            return anilist_result
        warnings.extend(anilist_result["warnings"])
        warnings.append("Falling back to English Wikipedia search.")
        anilist_err: Exception | None = None
    except GlossaryLookupError as exc:
        anilist_err = exc
        warnings.append(f"AniList lookup failed ({exc}); falling back to English Wikipedia.")

    try:
        wiki_result = await _lookup_wikipedia(
            client,
            cleaned_query,
            host=DEFAULT_WIKIPEDIA_HOST,
            is_direct_page=False,
        )
    except GlossaryLookupError as wiki_exc:
        if anilist_err is not None:
            raise GlossaryLookupError(
                f"AniList lookup failed ({anilist_err}) and Wikipedia fallback failed ({wiki_exc})."
            ) from wiki_exc
        raise

    wiki_result["warnings"] = warnings + list(wiki_result.get("warnings") or [])
    return wiki_result


async def lookup_character_glossary(
    query: str,
    source: LookupSource = "auto",
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Look up CJK/Hangul -> English character name pairs from AniList, Wikipedia, or Fandom."""
    if source not in _VALID_SOURCES:
        raise InvalidLookupQueryError(f"Unsupported glossary source: {source!r}")
    if not isinstance(query, str):
        raise InvalidLookupQueryError("Lookup query must be a non-empty string.")
    cleaned_query = query.strip()
    if not cleaned_query:
        raise InvalidLookupQueryError("Lookup query must not be empty.")

    if client is not None:
        return await _execute_lookup(client, cleaned_query, source)

    async with httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
    ) as owned_client:
        return await _execute_lookup(owned_client, cleaned_query, source)

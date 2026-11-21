"""Per-chapter rolling dialogue context and pinned character-name glossary store."""

from __future__ import annotations

import re
import threading
from collections import deque

_CJK_SCRIPT_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u309f\u30a0-\u30ff\uff65-\uff9f"
    r"\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]"
)


def contains_non_english_cjk(text: str) -> bool:
    """Return True if text contains Chinese, Japanese (Kana/Kanji), or Korean (Hangul) characters."""
    if not text:
        return False
    return _CJK_SCRIPT_RE.search(text) is not None


def _normalize_series_id(series_id: str | None) -> str | None:
    if series_id is None:
        return None
    cleaned = str(series_id).strip()
    return cleaned or None


def _clean_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    cleaned: list[tuple[str, str]] = []
    for orig, trans in pairs:
        o = str(orig).strip()
        t = str(trans).strip()
        if o and t and not contains_non_english_cjk(t):
            cleaned.append((o, t))
    return cleaned


class ChapterMemoryStore:
    """Thread-safe rolling multi-page dialogue history and pinned glossary per chapter/series."""

    def __init__(self, max_recent: int = 36) -> None:
        if max_recent < 0:
            raise ValueError("max_recent must be non-negative")
        self.max_recent = max_recent
        self._lock = threading.RLock()
        self._contexts: dict[str, deque[tuple[str, str]]] = {}
        self._pages: dict[str, deque[list[tuple[str, str]]]] = {}
        self._series_contexts: dict[str, deque[tuple[str, str]]] = {}
        self._series_pages: dict[str, deque[tuple[str, list[tuple[str, str]]]]] = {}
        self._glossaries: dict[str, dict[str, str]] = {}
        self._series_glossaries: dict[str, dict[str, str]] = {}

    def _prune_chapter_pages(self, pages: deque[list[tuple[str, str]]]) -> None:
        total = sum(len(p) for p in pages)
        while pages and total > self.max_recent:
            overflow = total - self.max_recent
            oldest = pages[0]
            if len(oldest) <= overflow:
                pages.popleft()
                total -= len(oldest)
            else:
                pages[0] = oldest[overflow:]
                total -= overflow

    def _prune_series_pages(self, pages: deque[tuple[str, list[tuple[str, str]]]]) -> None:
        total = sum(len(p) for _, p in pages)
        while pages and total > self.max_recent:
            overflow = total - self.max_recent
            rec_ch, oldest = pages[0]
            if len(oldest) <= overflow:
                pages.popleft()
                total -= len(oldest)
            else:
                pages[0] = (rec_ch, oldest[overflow:])
                total -= overflow

    def get_context(
        self,
        chapter_id: str,
        series_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """Return recent (original, translated) English dialogue pairs for a chapter/series."""
        with self._lock:
            buf = self._contexts.get(chapter_id)
            chapter_pairs = (
                [(o, t) for o, t in buf if o and t and not contains_non_english_cjk(t)]
                if buf is not None
                else []
            )
            sid = _normalize_series_id(series_id)
            if sid is None or self.max_recent == 0:
                return chapter_pairs
            if len(chapter_pairs) >= self.max_recent:
                return chapter_pairs[-self.max_recent :]

            prior_pairs: list[tuple[str, str]] = []
            series_pages = self._series_pages.get(sid)
            if series_pages:
                for rec_ch, page_pairs in series_pages:
                    if rec_ch == chapter_id:
                        continue
                    prior_pairs.extend(
                        (o, t) for o, t in page_pairs if o and t and not contains_non_english_cjk(t)
                    )
            elif not chapter_pairs:
                series_buf = self._series_contexts.get(sid)
                if series_buf:
                    prior_pairs.extend(
                        (o, t) for o, t in series_buf if o and t and not contains_non_english_cjk(t)
                    )

            combined = prior_pairs + chapter_pairs
            return combined[-self.max_recent :]

    def get_formatted_context(
        self,
        chapter_id: str,
        series_id: str | None = None,
        max_pairs: int = 28,
    ) -> str:
        """Return page-grouped recent dialogue lines ([Page -2], [Page -1], ...) across pages/chapters."""
        if max_pairs <= 0:
            return ""
        with self._lock:
            chapter_pages: list[list[tuple[str, str]]] = []
            raw_ch_pages = self._pages.get(chapter_id)
            if raw_ch_pages:
                for page_pairs in raw_ch_pages:
                    cleaned = [
                        (o, t) for o, t in page_pairs if o and t and not contains_non_english_cjk(t)
                    ]
                    if cleaned:
                        chapter_pages.append(cleaned)
            elif chapter_id in self._contexts:
                buf_pairs = [
                    (o, t)
                    for o, t in self._contexts[chapter_id]
                    if o and t and not contains_non_english_cjk(t)
                ]
                if buf_pairs:
                    chapter_pages.append(buf_pairs)

            prior_pages: list[list[tuple[str, str]]] = []
            sid = _normalize_series_id(series_id)
            if sid is not None:
                series_pages = self._series_pages.get(sid)
                if series_pages:
                    for rec_ch, page_pairs in series_pages:
                        if rec_ch == chapter_id:
                            continue
                        cleaned = [
                            (o, t)
                            for o, t in page_pairs
                            if o and t and not contains_non_english_cjk(t)
                        ]
                        if cleaned:
                            prior_pages.append(cleaned)

            all_pages = prior_pages + chapter_pages
            if not all_pages:
                return ""

            selected: list[list[tuple[str, str]]] = []
            remaining = max_pairs
            for page_pairs in reversed(all_pages):
                if remaining <= 0:
                    break
                if len(page_pairs) <= remaining:
                    selected.append(page_pairs)
                    remaining -= len(page_pairs)
                else:
                    selected.append(page_pairs[-remaining:])
                    remaining = 0

            selected.reverse()
            total_pages = len(selected)
            blocks: list[str] = []
            for idx, page_pairs in enumerate(selected):
                offset = total_pages - idx
                lines = [f"[Page -{offset}]"]
                lines.extend(f"- {orig} -> {trans}" for orig, trans in page_pairs)
                blocks.append("\n".join(lines))
            return "\n".join(blocks)

    def get_glossary(
        self,
        chapter_id: str,
        series_id: str | None = None,
    ) -> dict[str, str]:
        """Return pinned glossary mapping merging series-level defaults with chapter-level entries."""
        with self._lock:
            merged: dict[str, str] = {}
            sid = _normalize_series_id(series_id)
            if sid is not None:
                series_entries = self._series_glossaries.get(sid)
                if series_entries is not None:
                    merged.update(series_entries)
            chapter_entries = self._glossaries.get(chapter_id)
            if chapter_entries is not None:
                merged.update(chapter_entries)
            return merged

    def set_glossary(
        self,
        chapter_id: str,
        entries: dict[str, str],
        series_id: str | None = None,
    ) -> None:
        """Set cleaned non-empty glossary entries for a chapter (and series when provided)."""
        cleaned: dict[str, str] = {}
        for key, value in entries.items():
            k = str(key).strip()
            v = str(value).strip()
            if k and v:
                cleaned[k] = v
        with self._lock:
            self._glossaries[chapter_id] = dict(cleaned)
            sid = _normalize_series_id(series_id)
            if sid is not None:
                self._series_glossaries[sid] = dict(cleaned)

    def record_page(
        self,
        chapter_id: str,
        pairs: list[tuple[str, str]],
        series_id: str | None = None,
    ) -> None:
        """Append English-only translated dialogue pairs to the chapter and series rolling windows."""
        if self.max_recent == 0 or not pairs:
            return
        cleaned = _clean_pairs(pairs)
        if not cleaned:
            return
        with self._lock:
            buf = self._contexts.get(chapter_id)
            if buf is None:
                buf = deque(maxlen=self.max_recent)
                self._contexts[chapter_id] = buf
            buf.extend(cleaned)

            pages = self._pages.get(chapter_id)
            if pages is None:
                pages = deque()
                self._pages[chapter_id] = pages
            pages.append(list(cleaned[-self.max_recent :]))
            self._prune_chapter_pages(pages)

            sid = _normalize_series_id(series_id)
            if sid is not None:
                s_buf = self._series_contexts.get(sid)
                if s_buf is None:
                    s_buf = deque(maxlen=self.max_recent)
                    self._series_contexts[sid] = s_buf
                s_buf.extend(cleaned)

                s_pages = self._series_pages.get(sid)
                if s_pages is None:
                    s_pages = deque()
                    self._series_pages[sid] = s_pages
                s_pages.append((chapter_id, list(cleaned[-self.max_recent :])))
                self._prune_series_pages(s_pages)

    def clear(
        self,
        chapter_id: str,
        series_id: str | None = None,
    ) -> None:
        """Clear rolling dialogue context for a chapter (and optional series), preserving glossaries."""
        with self._lock:
            self._contexts.pop(chapter_id, None)
            self._pages.pop(chapter_id, None)
            sid = _normalize_series_id(series_id)
            if sid is not None:
                self._series_contexts.pop(sid, None)
                self._series_pages.pop(sid, None)

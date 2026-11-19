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


class ChapterMemoryStore:
    """Thread-safe rolling dialogue history and pinned glossary per chapter."""

    def __init__(self, max_recent: int = 10) -> None:
        if max_recent < 0:
            raise ValueError("max_recent must be non-negative")
        self.max_recent = max_recent
        self._lock = threading.RLock()
        self._contexts: dict[str, deque[tuple[str, str]]] = {}
        self._glossaries: dict[str, dict[str, str]] = {}

    def get_context(self, chapter_id: str) -> list[tuple[str, str]]:
        """Return recent (original, translated) English dialogue pairs for a chapter."""
        with self._lock:
            buf = self._contexts.get(chapter_id)
            if buf is None:
                return []
            return [(o, t) for o, t in buf if o and t and not contains_non_english_cjk(t)]

    def get_glossary(self, chapter_id: str) -> dict[str, str]:
        """Return a shallow copy of the pinned glossary mapping for a chapter."""
        with self._lock:
            entries = self._glossaries.get(chapter_id)
            return dict(entries) if entries is not None else {}

    def set_glossary(self, chapter_id: str, entries: dict[str, str]) -> None:
        """Set cleaned non-empty glossary entries for a chapter."""
        cleaned: dict[str, str] = {}
        for key, value in entries.items():
            k = str(key).strip()
            v = str(value).strip()
            if k and v:
                cleaned[k] = v
        with self._lock:
            self._glossaries[chapter_id] = cleaned

    def record_page(self, chapter_id: str, pairs: list[tuple[str, str]]) -> None:
        """Append English-only translated dialogue pairs to the chapter's rolling window."""
        if self.max_recent == 0 or not pairs:
            return
        with self._lock:
            buf = self._contexts.get(chapter_id)
            if buf is None:
                buf = deque(maxlen=self.max_recent)
                self._contexts[chapter_id] = buf
            for orig, trans in pairs:
                o = str(orig).strip()
                t = str(trans).strip()
                if o and t and not contains_non_english_cjk(t):
                    buf.append((o, t))

    def clear(self, chapter_id: str) -> None:
        """Clear rolling dialogue context for a chapter (preserving pinned glossary)."""
        with self._lock:
            self._contexts.pop(chapter_id, None)

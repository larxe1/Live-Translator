"""SQLite-backed LRU translation cache keyed by image bytes, lang, and glossary."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from server.schemas import TranslateResponse


class TranslationCache:
    """Persistent SHA-256 translation cache with bounded LRU eviction."""

    def __init__(self, db_path: Path, max_entries: int = 2000) -> None:
        self.db_path = Path(db_path)
        self.max_entries = max(0, int(max_entries))
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _init_db(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    last_accessed INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_translations_last_accessed
                ON translations (last_accessed ASC)
                """
            )
            conn.commit()

    @staticmethod
    def make_key(
        image_bytes: bytes,
        lang: str,
        glossary: dict[str, str] | None = None,
    ) -> str:
        """Compute deterministic SHA-256 key over image bytes, language, and glossary."""
        normalized_glossary = json.dumps(
            glossary or {},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        hasher = hashlib.sha256()
        hasher.update(image_bytes)
        hasher.update(b"\x00")
        hasher.update(lang.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(normalized_glossary.encode("utf-8"))
        return hasher.hexdigest()

    def _next_seq(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(last_accessed), 0) + 1 FROM translations"
        ).fetchone()
        return int(row[0]) if row else 1

    def get(self, key: str) -> TranslateResponse | None:
        """Fetch cached response and bump its LRU timestamp; returns None on miss."""
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT payload FROM translations WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None

            seq = self._next_seq(conn)
            conn.execute(
                "UPDATE translations SET last_accessed = ? WHERE cache_key = ?",
                (seq, key),
            )
            conn.commit()
            payload = row[0]

        response = TranslateResponse.model_validate_json(payload)
        return response.model_copy(update={"cached": True})

    def put(self, key: str, response: TranslateResponse) -> None:
        """Store response in SQLite and prune least-recently-used entries if over limit."""
        if self.max_entries == 0:
            return

        payload = response.model_dump_json()
        with self._lock, closing(self._connect()) as conn:
            seq = self._next_seq(conn)
            conn.execute(
                """
                INSERT INTO translations (cache_key, payload, last_accessed)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    last_accessed = excluded.last_accessed
                """,
                (key, payload, seq),
            )
            count_row = conn.execute("SELECT COUNT(*) FROM translations").fetchone()
            total = int(count_row[0]) if count_row else 0
            overflow = total - self.max_entries
            if overflow > 0:
                conn.execute(
                    """
                    DELETE FROM translations
                    WHERE cache_key IN (
                        SELECT cache_key FROM translations
                        ORDER BY last_accessed ASC
                        LIMIT ?
                    )
                    """,
                    (overflow,),
                )
            conn.commit()

    def clear(self) -> None:
        """Remove all entries from the persistent cache."""
        with self._lock, closing(self._connect()) as conn:
            conn.execute("DELETE FROM translations")
            conn.commit()

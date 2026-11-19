"""Unit tests for refusal detection, chapter memory, JSON translation repair, and llama manager."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing

import pytest

from server.cache import CACHE_KEY_VERSION, TranslationCache
from server.config import AppConfig
from server.pipeline import TranslationPipeline
from server.schemas import Box, BubbleResult, TranslateResponse
from server.translator.llama_manager import LlamaServerManager
from server.translator.memory import ChapterMemoryStore
from server.translator.translator import (
    DialogueTranslator,
    _parse_translation_json,
    contains_non_english_cjk,
    is_refusal,
    is_valid_english_translation,
    romanize_residual_kana,
)


def test_refusal_detector_catches_safety_boilerplate_without_false_positives():
    assert is_refusal("I'm sorry, I cannot translate explicit or violent content.") is True
    assert is_refusal("As an AI language model, I must decline.") is True
    assert is_refusal("I cannot translate profanity.") is True
    assert is_refusal("This request violates my content policy.") is True
    assert is_refusal("I am unable to translate NSFW content.") is True

    # Genuine dialogue, profanity, screams, and dramatic lines MUST NOT trigger refusal
    assert is_refusal("Get the fuck off me! A centipede!!") is False
    assert is_refusal("GYAAAAAH!") is False
    assert is_refusal("I can't move!") is False
    assert is_refusal("I cannot forgive you for what you did!") is False
    assert is_refusal("I'm sorry! Please don't kill me!") is False


def test_chapter_memory_store_rolling_window_glossary_and_clear():
    mem = ChapterMemoryStore(max_recent=3)
    mem.set_glossary("ch1", {"藍星": "Ranshin", "  ": "ignored"})
    assert mem.get_glossary("ch1") == {"藍星": "Ranshin"}

    mem.record_page("ch1", [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")])
    assert mem.get_context("ch1") == [("b", "B"), ("c", "C"), ("d", "D")]
    assert mem.get_context("ch2") == []

    mem.clear("ch1")
    assert mem.get_context("ch1") == []
    # Pinned glossary stays intact when clearing rolling dialogue history
    assert mem.get_glossary("ch1") == {"藍星": "Ranshin"}


def test_parse_translation_json_handles_fences_and_truncated_json():
    fenced = '```json\n{"0": "Ranshin...?", "1": "Hold on!"}\n```'
    assert _parse_translation_json(fenced) == {0: "Ranshin...?", 1: "Hold on!"}

    truncated = '{"0": "Get back!", "1": "A giant centipede!", "2": "unterminated'
    parsed = _parse_translation_json(truncated)
    assert parsed[0] == "Get back!"
    assert parsed[1] == "A giant centipede!"


@pytest.mark.asyncio
async def test_translator_parses_page_json_and_repairs_missing_or_refused_bubble():
    memory = ChapterMemoryStore()
    memory.set_glossary("ch1", {"藍星": "Ranshin"})
    calls: list[list[dict[str, str]]] = []

    async def fake_llm(messages):
        calls.append(messages)
        if len(calls) == 1:
            # Simulates missing id=2 AND a refusal on id=1
            return '{"0": "Ranshin...?", "1": "I cannot translate profanity."}'
        return '{"1": "Shit! A centipede!", "2": "GYAAAAAH!"}'

    tr = DialogueTranslator("http://127.0.0.1:8766", memory, llm_caller=fake_llm)
    out = await tr.translate_page(
        [(0, "藍星……？"), (1, "くそっ！蜈蚣が！"), (2, "ぎゃああああっ")],
        lang="ja",
        chapter_id="ch1",
    )
    assert out[0] == "Ranshin...?"
    assert not is_refusal(out[1]) and "centipede" in out[1].lower()
    assert out[2] == "GYAAAAAH!"
    assert len(calls) == 2
    # Retry prompt should include ONLY failed IDs (1 and 2), not already-succeeded ID 0
    assert '"0"' not in calls[1][1]["content"]
    assert '"1"' in calls[1][1]["content"] and '"2"' in calls[1][1]["content"]
    assert len(memory.get_context("ch1")) == 3


@pytest.mark.asyncio
async def test_translator_never_returns_refusal_even_if_retry_refuses():
    memory = ChapterMemoryStore()

    async def always_refuse(_messages):
        return '{"0": "As an AI language model, I cannot translate explicit content."}'

    tr = DialogueTranslator("http://127.0.0.1:8766", memory, llm_caller=always_refuse)
    out = await tr.translate_page([(0, "くそったれ！")], lang="ja", chapter_id="ch1")
    assert not is_refusal(out[0])
    assert out[0] == ""


def test_llama_manager_model_resolution_and_vram_capped_command(tmp_path):
    models_dir = tmp_path / "models"
    bin_dir = tmp_path / "bin" / "llama-cpp"
    models_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)

    gguf = models_dir / "Qwen2.5-7B-Instruct-abliterated-Q4_K_M.gguf"
    gguf.write_bytes(b"fake-gguf")
    exe = bin_dir / "llama-server.exe"
    exe.write_bytes(b"fake-exe")

    cfg = AppConfig(project_root=tmp_path, llama_port=8766, llm_ctx_size=4096, llm_gpu_layers=99)
    mgr = LlamaServerManager(cfg)

    assert mgr.base_url == "http://127.0.0.1:8766"
    assert mgr.resolve_model_path() == gguf
    assert mgr.model_name == gguf.name

    cmd = mgr.build_command(exe, gguf)
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--port" in cmd and "8766" in cmd
    assert "--ctx-size" in cmd and "4096" in cmd
    assert "--batch-size" in cmd and "512" in cmd
    assert "--parallel" in cmd and "1" in cmd
    assert "-ngl" in cmd and "99" in cmd


def test_contains_non_english_cjk_and_valid_english_translation():
    # Simplified Chinese, Traditional Chinese, Japanese Kanji/Kana/Halfwidth, Korean Hangul
    assert contains_non_english_cjk("我们是从富人那里偷来的财物分给平民的义贼团") is True
    assert contains_non_english_cjk("夢幻團") is True
    assert contains_non_english_cjk("夢幻団") is True
    assert contains_non_english_cjk("ぎゃああああっ") is True
    assert contains_non_english_cjk("テスト") is True
    assert contains_non_english_cjk("ﾃｽﾄ") is True
    assert contains_non_english_cjk("환영여단 안녕하세요") is True

    # Normal English with punctuation, ellipsis, numbers
    assert contains_non_english_cjk("Phantom Troupe") is False
    assert contains_non_english_cjk("Ranshin...? 100%!") is False
    assert contains_non_english_cjk("Get the fuck off me! A centipede!!") is False

    # Validation rejects empty, refusal, CJK, and unchanged CJK echo
    assert is_valid_english_translation("夢幻団", "") is False
    assert is_valid_english_translation("夢幻団", "   ") is False
    assert is_valid_english_translation("くそっ", "I cannot translate profanity.") is False
    assert is_valid_english_translation("夢幻団", "夢幻団") is False
    assert is_valid_english_translation("金持ちから盗んだ", "我们是从富人那里偷来的") is False
    assert is_valid_english_translation("환영여단", "환영여단") is False

    # Valid English passes
    assert is_valid_english_translation("夢幻団", "Phantom Troupe") is True
    assert is_valid_english_translation("藍星……？", "Ranshin...?") is True
    assert is_valid_english_translation("...", "...") is True


@pytest.mark.asyncio
async def test_translator_retries_chinese_drift_and_anchors_english_prompt():
    memory = ChapterMemoryStore()
    calls: list[list[dict[str, str]]] = []

    async def fake_llm(messages):
        calls.append(messages)
        if len(calls) == 1:
            # Reproduces cache/translations.db Row 11 & 14: pure-Kanji title left as-is + Simplified Chinese drift
            return (
                '{"0": "夢幻団", '
                '"1": "我们是从富人那里偷来的财物分给平民的义贼团", '
                '"2": "Hermes became their leader."}'
            )
        return (
            '{"0": "Phantom Troupe", '
            '"1": "A band of righteous thieves who stole from the rich and gave to the commoners."}'
        )

    tr = DialogueTranslator("http://127.0.0.1:8766", memory, llm_caller=fake_llm)
    out = await tr.translate_page(
        [
            (0, "夢幻団"),
            (1, "金持ちから盗んだものを庶民にバラ撒く義賊団であった"),
            (2, "ヘルメスがそんな義賊団のトップになった"),
        ],
        lang="ja",
        chapter_id="ch-row11",
    )

    assert out == {
        0: "Phantom Troupe",
        1: "A band of righteous thieves who stole from the rich and gave to the commoners.",
        2: "Hermes became their leader.",
    }
    assert len(calls) == 2

    # Primary prompt anchors English-only and includes the multi-bubble example
    primary_sys = calls[0][0]["content"]
    primary_usr = calls[0][1]["content"]
    assert "STRICTLY ENGLISH ONLY" in primary_sys
    assert "Phantom Troupe" in primary_sys
    assert "ENGLISH ONLY" in primary_usr

    # Batch retry includes ONLY failed IDs (0 and 1, not 2) and warns about non-English characters
    retry_sys = calls[1][0]["content"]
    retry_usr = calls[1][1]["content"]
    assert "Chinese, Japanese, or Korean" in retry_sys
    assert "non-English" in retry_usr
    assert '"0"' in retry_usr and '"1"' in retry_usr
    assert '"2"' not in retry_usr

    # Chapter memory stores only the clean English pairs
    ctx = memory.get_context("ch-row11")
    assert len(ctx) == 3
    assert all(not contains_non_english_cjk(trans) for _, trans in ctx)


@pytest.mark.asyncio
async def test_translator_falls_back_to_single_bubble_isolation_after_batch_retry():
    memory = ChapterMemoryStore()
    calls: list[list[dict[str, str]]] = []

    async def stubborn_cjk_llm(messages):
        calls.append(messages)
        if len(calls) == 1:
            # Primary: 0 and 1 drift to Chinese, 2 is English
            return '{"0": "夢幻団", "1": "之前的首领", "2": "Move out!"}'
        if len(calls) == 2:
            # Batch retry: 1 recovers to English, 0 STILL fails
            return '{"0": "梦幻团", "1": "The previous leader"}'
        # Single-bubble isolation fallback for 0
        return '{"0": "Phantom Troupe"}'

    tr = DialogueTranslator("http://127.0.0.1:8766", memory, llm_caller=stubborn_cjk_llm)
    out = await tr.translate_page(
        [(0, "夢幻団"), (1, "以前の長"), (2, "進め！")],
        lang="ja",
        chapter_id="ch-iso",
    )

    assert out == {
        0: "Phantom Troupe",
        1: "The previous leader",
        2: "Move out!",
    }
    assert len(calls) == 3
    # 3rd call is isolated to bubble 0 only
    assert '"0"' in calls[2][1]["content"]
    assert '"1"' not in calls[2][1]["content"]
    assert '"2"' not in calls[2][1]["content"]
    assert len(memory.get_context("ch-iso")) == 3


@pytest.mark.asyncio
async def test_translator_never_records_cjk_if_all_fallbacks_fail():
    memory = ChapterMemoryStore()

    async def always_chinese(_messages):
        return '{"0": "我们是义贼团"}'

    tr = DialogueTranslator("http://127.0.0.1:8766", memory, llm_caller=always_chinese)
    out = await tr.translate_page([(0, "義賊団だ")], lang="ja", chapter_id="ch-fail")

    assert out[0] == ""
    assert memory.get_context("ch-fail") == []


def test_chapter_memory_store_rejects_cjk_in_record_and_get_context():
    mem = ChapterMemoryStore(max_recent=5)
    mem.record_page(
        "ch1",
        [
            ("夢幻団", "夢幻団"),
            ("金持ちから", "我们是从富人那里偷来的"),
            ("환영여단", "환영여단"),
            ("進め！", "Move out!"),
        ],
    )
    assert mem.get_context("ch1") == [("進め！", "Move out!")]

    # Even if CJK were injected directly into internal deque, get_context filters it out
    mem._contexts["ch1"].append(("敵だ！", "是敌人！"))
    assert mem.get_context("ch1") == [("進め！", "Move out!")]


def _make_cache_response(translated_text: str, original_text: str = "夢幻団") -> TranslateResponse:
    return TranslateResponse(
        lang="ja",
        width=800,
        height=1200,
        elapsed_ms=42,
        cached=False,
        bubbles=[
            BubbleResult(
                id=0,
                bbox=Box(x1=10, y1=10, x2=100, y2=80),
                safe_box=Box(x1=15, y1=15, x2=95, y2=75),
                bg_color="#FFFFFF",
                text_color="#000000",
                is_dark=False,
                original=original_text,
                translated=translated_text,
            )
        ],
    )


def test_translation_cache_v2_key_evicts_on_get_and_purges_on_init(tmp_path):
    assert CACHE_KEY_VERSION == "v2-en-only"
    db_path = tmp_path / "translations.db"
    cache = TranslationCache(db_path, max_entries=20)

    img_bytes = b"fake-manga-page"
    v2_key = cache.make_key(img_bytes, "ja", {})
    legacy_hasher = hashlib.sha256()
    legacy_hasher.update(img_bytes)
    legacy_hasher.update(b"\x00ja\x00{}")
    assert v2_key != legacy_hasher.hexdigest()

    # Poisoned row is rejected and deleted on get()
    poison_key = cache.make_key(b"poisoned-page", "ja")
    poison_resp = _make_cache_response("我们是从富人那里偷来的财物")
    cache.put(poison_key, poison_resp)
    assert cache.get(poison_key) is None
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT payload FROM translations WHERE cache_key = ?", (poison_key,)
        ).fetchone()
        assert row is None

    # Pre-seed DB directly with one CJK row and one clean English row, then re-open cache
    good_key = cache.make_key(b"good-page", "ja")
    good_resp = _make_cache_response("Phantom Troupe")
    bad_key = "stale-v1-poisoned-row"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO translations (cache_key, payload, last_accessed) VALUES (?, ?, ?)",
            (bad_key, poison_resp.model_dump_json(), 1),
        )
        conn.execute(
            "INSERT INTO translations (cache_key, payload, last_accessed) VALUES (?, ?, ?)",
            (good_key, good_resp.model_dump_json(), 2),
        )
        conn.commit()

    # Re-initializing TranslationCache automatically purges CJK row while preserving English row
    reopened = TranslationCache(db_path, max_entries=20)
    with closing(sqlite3.connect(db_path)) as conn:
        keys = {
            r[0] for r in conn.execute("SELECT cache_key FROM translations").fetchall()
        }
    assert bad_key not in keys
    assert good_key in keys
    hit = reopened.get(good_key)
    assert hit is not None
    assert hit.bubbles[0].translated == "Phantom Troupe"


def test_pipeline_skips_caching_when_translation_contains_cjk(tmp_path):
    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=20)
    pipeline = TranslationPipeline(
        detector=None,
        ocr=None,
        translator=None,
        memory=memory,
        cache=cache,
    )

    bad_key = cache.make_key(b"page-bad", "ja", {})
    bad_resp = _make_cache_response("梦幻团")
    pipeline._store_cache(b"page-bad", "auto", "ja", {}, bad_key, bad_resp)
    assert cache.get(bad_key) is None
    assert cache.get(cache.make_key(b"page-bad", "auto", {})) is None

    good_key = cache.make_key(b"page-good", "auto", {})
    good_resp = _make_cache_response("Phantom Troupe")
    pipeline._store_cache(b"page-good", "auto", "ja", {}, good_key, good_resp)
    assert cache.get(good_key) is not None


def test_romanize_residual_kana_handles_names_digraphs_and_sentences():
    assert romanize_residual_kana("ナナス") == "Nanasu"
    assert romanize_residual_kana("ヘルメス") == "Herumesu"
    assert romanize_residual_kana("カッパ") == "Kappa"
    assert romanize_residual_kana("パーティー") == "Paatii"
    assert romanize_residual_kana("シャウト") == "Shauto"
    assert (
        romanize_residual_kana("Currently they chose ナナス (ヘルメス's alias) as leader.")
        == "Currently they chose Nanasu (Herumesu's alias) as leader."
    )


@pytest.mark.asyncio
async def test_translator_plain_text_fourth_stage_fallback_and_kana_salvage():
    memory = ChapterMemoryStore()
    calls: list[list[dict[str, str]]] = []
    line = "現在は2人が勝手にナナス（ヘルメスの偽名）を団長に抜擢し"

    async def fake_llm(messages):
        calls.append(messages)
        if len(calls) <= 3:
            # Primary, batch retry, and single-bubble JSON retry all fail with CJK parenthetical
            return '{"0": "Currently the two made ナナス（ヘルメスの偽名）the leader."}'
        # 4th-stage plain-text single-bubble retry succeeds (with stray katakana name salvaged)
        return "Currently, those two arbitrarily made ナナス (Hermes's alias) the leader."

    tr = DialogueTranslator("http://127.0.0.1:8766", memory, llm_caller=fake_llm)
    out = await tr.translate_page([(0, line)], lang="ja", chapter_id="ch-plain")

    assert len(calls) == 4
    assert "Transliterate all character names" in calls[3][0]["content"]
    assert out[0] == "Currently, those two arbitrarily made Nanasu (Hermes's alias) the leader."
    assert not contains_non_english_cjk(out[0])


@pytest.mark.asyncio
async def test_translator_streaming_emits_each_bubble_before_next_chunk_arrives():
    memory = ChapterMemoryStore()
    events: list[tuple[str, int, str, bool]] = []

    async def fake_stream(_messages):
        yield '{"0": "Phantom Troupe", '
        events.append(("after_chunk_0", -1, "", False))
        yield '"1": "We steal from the rich."}'
        events.append(("after_chunk_1", -1, "", False))

    async def on_bubble(bid: int, text: str, final: bool) -> None:
        events.append(("bubble", bid, text, final))

    tr = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_stream_caller=fake_stream,
    )
    out = await tr.translate_page(
        [(0, "夢幻団"), (1, "金持ちから盗む")],
        lang="ja",
        chapter_id="ch-stream",
        on_bubble=on_bubble,
    )

    assert out == {0: "Phantom Troupe", 1: "We steal from the rich."}
    assert events == [
        ("bubble", 0, "Phantom Troupe", False),
        ("after_chunk_0", -1, "", False),
        ("bubble", 1, "We steal from the rich.", False),
        ("after_chunk_1", -1, "", False),
    ]
    assert len(memory.get_context("ch-stream")) == 2


@pytest.mark.asyncio
async def test_translator_streaming_retries_failed_bubble_and_emits_final_true():
    memory = ChapterMemoryStore()
    emitted: list[tuple[int, str, bool]] = []

    async def fake_stream(_messages):
        yield '{"0": "Phantom Troupe", "1": "我们偷来的"}'

    async def fake_retry(_messages):
        return '{"1": "What we stole"}'

    async def on_bubble(bid: int, text: str, final: bool) -> None:
        emitted.append((bid, text, final))

    tr = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=fake_retry,
        llm_stream_caller=fake_stream,
    )
    out = await tr.translate_page(
        [(0, "夢幻団"), (1, "盗んだもの")],
        lang="ja",
        chapter_id="ch-stream-retry",
        on_bubble=on_bubble,
    )

    assert out == {0: "Phantom Troupe", 1: "What we stole"}
    assert emitted == [
        (0, "Phantom Troupe", False),
        (1, "What we stole", True),
    ]

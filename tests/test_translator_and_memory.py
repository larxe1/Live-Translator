"""Unit tests for refusal detection, chapter memory, JSON translation repair, and llama manager."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

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
    looks_like_romaji,
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
    expected_batch = str(getattr(cfg, "llm_batch_size", 512))
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--port" in cmd and "8766" in cmd
    assert "--ctx-size" in cmd and "4096" in cmd
    assert "--batch-size" in cmd and expected_batch in cmd
    assert "--parallel" in cmd and "1" in cmd
    assert "-ngl" in cmd
    ngl_val = int(cmd[cmd.index("-ngl") + 1])
    assert 0 <= ngl_val <= 99


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
async def test_translator_caps_retry_pass_and_isolates_single_failed_bubble():
    memory = ChapterMemoryStore()
    calls: list[list[dict[str, str]]] = []

    async def single_failure_llm(messages):
        calls.append(messages)
        if len(calls) == 1:
            # Primary: only bubble 0 fails with CJK drift; 1 & 2 succeed
            return '{"0": "夢幻団", "1": "The previous leader", "2": "Move out!"}'
        # Direct single-bubble English retry for bubble 0 (max 1 retry pass)
        return '{"0": "Phantom Troupe"}'

    tr = DialogueTranslator("http://127.0.0.1:8766", memory, llm_caller=single_failure_llm)
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
    assert len(calls) == 2
    assert '"0"' in calls[1][1]["content"]
    assert '"1"' not in calls[1][1]["content"]
    assert '"2"' not in calls[1][1]["content"]
    assert "romanize" not in calls[0][0]["content"].lower()
    assert "romanize" not in calls[1][0]["content"].lower()
    assert len(memory.get_context("ch-iso")) == 3

    # When >=2 bubbles fail and 1 still fails batch retry, emit "" instead of snowballing retries
    batch_calls: list[list[dict[str, str]]] = []

    async def stubborn_batch_llm(messages):
        batch_calls.append(messages)
        if len(batch_calls) == 1:
            return '{"0": "夢幻団", "1": "之前的首领", "2": "Move out!"}'
        return '{"0": "梦幻团", "1": "The previous leader"}'

    tr2 = DialogueTranslator("http://127.0.0.1:8766", ChapterMemoryStore(), llm_caller=stubborn_batch_llm)
    out2 = await tr2.translate_page(
        [(0, "夢幻団"), (1, "以前の長"), (2, "進め！")],
        lang="ja",
        chapter_id="ch-batch-cap",
    )
    assert out2 == {0: "", 1: "The previous leader", 2: "Move out!"}
    assert len(batch_calls) == 2
    assert "romanize" not in batch_calls[1][0]["content"].lower()


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


def test_translation_cache_v4_key_evicts_cjk_and_romaji_and_purges_on_init(tmp_path):
    assert CACHE_KEY_VERSION == "v4-chunked-en"
    db_path = tmp_path / "translations.db"
    cache = TranslationCache(db_path, max_entries=20)

    img_bytes = b"fake-manga-page"
    v4_key = cache.make_key(img_bytes, "ja", {})
    v4_key_floating_true = cache.make_key(img_bytes, "ja", {}, include_floating=True)
    v4_key_floating_false = cache.make_key(img_bytes, "ja", {}, include_floating=False)
    assert v4_key == v4_key_floating_true
    assert v4_key != v4_key_floating_false

    legacy_hasher = hashlib.sha256()
    legacy_hasher.update(img_bytes)
    legacy_hasher.update(b"\x00ja\x00{}")
    assert v4_key != legacy_hasher.hexdigest()

    # Poisoned CJK and Romaji rows are refused on put()
    poison_key = cache.make_key(b"poisoned-page", "ja")
    poison_resp = _make_cache_response("我们是从富人那里偷来的财物")
    cache.put(poison_key, poison_resp)
    assert cache.get(poison_key) is None

    romaji_key = cache.make_key(b"romaji-page", "ja")
    romaji_resp = _make_cache_response("Chotto chotto matte", original_text="ちょっとちょっと待って")
    cache.put(romaji_key, romaji_resp)
    assert cache.get(romaji_key) is None

    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute("SELECT cache_key FROM translations").fetchall()
        assert rows == []

    # Direct DB-seeded Romaji row is evicted on get()
    seeded_romaji_key = cache.make_key(b"seeded-romaji", "ja")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO translations (cache_key, payload, last_accessed) VALUES (?, ?, ?)",
            (seeded_romaji_key, romaji_resp.model_dump_json(), 1),
        )
        conn.commit()
    assert cache.get(seeded_romaji_key) is None

    # Pre-seed DB directly with CJK row, Romaji row, and clean English row, then re-open cache
    good_key = cache.make_key(b"good-page", "ja")
    good_resp = _make_cache_response("Phantom Troupe")
    bad_cjk_key = "stale-v1-cjk-row"
    bad_romaji_key = "stale-v3-romaji-row"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO translations (cache_key, payload, last_accessed) VALUES (?, ?, ?)",
            (bad_cjk_key, poison_resp.model_dump_json(), 1),
        )
        conn.execute(
            "INSERT INTO translations (cache_key, payload, last_accessed) VALUES (?, ?, ?)",
            (bad_romaji_key, romaji_resp.model_dump_json(), 2),
        )
        conn.execute(
            "INSERT INTO translations (cache_key, payload, last_accessed) VALUES (?, ?, ?)",
            (good_key, good_resp.model_dump_json(), 3),
        )
        conn.commit()

    # Re-initializing TranslationCache automatically purges CJK + Romaji rows while preserving English row
    reopened = TranslationCache(db_path, max_entries=20)
    with closing(sqlite3.connect(db_path)) as conn:
        keys = {
            r[0] for r in conn.execute("SELECT cache_key FROM translations").fetchall()
        }
    assert bad_cjk_key not in keys
    assert bad_romaji_key not in keys
    assert good_key in keys
    hit = reopened.get(good_key)
    assert hit is not None
    assert hit.bubbles[0].translated == "Phantom Troupe"


def test_pipeline_skips_caching_when_translation_contains_cjk_or_romaji(tmp_path):
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

    romaji_key = cache.make_key(b"page-romaji", "ja", {})
    romaji_resp = _make_cache_response("Koko ja yada", original_text="ここじゃやだ")
    pipeline._store_cache(b"page-romaji", "auto", "ja", {}, romaji_key, romaji_resp)
    assert cache.get(romaji_key) is None

    good_key = cache.make_key(b"page-good", "auto", {})
    good_resp = _make_cache_response("Phantom Troupe")
    pipeline._store_cache(b"page-good", "auto", "ja", {}, good_key, good_resp)
    assert cache.get(good_key) is not None


def test_looks_like_romaji_catches_leaks_and_preserves_valid_english_and_names():
    # Must catch Japanese & Korean Romaji leaks
    romaji_cases = [
        ("ちょっとちょっと待って", "Chotto chotto matte"),
        ("ちょっと待って", "Chotto matte"),
        ("ここじゃやだ", "Koko ja yada"),
        ("ベッド行こ？", "Bed ikko?"),
        ("むしゃむじゃ", "Mushamu musha"),
        ("やめてください", "Yamete kudasai"),
        ("なにこれ", "Nani kore"),
        ("大丈夫ですか", "Daijoubu desu ka"),
        ("本当に？", "Hontou ni?"),
        ("それはダメ", "Sore wa dame"),
        ("コラーッ！", "Koraa!!"),
        ("コラーッ！", "KORAA!!"),
        ("一亀見てクロ…", "Ichigame Mite Kuro...coms"),
        ("カサ・・・", "Kasa... ..."),
        ("제발 하지마", "Jebal hajima"),
        ("괜찮아?", "Gwaenchana?"),
    ]
    for orig, cand in romaji_cases:
        assert looks_like_romaji(orig, cand) is True, f"Failed to detect Romaji: {orig!r} -> {cand!r}"
        assert is_valid_english_translation(orig, cand) is False

    # Must NEVER false-positive on valid English sentences, names, or loanwords
    valid_cases = [
        ("マサキ…！！", "Masaki...!!"),
        ("七海！", "Nanami!"),
        ("藍星……？", "Ranshin...?"),
        ("ちょっと", "A little"),
        ("ちょっと待って", "Wait a second!"),
        ("ここじゃやだ", "Not here!"),
        ("ベッド行こ？", "Wanna go to bed?"),
        ("ベッド行こ？", "Let's go to bed?"),
        ("健二、待って！", "Kenji, wait for me!"),
        ("先生、大丈夫？", "Sensei, are you okay?"),
        ("先生！", "Sensei!"),
        ("東京", "Tokyo"),
        ("ラーメン", "Ramen"),
        ("うそ！", "No way!"),
        ("何！？", "What?!"),
        ("おい！", "Hey!"),
        ("猪股大喜", "Taiki Inomata"),
        ("七海ちゃんより好きって言って。", "Tell me you love me more than Nanami."),
    ]
    for orig, cand in valid_cases:
        assert looks_like_romaji(orig, cand) is False, f"False positive on valid English/name: {orig!r} -> {cand!r}"
        assert is_valid_english_translation(orig, cand) is True


@pytest.mark.asyncio
async def test_translator_rejects_romaji_and_residual_kana_without_auto_romanizing():
    memory = ChapterMemoryStore()
    calls: list[list[dict[str, str]]] = []

    async def romaji_then_english(messages):
        calls.append(messages)
        if len(calls) == 1:
            return (
                '{"0": "Chotto chotto matte", '
                '"1": "Koko ja yada", '
                '"2": "They chose ナナス as leader.", '
                '"3": "Masaki...!!"}'
            )
        return (
            '{"0": "Wait, hold on!", '
            '"1": "Not here!", '
            '"2": "They chose Nanasu as leader."}'
        )

    tr = DialogueTranslator("http://127.0.0.1:8766", memory, llm_caller=romaji_then_english)
    out = await tr.translate_page(
        [
            (0, "ちょっとちょっと待って"),
            (1, "ここじゃやだ"),
            (2, "ナナスを団長に選んだ"),
            (3, "マサキ…！！"),
        ],
        lang="ja",
        chapter_id="ch-romaji",
    )

    assert len(calls) == 2
    assert out == {
        0: "Wait, hold on!",
        1: "Not here!",
        2: "They chose Nanasu as leader.",
        3: "Masaki...!!",
    }
    # Retry prompt should NOT include bubble 3 ("Masaki...!!" is a valid name)
    assert '"3"' not in calls[1][1]["content"]
    assert '"0"' in calls[1][1]["content"]
    assert '"1"' in calls[1][1]["content"]
    assert '"2"' in calls[1][1]["content"]
    assert "romanize" not in calls[0][0]["content"].lower()
    assert "romanize" not in calls[1][0]["content"].lower()
    # Memory must never store Romaji pairs
    memory.record_page("ch-romaji", [("ここじゃやだ", "Koko ja yada")])
    assert all(t != "Koko ja yada" for _, t in memory.get_context("ch-romaji"))


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


def test_chapter_memory_multi_page_formatted_context_and_series_continuity():
    mem = ChapterMemoryStore()
    assert mem.max_recent == 36

    mem.set_glossary("ch1", {"藍星": "Ranshin", "夢幻団": "Phantom Troupe"}, series_id="series-a")
    # Chapter 2 inherits series glossary, and chapter-level entries can override/extend it
    assert mem.get_glossary("ch2", series_id="series-a") == {
        "藍星": "Ranshin",
        "夢幻団": "Phantom Troupe",
    }
    mem.set_glossary("ch2", {"藍星": "Ranshin-sama"})
    assert mem.get_glossary("ch2", series_id="series-a") == {
        "藍星": "Ranshin-sama",
        "夢幻団": "Phantom Troupe",
    }

    mem.record_page("ch1", [("行くぞ", "Let's move."), ("わかった", "Got it.")], series_id="series-a")
    mem.record_page("ch1", [("危ない！", "Watch out!")], series_id="series-a")
    mem.record_page("ch2", [("間に合ったか", "Did we make it in time?")], series_id="series-a")

    ctx_ch2 = mem.get_context("ch2", series_id="series-a")
    assert ctx_ch2 == [
        ("行くぞ", "Let's move."),
        ("わかった", "Got it."),
        ("危ない！", "Watch out!"),
        ("間に合ったか", "Did we make it in time?"),
    ]

    formatted = mem.get_formatted_context("ch2", series_id="series-a", max_pairs=28)
    assert "[Page -3]\n- 行くぞ -> Let's move.\n- わかった -> Got it." in formatted
    assert "[Page -2]\n- 危ない！ -> Watch out!" in formatted
    assert "[Page -1]\n- 間に合ったか -> Did we make it in time?" in formatted

    # Respects max_pairs cap from the newest entries backward
    capped = mem.get_formatted_context("ch2", series_id="series-a", max_pairs=2)
    assert "[Page -2]\n- 危ない！ -> Watch out!" in capped
    assert "[Page -1]\n- 間に合ったか -> Did we make it in time?" in capped
    assert "Let's move." not in capped

    # Clearing chapter + series wipes both while preserving glossaries
    mem.clear("ch2", series_id="series-a")
    assert mem.get_context("ch2", series_id="series-a") == []
    assert mem.get_glossary("ch2", series_id="series-a")["夢幻団"] == "Phantom Troupe"


@pytest.mark.asyncio
async def test_translator_prompt_omits_story_context_by_default_and_honors_opt_in():
    memory = ChapterMemoryStore()
    memory.set_glossary("ch1", {"藍星": "Ranshin"}, series_id="my-series")
    memory.record_page("ch1", [("どこだ？", "Where are you?")], series_id="my-series")
    captured_default: list[list[dict[str, str]]] = []

    async def fake_llm_default(messages):
        captured_default.append(messages)
        return '{"0": "Right here, Ranshin.", "1": "*RUSTLE*"}'

    # Default use_story_context=False: omits [Page -1] Story Context, keeps glossary & floating hints
    tr = DialogueTranslator("http://127.0.0.1:8766", memory, llm_caller=fake_llm_default)
    out = await tr.translate_page(
        [(0, "ここだよ、藍星"), (1, "カサカサ")],
        lang="ja",
        chapter_id="ch2",
        series_id="my-series",
        bubble_types={0: "bubble", 1: "floating"},
    )
    assert out == {0: "Right here, Ranshin.", 1: "*RUSTLE*"}
    sys_prompt = captured_default[0][0]["content"]
    usr_prompt = captured_default[0][1]["content"]
    assert "STRICTLY ENGLISH ONLY" in sys_prompt
    assert "[floating]" in sys_prompt
    assert "romanize" not in sys_prompt.lower()
    assert "[Page -1]" not in usr_prompt
    assert "Story Context" not in usr_prompt
    assert '"1": [floating]' in usr_prompt
    assert "Ranshin" in usr_prompt

    # Opt-in use_story_context=True restores [Page -1] Story Context
    captured_ctx: list[list[dict[str, str]]] = []

    async def fake_llm_ctx(messages):
        captured_ctx.append(messages)
        return '{"0": "Right here, Ranshin.", "1": "*RUSTLE*"}'

    tr_ctx = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=fake_llm_ctx,
        cfg=SimpleNamespace(use_story_context=True, llm_chunk_size=10),
    )
    await tr_ctx.translate_page(
        [(0, "ここだよ、藍星"), (1, "カサカサ")],
        lang="ja",
        chapter_id="ch2",
        series_id="my-series",
        bubble_types={0: "bubble", 1: "floating"},
    )
    assert "[Page -1]" in captured_ctx[0][1]["content"]
    assert "Where are you?" in captured_ctx[0][1]["content"]


@pytest.mark.asyncio
async def test_translator_chunks_25_bubbles_with_sampling_params_and_thermal_guard(monkeypatch):
    import httpx

    seen_payloads: list[dict[str, object]] = []
    guard_calls: list[int] = []

    class _FakeGuard:
        async def wait_if_hot(self, cancel_event=None, max_wait_s=45.0):
            guard_calls.append(len(seen_payloads))
            return "ok"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(body)
        user_msg = body["messages"][1]["content"]
        json_part = user_msg[user_msg.rfind("{") :]
        in_map = json.loads(json_part)
        out_map = {k: f"Line {k} translated!" for k in in_map}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(out_map)}}]},
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    memory = ChapterMemoryStore()
    memory.record_page("ch-chunk", [("前のページ", "Previous page line")])
    cfg = SimpleNamespace(use_story_context=False, llm_chunk_size=10)
    tr = DialogueTranslator("http://127.0.0.1:8766", memory, cfg=cfg)

    bubbles = [(i, f"台詞{i}") for i in range(25)]
    out = await tr.translate_page(bubbles, lang="ja", chapter_id="ch-chunk", gpu_guard=_FakeGuard())

    assert len(out) == 25
    assert len(seen_payloads) == 3  # 10 + 10 + 5
    assert len(guard_calls) == 3

    expected_max_tokens = [560, 560, 280]  # min(640, max(160, n * 56))
    for payload, exp_max in zip(seen_payloads, expected_max_tokens):
        assert payload["temperature"] == 0.2
        assert payload["top_p"] == 0.85
        assert payload["repeat_penalty"] == 1.12
        assert payload["max_tokens"] == exp_max
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert "[Page -1]" not in payload["messages"][1]["content"]


@pytest.mark.asyncio
async def test_translator_cancel_event_aborts_remaining_chunks():
    memory = ChapterMemoryStore()
    cancel_ev = asyncio.Event()
    calls = 0

    async def fake_llm(messages):
        nonlocal calls
        calls += 1
        user_msg = messages[1]["content"]
        in_map = json.loads(user_msg[user_msg.rfind("{") :])
        if calls == 1:
            cancel_ev.set()
        return json.dumps({k: f"Done {k}" for k in in_map})

    tr = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=fake_llm,
        cfg=SimpleNamespace(use_story_context=False, llm_chunk_size=10),
    )
    bubbles = [(i, f"台詞{i}") for i in range(25)]
    out = await tr.translate_page(bubbles, lang="ja", chapter_id="ch-cancel", cancel_event=cancel_ev)

    assert calls == 1
    assert len(out) == 10


@pytest.mark.asyncio
async def test_llama_http_payload_disables_thinking_and_sends_sampling_params(monkeypatch):
    import httpx

    seen_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(body)
        if body.get("stream"):
            sse = (
                'data: {"choices":[{"delta":{"content":"{\\"0\\": \\"Hello\\"}"}}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"0": "Hello"}'}}]},
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    tr = DialogueTranslator("http://127.0.0.1:8766", ChapterMemoryStore())
    raw = await tr._call_llama([{"role": "user", "content": "hi"}], max_tokens=280)
    assert raw == '{"0": "Hello"}'

    chunks = [c async for c in tr._call_llama_stream([{"role": "user", "content": "hi"}], max_tokens=280)]
    assert "".join(chunks) == '{"0": "Hello"}'

    assert len(seen_payloads) == 2
    for payload in seen_payloads:
        assert payload["max_tokens"] == 280
        assert payload["temperature"] == 0.2
        assert payload["top_p"] == 0.85
        assert payload["repeat_penalty"] == 1.12
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

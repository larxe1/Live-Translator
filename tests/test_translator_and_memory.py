"""Unit tests for refusal detection, chapter memory, JSON translation repair, and llama manager."""

from __future__ import annotations

import pytest

from server.config import AppConfig
from server.translator.llama_manager import LlamaServerManager
from server.translator.memory import ChapterMemoryStore
from server.translator.translator import DialogueTranslator, _parse_translation_json, is_refusal


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
    assert out[0] == "くそったれ！"


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

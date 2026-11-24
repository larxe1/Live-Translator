"""Unit and integration tests for Gemini Vision Interactions API and Dynamic Hybrid Shifting."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import httpx
from PIL import Image, ImageDraw
import pytest

from server.cache import CACHE_KEY_VERSION, TranslationCache
from server.pipeline import TranslationPipeline
from server.schemas import Box
from server.translator.gemini_translator import (
    GEMINI_INTERACTIONS_URL,
    GeminiBlockedError,
    GeminiRateLimitError,
    GeminiUnavailableError,
    GeminiVisionTranslator,
    prepare_vision_image,
)
from server.translator.memory import (
    ChapterMemoryStore,
    extract_character_names,
    looks_like_romaji,
)
from server.translator.translator import DialogueTranslator, is_valid_english_translation
from server.vision.slicer import RawDetection


def _make_v1_interaction_response(text: str, status: str = "completed") -> dict[str, Any]:
    return {
        "id": "int-test-1",
        "status": status,
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


def _make_sample_png_bytes(width: int = 600, height: int = 800) -> bytes:
    img = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle([360, 60, 540, 220], fill=(250, 250, 250), outline=(10, 10, 10), width=4)
    draw.rectangle([40, 260, 220, 440], fill=(15, 15, 15))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeLlamaManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.active_model_role: str = ""
        self.active_model_filename: str = ""
        self.model_name: str = ""

    def ensure_model(self, lang: str = "ja", uncensored: bool = False) -> str:
        self.calls.append((lang, uncensored))
        if lang == "ko":
            self.active_model_role = "ko"
            self.active_model_filename = "EXAONE-3.5-7.8B-Instruct-abliterated.i1-IQ4_XS.gguf"
        elif lang == "ja" and uncensored:
            self.active_model_role = "ja_uncensored"
            self.active_model_filename = "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf"
        else:
            self.active_model_role = "ja"
            self.active_model_filename = "shisa-v2-qwen2.5-7b.i1-Q4_K_M.gguf"
        self.model_name = self.active_model_filename
        return self.active_model_role


def test_prepare_vision_image_resizes_standard_page_and_crops_tall_strip():
    # Standard wide page: resized to max_width=1024, JPEG encoded
    wide_img = Image.new("RGB", (1800, 2200), (220, 230, 240))
    b64_std = prepare_vision_image(wide_img)
    raw_std = base64.b64decode(b64_std)
    with Image.open(BytesIO(raw_std)) as decoded_std:
        assert decoded_std.format == "JPEG"
        assert decoded_std.width <= 1024
        assert decoded_std.height <= 2048

    # Tall webtoon strip (height > 2400): crops [min_y - 200, max_y + 200] band around chunk bubbles
    strip = Image.new("RGB", (800, 4000), (20, 20, 20))
    draw = ImageDraw.Draw(strip)
    # Paint distinctive red band around y=2000..2600 where chunk bubbles live
    draw.rectangle([0, 1800, 800, 2800], fill=(230, 30, 30))
    boxes = {
        0: Box(x1=100, y1=200, x2=300, y2=400),
        1: Box(x1=100, y1=2050, x2=400, y2=2250),
        2: Box(x1=100, y1=2400, x2=400, y2=2550),
    }
    b64_band = prepare_vision_image(strip, bubble_boxes=boxes, chunk_ids=[1, 2])
    raw_band = base64.b64decode(b64_band)
    with Image.open(BytesIO(raw_band)) as decoded_band:
        assert decoded_band.format == "JPEG"
        # Expected vertical span: (2550 + 200) - (2050 - 200) = 900px
        assert decoded_band.width == 800
        assert decoded_band.height == 900
        r, g, b = decoded_band.getpixel((400, 450))
        assert r > 180 and g < 80 and b < 80


@pytest.mark.asyncio
async def test_gemini_v1_interactions_payload_shape_and_steps_parsing(monkeypatch):
    seen_requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_requests.append((str(request.url), dict(request.headers), body))
        return httpx.Response(
            200,
            json=_make_v1_interaction_response('{"0": "Wait a second!", "1": "Are you okay?"}'),
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-gemini-key-12345")

    cfg = SimpleNamespace(
        gemini_models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        gemini_use_vision=True,
        gemini_timeout_s=15.0,
    )
    gem = GeminiVisionTranslator(cfg=cfg)
    page_img = Image.new("RGB", (800, 1200), (245, 245, 245))

    translated, used_model = await gem.translate_chunk(
        [(0, "ちょっと待って！"), (1, "大丈夫？")],
        lang="ja",
        image=page_img,
        bubble_boxes={0: Box(x1=50, y1=50, x2=200, y2=150)},
    )

    assert used_model == "gemini-3.5-flash-lite"
    assert translated == {0: "Wait a second!", 1: "Are you okay?"}
    assert len(seen_requests) == 1

    url, headers, body = seen_requests[0]
    assert url == GEMINI_INTERACTIONS_URL
    assert headers["x-goog-api-key"] == "test-dummy-gemini-key-12345"
    assert body["model"] == "gemini-3.5-flash-lite"
    assert body["store"] is False
    assert body["generation_config"]["temperature"] == 0.3
    assert body["generation_config"]["thinking_level"] == "minimal"
    assert body["generation_config"]["max_output_tokens"] >= 256
    assert "visual context" in body["system_instruction"].lower()
    assert len(body["input"]) == 1
    assert body["input"][0]["type"] == "user_input"
    content = body["input"][0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "image"
    assert content[0]["mime_type"] == "image/jpeg"
    assert len(content[0]["data"]) > 20
    assert content[1]["type"] == "text"
    assert '"0": "ちょっと待って!"' in content[1]["text"]

    # Text-only omits image block
    seen_requests.clear()
    await gem.translate_chunk([(0, "ちょっと待って！")], lang="ja", image=page_img, use_vision=False)
    text_only_content = seen_requests[0][2]["input"][0]["content"]
    assert len(text_only_content) == 1
    assert text_only_content[0]["type"] == "text"


@pytest.mark.asyncio
async def test_gemini_503_circuit_breaker_and_error_classifications(monkeypatch):
    attempted_models: list[str] = []

    def handler_503_then_ok(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        model = body["model"]
        attempted_models.append(model)
        if model == "gemini-3.5-flash-lite":
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        return httpx.Response(200, json=_make_v1_interaction_response('{"0": "Recovered!"}'))

    real_async_client = httpx.AsyncClient

    def install_handler(fn):
        transport = httpx.MockTransport(fn)
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda *a, **kw: real_async_client(*a, transport=transport, **kw),
        )

    install_handler(handler_503_then_ok)
    cfg = SimpleNamespace(
        gemini_models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        gemini_use_vision=False,
    )
    gem = GeminiVisionTranslator(cfg=cfg, api_key="test-dummy-key-99999", model_breaker_s=120.0)
    out1, m1 = await gem.translate_chunk([(0, "テスト")], lang="ja")
    assert out1 == {0: "Recovered!"}
    assert m1 == "gemini-3.1-flash-lite"
    assert attempted_models == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

    # Second call skips 503'd model immediately thanks to 120s breaker!
    attempted_models.clear()
    out2, m2 = await gem.translate_chunk([(0, "テスト")], lang="ja")
    assert out2 == {0: "Recovered!"}
    assert m2 == "gemini-3.1-flash-lite"
    assert attempted_models == ["gemini-3.1-flash-lite"]

    # All 429 -> GeminiRateLimitError
    install_handler(lambda req: httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}}))
    gem_429 = GeminiVisionTranslator(cfg=cfg, api_key="test-dummy-key-99999")
    with pytest.raises(GeminiRateLimitError):
        await gem_429.translate_chunk([(0, "テスト")], lang="ja")

    # HTTP 400 safety -> GeminiBlockedError
    install_handler(lambda req: httpx.Response(400, json={"error": {"message": "PROHIBITED_CONTENT safety block"}}))
    gem_blk = GeminiVisionTranslator(cfg=cfg, api_key="test-dummy-key-99999")
    with pytest.raises(GeminiBlockedError):
        await gem_blk.translate_chunk([(0, "テスト")], lang="ja")

    # HTTP 200 status=failed with empty steps -> GeminiBlockedError
    install_handler(lambda req: httpx.Response(200, json={"status": "failed", "steps": []}))
    with pytest.raises(GeminiBlockedError):
        await gem_blk.translate_chunk([(0, "テスト")], lang="ja")

    # HTTP 200 refusal text -> GeminiBlockedError
    install_handler(
        lambda req: httpx.Response(
            200,
            json=_make_v1_interaction_response("I'm sorry, I cannot translate explicit content."),
        )
    )
    with pytest.raises(GeminiBlockedError):
        await gem_blk.translate_chunk([(0, "挿れたい")], lang="ja")

    # Network error redacts API key
    def handler_boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("failed with key test-dummy-key-99999")

    install_handler(handler_boom)
    gem_err = GeminiVisionTranslator(cfg=cfg, api_key="test-dummy-key-99999")
    with pytest.raises(GeminiUnavailableError) as exc_info:
        await gem_err.translate_chunk([(0, "テスト")], lang="ja")
    assert "test-dummy-key-99999" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_dynamic_hybrid_blocked_shifts_only_blocked_chunk_without_cooldown():
    memory = ChapterMemoryStore()
    llama_mgr = _FakeLlamaManager()
    gem_calls: list[list[int]] = []
    local_calls: list[list[int]] = []
    providers: list[str] = []

    class _FakeGemini:
        def is_configured(self) -> bool:
            return True

        def _active_models(self) -> list[str]:
            return ["gemini-3.5-flash-lite"]

        async def translate_chunk(self, bubbles, lang, **kwargs):
            ids = [bid for bid, _ in bubbles]
            gem_calls.append(ids)
            # Block only explicit bubble chunk
            if any("挿れたい" in txt for _, txt in bubbles):
                raise GeminiBlockedError("safety block on explicit panel")
            return {bid: f"Gemini line {bid}." for bid, _ in bubbles}, "gemini-3.5-flash-lite"

    async def fake_local_llm(messages):
        user_msg = messages[1]["content"]
        in_map = json.loads(user_msg[user_msg.rfind("{") :])
        local_calls.append([int(k) for k in in_map])
        return json.dumps({k: f"Local uncensored {k}!" for k in in_map})

    cfg = SimpleNamespace(
        translation_mode="hybrid",
        gemini_chunk_size=2,
        llm_chunk_size=2,
        gemini_cooldown_s=60.0,
        use_story_context=True,
        max_context_pairs=6,
    )
    tr = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=fake_local_llm,
        cfg=cfg,
        llama_mgr=llama_mgr,
        gemini_translator=_FakeGemini(),  # type: ignore[arg-type]
    )

    # Page 1: chunk 0 normal (Gemini), chunk 1 explicit (blocked -> Local), chunk 2 normal (Gemini!)
    out1 = await tr.translate_page(
        [
            (0, "おはよう"),
            (1, "元気？"),
            (2, "挿れたい"),
            (3, "もっと奥まで"),
            (4, "また明日ね"),
        ],
        lang="ja",
        chapter_id="ch-hyb",
        on_provider=lambda p: providers.append(p),
    )
    assert out1 == {
        0: "Gemini line 0.",
        1: "Gemini line 1.",
        2: "Local uncensored 2!",
        3: "Local uncensored 3!",
        4: "Gemini line 4.",
    }
    assert tr.last_provider == "hybrid"
    assert tr._gemini_cooldown_until == 0.0
    assert gem_calls == [[0, 1], [2, 3], [4]]
    assert local_calls == [[2, 3]]
    assert llama_mgr.calls == [("ja", False)]
    assert "gemini:gemini-3.5-flash-lite" in providers
    assert "local:ja" in providers

    # Page 2 (non-explicit): IMMEDIATELY uses Gemini again (zero cooldown!)
    gem_calls.clear()
    local_calls.clear()
    out2 = await tr.translate_page([(0, "いい天気だね")], lang="ja", chapter_id="ch-hyb")
    assert out2 == {0: "Gemini line 0."}
    assert tr.last_provider == "gemini:gemini-3.5-flash-lite"
    assert gem_calls == [[0]]
    assert local_calls == []


@pytest.mark.asyncio
async def test_dynamic_hybrid_rate_limit_sets_cooldown_and_recovers_after_expiry():
    memory = ChapterMemoryStore()
    llama_mgr = _FakeLlamaManager()
    gem_attempts = 0

    class _RateLimitingGemini:
        def __init__(self) -> None:
            self.fail_429 = True

        def is_configured(self) -> bool:
            return True

        def _active_models(self) -> list[str]:
            return ["gemini-3.5-flash-lite"]

        async def translate_chunk(self, bubbles, lang, **kwargs):
            nonlocal gem_attempts
            gem_attempts += 1
            if self.fail_429:
                raise GeminiRateLimitError("HTTP 429 across all models")
            return {bid: f"Cloud back {bid}!" for bid, _ in bubbles}, "gemini-3.5-flash-lite"

    async def fake_local_llm(messages):
        user_msg = messages[1]["content"]
        in_map = json.loads(user_msg[user_msg.rfind("{") :])
        return json.dumps({k: f"Local fallback {k}!" for k in in_map})

    gem = _RateLimitingGemini()
    cfg = SimpleNamespace(
        translation_mode="hybrid",
        gemini_chunk_size=2,
        llm_chunk_size=2,
        gemini_cooldown_s=60.0,
    )
    tr = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=fake_local_llm,
        cfg=cfg,
        llama_mgr=llama_mgr,
        gemini_translator=gem,  # type: ignore[arg-type]
    )

    # Page 1: hits 429 on chunk 0 -> shifts chunk 0 AND chunk 1 to local without re-hitting Gemini!
    out1 = await tr.translate_page(
        [(0, "あ"), (1, "い"), (2, "う")],
        lang="ja",
        chapter_id="ch-429",
    )
    assert out1 == {0: "Local fallback 0!", 1: "Local fallback 1!", 2: "Local fallback 2!"}
    assert gem_attempts == 1
    assert tr._gemini_cooldown_until > time.monotonic()
    assert tr.last_provider == "local:ja"

    # Page 2 while in cooldown: skips Gemini completely
    out2 = await tr.translate_page([(0, "え")], lang="ja", chapter_id="ch-429")
    assert out2 == {0: "Local fallback 0!"}
    assert gem_attempts == 1

    # Cooldown expires -> Page 3 uses Gemini again!
    gem.fail_429 = False
    tr._gemini_cooldown_until = time.monotonic() - 1.0
    out3 = await tr.translate_page([(0, "お")], lang="ja", chapter_id="ch-429")
    assert out3 == {0: "Cloud back 0!"}
    assert gem_attempts == 2
    assert tr.last_provider == "gemini:gemini-3.5-flash-lite"


@pytest.mark.asyncio
async def test_shisa_to_ja_uncensored_double_safety_net_on_refusal():
    memory = ChapterMemoryStore()
    llama_mgr = _FakeLlamaManager()
    providers: list[str] = []

    async def role_aware_local_llm(messages):
        user_msg = messages[1]["content"]
        in_map = json.loads(user_msg[user_msg.rfind("{") :])
        if llama_mgr.active_model_role == "ja":
            # Shisa translates normal line 0 but refuses explicit line 1!
            out = {}
            for k in in_map:
                if k == "0":
                    out[k] = "Wait a second!"
                else:
                    out[k] = "I'm sorry, I cannot translate sexually explicit content."
            return json.dumps(out)
        # Qwen2.5-7B-abliterated (ja_uncensored) translates everything faithfully!
        return json.dumps({k: "I want to put it in." for k in in_map})

    cfg = SimpleNamespace(translation_mode="local", llm_chunk_size=12)
    tr = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=role_aware_local_llm,
        cfg=cfg,
        llama_mgr=llama_mgr,
    )

    out = await tr.translate_page(
        [(0, "ちょっと待って！"), (1, "挿れたい")],
        lang="ja",
        chapter_id="ch-nsfw",
        on_provider=lambda p: providers.append(p),
    )

    assert out == {0: "Wait a second!", 1: "I want to put it in."}
    assert llama_mgr.calls == [("ja", False), ("ja", True)]
    assert providers == ["local:ja", "local:ja_uncensored"]
    assert tr.last_provider == "local:ja_uncensored"


@pytest.mark.asyncio
async def test_korean_routing_to_exaone_and_korean_romaji_guard():
    memory = ChapterMemoryStore()
    llama_mgr = _FakeLlamaManager()
    captured_messages: list[list[dict[str, str]]] = []

    async def fake_exaone(messages):
        captured_messages.append(messages)
        return json.dumps(
            {
                "0": "Hold on, Seojun...",
                "1": "Are you crazy?! We can't do this here!",
                "2": "No way, haha!",
            }
        )

    cfg = SimpleNamespace(translation_mode="local", llm_chunk_size=12)
    tr = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=fake_exaone,
        cfg=cfg,
        llama_mgr=llama_mgr,
    )

    out = await tr.translate_page(
        [
            (0, "서준 오빠, 잠깐만..."),
            (1, "미쳤어?! 여기서 이러면 안 돼!"),
            (2, "말도 안 돼ㅋㅋ"),
        ],
        lang="ko",
        chapter_id="ch-ko",
    )

    assert llama_mgr.calls == [("ko", False)]
    assert tr.last_provider == "local:ko"
    assert tr._is_qwen3_model() is False
    assert out[0] == "Hold on, Seojun..."
    assert "ㅋㅋ" in captured_messages[0][0]["content"]
    assert "오빠" in captured_messages[0][0]["content"]

    # Korean romanization caught; Korean character names preserved!
    for orig, bad in [
        ("잠깐만 기다려", "Jamkkanman gidaryeo"),
        ("여긴 싫어", "Yeogin silheo"),
        ("그만해 제발", "Geumanhae jebal"),
        ("나도 좋아해", "Nado johahae"),
    ]:
        assert looks_like_romaji(orig, bad) is True
        assert is_valid_english_translation(orig, bad) is False

    for orig, good in [
        ("민지야!", "Minji!"),
        ("지원아, 잠깐만!", "Wait, Jiwon!"),
        ("서준...?", "Seojun...?"),
        ("민지한테 말해.", "Tell Minji."),
    ]:
        assert looks_like_romaji(orig, good) is False
        assert is_valid_english_translation(orig, good) is True

    names = extract_character_names(
        [("민지야!", "Minji!"), ("지원아, 잠깐만!", "Wait, Jiwon!"), ("서준...?", "Seojun...?")]
    )
    assert "Minji" in names
    assert "Jiwon" in names
    assert "Seojun" in names


@pytest.mark.asyncio
async def test_shared_sliding_context_across_gemini_local_gemini_switches():
    memory = ChapterMemoryStore()
    llama_mgr = _FakeLlamaManager()
    gem_seen_contexts: list[tuple[list[tuple[str, str]], list[str] | None]] = []
    local_seen_prompts: list[str] = []

    class _ContextTrackingGemini:
        def is_configured(self) -> bool:
            return True

        def _active_models(self) -> list[str]:
            return ["gemini-3.5-flash-lite"]

        async def translate_chunk(self, bubbles, lang, **kwargs):
            gem_seen_contexts.append((list(kwargs.get("context") or []), kwargs.get("known_names")))
            if any("挿れたい" in txt for _, txt in bubbles):
                raise GeminiBlockedError("blocked explicit chunk")
            responses = {
                0: "Masaki...!!",
                1: "...I love you.",
                4: "Wait for me, Masaki!",
            }
            return {bid: responses[bid] for bid, _ in bubbles}, "gemini-3.5-flash-lite"

    async def local_tracking_llm(messages):
        local_seen_prompts.append(messages[1]["content"])
        return json.dumps({"2": "I want to hold you close.", "3": "Me too, Masaki."})

    cfg = SimpleNamespace(
        translation_mode="hybrid",
        gemini_chunk_size=2,
        llm_chunk_size=2,
        use_story_context=True,
        max_context_pairs=6,
    )
    tr = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=local_tracking_llm,
        cfg=cfg,
        llama_mgr=llama_mgr,
        gemini_translator=_ContextTrackingGemini(),  # type: ignore[arg-type]
    )

    out = await tr.translate_page(
        [
            (0, "まさき．．．！！"),
            (1, "．．．すき"),
            (2, "挿れたい"),
            (3, "俺も、将暉"),
            (4, "待って、将暉！"),
        ],
        lang="ja",
        chapter_id="ch-ctx",
    )

    assert out[0] == "Masaki...!!"
    assert out[2] == "I want to hold you close."
    assert out[4] == "Wait for me, Masaki!"

    # Chunk 1 (Local fallback) saw Gemini Chunk 0's pairs AND Established Character Name "Masaki"!
    assert len(local_seen_prompts) == 1
    assert "Masaki...!!" in local_seen_prompts[0]
    assert "...I love you." in local_seen_prompts[0]
    assert "Masaki" in local_seen_prompts[0]

    # Chunk 2 (back on Gemini) saw Local Chunk 1's pairs AND "Masaki" in known_names!
    chunk2_ctx, chunk2_names = gem_seen_contexts[2]
    assert ("挿れたい", "I want to hold you close.") in chunk2_ctx
    assert ("俺も、将暉", "Me too, Masaki.") in chunk2_ctx
    assert chunk2_names is not None and "Masaki" in chunk2_names


@pytest.mark.asyncio
async def test_gemini_only_mode_retries_text_only_once_on_block():
    memory = ChapterMemoryStore()
    llama_mgr = _FakeLlamaManager()
    vision_flags: list[bool | None] = []

    class _BlockVisionAllowTextGemini:
        def is_configured(self) -> bool:
            return True

        def _active_models(self) -> list[str]:
            return ["gemini-3.5-flash-lite"]

        async def translate_chunk(self, bubbles, lang, **kwargs):
            vision_flags.append(kwargs.get("use_vision"))
            if kwargs.get("image") is not None:
                raise GeminiBlockedError("image blocked")
            return {bid: "Text-only succeeded!" for bid, _ in bubbles}, "gemini-3.5-flash-lite"

    cfg = SimpleNamespace(translation_mode="gemini", gemini_chunk_size=20)
    tr = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        cfg=cfg,
        llama_mgr=llama_mgr,
        gemini_translator=_BlockVisionAllowTextGemini(),  # type: ignore[arg-type]
    )
    out = await tr.translate_page(
        [(0, "テスト")],
        lang="ja",
        chapter_id="ch-gem-only",
        image=Image.new("RGB", (400, 600), (255, 255, 255)),
    )
    assert out == {0: "Text-only succeeded!"}
    assert vision_flags == [None, False]
    assert llama_mgr.calls == []


@pytest.mark.asyncio
async def test_pipeline_lazy_llm_startup_cache_mode_isolation_and_provider_events(tmp_path: Path):
    assert CACHE_KEY_VERSION == "v6-hybrid-multi"
    memory = ChapterMemoryStore()
    cache = TranslationCache(tmp_path / "translations.db", max_entries=50)

    ensure_llm_calls = 0

    async def fake_ensure_llm_ready() -> None:
        nonlocal ensure_llm_calls
        ensure_llm_calls += 1

    class _Detector:
        def detect(self, image: Image.Image):
            return [RawDetection(Box(x1=360, y1=60, x2=540, y2=220), "bubble", 0.95)]

    class _Ocr:
        def read_page_bubbles(self, crops, lang_hint="auto", aspect_ratio=1.0):
            return "ja", ["藍星……？"]

    class _FakeGemini:
        def is_configured(self) -> bool:
            return True

        def _active_models(self) -> list[str]:
            return ["gemini-3.5-flash-lite"]

        async def translate_chunk(self, bubbles, lang, **kwargs):
            assert kwargs.get("image") is not None
            assert kwargs.get("bubble_boxes") is not None
            return {bid: "Ranshin...? (Gemini)" for bid, _ in bubbles}, "gemini-3.5-flash-lite"

    async def fake_local(messages):
        return '{"0": "Ranshin...? (Local)"}'

    cfg = SimpleNamespace(translation_mode="hybrid", gemini_chunk_size=20, llm_chunk_size=12)
    translator = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=fake_local,
        cfg=cfg,
        gemini_translator=_FakeGemini(),  # type: ignore[arg-type]
    )
    pipeline = TranslationPipeline(
        detector=_Detector(),
        ocr=_Ocr(),
        translator=translator,
        memory=memory,
        cache=cache,
        ensure_llm_ready=fake_ensure_llm_ready,
        cfg=cfg,
    )
    await pipeline.start()

    img_bytes = _make_sample_png_bytes()

    # 1. Hybrid mode with Gemini key -> ensure_llm_ready NOT called upfront!
    events = [
        ev
        async for ev in pipeline.translate_image_stream(
            img_bytes, lang="ja", chapter_id="ch-pipe", translation_mode="hybrid"
        )
    ]
    assert ensure_llm_calls == 0
    translate_statuses = [e for e in events if e["type"] == "status" and e["stage"] == "translate"]
    assert translate_statuses == [
        {"type": "status", "stage": "translate", "provider": "gemini:gemini-3.5-flash-lite"}
    ]
    done_resp = events[-1]["response"]
    assert done_resp["provider"] == "gemini:gemini-3.5-flash-lite"
    assert done_resp["bubbles"][0]["translated"] == "Ranshin...? (Gemini)"

    # 2. Local override on same image -> distinct cache key (cache miss) AND calls ensure_llm_ready!
    local_resp = await pipeline.translate_image(
        img_bytes, lang="ja", chapter_id="ch-pipe", translation_mode="local"
    )
    assert ensure_llm_calls == 1
    assert local_resp.cached is False
    assert local_resp.provider == "local:ja"
    assert local_resp.bubbles[0].translated == "Ranshin...? (Local)"

    # 3. Hybrid request hits Hybrid cache (not Local cache!)
    hybrid_cached = await pipeline.translate_image(
        img_bytes, lang="ja", chapter_id="ch-pipe", translation_mode="hybrid"
    )
    assert hybrid_cached.cached is True
    assert hybrid_cached.bubbles[0].translated == "Ranshin...? (Gemini)"

    await pipeline.stop()


@pytest.mark.asyncio
async def test_dynamic_hybrid_partial_chunk_refusal_shifts_only_missing_bubble_to_local():
    memory = ChapterMemoryStore()
    llama_mgr = _FakeLlamaManager()
    local_seen_ids: list[list[int]] = []

    class _PartialGemini:
        def is_configured(self) -> bool:
            return True

        def _active_models(self) -> list[str]:
            return ["gemini-3.5-flash-lite"]

        async def translate_chunk(self, bubbles, lang, **kwargs):
            # Translates normal bubbles 0 and 1, omits explicit bubble 2
            return {0: "Wait a second!", 1: "Are you sure?"}, "gemini-3.5-flash-lite"

    async def fake_local(messages):
        user_msg = messages[1]["content"]
        in_map = json.loads(user_msg[user_msg.rfind("{") :])
        local_seen_ids.append([int(k) for k in in_map])
        return json.dumps({k: "I want to put it in." for k in in_map})

    cfg = SimpleNamespace(translation_mode="hybrid", gemini_chunk_size=20, llm_chunk_size=12)
    tr = DialogueTranslator(
        "http://127.0.0.1:8766",
        memory,
        llm_caller=fake_local,
        cfg=cfg,
        llama_mgr=llama_mgr,
        gemini_translator=_PartialGemini(),  # type: ignore[arg-type]
    )
    out = await tr.translate_page(
        [(0, "ちょっと待って！"), (1, "本当に？"), (2, "挿れたい")],
        lang="ja",
        chapter_id="ch-partial",
    )
    assert out == {0: "Wait a second!", 1: "Are you sure?", 2: "I want to put it in."}
    assert local_seen_ids == [[2]]
    assert tr.last_provider == "hybrid"
    assert llama_mgr.calls == [("ja", False)]

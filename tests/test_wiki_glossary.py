"""Unit tests for server/translator/wiki_glossary.py using httpx.MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from server.translator.wiki_glossary import (
    USER_AGENT,
    GlossaryLookupError,
    InvalidLookupQueryError,
    lookup_character_glossary,
)


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_anilist_title_search_normalizes_and_disambiguates_safely():
    captured_requests: list[tuple[str, dict[str, Any], str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append((str(request.url), body, request.headers.get("user-agent", "")))
        return httpx.Response(
            200,
            json={
                "data": {
                    "Media": {
                        "title": {
                            "romaji": "Kimetsu no Yaiba",
                            "english": "Demon Slayer: Kimetsu no Yaiba",
                            "native": "鬼滅の刃",
                        },
                        "siteUrl": "https://anilist.co/manga/87216",
                        "characters": {
                            "nodes": [
                                {"name": {"full": "Tanjiro Kamado", "native": "竈門\u3000炭治郎", "alternative": []}},
                                {"name": {"full": "Nezuko Kamado", "native": "竈門 禰豆子", "alternative": []}},
                                # Single isolated 2-part katakana name: keep full entry without guessing splits
                                {"name": {"full": "Eren Yeager", "native": "エレン イェーガー", "alternative": []}},
                                # Missing native CJK -> skipped
                                {"name": {"full": "No Native", "native": "", "alternative": []}},
                                # Non-CJK native -> skipped
                                {"name": {"full": "Latin Only", "native": "Latin Only", "alternative": []}},
                            ]
                        },
                    }
                }
            },
        )

    async with _mock_client(handler) as client:
        result = await lookup_character_glossary("Demon Slayer", source="anilist", client=client)

    assert len(captured_requests) == 1
    url, body, ua = captured_requests[0]
    assert url == "https://graphql.anilist.co"
    assert body["variables"] == {"search": "Demon Slayer"}
    assert ua == USER_AGENT

    assert result["title"] == "Demon Slayer: Kimetsu no Yaiba"
    assert result["source"] == "anilist"
    assert result["url"] == "https://anilist.co/manga/87216"
    assert result["warnings"] == []
    assert result["entries"]["竈門 炭治郎"] == "Tanjiro Kamado"
    assert result["entries"]["竈門 禰豆子"] == "Nezuko Kamado"
    assert result["entries"]["エレン イェーガー"] == "Eren Yeager"
    # Shared surname disambiguated cleanly without touching isolated Western/katakana pair
    assert result["entries"]["竈門"] == "Kamado"
    assert result["entries"]["炭治郎"] == "Tanjiro"
    assert result["entries"]["禰豆子"] == "Nezuko"
    assert "エレン" not in result["entries"]
    assert "No Native" not in result["entries"].values()
    assert result["count"] == len(result["entries"])


@pytest.mark.asyncio
async def test_anilist_direct_url_queries_by_id():
    seen_variables: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_variables.append(body["variables"])
        assert "Media(id: $id, type: MANGA)" in body["query"]
        return httpx.Response(
            200,
            json={
                "data": {
                    "Media": {
                        "title": {"romaji": "Chainsaw Man", "english": None, "native": "チェンソーマン"},
                        "siteUrl": "https://anilist.co/manga/12345",
                        "characters": {
                            "nodes": [
                                {"name": {"full": "Denji", "native": "デンジ", "alternative": []}},
                                {"name": {"full": "Makima", "native": "マキマ", "alternative": []}},
                            ]
                        },
                    }
                }
            },
        )

    async with _mock_client(handler) as client:
        result = await lookup_character_glossary("https://anilist.co/manga/12345/Chainsaw-Man", client=client)

    assert seen_variables == [{"id": 12345}]
    assert result["title"] == "Chainsaw Man"
    assert result["source"] == "anilist"
    assert result["entries"] == {"デンジ": "Denji", "マキマ": "Makima"}
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_wikipedia_title_search_wikitext_and_parenthetical_lists():
    seen_params: list[dict[str, str]] = []

    wikitext = """
{{Infobox animanga/Header}}
'''Chainsaw Man''' ({{nihongo||チェンソーマン}}) is a Japanese manga series written by Tatsuki Fujimoto.

== Characters ==
=== Main characters ===
; {{nihongo|[[Denji (Chainsaw Man)|Denji]]|デンジ}}
: Voiced by: Kikunosuke Toya (Japanese: 戸谷 菊之介)
; '''Makima''' ({{nihongo||マキマ|Makima}})
: Leader of Public Safety.
* '''Power''' (Japanese: パワー, Hepburn: Pawā)
* Aki Hayakawa (早川 アキ / はやかわ アキ)
* Sung Jin-woo (Korean: 성진우, RR: Seong Jin-u)
* '''Himmel''' ({{Korean|hangul=힘멜}})

== Publication ==
Published by Shueisha (集英社) in Weekly Shonen Jump (週刊少年ジャンプ).
"""

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen_params.append(params)
        assert request.url.host == "en.wikipedia.org"
        if params.get("list") == "search":
            return httpx.Response(
                200,
                json={
                    "query": {
                        "search": [
                            {"title": "Chainsaw Man"},
                            {"title": "List of Chainsaw Man characters"},
                        ]
                    }
                },
            )
        assert params.get("action") == "parse"
        assert params.get("page") == "List of Chainsaw Man characters"
        return httpx.Response(
            200,
            json={
                "parse": {
                    "title": "List of Chainsaw Man characters",
                    "wikitext": {"*": wikitext},
                    "text": {"*": ""},
                }
            },
        )

    async with _mock_client(handler) as client:
        result = await lookup_character_glossary("Chainsaw Man", source="wikipedia", client=client)

    assert len(seen_params) == 2
    assert result["source"] == "wikipedia"
    assert result["title"] == "List of Chainsaw Man characters"
    assert result["entries"]["デンジ"] == "Denji"
    assert result["entries"]["マキマ"] == "Makima"
    assert result["entries"]["パワー"] == "Power"
    assert result["entries"]["早川 アキ"] == "Aki Hayakawa"
    assert result["entries"]["성진우"] == "Sung Jin-woo"
    assert result["entries"]["힘멜"] == "Himmel"
    # Voice actors and publisher sections must NOT pollute character glossary
    assert "戸谷 菊之介" not in result["entries"]
    assert "集英社" not in result["entries"]
    assert "チェンソーマン" not in result["entries"]
    assert result["count"] == 6


@pytest.mark.asyncio
async def test_wikipedia_direct_mobile_url_html_fallback():
    def handler(request: httpx.Request) -> httpx.Response:
        # Mobile .m.wikipedia.org should normalize to canonical API host
        assert request.url.host == "en.wikipedia.org"
        assert request.url.params["page"] == "Solo Leveling"
        return httpx.Response(
            200,
            json={
                "parse": {
                    "title": "Solo Leveling",
                    "wikitext": {"*": ""},
                    "text": {
                        "*": (
                            "<div><p><b>Sung Jinwoo</b> (Korean: <span>성진우</span>)</p>"
                            "<dl><dt>Cha Hae-in</dt><dd>(Korean: 차해인)</dd></dl></div>"
                        )
                    },
                }
            },
        )

    async with _mock_client(handler) as client:
        result = await lookup_character_glossary("https://en.m.wikipedia.org/wiki/Solo_Leveling", client=client)

    assert result["source"] == "wikipedia"
    assert result["entries"] == {"성진우": "Sung Jinwoo", "차해인": "Cha Hae-in"}


@pytest.mark.asyncio
async def test_fandom_url_parses_infobox_and_character_list():
    wikitext = """
{{Character
| name = [[Denji (Manga)|Denji]]
| kanji = デンジ<ref>{{cite web|url=https://example.com|title=Bio}}</ref><br/>チェンソーマン
| romaji = Denji
}}
== Associates ==
* '''[[Power]]''' (パワー)
* '''Sung Jinwoo''' (Korean: 성진우)
{| class="wikitable"
| [[Himeno]] || 姫野
|}
"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "chainsaw-man.fandom.com"
        assert request.url.path == "/api.php"
        assert request.url.params["action"] == "parse"
        assert request.url.params["page"] == "Denji"
        return httpx.Response(
            200,
            json={
                "parse": {
                    "title": "Denji",
                    "wikitext": {"*": wikitext},
                }
            },
        )

    async with _mock_client(handler) as client:
        result = await lookup_character_glossary("https://chainsaw-man.fandom.com/wiki/Denji", client=client)

    assert result["source"] == "fandom"
    assert result["title"] == "Denji"
    assert result["url"] == "https://chainsaw-man.fandom.com/wiki/Denji"
    assert result["entries"]["デンジ"] == "Denji"
    assert result["entries"]["パワー"] == "Power"
    assert result["entries"]["성진우"] == "Sung Jinwoo"
    assert result["entries"]["姫野"] == "Himeno"
    assert "チェンソーマン" not in result["entries"]
    assert result["count"] == 4


@pytest.mark.asyncio
async def test_fandom_infobox_korean_hangul_and_default_page_title():
    wikitext = """
{{Infobox character
| hangul = 성진우
| hanja = 成振宇
}}
"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"parse": {"title": "Sung Jinwoo", "wikitext": {"*": wikitext}}},
        )

    async with _mock_client(handler) as client:
        result = await lookup_character_glossary("https://solo-leveling.fandom.com/wiki/Sung_Jinwoo", client=client)

    assert result["entries"]["성진우"] == "Sung Jinwoo"
    assert result["entries"]["成振宇"] == "Sung Jinwoo"


@pytest.mark.asyncio
async def test_fandom_infobox_markup_only_and_leading_br_english_fields():
    wikitext = """
{{Infobox character
| name = <span></span><!-- hidden --><ref>ignored</ref>
| title = <br/>Denji
| kanji = デンジ
}}
{{Infobox character
| name = <br/>
| kanji = パワー
}}
"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"parse": {"title": "Power (Manga)", "wikitext": {"*": wikitext}}},
        )

    async with _mock_client(handler) as client:
        result = await lookup_character_glossary("https://chainsaw-man.fandom.com/wiki/Power", client=client)

    assert result["entries"]["デンジ"] == "Denji"
    assert result["entries"]["パワー"] == "Power"


@pytest.mark.asyncio
async def test_auto_falls_back_to_wikipedia_when_anilist_returns_zero_hits():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "graphql.anilist.co":
            return httpx.Response(200, json={"data": {"Media": None}})
        if request.url.host == "en.wikipedia.org":
            if request.url.params.get("list") == "search":
                return httpx.Response(200, json={"query": {"search": [{"title": "Obscure Manga"}]}})
            return httpx.Response(
                200,
                json={
                    "parse": {
                        "title": "Obscure Manga",
                        "wikitext": {"*": "* '''Taro Yamada''' ({{nihongo||山田 太郎}})"},
                        "text": {"*": ""},
                    }
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    async with _mock_client(handler) as client:
        result = await lookup_character_glossary("Obscure Manga", source="auto", client=client)

    assert result["source"] == "wikipedia"
    assert result["entries"] == {"山田 太郎": "Taro Yamada"}
    assert any("Wikipedia" in w for w in result["warnings"])


@pytest.mark.asyncio
async def test_auto_falls_back_to_wikipedia_when_anilist_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "graphql.anilist.co":
            return httpx.Response(503, text="Service Unavailable")
        if request.url.host == "en.wikipedia.org":
            if request.url.params.get("list") == "search":
                return httpx.Response(200, json={"query": {"search": [{"title": "Fallback Series"}]}})
            return httpx.Response(
                200,
                json={
                    "parse": {
                        "title": "Fallback Series",
                        "wikitext": {"*": "* '''Hanako''' (花子)"},
                        "text": {"*": ""},
                    }
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    async with _mock_client(handler) as client:
        result = await lookup_character_glossary("Fallback Series", source="auto", client=client)

    assert result["source"] == "wikipedia"
    assert result["entries"] == {"花子": "Hanako"}
    assert any("AniList lookup failed" in w for w in result["warnings"])


@pytest.mark.asyncio
async def test_auto_raises_when_both_anilist_and_wikipedia_fail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    async with _mock_client(handler) as client:
        with pytest.raises(GlossaryLookupError, match="Wikipedia fallback failed"):
            await lookup_character_glossary("Anything", source="auto", client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_query",
    [
        "",
        "   \t\n  ",
        "http://en.wikipedia.org/wiki/Chainsaw_Man",
        "http://anilist.co/manga/12345",
        "https://localhost/wiki/Chainsaw_Man",
        "https://127.0.0.1/wiki/Chainsaw_Man",
        "https://169.254.169.254/latest/meta-data",
        "https://evil.com/wiki/Chainsaw_Man",
        "https://en.wikipedia.org.evil.com/wiki/Chainsaw_Man",
        "https://evil-fandom.com/wiki/Denji",
        "https://anilist.co@127.0.0.1/manga/12345",
        "https://en.wikipedia.org:8443/wiki/Chainsaw_Man",
        "file:///etc/passwd",
        "ftp://en.wikipedia.org/wiki/Chainsaw_Man",
        "https://anilist.co/anime/12345",
        "https://en.wikipedia.org/",
        "https://chainsaw-man.fandom.com/",
    ],
)
async def test_ssrf_and_invalid_queries_rejected(bad_query: str):
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    async with _mock_client(handler) as client:
        with pytest.raises(InvalidLookupQueryError) as exc_info:
            await lookup_character_glossary(bad_query, client=client)
        assert isinstance(exc_info.value, ValueError)

    assert called is False


@pytest.mark.asyncio
async def test_mismatched_source_and_fandom_plain_query_rejected():
    async with _mock_client(lambda req: httpx.Response(200, json={})) as client:
        with pytest.raises(InvalidLookupQueryError):
            await lookup_character_glossary("https://en.wikipedia.org/wiki/Denji", source="anilist", client=client)
        with pytest.raises(InvalidLookupQueryError):
            await lookup_character_glossary("Chainsaw Man", source="fandom", client=client)
        with pytest.raises(InvalidLookupQueryError):
            await lookup_character_glossary("Chainsaw Man", source="unknown", client=client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_default_client_config_sets_timeout_user_agent_and_no_redirects(monkeypatch):
    seen_kwargs: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "Media": {
                        "title": {"english": "Test", "romaji": "Test", "native": "テスト"},
                        "siteUrl": "https://anilist.co/manga/1",
                        "characters": {"nodes": [{"name": {"full": "Denji", "native": "デンジ"}}]},
                    }
                }
            },
        )

    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs: Any) -> httpx.AsyncClient:
        seen_kwargs.update(kwargs)
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    result = await lookup_character_glossary("Test", source="anilist")

    assert seen_kwargs["timeout"] == 10.0
    assert seen_kwargs["follow_redirects"] is False
    assert seen_kwargs["headers"]["User-Agent"] == USER_AGENT
    assert result["entries"] == {"デンジ": "Denji"}


"""Per-chapter rolling dialogue context and pinned character-name glossary store."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import threading
import unicodedata
from collections import deque

_CJK_SCRIPT_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u309f\u30a0-\u30ff\uff65-\uff9f"
    r"\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]"
)
_HIRAGANA_RE = re.compile(r"[\u3041-\u3096]+")
_KATAKANA_ONLY_SFX_RE = re.compile(r"^[\u30a1-\u30fa\u30fc\s…・.!?！？ー\-~～]+$")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_JP_MORA_WORD_RE = re.compile(
    r"^(?:(?:sh|ch|ts|ky|gy|ny|hy|my|ry|by|py|jy|fu|[kgsztdnhbpmyrwjfvc])?[aeiou]+|n(?!$)|n$)+$"
)

# Minimal kana->Hepburn table used ONLY internally for detecting untranslated Romaji echoes.
_DETECT_KANA_DIGRAPHS: dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "ぢゃ": "ja", "ぢゅ": "ju", "ぢょ": "jo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "キャ": "kya", "キュ": "kyu", "キョ": "kyo",
    "シャ": "sha", "シュ": "shu", "ショ": "sho",
    "チャ": "cha", "チュ": "chu", "チョ": "cho",
    "ニャ": "nya", "ニュ": "nyu", "ニョ": "nyo",
    "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo",
    "ミャ": "mya", "ミュ": "myu", "ミョ": "myo",
    "リャ": "rya", "リュ": "ryu", "リョ": "ryo",
    "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo",
    "ジャ": "ja", "ジュ": "ju", "ジョ": "jo",
    "ビャ": "bya", "ビュ": "byu", "ビョ": "byo",
    "ピャ": "pya", "ピュ": "pyu", "ピョ": "pyo",
}

_DETECT_KANA_MONOS: dict[str, str] = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "wo", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo",
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヲ": "wo", "ン": "n",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
    "ャ": "ya", "ュ": "yu", "ョ": "yo",
}

# Unambiguous Japanese & Korean Romaji spoken words (zero overlap with valid English words/names).
_STRONG_ROMAJI_WORDS: frozenset[str] = frozenset(
    {
        # Japanese dialogue, verbs, adjectives, adverbs, interjections
        "chotto", "matte", "mattero", "mattete", "yada", "iyada", "yamete", "yamero", "yameru",
        "kudasai", "ikko", "ikou", "ikuzo", "ikimashou", "ikimasu", "ittekimasu", "itterasshai",
        "itadakimasu", "gochisousama", "otsukaresama", "yoroshiku", "onegaishimasu", "onegai",
        "soko", "asoko", "doko", "konna", "sonna", "donna",
        "nani", "nande", "naze", "doushite", "doshite", "dareka", "nanika", "dokoka", "itsuka",
        "daijoubu", "daijobu", "hontou", "honto", "hontouni", "hontoni", "majide", "yappari",
        "yahari", "sasuga", "masaka", "zettai", "kanarazu", "zenzen", "mattaku", "sukoshi",
        "ichiban", "takusan", "chanto", "shikkari", "yukkuri", "betsuni", "tonikaku", "kekkyoku",
        "jissai", "naruhodo", "mochiron", "tashika", "tabun", "kitto", "sugu", "suguni",
        "tokidoki", "itsumo", "mouichido",
        "desu", "deshou", "desho", "deshita", "masu", "mashita", "masen", "janai", "janaika",
        "darou", "daro", "kamoshirenai", "dakedo", "dakara", "keredo", "keredomo", "shikashi",
        "soshite", "sorede", "sorekara", "soretomo", "tatoeba", "nanka", "nandaka", "nantoka",
        "muri", "yabai", "yabe", "yabee", "sugoi", "sugee", "subarashii", "kowai", "itai", "ittai",
        "atsui", "samui", "tsurai", "ureshii", "kanashii", "tanoshii", "oishii", "mazui", "hayai",
        "hayaku", "osoi", "chikai", "tooi", "takai", "hikui", "ookii", "chiisai", "atarashii",
        "furui", "warui", "yasashii", "muzukashii", "abunai", "urusai", "uzai", "kimoi", "jouzu",
        "heiki",
        "wakatta", "wakarimashita", "wakaranai", "wakaru", "wakaran", "shiranai", "shiran",
        "shitteru", "shitteiru", "mieru", "mienai", "kikoeru", "kikoenai", "dekinai", "dekiru",
        "arimasu", "arimasen", "nakatta", "yatta", "yarou", "shiteru", "shiteiru", "shimasu",
        "shimashita", "taberu", "tabete", "nomu", "neru", "okiru", "hashiru", "aruku", "kaeru",
        "kaette", "modoru", "tsukau", "tsukuru", "omou", "omotta", "kangaeru", "wasureru",
        "wasureta", "oboeru", "oshieru", "asobu", "hataraku", "tasukete", "tasukeru", "mamoru",
        "tatakau", "korosu", "shinde", "shinda", "nigeru", "nigero", "tsukamaeru", "mitsukeru",
        "sagasu", "yurusanai", "yurushite", "ayamaru", "tanomu", "damare", "ganbare", "ganbatte",
        "gambare", "gambatte", "oshiete", "misete", "kiite", "hanashite", "yonde", "kaite",
        "arigatou", "arigato", "sumimasen", "suimasen", "gomen", "gomennasai", "gomenasai",
        "shitsurei", "ohayou", "ohayo", "konnichiwa", "konbanwa", "oyasumi", "sayonara",
        "sayounara", "tadaima", "okaeri", "okaerinasai",
        "watashi", "atashi", "boku", "anata", "omae", "kisama", "temee", "teme", "minna", "zenbu",
        "subete", "hitori", "futari", "daisuki", "kirai", "daikirai", "aishiteru", "aishiteiru",
        "tomodachi", "kazoku", "otousan", "okaasan", "oniisan", "oneesan", "oniichan", "oneechan",
        "otouto", "imouto", "ojiisan", "obaasan", "kodomo", "otona", "otoko", "onna", "ningen",
        "kamisama", "mahou", "chikara", "kokoro", "karada", "atama", "kotoba", "namae", "basho",
        "jikan", "sekai", "zetsubou", "shiawase", "himitsu", "yakusoku", "shinjitsu", "riyuu",
        "mondai", "kotae",
        "kuso", "kusottare", "chikushou", "chikusho", "shimatta", "yareyare", "usoda", "baka",
        "bakayaro", "bakayarou", "manuke", "hentai", "sukebe", "koraa", "koraaa", "bikkuri",
        "mushamu", "musha", "mushamusha", "dokidoki", "gorogoro", "pikapika", "fuwafuwa",
        "nikoniko", "wakuwaku", "iraira", "pachipachi", "kasakasa", "zawazawa", "gatagata",
        "patapata", "guragura", "buruburu", "furufuru", "jabujabu", "shiiin",
        # Korean romanization
        "jebal", "gwaenchana", "gwaenchanha", "mianhae", "mianhamnida", "joesonghamnida",
        "saranghae", "saranghaeyo", "hajima", "hajimara", "andwae", "andwaeyo", "gajima", "babo",
        "michin", "michyeosseo", "jinjja", "daebak", "aigo", "aigoo", "eotteoke", "eotteokhae",
        "molla", "mollayo", "alasseo", "algesseoyo", "gomawo", "gomapseumnida", "kamsahamnida",
        "gamsahamnida", "annyeong", "annyeonghaseyo", "wae", "mwo", "mwoya", "geurae", "geureom",
        "geuman", "ppalli", "jamkkan", "jamkkanman", "yeogi", "geogi", "jeogi", "nugu", "eodi",
        "eonje", "museun", "aniyo", "aniya", "jeongmal", "heol", "ssibal", "shibal", "saekki",
        "jugeo", "salryeojwo",
    }
)

# Romaji bigrams catching short/ambiguous words in unmistakable Japanese/Korean syntax.
_ROMAJI_BIGRAMS: frozenset[tuple[str, str]] = frozenset(
    {
        ("koko", "ja"), ("koko", "wa"), ("koko", "ni"), ("koko", "de"),
        ("sore", "wa"), ("sore", "ja"), ("sore", "ga"), ("sore", "ni"), ("sore", "wo"),
        ("kore", "wa"), ("kore", "ja"), ("kore", "ga"), ("kore", "ni"), ("kore", "wo"),
        ("are", "wa"), ("are", "ga"), ("wa", "dame"), ("dame", "da"), ("dame", "yo"),
        ("dame", "desu"), ("dame", "na"), ("ja", "dame"), ("ja", "nai"),
        ("suki", "da"), ("suki", "yo"), ("suki", "desu"), ("suki", "na"),
        ("maji", "de"), ("uso", "da"), ("uso", "desho"), ("uso", "deshou"),
        ("ore", "wa"), ("ore", "no"), ("ore", "ga"), ("ore", "ni"), ("ore", "wo"),
        ("kimi", "wa"), ("kimi", "no"), ("kimi", "ga"), ("kimi", "ni"),
        ("mou", "ii"), ("mada", "mada"), ("ii", "yo"), ("ii", "ne"), ("ii", "desu"),
        ("sou", "da"), ("sou", "desu"), ("sou", "ka"), ("sou", "ne"), ("sou", "yo"),
        ("jaa", "ne"), ("mata", "ne"), ("demo", "ne"), ("demo", "sa"),
        ("kono", "yaro"), ("kono", "yarou"), ("sono", "toori"),
        ("mite", "kure"), ("mite", "kuro"), ("shite", "kure"), ("yatte", "kure"),
        ("nani", "mo"), ("nani", "ka"), ("nani", "ga"), ("nani", "wo"),
        ("dare", "mo"), ("dare", "ga"), ("dare", "da"),
        ("wae", "irae"), ("wae", "geurae"), ("ani", "geuge"),
    }
)

# Non-English Japanese particles/weak Romaji stems (safe when paired in >=2 non-English Romaji context).
_WEAK_ROMAJI_TOKENS: frozenset[str] = frozenset(
    {
        "koko", "kore", "dore", "kono", "sono", "suki", "kimi", "mina", "kuro", "shiro",
        "akai", "aoi", "mite", "mita", "miru", "itta", "itte", "iku", "kita", "kite", "kuru",
        "yaru", "suru", "shita", "shite", "nai", "aru", "iru", "naru", "kara", "koto", "mono",
        "hito", "toki", "naka", "mae", "mou", "mada", "tada", "nara", "kora", "kasa", "jabu",
        "pata", "gata", "zawa", "doki", "biku", "piku", "gura", "buru", "furu", "yoshi", "yosha",
        "iie", "iya", "hai", "wa", "ga", "wo", "ni", "de", "ka", "da", "ja", "ne", "zo", "ze",
    }
)

# Japanese SFX romaji stems rejected when the source bubble is short Katakana SFX.
_SFX_ROMAJI_WORDS: frozenset[str] = frozenset(
    {
        "kasa", "gata", "batan", "pata", "zawa", "doki", "gura", "buru", "jabu", "kora",
        "musha", "piku", "biku", "zaza", "dosa", "basa", "gasa", "pachi", "kachi", "gacha",
    }
)

# Common English words and accepted English manga loanwords that can match Japanese CV phonotactics.
_ENGLISH_OR_LOANWORD_MORAIC: frozenset[str] = frozenset(
    {
        "a", "i", "to", "no", "so", "do", "go", "me", "he", "we", "be", "hi", "ha", "ho",
        "ah", "oh", "uh", "yo", "ok", "too", "see", "tea", "sea", "bee", "fee", "pee", "wee",
        "are", "sore", "demo", "made", "dame", "ore", "motto", "mama", "papa", "data", "area",
        "idea", "menu", "hero", "zero", "auto", "solo", "echo", "foto", "halo", "logo", "memo",
        "nano", "pogo", "polo", "silo", "taco", "tofu", "miso", "sake", "soda", "sofa", "toga",
        "tuna", "yoyo", "banana", "tomato", "potato", "cinema", "camera", "pajama", "panama",
        "karate", "kimono", "katana", "sudoku", "tsunami", "origami", "karaoke", "teriyaki",
        "wasabi", "edamame", "matcha", "mochi", "ramen", "sushi", "bento", "dojo", "anime",
        "manga", "ninja", "samurai", "sensei", "senpai", "tokyo", "kyoto", "osaka", "yakuza",
        "shogun", "sumo", "judo", "kendo", "ronin", "oni", "yokai", "kami", "koi", "futon",
        "tatami", "onsen", "ryokan", "bonsai", "haiku", "banzai", "tempura", "udon", "soba",
        "gyoza", "katsu", "sashimi", "yakitori", "kawaii", "chibi", "otaku", "mecha", "isekai",
        "here", "there", "where", "come", "some", "home", "time", "name", "same", "game",
        "take", "make", "like", "fine", "mine", "line", "nine", "more", "before", "sure",
        "care", "dare", "rare", "bare", "fare", "hire", "fire", "wire", "tire", "pure",
        "cute", "mute", "note", "vote", "hope", "rope", "nope", "cope", "dope", "tone",
        "bone", "cone", "zone", "done", "none", "gone", "one", "two", "you", "who", "how",
        "now", "wow", "hey", "yes", "yet", "not", "but", "out", "for", "way", "say", "day",
        "may", "pay", "big", "bad", "mad", "sad", "man", "men", "can", "ran", "run", "fun",
        "sun", "son", "won", "got", "get", "let", "set", "met", "bet", "pet", "wet", "yet",
        "did", "had", "has", "was", "his", "her", "she", "the", "and", "all", "our", "its",
        "what", "wait", "want", "wanna", "gonna", "gotta", "okay", "yeah", "nope", "wait",
        "little", "second", "bed", "please", "sorry", "thank", "thanks", "hello", "goodbye",
    }
)


def _kana_to_romaji_lower(text: str) -> str:
    """Internal read-only helper converting kana spans in source text to lowercase Hepburn."""
    normalized = unicodedata.normalize("NFKC", text)
    tokens: list[str] = []
    i = 0
    n = len(normalized)
    while i < n:
        ch = normalized[i]
        if ch in ("っ", "ッ"):
            tokens.append("_GEM_")
            i += 1
            continue
        if ch == "ー":
            tokens.append("_LONG_")
            i += 1
            continue
        if i + 1 < n:
            digraph = _DETECT_KANA_DIGRAPHS.get(normalized[i : i + 2])
            if digraph is not None:
                tokens.append(digraph)
                i += 2
                continue
        mono = _DETECT_KANA_MONOS.get(ch)
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
            if out and out[-1][-1] in "aeiou":
                out.append(out[-1][-1])
        else:
            out.append(tok)
    return "".join(out)


def contains_non_english_cjk(text: str) -> bool:
    """Return True if text contains Chinese, Japanese (Kana/Kanji), or Korean (Hangul) characters."""
    if not text:
        return False
    return _CJK_SCRIPT_RE.search(text) is not None


def looks_like_romaji(orig: str, candidate: str) -> bool:
    """Return True if candidate appears to be Japanese/Korean Romaji instead of English."""
    if not orig or not candidate:
        return False
    if not contains_non_english_cjk(orig):
        return False

    cleaned = candidate.strip()
    if not cleaned:
        return False

    raw_words = _LATIN_WORD_RE.findall(cleaned)
    if not raw_words:
        return False
    words = [w.lower() for w in raw_words]

    # Signal 1: Any unambiguous Japanese/Korean Romaji spoken word (including elongated vowels).
    for w in words:
        if w in _STRONG_ROMAJI_WORDS:
            return True
        if len(w) >= 5:
            collapsed = re.sub(r"([aeiou])\1+$", r"\1", w)
            if collapsed != w and (collapsed in _STRONG_ROMAJI_WORDS or collapsed == "kora"):
                return True

    # Signal 2: Unmistakable Japanese/Korean Romaji syntactic bigrams (e.g., "sore wa", "koko ja").
    for i in range(len(words) - 1):
        if (words[i], words[i + 1]) in _ROMAJI_BIGRAMS:
            return True

    # Signal 3: Katakana SFX directly transliterated (e.g., "カサ..." -> "Kasa...").
    if len(words) <= 2 and _KATAKANA_ONLY_SFX_RE.match(orig.strip()):
        if any(w in _SFX_ROMAJI_WORDS for w in words):
            return True

    # Signal 4: Direct Hiragana phonetic echo (multi-word candidate mirroring source Hiragana).
    hiragana_spans = _HIRAGANA_RE.findall(orig)
    hiragana_joined = "".join(hiragana_spans)
    if len(hiragana_joined) >= 4 and len(words) >= 2:
        orig_hira_romaji = _kana_to_romaji_lower(hiragana_joined)
        cand_alpha = "".join(words)
        if len(orig_hira_romaji) >= 5 and len(cand_alpha) >= 5:
            ratio = SequenceMatcher(None, orig_hira_romaji, cand_alpha).ratio()
            if ratio >= 0.72:
                return True

    # Signal 5: Multi-word Japanese moraic transliteration (e.g., "Ichigame Mite Kuro...coms").
    if len(words) >= 3 and "," not in cleaned:
        moraic_words = [
            w for w in words if len(w) >= 2 and "'" not in w and _JP_MORA_WORD_RE.match(w) is not None
        ]
        non_english_moraic = [w for w in moraic_words if w not in _ENGLISH_OR_LOANWORD_MORAIC]
        has_weak_marker = any(w in _WEAK_ROMAJI_TOKENS for w in words)
        if len(moraic_words) >= 3 and len(non_english_moraic) >= 2 and has_weak_marker:
            return True

    return False


_SFX_SPAN_RE = re.compile(r"\*[^*]+\*")
_HONORIFIC_NAME_RE = re.compile(
    r"\b([A-Z][a-z]{2,})-(?:chan|kun|san|sama|senpai|sensei|ssi|nim|hyung|oppa|noona|unnie)\b"
)
_MID_SENTENCE_CAP_RE = re.compile(
    r"(?:(?:^|[\s,(])[a-z]+\s+|,\s+)([A-Z][a-z]{2,})\b"
)
_SENTENCE_START_CAP_RE = re.compile(r"(?:^|[.!?]\s+)([A-Z][a-z]{2,})\b")
_REPEATED_TRIPLE_RE = re.compile(r"(.)\1\1", re.IGNORECASE)

# Common English words that may appear capitalized in dialogue but are not character names.
_COMMON_ENGLISH_NON_NAMES: frozenset[str] = frozenset(
    {
        "the", "and", "but", "for", "nor", "not", "yet", "you", "your", "yours", "she", "her",
        "hers", "his", "him", "its", "our", "ours", "they", "them", "their", "theirs", "this",
        "that", "these", "those", "here", "there", "where", "when", "what", "which", "who",
        "whom", "whose", "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "only", "own", "same", "than", "too", "very", "just", "now",
        "then", "once", "ever", "never", "always", "still", "even", "also", "too", "again",
        "well", "yes", "yeah", "yep", "nah", "nope", "okay", "sure", "fine", "good", "great",
        "nice", "bad", "sorry", "please", "thanks", "thank", "hello", "hey", "wait", "stop",
        "hold", "come", "look", "listen", "shut", "move", "run", "get", "let", "don", "can",
        "won", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn", "couldn", "wouldn",
        "shouldn", "mustn", "didn", "doesn", "ain", "really", "actually", "honestly", "maybe",
        "probably", "perhaps", "definitely", "totally", "seriously", "literally", "basically",
        "because", "though", "although", "however", "anyway", "besides", "meanwhile", "instead",
        "nothing", "something", "anything", "everything", "someone", "anyone", "everyone",
        "nobody", "somebody", "anybody", "everybody", "somewhere", "anywhere", "everywhere",
        "nowhere", "myself", "yourself", "himself", "herself", "itself", "ourselves",
        "themselves", "today", "tomorrow", "yesterday", "morning", "afternoon", "evening",
        "night", "midnight", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "english", "japanese", "korean",
        "chinese", "america", "american", "japan", "korea", "china", "tokyo", "kyoto", "osaka",
        "seoul", "god", "lord", "jesus", "christ", "hell", "damn", "shit", "fuck", "crap",
        "bitch", "bastard", "idiot", "moron", "fool", "stupid", "crazy", "insane", "weird",
        "cute", "pretty", "beautiful", "handsome", "cool", "hot", "cold", "warm", "sweet",
        "dear", "honey", "darling", "baby", "sweetheart", "cutie", "friend", "friends",
        "brother", "sister", "mother", "father", "mom", "dad", "mama", "papa", "grandpa",
        "grandma", "uncle", "aunt", "cousin", "husband", "wife", "boyfriend", "girlfriend",
        "master", "boss", "leader", "captain", "chief", "king", "queen", "prince", "princess",
        "sir", "miss", "mister", "mrs", "doctor", "professor", "teacher", "student", "sensei",
        "senpai", "school", "class", "club", "team", "room", "house", "home", "store", "shop",
        "station", "street", "city", "town", "world", "life", "time", "year", "month", "week",
        "day", "hour", "minute", "second", "moment", "episode", "chapter", "part", "page",
        "volume", "title", "end", "continued", "line", "bubble", "text", "voice", "heart",
        "love", "like", "hate", "want", "need", "know", "think", "feel", "mean", "tell",
        "give", "take", "make", "keep", "leave", "stay", "back", "down", "away", "out",
        "off", "over", "under", "into", "onto", "from", "with", "about", "after", "before",
        "during", "until", "since", "while", "where", "could", "would", "should", "might",
        "must", "will", "shall", "have", "been", "being", "were", "does", "done", "doing",
        "going", "gonna", "wanna", "gotta", "whoa", "ouch", "oops", "huh", "heh", "meh",
        "ugh", "aww", "ooh", "eek", "gasp", "sigh", "gulp", "pant", "huff", "phew", "hmm",
        "mhm", "mmh", "nnn", "ahh", "uhh", "err", "tch", "tsk", "pfft", "haha", "hehe",
        "hoho", "hahaha", "lol", "lmao", "omg", "wtf", "phantom", "troupe", "right", "left",
        "first", "last", "next", "little", "long", "much", "many", "another", "excuse",
        "pardon", "welcome", "congratulations", "translated", "bottom", "tier", "third",
        "rate", "junior", "senior", "high", "middle", "dot",
    }
)


def extract_character_names(
    pairs: list[tuple[str, str]],
    existing: list[str] | None = None,
    max_names: int = 8,
) -> list[str]:
    """Extract compact proper character names from (orig, trans) dialogue pairs.

    Preserves first-seen order and deduplicates case-insensitively so earlier
    kana readings (e.g., ``まさき -> Masaki``) carry into later kanji chunks.
    """
    names: list[str] = []
    seen_lower: set[str] = set()

    def _add(name: str) -> None:
        cleaned = name.strip()
        if len(cleaned) < 3:
            return
        low = cleaned.lower()
        if low in seen_lower:
            return
        if low in _COMMON_ENGLISH_NON_NAMES or low in _ENGLISH_OR_LOANWORD_MORAIC:
            return
        if low in _STRONG_ROMAJI_WORDS or low in _WEAK_ROMAJI_TOKENS:
            return
        if _REPEATED_TRIPLE_RE.search(cleaned):
            return
        seen_lower.add(low)
        names.append(cleaned)

    if existing:
        for item in existing:
            if len(names) >= max_names:
                return names
            _add(item)

    for orig, trans in pairs:
        if len(names) >= max_names:
            break
        if not trans:
            continue
        text = _SFX_SPAN_RE.sub(" ", trans)

        # 1. Honorific-suffixed names (e.g., Kotone-chan, Masaki-kun)
        for m in _HONORIFIC_NAME_RE.finditer(text):
            if len(names) >= max_names:
                break
            _add(m.group(1))

        # 2. Mid-sentence capitalized proper nouns preceded by a lowercase word
        for m in _MID_SENTENCE_CAP_RE.finditer(text):
            if len(names) >= max_names:
                break
            _add(m.group(1))

        # 3. Standalone / sentence-start Japanese-moraic proper names (e.g., "Masaki...!!", "Ranshin...?")
        if contains_non_english_cjk(orig):
            for m in _SENTENCE_START_CAP_RE.finditer(text):
                if len(names) >= max_names:
                    break
                word = m.group(1)
                low = word.lower()
                if len(word) >= 4 and _JP_MORA_WORD_RE.match(low) is not None:
                    _add(word)

    return names[:max_names]


def _is_clean_pair(orig: str, trans: str) -> bool:
    return bool(
        orig
        and trans
        and not contains_non_english_cjk(trans)
        and not looks_like_romaji(orig, trans)
    )


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
        if _is_clean_pair(o, t):
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
                [(o, t) for o, t in buf if _is_clean_pair(o, t)]
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
                    prior_pairs.extend((o, t) for o, t in page_pairs if _is_clean_pair(o, t))
            elif not chapter_pairs:
                series_buf = self._series_contexts.get(sid)
                if series_buf:
                    prior_pairs.extend((o, t) for o, t in series_buf if _is_clean_pair(o, t))

            combined = prior_pairs + chapter_pairs
            return combined[-self.max_recent :]

    def get_formatted_context(
        self,
        chapter_id: str,
        series_id: str | None = None,
        max_pairs: int = 6,
    ) -> str:
        """Return page-grouped recent dialogue lines ([Page -2], [Page -1], ...) across pages/chapters."""
        if max_pairs <= 0:
            return ""
        with self._lock:
            chapter_pages: list[list[tuple[str, str]]] = []
            raw_ch_pages = self._pages.get(chapter_id)
            if raw_ch_pages:
                for page_pairs in raw_ch_pages:
                    cleaned = [(o, t) for o, t in page_pairs if _is_clean_pair(o, t)]
                    if cleaned:
                        chapter_pages.append(cleaned)
            elif chapter_id in self._contexts:
                buf_pairs = [(o, t) for o, t in self._contexts[chapter_id] if _is_clean_pair(o, t)]
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
                        cleaned = [(o, t) for o, t in page_pairs if _is_clean_pair(o, t)]
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

# Live Manga & Manhwa Dialogue Translator Implementation Plan

> **For agentic workers:** Execute this plan task-by-task using the user-selected `/swarm` workflow (contract-first parallel Hydrogen workers + proactive `neon-coder` for complex vision/overlay logic + `neon-reviewer` diff gate + independent `verifier`). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 100% free, local, uncensored Japanese manga and Korean manhwa live dialogue translator for Microsoft Edge on Windows (RTX 3070, 8 GB VRAM) that translates dialogue bubbles as the user scrolls while leaving floating sound effects untouched.

**Architecture:** A native Microsoft Edge Manifest V3 extension observes scrolling `<img>` and `<canvas>` reader pages (with 2–3 page look-ahead pre-translation and paginated `src`-swap support), fetches image bytes via its background service worker to bypass CORS/CSP, and posts them to a local Python FastAPI server (`127.0.0.1:8765`). The server runs a GPU-serialized 4-stage pipeline on the RTX 3070—strip slicing + `RT-DETR` comic bubble detection (dropping artwork SFX), `manga-ocr` (Japanese) / `EasyOCR` (Korean) with automatic dark-bubble inversion, single-call page-context translation via a bundled CUDA `llama-server` running an uncensored abliterated `Qwen2.5-7B-Instruct` `Q4_K_M` GGUF model with chapter glossary memory, and OpenCV bubble-interior style/mask geometry—and the extension renders crisp comic lettering overlays with controlled slight expansion for cramped narrow bubbles.

**Tech Stack:** Python 3.12 (FastAPI, Uvicorn, PyTorch CUDA, HuggingFace `transformers`, `manga-ocr`, `easyocr`, OpenCV, Pillow, `httpx`, `pytest`), prebuilt Windows CUDA `llama.cpp` (`llama-server.exe`) + `richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF` (`Q4_K_M`), Microsoft Edge Manifest V3 Extension (vanilla JS/CSS/HTML, bundled OFL comic font), Node.js test runner + Playwright for extension/overlay verification.

**Spec:** `D:\Code\manga-live-translator\docs\superpowers\specs\2026-11-19-manga-live-translator-design.md`

## Global Constraints

- **Zero `C:` drive bloat:** ALL project files, `.venv`, pip cache (`PIP_CACHE_DIR`), HuggingFace cache (`HF_HOME`), PyTorch cache (`TORCH_HOME`), EasyOCR cache (`EASYOCR_MODULE_PATH`), SQLite cache, `llama-server` binaries, and GGUF models MUST live strictly under `D:\Code\manga-live-translator\`.
- **8 GB VRAM Ceiling (RTX 3070):** Peak GPU memory across detector + OCR + `llama-server` MUST stay under `6.8 GB` (leave ≥ 1.2 GB for Windows + Edge); GPU jobs MUST be serialized inside a single worker queue so concurrent look-ahead requests never cause CUDA OOM.
- **Localhost Only & Zero Paid APIs:** Server binds strictly to `127.0.0.1:8765` and `llama-server` strictly to `127.0.0.1:8766`. Zero cloud translation APIs required.
- **Dialogue Only:** Floating background sound effects/onomatopoeia (`text_free` over artwork like `カサ…`) MUST be skipped untouched; speech bubbles and clean uniform-fill rectangular narration boxes MUST be translated.
- **Uncensored & Refusal-Free:** Must faithfully translate NSFW, horror/gore, and strong profanity without safety refusals ever appearing in a speech bubble.
- **Opt-In Per Site in Microsoft Edge:** Inactive by default on ordinary websites; enabled per domain via Edge toolbar popup; zero floating UI on disabled sites.
- **Bubble Typography & Controlled Overflow:** Default strictly inside bubble bounds; when narrow vertical bubbles cannot fit readable lettering (`>= 11px` relative), allow modest controlled expansion (up to `+25%` width / `+18%` height) with a matching rounded backdrop + subtle border, plus hover tooltip showing full English + original text.
- **Non-Coder Usability:** 1-click `Start Manga Translator.bat` and `Stop Manga Translator.bat` (releasing 100% of VRAM when stopped) plus a built-in `/test` visual verification harness pre-loaded with the user's Japanese centipede sample and a tall Korean webtoon strip sample.

## Review Focus

1. **Hotlink-protected / cross-origin manga CDN images (`403` without `Referer` or `CORS` canvas taint):** Background service worker image fetch must forward the tab's page URL as `Referer` (via `fetch` / `declarativeNetRequest` session rule fallback) and return clean base64/binary payloads even when `<img>` has no `crossOrigin` attribute. *(Pinned in Task 7 tests.)*
2. **Dark/black screaming speech bubbles (`ぎゃああああっ`) and narrow vertical bubbles:** OCR must auto-invert dark bubble crops (`luminance < 110`) before `manga-ocr` inference, geometry must emit white `#FFFFFF` ink on sampled dark background, and the overlay fitter must trigger controlled expansion (`<= +25%` width) without clipping text or bleeding wildly across panels. *(Pinned in Task 2, Task 3, and Task 8 tests.)*
3. **LLM JSON truncation, missing bubble IDs, or safety refusal leakage on NSFW/profanity pages:** Translator must detect refusal phrases (e.g., `"I cannot"`, `"I'm sorry"`, `"as an AI"`) or missing IDs, automatically retry with strict raw-translation framing, and fall back to per-bubble repair so every bubble gets clean, uncensored dialogue. *(Pinned in Task 4 tests.)*
4. **Tall Korean webtoon strips (`800×8000px+`) crossing slice boundaries:** Slicer must split tall strips into overlapping chunks (`1600px` height, `250px` overlap) and deduplicate bubbles split across seams without duplicating overlays or dropping seam bubbles. *(Pinned in Task 2 tests.)*
5. **Paginated single-page readers swapping `img.src` in place & stale in-flight look-ahead races:** When `img.src` mutates or the user scrolls rapidly past a queued image, stale overlays must clear immediately and late responses from an older `src`/hash must be ignored instead of painting old chapter dialogue onto the new page. *(Pinned in Task 8 tests.)*

---

## File Structure Map

```text
D:\Code\manga-live-translator\
├── config.json                        # User/runtime config (ports, model path, VRAM/ctx settings)
├── requirements.txt                   # Pinned Python 3.12 dependencies
├── setup.ps1                          # One-time D:-only installer (.venv, CUDA torch, llama-server, GGUF, samples)
├── Start Manga Translator.bat         # 1-click non-coder launcher (sets D: env vars, starts server, opens status UI)
├── Stop Manga Translator.bat          # 1-click graceful shutdown (frees 100% RTX 3070 VRAM)
├── assets\
│   ├── fonts\ComicNeue-Bold.woff2     # Bundled OFL comic lettering font (also copied to extension/fonts/)
│   └── samples\
│       ├── jp_centipede.jpg           # Copied user Japanese manga sample
│       └── ko_webtoon_strip.png       # Generated tall Korean webtoon strip (>3500px) for automated tests
├── server\
│   ├── __init__.py
│   ├── config.py                      # AppConfig, D:-drive env enforcement, path resolution
│   ├── schemas.py                     # Shared Pydantic models (TranslateRequest, BubbleResult, TranslateResponse, HealthResponse)
│   ├── cache.py                       # SQLite + LRU in-memory translation cache (D:\...\cache\translations.db)
│   ├── vision\
│   │   ├── __init__.py
│   │   ├── slicer.py                  # Tall webtoon strip slicer & IoU seam deduplicator
│   │   ├── detector.py                # RT-DETR bubble detector, SFX filter, narration box rescue, RTL/spread/KO sorter
│   │   └── geometry.py                # Bubble interior color sampling, dark-bubble detection, safe_box & mask geometry
│   ├── ocr\
│   │   ├── __init__.py
│   │   └── engine.py                  # manga-ocr (JP, auto dark-inversion) + EasyOCR (KO) + auto script detection
│   ├── translator\
│   │   ├── __init__.py
│   │   ├── llama_manager.py           # Bundled llama-server.exe process lifecycle (127.0.0.1:8766, CUDA)
│   │   ├── memory.py                  # Per-chapter rolling dialogue context + pinned character-name glossary
│   │   └── translator.py              # Single-call page JSON translation, uncensored prompt, refusal guard, fallback repair
│   ├── pipeline.py                    # Priority GPU worker queue (priority 0 visible > priority 1 lookahead) + 4-stage orchestrator
│   ├── app.py                         # FastAPI app on 127.0.0.1:8765 (/health, /translate, /glossary, /memory/clear, /shutdown, /, /test)
│   └── static\
│       ├── index.html                 # Friendly status dashboard + 3-step Edge install guide + Start/Stop GPU controls
│       └── test.html                  # Built-in visual test harness (jp_centipede.jpg + ko_webtoon_strip.png)
├── extension\
│   ├── manifest.json                  # Microsoft Edge Manifest V3 manifest
│   ├── background.js                  # Service worker: cross-origin image fetch (with Referer), API bridge, context menu
│   ├── overlay.js                     # Shared DOM overlay renderer + binary-search text fitter + controlled overflow + Alt peek
│   ├── content.js                     # Opt-in site controller, MutationObserver (src swap), IntersectionObserver look-ahead, floating pill
│   ├── styles.css                     # Crisp comic overlay, controlled-overflow backdrop, hover tooltip, collapsible pill styles
│   ├── popup.html                     # Edge toolbar popup UI (site toggle, status, lang, glossary editor, stop server)
│   ├── popup.js                       # Popup logic synced with chrome.storage.local & background service worker
│   ├── fonts\ComicNeue-Bold.woff2     # Local web-accessible OFL comic font
│   └── icons\                         # 16/48/128px PNG extension icons
└── tests\
    ├── test_config_and_cache.py
    ├── test_vision_slicer_geometry.py
    ├── test_detector_sorting.py
    ├── test_ocr_engine.py
    ├── test_translator_and_memory.py
    ├── test_pipeline_and_api.py
    ├── test_extension_overlay.mjs     # Node/Playwright behavioral tests for overlay.js, content.js, background.js
    └── test_e2e_gpu_samples.py        # Live RTX 3070 E2E verification on jp_centipede.jpg & ko_webtoon_strip.png
```

---

### Task 1: Project Configuration, D:-Drive Guard, Schemas & Persistent Cache

**Files:**
- Create: `D:\Code\manga-live-translator\config.json`
- Create: `D:\Code\manga-live-translator\server\__init__.py`
- Create: `D:\Code\manga-live-translator\server\config.py`
- Create: `D:\Code\manga-live-translator\server\schemas.py`
- Create: `D:\Code\manga-live-translator\server\cache.py`
- Create: `D:\Code\manga-live-translator\tests\test_config_and_cache.py`

**Interfaces:**
- Consumes: None (foundation module).
- Produces:
  - `server.config.AppConfig` dataclass + `load_config(path: Path | None = None) -> AppConfig` + `enforce_d_drive_env(cfg: AppConfig) -> dict[str, str]`
  - `server.schemas`: `Box(x1: int, y1: int, x2: int, y2: int)`, `BubbleResult(id: int, bbox: Box, safe_box: Box, bg_color: str, text_color: str, is_dark: bool, original: str, translated: str)`, `TranslateResponse(lang: Literal["ja", "ko"], width: int, height: int, elapsed_ms: int, cached: bool, bubbles: list[BubbleResult])`, `HealthResponse(status: str, ready: bool, gpu_name: str, vram_used_mb: int, vram_total_mb: int, llm_model: str, llm_ready: bool)`, `GlossaryUpdateRequest(chapter_id: str, entries: dict[str, str])`
  - `server.cache.TranslationCache(db_path: Path, max_entries: int = 2000)` with `make_key(image_bytes: bytes, lang: str, glossary: dict[str, str]) -> str`, `get(key: str) -> TranslateResponse | None`, `put(key: str, response: TranslateResponse) -> None`, `clear() -> None`

- [ ] **Step 1: Write failing tests in `tests/test_config_and_cache.py`**

```python
from pathlib import Path
import os
from server.config import load_config, enforce_d_drive_env
from server.schemas import Box, BubbleResult, TranslateResponse
from server.cache import TranslationCache

def test_enforce_d_drive_env_sets_all_caches_under_project_root(tmp_path):
    cfg = load_config(project_root=tmp_path)
    env = enforce_d_drive_env(cfg)
    for var in ("HF_HOME", "TORCH_HOME", "EASYOCR_MODULE_PATH", "PIP_CACHE_DIR"):
        assert Path(env[var]).is_relative_to(tmp_path / "cache")
        assert os.environ[var] == env[var]
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8765
    assert cfg.llama_port == 8766

def test_cache_roundtrip_and_glossary_invalidation(tmp_path):
    cache = TranslationCache(tmp_path / "translations.db", max_entries=5)
    resp = TranslateResponse(
        lang="ja", width=800, height=1200, elapsed_ms=950, cached=False,
        bubbles=[BubbleResult(id=0, bbox=Box(x1=10, y1=10, x2=100, y2=200),
                              safe_box=Box(x1=16, y1=18, x2=94, y2=192),
                              bg_color="#FFFFFF", text_color="#111111", is_dark=False,
                              original="藍星……？", translated="Ranshin...?")]
    )
    k1 = cache.make_key(b"fake-jpg", "ja", {"藍星": "Ranshin"})
    k2 = cache.make_key(b"fake-jpg", "ja", {"藍星": "Aiboshi"})
    assert k1 != k2
    cache.put(k1, resp)
    hit = cache.get(k1)
    assert hit is not None and hit.bubbles[0].translated == "Ranshin...?"
    assert cache.get(k2) is None
```

- [ ] **Step 2: Run test to verify failure**
  Run: `pytest tests/test_config_and_cache.py -v`
  Expected: `FAIL` (`ModuleNotFoundError: No module named 'server'`)

- [ ] **Step 3: Implement `config.json`, `server/config.py`, `server/schemas.py`, and `server/cache.py`**
  - Default `config.json` pins `host: "127.0.0.1"`, `port: 8765`, `llama_port: 8766`, `detector_model_id: "ogkalu/comic-text-and-bubble-detector"`, `llm_repo_id: "richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF"`, `llm_filename_pattern: "*Q4_K_M*.gguf"`, `llm_ctx_size: 4096`, `llm_gpu_layers: 99`, `max_cache_entries: 2000`.
  - `TranslationCache` stores serialized `TranslateResponse` JSON in SQLite with `last_accessed` LRU pruning so disk usage stays bounded.

- [ ] **Step 4: Run tests to verify pass**
  Run: `pytest tests/test_config_and_cache.py -v`
  Expected: `PASS` (all tests green).

- [ ] **Step 5: Commit foundation**
  Initialize git repo in `D:\Code\manga-live-translator` (with `.gitignore` excluding `.venv/`, `cache/`, `models/`, `bin/`, `__pycache__/`, `*.gguf`, `*.env`) and commit Task 1.

---

### Task 2: Vision Stage — Strip Slicer, Bubble Detector (SFX Filter & Reading Order), and Bubble Style Geometry

**Files:**
- Create: `D:\Code\manga-live-translator\server\vision\__init__.py`
- Create: `D:\Code\manga-live-translator\server\vision\slicer.py`
- Create: `D:\Code\manga-live-translator\server\vision\detector.py`
- Create: `D:\Code\manga-live-translator\server\vision\geometry.py`
- Create: `D:\Code\manga-live-translator\tests\test_vision_slicer_geometry.py`
- Create: `D:\Code\manga-live-translator\tests\test_detector_sorting.py`

**Interfaces:**
- Consumes: `server.schemas.Box`
- Produces:
  - `server.vision.slicer`: `ImageSlice(y_offset: int, image: PIL.Image.Image)`, `RawDetection(box: Box, label: str, score: float)`, `slice_tall_image(image: Image.Image, slice_height: int = 1600, overlap: int = 250) -> list[ImageSlice]`, `merge_slice_detections(detections: list[RawDetection], iou_threshold: float = 0.45) -> list[RawDetection]`
  - `server.vision.geometry`: `BubbleStyle(safe_box: Box, bg_color: str, text_color: str, is_dark: bool, mean_luminance: float, uniform_bg_ratio: float)`, `analyze_bubble_appearance(image_rgb: np.ndarray, bbox: Box) -> BubbleStyle`, `is_clean_narration_box(image_rgb: np.ndarray, bbox: Box) -> bool`
  - `server.vision.detector`: `BubbleDetector(model_id: str, device: str = "cuda")` with `detect(image: Image.Image) -> list[RawDetection]`, `filter_dialogue_regions(image_rgb: np.ndarray, detections: list[RawDetection]) -> list[RawDetection]`, `sort_reading_order(detections: list[RawDetection], img_w: int, img_h: int, lang: Literal["ja", "ko"]) -> list[RawDetection]`

- [ ] **Step 1: Write failing tests in `tests/test_vision_slicer_geometry.py` and `tests/test_detector_sorting.py`**

```python
# tests/test_vision_slicer_geometry.py (covers Review Focus #2 & #4)
import numpy as np
from PIL import Image
from server.schemas import Box
from server.vision.slicer import slice_tall_image, merge_slice_detections, RawDetection
from server.vision.geometry import analyze_bubble_appearance, is_clean_narration_box

def test_tall_korean_strip_slices_and_deduplicates_seam_bubble():
    img = Image.new("RGB", (800, 3800), (255, 255, 255))
    slices = slice_tall_image(img, slice_height=1600, overlap=250)
    assert len(slices) == 3
    assert slices[1].y_offset == 1350
    # Same bubble detected in slice 0 and slice 1 near y=1450
    d1 = RawDetection(Box(x1=100, y1=1400, x2=420, y2=1560), "bubble", 0.92)
    d2 = RawDetection(Box(x1=104, y1=1405, x2=418, y2=1558), "bubble", 0.88)
    merged = merge_slice_detections([d1, d2], iou_threshold=0.45)
    assert len(merged) == 1 and merged[0].score == 0.92

def test_dark_bubble_detected_with_white_text_and_light_bubble_dark_text():
    canvas = np.full((300, 300, 3), 250, dtype=np.uint8)
    # Dark screaming bubble with white glyphs inside
    canvas[20:140, 20:140] = 15
    canvas[50:110, 65:95] = 245
    dark_style = analyze_bubble_appearance(canvas, Box(x1=20, y1=20, x2=140, y2=140))
    assert dark_style.is_dark is True
    assert dark_style.text_color == "#FFFFFF"

    light_style = analyze_bubble_appearance(canvas, Box(x1=160, y1=160, x2=280, y2=280))
    assert light_style.is_dark is False
    assert light_style.text_color == "#111111"

def test_clean_narration_box_rescued_but_artwork_sfx_rejected():
    # Gradient artwork background with stylized SFX text -> low uniform background ratio
    art = np.tile(np.linspace(30, 210, 200, dtype=np.uint8), (200, 1))
    art_rgb = np.stack([art, art // 2, 255 - art], axis=-1)
    assert is_clean_narration_box(art_rgb, Box(x1=20, y1=20, x2=180, y2=180)) is False
    # Solid white rectangular narration box with dark border
    box_rgb = np.full((200, 200, 3), 252, dtype=np.uint8)
    box_rgb[20:180, 20:23] = 10; box_rgb[20:180, 177:180] = 10
    box_rgb[20:23, 20:180] = 10; box_rgb[177:180, 20:180] = 10
    assert is_clean_narration_box(box_rgb, Box(x1=20, y1=20, x2=180, y2=180)) is True
```

```python
# tests/test_detector_sorting.py
from server.schemas import Box
from server.vision.slicer import RawDetection
from server.vision.detector import filter_dialogue_regions, sort_reading_order
import numpy as np

def test_filter_drops_text_free_over_artwork_and_keeps_bubbles():
    art = np.random.default_rng(42).integers(0, 255, size=(400, 400, 3), dtype=np.uint8)
    dets = [
        RawDetection(Box(x1=10, y1=10, x2=120, y2=150), "bubble", 0.9),
        RawDetection(Box(x1=200, y1=50, x2=300, y2=180), "text_free", 0.85),
    ]
    kept = filter_dialogue_regions(art, dets)
    assert len(kept) == 1 and kept[0].label == "bubble"

def test_japanese_rtl_and_two_page_spread_sorting():
    # Wide two-page spread (2000x1400): right page bubbles MUST all precede left page bubbles
    right_top = RawDetection(Box(x1=1600, y1=100, x2=1850, y2=300), "bubble", 0.9)
    right_bottom = RawDetection(Box(x1=1100, y1=900, x2=1350, y2=1100), "bubble", 0.9)
    left_top = RawDetection(Box(x1=600, y1=80, x2=850, y2=280), "bubble", 0.9)
    ordered = sort_reading_order([left_top, right_bottom, right_top], img_w=2000, img_h=1400, lang="ja")
    assert [d.box.x1 for d in ordered] == [1600, 1100, 600]
```

- [ ] **Step 2: Run tests to verify failure**
  Run: `pytest tests/test_vision_slicer_geometry.py tests/test_detector_sorting.py -v`
  Expected: `FAIL`

- [ ] **Step 3: Implement `slicer.py`, `geometry.py`, and `detector.py`**
  - `slicer.py`: For `image.height > 2000`, slice into overlapping vertical chunks (`slice_height=1600, overlap=250`), shift boxes back by `y_offset`, and run class-aware NMS / containment deduplication so overlapping boxes across seams merge cleanly.
  - `geometry.py`: Sample perimeter/border-inset ring pixels (avoiding the outermost black ink stroke and inner glyphs) via median/mode cluster to determine true bubble interior color `bg_color`, compute `is_dark = mean_luminance < 110`, set `text_color = "#FFFFFF" if is_dark else "#111111"`, compute `safe_box` (inset ~10–14% from bubble edges to stay inside rounded oval strokes), and measure `uniform_bg_ratio` (fraction of pixels near dominant color) to separate clean rectangular narration boxes from artwork-overlapping SFX.
  - `detector.py`: Load `ogkalu/comic-text-and-bubble-detector` via `RTDetrImageProcessor` + `RTDetrForObjectDetection` on `cuda` (`fp16` where supported for speed & ~350 MB VRAM). Keep `bubble` and `text_bubble` (preferring `text_bubble` paired with enclosing `bubble` for tight OCR crop + clean bubble bounds); for `text_free`, drop unless `is_clean_narration_box` passes strictly. Sort Japanese manga right-to-left / top-to-bottom (splitting wide spreads `img_w / img_h > 1.25` right-half first), and Korean manhwa top-to-bottom / left-to-right.

- [ ] **Step 4: Run unit tests to verify pass**
  Run: `pytest tests/test_vision_slicer_geometry.py tests/test_detector_sorting.py -v`
  Expected: `PASS`

- [ ] **Step 5: Commit Task 2**

---

### Task 3: OCR Stage — Japanese (`manga-ocr`), Korean (`EasyOCR`), Dark-Bubble Inversion & Script Auto-Detection

**Files:**
- Create: `D:\Code\manga-live-translator\server\ocr\__init__.py`
- Create: `D:\Code\manga-live-translator\server\ocr\engine.py`
- Create: `D:\Code\manga-live-translator\tests\test_ocr_engine.py`

**Interfaces:**
- Consumes: `server.vision.geometry.BubbleStyle`
- Produces:
  - `server.ocr.engine`:
    - `preprocess_crop_for_ocr(crop: Image.Image, is_dark: bool) -> Image.Image`
    - `is_meaningful_dialogue(text: str) -> bool`
    - `classify_script(texts: list[str], aspect_ratio: float) -> Literal["ja", "ko"]`
    - `OcrEngine(device: str = "cuda", cache_dir: Path = ...)` with `read_bubble(crop: Image.Image, lang: Literal["ja", "ko"], is_dark: bool) -> str` and `read_page_bubbles(crops: list[tuple[Image.Image, bool]], lang_hint: Literal["auto", "ja", "ko"], aspect_ratio: float) -> tuple[Literal["ja", "ko"], list[str]]`

- [ ] **Step 1: Write failing tests in `tests/test_ocr_engine.py`**

```python
import numpy as np
from PIL import Image
from server.ocr.engine import preprocess_crop_for_ocr, is_meaningful_dialogue, classify_script

def test_dark_bubble_crop_is_inverted_for_ocr():
    dark = Image.fromarray(np.full((64, 64, 3), 10, dtype=np.uint8))
    inverted = preprocess_crop_for_ocr(dark, is_dark=True)
    assert np.array(inverted).mean() > 235
    normal = preprocess_crop_for_ocr(dark, is_dark=False)
    assert np.array(normal).mean() < 20

def test_meaningful_dialogue_filters_punctuation_and_keeps_short_screams():
    assert is_meaningful_dialogue("ぎ") is True
    assert is_meaningful_dialogue("蟲") is True
    assert is_meaningful_dialogue("뭐?!") is True
    assert is_meaningful_dialogue("...") is False
    assert is_meaningful_dialogue("！？") is False
    assert is_meaningful_dialogue("   ") is False

def test_classify_script_distinguishes_japanese_and_korean():
    assert classify_script(["今！助けます！", "蜈蚣が！"], aspect_ratio=0.7) == "ja"
    assert classify_script(["지금 바로 구해줄게!", "지네야!"], aspect_ratio=0.15) == "ko"
```

- [ ] **Step 2: Run test to verify failure**
  Run: `pytest tests/test_ocr_engine.py -v`
  Expected: `FAIL`

- [ ] **Step 3: Implement `server/ocr/engine.py`**
  - Lazy-load `MangaOcr` (PyTorch CUDA, strips furigana automatically) and `easyocr.Reader(['ko'], gpu=True, model_storage_directory=...)` inside `D:\Code\manga-live-translator\cache\easyocr`.
  - When `is_dark=True`, invert crop (`ImageOps.invert`) + light contrast stretch so white-on-black screams (`ぎゃああああっ`, `ぎ`) read cleanly.
  - When `lang_hint == "auto"`: sample up to 2 largest bubbles with fast script probe (aspect ratio > 2.2 tall strip tries Korean first; standard manga page tries Japanese first; verifies Hangul vs. Kana/Kanji character counts and falls back to the other OCR if zero target-script glyphs appear).

- [ ] **Step 4: Run unit tests to verify pass**
  Run: `pytest tests/test_ocr_engine.py -v`
  Expected: `PASS`

- [ ] **Step 5: Commit Task 3**

---

### Task 4: Uncensored Local LLM Translator, `llama-server` Lifecycle & Chapter Memory

**Files:**
- Create: `D:\Code\manga-live-translator\server\translator\__init__.py`
- Create: `D:\Code\manga-live-translator\server\translator\llama_manager.py`
- Create: `D:\Code\manga-live-translator\server\translator\memory.py`
- Create: `D:\Code\manga-live-translator\server\translator\translator.py`
- Create: `D:\Code\manga-live-translator\tests\test_translator_and_memory.py`

**Interfaces:**
- Consumes: `server.config.AppConfig`
- Produces:
  - `server.translator.llama_manager.LlamaServerManager(cfg: AppConfig)` with `start() -> None`, `stop() -> None`, `is_healthy() -> bool`, `base_url: str`
  - `server.translator.memory.ChapterMemoryStore(max_recent: int = 10)` with `get_context(chapter_id: str) -> list[tuple[str, str]]`, `get_glossary(chapter_id: str) -> dict[str, str]`, `set_glossary(chapter_id: str, entries: dict[str, str]) -> None`, `record_page(chapter_id: str, pairs: list[tuple[str, str]]) -> None`, `clear(chapter_id: str) -> None`
  - `server.translator.translator`: `is_refusal(text: str) -> bool`, `DialogueTranslator(llama_base_url: str, memory: ChapterMemoryStore)` with `async translate_page(bubbles: list[tuple[int, str]], lang: Literal["ja", "ko"], chapter_id: str, extra_glossary: dict[str, str] | None = None) -> dict[int, str]`

- [ ] **Step 1: Write failing tests in `tests/test_translator_and_memory.py` (covers Review Focus #3)**

```python
import pytest
from server.translator.memory import ChapterMemoryStore
from server.translator.translator import DialogueTranslator, is_refusal

def test_refusal_detector_catches_safety_boilerplate():
    assert is_refusal("I'm sorry, I cannot translate explicit or violent content.") is True
    assert is_refusal("As an AI language model, I must decline.") is True
    assert is_refusal("Get the fuck off me! A centipede!!") is False
    assert is_refusal("GYAAAAAH!") is False

@pytest.mark.asyncio
async def test_translator_parses_page_json_and_repairs_missing_or_refused_bubble():
    memory = ChapterMemoryStore()
    memory.set_glossary("ch1", {"藍星": "Ranshin"})
    calls = []

    async def fake_llm(messages):
        calls.append(messages)
        if len(calls) == 1:
            # Simulates missing id=2 AND a refusal on id=1
            return '{"0": "Ranshin...?", "1": "I cannot translate profanity."}'
        return '{"1": "Shit! A centipede!", "2": "GYAAAAAH!"}'

    tr = DialogueTranslator("http://127.0.0.1:8766", memory, llm_caller=fake_llm)
    out = await tr.translate_page(
        [(0, "藍星……？"), (1, "くそっ！蜈蚣が！"), (2, "ぎゃああああっ")],
        lang="ja", chapter_id="ch1"
    )
    assert out[0] == "Ranshin...?"
    assert not is_refusal(out[1]) and "centipede" in out[1].lower()
    assert out[2] == "GYAAAAAH!"
    assert len(calls) == 2
    assert len(memory.get_context("ch1")) == 3
```

- [ ] **Step 2: Run test to verify failure**
  Run: `pytest tests/test_translator_and_memory.py -v`
  Expected: `FAIL`

- [ ] **Step 3: Implement `llama_manager.py`, `memory.py`, and `translator.py`**
  - `llama_manager.py`: Launches `D:\Code\manga-live-translator\bin\llama-cpp\llama-server.exe --model <gguf> --host 127.0.0.1 --port 8766 -ngl 99 -c 4096 --flash-attn on --parallel 1`, polls `/health` until ready, and terminates cleanly on `stop()` (releasing 100% of VRAM).
  - `translator.py`: Sends a single `/v1/chat/completions` call (`temperature=0.25`, `response_format={"type": "json_object"}`) with an explicit uncensored manga-localization system prompt instructing: concise punchy comic dialogue (so text fits narrow bubbles cleanly), faithful profanity/horror/NSFW translation with zero censorship or commentary, strict adherence to pinned glossary entries, and strict JSON mapping `{"0": "...", "1": "..."}`. If any bubble ID is missing, empty, or triggers `is_refusal()`, retries ONLY the failed/missing IDs once with neutral literal-translation framing.

- [ ] **Step 4: Run tests to verify pass**
  Run: `pytest tests/test_translator_and_memory.py -v`
  Expected: `PASS`

- [ ] **Step 5: Commit Task 4**

---

### Task 5: GPU Priority Queue Pipeline, FastAPI Server, Status Dashboard & `/test` Visual Harness

**Files:**
- Create: `D:\Code\manga-live-translator\server\pipeline.py`
- Create: `D:\Code\manga-live-translator\server\app.py`
- Create: `D:\Code\manga-live-translator\server\static\index.html`
- Create: `D:\Code\manga-live-translator\server\static\test.html`
- Create: `D:\Code\manga-live-translator\tests\test_pipeline_and_api.py`

**Interfaces:**
- Consumes: `AppConfig`, `TranslationCache`, `BubbleDetector`, `OcrEngine`, `DialogueTranslator`, `LlamaServerManager`, `ChapterMemoryStore`, `analyze_bubble_appearance`
- Produces:
  - `server.pipeline.TranslationPipeline` with `async submit(image_bytes: bytes, lang: str, chapter_id: str, priority: int, glossary: dict[str, str], request_id: str | None) -> TranslateResponse`, `cancel(request_id: str) -> None`
  - FastAPI `create_app(pipeline, cache, memory, llama_manager, cfg)` serving:
    - `GET /health` → `HealthResponse`
    - `POST /translate` (JSON `{image_base64, lang, chapter_id, priority, glossary, request_id}` or multipart form) → `TranslateResponse`
    - `GET /glossary?chapter_id=...`, `POST /glossary`, `POST /memory/clear`
    - `POST /shutdown` (gracefully stops `llama_manager` + terminates server process cleanly)
    - `GET /` (status dashboard + 3-step Edge install guide + Stop GPU Engine button)
    - `GET /test` (visual harness rendering `jp_centipede.jpg` and `ko_webtoon_strip.png` via shared `overlay.js` + `styles.css`!)

- [ ] **Step 1: Write failing tests in `tests/test_pipeline_and_api.py`**

```python
import asyncio, base64
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
from server.app import create_app
from server.pipeline import PriorityGpuQueue

@pytest.mark.asyncio
async def test_priority_queue_runs_visible_page_before_lookahead_and_serializes_gpu():
    active = 0
    max_concurrent = 0
    order = []

    async def worker(tag):
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.03)
        order.append(tag)
        active -= 1
        return tag

    q = PriorityGpuQueue()
    await q.start()
    t_running = asyncio.create_task(q.enqueue(priority=1, fn=lambda: worker("first")))
    await asyncio.sleep(0.005)
    t_low = asyncio.create_task(q.enqueue(priority=1, fn=lambda: worker("lookahead")))
    t_high = asyncio.create_task(q.enqueue(priority=0, fn=lambda: worker("visible")))
    await asyncio.gather(t_running, t_low, t_high)
    await q.stop()
    assert max_concurrent == 1
    assert order == ["first", "visible", "lookahead"]
```
*(Plus `TestClient` API tests verifying `/health`, `/translate` cache hit on second call, `/glossary` persistence, `/memory/clear`, and `GET /` + `GET /test` returning HTTP 200.)*

- [ ] **Step 2: Run test to verify failure**
  Run: `pytest tests/test_pipeline_and_api.py -v`
  Expected: `FAIL`

- [ ] **Step 3: Implement `pipeline.py`, `app.py`, `static/index.html`, and `static/test.html`**
  - `PriorityGpuQueue`: `asyncio.PriorityQueue` driven by a single consumer coroutine guaranteeing `max_concurrent == 1` on the RTX 3070, with `(priority, seq_num)` ordering and cooperative cancellation when a tab scrolls away or aborts.
  - `app.py`: Add permissive local CORS headers for `chrome-extension://*` and `http://127.0.0.1:8765`, mount `/assets` and `/ext` (so `/test` loads the exact same `extension/overlay.js` and `extension/styles.css` with zero code duplication!).
  - `static/index.html`: Plain-English non-coder dashboard showing live GPU card name, VRAM used, Ready/Loading badge, big **"Open Visual Test Page"** button, **"Stop Translator & Free GPU"** button, and clear 3-step illustrated Edge installation instructions (`edge://extensions` → Developer Mode → Load unpacked → select `D:\Code\manga-live-translator\extension`).

- [ ] **Step 4: Run tests to verify pass**
  Run: `pytest tests/test_pipeline_and_api.py -v`
  Expected: `PASS`

- [ ] **Step 5: Commit Task 5**

---

### Task 6: Shared Comic Overlay Renderer (`extension/overlay.js` + `styles.css` + Bundled OFL Comic Font)

**Files:**
- Create: `D:\Code\manga-live-translator\extension\overlay.js`
- Create: `D:\Code\manga-live-translator\extension\styles.css`
- Create: `D:\Code\manga-live-translator\assets\fonts\ComicNeue-Bold.woff2` (and copy to `extension\fonts\ComicNeue-Bold.woff2`)
- Create: `D:\Code\manga-live-translator\tests\test_extension_overlay.mjs` (overlay behavioral suite)

**Interfaces:**
- Consumes: `TranslateResponse` JSON from Task 5 (`/translate`)
- Produces:
  - Global/export `window.MangaOverlay` with:
    - `renderOverlay(targetEl: HTMLImageElement | HTMLCanvasElement, data: TranslateResponse) -> OverlayHandle`
    - `clearOverlay(targetEl: HTMLElement) -> void`
    - `setPeekOriginal(enabled: boolean) -> void`
    - `computeFittedBox(text: string, safeBox: {x1,y1,x2,y2}, imgSize: {w,h}, measureFn) -> {box, fontSizePx, expanded: boolean}`

- [ ] **Step 1: Write failing behavioral tests in `tests/test_extension_overlay.mjs` (covers Review Focus #2 & user's controlled-overflow rule)**
  - Test 1 (`narrow_bubble_controlled_expansion`): Given a tall skinny vertical Japanese bubble (`55×260px` on a `900×1300px` image) and a 6-word English line, `computeFittedBox` keeps `fontSizePx >= 11`, sets `expanded: true`, and expands box width by **at most +25%** (never sprawling across the page) with matching `bg_color` backdrop class.
  - Test 2 (`dark_bubble_high_contrast`): Dark bubble (`is_dark: true`, `bg_color: "#111111"`, `text_color: "#FFFFFF"`) renders white text on dark rounded patch.
  - Test 3 (`alt_peek_hides_and_restores`): `setPeekOriginal(true)` toggles `.mlt-peek-original` hiding overlays cleanly without losing DOM nodes; `setPeekOriginal(false)` restores immediately.
  - Test 4 (`hover_tooltip_contains_english_and_original`): Hovering a rendered bubble exposes full translated text + original OCR text in its tooltip element.

- [ ] **Step 2: Run test to verify failure**
  Run: `node --test tests/test_extension_overlay.mjs`
  Expected: `FAIL`

- [ ] **Step 3: Implement `extension/overlay.js` and `extension/styles.css`**
  - Overlay container positions accurately over `targetEl` via a lightweight wrapper/positioned sibling synced with `ResizeObserver`, converting normalized natural-image coordinates (`bbox` / `safe_box`) to rendered CSS percentages so browser zoom and responsive layout resizing stay pixel-locked at 60fps.
  - Renders an inner clean mask patch (`bg_color`) over `safe_box` (slightly rounded to preserve black oval ink borders) + crisp comic lettering using bundled `ComicNeue-Bold.woff2`.
  - Binary-search font fitting (`max 26px` down to `min 11px` relative to page width): if text overflows `safe_box` at `11px`, smoothly expands the text container up to `+25%` width and `+18%` height (clamped to page edges) with a matching bubble-colored rounded pill backdrop and subtle `1px` contrast border.
  - `Alt` keydown/keyup global listener toggles peek-original instantly.

- [ ] **Step 4: Run tests to verify pass**
  Run: `node --test tests/test_extension_overlay.mjs`
  Expected: `PASS`

- [ ] **Step 5: Commit Task 6**

---

### Task 7: Microsoft Edge Manifest V3 Extension (`manifest.json`, `background.js`, `content.js`, `popup.html/js`)

**Files:**
- Create: `D:\Code\manga-live-translator\extension\manifest.json`
- Create: `D:\Code\manga-live-translator\extension\background.js`
- Create: `D:\Code\manga-live-translator\extension\content.js`
- Create: `D:\Code\manga-live-translator\extension\popup.html`
- Create: `D:\Code\manga-live-translator\extension\popup.js`
- Create: `D:\Code\manga-live-translator\extension\icons\icon16.png`, `icon48.png`, `icon128.png`
- Modify: `D:\Code\manga-live-translator\tests\test_extension_overlay.mjs` (add Playwright / headless DOM tests for opt-in activation, `src`-swap race protection, and look-ahead priority)

**Interfaces:**
- Consumes: `window.MangaOverlay` (Task 6), local API `http://127.0.0.1:8765` (Task 5)
- Produces: Complete loadable Microsoft Edge Manifest V3 extension in `D:\Code\manga-live-translator\extension\`.

- [ ] **Step 1: Add failing behavioral tests to `tests/test_extension_overlay.mjs` (covers Review Focus #1 & #5)**
  - Test 5 (`opt_in_dormant_on_disabled_site`): On a domain NOT in `enabledDomains`, `content.js` mounts zero floating pills and issues zero translate calls.
  - Test 6 (`paginated_src_swap_drops_stale_response`): On an enabled domain, when `img.src` mutates from `page1.jpg` to `page2.jpg` while `page1.jpg`'s translate promise is still in flight, the stale `page1.jpg` overlay clears immediately and when `page1.jpg`'s late response resolves it is discarded (never painted over `page2.jpg`).
  - Test 7 (`lookahead_priority_ordering`): Visible viewport image sends `priority: 0`, next 2–3 below-viewport images send `priority: 1`, and images > 4 pages away are skipped until scrolled closer.
  - Test 8 (`background_fetch_forwards_referer`): Background fetch helper configures temporary `declarativeNetRequest` session rule / `Referer` header when fetching cross-origin manga CDN URLs.

- [ ] **Step 2: Run test to verify failure**
  Run: `node --test tests/test_extension_overlay.mjs`
  Expected: `FAIL`

- [ ] **Step 3: Implement `manifest.json`, `background.js`, `content.js`, `popup.html`, `popup.js`, and icons**
  - `manifest.json`: MV3 with `permissions: ["storage", "activeTab", "contextMenus", "declarativeNetRequest"]`, `host_permissions: ["<all_urls>"]`, `web_accessible_resources` exposing `fonts/ComicNeue-Bold.woff2`.
  - `background.js`: Fetches image URLs (with `Referer` forwarded via `declarativeNetRequest` session rule fallback for hotlink-protected CDNs), converts to base64, posts to `http://127.0.0.1:8765/translate`, handles AbortController cancellation, exposes context menu **"Translate this manga page"**, and proxies `/health`, `/glossary`, `/memory/clear`, `/shutdown`.
  - `content.js`: Opt-in per hostname via `chrome.storage.local`; observes DOM additions AND `src`/`srcset` attribute changes (`MutationObserver`), tracks generation token per `<img>` / `<canvas>` to reject stale late responses, queues visible images (`priority: 0`) + next 2–3 upcoming images (`priority: 1`) via `IntersectionObserver`, and mounts a sleek collapsible bottom-right pill ONLY when enabled on the current site.
  - `popup.html` & `popup.js`: Clean plain-English Edge popup: GPU status indicator, big **"Translate This Site"** switch, Language (`Auto` / `Japanese` / `Korean`), simple Character Name Glossary editor (`藍星 = Ranshin`), **"Clear Chapter Memory"**, **"Open Test Page"**, and **"Stop Local GPU Engine"**.

- [ ] **Step 4: Run tests to verify pass**
  Run: `node --test tests/test_extension_overlay.mjs`
  Expected: `PASS`

- [ ] **Step 5: Commit Task 7**

---

### Task 8: One-Time `D:`-Only Setup Script, 1-Click Launchers, Sample Fixtures & Live RTX 3070 E2E Verification

**Files:**
- Create: `D:\Code\manga-live-translator\requirements.txt`
- Create: `D:\Code\manga-live-translator\setup.ps1`
- Create: `D:\Code\manga-live-translator\Start Manga Translator.bat`
- Create: `D:\Code\manga-live-translator\Stop Manga Translator.bat`
- Create: `D:\Code\manga-live-translator\assets\samples\jp_centipede.jpg` (copied from `C:\Users\ecmat\.gemini\antigravity\brain\a14219b1-1d67-4a8f-970a-a1ad8b0dcea1\.user_uploaded\media_1795062710266_cdab5fd3.jpg`)
- Create: `D:\Code\manga-live-translator\assets\samples\ko_webtoon_strip.png` (generated tall Korean Hangul webtoon strip `800×3800px` with dialogue bubbles across slice boundaries)
- Create: `D:\Code\manga-live-translator\tests\test_e2e_gpu_samples.py`

**Interfaces:**
- Consumes: Entire stack from Tasks 1–7.
- Produces: Ready-to-run Windows installation on `D:\Code\manga-live-translator` verified live on the RTX 3070.

- [ ] **Step 1: Write `setup.ps1`, `requirements.txt`, `Start Manga Translator.bat`, `Stop Manga Translator.bat`, and sample fixture builder**
  - `setup.ps1`:
    1. Enforces `TEMP`, `TMP`, `PIP_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`, `EASYOCR_MODULE_PATH` all inside `D:\Code\manga-live-translator\cache` (verifying `C:` free space does not shrink!).
    2. Creates `D:\Code\manga-live-translator\.venv` using Python 3.12 and installs PyTorch CUDA (`cu124`/`cu126`) + `requirements.txt`.
    3. Downloads latest prebuilt Windows CUDA `llama.cpp` release (`llama-server.exe` + CUDA DLLs) into `D:\Code\manga-live-translator\bin\llama-cpp\`.
    4. Downloads `richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF` (`*Q4_K_M*.gguf`) into `D:\Code\manga-live-translator\models\`, plus pre-warms `ogkalu/comic-text-and-bubble-detector`, `kha-white/manga-ocr`, and `EasyOCR` Korean into `D:\Code\manga-live-translator\cache\`.
    5. Copies `jp_centipede.jpg` and generates `ko_webtoon_strip.png`, and creates a Desktop shortcut to `Start Manga Translator.bat`.

- [ ] **Step 2: Write live RTX 3070 E2E test in `tests/test_e2e_gpu_samples.py`**
  - Starts the real server on `127.0.0.1:8765` (with real RTX 3070 GPU models + `llama-server.exe`).
  - Verifies `GET /health` reports `ready: true`, `RTX 3070`, and `vram_used_mb < 7000`.
  - Posts `assets/samples/jp_centipede.jpg` (`lang: "auto"`, `glossary: {"藍星": "Ranshin"}`):
    - Asserts `lang == "ja"`.
    - Asserts `elapsed_ms < 4500` (warm pass `< 3000`).
    - Asserts detected dialogue bubbles include `"Ranshin"` (or top-right bubble), `"centipede"` (from `蜈蚣`), and a screaming dark bubble (`is_dark == True`, `text_color == "#FFFFFF"`).
    - Asserts floating SFX (`カサ…` region in top-left gradient panel) is **NOT** covered by any returned bubble box!
    - Asserts zero refusal strings in any bubble.
  - Posts `assets/samples/ko_webtoon_strip.png` (`lang: "auto"`):
    - Asserts `lang == "ko"`, all Hangul bubbles across slice seams detected without duplicates and translated into natural English.
  - Opens `http://127.0.0.1:8765/test` in headless Chromium/Edge via Playwright to verify overlays render cleanly with zero console errors and zero clipped boxes, then saves a visual verification screenshot.
  - Calls `POST /shutdown` and verifies `llama-server.exe` exits cleanly and VRAM drops back to baseline.

- [ ] **Step 3: Run `setup.ps1` and execute full unit + Node + live RTX 3070 E2E verification suite**
  - Run: `D:\Code\manga-live-translator\.venv\Scripts\pytest.exe tests/ -v`
  - Run: `node --test tests/test_extension_overlay.mjs`
  - Confirm `C:` drive free space remained untouched (~8.5 GB) and all tests pass.

- [ ] **Step 4: Commit final verified project**

---

## Self-Review Checklist (Completed)

1. **Spec coverage:** Every approved requirement in `2026-11-19-manga-live-translator-design.md` maps to a concrete task: `D:`-only storage (Tasks 1 & 8), RTX 3070 8 GB VRAM serialization & budget (Tasks 2, 4, 5, 8), dialogue-only / SFX skip + narration box rescue (Task 2), Japanese vertical/furigana + Korean Hangul + dark-bubble inversion (Task 3), uncensored/abliterated local GGUF + refusal guard + chapter glossary (Task 4), Microsoft Edge MV3 opt-in extension + flexible capture (`<img>`, `src` swap, `<canvas>`, wide spread, look-ahead) (Tasks 2, 6, 7), controlled slight bubble overflow + hover tooltip + `Alt` peek (Task 6), 1-click Start/Stop launchers & `/test` visual harness on user's centipede sample + Korean strip (Task 8).
2. **Step scan:** Every task has exact file paths, explicit `Consumes`/`Produces` interfaces, concrete assertions, and clear verification commands.
3. **Type consistency:** `Box`, `BubbleResult`, `TranslateResponse`, `HealthResponse`, `RawDetection`, `BubbleStyle`, `ChapterMemoryStore`, and `window.MangaOverlay` match identically across Tasks 1–8.
4. **Review Focus:** All 5 high-risk failure modes (hotlink Referer/CORS, dark & narrow vertical bubbles, LLM refusals/missing IDs, tall Korean strip seam deduplication, paginated `src`-swap race conditions) are explicitly pinned by automated tests in their owning tasks.
5. **Proportion:** Concise, interface-driven plan (~310 lines) focused purely on decisions, contracts, and verification rather than pasting implementation bodies.

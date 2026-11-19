# Live Manga & Manhwa Dialogue Translator — Design Document

**Date:** 2026-11-19  
**Project Directory:** `D:\Code\manga-live-translator` *(chosen instead of `C:` because `C:` has only 8.5 GB free while `D:` has 334.2 GB free)*  
**Status:** Approved by User (2026-11-19)

---

## 1. Intent & The Why

The user reads untranslated Japanese manga and Korean manhwa on manga websites on their Windows PC. Because manga pages are flat images (not selectable text), normal browser translation tools cannot read or translate them. The user wants a **free, completely local** system—with **no paid API keys or cloud accounts**—that automatically detects dialogue in manga/manhwa images and overlays clean, readable English text **live as they read and scroll**.

The user's core priorities are:
1. **Speed:** Pages must translate fast enough to feel smooth while scrolling (assisted by look-ahead pre-translation).
2. **Accuracy:** Natural, context-aware English dialogue—not stiff, word-for-word machine translation—that handles vertical Japanese (with furigana), Korean Hangul, split sentences across bubbles, and consistent character names.
3. **Readability:** Crisp, properly sized comic lettering matched to bubble colors (dark ink on white/light bubbles; crisp white ink on black/dark screaming bubbles) that stays sharp and never overlaps or obscures surrounding artwork.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- Translate **Japanese manga** (right-to-left pages, vertical text, furigana, white and black speech bubbles, narration boxes) and **Korean manhwa** (long vertical webtoon strips, horizontal/vertical Hangul bubbles) live while scrolling on manga websites.
- Default cleanly to **regular vertical-scrolling manga/manhwa websites** while keeping capture flexible enough to handle **paginated single-page / image-swap readers, wide two-page spreads, and unscrambled `<canvas>` reader pages**.
- Provide **uncensored, faithful, refusal-free translation** across all genres including dark horror/gore, strong profanity, and mature/NSFW manga & manhwa.
- Run **100% locally and free** on the user's Windows PC (NVIDIA RTX 3070, 8 GB VRAM) with zero paid API keys or online translation dependencies after initial model download.
- Keep scrolling responsive via viewport-prioritized queueing, **look-ahead pre-translation** of the next 2–3 upcoming pages, and hash-based caching so scrolling back up is instant.
- Provide simple, non-coder-friendly controls directly on the page: on/off toggle (global and per-site), quick peek at the original image (hotkey or button), language selector (`Auto` / `Japanese` / `Korean`), and one-click re-translate for a page.
- Store **all code, virtual environments, model weights, and caches exclusively on `D:\Code\manga-live-translator`** so `C:` drive space is never consumed.

### Non-Goals
- **No onomatopoeia / floating sound effect (SFX) translation:** Floating SFX drawn over artwork (e.g., *"カサ…"* crawling sound over a shaded panel) must be intentionally skipped and left untouched.
- **No heavy artwork inpainting/redrawing:** Because SFX over artwork is skipped, we do not need slow generative inpainting that risks smudging illustrations. Clean bubble-interior erasure is sufficient and far faster.
- **No mobile app, cloud backend, or paid API requirement** in v1 (an optional cloud fallback can be considered later, never required).
- **No batch-exporting translated CBZ/PDF archives** in v1—focus is strictly live web reading.
- **No DRM-bypass or tile-descrambling** for proprietary encrypted official apps/viewers (e.g., BookWalker/KakaoPage DRM canvases) in v1.

### Anti-Goals (Failure Modes Even If Technically Working)
- **Scrolling stalls or lag:** Waiting 8–12 seconds staring at a blank or untranslated page every time the user scrolls down.
- **Smudged or covered artwork:** Translating background SFX or placing bulky opaque rectangles that bleed outside speech bubbles onto character faces or panel lines.
- **Unreadable lettering:** Microscopic squeezed text, overflowing/clipped words, dark text placed on black bubbles, or blurry upscaled raster text.
- **Robotic, broken, or censored dialogue:** Furigana getting read as duplicate garbage syllables, split bubbles losing sentence flow, character names changing spelling every two pages, or AI safety refusals/sanitized dialogue appearing inside speech bubbles on NSFW, horror, or profanity-heavy series.
- **Technical friction for a non-coder:** Requiring terminal commands every time the user wants to read, broken unsigned browser extensions vanishing on restart, or filling up the nearly-full `C:` drive.

---

## 3. Constraints & Environment (Verified on User's PC)

- **OS:** Windows (64-bit)
- **GPU:** NVIDIA GeForce RTX 3070 — **8,192 MiB (8 GB) VRAM**, Driver `616.64`
- **CPU / RAM:** Intel Core i3-12100F, **31.9 GB RAM**
- **Storage:**
  - `C:` **8.5 GB free** — **STRICT CONSTRAINT: Do NOT install Python packages, PyTorch, HuggingFace caches, or AI models on `C:`.**
  - `D:` **334.2 GB free** (`D:\Code` already exists) — All project files, `.venv`, `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, and GGUF models live under `D:\Code\manga-live-translator`.
- **Installed Runtimes:** Python 3.12.10, Node.js v24.19.0, Git 2.56.0, winget v1.29.380. (Ollama is not installed.)
- **Browsers Installed:**
  - **Target Reading Browser (User Confirmed):** **Microsoft Edge** (`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe` — Chromium-based, supports permanent unpacked Manifest V3 extensions with native toolbar popup & background service worker).
  - Also installed: Google Chrome, Zen Browser.
- **Cost Constraint:** Zero paid APIs; 100% free open-source models running locally.
- **User Persona:** Non-coder. Setup, launching, and daily use must be simple, plain-English, and 1-click where possible.

---

## 4. Approaches Considered & Selection

We evaluated three local architectures that fit the user's RTX 3070 with zero paid APIs:

1. **Option 1 — Specialized Bubble Reader + Smart Local AI Translator *(Selected by User)***
   - Uses a dedicated comic bubble/text detector (explicitly separating dialogue bubbles from floating SFX), specialized Japanese/Korean OCR, and a compact local LLM that translates all dialogue bubbles on a page together in reading order, rendered via a crisp live browser overlay.
   - *Trade-offs:* ~1–3s/page (near-instant with look-ahead pre-translation), high contextual dialogue accuracy, clean bubble-matched lettering, ~6–8 GB disk space on `D:`.
2. **Option 2 — Specialized Bubble Reader + Ultra-Light Classic Translator *(Rejected)***
   - Same detector/OCR/overlay, but swaps the local LLM for a tiny bubble-by-bubble neural translator (OPUS-MT / NLLB style).
   - *Trade-offs:* Under 1s/page and smaller (~2 GB), but translates each bubble in isolation—producing stiff, context-blind manga dialogue and broken pronouns/fragments.
3. **Option 3 — All-in-One Local Vision-Language Model *(Rejected)***
   - Sends raw pages directly to a single local VLM (e.g., Qwen2.5-VL 7B) to detect, read, and translate in one pass.
   - *Trade-offs:* Tight fit in 8 GB VRAM, slower (~5–12s/page), struggles with very tall Korean webtoon strips, and produces less accurate bubble coordinates (causing misaligned overlays).

**Why Option 1 won:** It directly maximizes all three of the user's stated priorities—**speed, accuracy, and readability**—on an RTX 3070 while cleanly honoring the dialogue-only (no SFX) rule.

---

## 5. System Architecture & Design Details

> **Decision Attribution:**
> - **User Decisions:** Windows + regular vertical-scrolling manga websites in **Microsoft Edge** (flexible capture for other reader layouts, scroll-down default); **Edge extension + local background helper** (preferred over standalone `.exe`); live translate-as-you-scroll; dialogue only (skip SFX/onomatopoeia); **uncensored/refusal-free translation** including NSFW, horror/gore, and strong profanity; free/local with no paid API; Option 1 architecture; top priorities = speed, accuracy, readability.
> - **Our Engineering Calls (summarized in Section 7 for user check):** Project location on `D:\Code\manga-live-translator`; opt-in per-site activation (silent on normal websites); stripping furigana via `manga-ocr` + editable Character Name Glossary for custom ateji names; rescuing clean rectangular narration boxes while dropping floating artwork SFX; keeping text strictly inside bubbles + hover-to-read tooltip for cramped vertical bubbles; single tuned uncensored local GGUF model in v1; 1-click desktop launcher + stop control.

### 5.1 Browser Side — Native Microsoft Edge Extension (Manifest V3)

- **Delivery Mechanism (User Confirmed):** Ship as a standalone **Manifest V3 browser extension** (`D:\Code\manga-live-translator\extension\`) loaded once in Microsoft Edge (`edge://extensions` → Developer Mode → Load unpacked).
  - *Why:* Edge is Chromium-based, so an unpacked Manifest V3 extension stays installed permanently across restarts, keeps the user's real Edge bookmarks/logins/ad-blocker intact, requires zero third-party script managers, provides a clean toolbar icon & popup menu, and uses a privileged background service worker to fetch cross-origin manga CDN images and communicate with `http://127.0.0.1:8765` cleanly without website CORS or Content-Security-Policy (CSP) errors.

- **Automatic Page & Image Detection (Scroll-Down Default + Flexible Capture):**
  - Uses `MutationObserver` (including `src`/`srcset` attribute changes for paginated single-page readers!) + `IntersectionObserver` to discover chapter pages dynamically as manga sites lazy-load or swap pages.
  - Filters out thumbnails, icons, ads, avatars, and UI banners using size/aspect heuristics (e.g., rendered width ≥ 300px and height ≥ 350px, or tall webtoon strip aspect ratios), while allowing the user to right-click → **"Translate this image"** on any specific page if a site uses unusual markup.
  - Pluggable capture layer supporting:
    1. **Standard scrolling `<img>` pages (Default)** — fetches raw bytes via the extension's background service worker (`host_permissions: ["<all_urls>"]`, preserving cookies/referer so hotlink-protected CDNs work cleanly).
    2. **Paginated / page-swap readers** — detects `<img>` source updates in place, clears the stale overlay immediately, and translates the newly loaded page.
    3. **Unscrambled `<canvas>` reader pages (Best-effort)** — captures via `canvas.toBlob()` when untainted.
    4. **Wide two-page manga spreads** — when a Japanese manga page has width/height > 1.25, the server splits reading-order sorting right-half first, then left-half so dialogue flows naturally across spreads.

- **Viewport Priority & Look-Ahead Queue:**
  - **Priority 0 (Immediate):** Any manga page currently intersecting the viewport.
  - **Priority 1 (Look-Ahead):** Next **2–3 pages below/ahead** in DOM reading order so pages are already translated before the user scrolls to them.
  - **Cancellation / Deduplication:** In-flight requests for pages scrolled far off-screen are cancelled/deprioritized; completed translations are cached both in-memory (browser tab) and on the local server by image hash (`SHA-256`) + target settings, making scroll-back instantaneous.

- **Default Activation & Site Enablement (Proposed Call — Opt-In Per Site):**
  - Inactive by default on ordinary websites (Gmail, YouTube, Reddit, news, etc.) so it never triggers GPU work or displays floating UI during normal browsing.
  - Clicking the toolbar icon (or app toggle) turns translation ON for that manga site and remembers that domain automatically for future visits.

- **Non-Destructive Visual Overlay & Typography:**
  - Mounts a pointer-transparent overlay container pinned precisely over each manga image (automatically rescaling with `ResizeObserver` if the user zooms or resizes the window).
  - For each detected dialogue bubble:
    1. Covers the original Japanese/Korean glyphs cleanly using the server-computed inner text mask / bubble patch filled with the bubble's sampled interior color (white/off-white for normal bubbles, black/dark gray for dark screaming bubbles)—leaving the bubble's drawn black border and surrounding art crisp and intact.
    2. Renders vector-crisp English text in a readable bundled comic/manga font (open-source comic lettering font served locally, clean easy-to-read manga lettering).
    3. Auto-fits font size and line wrapping inside the bubble's safe inner bounds (LLM is prompted for punchy, manga-concise English phrasing). **Controlled overflow rule (User Approved):** If a translation is too long to fit inside a narrow vertical bubble without dropping below a comfortable minimum readable font size, the text box is permitted to expand **slightly outside the bubble edges (e.g., up to ~20–25% wider/taller)** with a clean matching bubble-colored rounded backdrop and subtle border so readability wins without sprawling across the artwork—plus a hover tooltip showing full English + original text.
  - **Controls (Minimal, Unobtrusive):**
    - Toolbar popup (or top bar) + tiny collapsible floating pill (shown **only** on enabled manga sites) displaying live status (`Ready` / `Translating 2/3` / `Offline`), quick on/off toggle for the current site, language mode (`Auto` / `JP` / `KO`), a simple **Character Name Glossary** editor (e.g., pin `藍星 = Ranshin` for custom ateji readings), and a button to clear chapter context.
    - **Hold `Alt`** (or click eye icon) to temporarily hide overlays and peek at the original Japanese/Korean artwork instantly.
    - **Click/Right-click a bubble** to see the original OCR text or force re-translate that page.

### 5.2 Local Windows GPU Server (`http://127.0.0.1:8765`)

A lightweight Python (`FastAPI` + `uvicorn`) local backend bound **strictly to `127.0.0.1`** (never reachable from the network), orchestrating a 4-stage pipeline carefully budgeted to fit comfortably inside **8 GB VRAM** alongside Windows desktop (~0.6–1.0 GB) and the browser:

| Stage | Component (Proposed Call) | Role | Target VRAM | Target Latency (RTX 3070) |
|---|---|---|---|---|
| **1. Strip Slicing & Bubble Detection** | `ogkalu/comic-text-and-bubble-detector` (ONNX/PyTorch GPU) | Keeps all `bubble` & `text_bubble` regions. For `text_free` regions: **drops floating/stylized SFX over artwork** (like `カサ…`), while rescuing only clean enclosed rectangular narration boxes or plain-background gutter dialogue (high uniform-fill ratio, zero artwork smudging). Automatically slices tall Korean webtoon strips (~1600px chunks with ~250px overlap) and merges/deduplicates detections across slice seams. | ~0.35 GB | ~40–80 ms |
| **2. Script Routing & Specialized OCR** | **Japanese:** `manga-ocr` (`kha-white/manga-ocr`)<br>**Korean:** `RapidOCR` / `PaddleOCR` Korean Hangul ONNX | Reads vertical/horizontal manga text inside each detected bubble crop. Detects dark bubbles (mean interior luminance < 110) and automatically inverts the crop before OCR so white-on-black screams (like `ぎゃああああっ`) read cleanly. Strips furigana automatically to prevent duplicate syllable noise; custom ateji character names (e.g., `藍星` → `Ranshin`) are supported via the user-editable Character Name Glossary rather than a costly separate furigana VLM pass. | ~0.55 GB | ~150–400 ms |
| **3. Context-Aware Uncensored Page Translation** | Single tuned **uncensored / abliterated** local GGUF LLM via bundled `llama-server` (CUDA) — e.g., abliterated `Qwen2.5-7B-Instruct` or abliterated `Qwen3-8B` (`Q4_K_M`, ~4.7 GB, configurable in `config.json`) | Translates **all bubbles on the page in a single structured JSON call**, ordered in true reading order (right-to-left top-to-bottom for Japanese manga, splitting wide spreads right-half → left-half; top-to-bottom left-to-right for Korean manhwa). Faithful, uncensored handling of horror/gore, strong profanity, and NSFW dialogue, backed by refusal-detection guardrails so safety boilerplate never lands in a speech bubble. Prompts for punchy, manga-concise natural English. Maintains a lightweight rolling **per-chapter context & character-name glossary** (last ~10 translated lines + pinned/recurring names) to preserve pronouns, tone, and consistent name spellings across pages. | ~4.8 GB | ~0.8–1.8 s |
| **4. Bubble Masking & Layout Geometry** | OpenCV + Pillow (CPU) | Samples exact bubble interior background color, computes clean inner glyph-erasure mask (shrunk/flood-filled inside the bubble boundary so black bubble outlines stay sharp), computes safe text-box rectangle + text color, and returns compact JSON + optional lightweight mask overlays. | 0 GB (CPU) | ~20–50 ms |
| **Total Peak Budget** | All stages resident in memory | Keeps total GPU usage ~5.7–6.0 GB (leaving ~2.0 GB headroom for Windows + browser on 8 GB RTX 3070, preventing VRAM-to-RAM spill slowdowns). | **~5.7 GB** | **~1.2–2.3 s / page** |

- **Auto Language Detection:**
  - When set to `Auto`, inspects the page aspect ratio and runs a fast lightweight check on the first 1–2 bubbles (checking Hangul vs. Hiragana/Katakana/Kanji character distribution + confidence) and locks the detected language for that chapter tab (overridable anytime in the floating pill).
- **Resilience & Fallbacks:**
  - If the LLM ever returns malformed JSON or misses a bubble index, the server automatically retries once with a stricter prompt and falls back to per-bubble repair so no bubble is left mysteriously untranslated.
  - Empty/non-text false-positive bubbles (e.g., pure symbols or punctuation-only artifacts) are filtered out cleanly so bubbles aren't blanked unnecessarily.
  - Serializes GPU inference cleanly inside a single worker queue (so concurrent browser look-ahead requests never cause CUDA out-of-memory errors) while allowing urgent visible-page requests to jump ahead of lower-priority look-ahead pages.

### 5.3 Non-Coder Setup, Launcher & Verification Harness

- **Zero-`C:`-Drive Footprint:** All setup scripts explicitly set `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `TMP/TEMP` (for installs), and model directories inside `D:\Code\manga-live-translator\`.
- **One-Click Daily Use:**
  - A simple desktop shortcut / `Start Manga Translator.bat` starts the local GPU server and opens a clean local status page at `http://127.0.0.1:8765` showing GPU readiness and a simple 3-step visual guide for loading the Edge extension once.
  - Includes a clean `Stop` button / graceful shutdown so VRAM is completely freed when the user is done reading (e.g., before launching PC games!).
- **Built-In Visual Test Harness:**
  - Before the user even visits a manga site, the local server includes a `/test` page pre-loaded with:
    - **Japanese sample:** The user's uploaded centipede manga page (copied from `C:\Users\ecmat\.gemini\antigravity\brain\a14219b1-1d67-4a8f-970a-a1ad8b0dcea1\.user_uploaded\media_1795062710266_cdab5fd3.jpg` into `D:\Code\manga-live-translator\assets\samples\jp_centipede.jpg`).
    - **Korean sample:** A generated/bundled tall vertical Korean webtoon test strip (`D:\Code\manga-live-translator\assets\samples\ko_webtoon_strip.png`, >3500px tall with multiple Hangul speech bubbles across slice seams) so strip-slicing, seam deduplication, and Hangul OCR/translation are verified end-to-end.
  - Verification criteria on `/test`:
    1. Vertical Japanese base kanji accuracy with furigana cleanly stripped (no duplicate ruby noise: `今！今 助けます！`, `む…蜈蚣` → centipede, `蟲` → bug, `藍星` → transliterated name or pinned glossary entry `Ranshin`).
    2. Black bubble white-text OCR & white-on-black English rendering (`ぎゃああああっ`, `ぎ`).
    3. Floating SFX (`カサ…`) completely skipped and artwork untouched.
    4. Korean tall-strip slicing & Hangul dialogue accuracy across slice boundaries.
    5. End-to-end single-page translation latency under ~3 seconds on the RTX 3070.

---

## 6. What's Left to the Builder

The builder has full freedom to make clean implementation calls on:
- Exact PyTorch CUDA wheel / ONNX Runtime GPU / `llama-server` prebuilt CUDA binary versions compatible with Python 3.12 and NVIDIA driver `616.64`.
- Exact comic detector checkpoint variant (`ogkalu/comic-text-and-bubble-detector` RT-DETR vs YOLOv8) based on live verification against the sample manga page and tall strip slicing accuracy.
- Exact Korean OCR package (`rapidocr-onnxruntime` vs `paddleocr` vs `easyocr`) chosen for easiest clean Windows Python 3.12 install without C++ build tool headaches and highest Hangul accuracy.
- Exact OpenCV bubble-interior segmentation / flood-fill mask algorithm and binary-search CSS/Canvas text-fitting implementation.
- Exact `http://127.0.0.1:8765` API contract, streaming/status endpoints, and on-disk LRU cache pruning policy.
- Bundling an open-source comic lettering font (e.g., `Comic Neue` or `Anime Ace`-style OFL font) locally so typography works offline.

---

## 7. Summary of Engineering Calls Made (For User Check)

These practical defaults were settled during design review and are ready for your sign-off (tell me if you want any of them changed):

1. **Everything lives on `D:\Code\manga-live-translator`:** All code, Python packages, caches, and ~6–7 GB of AI models stay off your nearly-full `C:` drive.
2. **Off by default on normal websites (opt-in per site):** The Edge extension stays completely dormant on non-manga sites (YouTube, Gmail, Reddit, etc.). Click the Edge toolbar icon once on a manga site to enable it and remember that site.
3. **Skip floating artwork SFX, keep clean narration boxes:** Floating sound effects drawn over art (like `カサ…`) are skipped untouched, while clean rectangular narration boxes and speech bubbles are translated.
4. **Clean fit inside bubbles, with slight controlled expansion when needed (User Refined):** English fits inside the bubble outline by default; when a narrow bubble doesn't have enough room without becoming tiny/unreadable, the text box may expand slightly outside the bubble edge with a clean matching rounded backdrop (plus hover tooltip).
5. **Furigana stripped + editable Name Glossary:** Tiny furigana guides are stripped automatically so words don't double-read into gibberish; if a character has a special kanji pronunciation (like `藍星` → `Ranshin`), you can pin it once in the extension's Name Glossary.
6. **Single tuned uncensored local model in v1:** Ships one fast, high-accuracy, uncensored ~7B–8B GGUF model tuned for your 8 GB RTX 3070 (so no bloated second multi-GB model download), plus a refusal check so no safety message ever appears in a bubble.
7. **Easy Start / Stop launcher:** One desktop shortcut launches the local GPU engine, and a clean Stop button shuts it down to free 100% of your RTX 3070 VRAM whenever you're done reading.

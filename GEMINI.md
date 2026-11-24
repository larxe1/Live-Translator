# Project Rules: Live Manga & Manhwa Translator

## 1. Version Bump Checklist (Required on Feature/Fix Releases)
Update the extension version consistently across all six locations before committing a release:
- `extension/manifest.json` (`"version"`)
- `extension/popup.html` (`#versionBadge` initial text)
- `extension/popup.js` (`getLocalVersion()` fallback strings)
- `extension/background.js` (`getLocalVersion()` fallback string)
- `tests/test_extension_autoupdate.mjs` (manifest & status version assertions)
- `tests/test_extension_native_host.mjs` (manifest version assertion)

Versioning policy:
- Use a **patch bump** (`x.y.Z`) for bugfixes.
- Use a **minor bump** (`x.Y.0`) for new features or behavior upgrades.

## 2. Full-Stack Simultaneous GPU/VRAM Verification (RTX 3070 8 GB)
Whenever changing `config.json`, `server/gpu_guard.py`, `server/translator/llama_manager.py`, `server/vision/detector.py`, `server/ocr/engine.py`, or local GGUF model configs:
- Live verification MUST load `llama-server` + `BubbleDetector` + `OcrEngine` (`manga-ocr`) **simultaneously** (not `llama-server` in isolation).
- Confirm `manga-ocr` stays on CUDA (never silently falling back to CPU) with safe free VRAM headroom reported by `nvidia-smi` and healthy GPU temperature.

## 3. Secrets & `.env` Hygiene
- `GEMINI_API_KEY` lives ONLY in gitignored `.env` (loaded via `server/config.py` and updated via `POST /settings`).
- Never write raw API keys into `config.json`, `chrome.storage.local`, logs, tests, or any git-tracked file.
- Always scan `git diff --cached` for `AQ.` and `AIza` patterns (plus standard token/key patterns) before committing.

## 4. Verification Commands
- Python test suite: `.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_e2e_gpu_samples.py`
- Edge extension JS suite: `node --test tests/test_extension_*.mjs`

## 5. Default Target Language — English Only
- All translated dialogue and text across JP, KR, Spanish/other scanlations, Gemini Vision, and local LLMs MUST default to **English ONLY** unless the user explicitly requests another target language.
- Validators and caches must reject/purge non-English outputs (including CJK, Romaji, and Spanish/other non-English Latin languages).

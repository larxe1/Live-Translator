# Live-Translator

Local Japanese manga and Korean manhwa translator for Microsoft Edge. It detects speech bubbles, extracts text with local OCR (manga-ocr and EasyOCR), translates to English using a local `llama.cpp` server (`Qwen2.5-7B-Instruct-abliterated`), and overlays the translated text onto web pages.

## Requirements

- Windows 10 or 11
- Microsoft Edge
- Python 3.12 (`py -3.12` or `python` on `PATH`)
- NVIDIA GPU with CUDA 12.4 recommended (installs PyTorch `cu124` and Windows CUDA `llama-server.exe`; `setup.ps1` falls back to Vulkan `llama.cpp` if no CUDA asset matches)
- Git (optional, required only if pulling updates from a GitHub remote)

## Quick Setup

1. Run the setup script from the repository root in PowerShell:

   ```powershell
   .\setup.ps1
   ```

   This creates `.venv`, installs PyTorch CUDA and `requirements.txt`, downloads `llama-server.exe` to `bin\llama-cpp\`, downloads the `Q4_K_M` GGUF model to `models\`, warms OCR/detector caches under `cache\`, generates test samples, and creates a `Start Manga Translator.lnk` shortcut on your Desktop. All caches and temp directories stay inside the project folder.

2. Load the unpacked extension in Microsoft Edge (one-time):
   - Open `edge://extensions`
   - Enable **Developer mode**
   - Click **Load unpacked** and select `extension\` (for example, `D:\Code\manga-live-translator\extension`)

3. Start translating. You do not need to launch any `.bat` file.
   - `setup.ps1` registers a Native Messaging host (`native_host\host.cmd`) for the current Windows user (HKCU registry, no admin rights). This lets the extension start and stop the local server on its own.
   - Click the extension toolbar icon and press **Start Local Engine**, or just turn on translation for a site. If the server is offline, the extension starts it automatically in a hidden background process with no console window. Output goes to `cache\server.log`.
   - Click the extension toolbar icon on any manga site to enable translation for that domain. Hold `Alt` to peek at the original page.
   - If you load the extension after running `setup.ps1`, or its ID changes, re-run `.venv\Scripts\python.exe native_host\register_host.py` (add `--extension-id <id>` if the ID is not detected automatically).

4. Stop the server with the Stop button in the extension popup. This shuts down the Python server and `llama-server.exe` and frees GPU VRAM.

5. Optional fallback: `Start Manga Translator.bat` (or the Desktop shortcut) and `Stop Manga Translator.bat` still work if you prefer a visible console or Native Messaging is unavailable.

## Auto-Update Behavior

- **Local extension auto-reload:** The backend computes a content hash (`extension_build_id`) over `extension/` and exposes it on `/health` and `/update/status`. When unpacked extension files change locally and auto-update is enabled, the background worker reloads the extension with `chrome.runtime.reload()` (with a 15-second cooldown and deferral while translations are active). Open tabs show an `Updated — Click to refresh` pill to reload the page cleanly.
- **GitHub fast-forward pulls:** When a git remote is configured and the working tree is clean, `Start Manga Translator.bat` checks for updates on startup, and the extension popup (or `POST /update/check` and `POST /update/apply`) can pull updates via `git pull --ff-only`. If `server/`, `requirements.txt`, or `config.json` changes, the server exits with code `42` under `MLT_AUTO_RESTART=1` so the launcher restarts the backend without reopening Edge.

## Troubleshooting

- **Virtual environment missing:** Run `.\setup.ps1` before launching `Start Manga Translator.bat`.
- **Tab shows `Updated — Click to refresh`:** Click the pill or refresh the tab after the extension reloads.
- **Extension says the native host is missing:** Re-run `.venv\Scripts\python.exe native_host\register_host.py`, then reload the extension. Check `cache\native-host.log` and `cache\server.log`.
- **Server offline or stuck process on port 8765:** Run `Stop Manga Translator.bat`, check `cache\llama-server.log`, and run `Start Manga Translator.bat` again.
- **GPU VRAM limits:** Adjust `llm_gpu_layers` (default `23`), `llm_ctx_size` (default `4096`), or `idle_unload_minutes` (default `3`) in `config.json`.
- **Remote update skipped:** Automatic pulls only run fast-forward merges when `git status --porcelain` is empty.

## Project Layout

- `server/` - FastAPI service (`app.py`), pipeline, vision detector, OCR, `llama.cpp` manager, SQLite cache, and updater
- `extension/` - Microsoft Edge MV3 extension (`manifest.json`, `background.js`, `content.js`, `overlay.js`, `popup.html`, `popup.js`)
- `native_host/` - Native Messaging host (`host.py`, `host.cmd`), silent server supervisor (`supervisor.py`), and registry installer (`register_host.py`)
- `setup.ps1` - Windows environment, model, binary, and native host installer
- `Start Manga Translator.bat` / `Stop Manga Translator.bat` - Start and stop scripts
- `config.json` - Local server, port, model, and GPU configuration
- `tests/` - Python and Node tests

@echo off
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "CACHE_DIR=%PROJECT_ROOT%\cache"
set "TEMP=%CACHE_DIR%\tmp"
set "TMP=%CACHE_DIR%\tmp"
set "PIP_CACHE_DIR=%CACHE_DIR%\pip"
set "HF_HOME=%CACHE_DIR%\huggingface"
set "TORCH_HOME=%CACHE_DIR%\torch"
set "EASYOCR_MODULE_PATH=%CACHE_DIR%\easyocr"
set "XDG_CACHE_HOME=%CACHE_DIR%\xdg"
set "PLAYWRIGHT_BROWSERS_PATH=%CACHE_DIR%\playwright"

if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"
if not exist "%EASYOCR_MODULE_PATH%" mkdir "%EASYOCR_MODULE_PATH%"
if not exist "%XDG_CACHE_HOME%" mkdir "%XDG_CACHE_HOME%"
if not exist "%PLAYWRIGHT_BROWSERS_PATH%" mkdir "%PLAYWRIGHT_BROWSERS_PATH%"

if not exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at "%PROJECT_ROOT%\.venv".
    echo Please run setup.ps1 first.
    pause
    exit /b 1
)

echo =====================================================
echo   Starting Live Manga & Manhwa Dialogue Translator
echo   Local Server: http://127.0.0.1:8765
echo   All caches pinned to: %CACHE_DIR%
echo =====================================================

cd /d "%PROJECT_ROOT%"
call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"

REM Open Microsoft Edge once port 8765 is listening
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "for ($i = 0; $i -lt 60; $i++) { try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8765/health' -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { break } } catch {}; Start-Sleep -Milliseconds 500 }; $edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'; if (Test-Path $edge) { Start-Process $edge 'http://127.0.0.1:8765' } else { Start-Process 'http://127.0.0.1:8765' }"

python -m server.app
endlocal

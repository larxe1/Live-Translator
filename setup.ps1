$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ProjectRoot) {
    $ProjectRoot = "D:\Code\manga-live-translator"
}

$CacheDir = Join-Path $ProjectRoot "cache"
$TempDir = Join-Path $CacheDir "tmp"
$DownloadsDir = Join-Path $CacheDir "downloads"
$PipCacheDir = Join-Path $CacheDir "pip"
$HfHomeDir = Join-Path $CacheDir "huggingface"
$TorchHomeDir = Join-Path $CacheDir "torch"
$EasyOcrDir = Join-Path $CacheDir "easyocr"
$XdgCacheDir = Join-Path $CacheDir "xdg"
$PlaywrightDir = Join-Path $CacheDir "playwright"

$VenvDir = Join-Path $ProjectRoot ".venv"
$ModelsDir = Join-Path $ProjectRoot "models"
$LlamaBinDir = Join-Path $ProjectRoot "bin\llama-cpp"

foreach ($dir in @($CacheDir, $TempDir, $DownloadsDir, $PipCacheDir, $HfHomeDir, $TorchHomeDir, $EasyOcrDir, $XdgCacheDir, $PlaywrightDir, $ModelsDir, $LlamaBinDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# Pin ALL caches and temp files strictly to D:\Code\manga-live-translator\cache
$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:PIP_CACHE_DIR = $PipCacheDir
$env:HF_HOME = $HfHomeDir
$env:TORCH_HOME = $TorchHomeDir
$env:EASYOCR_MODULE_PATH = $EasyOcrDir
$env:XDG_CACHE_HOME = $XdgCacheDir
$env:PLAYWRIGHT_BROWSERS_PATH = $PlaywrightDir

Write-Host "=== Manga Live Translator Setup (D:-Drive Only) ===" -ForegroundColor Cyan
Write-Host "Project Root : $ProjectRoot"
Write-Host "Cache Root   : $CacheDir"

# 1. Create .venv on D:
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "`n[1/6] Creating Python 3.12 virtual environment at $VenvDir ..." -ForegroundColor Yellow
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv $VenvDir
    } else {
        & python -m venv $VenvDir
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        throw "Failed to create Python virtual environment."
    }
} else {
    Write-Host "`n[1/6] Virtual environment already exists at $VenvDir." -ForegroundColor Green
}

# 2. Install PyTorch CUDA + Python dependencies
Write-Host "`n[2/6] Installing PyTorch CUDA (cu124) & requirements into .venv ..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip --cache-dir $PipCacheDir
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

& $VenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --cache-dir $PipCacheDir
if ($LASTEXITCODE -ne 0) { throw "Failed to install PyTorch CUDA." }

& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt") --cache-dir $PipCacheDir
if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements.txt." }

# 3. Download latest prebuilt Windows CUDA llama.cpp release
$LlamaServerExe = Join-Path $LlamaBinDir "llama-server.exe"
if (-not (Test-Path $LlamaServerExe)) {
    Write-Host "`n[3/6] Downloading prebuilt Windows CUDA llama.cpp release ..." -ForegroundColor Yellow
    $Headers = @{ "User-Agent" = "manga-live-translator-setup" }
    $Releases = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15" -Headers $Headers
    if (-not $Releases) {
        $Latest = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" -Headers $Headers
        $Releases = @($Latest)
    }

    $MainAsset = $null
    $MatchedAssets = $null
    $Patterns = @(
        "^llama-.*-bin-win-cuda-(cu)?12\..*-x64\.zip$",
        "^llama-.*-bin-win-cuda-.*-x64\.zip$",
        "^llama-.*-bin-win-vulkan-x64\.zip$"
    )
    foreach ($Pat in $Patterns) {
        foreach ($Rel in $Releases) {
            if (-not $Rel.assets) { continue }
            $Candidate = $Rel.assets | Where-Object { $_.name -match $Pat } | Select-Object -First 1
            if ($Candidate) {
                $MainAsset = $Candidate
                $MatchedAssets = $Rel.assets
                break
            }
        }
        if ($MainAsset) { break }
    }

    if (-not $MainAsset) {
        throw "Could not find a compatible Windows CUDA or Vulkan llama.cpp release asset."
    }

    $CudartAsset = $null
    if ($MainAsset.name -match "cuda") {
        $CudartAsset = $MatchedAssets | Where-Object { $_.name -match "^cudart-llama-bin-win-cuda-(cu)?12\..*-x64\.zip$" } | Select-Object -First 1
        if (-not $CudartAsset) {
            $CudartAsset = $MatchedAssets | Where-Object { $_.name -match "^cudart-llama-bin-win-cuda-.*-x64\.zip$" } | Select-Object -First 1
        }
    }

    $ZipFiles = @($MainAsset)
    if ($CudartAsset) { $ZipFiles += $CudartAsset }

    foreach ($Asset in $ZipFiles) {
        $ZipPath = Join-Path $DownloadsDir $Asset.name
        Write-Host "  Downloading $($Asset.name) ..."
        Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $ZipPath -Headers $Headers
        $ExtractTmp = Join-Path $TempDir ("extract_" + [System.IO.Path]::GetFileNameWithoutExtension($Asset.name))
        if (Test-Path $ExtractTmp) { Remove-Item -Recurse -Force $ExtractTmp }
        Expand-Archive -Path $ZipPath -DestinationPath $ExtractTmp -Force

        # Flatten files into $LlamaBinDir
        Get-ChildItem -Path $ExtractTmp -Recurse -File | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination (Join-Path $LlamaBinDir $_.Name) -Force
        }
        Remove-Item -Recurse -Force $ExtractTmp
    }

    if (-not (Test-Path $LlamaServerExe)) {
        throw "llama-server.exe was not found in $LlamaBinDir after extraction."
    }
} else {
    Write-Host "`n[3/6] llama-server.exe already present at $LlamaServerExe." -ForegroundColor Green
}

# 4. Download uncensored abliterated Qwen2.5-7B-Instruct Q4_K_M GGUF model
Write-Host "`n[4/6] Downloading abliterated Qwen2.5-7B-Instruct Q4_K_M GGUF into $ModelsDir ..." -ForegroundColor Yellow
$DownloadModelPy = @"
import fnmatch
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

models_dir = Path(r"$ModelsDir")
existing = list(models_dir.glob("*Q4_K_M*.gguf")) + list(models_dir.glob("*.gguf"))
if existing:
    print("Model already present:", existing[0])
    raise SystemExit(0)

repos = [
    ("richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF", ["*Q4_K_M*.gguf"]),
    ("bartowski/Qwen2.5-7B-Instruct-abliterated-GGUF", ["*Q4_K_M*.gguf"]),
    ("mradermacher/Qwen2.5-7B-Instruct-abliterated-GGUF", ["*Q4_K_M*.gguf"]),
]

for repo_id, patterns in repos:
    try:
        files = list_repo_files(repo_id)
        matches = []
        for pat in patterns:
            matches.extend([f for f in files if fnmatch.fnmatch(Path(f).name.lower(), pat.lower()) and "-00001-of-" not in f.lower()])
        if not matches:
            for pat in patterns:
                matches.extend([f for f in files if fnmatch.fnmatch(Path(f).name.lower(), pat.lower())])
        if matches:
            target = matches[0]
            print(f"Downloading {repo_id} / {target} ...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=target,
                local_dir=str(models_dir),
            )
            print("Saved GGUF:", local_path)
            raise SystemExit(0)
    except Exception as exc:
        print(f"Repo {repo_id} fallback triggered: {exc}")

raise RuntimeError("Could not download any Q4_K_M abliterated Qwen2.5-7B GGUF model.")
"@
$DownloadModelScript = Join-Path $TempDir "download_gguf.py"
Set-Content -Path $DownloadModelScript -Value $DownloadModelPy -Encoding UTF8
& $VenvPython $DownloadModelScript
if ($LASTEXITCODE -ne 0) { throw "Failed to download GGUF model." }

# 5. Pre-warm vision/OCR models onto D:\cache and build samples/fonts/icons
Write-Host "`n[5/6] Pre-warming detector/OCR caches onto D: and generating test samples ..." -ForegroundColor Yellow
$PrewarmPy = @"
import os
from huggingface_hub import snapshot_download

for repo_id in (
    "ogkalu/comic-text-and-bubble-detector",
    "kha-white/manga-ocr-base",
):
    try:
        print(f"Caching {repo_id} ...")
        snapshot_download(repo_id)
    except Exception as exc:
        print(f"HuggingFace prewarm note ({repo_id}):", exc)

try:
    import easyocr
    print("Caching EasyOCR Korean reader ...")
    easyocr.Reader(
        ["ko"],
        gpu=False,
        model_storage_directory=os.environ["EASYOCR_MODULE_PATH"],
        download_enabled=True,
    )
except Exception as exc:
    print("EasyOCR prewarm note:", exc)
"@
$PrewarmScript = Join-Path $TempDir "prewarm_models.py"
Set-Content -Path $PrewarmScript -Value $PrewarmPy -Encoding UTF8
& $VenvPython $PrewarmScript
if ($LASTEXITCODE -ne 0) { Write-Warning "Model pre-warm returned non-zero; models will download on first run." }

& $VenvPython (Join-Path $ProjectRoot "scripts\make_samples.py")
if ($LASTEXITCODE -ne 0) { throw "make_samples.py failed." }

# 6. Register the Edge native messaging host (HKCU, no admin) and create Desktop shortcut
Write-Host "`n[6/6] Registering native messaging host and creating Desktop shortcut ..." -ForegroundColor Yellow
$RegisterHost = Join-Path $ProjectRoot "native_host\register_host.py"
& $VenvPython $RegisterHost
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Native host registration failed. The extension cannot auto-start the server; use Start Manga Translator.bat instead."
} else {
    Write-Host "Native host registered. The extension can now start and stop the local server by itself." -ForegroundColor Green
}

$DesktopDir = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopDir "Start Manga Translator.lnk"
$TargetBat = Join-Path $ProjectRoot "Start Manga Translator.bat"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetBat
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Launch Live Manga & Manhwa Dialogue Translator (Local GPU)"
$Shortcut.Save()

Write-Host "`nSetup complete! All files & caches reside on $ProjectRoot." -ForegroundColor Green
Write-Host "Shortcut created: $ShortcutPath" -ForegroundColor Green

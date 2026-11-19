@echo off
setlocal EnableExtensions

echo Stopping Live Manga & Manhwa Dialogue Translator and freeing RTX 3070 VRAM...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8765/shutdown' -TimeoutSec 4 | Out-Null; Write-Host 'Sent /shutdown signal to local server.' } catch { Write-Host 'Local server not responding on :8765 (may already be stopped).' }"

timeout /t 2 /nobreak >nul

REM Ensure any lingering llama-server.exe or server.app started from this project is terminated
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$procs = Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'llama-server.exe' -and $_.ExecutablePath -like '*manga-live-translator\bin\llama-cpp*') -or ($_.Name -eq 'python.exe' -and $_.CommandLine -like '*manga-live-translator*' -and $_.CommandLine -like '*server.app*') }; foreach ($p in $procs) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Host ('Stopped process ' + $p.Name + ' (PID ' + $p.ProcessId + ')') } catch {} }"

echo Done. RTX 3070 VRAM is free.
endlocal

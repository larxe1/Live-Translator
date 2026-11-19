"""Lifecycle manager for the bundled local CUDA llama-server.exe process."""

from __future__ import annotations

import fnmatch
import os
import subprocess
import time
from pathlib import Path
from typing import IO

import httpx

from server.config import AppConfig


class LlamaServerManager:
    """Starts, health-checks, and cleanly stops the local llama-server child process."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.base_url: str = f"http://127.0.0.1:{cfg.llama_port}"
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file: IO[bytes] | None = None
        self.model_name: str = self._initial_model_name()

    def _initial_model_name(self) -> str:
        try:
            return self.resolve_model_path().name
        except FileNotFoundError:
            return self.cfg.llm_repo_id

    def resolve_model_path(self) -> Path:
        """Resolve explicit cfg.llm_model_path or locate the first .gguf in cfg.models_dir."""
        if self.cfg.llm_model_path is not None:
            candidate = Path(self.cfg.llm_model_path)
            if not candidate.is_absolute():
                candidate = (self.cfg.project_root / candidate).resolve()
            if candidate.is_file():
                return candidate

        models_dir = self.cfg.models_dir or (self.cfg.project_root / "models")
        if models_dir.is_dir():
            ggufs = sorted(p for p in models_dir.rglob("*.gguf") if p.is_file())
            if ggufs:
                pattern = self.cfg.llm_filename_pattern
                if pattern:
                    for path in ggufs:
                        if fnmatch.fnmatch(path.name, pattern):
                            return path
                return ggufs[0]

        raise FileNotFoundError(
            f"No GGUF model found (checked llm_model_path={self.cfg.llm_model_path} "
            f"and models_dir={models_dir}). Run setup.ps1 to download the local model."
        )

    def _resolve_server_exe(self) -> Path:
        """Locate llama-server.exe inside cfg.llama_bin_dir."""
        bin_dir = self.cfg.llama_bin_dir or (self.cfg.project_root / "bin" / "llama-cpp")
        exe_names = ("llama-server.exe", "llama-server")
        for name in exe_names:
            direct = bin_dir / name
            if direct.is_file():
                return direct
        if bin_dir.is_dir():
            for name in exe_names:
                matches = sorted(bin_dir.rglob(name))
                if matches:
                    return matches[0]
        raise FileNotFoundError(
            f"llama-server executable not found under {bin_dir}. Run setup.ps1 first."
        )

    def build_command(self, exe_path: Path, model_path: Path) -> list[str]:
        """Build the llama-server CLI invocation with fixed VRAM-safe context/batch caps."""
        ctx_size = min(int(self.cfg.llm_ctx_size), 4096) if self.cfg.llm_ctx_size > 0 else 4096
        gpu_layers = int(self.cfg.llm_gpu_layers)
        return [
            str(exe_path),
            "--model",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.cfg.llama_port),
            "-ngl",
            str(gpu_layers),
            "--ctx-size",
            str(ctx_size),
            "--batch-size",
            "512",
            "--parallel",
            "1",
        ]

    def is_healthy(self) -> bool:
        """Return True when llama-server responds HTTP 200 on /health."""
        if self._process is not None and self._process.poll() is not None:
            return False
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=2.0)
            return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def start(self, timeout_s: float = 90.0) -> None:
        """Start llama-server.exe and block until /health returns 200 or timeout expires."""
        if self.is_healthy():
            return

        model_path = self.resolve_model_path()
        self.model_name = model_path.name
        exe_path = self._resolve_server_exe()
        cmd = self.build_command(exe_path, model_path)

        cache_dir = self.cfg.cache_dir or (self.cfg.project_root / "cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        log_path = cache_dir / "llama-server.log"
        self._close_log_file()
        self._log_file = open(log_path, "wb")

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self._process = subprocess.Popen(
            cmd,
            cwd=str(exe_path.parent),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            exit_code = self._process.poll()
            if exit_code is not None:
                self._close_log_file()
                tail = ""
                try:
                    raw = log_path.read_text(encoding="utf-8", errors="replace")
                    tail = raw[-1000:].strip()
                except OSError:
                    pass
                self._process = None
                raise RuntimeError(
                    f"llama-server exited early with code {exit_code}. Log tail:\n{tail}"
                )
            if self.is_healthy():
                return
            time.sleep(0.5)

        self.stop()
        raise TimeoutError(
            f"llama-server did not become healthy at {self.base_url}/health within {timeout_s}s"
        )

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    def stop(self) -> None:
        """Gracefully terminate llama-server.exe and wait so 100% of VRAM is freed."""
        proc = self._process
        self._process = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=8.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5.0)
            except OSError:
                pass
        self._close_log_file()

"""Lifecycle manager for the bundled local CUDA llama-server.exe process."""

from __future__ import annotations

import fnmatch
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import IO

import httpx

from server.config import AppConfig, resolve_local_model
from server.gpu_guard import GpuSafetyGuard

logger = logging.getLogger(__name__)


def infer_total_layers(model_filename: str) -> int:
    """Return transformer layer count for VRAM offloading estimates."""
    fname = (model_filename or "").lower()
    if "qwen2.5-7b" in fname or "shisa" in fname:
        return 29
    if "exaone" in fname:
        return 33
    return 37


class LlamaServerManager:
    """Starts, health-checks, switches, and cleanly stops the local llama-server child process."""

    def __init__(self, cfg: AppConfig, gpu_guard: GpuSafetyGuard | None = None) -> None:
        self.cfg = cfg
        self.gpu_guard = gpu_guard if gpu_guard is not None else GpuSafetyGuard(cfg)
        self.base_url: str = f"http://127.0.0.1:{cfg.llama_port}"
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file: IO[bytes] | None = None
        self._lock = threading.RLock()
        self.active_model_filename: str = ""
        self.active_model_role: str = ""
        self.model_name: str = self._initial_model_name()

    def _initial_model_name(self) -> str:
        try:
            return self.resolve_model_path().name
        except FileNotFoundError:
            return getattr(self.cfg, "llm_filename", "") or self.cfg.llm_repo_id

    def _models_dir(self) -> Path:
        return self.cfg.models_dir or (self.cfg.project_root / "models")

    def _list_ggufs(self) -> list[Path]:
        models_dir = self._models_dir()
        if not models_dir.is_dir():
            return []
        return sorted(p for p in models_dir.rglob("*.gguf") if p.is_file())

    def _find_gguf_by_name(self, filename: str) -> Path | None:
        target = (filename or "").strip().lower()
        if not target:
            return None
        for path in self._list_ggufs():
            if path.name.lower() == target:
                return path
        if self.cfg.llm_model_path is not None:
            candidate = Path(self.cfg.llm_model_path)
            if not candidate.is_absolute():
                candidate = (self.cfg.project_root / candidate).resolve()
            if candidate.is_file() and candidate.name.lower() == target:
                return candidate
        return None

    def _infer_role_for_filename(self, filename: str) -> str:
        target = (filename or "").strip().lower()
        local_models = getattr(self.cfg, "local_models", None) or {}
        for role in ("ja", "ja_uncensored", "ko", "default"):
            entry = local_models.get(role)
            if isinstance(entry, dict) and str(entry.get("filename", "")).strip().lower() == target:
                return role
        for role, entry in local_models.items():
            if isinstance(entry, dict) and str(entry.get("filename", "")).strip().lower() == target:
                return str(role)
        return "default"

    def available_local_models(self) -> dict[str, bool]:
        """Return availability map {role: is_downloaded} for configured local_models."""
        local_models = getattr(self.cfg, "local_models", None) or {}
        present_names = {p.name.lower() for p in self._list_ggufs()}
        availability: dict[str, bool] = {}
        for role, entry in local_models.items():
            if not isinstance(entry, dict):
                continue
            fname = str(entry.get("filename", "") or "").strip().lower()
            availability[str(role)] = bool(fname and fname in present_names)
        return availability

    def ensure_model_downloaded(self, repo_id: str, filename: str) -> Path:
        """Download a GGUF file from HuggingFace into models_dir if missing."""
        existing = self._find_gguf_by_name(filename)
        if existing is not None:
            return existing
        models_dir = self._models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(models_dir),
        )
        return Path(downloaded)

    def resolve_model_path(self, model_filename: str | None = None) -> Path:
        """Resolve an explicit model_filename, cfg.llm_model_path, or best .gguf in models_dir."""
        if model_filename is not None:
            matched = self._find_gguf_by_name(model_filename)
            if matched is not None:
                return matched
            raise FileNotFoundError(
                f"Requested GGUF model {model_filename!r} not found in {self._models_dir()}."
            )

        if self.cfg.llm_model_path is not None:
            candidate = Path(self.cfg.llm_model_path)
            if not candidate.is_absolute():
                candidate = (self.cfg.project_root / candidate).resolve()
            if candidate.is_file():
                return candidate

        ggufs = self._list_ggufs()
        if ggufs:
            target_name = str(getattr(self.cfg, "llm_filename", "") or "").strip()
            pattern = str(getattr(self.cfg, "llm_filename_pattern", "") or "").strip()
            # If caller overrode pattern without matching target_name, honor pattern first.
            if pattern and (not target_name or not fnmatch.fnmatch(target_name.lower(), pattern.lower())):
                for path in ggufs:
                    if fnmatch.fnmatch(path.name.lower(), pattern.lower()):
                        return path
            if target_name:
                for path in ggufs:
                    if path.name.lower() == target_name.lower():
                        return path
            if pattern:
                for path in ggufs:
                    if fnmatch.fnmatch(path.name.lower(), pattern.lower()):
                        return path
            return ggufs[0]

        raise FileNotFoundError(
            f"No GGUF model found (checked llm_model_path={self.cfg.llm_model_path} "
            f"and models_dir={self._models_dir()}). Run setup.ps1 to download the local model."
        )

    def _pick_fallback_on_disk(self) -> tuple[str, Path] | None:
        """Pick the best already-downloaded GGUF in models_dir without blocking on network."""
        ggufs = self._list_ggufs()
        if not ggufs:
            return None

        local_models = getattr(self.cfg, "local_models", None) or {}
        preferred_names: list[tuple[str, str]] = []
        for role in ("ja_uncensored", "default"):
            entry = local_models.get(role)
            if isinstance(entry, dict) and entry.get("filename"):
                preferred_names.append((role, str(entry["filename"])))
        preferred_names.append(("ja_uncensored", "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf"))
        for role in ("ja", "ko"):
            entry = local_models.get(role)
            if isinstance(entry, dict) and entry.get("filename"):
                preferred_names.append((role, str(entry["filename"])))

        by_lower = {p.name.lower(): p for p in ggufs}
        for role, fname in preferred_names:
            hit = by_lower.get(fname.strip().lower())
            if hit is not None:
                return role, hit

        first = ggufs[0]
        return self._infer_role_for_filename(first.name), first

    def ensure_model(self, lang: str = "ja", uncensored: bool = False) -> str:
        """Ensure llama-server is running the language/role-appropriate local GGUF model."""
        with self._lock:
            role, repo_id, filename = resolve_local_model(self.cfg, lang, uncensored=uncensored)
            target_path = self._find_gguf_by_name(filename)

            if target_path is None:
                fallback = self._pick_fallback_on_disk()
                if fallback is not None:
                    fallback_role, fallback_path = fallback
                    logger.warning(
                        "Target model %s (%s) not on disk; falling back to %s (%s)",
                        filename,
                        role,
                        fallback_path.name,
                        fallback_role,
                    )
                    role = fallback_role
                    filename = fallback_path.name
                else:
                    downloaded = self.ensure_model_downloaded(repo_id, filename)
                    filename = downloaded.name

            if (
                self.is_healthy()
                and self.active_model_filename
                and self.active_model_filename.lower() == filename.lower()
            ):
                self.active_model_role = role
                return self.active_model_role

            if self._process is not None or self.is_healthy():
                logger.info(
                    "Switching llama-server model: %s (%s) -> %s (%s)",
                    self.active_model_filename or "none",
                    self.active_model_role or "none",
                    filename,
                    role,
                )
                self.stop()

            self.start(model_filename=filename, repo_id=repo_id, role=role)
            return self.active_model_role

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
        """Build the llama-server CLI invocation with VRAM-safe context, KV, and layer caps."""
        raw_ctx = int(getattr(self.cfg, "llm_ctx_size", 2048))
        ctx_size = min(raw_ctx, 4096) if raw_ctx > 0 else 2048
        batch_size = max(1, int(getattr(self.cfg, "llm_batch_size", 256)))
        requested_layers = int(getattr(self.cfg, "llm_gpu_layers", 99))
        estimated_model_mb = (
            int(model_path.stat().st_size / (1024 * 1024)) + 200
            if model_path.is_file()
            else 4500
        )
        fname = model_path.name.lower()
        total_layers = infer_total_layers(fname)
        gpu_layers = self.gpu_guard.choose_safe_gpu_layers(
            requested_layers,
            total_layers=total_layers,
            estimated_model_mb=estimated_model_mb,
        )
        cmd = [
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
            str(batch_size),
            "--ubatch-size",
            str(batch_size),
            "--parallel",
            "1",
            "--flash-attn",
            "on",
            "--cache-prompt",
        ]
        if "qwen3" in fname:
            cmd.extend(["--reasoning", "off"])
        kv_type = str(getattr(self.cfg, "llm_kv_cache_type", "") or "").strip()
        if kv_type:
            cmd.extend(["-ctk", kv_type, "-ctv", kv_type])
        return cmd

    def is_healthy(self) -> bool:
        """Return True when llama-server responds HTTP 200 on /health."""
        if self._process is not None and self._process.poll() is not None:
            return False
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=2.0)
            return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def start(
        self,
        timeout_s: float = 90.0,
        model_filename: str | None = None,
        repo_id: str | None = None,
        role: str | None = None,
    ) -> None:
        """Start llama-server.exe and block until /health returns 200 or timeout expires."""
        del repo_id  # Reserved for caller clarity / download symmetry.
        with self._lock:
            model_path = self.resolve_model_path(model_filename=model_filename)
            resolved_role = role or self._infer_role_for_filename(model_path.name)

            if self.is_healthy():
                if (
                    model_filename is None
                    or self.active_model_filename.lower() == model_path.name.lower()
                ):
                    self.model_name = model_path.name
                    self.active_model_filename = model_path.name
                    if not self.active_model_role or role is not None:
                        self.active_model_role = resolved_role
                    return
                self.stop()

            self.model_name = model_path.name
            self.active_model_filename = model_path.name
            self.active_model_role = resolved_role
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
                    self.active_model_filename = ""
                    self.active_model_role = ""
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
        with self._lock:
            proc = self._process
            self._process = None
            self.active_model_filename = ""
            self.active_model_role = ""
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

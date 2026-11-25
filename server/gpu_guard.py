"""GPU telemetry, VRAM headroom checks, and thermal safety guard."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from server.config import AppConfig

logger = logging.getLogger(__name__)

_SNAPSHOT_TTL_SEC = 5.0
_snapshot_lock = threading.Lock()
_snapshot_cache: tuple[float, GpuSnapshot] | None = None


@dataclass
class GpuSnapshot:
    """Point-in-time GPU telemetry from nvidia-smi."""

    available: bool
    name: str
    vram_used_mb: int
    vram_total_mb: int
    vram_free_mb: int
    temp_c: int
    utilization_pct: int
    power_w: float
    power_limit_w: float
    sampled_at: float


def _unavailable_snapshot(sampled_at: float | None = None) -> GpuSnapshot:
    ts = time.monotonic() if sampled_at is None else sampled_at
    return GpuSnapshot(
        available=False,
        name="Unavailable",
        vram_used_mb=0,
        vram_total_mb=0,
        vram_free_mb=0,
        temp_c=0,
        utilization_pct=0,
        power_w=0.0,
        power_limit_w=0.0,
        sampled_at=ts,
    )


def _safe_int(raw: str, default: int = 0) -> int:
    val = raw.strip()
    if not val or val.startswith("["):
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _safe_float(raw: str, default: float = 0.0) -> float:
    val = raw.strip()
    if not val or val.startswith("["):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _query_nvidia_smi() -> GpuSnapshot:
    """Query nvidia-smi once; never raises."""
    now = time.monotonic()
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,memory.free,temperature.gpu,utilization.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
            creationflags=creationflags,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return _unavailable_snapshot(now)

        first_line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 8:
            return _unavailable_snapshot(now)

        name = parts[0] or "Unknown GPU"
        used_mb = _safe_int(parts[1])
        total_mb = _safe_int(parts[2])
        free_mb = _safe_int(parts[3], default=max(0, total_mb - used_mb))
        temp_c = _safe_int(parts[4])
        util_pct = _safe_int(parts[5])
        power_w = _safe_float(parts[6])
        power_limit_w = _safe_float(parts[7])
        return GpuSnapshot(
            available=True,
            name=name,
            vram_used_mb=used_mb,
            vram_total_mb=total_mb,
            vram_free_mb=free_mb,
            temp_c=temp_c,
            utilization_pct=util_pct,
            power_w=power_w,
            power_limit_w=power_limit_w,
            sampled_at=time.monotonic(),
        )
    except Exception:
        return _unavailable_snapshot(now)


def get_gpu_snapshot(force_refresh: bool = False) -> GpuSnapshot:
    """Return cached GPU telemetry (5s TTL) via nvidia-smi; never raises."""
    global _snapshot_cache
    now = time.monotonic()
    if not force_refresh:
        with _snapshot_lock:
            if _snapshot_cache is not None and (now - _snapshot_cache[0]) < _SNAPSHOT_TTL_SEC:
                return _snapshot_cache[1]

    snap = _query_nvidia_smi()
    with _snapshot_lock:
        _snapshot_cache = (time.monotonic(), snap)
    return snap


def has_safe_vram_headroom(min_free_mb: int) -> bool:
    """Check free VRAM via nvidia-smi first (WDDM-accurate), falling back to torch.cuda."""
    snap = _read_snapshot(force_refresh=True)
    if snap.available:
        return snap.vram_free_mb >= int(min_free_mb)

    try:
        import torch

        if not torch.cuda.is_available():
            return False
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return (free_bytes / (1024 * 1024)) >= float(min_free_mb)
    except Exception:
        return False


def _read_snapshot(force_refresh: bool = False) -> GpuSnapshot:
    """Call get_gpu_snapshot safely even if monkeypatched with a 0-arg callable in tests."""
    if force_refresh:
        try:
            return get_gpu_snapshot(force_refresh=True)
        except TypeError:
            return get_gpu_snapshot()
    return get_gpu_snapshot()


class GpuSafetyGuard:
    """VRAM headroom and thermal throttling/pause guard for local GPU inference."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.min_free_vram_mb = int(getattr(cfg, "gpu_min_free_vram_mb", 1024))
        self.llama_headroom_mb = int(getattr(cfg, "gpu_llama_headroom_mb", 600))
        self.soft_temp_c = int(getattr(cfg, "gpu_temp_soft_c", 78))
        self.hard_temp_c = int(getattr(cfg, "gpu_temp_hard_c", 84))
        self.resume_temp_c = int(getattr(cfg, "gpu_temp_resume_c", 76))
        self.cooldown_seconds = max(0.0, float(getattr(cfg, "gpu_cooldown_seconds", 1.5)))

    def status(self) -> dict[str, Any]:
        """Return structured GPU safety telemetry for /health and extension UI."""
        snap = _read_snapshot()
        if not snap.available:
            state = "unavailable"
            message = "GPU telemetry unavailable"
        elif snap.temp_c >= self.hard_temp_c:
            state = "paused"
            message = f"GPU temperature {snap.temp_c}C >= hard limit {self.hard_temp_c}C; cooling down"
        elif snap.temp_c >= self.soft_temp_c:
            state = "throttled"
            message = f"GPU temperature {snap.temp_c}C >= soft limit {self.soft_temp_c}C; throttling"
        else:
            state = "ok"
            if snap.vram_free_mb < self.min_free_vram_mb:
                message = (
                    f"GPU nominal ({snap.temp_c}C), low free VRAM "
                    f"({snap.vram_free_mb} MiB < {self.min_free_vram_mb} MiB)"
                )
            else:
                message = f"GPU nominal ({snap.temp_c}C, {snap.vram_free_mb} MiB free)"

        return {
            "state": state,
            "temp_c": int(snap.temp_c),
            "vram_used_mb": int(snap.vram_used_mb),
            "vram_total_mb": int(snap.vram_total_mb),
            "vram_free_mb": int(snap.vram_free_mb),
            "min_free_vram_mb": self.min_free_vram_mb,
            "soft_temp_c": self.soft_temp_c,
            "hard_temp_c": self.hard_temp_c,
            "message": message,
        }

    async def _sleep_or_cancel(
        self,
        delay_s: float,
        cancel_event: asyncio.Event | None,
    ) -> bool:
        """Sleep for delay_s unless cancel_event triggers first. Returns True if cancelled."""
        if cancel_event is None:
            await asyncio.sleep(max(0.0, delay_s))
            return False
        if cancel_event.is_set():
            return True
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=max(0.0, delay_s))
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_if_hot(
        self,
        cancel_event: asyncio.Event | None = None,
        max_wait_s: float = 45.0,
    ) -> str:
        """Pause or throttle when GPU temperature crosses soft/hard safety thresholds."""
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"

        snap = await asyncio.to_thread(_read_snapshot, True)
        if not snap.available or snap.temp_c < self.soft_temp_c:
            return "ok"

        if snap.temp_c < self.hard_temp_c:
            logger.info(
                "GPU temp %dC >= soft limit %dC; cooling for %.1fs",
                snap.temp_c,
                self.soft_temp_c,
                self.cooldown_seconds,
            )
            if await self._sleep_or_cancel(self.cooldown_seconds, cancel_event):
                return "cancelled"
            return "throttled"

        logger.warning(
            "GPU temp %dC >= hard limit %dC; pausing up to %.1fs until <= %dC",
            snap.temp_c,
            self.hard_temp_c,
            max_wait_s,
            self.resume_temp_c,
        )
        deadline = time.monotonic() + max(0.0, float(max_wait_s))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "paused_timeout"
            step = min(1.0, max(0.05, remaining))
            if await self._sleep_or_cancel(step, cancel_event):
                return "cancelled"
            snap = await asyncio.to_thread(_read_snapshot, True)
            if not snap.available or snap.temp_c <= self.resume_temp_c:
                logger.info("GPU cooled to %dC; resuming", snap.temp_c)
                return "paused_cooled"

    def choose_safe_gpu_layers(
        self,
        requested_layers: int,
        total_layers: int = 37,
        estimated_model_mb: int = 4850,
    ) -> int:
        """Keep full offload when free VRAM suffices, otherwise scale down safely."""
        req = max(0, int(requested_layers))
        if req == 0:
            return 0

        snap = _read_snapshot(force_refresh=True)
        if not snap.available:
            return req

        required_mb = max(0, int(estimated_model_mb)) + max(0, self.llama_headroom_mb)
        if snap.vram_free_mb >= required_mb:
            return req

        budget_mb = max(0, snap.vram_free_mb - max(0, self.llama_headroom_mb))
        if budget_mb <= 0 or estimated_model_mb <= 0:
            safe_layers = 0
        else:
            ratio = budget_mb / float(estimated_model_mb)
            safe_layers = max(0, min(req, int(total_layers * ratio)))

        logger.warning(
            "Low free VRAM (%d MiB free < %d MiB required for full offload); "
            "scaling llama GPU layers from %d to %d/%d",
            snap.vram_free_mb,
            required_mb,
            req,
            safe_layers,
            total_layers,
        )
        return safe_layers

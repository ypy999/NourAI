"""Hardware optimization & monitoring for ~8 GB RAM CPU-only environments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import platform
import threading
import time
from typing import Any, Callable

import torch

from training.metrics import memory_usage_mb


@dataclass
class HardwareSnapshot:
    timestamp: float
    os_name: str
    cpu_count: int
    process_rss_mb: float | None
    cuda_available: bool
    torch_num_threads: int
    degraded_mode: bool
    recommended_preset: str
    recommended_context_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recommend_batch_size(
    *,
    available_ram_mb: float,
    model_params: int,
    context_length: int,
    max_batch_size: int = 32,
) -> int:
    """Estimate a safe CPU training/inference batch size based on RAM budget."""
    if available_ram_mb <= 0 or model_params <= 0 or context_length <= 0:
        return 1
    # Rough FP32 activation + optimizer headroom heuristic (conservative for 8GB CPU)
    model_mb = (model_params * 4.0) / (1024.0 * 1024.0)
    per_sample_mb = max(0.5, (context_length * 0.08) + (model_mb * 0.15))
    usable_mb = max(64.0, available_ram_mb * 0.35 - model_mb * 3.0)
    est = int(usable_mb // per_sample_mb)
    return max(1, min(max_batch_size, est))


class HardwareMonitor:
    """Tracks process RAM, CPU capabilities, and triggers graceful degradation when needed."""

    def __init__(self, *, soft_ram_limit_mb: float = 2048.0, hard_ram_limit_mb: float = 4096.0):
        self.soft_ram_limit_mb = float(soft_ram_limit_mb)
        self.hard_ram_limit_mb = float(hard_ram_limit_mb)

    def snapshot(self) -> HardwareSnapshot:
        rss = memory_usage_mb()
        cuda_ok = bool(torch.cuda.is_available())
        degraded = bool(rss is not None and rss >= self.soft_ram_limit_mb)
        if cuda_ok:
            preset, ctx = "base", 512
        elif degraded:
            preset, ctx = "nano", 64
        else:
            preset, ctx = "nano", 128
        return HardwareSnapshot(
            timestamp=time.time(),
            os_name=f"{platform.system()} {platform.release()}",
            cpu_count=os.cpu_count() or 1,
            process_rss_mb=round(rss, 2) if rss is not None else None,
            cuda_available=cuda_ok,
            torch_num_threads=int(torch.get_num_threads()),
            degraded_mode=degraded,
            recommended_preset=preset,
            recommended_context_length=ctx,
        )

    def can_load_model(self, estimated_model_mb: float) -> tuple[bool, str]:
        rss = memory_usage_mb()
        if rss is None:
            return True, "rss_unavailable_proceeding_cautiously"
        if rss + estimated_model_mb > self.hard_ram_limit_mb:
            return False, f"loading model ({estimated_model_mb:.1f} MiB) would exceed hard RAM budget ({self.hard_ram_limit_mb:.1f} MiB)"
        return True, "within_budget"


class LazyModelLoader:
    """Thread-safe lazy loader with memory budget checks and graceful fallback."""

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        estimated_mb: float = 25.0,
        monitor: HardwareMonitor | None = None,
    ):
        self._factory = factory
        self._estimated_mb = float(estimated_mb)
        self._monitor = monitor or HardwareMonitor()
        self._lock = threading.Lock()
        self._instance: Any | None = None
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._instance is not None

    def get(self) -> Any | None:
        if self._instance is not None:
            return self._instance
        with self._lock:
            if self._instance is not None:
                return self._instance
            ok, reason = self._monitor.can_load_model(self._estimated_mb)
            if not ok:
                self._load_error = reason
                return None
            try:
                self._instance = self._factory()
                self._load_error = None
                return self._instance
            except Exception as exc:
                self._load_error = str(exc)
                return None

    def unload(self) -> None:
        with self._lock:
            self._instance = None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "estimated_mb": self._estimated_mb,
            "load_error": self._load_error,
        }

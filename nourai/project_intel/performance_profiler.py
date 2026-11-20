"""JABRIX AI v3.7.0 — Static & Optional Isolated Performance Profiler Engine.

Performs static CPU/memory bottleneck detection via AST, supports optional
safe micro-profiling of authorized Python callables via tracemalloc/perf_counter,
and computes reproducible before/after latency & memory comparisons.
"""
from __future__ import annotations

from pathlib import Path
import time
import tracemalloc
from typing import Any, Callable

from nourai.software_intel.code_optimizer import CodeOptimizerEngine
from ._incremental_index import IncrementalProjectIndex


class PerformanceProfilerEngine:
    """Static bottleneck profiler + optional isolated callable micro-profiler + before/after comparator."""

    def __init__(
        self,
        index: IncrementalProjectIndex | None = None,
        optimizer: CodeOptimizerEngine | None = None,
    ) -> None:
        self.index = index or IncrementalProjectIndex()
        self.optimizer = optimizer or CodeOptimizerEngine()

    def static_profile(self, project_root: str | Path, *, max_files: int = 300) -> dict[str, Any]:
        opt = self.optimizer.analyze_project(project_root, max_files=max_files)
        cpu_hotspots = [
            s for s in opt["suggestions"] if s["category"] in ("bottleneck_nested_loops", "complexity")
        ]
        memory_hotspots = [
            s for s in opt["suggestions"] if s["category"] in ("string_concat_loop",)
        ]
        return {
            "project_root": opt["project_root"],
            "files_analyzed": opt["files_analyzed"],
            "functions_analyzed": opt["functions_analyzed"],
            "avg_cyclomatic_complexity": opt["avg_cyclomatic_complexity"],
            "max_cyclomatic_complexity": opt["max_cyclomatic_complexity"],
            "cpu_bottlenecks_count": len(cpu_hotspots),
            "memory_bottlenecks_count": len(memory_hotspots),
            "cpu_bottlenecks": cpu_hotspots[:50],
            "memory_bottlenecks": memory_hotspots[:50],
            "executed_untrusted_code": False,
        }

    @staticmethod
    def profile_callable(fn: Callable[[], Any], *, iterations: int = 5) -> dict[str, Any]:
        """Measure CPU wall-time and peak Python heap allocation for an authorized in-process callable."""
        iters = max(1, min(50, int(iterations)))
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            for _ in range(iters):
                fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        return {
            "iterations": iters,
            "total_ms": round(elapsed_ms, 3),
            "avg_ms": round(elapsed_ms / iters, 3),
            "peak_heap_kb": round(peak_bytes / 1024.0, 2),
        }

    @staticmethod
    def compare_before_after(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        b_ms = float(before.get("avg_ms", before.get("elapsed_ms", 0.0)))
        a_ms = float(after.get("avg_ms", after.get("elapsed_ms", 0.0)))
        b_mem = float(before.get("peak_heap_kb", before.get("rss_mb", 0.0)))
        a_mem = float(after.get("peak_heap_kb", after.get("rss_mb", 0.0)))

        speedup = round(b_ms / max(0.001, a_ms), 3) if b_ms > 0 else 1.0
        latency_delta_pct = round(((a_ms - b_ms) / max(0.001, b_ms)) * 100.0, 2) if b_ms > 0 else 0.0
        mem_delta = round(a_mem - b_mem, 2)
        improved = a_ms < b_ms and a_mem <= b_mem * 1.1 + 1.0
        return {
            "before_ms": round(b_ms, 3),
            "after_ms": round(a_ms, 3),
            "speedup_x": speedup,
            "latency_delta_pct": latency_delta_pct,
            "memory_delta": mem_delta,
            "improved": improved,
        }

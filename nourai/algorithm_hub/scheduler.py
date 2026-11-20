"""JABRIX AI v3.3.0 — Resource-Aware Scheduler, Lazy Intelligent Router, and Performance Monitor."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import Any, Callable

from nourai.monitoring.hardware import HardwareMonitor
from .core import AlgorithmResult, AlgorithmRouter, AlgorithmSpec, BaseAlgorithm

# Safe automatic fallback chain when a primary algorithm fails
ALGORITHM_FALLBACKS: dict[str, str] = {
    "dijkstra": "bfs_shortest_path",
    "jaro_winkler": "levenshtein",
    "minhash": "levenshtein",
    "kalman_filter": "exponential_smoothing",
    "textrank": "bm25",
}


class PerformanceMonitor:
    """Tracks real-time latency, throughput, cache hit rate, per-algorithm stats, and CPU/RAM usage."""

    def __init__(self, *, window_size: int = 200):
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._per_algo: dict[str, list[float]] = {}
        self.total_requests = 0
        self.total_failures = 0
        self.total_fallbacks = 0
        self.hardware = HardwareMonitor()

    def record(self, *, elapsed_ms: float, ok: bool, algo_timings: dict[str, float], fallbacks_used: int = 0) -> None:
        with self._lock:
            self.total_requests += 1
            if not ok:
                self.total_failures += 1
            self.total_fallbacks += int(fallbacks_used)
            self._latencies.append(float(elapsed_ms))
            for name, ms in algo_timings.items():
                lst = self._per_algo.setdefault(name, [])
                lst.append(float(ms))
                if len(lst) > 100:
                    del lst[:-100]

    def report(self, *, cache_stats: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            lats = sorted(self._latencies)
            n = len(lats)
            avg_ms = round(sum(lats) / n, 3) if n else 0.0
            p95_ms = round(lats[min(n - 1, int(n * 0.95))], 3) if n else 0.0
            uptime_sec = max(0.001, time.time() - self._started_at)
            throughput = round(self.total_requests / uptime_sec, 3)
            per_algo_avg = {
                k: round(sum(v) / max(1, len(v)), 3) for k, v in self._per_algo.items()
            }
            return {
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "total_fallbacks": self.total_fallbacks,
                "avg_latency_ms": avg_ms,
                "p95_latency_ms": p95_ms,
                "throughput_rps": throughput,
                "per_algorithm_avg_ms": per_algo_avg,
                "cache": cache_stats or {},
                "hardware": self.hardware.snapshot().to_dict(),
            }


class IntelligentAlgorithmRouter(AlgorithmRouter):
    """Lazy-loading, resource-aware Algorithm Router.

    Instantiates algorithms only on first use (never loading all 150 algorithms into RAM)
    and adapts algorithm selection to task_type, complexity, latency_budget_ms, and available RAM.
    """

    def __init__(self, *, monitor: HardwareMonitor | None = None):
        super().__init__()
        self._factories: dict[str, tuple[AlgorithmSpec, Callable[[], BaseAlgorithm]]] = {}
        self._lock = threading.Lock()
        self._hw = monitor or HardwareMonitor()

    def register_lazy(self, spec: AlgorithmSpec, factory: Callable[[], BaseAlgorithm]) -> None:
        with self._lock:
            self._factories[spec.name] = (spec, factory)

    def get(self, name: str) -> BaseAlgorithm | None:
        inst = self._registry.get(name)
        if inst is not None:
            return inst
        with self._lock:
            inst = self._registry.get(name)
            if inst is not None:
                return inst
            entry = self._factories.get(name)
            if entry is None:
                return None
            _, factory = entry
            inst = factory()
            self._registry[name] = inst
            return inst

    def catalog(self) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for name, (spec, _) in self._factories.items():
            d = spec.to_dict()
            d["loaded"] = name in self._registry
            seen[name] = d
        for name, algo in self._registry.items():
            d = algo.spec.to_dict()
            d["loaded"] = True
            seen[name] = d
        return list(seen.values())

    def loaded_count(self) -> int:
        return len(self._registry)

    def select_dynamic(
        self,
        task_type: str,
        payload: dict[str, Any],
        *,
        recommended: list[str],
        latency_budget_ms: float | None = None,
    ) -> tuple[list[BaseAlgorithm], dict[str, Any]]:
        """Dynamically prune or keep algorithms according to complexity, latency budget, and RAM."""
        snap = self._hw.snapshot()
        doc_count = len(payload.get("documents", [])) if isinstance(payload.get("documents"), list) else 0
        tight_latency = latency_budget_ms is not None and float(latency_budget_ms) < 25.0
        high_pressure = snap.degraded_mode or tight_latency

        chosen_names = list(recommended)
        reason = "standard_multi_algorithm_selection"

        if high_pressure and len(chosen_names) > 1:
            # Keep only the fastest primary algorithm under tight latency or high RAM pressure
            chosen_names = [chosen_names[0]]
            reason = "pruned_to_primary_for_latency_or_ram_budget"
        elif task_type == "search_rank" and doc_count > 500:
            chosen_names = ["bm25"]
            reason = "large_corpus_fast_bm25_mode"

        algos = [a for n in chosen_names if (a := self.get(n)) is not None]
        meta = {
            "selected": [a.spec.name for a in algos],
            "reason": reason,
            "degraded_mode": snap.degraded_mode,
            "process_rss_mb": snap.process_rss_mb,
            "latency_budget_ms": latency_budget_ms,
        }
        return algos, meta


class ResourceAwareScheduler:
    """Executes independent algorithms in parallel on CPU while adapting worker count to Core i3 / 8GB RAM."""

    def __init__(self, *, max_workers: int = 3, hardware: HardwareMonitor | None = None):
        self.max_workers = max(1, min(4, int(max_workers)))
        self.hardware = hardware or HardwareMonitor()

    def effective_workers(self, task_count: int) -> int:
        snap = self.hardware.snapshot()
        if snap.degraded_mode or task_count <= 1:
            return 1
        return max(1, min(self.max_workers, snap.cpu_count, task_count))

    def execute_batch(
        self,
        algorithms: list[BaseAlgorithm],
        payload: dict[str, Any],
        *,
        router: AlgorithmRouter,
    ) -> tuple[list[AlgorithmResult], bool, int]:
        """Run independent algorithms with automatic fallback on failure."""
        fallbacks_used = 0
        fb_lock = threading.Lock()

        def _run_with_fallback(algo: BaseAlgorithm) -> AlgorithmResult:
            nonlocal fallbacks_used
            t0 = time.perf_counter()
            try:
                out = algo.run(payload)
                return AlgorithmResult(
                    algorithm=algo.spec.name,
                    engine=algo.spec.engine,
                    ok=True,
                    output=out,
                    confidence=float(out.get("confidence", 0.92)) if isinstance(out, dict) else 0.92,
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                )
            except Exception as exc:
                fb_name = ALGORITHM_FALLBACKS.get(algo.spec.name)
                fb_algo = router.get(fb_name) if fb_name else None
                if fb_algo is not None:
                    try:
                        fb_out = fb_algo.run(payload)
                        with fb_lock:
                            fallbacks_used += 1
                        if isinstance(fb_out, dict):
                            fb_out = {**fb_out, "fallback_from": algo.spec.name, "primary_error": str(exc)}
                        return AlgorithmResult(
                            algorithm=fb_algo.spec.name,
                            engine=fb_algo.spec.engine,
                            ok=True,
                            output=fb_out,
                            confidence=float(fb_out.get("confidence", 0.85)) if isinstance(fb_out, dict) else 0.85,
                            duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                        )
                    except Exception:
                        pass
                return AlgorithmResult(
                    algorithm=algo.spec.name,
                    engine=algo.spec.engine,
                    ok=False,
                    error=str(exc),
                    confidence=0.0,
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                )

        workers = self.effective_workers(len(algorithms))
        parallel = workers > 1 and all(a.spec.parallel_safe for a in algorithms)
        results: list[AlgorithmResult] = []
        if parallel:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_run_with_fallback, a) for a in algorithms]
                for fut in as_completed(futs):
                    results.append(fut.result())
        else:
            for a in algorithms:
                results.append(_run_with_fallback(a))
        return results, parallel, fallbacks_used

"""Smart 10-Stage Central Algorithm Orchestrator:
Input Gateway -> Task Analyzer -> Algorithm Router -> Parallel Execution ->
Result Fusion Engine -> Verification & Conflict Resolution -> LRU Cache -> Final Response."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
import json
import threading
import time
from typing import Any


@dataclass(frozen=True)
class AlgorithmSpec:
    algo_id: int
    name: str
    engine: str
    category: str
    description: str
    parallel_safe: bool = True
    cpu_friendly: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlgorithmResult:
    algorithm: str
    engine: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseAlgorithm:
    spec: AlgorithmSpec

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class LRUCache:
    """Engine 6 (#51): Thread-safe O(1) Least-Recently-Used Cache for repeated algorithm requests."""

    def __init__(self, capacity: int = 256):
        self.capacity = max(1, int(capacity))
        self._lock = threading.Lock()
        self._store: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._store:
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }


class TaskAnalyzer:
    """Stage 2: Classifies incoming requests and estimates complexity without running unnecessary engines."""

    _ROUTING_MAP: dict[str, list[str]] = {
        "search_rank": ["bm25", "textrank"],
        "text_similarity": ["levenshtein", "jaro_winkler", "minhash"],
        "graph_search": ["bfs_shortest_path", "dijkstra"],
        "integrity_security": ["merkle_tree", "hkdf_sha256", "bloom_filter"],
        "forecast_telemetry": ["kalman_filter", "exponential_smoothing"],
        "compress_series": ["delta_encoding", "rle_encoding"],
        "parallel_process": ["mapreduce"],
        "evaluate_metrics": ["f1_score", "bootstrap_ci"],
    }

    def analyze(self, task_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        tt = (task_type or "").strip().lower()
        if tt not in self._ROUTING_MAP:
            # Infer task_type from payload keys when not explicitly specified
            if "documents" in payload and "query" in payload:
                tt = "search_rank"
            elif "text_a" in payload and "text_b" in payload:
                tt = "text_similarity"
            elif "graph" in payload and "start" in payload:
                tt = "graph_search"
            elif "series" in payload:
                tt = "forecast_telemetry"
            elif "items" in payload:
                tt = "integrity_security"
            else:
                tt = "search_rank"
        selected = list(self._ROUTING_MAP.get(tt, []))
        return {
            "task_type": tt,
            "recommended_algorithms": selected,
            "complexity": "parallel_multi_algo" if len(selected) > 1 else "single_algo",
        }


class AlgorithmRouter:
    """Stage 3: Selects only registered, CPU-compatible algorithms needed for the task."""

    def __init__(self) -> None:
        self._registry: dict[str, BaseAlgorithm] = {}

    def register(self, algo: BaseAlgorithm) -> None:
        self._registry[algo.spec.name] = algo

    def get(self, name: str) -> BaseAlgorithm | None:
        return self._registry.get(name)

    def catalog(self) -> list[dict[str, Any]]:
        return [a.spec.to_dict() for a in self._registry.values()]

    def select(self, requested: list[str]) -> list[BaseAlgorithm]:
        return [self._registry[n] for n in requested if n in self._registry]


class ResultFusionEngine:
    """Stages 6-9: Combines multi-algorithm outputs, detects conflicts, and computes ensemble confidence."""

    def fuse(self, results: list[AlgorithmResult]) -> dict[str, Any]:
        ok_results = [r for r in results if r.ok]
        errors = [{"algorithm": r.algorithm, "error": r.error} for r in results if not r.ok]
        conflicts: list[str] = []

        # Detect conflicting top rankings or high similarity disagreement when applicable
        if len(ok_results) >= 2:
            scores = [
                float(r.output["similarity"])
                for r in ok_results
                if isinstance(r.output, dict) and "similarity" in r.output
            ]
            if len(scores) >= 2 and (max(scores) - min(scores)) > 0.45:
                conflicts.append(f"similarity_divergence: spread={round(max(scores) - min(scores), 3)}")

        avg_conf = round(sum(r.confidence for r in ok_results) / max(1, len(ok_results)), 4) if ok_results else 0.0
        best = max(ok_results, key=lambda r: r.confidence) if ok_results else None

        return {
            "fused_ok": len(ok_results) > 0,
            "algorithms_succeeded": len(ok_results),
            "algorithms_failed": len(errors),
            "ensemble_confidence": avg_conf,
            "conflicts": conflicts,
            "primary_result": best.to_dict() if best else None,
            "by_algorithm": {r.algorithm: r.output for r in ok_results},
            "errors": errors,
        }


class AlgorithmOrchestrator:
    """Central 10-Stage Orchestrator with Adaptive Bounded Cache, Resource-Aware Scheduler,
    Lazy Intelligent Routing, Automatic Fallback, Result Verification, and Performance Monitoring."""

    def __init__(
        self,
        router: AlgorithmRouter,
        *,
        cache_capacity: int = 256,
        max_cache_bytes: int = 4 * 1024 * 1024,
        max_workers: int = 3,
    ):
        from .adaptive_cache import AdaptiveCache
        from .scheduler import PerformanceMonitor, ResourceAwareScheduler

        self.router = router
        self.analyzer = TaskAnalyzer()
        self.fusion = ResultFusionEngine()
        self.cache = AdaptiveCache(capacity=cache_capacity, max_memory_bytes=max_cache_bytes)
        self.scheduler = ResourceAwareScheduler(max_workers=max_workers)
        self.monitor = PerformanceMonitor()
        self.max_workers = max(1, min(4, int(max_workers)))

    @staticmethod
    def _cache_key(task_type: str, algorithms: list[str] | None, payload: dict[str, Any]) -> str:
        raw = json.dumps({"t": task_type, "a": algorithms or [], "p": payload}, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def execute(
        self,
        task_type: str,
        payload: dict[str, Any],
        *,
        algorithms: list[str] | None = None,
        use_cache: bool = True,
        latency_budget_ms: float | None = None,
        namespace: str = "algorithms",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        ckey = self._cache_key(task_type, algorithms, payload)
        if use_cache:
            cached = self.cache.get(ckey)
            if cached is not None:
                elapsed = round((time.perf_counter() - started) * 1000.0, 3)
                self.monitor.record(elapsed_ms=elapsed, ok=True, algo_timings={})
                resp = dict(cached)
                resp["cache_hit"] = True
                resp["elapsed_ms"] = elapsed
                return resp

        analysis = self.analyzer.analyze(task_type, payload)
        target_names = algorithms if algorithms else analysis["recommended_algorithms"]

        routing_meta: dict[str, Any] = {"selected": target_names, "reason": "explicit_or_default"}
        if algorithms is None and hasattr(self.router, "select_dynamic"):
            selected, routing_meta = self.router.select_dynamic(
                analysis["task_type"],
                payload,
                recommended=target_names,
                latency_budget_ms=latency_budget_ms,
            )
        else:
            selected = self.router.select(target_names)

        if not selected:
            elapsed = round((time.perf_counter() - started) * 1000.0, 3)
            self.monitor.record(elapsed_ms=elapsed, ok=False, algo_timings={})
            return {
                "ok": False,
                "error": f"no matching registered algorithms for {target_names}",
                "cache_hit": False,
                "elapsed_ms": elapsed,
            }

        results, parallel_used, fallbacks_used = self.scheduler.execute_batch(
            selected, payload, router=self.router
        )

        fused = self.fusion.fuse(results)
        elapsed = round((time.perf_counter() - started) * 1000.0, 3)
        self.monitor.record(
            elapsed_ms=elapsed,
            ok=fused["fused_ok"],
            algo_timings={r.algorithm: r.duration_ms for r in results},
            fallbacks_used=fallbacks_used,
        )

        response = {
            "ok": fused["fused_ok"],
            "task_analysis": analysis,
            "routing": routing_meta,
            "parallel_execution": parallel_used,
            "fallbacks_used": fallbacks_used,
            "algorithms_executed": [r.algorithm for r in results],
            "fusion": fused,
            "cache_hit": False,
            "elapsed_ms": elapsed,
        }
        if use_cache and response["ok"]:
            self.cache.put(ckey, response, namespace=namespace)
        return response

    def metrics(self) -> dict[str, Any]:
        return self.monitor.report(cache_stats=self.cache.stats())

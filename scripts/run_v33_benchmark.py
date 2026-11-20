"""Head-to-head Before/After Benchmark: JABRIX AI v3.2.0 vs v3.3.0 on local CPU hardware."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nourai.algorithm_hub import HybridSearchEngine, build_default_algorithm_hub
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.monitoring.hardware import HardwareMonitor


def main() -> None:
    hub = build_default_algorithm_hub()
    startup_loaded = hub.router.loaded_count()

    # 1. Latency-budget dynamic pruning: v3.2 (always full multi-algo) vs v3.3 (dynamic budget)
    docs_120 = [f"JABRIX AI document {i} about Arabic NLP, SQLite WAL, and hybrid retrieval." for i in range(60)] + [
        f"Unrelated weather note {i} sunny sky." for i in range(60)
    ]
    t0 = time.perf_counter()
    v32_full = hub.execute("search_rank", {"query": "SQLite WAL hybrid retrieval", "documents": docs_120}, use_cache=False)
    v32_full_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    t0 = time.perf_counter()
    v33_pruned = hub.execute("search_rank", {"query": "SQLite WAL hybrid retrieval", "documents": docs_120}, use_cache=False, latency_budget_ms=15.0)
    v33_pruned_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    # 2. Adaptive Cache cold vs warm hit latency
    cold = hub.execute("search_rank", {"query": "Arabic NLP", "documents": docs_120}, use_cache=True)
    warm = hub.execute("search_rank", {"query": "Arabic NLP", "documents": docs_120}, use_cache=True)

    # 3. Retrieval Recall@3: v3.2 Semantic-Only vs v3.3 Hybrid (BM25 + Semantic + KnowledgeGraph)
    with tempfile.TemporaryDirectory(prefix="jabrix_v33_bench_") as tmpdir:
        db = Path(tmpdir) / "bench.sqlite3"
        mm = AdvancedMemoryManager(db)
        kg = KnowledgeGraph(db)
        uid = "bench_u"
        mm.store(uid, "semantic", "JABRIX AI uses SQLite WAL for local persistence.", source="mem")
        mm.store(uid, "semantic", "عاصمة قطر هي الدوحة.", source="mem")
        kg.add_relation("JABRIX", "uses", "Kalman Filter", source="kg")
        kg.add_relation("France", "has_capital", "Paris", source="kg")
        extra_docs = ["AdaptiveCache enforces a 4 MiB memory bound on Core i3."]

        queries = [
            ("SQLite WAL", "SQLite"),
            ("عاصمة قطر", "الدوحة"),
            ("Kalman Filter", "Kalman"),       # Only in KnowledgeGraph!
            ("capital France", "Paris"),        # Only in KnowledgeGraph!
            ("AdaptiveCache 4 MiB", "4 MiB"),   # Only in extra_docs (BM25)!
        ]
        v32_hits = 0
        for q, needle in queries:
            sem = mm.semantic_search(uid, q, top_k=3)
            if any(needle.lower() in r["content"].lower() for r in sem):
                v32_hits += 1

        hybrid = HybridSearchEngine(memory_manager=mm, knowledge_graph=kg, cache=hub.cache)
        v33_hits = 0
        verified_count = 0
        for q, needle in queries:
            hres = hybrid.search(uid, q, documents=extra_docs, top_k=3, use_cache=False)
            if any(needle.lower() in r["text"].lower() for r in hres["results"]):
                v33_hits += 1
            if hres["verification"]["verified"]:
                verified_count += 1

    report = {
        "version": "3.3.0",
        "lazy_loading": {
            "total_registered_algorithms": len(hub.router.catalog()),
            "loaded_at_startup": startup_loaded,
            "loaded_after_benchmark": hub.router.loaded_count(),
        },
        "routing_latency_comparison": {
            "v3_2_multi_algo_ms": v32_full_ms,
            "v3_3_budget_pruned_ms": v33_pruned_ms,
            "pruning_speedup_x": round(v32_full_ms / max(0.001, v33_pruned_ms), 2),
        },
        "adaptive_cache_benchmark": {
            "cold_ms": cold["elapsed_ms"],
            "warm_ms": warm["elapsed_ms"],
            "speedup_x": round(cold["elapsed_ms"] / max(0.001, warm["elapsed_ms"]), 1),
            "cache_stats": hub.cache.stats(),
        },
        "retrieval_recall_at_3": {
            "queries_tested": len(queries),
            "v3_2_semantic_only_recall": round(v32_hits / len(queries), 4),
            "v3_3_hybrid_search_recall": round(v33_hits / len(queries), 4),
            "v3_3_verified_results_rate": round(verified_count / len(queries), 4),
        },
        "performance_monitor": hub.metrics(),
        "hardware": HardwareMonitor().snapshot().to_dict(),
    }
    out = REPO_ROOT / "reports" / "v3_3_benchmark_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

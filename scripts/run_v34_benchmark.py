"""Before/After Benchmark: JABRIX AI v3.3.0 vs v3.4.0 Advanced Reasoning Engine on Core i3 / 8GB RAM CPU."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nourai.algorithm_hub import (
    ContextManager,
    HybridSearchEngine,
    SemanticMemoryEngine,
    build_default_algorithm_hub,
)
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.monitoring.hardware import HardwareMonitor
from nourai.reasoning_v34 import AdvancedReasoningOrchestrator, SelfVerificationEngine


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="jabrix_v34_bench_") as tmpdir:
        db = Path(tmpdir) / "v34_bench.sqlite3"
        mm = AdvancedMemoryManager(db)
        kg = KnowledgeGraph(db)
        hub = build_default_algorithm_hub()
        ctx = ContextManager()
        sem = SemanticMemoryEngine(mm, context_manager=ctx, cache=hub.cache)
        hs = HybridSearchEngine(memory_manager=mm, knowledge_graph=kg, cache=hub.cache)
        orch = AdvancedReasoningOrchestrator(
            algorithm_hub=hub,
            hybrid_search=hs,
            semantic_memory=sem,
            context_manager=ctx,
        )
        uid = "bench_v34"
        docs = [
            "KV-Cache causes 2.3x faster CPU inference. Faster inference leads to lower latency.",
            "PrimaryPort is 7860. JABRIX AI persists memory in SQLite WAL.",
            "التخزين المؤقت يؤدي إلى تقليل زمن الاستجابة.",
        ]
        for d in docs:
            sem.store_fact(uid, d, source="verified_doc")

        # 1. Hallucination & Contradiction Detection Benchmark (6 test cases: 3 grounded, 3 hallucinated/contradicted)
        sv = SelfVerificationEngine()
        ev = [{"text": d} for d in docs]
        verif_cases = [
            ("PrimaryPort is 7860.", True),
            ("JABRIX AI persists memory in SQLite WAL.", True),
            ("KV-Cache causes 2.3x faster CPU inference.", True),
            ("PrimaryPort is 9999.", False),
            ("JABRIX AI requires 16 Nvidia H100 GPUs.", False),
            ("SQLite WAL is disabled in JABRIX AI and PrimaryPort is 1234.", False),
        ]
        correct = 0
        for ans, expected_ok in verif_cases:
            res = sv.verify_response(ans, ev)
            if bool(res["verified"]) == expected_ok:
                correct += 1
        hallucination_detection_accuracy = round(correct / len(verif_cases), 4)

        # 2. Full Reasoning Latency (Cold vs Warm Cache)
        cold = orch.reason(uid, "Why does KV-Cache reduce latency and what is PrimaryPort?", documents=docs, use_cache=True)
        warm = orch.reason(uid, "Why does KV-Cache reduce latency and what is PrimaryPort?", documents=docs, use_cache=True)

        # 3. Missing-evidence calibration check
        no_ev = orch.reason(uid, "What is the top speed of warp-drive alpha-99?", use_cache=False)

    report = {
        "version": "3.4.0",
        "before_v3_3_vs_after_v3_4": {
            "v3_3_claim_level_hallucination_detection": "not_implemented",
            "v3_4_hallucination_and_contradiction_accuracy": hallucination_detection_accuracy,
            "v3_3_causal_graph_extraction": "not_implemented",
            "v3_4_causal_edges_extracted": cold["causal_analysis"]["edge_count"],
            "v3_4_causal_chains_extracted": len(cold["causal_analysis"]["causal_chains"]),
            "v3_4_missing_evidence_confidence_band": no_ev["uncertainty"]["confidence_band"],
            "v3_4_missing_evidence_epistemic_uncertainty": no_ev["uncertainty"]["epistemic_uncertainty"],
        },
        "latency_and_cache": {
            "cold_reasoning_ms": cold["elapsed_ms"],
            "warm_cached_ms": warm["elapsed_ms"],
            "cache_speedup_x": round(cold["elapsed_ms"] / max(0.001, warm["elapsed_ms"]), 1),
        },
        "hardware": HardwareMonitor().snapshot().to_dict(),
    }
    out = REPO_ROOT / "reports" / "v3_4_benchmark_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

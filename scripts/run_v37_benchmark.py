"""JABRIX AI v3.7.0 — Reproducible Incremental vs Full Indexing, Cross-File & Optimization Benchmark."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nourai.monitoring.hardware import HardwareMonitor
from nourai.project_intel import ProjectIntelligenceHub


def run_benchmark() -> dict:
    hw = HardwareMonitor()
    rss_start = hw.snapshot().to_dict().get("process_rss_mb", 0.0)

    with tempfile.TemporaryDirectory(prefix="jabrix_v37_bench_") as tmp:
        tmp_path = Path(tmp)
        hub = ProjectIntelligenceHub(db_path=tmp_path / "bench_v37.db", authorized_root=REPO_ROOT)
        loaded_at_init = hub.loaded_engines_count()
        rss_after_init = hw.snapshot().to_dict().get("process_rss_mb", 0.0)

        # 1. Full vs Incremental Indexing benchmark on real JABRIX repository
        full_idx = hub.index.update(REPO_ROOT, max_files=300, force_full=True)
        inc_warm_idx = hub.index.update(REPO_ROOT, max_files=300, force_full=False)

        # Also simulate 1-file modification in an isolated 60-file copy to prove 1-file incremental reparse
        sim_proj = tmp_path / "sim_60"
        sim_proj.mkdir()
        for i in range(60):
            (sim_proj / f"mod_{i:02d}.py").write_text(
                f"def func_{i}(x):\n    return x + {i}\n" * 5, encoding="utf-8"
            )
        sim_hub = ProjectIntelligenceHub(db_path=tmp_path / "sim.db", authorized_root=sim_proj)
        sim_full = sim_hub.index.update(sim_proj, force_full=True)
        time.sleep(0.02)
        (sim_proj / "mod_00.py").write_text("def func_0(x):\n    return x + 999\n", encoding="utf-8")
        sim_inc_1file = sim_hub.index.update(sim_proj, force_full=False)

        # 2. Cross-file reasoning & deep analysis on real JABRIX repository
        t0 = time.perf_counter()
        arch = hub.architecture.analyze(REPO_ROOT, max_files=300)
        cf = hub.cross_file.analyze(REPO_ROOT, max_files=300)
        impact = hub.change_impact.analyze_impact(
            REPO_ROOT, changed_files=["nourai/software_intel/_sandbox.py"], max_files=300
        )
        quality = hub.project_quality.evaluate(REPO_ROOT, max_files=300)
        release = hub.release_intel.evaluate_release(
            REPO_ROOT, version_label="3.7.0", changed_files=["nourai/software_intel/_sandbox.py"], max_files=300
        )
        deep_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # 3. Reproducible before/after optimization comparison (O(n^2) string concat vs ''.join)
        items = [str(i) for i in range(400)]

        def slow_concat():
            s = ""
            for x in items:
                s += x
            return s

        def fast_join():
            return "".join(items)

        b_prof = hub.profiler.profile_callable(slow_concat, iterations=10)
        a_prof = hub.profiler.profile_callable(fast_join, iterations=10)
        opt_cmp = hub.profiler.compare_before_after(b_prof, a_prof)

        rss_end = hw.snapshot().to_dict().get("process_rss_mb", 0.0)
        snap = hw.snapshot().to_dict()

        report = {
            "version": "3.7.0",
            "hardware": snap,
            "lazy_loading": {
                "loaded_engines_at_init": loaded_at_init,
                "loaded_engines_after_benchmark": hub.loaded_engines_count(),
            },
            "memory_mb": {
                "rss_start": round(rss_start, 2),
                "rss_after_init": round(rss_after_init, 2),
                "rss_end": round(rss_end, 2),
                "rss_delta_total": round(rss_end - rss_start, 2),
            },
            "incremental_vs_full_indexing": {
                "real_repo_full_files": full_idx["files_total"],
                "real_repo_full_ms": full_idx["elapsed_ms"],
                "real_repo_incremental_warm_ms": inc_warm_idx["elapsed_ms"],
                "real_repo_warm_speedup_x": round(full_idx["elapsed_ms"] / max(0.01, inc_warm_idx["elapsed_ms"]), 2),
                "sim_60_full_ms": sim_full["elapsed_ms"],
                "sim_60_1file_changed_ms": sim_inc_1file["elapsed_ms"],
                "sim_60_1file_reparsed": sim_inc_1file["files_reparsed"],
                "sim_60_1file_reused": sim_inc_1file["files_reused_from_cache"],
                "sim_60_1file_speedup_x": round(sim_full["elapsed_ms"] / max(0.01, sim_inc_1file["elapsed_ms"]), 2),
            },
            "real_repo_intelligence": {
                "deep_analysis_ms": deep_ms,
                "components_count": arch["components_count"],
                "circular_components": len(arch["circular_components"]),
                "cross_file_call_edges": cf["cross_file_call_edges_count"],
                "api_route_files": len(cf["api_dependencies"]),
                "sandbox_change_downstream_files": len(impact["downstream_impacted_files"]),
                "overall_quality_score": quality["overall_quality_score"],
                "release_readiness": release["readiness"],
            },
            "optimization_before_after_proof": opt_cmp,
        }

    out_path = REPO_ROOT / "reports" / "v3_7_benchmark_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run_benchmark()

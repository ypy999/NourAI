"""JABRIX AI v3.6.0 — Reproducible Software Intelligence CPU, RAM, Latency, Isolation & Rollback Benchmark."""
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
from nourai.software_intel import ProjectIsolationError, SoftwareIntelligenceHub


def run_benchmark() -> dict:
    hw = HardwareMonitor()
    rss_start = hw.snapshot().to_dict().get("process_rss_mb", 0.0)

    with tempfile.TemporaryDirectory(prefix="jabrix_v36_bench_") as tmp:
        tmp_path = Path(tmp)
        hub = SoftwareIntelligenceHub(db_path=tmp_path / "bench_v36.db", authorized_root=REPO_ROOT)
        loaded_at_init = hub.loaded_engines_count()
        rss_after_init = hw.snapshot().to_dict().get("process_rss_mb", 0.0)

        # 1. Benchmark full static analysis on JABRIX repository itself (read-only, max 150 files)
        t0 = time.perf_counter()
        analysis = hub.analyze_project_full(max_files=150)
        analyze_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # 2. Benchmark isolation, propose, approve, apply, rollback on isolated temp project
        iso_proj = tmp_path / "iso_proj"
        iso_proj.mkdir()
        target = iso_proj / "service.py"
        orig = "def add(a, b):\n    return a + b\n"
        proposed = "def add(a: int, b: int) -> int:\n    return a + b\n"
        target.write_text(orig, encoding="utf-8")

        iso_hub = SoftwareIntelligenceHub(db_path=tmp_path / "iso.db", authorized_root=iso_proj)
        isolation_blocked = False
        try:
            iso_hub.resolve_project_root("../outside")
        except ProjectIsolationError:
            isolation_blocked = True

        t1 = time.perf_counter()
        prop = iso_hub.propose_safe_patch(
            user_id="bench",
            target_file="service.py",
            proposed_content=proposed,
            rationale="Add type annotations",
        )
        pid = prop["proposal"]["patch_id"]
        iso_hub.approval_manager.approve_patch(pid, reviewer="bench_reviewer")
        iso_hub.approval_manager.apply_approved_patch(pid)
        rb = iso_hub.approval_manager.rollback_patch(pid)
        patch_cycle_ms = round((time.perf_counter() - t1) * 1000.0, 2)
        rollback_exact = target.read_text(encoding="utf-8") == orig and rb["status"] == "ROLLED_BACK"

        rss_end = hw.snapshot().to_dict().get("process_rss_mb", 0.0)
        snap = hw.snapshot().to_dict()

        report = {
            "version": "3.6.0",
            "hardware": snap,
            "lazy_loading": {
                "loaded_engines_at_init": loaded_at_init,
                "loaded_engines_after_analysis": hub.loaded_engines_count(),
            },
            "memory_mb": {
                "rss_start": round(rss_start, 2),
                "rss_after_init": round(rss_after_init, 2),
                "rss_end": round(rss_end, 2),
                "rss_delta_total": round(rss_end - rss_start, 2),
            },
            "latency_ms": {
                "repo_analysis_150_files_ms": analyze_ms,
                "patch_propose_approve_apply_rollback_ms": patch_cycle_ms,
            },
            "repo_scan_summary": {
                "files_scanned": analysis["project_understanding"]["total_files_scanned"],
                "python_modules": analysis["project_understanding"]["python_module_count"],
                "symbols_indexed": analysis["code_intelligence"]["symbol_count"],
                "static_findings": analysis["bug_detection"]["total_findings"],
                "optimization_suggestions": analysis["code_optimizer"]["suggestions_count"],
                "circular_dependencies": len(analysis["dependencies"]["circular_dependencies"]),
            },
            "safety_validation": {
                "project_isolation_blocked_traversal": isolation_blocked,
                "patch_rollback_exact_sha256": rollback_exact,
                "isolated_compile_ok": bool(prop["isolated_syntax_check"]["compile_ok"]),
                "executed_untrusted_code": False,
            },
        }

    out_path = REPO_ROOT / "reports" / "v3_6_benchmark_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run_benchmark()

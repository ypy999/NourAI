"""JABRIX AI v3.7.0 — Project Quality & Maintainability Diagnostics Engine.

Computes maintainability indicators, complexity scoring, test quality metrics,
and security posture diagnostics across the authorized project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nourai.software_intel.bug_detection import BugDetectionEngine
from nourai.software_intel.test_generator import TestGeneratorEngine
from ._incremental_index import IncrementalProjectIndex


class ProjectQualityEngine:
    """Computes quantitative maintainability, complexity, test quality, and security scores."""

    def __init__(
        self,
        index: IncrementalProjectIndex | None = None,
        bug_engine: BugDetectionEngine | None = None,
        test_gen: TestGeneratorEngine | None = None,
    ) -> None:
        self.index = index or IncrementalProjectIndex()
        self.bug_engine = bug_engine or BugDetectionEngine()
        self.test_gen = test_gen or TestGeneratorEngine()

    def evaluate(self, project_root: str | Path, *, max_files: int = 300) -> dict[str, Any]:
        idx = self.index.update(project_root, max_files=max_files)
        records = idx["records"]
        bugs = self.bug_engine.scan_project(project_root, max_files=max_files)
        cov = self.test_gen.analyze_coverage(project_root, max_files=max_files)

        all_syms = [s for r in records.values() for s in r["symbols"]]
        fn_syms = [s for s in all_syms if s["kind"] in ("function", "method")]
        avg_cc = round(sum(s["cyclomatic_complexity"] for s in fn_syms) / max(1, len(fn_syms)), 2)
        max_cc = max((s["cyclomatic_complexity"] for s in fn_syms), default=0)
        high_cc_count = sum(1 for s in fn_syms if s["cyclomatic_complexity"] >= 10)
        doc_ratio = round(sum(1 for s in all_syms if s.get("has_docstring")) / max(1, len(all_syms)), 4)

        test_files = [r for rel, r in records.items() if rel.startswith("tests/") or Path(rel).name.startswith("test_")]
        src_files = [r for rel, r in records.items() if not (rel.startswith("tests/") or Path(rel).name.startswith("test_"))]
        total_assertions = sum(r.get("assertions", 0) for r in test_files)
        test_to_src_ratio = round(len(test_files) / max(1, len(src_files)), 3)

        sec_findings = [f for f in bugs["findings"] if str(f.get("rule_id", "")).startswith("SEC")]
        crit_findings = [f for f in bugs["findings"] if f.get("severity") == "CRITICAL"]

        # Transparent, bounded 0-100 sub-scores
        complexity_score = round(max(0.0, min(100.0, 100.0 - max(0.0, avg_cc - 3.0) * 6.0 - high_cc_count * 1.5)), 2)
        maintainability_score = round(max(0.0, min(100.0, 0.65 * complexity_score + 35.0 * doc_ratio)), 2)
        test_quality_score = round(
            max(0.0, min(100.0, 60.0 * cov["estimated_symbol_coverage"] + min(25.0, test_to_src_ratio * 40.0) + min(15.0, total_assertions * 0.25))),
            2,
        )
        security_score = round(max(0.0, min(100.0, 100.0 - len(crit_findings) * 20.0 - len(sec_findings) * 8.0)), 2)
        overall_score = round(
            0.30 * maintainability_score + 0.25 * complexity_score + 0.25 * test_quality_score + 0.20 * security_score,
            2,
        )

        return {
            "project_root": idx["project_root"],
            "overall_quality_score": overall_score,
            "maintainability": {
                "score": maintainability_score,
                "docstring_coverage_ratio": doc_ratio,
                "total_loc": sum(r["loc"] for r in records.values()),
            },
            "complexity": {
                "score": complexity_score,
                "avg_cyclomatic_complexity": avg_cc,
                "max_cyclomatic_complexity": max_cc,
                "high_complexity_functions": high_cc_count,
            },
            "test_quality": {
                "score": test_quality_score,
                "test_files": len(test_files),
                "source_files": len(src_files),
                "test_to_source_ratio": test_to_src_ratio,
                "total_assertions": total_assertions,
                "estimated_symbol_coverage": cov["estimated_symbol_coverage"],
            },
            "security": {
                "score": security_score,
                "critical_findings": len(crit_findings),
                "security_findings": len(sec_findings),
                "total_static_findings": bugs["total_findings"],
            },
        }

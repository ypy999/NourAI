"""JABRIX AI v3.7.0 — Autonomous Test Intelligence Engine.

Provides failure-driven test generation, AST-based edge-case identification,
change-impact regression test prioritization, and coverage gap detection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nourai.software_intel.test_generator import TestGeneratorEngine
from ._incremental_index import IncrementalProjectIndex
from .change_impact import ChangeImpactEngine


class AutoTestIntelligenceEngine:
    """Prioritizes regression tests, identifies edge cases, and generates failure-driven tests."""

    def __init__(
        self,
        index: IncrementalProjectIndex | None = None,
        impact_engine: ChangeImpactEngine | None = None,
        test_gen: TestGeneratorEngine | None = None,
    ) -> None:
        self.index = index or IncrementalProjectIndex()
        self.impact_engine = impact_engine or ChangeImpactEngine(index=self.index)
        self.test_gen = test_gen or TestGeneratorEngine()

    @staticmethod
    def identify_edge_cases(symbol: dict[str, Any]) -> list[str]:
        args = [a for a in symbol.get("args", []) if a != "self"]
        cases = ["empty_or_none_inputs"]
        for a in args:
            low = a.lower()
            if any(k in low for k in ("path", "file", "dir", "root")):
                cases.append(f"{a}:path_traversal_and_missing_file")
            elif any(k in low for k in ("text", "query", "str", "name", "prompt")):
                cases.append(f"{a}:empty_and_multilingual_arabic_french_unicode")
            elif any(k in low for k in ("limit", "count", "size", "max", "timeout", "budget")):
                cases.append(f"{a}:zero_negative_and_upper_bound")
            elif any(k in low for k in ("items", "list", "docs", "series", "files")):
                cases.append(f"{a}:empty_single_and_large_collection")
        return sorted(set(cases))

    def prioritize_regression_tests(
        self,
        project_root: str | Path,
        *,
        changed_files: list[str],
        max_files: int = 400,
    ) -> dict[str, Any]:
        impact = self.impact_engine.analyze_impact(project_root, changed_files=changed_files, max_files=max_files)
        idx = self.index.update(project_root, max_files=max_files)
        all_tests = sorted(
            rel for rel in idx["records"] if rel.startswith("tests/") or Path(rel).name.startswith("test_")
        )
        direct_tests = set(impact["impacted_tests"])

        prioritized: list[dict[str, Any]] = []
        for tf in all_tests:
            priority = 1 if tf in direct_tests else 2
            reason = "direct_or_transitive_dependency_match" if priority == 1 else "baseline_regression_suite"
            prioritized.append({"test_file": tf, "priority": priority, "reason": reason})
        prioritized.sort(key=lambda x: (x["priority"], x["test_file"]))

        return {
            "project_root": impact["project_root"],
            "changed_files": impact["changed_files"],
            "regression_risk_level": impact["regression_risk_level"],
            "prioritized_tests": prioritized,
            "high_priority_count": sum(1 for p in prioritized if p["priority"] == 1),
        }

    def analyze_gaps_and_edge_cases(self, project_root: str | Path, *, max_files: int = 300) -> dict[str, Any]:
        cov = self.test_gen.analyze_coverage(project_root, max_files=max_files)
        edge_recommendations = [
            {
                "symbol": s["name"],
                "file": s["file"],
                "line": s["line"],
                "suggested_edge_cases": self.identify_edge_cases(s),
            }
            for s in cov["uncovered_symbols"][:30]
        ]
        return {
            "coverage": cov,
            "edge_case_recommendations": edge_recommendations,
        }

    @staticmethod
    def generate_failure_driven_test(root_cause_report: dict[str, Any]) -> dict[str, Any]:
        top = root_cause_report.get("top_root_cause")
        if not top:
            return {"generated": False, "test_code": "", "requires_human_review": True}
        fname = top["file"]
        safe_id = Path(fname).stem.replace("-", "_")
        code = (
            f'"""Failure-driven regression test for {fname} (requires human review)."""\n'
            f"def test_regression_{safe_id}_root_cause():\n"
            f"    # Ranked root cause confidence={top['confidence']}, evidence={len(top['evidence'])} item(s)\n"
            f"    # Symbols: {top.get('symbols', [])}\n"
            f"    assert True  # Replace with failing input reproduction\n"
        )
        return {
            "generated": True,
            "target_file": fname,
            "test_code": code,
            "requires_human_review": True,
        }

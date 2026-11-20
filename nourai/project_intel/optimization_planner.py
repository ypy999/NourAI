"""JABRIX AI v3.7.0 — Resource-Aware Optimization Planner Engine.

Combines performance hotspots, architectural cycles, and cross-file fan-in risk
into prioritized optimization plans with expected benefit, risk level, and
Core i3 / 8 GB RAM execution steps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._incremental_index import IncrementalProjectIndex
from .architecture_intelligence import ArchitectureIntelligenceEngine
from .cross_file_reasoning import CrossFileReasoningEngine
from .performance_profiler import PerformanceProfilerEngine


class OptimizationPlannerEngine:
    """Produces risk-assessed, resource-aware optimization roadmaps."""

    def __init__(
        self,
        index: IncrementalProjectIndex | None = None,
        profiler: PerformanceProfilerEngine | None = None,
        arch_engine: ArchitectureIntelligenceEngine | None = None,
        cross_file: CrossFileReasoningEngine | None = None,
    ) -> None:
        self.index = index or IncrementalProjectIndex()
        self.profiler = profiler or PerformanceProfilerEngine(index=self.index)
        self.arch_engine = arch_engine or ArchitectureIntelligenceEngine(index=self.index)
        self.cross_file = cross_file or CrossFileReasoningEngine(index=self.index)

    def create_plan(self, project_root: str | Path, *, max_files: int = 300) -> dict[str, Any]:
        prof = self.profiler.static_profile(project_root, max_files=max_files)
        arch = self.arch_engine.analyze(project_root, max_files=max_files)
        cf = self.cross_file.analyze(project_root, max_files=max_files)

        fan_in_map = {item["symbol"]: item["cross_file_callers"] for item in cf["top_referenced_symbols"]}
        recommendations: list[dict[str, Any]] = []

        for item in prof["cpu_bottlenecks"][:15] + prof["memory_bottlenecks"][:15]:
            sym_key = f"{item['file']}:{item['function']}"
            callers = fan_in_map.get(sym_key, 0)
            risk = "HIGH" if callers >= 5 else ("MEDIUM" if callers >= 2 else "LOW")
            benefit = (
                "Reduce algorithmic complexity (O(n^k) -> O(n)) and CPU latency"
                if item["category"] == "bottleneck_nested_loops"
                else (
                    "Eliminate quadratic string reallocations in loop"
                    if item["category"] == "string_concat_loop"
                    else "Improve maintainability & cacheability by splitting complex branch logic"
                )
            )
            recommendations.append(
                {
                    "file": item["file"],
                    "function": item["function"],
                    "line": item["line"],
                    "category": item["category"],
                    "expected_benefit": benefit,
                    "risk_level": risk,
                    "downstream_callers": callers,
                    "requires_human_approval": True,
                }
            )

        for cyc in arch["circular_components"]:
            recommendations.append(
                {
                    "file": "architecture",
                    "function": " -> ".join(cyc),
                    "line": 0,
                    "category": "circular_architecture",
                    "expected_benefit": "Break circular component imports to enable lazy loading and clean isolation",
                    "risk_level": "HIGH",
                    "downstream_callers": len(cyc),
                    "requires_human_approval": True,
                }
            )

        # Sort LOW-risk high-impact items first for safe incremental execution
        risk_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        recommendations.sort(key=lambda r: (risk_rank.get(r["risk_level"], 2), r["file"]))

        execution_steps = [
            {"step": 1, "action": "Capture baseline benchmark via PerformanceProfilerEngine (CPU/RAM/latency)"},
            {"step": 2, "action": "Generate validated unified diff via PatchGeneratorEngine (original file untouched)"},
            {"step": 3, "action": "Verify API compatibility & tests via RegressionGuard + prioritized regression suite"},
            {"step": 4, "action": "Request human approval via ApprovalManager before applying change"},
            {"step": 5, "action": "Compare before/after benchmark; rollback immediately if latency or RAM regresses"},
        ]

        return {
            "project_root": prof["project_root"],
            "recommendations_count": len(recommendations),
            "recommendations": recommendations[:30],
            "resource_aware_plan": {
                "target_hardware": "Intel Core i3 / 8 GB RAM / CPU-only",
                "bounded_concurrency": "sequential_or_2_workers_max",
                "steps": execution_steps,
            },
        }

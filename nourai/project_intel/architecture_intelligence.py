"""JABRIX AI v3.7.0 — Global Architecture Intelligence Engine.

Performs component dependency mapping, architectural layer boundary detection,
and component-level circular architecture detection via IncrementalProjectIndex.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from nourai.software_intel.dependency_analyzer import DependencyAnalyzer
from ._incremental_index import IncrementalProjectIndex

_LAYER_ORDER: dict[str, int] = {
    "core": 1,
    "memory": 2,
    "knowledge": 2,
    "algorithm_hub": 3,
    "reasoning": 4,
    "software_intel": 4,
    "project_intel": 4,
    "multi_agent": 5,
    "api": 6,
    "ui": 7,
    "tests": 8,
    "scripts": 8,
}


class ArchitectureIntelligenceEngine:
    """Discovers global component boundaries, layer violations, and architectural cycles."""

    def __init__(self, index: IncrementalProjectIndex | None = None) -> None:
        self.index = index or IncrementalProjectIndex()

    @staticmethod
    def _component_for_module(mod: str) -> str:
        parts = [p for p in mod.split(".") if p]
        if not parts:
            return "root"
        if len(parts) >= 2 and parts[0] in ("nourai", "src", "app"):
            return f"{parts[0]}.{parts[1]}"
        return parts[0]

    @staticmethod
    def _layer_for_component(comp: str) -> str:
        low = comp.lower()
        for key in ("tests", "scripts", "ui", "api", "multi_agent", "project_intel", "software_intel", "reasoning", "algorithm_hub", "knowledge", "memory", "core"):
            if key in low:
                return key
        return "domain"

    def analyze(self, project_root: str | Path, *, max_files: int = 400) -> dict[str, Any]:
        idx = self.index.update(project_root, max_files=max_files)
        records = idx["records"]

        mod_to_comp: dict[str, str] = {}
        comp_files: dict[str, list[str]] = defaultdict(list)
        for rel, rec in records.items():
            mod = rec["module"]
            if not mod:
                continue
            comp = self._component_for_module(mod)
            mod_to_comp[mod] = comp
            comp_files[comp].append(rel)

        all_components = set(comp_files.keys())
        comp_deps: dict[str, set[str]] = {c: set() for c in all_components}
        layer_violations: list[dict[str, Any]] = []

        for rel, rec in records.items():
            src_mod = rec["module"]
            if not src_mod or src_mod not in mod_to_comp:
                continue
            src_comp = mod_to_comp[src_mod]
            src_layer = self._layer_for_component(src_comp)
            src_rank = _LAYER_ORDER.get(src_layer, 3)

            for imp in rec["imports"]:
                # Match imported module or prefix to an internal component
                tgt_comp = None
                if imp in mod_to_comp:
                    tgt_comp = mod_to_comp[imp]
                else:
                    cand = self._component_for_module(imp)
                    if cand in all_components:
                        tgt_comp = cand
                if tgt_comp and tgt_comp != src_comp:
                    comp_deps[src_comp].add(tgt_comp)
                    tgt_layer = self._layer_for_component(tgt_comp)
                    tgt_rank = _LAYER_ORDER.get(tgt_layer, 3)
                    # Lower architectural layer importing higher presentation/API layer
                    if src_layer not in ("tests", "scripts", "domain") and tgt_layer in ("api", "ui") and src_rank < tgt_rank:
                        layer_violations.append(
                            {
                                "source_component": src_comp,
                                "source_layer": src_layer,
                                "target_component": tgt_comp,
                                "target_layer": tgt_layer,
                                "file": rel,
                                "severity": "MEDIUM",
                            }
                        )

        comp_graph = {c: sorted(deps) for c, deps in comp_deps.items()}
        cycles = DependencyAnalyzer.detect_cycles(comp_graph)

        layers_map: dict[str, list[str]] = defaultdict(list)
        for c in sorted(all_components):
            layers_map[self._layer_for_component(c)].append(c)

        return {
            "project_root": idx["project_root"],
            "components_count": len(all_components),
            "component_file_counts": {c: len(fs) for c, fs in sorted(comp_files.items())},
            "component_dependency_graph": comp_graph,
            "layers": dict(layers_map),
            "layer_boundary_violations": layer_violations[:100],
            "circular_components": cycles,
            "has_circular_architecture": len(cycles) > 0,
            "incremental_stats": {
                "files_total": idx["files_total"],
                "files_reparsed": idx["files_reparsed"],
                "files_reused_from_cache": idx["files_reused_from_cache"],
                "elapsed_ms": idx["elapsed_ms"],
            },
        }

"""JABRIX AI v3.7.0 — Cross-File Reasoning Engine.

Builds cross-file symbol relationships, a multi-module function call graph,
static AST data-flow summaries, and REST API route-to-handler dependency mappings.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ._incremental_index import IncrementalProjectIndex


class CrossFileReasoningEngine:
    """Analyzes cross-file symbol graphs, call graphs, static data flow, and API dependencies."""

    def __init__(self, index: IncrementalProjectIndex | None = None) -> None:
        self.index = index or IncrementalProjectIndex()

    def analyze(self, project_root: str | Path, *, max_files: int = 400) -> dict[str, Any]:
        idx = self.index.update(project_root, max_files=max_files)
        records = idx["records"]

        symbol_defs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rel, rec in records.items():
            for sym in rec["symbols"]:
                symbol_defs[sym["name"]].append(
                    {"file": rel, "module": rec["module"], "qualname": sym["qualname"], "kind": sym["kind"], "line": sym["line"]}
                )

        call_graph: dict[str, set[str]] = defaultdict(set)
        cross_file_edges: list[dict[str, Any]] = []
        data_flows: list[dict[str, Any]] = []
        api_dependencies: list[dict[str, Any]] = []

        for rel, rec in records.items():
            data_flows.extend(rec["data_flows"])
            if rec["api_routes"]:
                api_dependencies.append(
                    {
                        "file": rel,
                        "module": rec["module"],
                        "routes": rec["api_routes"],
                        "imported_modules": rec["imports"][:30],
                    }
                )
            for call in rec["calls"]:
                callee = call["callee"]
                caller_id = f"{rel}:{call['caller']}"
                if callee in symbol_defs:
                    for target in symbol_defs[callee]:
                        target_id = f"{target['file']}:{target['qualname']}"
                        call_graph[caller_id].add(target_id)
                        if target["file"] != rel:
                            cross_file_edges.append(
                                {
                                    "source_file": rel,
                                    "caller": call["caller"],
                                    "target_file": target["file"],
                                    "callee": target["qualname"],
                                    "line": call["line"],
                                }
                            )

        # Compute most-referenced symbols (high fan-in)
        fan_in: dict[str, int] = defaultdict(int)
        for edge in cross_file_edges:
            fan_in[f"{edge['target_file']}:{edge['callee']}"] += 1
        top_referenced = sorted(fan_in.items(), key=lambda kv: -kv[1])[:25]

        return {
            "project_root": idx["project_root"],
            "defined_symbols_count": sum(len(v) for v in symbol_defs.values()),
            "cross_file_call_edges_count": len(cross_file_edges),
            "cross_file_call_edges": cross_file_edges[:300],
            "call_graph_nodes_count": len(call_graph),
            "top_referenced_symbols": [{"symbol": s, "cross_file_callers": c} for s, c in top_referenced],
            "data_flows_count": len(data_flows),
            "data_flows_sample": data_flows[:100],
            "api_dependencies": api_dependencies,
            "incremental_stats": {
                "files_total": idx["files_total"],
                "files_reparsed": idx["files_reparsed"],
                "files_reused_from_cache": idx["files_reused_from_cache"],
            },
        }

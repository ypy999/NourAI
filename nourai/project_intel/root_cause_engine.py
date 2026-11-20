"""JABRIX AI v3.7.0 — Evidence-Backed Root Cause Analysis Engine.

Correlates failure traces with cross-file dependencies, recent changes, and static
bug diagnostics to rank candidate root causes with traceable file/line evidence.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from nourai.software_intel.bug_detection import BugDetectionEngine
from ._incremental_index import IncrementalProjectIndex

_TRACE_FILE_RE = re.compile(r'File\s+"([^"]+\.py)",\s+line\s+(\d+)(?:,\s+in\s+(\w+))?')
_REL_PY_RE = re.compile(r"([A-Za-z0-9_./\\-]+\.py):(\d+)")


class RootCauseEngine:
    """Correlates runtime failures with dependency graph and static diagnostics."""

    def __init__(
        self,
        index: IncrementalProjectIndex | None = None,
        bug_engine: BugDetectionEngine | None = None,
    ) -> None:
        self.index = index or IncrementalProjectIndex()
        self.bug_engine = bug_engine or BugDetectionEngine()

    @staticmethod
    def extract_trace_locations(error_text: str) -> list[dict[str, Any]]:
        locs: list[dict[str, Any]] = []
        for m in _TRACE_FILE_RE.finditer(error_text or ""):
            locs.append({"file": m.group(1).replace("\\", "/"), "line": int(m.group(2)), "symbol": m.group(3) or ""})
        if not locs:
            for m in _REL_PY_RE.finditer(error_text or ""):
                locs.append({"file": m.group(1).replace("\\", "/"), "line": int(m.group(2)), "symbol": ""})
        return locs

    def diagnose(
        self,
        project_root: str | Path,
        *,
        error_text: str,
        changed_files: list[str] | None = None,
        max_files: int = 300,
    ) -> dict[str, Any]:
        idx = self.index.update(project_root, max_files=max_files)
        records = idx["records"]
        bugs = self.bug_engine.scan_project(project_root, max_files=max_files)
        bugs_by_file: dict[str, list[dict[str, Any]]] = {}
        for f in bugs["findings"]:
            bugs_by_file.setdefault(f["file"], []).append(f)

        trace_locs = self.extract_trace_locations(error_text)
        changed_set = set(changed_files or [])
        candidates: dict[str, dict[str, Any]] = {}

        def _ensure(rel: str) -> dict[str, Any]:
            if rel not in candidates:
                candidates[rel] = {"file": rel, "score": 0.0, "evidence": [], "symbols": set()}
            return candidates[rel]

        # 1. Direct traceback file correlation
        for pos, loc in enumerate(trace_locs):
            matched_rel = None
            for rel in records:
                if loc["file"].endswith(rel) or rel.endswith(loc["file"]):
                    matched_rel = rel
                    break
            if matched_rel:
                c = _ensure(matched_rel)
                # Deeper stack frames (later in traceback) weigh more heavily
                weight = 0.45 + 0.1 * min(3, pos)
                c["score"] += weight
                c["evidence"].append(
                    {"type": "stack_frame", "file": matched_rel, "line": loc["line"], "symbol": loc["symbol"]}
                )
                if loc["symbol"]:
                    c["symbols"].add(loc["symbol"])

                # Correlate upstream imports of the failing file
                mod_to_file = {r["module"]: f for f, r in records.items() if r["module"]}
                for imp in records[matched_rel]["imports"]:
                    dep_file = mod_to_file.get(imp)
                    if dep_file and dep_file != matched_rel:
                        dc = _ensure(dep_file)
                        dc["score"] += 0.2
                        dc["evidence"].append(
                            {"type": "upstream_dependency", "imported_by": matched_rel, "module": imp}
                        )

        # 2. Recent change correlation + static bug correlation
        for rel in list(candidates.keys()) + list(changed_set):
            if rel not in records:
                continue
            if rel in changed_set:
                c = _ensure(rel)
                c["score"] += 0.35
                c["evidence"].append({"type": "recently_changed_file", "file": rel})
            if rel in bugs_by_file:
                c = _ensure(rel)
                for bf in bugs_by_file[rel][:3]:
                    sev_w = {"CRITICAL": 0.4, "HIGH": 0.3, "MEDIUM": 0.15, "LOW": 0.05}.get(bf["severity"], 0.1)
                    c["score"] += sev_w
                    c["evidence"].append(
                        {"type": "static_bug", "rule_id": bf["rule_id"], "line": bf["line"], "message": bf["message"]}
                    )

        ranked = sorted(candidates.values(), key=lambda x: -x["score"])
        out_ranked = [
            {
                "rank": i,
                "file": item["file"],
                "confidence": round(min(0.98, item["score"]), 3),
                "symbols": sorted(item["symbols"]),
                "evidence": item["evidence"],
            }
            for i, item in enumerate(ranked[:10], start=1)
        ]
        return {
            "project_root": idx["project_root"],
            "trace_locations_found": len(trace_locs),
            "ranked_root_causes": out_ranked,
            "top_root_cause": out_ranked[0] if out_ranked else None,
        }

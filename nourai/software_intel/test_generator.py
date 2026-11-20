"""JABRIX AI v3.6.0 — Test Generator & Static Coverage Analysis Engine.

Generates unit test templates from AST signatures, generates regression test cases
from static bug diagnostics, and estimates static module/symbol test coverage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._sandbox import AuthorizedProjectScope
from .code_intelligence import CodeIntelligenceEngine


class TestGeneratorEngine:
    """Generates safe unit/regression test stubs and analyzes static test coverage."""

    __test__ = False

    def __init__(self, code_intel: CodeIntelligenceEngine | None = None) -> None:
        self.code_intel = code_intel or CodeIntelligenceEngine()

    def analyze_coverage(self, project_root: str | Path, *, max_files: int = 300) -> dict[str, Any]:
        scope = AuthorizedProjectScope(project_root, max_files=max_files)
        idx = self.code_intel.index_project(project_root, max_files=max_files)

        test_texts: list[str] = []
        test_files: list[str] = []
        for fpath in scope.iter_files(suffix=".py"):
            rel = scope.relative_posix(fpath)
            if rel.startswith("tests/") or Path(rel).name.startswith("test_"):
                test_files.append(rel)
                txt = scope.read_text_safe(rel)
                if txt:
                    test_texts.append(txt)
        combined_tests = "\n".join(test_texts)

        public_source_symbols = [
            s
            for s in idx["symbols"]
            if s["is_public"]
            and not s["file"].startswith("tests/")
            and not Path(s["file"]).name.startswith("test_")
        ]
        covered: list[dict[str, Any]] = []
        uncovered: list[dict[str, Any]] = []
        for sym in public_source_symbols:
            if sym["name"] in combined_tests:
                covered.append(sym)
            else:
                uncovered.append(sym)

        ratio = round(len(covered) / max(1, len(public_source_symbols)), 4)
        return {
            "project_root": str(scope.root),
            "test_files_count": len(test_files),
            "public_symbols_total": len(public_source_symbols),
            "covered_symbols_count": len(covered),
            "uncovered_symbols_count": len(uncovered),
            "estimated_symbol_coverage": ratio,
            "uncovered_symbols": uncovered[:100],
        }

    def generate_unit_tests(
        self,
        project_root: str | Path,
        target_file: str,
    ) -> dict[str, Any]:
        """Generate pytest unit test scaffolding for public functions in an authorized Python file."""
        scope = AuthorizedProjectScope(project_root)
        safe = scope.resolve_safe(target_file)
        rel = scope.relative_posix(safe)
        idx = self.code_intel.index_project(project_root)
        syms = [s for s in idx["symbols"] if s["file"] == rel and s["is_public"] and s["kind"] == "function"]

        mod_path = ".".join(Path(rel).with_suffix("").parts)
        lines = [
            f'"""Auto-generated unit test scaffold for {rel} (requires human review)."""',
            "import pytest",
            "",
        ]
        if syms:
            imports = ", ".join(s["name"] for s in syms)
            lines.append(f"from {mod_path} import {imports}")
            lines.append("")
            for s in syms:
                arg_list = ", ".join(f"{a}=None" for a in s.get("args", []) if a != "self")
                lines.extend(
                    [
                        f"def test_{s['name']}_basic():",
                        f"    # TODO: supply valid inputs for {s['name']}({', '.join(s.get('args', []))})",
                        f"    assert callable({s['name']})",
                        f"    # result = {s['name']}({arg_list})",
                        "",
                    ]
                )
        else:
            lines.append(f"def test_{Path(rel).stem}_module_placeholder():")
            lines.append("    assert True")
            lines.append("")

        return {
            "target_file": rel,
            "module": mod_path,
            "functions_targeted": [s["name"] for s in syms],
            "generated_code": "\n".join(lines),
            "requires_human_review": True,
        }

    @staticmethod
    def generate_regression_tests(findings: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate regression test stubs from BugDetectionEngine findings."""
        lines = [
            '"""Auto-generated regression tests from static diagnostics (requires human review)."""',
            "from pathlib import Path",
            "from nourai.software_intel.bug_detection import BugDetectionEngine",
            "",
        ]
        targeted: list[str] = []
        for idx, f in enumerate(findings[:20], start=1):
            rule = str(f.get("rule_id", "RULE")).lower()
            fpath = str(f.get("file", "unknown.py"))
            test_name = f"test_regression_{idx}_{rule}"
            targeted.append(test_name)
            lines.extend(
                [
                    f"def {test_name}():",
                    f"    engine = BugDetectionEngine()",
                    f"    src = Path({fpath!r}).read_text(encoding='utf-8')",
                    f"    hits = [x for x in engine.scan_source(src, filename={fpath!r}) if x['rule_id'] == {f.get('rule_id')!r}]",
                    f"    assert len(hits) == 0, f'Regression detected for {f.get('rule_id')}: {{hits}}'",
                    "",
                ]
            )
        return {
            "tests_generated": len(targeted),
            "test_names": targeted,
            "generated_code": "\n".join(lines),
            "requires_human_review": True,
        }

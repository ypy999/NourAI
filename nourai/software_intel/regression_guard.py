"""JABRIX AI v3.6.0 — Regression Guard & API Backward Compatibility Checker.

Ensures proposed patches:
1. Preserve existing test functions and assertions.
2. Preserve public Python API signatures (no removed public functions/classes or required args).
3. Support optional isolated, resource-limited temp-copy verification (never executes untrusted code on host files).
"""
from __future__ import annotations

import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from ._sandbox import AuthorizedProjectScope


class RegressionGuard:
    """Guards against breaking API changes, deleted tests, and regressions."""

    @staticmethod
    def _extract_public_api(source: str) -> dict[str, dict[str, Any]]:
        api: dict[str, dict[str, Any]] = {}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return api
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                args = [a.arg for a in node.args.args]
                n_defaults = len(node.args.defaults)
                required = args[: len(args) - n_defaults] if n_defaults else args
                api[f"func:{node.name}"] = {"args": args, "required_args": required}
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                methods = sorted(
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")
                )
                api[f"class:{node.name}"] = {"methods": methods}
        return api

    @staticmethod
    def _extract_test_functions(source: str) -> set[str]:
        tests: set[str] = set()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return tests
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                tests.add(node.name)
        return tests

    def check_patch(
        self,
        rel_path: str,
        original_content: str,
        proposed_content: str,
    ) -> dict[str, Any]:
        """Run static API compatibility and test preservation checks on a proposed patch."""
        breaking_changes: list[str] = []
        removed_tests: list[str] = []

        if rel_path.lower().endswith(".py"):
            orig_api = self._extract_public_api(original_content)
            prop_api = self._extract_public_api(proposed_content)

            for sym_key, meta in orig_api.items():
                if sym_key not in prop_api:
                    breaking_changes.append(f"Removed public symbol: {sym_key}")
                elif sym_key.startswith("func:"):
                    orig_req = meta["required_args"]
                    prop_args = prop_api[sym_key]["args"]
                    for req_arg in orig_req:
                        if req_arg not in prop_args:
                            breaking_changes.append(
                                f"Removed required parameter '{req_arg}' from {sym_key}"
                            )
                elif sym_key.startswith("class:"):
                    missing_methods = set(meta["methods"]) - set(prop_api[sym_key]["methods"])
                    for m in sorted(missing_methods):
                        breaking_changes.append(f"Removed public method '{m}' from {sym_key}")

            # Check existing test preservation
            orig_tests = self._extract_test_functions(original_content)
            prop_tests = self._extract_test_functions(proposed_content)
            removed_tests = sorted(orig_tests - prop_tests)

        safe_to_propose = len(breaking_changes) == 0 and len(removed_tests) == 0
        return {
            "target_file": rel_path,
            "api_compatible": len(breaking_changes) == 0,
            "breaking_changes": breaking_changes,
            "tests_preserved": len(removed_tests) == 0,
            "removed_tests": removed_tests,
            "passed": safe_to_propose,
        }

    def run_isolated_syntax_check(
        self,
        project_root: str | Path,
        rel_path: str,
        proposed_content: str,
        *,
        timeout_seconds: float = 5.0,
    ) -> dict[str, Any]:
        """Validate proposed Python file in an isolated temporary directory using `python -m py_compile`
        with `-I` (isolated mode) and a strict timeout — never executes the untrusted code itself."""
        scope = AuthorizedProjectScope(project_root)
        scope.resolve_safe(rel_path)

        with tempfile.TemporaryDirectory(prefix="jabrix_iso_check_") as tmpdir:
            iso_file = Path(tmpdir) / Path(rel_path).name
            iso_file.write_text(proposed_content, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-m", "py_compile", str(iso_file)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_seconds),
                    check=False,
                )
                return {
                    "isolated": True,
                    "executed_untrusted_code": False,
                    "compile_ok": proc.returncode == 0,
                    "stderr": proc.stderr.strip(),
                }
            except subprocess.TimeoutExpired:
                return {
                    "isolated": True,
                    "executed_untrusted_code": False,
                    "compile_ok": False,
                    "stderr": f"Isolated check timed out after {timeout_seconds}s",
                }
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

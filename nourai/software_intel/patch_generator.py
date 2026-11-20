"""JABRIX AI v3.6.0 — Patch Generator & Validation Engine.

Generates unified diffs, validates syntax via AST without executing untrusted code,
and strictly preserves original files (read-only proposal stage).
"""
from __future__ import annotations

import ast
import difflib
from pathlib import Path
from typing import Any

from ._sandbox import AuthorizedProjectScope


class PatchGeneratorEngine:
    """Produces validated unified-diff patches while leaving original files untouched."""

    def generate_patch(
        self,
        project_root: str | Path,
        target_file: str,
        proposed_content: str,
        *,
        rationale: str = "",
    ) -> dict[str, Any]:
        scope = AuthorizedProjectScope(project_root)
        safe_path = scope.resolve_safe(target_file)
        rel = scope.relative_posix(safe_path)
        original_content = scope.read_text_safe(rel)
        if original_content is None:
            raise FileNotFoundError(f"Authorized file not found or unreadable: {rel}")

        original_sha256 = scope.sha256_text(original_content)
        proposed_sha256 = scope.sha256_text(proposed_content)

        diff_lines = list(
            difflib.unified_diff(
                original_content.splitlines(keepends=True),
                proposed_content.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=3,
            )
        )
        unified_diff = "".join(diff_lines)
        validation = self.validate_patch(rel, original_content, proposed_content, unified_diff)

        return {
            "project_root": str(scope.root),
            "target_file": rel,
            "rationale": rationale,
            "original_sha256": original_sha256,
            "proposed_sha256": proposed_sha256,
            "original_content": original_content,
            "proposed_content": proposed_content,
            "unified_diff": unified_diff,
            "validation": validation,
            "original_file_modified": False,
        }

    @staticmethod
    def validate_patch(
        rel_path: str,
        original_content: str,
        proposed_content: str,
        unified_diff: str,
    ) -> dict[str, Any]:
        issues: list[str] = []
        if not unified_diff.strip():
            issues.append("Patch produces an empty diff (no changes).")

        if rel_path.lower().endswith(".py"):
            try:
                ast.parse(proposed_content, filename=rel_path)
            except SyntaxError as exc:
                issues.append(f"Proposed Python code has SyntaxError at line {exc.lineno}: {exc.msg}")

        added = sum(1 for ln in unified_diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in unified_diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "lines_added": added,
            "lines_removed": removed,
        }

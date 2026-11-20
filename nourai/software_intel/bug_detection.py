"""JABRIX AI v3.6.0 — Static Bug Detection & Error Pattern Engine.

Performs AST-based static security and correctness analysis on untrusted repository
files without ever executing host code.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re
from typing import Any

from ._sandbox import AuthorizedProjectScope

_SECRET_ASSIGN_RE = re.compile(
    r"""(?i)\b(api_key|secret_key|password|passwd|private_key|access_token)\s*=\s*["']([^"']{8,})["']"""
)


class BugDetectionEngine:
    """Static AST + pattern analyzer producing structured diagnostic reports."""

    def scan_source(self, source: str, filename: str = "<memory>") -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []

        # 1. Hardcoded secret heuristic
        for lineno, line in enumerate(source.splitlines(), start=1):
            if _SECRET_ASSIGN_RE.search(line) and "example" not in line.lower() and "placeholder" not in line.lower():
                findings.append(
                    {
                        "file": filename,
                        "line": lineno,
                        "rule_id": "SEC001_HARDCODED_SECRET",
                        "severity": "HIGH",
                        "message": "Possible hardcoded secret or credential literal detected.",
                    }
                )

        # 2. AST parse check
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as exc:
            findings.append(
                {
                    "file": filename,
                    "line": exc.lineno or 1,
                    "rule_id": "ERR001_SYNTAX_ERROR",
                    "severity": "CRITICAL",
                    "message": f"SyntaxError: {exc.msg}",
                }
            )
            return findings

        # 3. AST bug & security patterns
        for node in ast.walk(tree):
            # Dangerous dynamic execution
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in ("eval", "exec"):
                    findings.append(
                        {
                            "file": filename,
                            "line": node.lineno,
                            "rule_id": "SEC002_DYNAMIC_EXEC",
                            "severity": "CRITICAL",
                            "message": f"Unsafe dynamic code execution via `{func_name}()`.",
                        }
                    )
                elif func_name == "system" and isinstance(node.func, ast.Attribute):
                    findings.append(
                        {
                            "file": filename,
                            "line": node.lineno,
                            "rule_id": "SEC003_OS_SYSTEM",
                            "severity": "HIGH",
                            "message": "Shell command execution via `os.system()` detected.",
                        }
                    )
                elif func_name in ("run", "Popen", "call", "check_output"):
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            findings.append(
                                {
                                    "file": filename,
                                    "line": node.lineno,
                                    "rule_id": "SEC004_SUBPROCESS_SHELL",
                                    "severity": "HIGH",
                                    "message": "Subprocess call with `shell=True` risks command injection.",
                                }
                            )

            # Bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append(
                    {
                        "file": filename,
                        "line": node.lineno,
                        "rule_id": "BUG001_BARE_EXCEPT",
                        "severity": "MEDIUM",
                        "message": "Bare `except:` catches SystemExit/KeyboardInterrupt; catch specific exceptions.",
                    }
                )

            # Mutable default arguments
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append(
                            {
                                "file": filename,
                                "line": node.lineno,
                                "rule_id": "BUG002_MUTABLE_DEFAULT",
                                "severity": "MEDIUM",
                                "message": f"Function `{node.name}` uses a mutable default argument (list/dict/set).",
                            }
                        )
                # Unreachable code after return/raise inside function body
                pass

            # Unreachable code after return/raise in any statement block
            body = getattr(node, "body", None)
            if isinstance(body, list) and len(body) >= 2:
                for idx, stmt in enumerate(body[:-1]):
                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        nxt = body[idx + 1]
                        if hasattr(nxt, "lineno"):
                            findings.append(
                                {
                                    "file": filename,
                                    "line": nxt.lineno,
                                    "rule_id": "BUG003_UNREACHABLE_CODE",
                                    "severity": "LOW",
                                    "message": "Unreachable statement after return/raise.",
                                }
                            )
                            break

        return findings

    def scan_project(self, project_root: str | Path, *, max_files: int = 300) -> dict[str, Any]:
        scope = AuthorizedProjectScope(project_root, max_files=max_files)
        all_findings: list[dict[str, Any]] = []
        files_scanned = 0
        for fpath in scope.iter_files(suffix=".py"):
            rel = scope.relative_posix(fpath)
            src = scope.read_text_safe(rel)
            if src is None:
                continue
            files_scanned += 1
            all_findings.extend(self.scan_source(src, filename=rel))

        sev_counts = Counter(f["severity"] for f in all_findings)
        return {
            "project_root": str(scope.root),
            "files_scanned": files_scanned,
            "total_findings": len(all_findings),
            "severity_counts": dict(sev_counts),
            "findings": all_findings[:300],
        }

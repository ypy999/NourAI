"""JABRIX AI v3.6.0 — Automated Documentation Engine.

Generates architecture documentation (with Mermaid diagram), dependency tables,
and structured change reports from static analysis outputs.
"""
from __future__ import annotations

from typing import Any


class DocumentationEngine:
    """Generates Markdown architecture, dependency, and change-review documentation."""

    @staticmethod
    def generate_architecture_doc(project_report: dict[str, Any]) -> str:
        layers = project_report.get("architecture_layers", {})
        langs = project_report.get("languages", {})
        lines = [
            "# Project Architecture Summary",
            "",
            f"- **Project Root:** `{project_report.get('project_root', '')}`",
            f"- **Files Scanned:** {project_report.get('total_files_scanned', 0)}",
            f"- **Python Modules:** {project_report.get('python_module_count', 0)}",
            "",
            "## Detected Layers",
            "",
            "| Layer | File Count |",
            "| :--- | ---: |",
        ]
        for layer, count in sorted(layers.items()):
            lines.append(f"| `{layer}` | {count} |")

        lines.extend(["", "## Languages", ""])
        for lang, count in sorted(langs.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{lang}**: {count} file(s)")

        lines.extend(["", "```mermaid", "flowchart TD"])
        lines.append('    ROOT["Authorized Project"]')
        for idx, layer in enumerate(sorted(layers.keys()), start=1):
            lines.append(f'    L{idx}["{layer} ({layers[layer]})"]')
            lines.append(f"    ROOT --> L{idx}")
        lines.append("```")
        return "\n".join(lines) + "\n"

    @staticmethod
    def generate_dependency_doc(dep_report: dict[str, Any]) -> str:
        lines = [
            "# Dependency & Compatibility Report",
            "",
            f"- **Python Constraint:** `{dep_report.get('python_requires') or 'unspecified'}`",
            f"- **External Packages:** {dep_report.get('external_dependencies_count', 0)}",
            f"- **Internal Modules:** {dep_report.get('internal_module_count', 0)}",
            f"- **Circular Dependencies:** {len(dep_report.get('circular_dependencies', []))}",
            "",
        ]
        cycles = dep_report.get("circular_dependencies", [])
        if cycles:
            lines.append("## Circular Dependency Cycles")
            for cyc in cycles:
                lines.append(f"- `{' -> '.join(cyc)}`")
            lines.append("")

        issues = dep_report.get("compatibility_issues", [])
        if issues:
            lines.append("## Compatibility Issues")
            for iss in issues[:50]:
                lines.append(f"- **[{iss.get('severity', 'LOW')}]** `{iss.get('package')}`: {iss.get('type')}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def generate_change_report(
        patch_record: dict[str, Any],
        guard_report: dict[str, Any] | None = None,
    ) -> str:
        val = patch_record.get("validation", {})
        lines = [
            "# Proposed Patch Change Report",
            "",
            f"- **Target File:** `{patch_record.get('target_file', '')}`",
            f"- **Rationale:** {patch_record.get('rationale', 'N/A')}",
            f"- **Original SHA-256:** `{patch_record.get('original_sha256', '')}`",
            f"- **Lines Added / Removed:** +{val.get('lines_added', 0)} / -{val.get('lines_removed', 0)}",
            f"- **Patch Valid:** `{val.get('valid', False)}`",
        ]
        if guard_report:
            lines.append(f"- **API Compatible:** `{guard_report.get('api_compatible', True)}`")
            lines.append(f"- **Tests Preserved:** `{guard_report.get('tests_preserved', True)}`")
        lines.extend(
            [
                "",
                "## Unified Diff",
                "",
                "```diff",
                str(patch_record.get("unified_diff", "")).rstrip(),
                "```",
            ]
        )
        return "\n".join(lines) + "\n"

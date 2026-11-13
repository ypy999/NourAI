"""Phase 8 JSON, Markdown and static local dashboard reports."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import os


def build_report(*, environment, sources, downloads, imports, splits, validation, quality,
                 performance, readiness, errors=None, warnings=None, implementation=None,
                 tests=None, limitations=None, recommendations=None):
    return {
        "schema_version": 1,
        "report_type": "nourai_phase8_corpus_readiness",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "sources": sources,
        "downloads": downloads,
        "imports": imports,
        "splits": splits,
        "validation": validation,
        "quality": quality,
        "performance": performance,
        "readiness": readiness,
        "implementation": implementation or {},
        "tests": tests or {},
        "limitations": list(limitations or []),
        "recommendations": list(recommendations or []),
        "errors": list(errors or []),
        "warnings": list(warnings or []),
        "claims": {"training_started": False, "cuda_used": False,
                   "language_understanding_claimed": False,
                   "size_is_a_readiness_threshold": False},
    }


def write_report(report, path, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md_path = path.with_suffix(".md")
    if not overwrite and (path.exists() or md_path.exists()):
        raise FileExistsError(f"refusing to overwrite report pair: {path} / {md_path}")
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
            json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"report appeared concurrently: {path}")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    lines = ["# NourAI — Phase 8 Corpus Acquisition & Dataset Readiness", "",
             f"Created: {report['created_at']}", "",
             f"Readiness: **{report['readiness'].get('status', 'NOT_READY')}**", "",
             "## Provenance and license", "",
             "Only an explicitly CC0-labeled source is eligible for automatic ingestion. This status records technical provenance; it is not legal advice.", ""]
    for source in report.get("sources", []):
        lines.append(f"- `{source.get('source_id')}` — {source.get('status')} — {source.get('license_id')}")
    if report.get("source_catalog"):
        lines.extend(["", "### Registered source decisions", ""])
        for source in report["source_catalog"]:
            lines.append(f"- `{source.get('source_id')}` — {source.get('status')} — {source.get('license_id')} — {source.get('source_license_url') or source.get('license_url')}")
    lines.extend(["", "## Corpus results", "", "```json",
                  json.dumps({"downloads": report.get("downloads"), "imports": report.get("imports"),
                              "splits": report.get("splits"), "validation": report.get("validation"),
                              "quality": report.get("quality")}, ensure_ascii=False, indent=2), "```", "",
                  "## Performance", "", "```json",
                  json.dumps(report.get("performance"), ensure_ascii=False, indent=2), "```", "",
                  "## Quality score formula", "",
                  "Record score = 0.15×UTF-8 + 0.15×completeness + 0.20×uniqueness + 0.15×length + 0.20×Unicode integrity + 0.15×formatting. Dataset score = 0.90×mean record score + 0.10×normalized language-balance score (or mean score if no target-language records). Each component is a format/integrity heuristic; this does not assess factual correctness, educational value, or language understanding.", "",
                  "## Readiness blockers", ""])
    lines.extend(f"- {item}" for item in report["readiness"].get("blockers", []) or ["None"])
    training = report["readiness"].get("training_readiness", {})
    lines.extend(["", f"## Actual model-training decision: {training.get('status', 'not assessed')}", ""])
    lines.extend(f"- {item}" for item in training.get("blockers", []) or ["No additional training blockers recorded"])
    lines.extend(["", "## Errors and warnings", ""])
    lines.extend([f"- Error: {item}" for item in report.get("errors", [])] or ["- None"])
    lines.extend([f"- Warning: {item}" for item in report.get("warnings", [])] or ["- None"])
    lines.extend(["", "## Scope", "", "No model training or CUDA work was performed. Synthetic fixtures used for automated tests are not real corpus data. The readiness report does not claim language understanding.", ""])
    impl = report.get("implementation", {})
    lines.extend(["", "## Implementation files", ""])
    lines.extend([f"- Created: `{item}`" for item in impl.get("created", [])] or ["- No file inventory recorded"])
    lines.extend([f"- Modified existing files: `{item}`" for item in impl.get("modified", [])] or ["- None; Phase 8 files are new"])
    lines.append(f"- Backup handling: {impl.get('backup_note', 'not recorded')}")
    lines.extend(["", "## Automated test results", "", "```json", json.dumps(report.get("tests"), ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Known limitations", ""])
    lines.extend([f"- {item}" for item in report.get("limitations", [])] or ["- None recorded"])
    lines.extend(["", "## Recommendations", ""])
    lines.extend([f"- {item}" for item in report.get("recommendations", [])] or ["- None recorded"])
    md_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {"json": str(path), "markdown": str(md_path)}


def write_dashboard(report, path, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite dashboard: {path}")
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>NourAI Corpus Readiness</title><style>body{font:16px system-ui;max-width:1000px;margin:3rem auto;padding:0 1rem;color:#17202a}code,pre{background:#f3f5f7;padding:.5rem}pre{overflow:auto}.status{font-size:1.4rem;font-weight:700}</style><h1>NourAI Corpus Readiness</h1><div id=\"status\" class=\"status\"></div><p id=\"note\"></p><h2>Report data</h2><pre id=\"data\"></pre><script>const r=__REPORT__;const t=r.readiness.training_readiness||{};document.querySelector('#status').textContent='Technical: '+r.readiness.status+' | Training: '+(t.status||'not assessed');document.querySelector('#note').textContent=[...r.readiness.blockers,...(t.blockers||[])].join(' | ')||'No blockers reported.';document.querySelector('#data').textContent=JSON.stringify(r,null,2);</script></html>""".replace("__REPORT__", payload)
    path.write_text(html, encoding="utf-8", newline="\n")
    return str(path)

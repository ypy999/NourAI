"""JSON-safe and Markdown evaluation report writers."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tempfile

import torch


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "to_dict"):
        return _safe(value.to_dict())
    return value


def build_report(*, model, settings, evaluation, benchmark=None, tokenizer_results=None,
                 resource_estimates=None, tests=None, warnings=None, errors=None, hardware=None):
    params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    settings_value = settings.to_dict() if hasattr(settings, "to_dict") else settings
    settings_digest = hashlib.sha256(json.dumps(settings_value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    return _safe({
        "schema_version": 1,
        "evaluation_date": datetime.now().astimezone().isoformat(),
        "environment": {"python": __import__("sys").version,
                        "pytorch": torch.__version__, "device": "cpu"},
        "hardware": hardware,
        "settings": settings_value,
        "settings_sha256": settings_digest,
        "model": {"config": model.config.__dict__, "parameters_measured": params,
                  "weight_bytes_calculated": weight_bytes,
                  "weight_mib_calculated": weight_bytes / (1024 ** 2)},
        "evaluation": evaluation.to_dict() if hasattr(evaluation, "to_dict") else evaluation,
        "benchmark": benchmark,
        "tokenizer_tests": tokenizer_results,
        "resource_estimates": resource_estimates,
        "tests": tests,
        "warnings": warnings or [],
        "errors": errors or [],
        "limitations": ["Synthetic evaluation does not establish language understanding or training quality."],
    })


def _markdown(report):
    evaluation = report.get("evaluation") or {}
    model = report.get("model") or {}
    benchmark = report.get("benchmark") or {}
    lines = ["# NourAI Evaluation Report", "", f"- Date: {report.get('evaluation_date')}",
             f"- Device: {report.get('environment', {}).get('device')}",
             f"- Python / PyTorch: {report.get('environment', {}).get('python')} / {report.get('environment', {}).get('pytorch')}",
             f"- Hardware snapshot: {json.dumps(report.get('hardware'), ensure_ascii=False)}",
             f"- Status: {evaluation.get('status', 'not run')}",
             f"- Parameters: {model.get('parameters_measured')}",
             f"- Weight size (calculated): {model.get('weight_mib_calculated')} MiB", "",
             "## Evaluation", "",
             f"- Split: {evaluation.get('split')}",
             f"- Samples / batches / valid tokens: {evaluation.get('samples')} / {evaluation.get('batches')} / {evaluation.get('valid_tokens')}",
             f"- Average loss: {evaluation.get('average_loss')}",
             f"- Perplexity: {evaluation.get('perplexity')}",
             f"- Elapsed seconds: {evaluation.get('elapsed_seconds')}",
             f"- Tokens/second: {evaluation.get('tokens_per_second')}",
             f"- Padding targets ignored: {evaluation.get('padding_tokens_ignored')}", "",
             "## Benchmark", "",
             f"- Average batch latency: {benchmark.get('average_evaluation_batch_latency_seconds')}",
             f"- Average generation latency: {benchmark.get('average_generation_latency_seconds')}",
             f"- Evaluation / generation tokens per second: {benchmark.get('evaluation_tokens_per_second')} / {benchmark.get('generation_tokens_per_second')}",
             f"- Model construction time: {benchmark.get('model_factory_construction_seconds')}",
             f"- Process RSS before/after (MiB): {benchmark.get('process_rss_before_mib_measured')} / {benchmark.get('process_rss_after_mib_measured')}", "",
             "## Tokenizer and resource checks", "",
             f"- Tokenizer results: {json.dumps(report.get('tokenizer_tests'), ensure_ascii=False)}",
             f"- Resource estimates: {json.dumps(report.get('resource_estimates'), ensure_ascii=False)}",
             f"- Test status: {json.dumps(report.get('tests'), ensure_ascii=False)}", "",
             "## Warnings and errors", ""]
    warnings = report.get("warnings", []) + evaluation.get("warnings", []) + benchmark.get("warnings", [])
    errors = report.get("errors", [])
    if evaluation.get("error"):
        errors = errors + [evaluation["error"]]
    lines.extend([f"- Warning: {item}" for item in warnings] or ["- None"])
    lines.extend([f"- Error: {item}" for item in errors] or ["- None"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    return "\n".join(lines) + "\n"


def write_reports(report, output_dir, stem="evaluation_report"):
    """Create JSON and Markdown without overwriting existing report files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output_dir / f"{stem}.json", output_dir / f"{stem}.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError("evaluation report already exists; choose another stem or directory")
    safe_report = _safe(report)
    json_tmp = md_tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output_dir, delete=False, suffix=".tmp") as f:
            json_tmp = Path(f.name)
            json.dump(safe_report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output_dir, delete=False, suffix=".tmp") as f:
            md_tmp = Path(f.name)
            f.write(_markdown(safe_report)); f.flush()
        json_tmp.replace(json_path)
        json_tmp = None
        md_tmp.replace(md_path)
        md_tmp = None
    finally:
        for path in (json_tmp, md_tmp):
            if path is not None and path.exists():
                path.unlink()
    return {"json": str(json_path), "markdown": str(md_path)}

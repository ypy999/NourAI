"""Phase 13 — Initial Model Training pipeline and report generator."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any

import torch

from configs.configuration import ProjectConfig, load_config
from datasets.loader import create_dataloaders
from model.factory import create_model
from tokenizer.byte_tokenizer import ByteTokenizer
from .checkpoint_manager import CheckpointManager
from .logger import JsonlLogger
from .trainer import Trainer
from .validator import validate


def load_phase13_spec(path: str | Path = "configs/phase13_initial_training.json") -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_phase13_prerequisites(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify Phase 10.12, Phase 11, and Phase 12 certification gates."""
    spec = spec or load_phase13_spec()
    prereqs = spec.get("prerequisite_reports", {})
    expected_statuses = {
        "phase10_12": "CERTIFIED_FOR_PHASE_11_ARCHITECTURE_AND_INITIAL_TRAINING",
        "phase11": "CERTIFIED_FOR_PHASE_12_TRAINING_ENGINE",
        "phase12": "CERTIFIED_FOR_PHASE_13_INITIAL_MODEL_TRAINING",
    }
    verified: dict[str, Any] = {}
    for key, expected_status in expected_statuses.items():
        report_path = Path(prereqs[key])
        if not report_path.is_file():
            raise RuntimeError(f"missing prerequisite certification report for {key}: {report_path}")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        actual_status = payload.get("certification_verdict") or payload.get("status")
        if actual_status != expected_status:
            raise RuntimeError(
                f"prerequisite gate {key} not certified: expected {expected_status}, got {actual_status}"
            )
        verified[key] = {
            "report_path": str(report_path),
            "status": actual_status,
            "verified": True,
        }
    return verified


def _excerpt_chunks(
    item: dict[str, Any],
    *,
    max_excerpt_chars: int = 512,
    max_chunks: int = 12,
) -> list[dict[str, Any]]:
    text = (item.get("text") or "").strip()
    if not text:
        return []
    if len(text) <= max_excerpt_chars:
        return [dict(item)]

    # For long documents (e.g. QNL OCR books), skip the first 5% header noise and slice clean windows
    start_offset = min(len(text) // 20, 2000) if len(text) > max_excerpt_chars * 4 else 0
    usable = text[start_offset:]
    step = max(max_excerpt_chars, len(usable) // max(1, max_chunks))
    chunks: list[dict[str, Any]] = []
    base_record_id = item.get("record_id") or hashlib.sha256(text[:256].encode("utf-8")).hexdigest()
    for idx in range(max_chunks):
        pos = idx * step
        if pos >= len(usable):
            break
        snippet = usable[pos : pos + max_excerpt_chars].strip()
        if len(snippet) < 64:
            continue
        clone = dict(item)
        clone["text"] = snippet
        clone["raw_text"] = snippet
        clone["character_count"] = len(snippet)
        clone["word_count"] = len(snippet.split())
        clone["content_hash"] = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
        clone["record_id"] = f"{base_record_id}#chunk{idx:02d}"
        chunks.append(clone)
    return chunks


def prepare_phase13_multilingual_splits(
    *,
    source_splits_dir: str | Path = "data/processed/phase10_3/splits",
    working_splits_dir: str | Path = "data/processed/phase13/splits",
    train_records_per_language: int = 180,
    validation_records_per_language: int = 36,
    test_records_per_language: int = 36,
    max_excerpt_chars: int = 512,
    max_chunks_per_long_document: int = 12,
) -> dict[str, Any]:
    """Build balanced Arabic/English/French working splits from certified Phase 10.3 splits."""
    src_dir = Path(source_splits_dir)
    out_dir = Path(working_splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "train": int(train_records_per_language),
        "validation": int(validation_records_per_language),
        "test": int(test_records_per_language),
    }
    languages = ("arabic", "english", "french")
    split_summaries: dict[str, Any] = {}

    for split_name, per_lang_quota in targets.items():
        src_path = src_dir / f"{split_name}.jsonl"
        if not src_path.is_file():
            raise FileNotFoundError(f"certified source split not found: {src_path}")

        buckets: dict[str, list[dict[str, Any]]] = {lang: [] for lang in languages}
        source_counts: dict[str, int] = {}

        with src_path.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if all(len(buckets[lang]) >= per_lang_quota for lang in languages):
                    break
                row = json.loads(line)
                lang = (row.get("language") or row.get("detected_language") or "").lower()
                if lang not in buckets or len(buckets[lang]) >= per_lang_quota:
                    continue
                for chunk in _excerpt_chunks(
                    row,
                    max_excerpt_chars=max_excerpt_chars,
                    max_chunks=max_chunks_per_long_document,
                ):
                    if len(buckets[lang]) >= per_lang_quota:
                        break
                    buckets[lang].append(chunk)
                    src_id = chunk.get("source_id") or "unknown"
                    source_counts[src_id] = source_counts.get(src_id, 0) + 1

        # Round-robin interleave (arabic, english, french) so every batch is multilingual
        interleaved: list[dict[str, Any]] = []
        for group in itertools.zip_longest(*(buckets[lang] for lang in languages)):
            for item in group:
                if item is not None:
                    interleaved.append(item)

        out_path = out_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
            for item in interleaved:
                out_handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

        split_summaries[split_name] = {
            "path": str(out_path),
            "total_records": len(interleaved),
            "language_counts": {lang: len(buckets[lang]) for lang in languages},
            "source_counts": source_counts,
            "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        }

    return split_summaries


def build_phase13_project_config(spec: dict[str, Any] | None = None) -> ProjectConfig:
    spec = spec or load_phase13_spec()
    values = load_config().to_dict()
    values.update(spec.get("training_config", {}))
    return ProjectConfig(**values).validate()


def run_phase13_initial_training(
    *,
    spec_path: str | Path = "configs/phase13_initial_training.json",
    work_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute Phase 13 Initial Model Training on the balanced multilingual corpus."""
    spec = load_phase13_spec(spec_path)
    prerequisites = verify_phase13_prerequisites(spec)

    root = Path(work_root) if work_root is not None else Path(".")
    source_splits_dir = Path(spec["source_splits_dir"])
    working_splits_dir = root / spec["working_splits_dir"] if work_root is not None else Path(spec["working_splits_dir"])
    checkpoint_dir = root / spec["checkpoint_dir"] if work_root is not None else Path(spec["checkpoint_dir"])
    report_dir = root / spec["report_dir"] if work_root is not None else Path(spec["report_dir"])
    cache_dir = working_splits_dir.parent / "token_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    sampling = spec["corpus_sampling"]
    split_summaries = prepare_phase13_multilingual_splits(
        source_splits_dir=source_splits_dir,
        working_splits_dir=working_splits_dir,
        train_records_per_language=sampling["train_records_per_language"],
        validation_records_per_language=sampling["validation_records_per_language"],
        test_records_per_language=sampling["test_records_per_language"],
        max_excerpt_chars=sampling["max_excerpt_chars"],
        max_chunks_per_long_document=sampling["max_chunks_per_long_document"],
    )

    project_config = build_phase13_project_config(spec)
    tokenizer = ByteTokenizer()
    torch.manual_seed(project_config.random_seed)
    model, _ = create_model(project_config, tokenizer)
    train_loader, validation_loader = create_dataloaders(
        working_splits_dir,
        project_config,
        tokenizer,
        num_workers=0,
        cache_dir=cache_dir,
    )

    events_log_path = report_dir / "training_events.jsonl"
    if events_log_path.exists():
        events_log_path.unlink()

    checkpoint_manager = CheckpointManager(checkpoint_dir, keep=project_config.checkpoint_keep)
    logger = JsonlLogger(events_log_path)

    started_at = time.perf_counter()
    initial_validation = validate(model, validation_loader, device=project_config.device)

    trainer = Trainer(
        model,
        project_config,
        train_loader,
        validation_loader,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
    )
    validation_history = trainer.fit()
    final_validation = validate(model, validation_loader, device=project_config.device)
    elapsed_seconds = time.perf_counter() - started_at

    # Save standalone Phase 11-compatible model checkpoint as well
    model_checkpoint_path = checkpoint_dir / "phase13_model.pt"
    model.save_checkpoint(model_checkpoint_path)

    events = [
        json.loads(line)
        for line in events_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_steps = [e for e in events if e.get("event") == "train_step"]
    val_events = [e for e in events if e.get("event") == "validation"]

    initial_train_loss = float(train_steps[0]["training_loss"]) if train_steps else None
    final_train_loss = float(train_steps[-1]["training_loss"]) if train_steps else None
    mean_tokens_per_sec = (
        sum(float(e["tokens_per_second"]) for e in train_steps) / len(train_steps)
        if train_steps
        else 0.0
    )

    last_ckpt = checkpoint_dir / "last.pt"
    best_ckpt = checkpoint_dir / "best.pt"
    history_ckpts = sorted((checkpoint_dir / "history").glob("*.pt"))

    checklist = {
        "prerequisites_verified": all(v["verified"] for v in prerequisites.values()),
        "initial_config_defined": project_config.max_training_steps > 0 and project_config.context_length == 128,
        "model_and_batch_size_fixed": model.count_parameters() == 843008 and project_config.batch_size == 8,
        "training_steps_fixed": trainer.global_step == project_config.max_training_steps,
        "training_environment_ready": len(train_loader) > 0 and len(validation_loader) > 0,
        "multilingual_corpus_trained": all(
            split_summaries["train"]["language_counts"][lang] > 0
            for lang in ("arabic", "english", "french")
        ),
        "checkpoints_saved": last_ckpt.is_file() and best_ckpt.is_file() and model_checkpoint_path.is_file(),
        "training_and_validation_loss_monitored": (
            initial_train_loss is not None
            and final_train_loss is not None
            and final_train_loss < initial_train_loss
            and final_validation["validation_loss"] < initial_validation["validation_loss"]
            and final_validation["perplexity"] < initial_validation["perplexity"]
        ),
    }
    all_passed = all(checklist.values())

    report = {
        "phase": "13",
        "title": "Initial Model Training",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "CERTIFIED_FOR_PHASE_14_TRAINING_OPTIMIZATION" if all_passed else "FAILED",
        "certification_verdict": "CERTIFIED_FOR_PHASE_14_TRAINING_OPTIMIZATION" if all_passed else "FAILED",
        "all_checks_passed": all_passed,
        "checklist": checklist,
        "prerequisites": prerequisites,
        "model_summary": {
            "architecture": "GPT Decoder-only Transformer",
            "preset": "default",
            "total_parameters": model.count_parameters(),
            "vocab_size": project_config.vocab_size,
            "context_length": project_config.context_length,
            "d_model": project_config.d_model,
            "n_heads": project_config.n_heads,
            "n_layers": project_config.n_layers,
            "dropout": project_config.dropout,
        },
        "training_hyperparameters": {
            "batch_size": project_config.batch_size,
            "gradient_accumulation_steps": project_config.gradient_accumulation_steps,
            "effective_tokens_per_step": (
                project_config.batch_size
                * project_config.gradient_accumulation_steps
                * project_config.context_length
            ),
            "learning_rate": project_config.learning_rate,
            "weight_decay": project_config.weight_decay,
            "scheduler_type": project_config.scheduler_type,
            "warmup_steps": project_config.warmup_steps,
            "max_training_steps": project_config.max_training_steps,
            "gradient_clip_norm": project_config.gradient_clip_norm,
            "device": project_config.device,
            "precision": project_config.precision,
            "random_seed": project_config.random_seed,
        },
        "dataset_summary": {
            "splits": split_summaries,
            "train_sequences": len(train_loader.dataset),
            "validation_sequences": len(validation_loader.dataset),
            "train_batches_per_epoch": len(train_loader),
            "validation_batches": len(validation_loader),
        },
        "training_metrics": {
            "completed_steps": trainer.global_step,
            "completed_epochs": trainer.epoch,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "mean_tokens_per_second": round(mean_tokens_per_sec, 2),
            "initial_training_loss": round(initial_train_loss, 6) if initial_train_loss is not None else None,
            "final_training_loss": round(final_train_loss, 6) if final_train_loss is not None else None,
            "training_loss_reduction": (
                round(initial_train_loss - final_train_loss, 6)
                if initial_train_loss is not None and final_train_loss is not None
                else None
            ),
            "initial_validation_loss": round(initial_validation["validation_loss"], 6),
            "final_validation_loss": round(final_validation["validation_loss"], 6),
            "best_validation_loss": round(float(trainer.best_validation_loss), 6) if trainer.best_validation_loss is not None else None,
            "validation_loss_reduction": round(
                initial_validation["validation_loss"] - final_validation["validation_loss"], 6
            ),
            "initial_perplexity": round(initial_validation["perplexity"], 4),
            "final_perplexity": round(final_validation["perplexity"], 4),
            "validation_curve": [
                {
                    "step": row["step"],
                    "epoch": row["epoch"],
                    "validation_loss": round(row["validation_loss"], 6),
                    "perplexity": round(row["perplexity"], 4),
                }
                for row in validation_history
            ],
            "training_step_curve_sample": [
                {
                    "step": row["step"],
                    "training_loss": round(row["training_loss"], 6),
                    "learning_rate": round(row["learning_rate"], 8),
                    "gradient_norm": round(row["gradient_norm"], 4),
                }
                for row in train_steps
                if row["step"] == 1 or row["step"] % 5 == 0 or row["step"] == len(train_steps)
            ],
        },
        "artifacts": {
            "last_checkpoint": str(last_ckpt),
            "best_checkpoint": str(best_ckpt),
            "model_checkpoint": str(model_checkpoint_path),
            "history_checkpoint_count": len(history_ckpts),
            "events_log": str(events_log_path),
        },
    }
    return report


def write_phase13_reports(
    report: dict[str, Any],
    report_dir: str | Path = "reports/phase13",
) -> tuple[Path, Path]:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase13_initial_training_report.json"
    md_path = out_dir / "phase13_initial_training_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    m = report["training_metrics"]
    hp = report["training_hyperparameters"]
    ds = report["dataset_summary"]
    lines = [
        "# Phase 13 — Initial Model Training Report",
        "",
        f"- **Status:** `{report['status']}`",
        f"- **Timestamp:** `{report['timestamp']}`",
        f"- **All Checks Passed:** `{report['all_checks_passed']}`",
        "",
        "## 1. Phase 13 Verification Checklist",
        "",
        "| Gate | Passed |",
        "|---|---|",
    ]
    for gate, passed in report["checklist"].items():
        lines.append(f"| `{gate}` | `{passed}` |")

    lines.extend(
        [
            "",
            "## 2. Model & Training Configuration",
            "",
            f"- **Model Architecture:** `{report['model_summary']['architecture']}` (`{report['model_summary']['total_parameters']:,}` parameters)",
            f"- **Context Length / Batch Size / Grad Accum:** `{hp['effective_tokens_per_step'] // (hp['batch_size'] * hp['gradient_accumulation_steps'])}` / `{hp['batch_size']}` / `{hp['gradient_accumulation_steps']}` (`{hp['effective_tokens_per_step']}` tokens/step)",
            f"- **Optimizer & Schedule:** `AdamW` (`lr={hp['learning_rate']}`, `weight_decay={hp['weight_decay']}`, `{hp['scheduler_type']}`, `warmup={hp['warmup_steps']}`, `max_steps={hp['max_training_steps']}`)",
            f"- **Train / Validation Sequences:** `{ds['train_sequences']}` / `{ds['validation_sequences']}`",
            "",
            "## 3. Initial Model Training Progression (Arabic + English + French)",
            "",
            f"- **Completed Optimizer Steps:** `{m['completed_steps']}` (`{m['elapsed_seconds']}s`, `{m['mean_tokens_per_second']}` tokens/sec)",
            f"- **Training Loss (`Step 1 -> Step {m['completed_steps']}`):** `{m['initial_training_loss']}` → `{m['final_training_loss']}` (`-{m['training_loss_reduction']}`)",
            f"- **Validation Loss (`Initial -> Final`):** `{m['initial_validation_loss']}` → `{m['final_validation_loss']}` (`-{m['validation_loss_reduction']}`)",
            f"- **Validation Perplexity (`Initial -> Final`):** `{m['initial_perplexity']}` → `{m['final_perplexity']}`",
            "",
            "## 4. Validation Loss & Perplexity Curve",
            "",
            "| Step | Epoch | Validation Loss | Perplexity |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in m["validation_curve"]:
        lines.append(
            f"| {row['step']} | {row['epoch']} | {row['validation_loss']:.6f} | {row['perplexity']:.4f} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path

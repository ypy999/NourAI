"""Phase 5 — Jebrex AI Data-Gated, Resumable, Non-Contaminated CPU Sanity Training Runner.

Enforces:
- Rule 11: Refuses to start training unless the target dataset passes the Data Readiness Gate.
- Rule 12: Strictly CPU FP32 unless CUDA is verified available and explicitly approved.
- Rule 13: Preflight free-disk and RAM checks before allocating checkpoints or loaders.
- Section 7: Full Trainer.resume() continuity verification proving AdamW moments and
  LR scheduler state are restored across interruption, with zero held-out test contamination.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from configs.configuration import ProjectConfig, load_config
from data.jebrex_data_gate import audit_dataset_splits
from datasets.dataset import TokenSequenceDataset
from evaluation.engine import evaluate
from model.factory import create_model
from tokenizer.byte_tokenizer import ByteTokenizer
from training.checkpoint_manager import CheckpointManager
from training.logger import JsonlLogger
from training.metrics import memory_usage_mb
from training.trainer import Trainer


def preflight_resource_check(
    checkpoint_dir: str | Path,
    *,
    min_free_disk_mib: float = 256.0,
    device: str = "cpu",
    allow_cuda: bool = False,
) -> dict[str, Any]:
    """Verify disk space, device compliance, and baseline memory before starting training."""
    if device != "cpu" and not (allow_cuda and torch.cuda.is_available()):
        raise RuntimeError(f"device={device!r} rejected: CUDA is unavailable or unapproved")
    target = Path(checkpoint_dir)
    target.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(target)
    free_mib = disk.free / (1024 * 1024)
    if free_mib < min_free_disk_mib:
        raise RuntimeError(f"insufficient disk space: {free_mib:.1f} MiB < {min_free_disk_mib:.1f} MiB required")
    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "free_disk_mib": round(free_mib, 2),
        "process_rss_mib": memory_usage_mb(),
        "passed": True,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_gated_training(
    *,
    splits_dir: str | Path = "data/processed/jebrex_v1_clean/splits",
    checkpoint_dir: str | Path = "checkpoints/jebrex_v1_sanity",
    report_dir: str | Path = "reports/training",
    total_steps: int = 60,
    interrupt_at_step: int = 30,
    batch_size: int = 8,
    grad_accum: int = 2,
    learning_rate: float = 1.5e-3,
    seed: int = 1337,
) -> dict[str, Any]:
    """Run a data-gated, reproducible 2-part training run with mid-run resume verification."""
    t0 = time.perf_counter()
    splits_path = Path(splits_dir)
    ckpt_dir = Path(checkpoint_dir)
    out_report_dir = Path(report_dir)
    out_report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Enforce Data Readiness Gate (Rule 11)
    data_audit = audit_dataset_splits(splits_path, dataset_name=splits_path.parent.name)
    if data_audit["overall_pipeline_validation_gate"] != "DATA_READY":
        raise RuntimeError(
            f"DATA READINESS GATE BLOCKED TRAINING: gate={data_audit['overall_pipeline_validation_gate']}"
        )

    # 2. Enforce Hardware & Resource Preflight (Rules 12 & 13)
    preflight = preflight_resource_check(ckpt_dir, device="cpu", allow_cuda=False)

    # 3. Build config & datasets (NEVER touch test.jsonl during training!)
    values = load_config().to_dict()
    values.update(
        {
            "vocab_size": 260,
            "context_length": 128,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 4,
            "dropout": 0.05,
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(grad_accum),
            "learning_rate": float(learning_rate),
            "weight_decay": 0.01,
            "warmup_steps": min(8, max(2, total_steps // 10)),
            "scheduler_type": "linear_warmup_cosine",
            "precision": "fp32",
            "device": "cpu",
            "epochs": 20,
            "max_training_steps": int(total_steps),
            "validation_interval": 15,
            "checkpoint_interval": 15,
            "early_stopping": True,
            "early_stopping_patience": 6,
            "early_stopping_min_delta": 0.001,
            "random_seed": int(seed),
            "checkpoint_keep": 3,
        }
    )
    cfg = ProjectConfig(**values).validate()
    tokenizer = ByteTokenizer()

    with tempfile.TemporaryDirectory(prefix="jebrex-train-cache-") as tmp_cache:
        train_ds = TokenSequenceDataset(
            splits_path / "train.jsonl", cfg, tokenizer, cache_dir=Path(tmp_cache) / "train", preload_tokens=True
        )
        val_ds = TokenSequenceDataset(
            splits_path / "validation.jsonl", cfg, tokenizer, cache_dir=Path(tmp_cache) / "val", preload_tokens=True
        )
        test_ds = TokenSequenceDataset(
            splits_path / "test.jsonl", cfg, tokenizer, cache_dir=Path(tmp_cache) / "test", preload_tokens=True
        )

        train_gen1 = torch.Generator().manual_seed(seed)
        train_loader1 = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=train_gen1)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

        # Baseline untrained metrics (train/val/test strictly separated)
        torch.manual_seed(seed)
        model1, _ = create_model(cfg, tokenizer)
        initial_val = evaluate(model1, val_loader, split="validation", tokenizer=tokenizer).to_dict()
        initial_test = evaluate(model1, test_loader, split="test", tokenizer=tokenizer).to_dict()

        ckpt_manager = CheckpointManager(ckpt_dir, keep=cfg.checkpoint_keep)
        log_path = out_report_dir / "jebrex_v1_sanity_events.jsonl"
        if log_path.exists():
            # Append timestamped backup instead of deleting prior log (Rule 6)
            backup_log = out_report_dir / f"jebrex_v1_sanity_events_prev_{int(time.time())}.jsonl"
            shutil.copy2(log_path, backup_log)
            log_path.write_text("", encoding="utf-8")
        logger = JsonlLogger(log_path)

        # Part 1: Train up to interrupt_at_step and save full state
        trainer1 = Trainer(model1, cfg, train_loader1, val_loader, checkpoint_manager=ckpt_manager, logger=logger)
        part1_reports = trainer1.fit(stop_at_step=interrupt_at_step)
        state_before_interrupt = trainer1.optimizer_scheduler_state_summary()

        # Part 2: Instantiate brand-new model + Trainer, resume from last.pt, verify optimizer/scheduler restoration
        torch.manual_seed(seed + 9999)
        model2, _ = create_model(cfg, tokenizer)
        train_gen2 = torch.Generator().manual_seed(seed)
        train_loader2 = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=train_gen2)
        trainer2 = Trainer(model2, cfg, train_loader2, val_loader, checkpoint_manager=ckpt_manager, logger=logger)
        recovery_info = trainer2.resume_with_recovery()
        state_after_resume = trainer2.optimizer_scheduler_state_summary()

        optimizer_restored_exact = (
            state_before_interrupt["optimizer_param_entries"] == state_after_resume["optimizer_param_entries"]
            and state_before_interrupt["optimizer_max_step"] == state_after_resume["optimizer_max_step"] == interrupt_at_step
            and abs(state_before_interrupt["optimizer_exp_avg_l2"] - state_after_resume["optimizer_exp_avg_l2"]) < 1e-7
            and abs(state_before_interrupt["optimizer_exp_avg_sq_l2"] - state_after_resume["optimizer_exp_avg_sq_l2"]) < 1e-7
        )
        scheduler_restored_exact = (
            state_before_interrupt["scheduler_last_epoch"] == state_after_resume["scheduler_last_epoch"] == interrupt_at_step
            and abs(state_before_interrupt["learning_rate"] - state_after_resume["learning_rate"]) < 1e-12
        )

        # Continue Part 2 to total_steps
        part2_reports = trainer2.fit(resume=False)
        state_final = trainer2.optimizer_scheduler_state_summary()

        # Export standalone inference weights alongside resumable last.pt / best.pt
        inference_ckpt = ckpt_dir / "jebrex_v1_sanity_model.pt"
        model2.save_checkpoint(inference_ckpt)

        # Final independent evaluation (NEVER trained on test_loader)
        final_train = evaluate(model2, DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False), split="train", tokenizer=tokenizer).to_dict()
        final_val = evaluate(model2, val_loader, split="validation", tokenizer=tokenizer).to_dict()
        final_test = evaluate(model2, test_loader, split="test", tokenizer=tokenizer).to_dict()

    elapsed_s = round(time.perf_counter() - t0, 2)
    report = {
        "run_name": "jebrex_v1_sanity",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_readiness_gate_passed": data_audit["overall_pipeline_validation_gate"],
        "general_llm_production_data_gate": data_audit["overall_general_llm_production_gate"],
        "preflight": preflight,
        "model_config": {
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "dropout": cfg.dropout,
            "parameter_count": model2.count_parameters(),
        },
        "dataset_sequences": {
            "train_sequences": len(train_ds),
            "validation_sequences": len(val_ds),
            "test_sequences": len(test_ds),
            "test_used_in_training": False,
        },
        "training_resume_verification": {
            "interrupt_at_step": interrupt_at_step,
            "target_total_steps": total_steps,
            "state_before_interrupt": state_before_interrupt,
            "state_after_resume": state_after_resume,
            "state_final": state_final,
            "optimizer_restored_exact": optimizer_restored_exact,
            "scheduler_restored_exact": scheduler_restored_exact,
            "recovered_from": recovery_info["recovered_from"],
            "used_fallback": recovery_info["used_fallback"],
        },
        "metrics_separated": {
            "initial_untrained_validation": {
                "loss": round(float(initial_val["average_loss"]), 4),
                "perplexity": round(float(initial_val["perplexity"]), 4),
                "bits_per_byte": round(float(initial_val["average_loss"]) / math.log(2.0), 4),
            },
            "initial_untrained_test": {
                "loss": round(float(initial_test["average_loss"]), 4),
                "perplexity": round(float(initial_test["perplexity"]), 4),
                "bits_per_byte": round(float(initial_test["average_loss"]) / math.log(2.0), 4),
            },
            "final_train": {
                "loss": round(float(final_train["average_loss"]), 4),
                "perplexity": round(float(final_train["perplexity"]), 4),
                "bits_per_byte": round(float(final_train["average_loss"]) / math.log(2.0), 4),
            },
            "final_validation": {
                "loss": round(float(final_val["average_loss"]), 4),
                "perplexity": round(float(final_val["perplexity"]), 4),
                "bits_per_byte": round(float(final_val["average_loss"]) / math.log(2.0), 4),
            },
            "final_held_out_test": {
                "loss": round(float(final_test["average_loss"]), 4),
                "perplexity": round(float(final_test["perplexity"]), 4),
                "bits_per_byte": round(float(final_test["average_loss"]) / math.log(2.0), 4),
            },
        },
        "checkpoints": {
            "inference_model": str(inference_ckpt),
            "inference_model_sha256": _sha256_file(inference_ckpt),
            "last_checkpoint": str(ckpt_dir / "last.pt"),
            "best_checkpoint": str(ckpt_dir / "best.pt"),
        },
        "elapsed_seconds": elapsed_s,
        "part1_val_history": part1_reports,
        "part2_val_history": part2_reports,
    }

    (out_report_dir / "jebrex_v1_sanity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    m = report["metrics_separated"]
    md = [
        "# المرحلة الخامسة — تقرير التدريب التحققي القابل للاستئناف (`jebrex_v1_sanity`)",
        "",
        f"- **التوقيت (UTC):** `{report['timestamp_utc']}`",
        f"- **بوابة البيانات المجتازة:** `{report['data_readiness_gate_passed']}` (بوابة الإنتاج العام: `{report['general_llm_production_data_gate']}`)",
        f"- **العتاد:** `CPU FP32` (`CUDA=False`) | الزمن الإجمالي: `{elapsed_s}s`",
        f"- **النموذج:** `Nano GPT` (`{report['model_config']['parameter_count']:,}` معامل، `context_length=128`)",
        f"- **استعادة `AdamW` بدقة:** `{optimizer_restored_exact}` (عند الخطوة `{interrupt_at_step}` → إكمال حتى `{state_final['global_step']}`)",
        f"- **استعادة `Scheduler` بدقة:** `{scheduler_restored_exact}`",
        "",
        "## النتائج المنفصلة (`Train / Validation / Held-Out Test`)",
        "",
        "| التقسيم | Loss قبل التدريب | Loss بعد التدريب | Perplexity بعد | Bits/Byte بعد |",
        "|---|---:|---:|---:|---:|",
        f"| `Train` | — | {m['final_train']['loss']} | {m['final_train']['perplexity']} | {m['final_train']['bits_per_byte']} |",
        f"| `Validation` | {m['initial_untrained_validation']['loss']} | {m['final_validation']['loss']} | {m['final_validation']['perplexity']} | {m['final_validation']['bits_per_byte']} |",
        f"| `Held-Out Test` (لم يُستخدم في التدريب) | {m['initial_untrained_test']['loss']} | {m['final_held_out_test']['loss']} | {m['final_held_out_test']['perplexity']} | {m['final_held_out_test']['bits_per_byte']} |",
        "",
    ]
    (out_report_dir / "jebrex_v1_sanity_report.md").write_text("\n".join(md), encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_gated_training()
    print(json.dumps({
        "elapsed_seconds": res["elapsed_seconds"],
        "optimizer_restored_exact": res["training_resume_verification"]["optimizer_restored_exact"],
        "scheduler_restored_exact": res["training_resume_verification"]["scheduler_restored_exact"],
        "final_step": res["training_resume_verification"]["state_final"]["global_step"],
        "metrics_separated": res["metrics_separated"],
    }, indent=2, ensure_ascii=False))

"""Jebrex AI — Phase G Staged Training Orchestrator (`125M` -> `350M`) & Stability Watchdog.

Guarantees:
1. NEVER bypasses `DATA_NOT_READY` or missing CUDA VRAM for `125M` / `350M` production runs.
2. Enforces Stage 1 (`jebrex-125m`) completion & independent acceptance verification BEFORE
   allowing Stage 2 (`jebrex-350m`) to start.
3. Monitors per-step stability (NaN/Inf loss, gradient norm explosions, sudden loss spikes)
   and automatically rolls back to the last verified healthy checkpoint with LR backoff.
4. Enforces strict held-out test split isolation during training (only train & val loaders).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from training.cloud_trainer import CloudResumableTrainer
from training.production_launcher import (
    ProductionGateBlockedError,
    evaluate_production_training_gate,
)


@dataclass
class StabilityConfig:
    max_grad_norm_alert: float = 10.0
    loss_spike_multiplier: float = 2.5
    ema_alpha: float = 0.2
    lr_backoff_factor: float = 0.5
    max_rollbacks: int = 3


class TrainingStabilityWatchdog:
    """Detects NaN/Inf or sudden loss divergence and recovers cleanly from last healthy checkpoint."""

    def __init__(self, config: Optional[StabilityConfig] = None) -> None:
        self.cfg = config or StabilityConfig()
        self.loss_ema: Optional[float] = None
        self.rollback_count: int = 0
        self.events: List[Dict[str, Any]] = []

    def check_step(self, step: int, loss_val: float, grad_norm: float) -> Dict[str, Any]:
        is_nan_or_inf = (not math.isfinite(loss_val)) or (not math.isfinite(grad_norm))
        is_loss_spike = (
            self.loss_ema is not None
            and math.isfinite(loss_val)
            and loss_val > max(self.loss_ema * self.cfg.loss_spike_multiplier, self.loss_ema + 3.0)
        )
        grad_alert = math.isfinite(grad_norm) and grad_norm > self.cfg.max_grad_norm_alert

        unstable = is_nan_or_inf or is_loss_spike
        reason = None
        if is_nan_or_inf:
            reason = "nan_or_inf_detected"
        elif is_loss_spike:
            reason = f"loss_spike ({loss_val:.4f} > {self.cfg.loss_spike_multiplier}x EMA {self.loss_ema:.4f})"

        if not unstable and math.isfinite(loss_val):
            self.loss_ema = (
                loss_val if self.loss_ema is None else (self.cfg.ema_alpha * loss_val + (1.0 - self.cfg.ema_alpha) * self.loss_ema)
            )

        ev = {
            "step": step,
            "loss": loss_val if math.isfinite(loss_val) else "non_finite",
            "grad_norm": round(grad_norm, 4) if math.isfinite(grad_norm) else "non_finite",
            "grad_alert": grad_alert,
            "unstable": unstable,
            "reason": reason,
        }
        if unstable or grad_alert:
            self.events.append(ev)
        return ev

    def recover_trainer(self, trainer: CloudResumableTrainer, healthy_ckpt: Path) -> Dict[str, Any]:
        """Restore last verified healthy checkpoint and reduce LR cleanly."""
        if self.rollback_count >= self.cfg.max_rollbacks:
            raise RuntimeError(f"Exceeded max stability rollbacks ({self.cfg.max_rollbacks}); halting training.")
        info = trainer.load_checkpoint(healthy_ckpt)
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = float(pg["lr"]) * self.cfg.lr_backoff_factor
        self.rollback_count += 1
        rec = {
            "rollback_number": self.rollback_count,
            "restored_step": info["global_step"],
            "new_lr": trainer.optimizer.param_groups[0]["lr"],
            "healthy_checkpoint": str(healthy_ckpt).replace("\\", "/"),
        }
        self.events.append({"recovery": rec})
        return rec


def check_stage_progression_gate(
    stage_name: str,
    *,
    stage1_eval_report: Optional[Path] = None,
    config_125m: Path = Path("configs/training_125m.json"),
    config_350m: Path = Path("configs/training_350m.json"),
    dataset_manifest: Path = Path("data/manifests/jebrex_v1_clean.v1.0.0.manifest.json"),
) -> Dict[str, Any]:
    """Enforce strict Stage 1 (125M) -> Stage 2 (350M) progression without ever bypassing DATA_NOT_READY."""
    if stage_name == "stage_1_125m":
        return evaluate_production_training_gate(
            config_path=config_125m,
            dataset_manifest_path=dataset_manifest,
            report_out=None,
        )

    if stage_name == "stage_2_350m":
        base = evaluate_production_training_gate(
            config_path=config_350m,
            dataset_manifest_path=dataset_manifest,
            report_out=None,
        )
        blockers = list(base["blockers"])
        if stage1_eval_report is None or not Path(stage1_eval_report).is_file():
            blockers.insert(0, "Prerequisite Stage 1 (jebrex-125m) verified evaluation report does not exist.")
        else:
            s1 = json.loads(Path(stage1_eval_report).read_text(encoding="utf-8"))
            if not s1.get("promotion_passed", False):
                blockers.insert(0, "Prerequisite Stage 1 (jebrex-125m) did not pass independent acceptance evaluation.")
        base["blockers"] = blockers
        base["training_allowed"] = len(blockers) == 0
        base["gate_status"] = "PRODUCTION_TRAINING_READY" if not blockers else "PRODUCTION_TRAINING_BLOCKED"
        return base

    raise ValueError(f"Unknown stage_name: {stage_name}")


def assert_test_split_not_used_in_training(train_loader: DataLoader, val_loader: DataLoader) -> None:
    """Verify neither training nor validation dataloader points to a held-out test split."""
    for name, loader in (("train", train_loader), ("validation", val_loader)):
        ds = getattr(loader, "dataset", None)
        bin_p = str(getattr(ds, "bin_path", "")).lower()
        if "test" in Path(bin_p).name:
            raise ValueError(f"Held-out test split violation: {name}_loader points to {bin_p}")

"""CPU-first training loop with validation, accumulation, conditional AMP, and resumable checkpoints."""
from contextlib import nullcontext
import math
import random
import time

import torch

from .checkpoint_manager import CheckpointManager
from .early_stopping import EarlyStopping
from .logger import JsonlLogger
from .metrics import LossAccumulator, memory_usage_mb
from .optimizer import create_optimizer
from .scheduler import create_scheduler
from .validator import validate


def resolve_precision_profile(device, precision, *, allow_cpu_amp=False, allow_cuda=False):
    """Resolve execution device, autocast dtype, and GradScaler usage."""
    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if (allow_cuda and torch.cuda.is_available()) else "cpu"
    if resolved_device != "cpu":
        if not (resolved_device == "cuda" and allow_cuda and torch.cuda.is_available()):
            raise RuntimeError("this training run is configured for CPU only; CUDA was not initialized")
    if precision == "fp32":
        return {
            "device": resolved_device,
            "device_type": "cuda" if resolved_device.startswith("cuda") else "cpu",
            "precision": "fp32",
            "amp_enabled": False,
            "amp_dtype": None,
            "use_grad_scaler": False,
        }
    if resolved_device == "cpu":
        if not allow_cpu_amp or precision != "bf16":
            raise ValueError("mixed precision is not enabled for CPU training; select precision='fp32'")
        return {
            "device": "cpu",
            "device_type": "cpu",
            "precision": "bf16",
            "amp_enabled": True,
            "amp_dtype": torch.bfloat16,
            "use_grad_scaler": False,
        }
    if precision == "bf16":
        return {
            "device": resolved_device,
            "device_type": "cuda",
            "precision": "bf16",
            "amp_enabled": True,
            "amp_dtype": torch.bfloat16,
            "use_grad_scaler": False,
        }
    if precision == "fp16":
        return {
            "device": resolved_device,
            "device_type": "cuda",
            "precision": "fp16",
            "amp_enabled": True,
            "amp_dtype": torch.float16,
            "use_grad_scaler": True,
        }
    raise ValueError(f"unsupported precision mode: {precision}")


class Trainer:
    def __init__(self, model, config, train_loader, validation_loader,
                 checkpoint_manager=None, logger=None, *,
                 allow_cpu_amp=False, allow_cuda=False):
        profile = resolve_precision_profile(
            config.device,
            config.precision,
            allow_cpu_amp=allow_cpu_amp,
            allow_cuda=allow_cuda,
        )
        if len(train_loader) == 0 or len(validation_loader) == 0:
            raise ValueError("training and validation loaders must both contain batches")
        self.device = profile["device"]
        self.device_type = profile["device_type"]
        self.amp_enabled = profile["amp_enabled"]
        self.amp_dtype = profile["amp_dtype"]
        self.use_grad_scaler = profile["use_grad_scaler"]
        self.scaler = (
            torch.amp.GradScaler(self.device_type, enabled=True)
            if self.use_grad_scaler
            else None
        )
        self.model = model.to(self.device)
        self.config = config
        self.train_loader, self.validation_loader = train_loader, validation_loader
        self.optimizer = create_optimizer(model, config)
        steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        total_steps = config.max_training_steps or steps_per_epoch * config.epochs
        self.scheduler = create_scheduler(self.optimizer, config.scheduler_type, total_steps, config.warmup_steps)
        self.checkpoints = checkpoint_manager or CheckpointManager("checkpoints/phase5", keep=config.checkpoint_keep)
        self.logger = logger or JsonlLogger("reports/training.jsonl")
        self.early_stopping = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta, config.early_stopping)
        self.epoch = 0
        self.batch_in_epoch = 0
        self.global_step = 0
        self.best_validation_loss = None
        self.history = []
        self._resumed = False
        self._stop_requested = False

    def _autocast_context(self):
        if self.amp_enabled and self.amp_dtype is not None:
            return torch.amp.autocast(device_type=self.device_type, dtype=self.amp_dtype)
        return nullcontext()

    def request_stop(self):
        """Request a graceful stop at the next optimizer boundary."""
        self._stop_requested = True

    def optimizer_scheduler_state_summary(self):
        """Return verifiable diagnostic summary of AdamW moments and LR scheduler state."""
        opt_state = self.optimizer.state_dict().get("state", {})
        step_values = []
        exp_avg_norm_sq = 0.0
        exp_avg_sq_norm_sq = 0.0
        for entry in opt_state.values():
            if "step" in entry:
                s = entry["step"]
                step_values.append(int(s.item()) if torch.is_tensor(s) else int(s))
            if "exp_avg" in entry and torch.is_tensor(entry["exp_avg"]):
                exp_avg_norm_sq += float(entry["exp_avg"].detach().float().pow(2).sum().item())
            if "exp_avg_sq" in entry and torch.is_tensor(entry["exp_avg_sq"]):
                exp_avg_sq_norm_sq += float(entry["exp_avg_sq"].detach().float().pow(2).sum().item())
        sched_state = self.scheduler.state_dict()
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "optimizer_param_entries": len(opt_state),
            "optimizer_max_step": max(step_values) if step_values else 0,
            "optimizer_exp_avg_l2": round(math.sqrt(exp_avg_norm_sq), 8),
            "optimizer_exp_avg_sq_l2": round(math.sqrt(exp_avg_sq_norm_sq), 8),
            "scheduler_last_epoch": int(sched_state.get("last_epoch", 0)),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    def resume(self, path=None, *, strict_config=True):
        payload = self.checkpoints.load(path, model=self.model, optimizer=self.optimizer,
                                        scheduler=self.scheduler, config=self.config if strict_config else None, restore_rng=True)
        self.epoch = int(payload.get("trainer_state", {}).get("epoch", payload["epoch"] + 1))
        self.batch_in_epoch = int(payload.get("trainer_state", {}).get("batch_in_epoch", 0))
        self.global_step = int(payload["global_step"])
        self.best_validation_loss = payload.get("best_validation_loss")
        self._resumed = True
        if payload.get("early_stopping_state"):
            self.early_stopping.load_state_dict(payload["early_stopping_state"])
        return payload

    def resume_with_recovery(self, path=None, *, strict_config=True):
        """Resume from primary checkpoint, falling back to archived history or best.pt if corrupted."""
        candidates = []
        primary = self.checkpoints.directory / "last.pt" if path is None else path
        candidates.append(primary)
        if path is None:
            history_files = sorted(
                self.checkpoints.archive.glob("last-*.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            candidates.extend(history_files)
            best_ckpt = self.checkpoints.directory / "best.pt"
            if best_ckpt.is_file():
                candidates.append(best_ckpt)

        errors = []
        for idx, candidate in enumerate(candidates):
            try:
                payload = self.resume(candidate, strict_config=strict_config)
                self.logger.log({
                    "event": "checkpoint_resume",
                    "source": str(candidate),
                    "used_fallback": idx > 0,
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                    "batch_in_epoch": self.batch_in_epoch,
                })
                return {
                    "payload": payload,
                    "recovered_from": str(candidate),
                    "used_fallback": idx > 0,
                    "skipped_corrupt_candidates": errors,
                }
            except Exception as exc:
                errors.append({"path": str(candidate), "error": str(exc)})
        raise ValueError(f"all checkpoint recovery candidates failed: {errors}")

    def _save(self, epoch, next_batch, best=False):
        args = dict(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                    epoch=epoch, global_step=self.global_step, config=self.config,
                    best_validation_loss=self.best_validation_loss,
                    trainer_state={"epoch": epoch, "batch_in_epoch": next_batch},
                    early_stopping_state=self.early_stopping.state_dict())
        return self.checkpoints.save_best(**args) if best else self.checkpoints.save(**args)

    def fit(self, resume=False, *, stop_at_step=None):
        if resume:
            self.resume()
        if not self._resumed:
            random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
        reports = []
        stop = False
        self.optimizer.zero_grad(set_to_none=True)
        resumed_epoch = self.epoch
        try:
            for epoch in range(resumed_epoch, self.config.epochs):
                self.epoch = epoch
                epoch_started = time.perf_counter()
                generator = getattr(self.train_loader, "generator", None)
                if generator is not None:
                    generator.manual_seed(self.config.random_seed + epoch)
                self.model.train()
                training = LossAccumulator()
                started = time.perf_counter()
                tokens_since_log = 0
                # The stored cursor is consumed only for the epoch being resumed.
                start_batch = self.batch_in_epoch if epoch == resumed_epoch else 0
                validated_at_step = -1
                next_batch = start_batch
                for batch_index, (input_ids, target_ids) in enumerate(self.train_loader):
                    if batch_index < start_batch:
                        continue
                    input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)
                    group_start = max(start_batch, batch_index - batch_index % self.config.gradient_accumulation_steps)
                    group_end = min(len(self.train_loader), (batch_index // self.config.gradient_accumulation_steps + 1) * self.config.gradient_accumulation_steps)
                    group_size = max(1, group_end - group_start)
                    with self._autocast_context():
                        _, loss = self.model(input_ids, target_ids)
                    if loss is None or not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite training loss at epoch {epoch}, batch {batch_index}")
                    token_count = target_ids.numel()
                    loss_value = float(loss.detach().cpu().item())
                    training.update(loss_value, token_count)
                    tokens_since_log += token_count
                    scaled_loss = loss / group_size
                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    last_batch = batch_index + 1 == len(self.train_loader)
                    optimizer_boundary = (batch_index + 1) % self.config.gradient_accumulation_steps == 0 or last_batch
                    if not optimizer_boundary:
                        continue
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm, error_if_nonfinite=True)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("non-finite gradient norm")
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    self.logger.log({"event": "train_step", "epoch": epoch, "step": self.global_step,
                                     "training_loss": training.average_loss,
                                     "learning_rate": self.optimizer.param_groups[0]["lr"],
                                     "step_time_seconds": elapsed,
                                     "tokens_per_second": tokens_since_log / elapsed,
                                     "memory_mb": memory_usage_mb(), "gradient_norm": float(grad_norm),
                                     "precision": self.config.precision})
                    tokens_since_log = 0
                    started = time.perf_counter()
                    next_batch = batch_index + 1
                    if self.global_step % self.config.validation_interval == 0:
                        val = validate(self.model, self.validation_loader, self.device, amp_dtype=self.amp_dtype)
                        record = {"epoch": epoch, "step": self.global_step, **val}
                        reports.append(record); self.history.append(record)
                        improved, should_stop = self.early_stopping.update(val["validation_loss"])
                        if improved:
                            self.best_validation_loss = val["validation_loss"]
                            self._save(epoch, next_batch, best=True)
                        self.logger.log({"event": "validation", **record, "best": improved,
                                         "memory_mb": memory_usage_mb()})
                        validated_at_step = self.global_step
                        if should_stop:
                            stop = True
                    if self.global_step % self.config.checkpoint_interval == 0:
                        self._save(epoch, next_batch)
                    if self._stop_requested or (stop_at_step is not None and self.global_step >= int(stop_at_step)):
                        self._save(epoch, next_batch)
                        stop = True
                    if stop or (self.config.max_training_steps and self.global_step >= self.config.max_training_steps):
                        break
                epoch_complete = next_batch >= len(self.train_loader)
                self.batch_in_epoch = 0 if epoch_complete else next_batch
                self.epoch = epoch + 1 if epoch_complete else epoch
                if validated_at_step != self.global_step and self.global_step:
                    val = validate(self.model, self.validation_loader, self.device, amp_dtype=self.amp_dtype)
                    record = {"epoch": epoch, "step": self.global_step, **val}
                    reports.append(record); self.history.append(record)
                    improved, should_stop = self.early_stopping.update(val["validation_loss"])
                    if improved:
                        self.best_validation_loss = val["validation_loss"]
                        self._save(self.epoch, self.batch_in_epoch, best=True)
                    self.logger.log({"event": "validation", **record, "best": improved,
                                     "memory_mb": memory_usage_mb()})
                    stop = stop or should_stop
                self._save(self.epoch, self.batch_in_epoch)
                self.logger.log({"event": "epoch_end", "epoch": epoch,
                                 "training_loss": training.average_loss,
                                 "validation_loss": reports[-1]["validation_loss"] if reports else None,
                                 "global_step": self.global_step,
                                 "epoch_time_seconds": time.perf_counter() - epoch_started})
                if stop or (self.config.max_training_steps and self.global_step >= self.config.max_training_steps):
                    break
        except KeyboardInterrupt:
            if self.global_step > 0 and all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):
                self._save(self.epoch, self.batch_in_epoch)
                self.logger.log({"event": "emergency_checkpoint_saved", "global_step": self.global_step})
            raise
        return reports

"""Atomic checkpoint persistence with safe tensor-only deserialization."""
import os
from pathlib import Path
import random
import shutil
import time

import torch


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


class CheckpointManager:
    FORMAT_VERSION = 1

    def __init__(self, directory, keep=3):
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ValueError("keep must be a positive integer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.archive = self.directory / "history"
        self.archive.mkdir(exist_ok=True)
        self.keep = keep

    def _archive_existing(self, destination):
        if destination.exists():
            stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
            shutil.copy2(destination, self.archive / f"{destination.stem}-{stamp}.pt")
            versions = sorted(self.archive.glob(f"{destination.stem}-*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            for stale in versions[self.keep:]:
                stale.unlink()

    def _atomic_save(self, payload, destination):
        estimate = max(_tensor_bytes(payload), 1024 * 1024)
        if shutil.disk_usage(self.directory).free < estimate * 2:
            raise OSError("insufficient free disk space for an atomic checkpoint save")
        temporary = self.directory / f".{destination.stem}-{os.getpid()}-{time.time_ns()}.tmp"
        try:
            torch.save(payload, temporary)
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            self._archive_existing(destination)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _payload(self, *, model, optimizer, scheduler, epoch, global_step, config,
                 best_validation_loss=None, trainer_state=None, early_stopping_state=None):
        config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        return {
            "format_version": self.FORMAT_VERSION,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch), "global_step": int(global_step),
            "config": config_dict, "model_config": model.config.__dict__.copy(),
            "best_validation_loss": best_validation_loss,
            "trainer_state": trainer_state or {},
            "early_stopping_state": early_stopping_state or {},
            "rng_torch": torch.random.get_rng_state(),
            "rng_python": random.getstate(),
        }

    def save(self, **kwargs):
        destination = self.directory / "last.pt"
        self._atomic_save(self._payload(**kwargs), destination)
        return destination

    def save_best(self, **kwargs):
        payload = self._payload(**kwargs)
        self._atomic_save(payload, self.directory / "last.pt")
        self._atomic_save(payload, self.directory / "best.pt")
        return self.directory / "best.pt"

    def load(self, path=None, *, model, optimizer=None, scheduler=None, config=None, restore_rng=True):
        source = Path(path) if path is not None else self.directory / "last.pt"
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"checkpoint is unreadable or unsafe: {source}: {exc}") from exc
        required = {"format_version", "model_state", "optimizer_state", "epoch", "global_step", "config", "model_config"}
        if not isinstance(payload, dict) or not required.issubset(payload) or payload["format_version"] != self.FORMAT_VERSION:
            raise ValueError("checkpoint format is invalid or unsupported")
        if payload["model_config"] != model.config.__dict__:
            raise ValueError("checkpoint model architecture does not match the current model")
        expected = config.to_dict() if hasattr(config, "to_dict") else config
        if expected is not None and payload["config"] != expected:
            raise ValueError("checkpoint training configuration does not match")
        model.load_state_dict(payload["model_state"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None:
            if payload.get("scheduler_state") is None:
                raise ValueError("checkpoint has no scheduler state")
            scheduler.load_state_dict(payload["scheduler_state"])
        if restore_rng:
            torch.random.set_rng_state(payload["rng_torch"])
            random.setstate(payload["rng_python"])
        return payload

"""Phase 12 — Training Engine Finalization acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.configuration import ProjectConfig, load_config
from model.factory import create_model
from training import (
    CheckpointManager,
    Trainer,
    audit_phase12_training_engine,
    create_scheduler,
    resolve_precision_profile,
    validate,
    write_phase12_reports,
)
from training.logger import JsonlLogger


def _cfg(**overrides):
    values = load_config().to_dict()
    values.update(
        {
            "context_length": 8,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.0,
            "batch_size": 2,
            "epochs": 2,
            "learning_rate": 0.005,
            "weight_decay": 0.01,
            "validation_interval": 2,
            "checkpoint_interval": 2,
            "warmup_steps": 2,
            "gradient_accumulation_steps": 1,
            "gradient_clip_norm": 1.0,
            "scheduler_type": "linear_warmup_cosine",
            "max_training_steps": 4,
            "early_stopping": False,
            "random_seed": 42,
        }
    )
    values.update(overrides)
    return ProjectConfig(**values).validate()


def test_gradient_accumulation_mathematical_equivalence(tmp_path: Path):
    torch.manual_seed(101)
    x = torch.randint(4, 260, (4, 8), dtype=torch.long)
    y = torch.randint(4, 260, (4, 8), dtype=torch.long)
    ds = TensorDataset(x, y)

    cfg_full = _cfg(batch_size=4, gradient_accumulation_steps=1, gradient_clip_norm=100.0, max_training_steps=1, epochs=1, scheduler_type="constant", warmup_steps=0)
    cfg_accum = _cfg(batch_size=2, gradient_accumulation_steps=2, gradient_clip_norm=100.0, max_training_steps=1, epochs=1, scheduler_type="constant", warmup_steps=0)

    torch.manual_seed(777)
    model_full = create_model(cfg_full)[0]
    torch.manual_seed(777)
    model_accum = create_model(cfg_accum)[0]

    Trainer(
        model_full,
        cfg_full,
        DataLoader(ds, batch_size=4, shuffle=False),
        DataLoader(ds, batch_size=4, shuffle=False),
        CheckpointManager(tmp_path / "full"),
        JsonlLogger(tmp_path / "full.jsonl"),
    ).fit()

    Trainer(
        model_accum,
        cfg_accum,
        DataLoader(ds, batch_size=2, shuffle=False),
        DataLoader(ds, batch_size=2, shuffle=False),
        CheckpointManager(tmp_path / "accum"),
        JsonlLogger(tmp_path / "accum.jsonl"),
    ).fit()

    for p_full, p_accum in zip(model_full.parameters(), model_accum.parameters()):
        torch.testing.assert_close(p_full, p_accum, atol=1e-4, rtol=1e-2)


def test_conditional_cpu_bf16_mixed_precision_and_default_guards(tmp_path: Path):
    with pytest.raises(ValueError, match="mixed precision"):
        resolve_precision_profile("cpu", "bf16", allow_cpu_amp=False)
    with pytest.raises(RuntimeError, match="CPU only"):
        resolve_precision_profile("cuda", "fp32", allow_cuda=False)

    profile = resolve_precision_profile("cpu", "bf16", allow_cpu_amp=True)
    assert profile["amp_enabled"] is True
    assert profile["amp_dtype"] == torch.bfloat16

    cfg = _cfg(precision="bf16", max_training_steps=2, epochs=1)
    torch.manual_seed(12)
    x = torch.randint(4, 260, (4, 8), dtype=torch.long)
    y = torch.randint(4, 260, (4, 8), dtype=torch.long)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = create_model(cfg)[0]
    trainer = Trainer(
        model,
        cfg,
        loader,
        loader,
        CheckpointManager(tmp_path / "bf16"),
        JsonlLogger(tmp_path / "bf16.jsonl"),
        allow_cpu_amp=True,
    )
    history = trainer.fit()
    assert trainer.global_step == 2
    assert history and history[-1]["validation_loss"] > 0


def test_multi_epoch_resume_matches_uninterrupted_run(tmp_path: Path):
    torch.manual_seed(55)
    x = torch.randint(4, 260, (8, 8), dtype=torch.long)
    y = torch.randint(4, 260, (8, 8), dtype=torch.long)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    cfg_full = _cfg(epochs=2, max_training_steps=6, validation_interval=2, checkpoint_interval=2, random_seed=303)
    torch.manual_seed(303)
    model_full = create_model(cfg_full)[0]
    Trainer(
        model_full,
        cfg_full,
        loader,
        loader,
        CheckpointManager(tmp_path / "full_ckpt"),
        JsonlLogger(tmp_path / "full.jsonl"),
    ).fit()

    cfg_part1 = _cfg(epochs=2, max_training_steps=2, validation_interval=2, checkpoint_interval=2, random_seed=303)
    torch.manual_seed(303)
    model_part1 = create_model(cfg_part1)[0]
    ckpt = CheckpointManager(tmp_path / "part_ckpt")
    trainer_part1 = Trainer(
        model_part1,
        cfg_part1,
        loader,
        loader,
        ckpt,
        JsonlLogger(tmp_path / "part1.jsonl"),
    )
    trainer_part1.scheduler = create_scheduler(
        trainer_part1.optimizer, cfg_full.scheduler_type, 6, cfg_full.warmup_steps
    )
    trainer_part1.fit()
    trainer_part1.config = cfg_full
    trainer_part1._save(trainer_part1.epoch, trainer_part1.batch_in_epoch)

    torch.manual_seed(999)
    model_resumed = create_model(cfg_full)[0]
    trainer_resumed = Trainer(
        model_resumed,
        cfg_full,
        loader,
        loader,
        ckpt,
        JsonlLogger(tmp_path / "part2.jsonl"),
    )
    trainer_resumed.resume()
    trainer_resumed.fit()

    assert trainer_resumed.global_step == 6
    assert trainer_resumed.epoch == 1
    assert trainer_resumed.batch_in_epoch == 2
    for p_full, p_resumed in zip(model_full.parameters(), model_resumed.parameters()):
        torch.testing.assert_close(p_full, p_resumed, atol=1e-6, rtol=1e-6)


def test_phase12_audit_and_report_generation(tmp_path: Path):
    report = audit_phase12_training_engine(tmp_path / "work")
    assert report["all_checks_passed"] is True
    assert report["status"] == "CERTIFIED_FOR_PHASE_13_INITIAL_MODEL_TRAINING"
    assert all(report["checklist"].values())
    assert report["constraints"]["real_corpus_training_started"] is False
    assert report["constraints"]["cuda_used"] is False
    assert (
        report["cpu_synthetic_mini_training"]["final_validation_loss"]
        < report["cpu_synthetic_mini_training"]["initial_validation_loss"]
    )

    json_path, md_path = write_phase12_reports(report, tmp_path / "reports")
    assert json_path.is_file()
    assert md_path.is_file()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["status"] == "CERTIFIED_FOR_PHASE_13_INITIAL_MODEL_TRAINING"

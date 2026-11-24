"""Phases 4 & 5 acceptance tests — Model forward ignore_index, Data Gate enforcement, exact AdamW/Scheduler resume & failure recovery."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.configuration import ProjectConfig, load_config
from model.factory import create_model
from model.transformer import GPT, GPTConfig
from tokenizer.byte_tokenizer import ByteTokenizer
from training import CheckpointManager, Trainer
from training.jebrex_train import preflight_resource_check, run_gated_training
from training.logger import JsonlLogger


def _mini_cfg(**overrides):
    values = load_config().to_dict()
    values.update(
        {
            "context_length": 8,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.0,
            "batch_size": 2,
            "epochs": 4,
            "learning_rate": 0.005,
            "weight_decay": 0.01,
            "validation_interval": 2,
            "checkpoint_interval": 2,
            "warmup_steps": 2,
            "gradient_accumulation_steps": 1,
            "gradient_clip_norm": 1.0,
            "scheduler_type": "linear_warmup_cosine",
            "max_training_steps": 6,
            "early_stopping": False,
            "random_seed": 77,
        }
    )
    values.update(overrides)
    return ProjectConfig(**values).validate()


def test_gpt_forward_supports_ignore_index_for_masked_sft_loss():
    tok = ByteTokenizer()
    cfg = GPTConfig(vocab_size=260, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = GPT(cfg)
    x = torch.tensor([[65, 66, 67, tok.pad_id]], dtype=torch.long)
    y_masked = torch.tensor([[tok.pad_id, tok.pad_id, 68, tok.pad_id]], dtype=torch.long)
    logits, loss = model(x, y_masked, ignore_index=tok.pad_id)
    assert logits.shape == (1, 4, 260)
    assert loss is not None and torch.isfinite(loss)

    # Also test standard -100 ignore_index
    y_neg = torch.tensor([[-100, -100, 68, -100]], dtype=torch.long)
    _, loss_neg = model(x, y_neg, ignore_index=-100)
    torch.testing.assert_close(loss, loss_neg)


def test_resume_preserves_exact_optimizer_moments_and_scheduler_lr(tmp_path: Path):
    torch.manual_seed(11)
    x = torch.randint(4, 250, (8, 8), dtype=torch.long)
    y = torch.randint(4, 250, (8, 8), dtype=torch.long)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    cfg = _mini_cfg(max_training_steps=6)

    # Reference uninterrupted run for 6 steps
    torch.manual_seed(77)
    ref_model = create_model(cfg)[0]
    ref_trainer = Trainer(
        ref_model, cfg, loader, loader,
        CheckpointManager(tmp_path / "ref"),
        JsonlLogger(tmp_path / "ref.jsonl"),
    )
    ref_trainer.fit()
    ref_summary = ref_trainer.optimizer_scheduler_state_summary()

    # Interrupted at step 3, then resumed in a fresh Trainer to step 6
    torch.manual_seed(77)
    part1_model = create_model(cfg)[0]
    shared_ckpt = CheckpointManager(tmp_path / "resumed")
    t1 = Trainer(part1_model, cfg, loader, loader, shared_ckpt, JsonlLogger(tmp_path / "p1.jsonl"))
    t1.fit(stop_at_step=3)
    s_before = t1.optimizer_scheduler_state_summary()
    assert s_before["global_step"] == 3
    assert s_before["optimizer_max_step"] == 3
    assert s_before["scheduler_last_epoch"] == 3
    assert s_before["optimizer_exp_avg_l2"] > 0.0

    torch.manual_seed(9999)
    part2_model = create_model(cfg)[0]
    t2 = Trainer(part2_model, cfg, loader, loader, shared_ckpt, JsonlLogger(tmp_path / "p2.jsonl"))
    t2.resume()
    s_after = t2.optimizer_scheduler_state_summary()
    assert s_after == s_before

    t2.fit()
    s_final = t2.optimizer_scheduler_state_summary()
    assert s_final["global_step"] == ref_summary["global_step"] == 6
    assert abs(s_final["learning_rate"] - ref_summary["learning_rate"]) < 1e-12
    assert abs(s_final["optimizer_exp_avg_l2"] - ref_summary["optimizer_exp_avg_l2"]) < 1e-6
    for p_ref, p_res in zip(ref_model.parameters(), part2_model.parameters()):
        torch.testing.assert_close(p_ref, p_res, atol=1e-6, rtol=1e-6)


def test_failure_recovery_from_corrupted_last_checkpoint(tmp_path: Path):
    torch.manual_seed(22)
    x = torch.randint(4, 250, (8, 8), dtype=torch.long)
    y = torch.randint(4, 250, (8, 8), dtype=torch.long)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    cfg = _mini_cfg(max_training_steps=4, checkpoint_interval=2, validation_interval=2)

    ckpt_mgr = CheckpointManager(tmp_path / "ckpt", keep=3)
    t1 = Trainer(create_model(cfg)[0], cfg, loader, loader, ckpt_mgr, JsonlLogger(tmp_path / "log1.jsonl"))
    t1.fit()

    # Corrupt last.pt to simulate mid-write / storage corruption
    (tmp_path / "ckpt" / "last.pt").write_bytes(b"CORRUPTED_BYTES")

    t2 = Trainer(create_model(cfg)[0], cfg, loader, loader, ckpt_mgr, JsonlLogger(tmp_path / "log2.jsonl"))
    recovery = t2.resume_with_recovery()
    assert recovery["used_fallback"] is True
    assert len(recovery["skipped_corrupt_candidates"]) == 1
    assert t2.global_step >= 2


def test_gated_training_blocks_unclean_data_and_rejects_unapproved_cuda(tmp_path: Path):
    with pytest.raises(RuntimeError, match="DATA READINESS GATE BLOCKED TRAINING"):
        run_gated_training(
            splits_dir="data/processed/v1_2_corpus/splits",
            checkpoint_dir=tmp_path / "blocked_ckpt",
            report_dir=tmp_path / "blocked_rep",
            total_steps=4,
            interrupt_at_step=2,
        )
    with pytest.raises(RuntimeError, match="CUDA"):
        preflight_resource_check(tmp_path, device="cuda", allow_cuda=False)

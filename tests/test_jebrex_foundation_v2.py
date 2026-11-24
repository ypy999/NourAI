"""Tests for Phase C & D: Jebrex v2 Foundation Transformer, Memmap Loader, and Cloud Trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from datasets.efficient_loader import (
    MemmapTokenDataset,
    build_memmap_dataloader,
    pack_jsonl_to_uint16_bin,
)
from model.jebrex_transformer import (
    JebrexConfig,
    JebrexTransformer,
    RMSNorm,
    SwiGLU,
    verify_preset_parameter_counts,
)
from tokenizer.byte_tokenizer import ByteTokenizer
from training.cloud_trainer import CloudResumableTrainer, CloudTrainConfig


def test_125m_and_350m_presets_exact_meta_param_counts():
    audit = verify_preset_parameter_counts()
    assert 120_000_000 <= audit["jebrex-125m"]["total_parameters"] <= 130_000_000
    assert 345_000_000 <= audit["jebrex-350m"]["total_parameters"] <= 360_000_000
    assert 500_000 <= audit["jebrex-micro"]["total_parameters"] <= 2_000_000


def test_causal_masking_and_kv_cache_exact_parity():
    torch.manual_seed(7)
    cfg = JebrexConfig(
        vocab_size=300,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=128,
        max_seq_len=64,
        dropout=0.0,
    )
    model = JebrexTransformer(cfg).eval()
    tokens = torch.randint(0, 300, (1, 16))

    # Causal invariance: changing future tokens must not change earlier logits
    out_full = model(tokens).logits
    tokens_perturbed = tokens.clone()
    tokens_perturbed[:, 10:] = (tokens_perturbed[:, 10:] + 1) % 300
    out_perturbed = model(tokens_perturbed).logits
    assert torch.allclose(out_full[:, :10, :], out_perturbed[:, :10, :], atol=1e-5)

    # Incremental KV-cache decode parity against full-sequence forward
    prefix_out = model(tokens[:, :10], use_cache=True)
    past = prefix_out.past_key_values
    cached_logits = [prefix_out.logits]
    for pos in range(10, 16):
        step_out = model(tokens[:, pos : pos + 1], past_key_values=past, use_cache=True)
        past = step_out.past_key_values
        cached_logits.append(step_out.logits)
    stitched = torch.cat(cached_logits, dim=1)
    assert torch.allclose(out_full, stitched, atol=1e-4)

    # Multi-token chunk continuation with existing KV-cache
    chunk_out = model(tokens[:, 10:16], past_key_values=prefix_out.past_key_values, use_cache=True)
    assert torch.allclose(out_full[:, 10:16, :], chunk_out.logits, atol=1e-4)


def test_gradient_checkpointing_matches_standard_gradients():
    torch.manual_seed(11)
    cfg = JebrexConfig(
        vocab_size=260,
        d_model=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=96,
        max_seq_len=32,
        dropout=0.0,
    )
    m_std = JebrexTransformer(cfg)
    m_ckpt = JebrexTransformer(cfg)
    m_ckpt.load_state_dict(m_std.state_dict())
    m_ckpt.enable_gradient_checkpointing(True)

    x = torch.randint(0, 260, (2, 16))
    y = torch.randint(0, 260, (2, 16))

    loss_std = m_std(x, targets=y).loss
    loss_ckpt = m_ckpt(x, targets=y).loss
    assert loss_std is not None and loss_ckpt is not None
    assert torch.allclose(loss_std, loss_ckpt, atol=1e-5)

    loss_std.backward()
    loss_ckpt.backward()
    for (n1, p1), (n2, p2) in zip(m_std.named_parameters(), m_ckpt.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5)


def test_memmap_loader_and_cloud_trainer_portable_bundle_resume(tmp_path: Path):
    tok = ByteTokenizer()
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"
    rows = [
        {"text": f"Jebrex AI multilingual sentence number {i} العربية Français English.\n"}
        for i in range(40)
    ]
    train_jsonl.write_text("\n".join(json.dumps(r) for r in rows[:32]), encoding="utf-8")
    val_jsonl.write_text("\n".join(json.dumps(r) for r in rows[32:]), encoding="utf-8")

    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    meta = pack_jsonl_to_uint16_bin(train_jsonl, train_bin, tok)
    pack_jsonl_to_uint16_bin(val_jsonl, val_bin, tok)
    assert meta["total_tokens"] > 100

    ds = MemmapTokenDataset(train_bin, block_size=32)
    x0, y0 = ds[0]
    assert x0.shape == (32,) and y0.shape == (32,)
    assert torch.equal(x0[1:], y0[:-1])

    train_loader = build_memmap_dataloader(train_bin, block_size=32, batch_size=2, shuffle=True, seed=9)
    val_loader = build_memmap_dataloader(val_bin, block_size=32, batch_size=2, shuffle=False, seed=9)

    cfg = JebrexConfig(
        vocab_size=tok.vocab_size,
        d_model=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=96,
        max_seq_len=64,
    )
    tcfg = CloudTrainConfig(
        batch_size=2,
        block_size=32,
        max_steps=6,
        warmup_steps=2,
        eval_interval=3,
        save_interval=3,
        gradient_checkpointing=True,
    )

    run_dir = tmp_path / "run"
    trainer1 = CloudResumableTrainer(JebrexTransformer(cfg), tcfg, run_dir)
    s1 = trainer1.train(train_loader, val_loader, stop_at_step=3)
    assert s1["global_step"] == 3
    bundle_dir = tmp_path / "bundle"
    manifest = trainer1.export_portable_bundle(bundle_dir)
    assert len(manifest["files"][0]["sha256"]) == 64

    # Corrupting a byte must trigger SHA-256 rejection
    bad_bundle = tmp_path / "bad_bundle"
    trainer1.export_portable_bundle(bad_bundle)
    ckpt_bytes = bytearray((bad_bundle / "checkpoint.pt").read_bytes())
    ckpt_bytes[0] ^= 0xFF
    (bad_bundle / "checkpoint.pt").write_bytes(bytes(ckpt_bytes))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        CloudResumableTrainer(JebrexTransformer(cfg), tcfg, run_dir).import_portable_bundle(bad_bundle)

    # Clean recovery from valid portable bundle
    trainer2 = CloudResumableTrainer(JebrexTransformer(cfg), tcfg, run_dir)
    info = trainer2.import_portable_bundle(bundle_dir)
    assert info["global_step"] == 3
    assert info["optimizer_state_entries"] > 0
    s2 = trainer2.train(train_loader, val_loader, stop_at_step=6)
    assert s2["global_step"] == 6
    assert s2["steps_executed_this_session"] == 3


def test_rmsnorm_and_swiglu_shapes():
    norm = RMSNorm(32)
    mlp = SwiGLU(32, 64)
    x = torch.randn(2, 8, 32)
    y = mlp(norm(x))
    assert y.shape == (2, 8, 32)
    assert torch.isfinite(y).all()

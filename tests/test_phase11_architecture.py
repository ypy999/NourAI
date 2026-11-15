import math
from pathlib import Path
import torch

from model.architecture_audit import audit_model_architecture
from model.factory import create_model, create_model_from_preset
from model.transformer import ARCHITECTURE_PRESETS, GPT, GPTConfig, causal_mask


def test_phase11_presets_and_parameter_counting():
    model, cfg = create_model_from_preset("default")
    assert cfg.n_layers == 4
    assert cfg.n_heads == 4
    assert cfg.d_model == 128
    assert cfg.context_length == 128
    assert cfg.vocab_size == 260
    assert cfg.head_dim == 32
    assert cfg.ffn_hidden_dim == 512

    breakdown = model.parameter_breakdown()
    # 260*128 (33280) + 128*128 (16384) + 4 * 198272 (793088) + 256 = 843008
    assert breakdown["token_embedding"] == 33280
    assert breakdown["positional_embedding"] == 16384
    assert breakdown["per_block"]["block_total"] == 198272
    assert breakdown["transformer_blocks_total"] == 793088
    assert breakdown["final_layer_norm"] == 256
    assert breakdown["lm_head_tied"] is True
    assert breakdown["total_unique_parameters"] == 843008
    assert model.count_parameters() == 843008

    small_model, _ = create_model_from_preset("small")
    base_model, _ = create_model_from_preset("base")
    assert small_model.count_parameters() > model.count_parameters()
    assert base_model.count_parameters() > small_model.count_parameters()


def test_phase11_gpt2_weight_initialization_and_causal_invariance():
    torch.manual_seed(1337)
    model = GPT(GPTConfig.from_preset("default")).eval()
    assert model.head.weight is model.token.weight
    assert torch.allclose(model.norm.weight, torch.ones_like(model.norm.weight))
    assert torch.allclose(model.norm.bias, torch.zeros_like(model.norm.bias))

    target_res_std = 0.02 / math.sqrt(2.0 * model.config.n_layers)
    actual_res_std = float(model.blocks[0].ff[2].weight.std().item())
    assert 0.5 * target_res_std <= actual_res_std <= 1.5 * target_res_std

    x1 = torch.randint(0, 260, (2, 16), dtype=torch.long)
    x2 = x1.clone()
    x2[:, 8:] = (x2[:, 8:] + 19) % 260
    with torch.no_grad():
        out1, _ = model(x1)
        out2, _ = model(x2)
    assert float((out1[:, :8, :] - out2[:, :8, :]).abs().max().item()) < 1e-6


def test_phase11_full_architecture_audit_and_checkpoint_roundtrip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    report = audit_model_architecture(root, temp_dir=tmp_path)
    assert report["status"] == "PASS", report["acceptance_criteria"]
    assert report["certification_verdict"] == "CERTIFIED_FOR_PHASE_12_TRAINING_ENGINE"
    assert all(report["acceptance_criteria"].values())

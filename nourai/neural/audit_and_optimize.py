"""Deep Transformer audit, KV-cache equivalence, CPU INT8 quantization, and SFT catastrophic forgetting protection."""
from __future__ import annotations

import copy
import time
from typing import Any

import torch
from torch import nn

from model.transformer import GPT, GPTConfig
from training.metrics import memory_usage_mb


def audit_transformer_model(model: GPT) -> dict[str, Any]:
    """Audit attention, Pre-LN, residual scaling, embeddings, weight tying, and KV-cache equivalence."""
    model.eval()
    c = model.config
    findings: list[str] = []

    # 1. Check weight tying
    weight_tied = model.head.weight is model.token.weight
    if not weight_tied:
        findings.append("LM head weight is not tied to token embedding weight.")

    # 2. Check Pre-LN placement and residual projections
    has_pre_ln = all(hasattr(b, "ln1") and hasattr(b, "ln2") for b in model.blocks)

    # 3. Check numerical equivalence between forward() and forward_with_cache()
    torch.manual_seed(42)
    seq_len = min(16, c.context_length)
    sample_ids = torch.randint(0, c.vocab_size, (1, seq_len), dtype=torch.long)
    with torch.no_grad():
        full_logits, _ = model(sample_ids)
        # Prefill first half, decode second half step-by-step
        split = max(1, seq_len // 2)
        prefill_logits, kvs = model.forward_with_cache(sample_ids[:, :split], past_kvs=None, pos_offset=0)
        step_logits = [prefill_logits]
        for pos in range(split, seq_len):
            lg, kvs = model.forward_with_cache(sample_ids[:, pos : pos + 1], past_kvs=kvs, pos_offset=pos)
            step_logits.append(lg)
        cached_logits = torch.cat(step_logits, dim=1)
        max_abs_diff = float((full_logits - cached_logits).abs().max().item())

    kv_cache_equivalent = max_abs_diff < 1e-4
    if not kv_cache_equivalent:
        findings.append(f"KV-cache logits diverge from full forward pass (max_abs_diff={max_abs_diff:.6f}).")

    # 4. Check gradient flow and finite loss
    model.train()
    x = torch.randint(0, c.vocab_size, (2, min(8, c.context_length)), dtype=torch.long)
    y = torch.randint(0, c.vocab_size, (2, min(8, c.context_length)), dtype=torch.long)
    _, loss = model(x, y)
    loss.backward()
    grad_norms = [float(p.grad.norm().item()) for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)
    model.eval()

    return {
        "architecture_summary": model.architecture_summary(),
        "weight_tied": weight_tied,
        "has_pre_ln": has_pre_ln,
        "kv_cache_max_abs_diff": round(max_abs_diff, 8),
        "kv_cache_equivalent": kv_cache_equivalent,
        "initial_loss_finite": bool(torch.isfinite(loss).item()),
        "initial_loss": round(float(loss.item()), 4),
        "nonzero_grad_tensors": sum(1 for g in grad_norms if g > 0),
        "total_grad_tensors": len(grad_norms),
        "findings": findings,
        "audit_passed": weight_tied and has_pre_ln and kv_cache_equivalent and bool(torch.isfinite(loss).item()),
    }


def quantize_model_dynamic_int8(model: GPT) -> nn.Module:
    """Apply CPU-safe dynamic INT8 quantization to Linear layers (leaves original model untouched)."""
    model_copy = copy.deepcopy(model).to("cpu").eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def benchmark_kv_cache_and_quantization(
    model: GPT,
    *,
    prompt_len: int = 16,
    gen_steps: int = 24,
) -> dict[str, Any]:
    """Benchmark uncached FP32 vs KV-cached FP32 vs dynamic INT8 CPU inference."""
    model.eval()
    c = model.config
    prompt_len = min(prompt_len, max(4, c.context_length - gen_steps - 1))
    gen_steps = min(gen_steps, max(1, c.context_length - prompt_len - 1))
    torch.manual_seed(123)
    prompt = torch.randint(0, c.vocab_size, (1, prompt_len), dtype=torch.long)

    # 1. Uncached FP32 decoding
    t0 = time.perf_counter()
    with torch.no_grad():
        seq = prompt.clone()
        for _ in range(gen_steps):
            logits, _ = model(seq[:, -c.context_length :])
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            seq = torch.cat((seq, nxt), dim=1)
    uncached_ms = (time.perf_counter() - t0) * 1000.0

    # 2. KV-cached FP32 decoding
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, kvs = model.forward_with_cache(prompt, past_kvs=None, pos_offset=0)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        pos = prompt.size(1)
        for _ in range(gen_steps - 1):
            logits, kvs = model.forward_with_cache(nxt, past_kvs=kvs, pos_offset=pos)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            pos += 1
    cached_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Dynamic INT8 quantized forward benchmark
    q_available = True
    q_ms = None
    q_error = None
    try:
        q_model = quantize_model_dynamic_int8(model)
        t0 = time.perf_counter()
        with torch.no_grad():
            seq_q = prompt.clone()
            for _ in range(gen_steps):
                logits, _ = q_model(seq_q[:, -c.context_length :])
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                seq_q = torch.cat((seq_q, nxt), dim=1)
        q_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        q_available = False
        q_error = str(exc)

    rss = memory_usage_mb()
    return {
        "preset_params": model.count_parameters(trainable_only=False),
        "prompt_len": prompt_len,
        "gen_steps": gen_steps,
        "fp32_uncached_ms": round(uncached_ms, 2),
        "fp32_kv_cached_ms": round(cached_ms, 2),
        "kv_cache_speedup": round(uncached_ms / max(0.01, cached_ms), 2),
        "int8_dynamic_available": q_available,
        "int8_dynamic_ms": round(q_ms, 2) if q_ms is not None else None,
        "int8_error": q_error,
        "process_rss_mb": round(rss, 2) if rss is not None else None,
    }


class L2AnchorRegularizer:
    """EWC-lite L2 parameter anchor regularizer to prevent catastrophic forgetting during SFT.

    Penalizes large parameter drift from the pretrained checkpoint:
        penalty = (lambda_anchor / 2) * sum((theta_i - theta_anchor_i)^2)
    """

    def __init__(self, pretrained_model: nn.Module, *, lambda_anchor: float = 1e-3):
        if lambda_anchor < 0:
            raise ValueError("lambda_anchor must be non-negative")
        self.lambda_anchor = float(lambda_anchor)
        self._anchors: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in pretrained_model.named_parameters()
            if param.requires_grad
        }

    def penalty(self, model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        total = torch.zeros((), device=device, dtype=torch.float32)
        if self.lambda_anchor == 0.0:
            return total
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            anchor = self._anchors.get(name)
            if anchor is None:
                continue
            diff = param - anchor.to(device=device, dtype=param.dtype)
            total = total + torch.sum(diff * diff)
        return 0.5 * self.lambda_anchor * total

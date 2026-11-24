"""Jebrex AI — Modern Foundation Transformer Architecture (Phase C, 125M–350M Ready).

Implements production-standard decoder-only Transformer components:
- RMSNorm (Pre-Normalization)
- Rotary Positional Embeddings (RoPE) with KV-cache offset support
- Causal Grouped-Query Attention (GQA) with KV-Cache incremental decoding
- SwiGLU Feed-Forward Network
- Weight-tied token embeddings and LM head
- Optional Gradient Checkpointing (torch.utils.checkpoint, non-reentrant)
- Presets: jebrex-micro (~1.1M CPU sanity), jebrex-125m (~124.3M), jebrex-350m (~351.3M)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


@dataclass
class JebrexConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 6
    intermediate_size: int = 2816
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    architecture_version: str = "jebrex-v2-modern"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "JebrexConfig":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


JEBBREX_MODEL_PRESETS: Dict[str, JebrexConfig] = {
    "jebrex-micro": JebrexConfig(
        vocab_size=2312,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=384,
        max_seq_len=256,
        dropout=0.0,
    ),
    "jebrex-125m": JebrexConfig(
        vocab_size=32768,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=6,
        intermediate_size=2816,
        max_seq_len=2048,
        dropout=0.0,
    ),
    "jebrex-350m": JebrexConfig(
        vocab_size=32768,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        intermediate_size=3456,
        max_seq_len=4096,
        dropout=0.0,
    ),
}


@dataclass
class JebrexModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.float()
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_fp32 * inv_rms).to(in_dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed Rotary Positional Embeddings (RoPE) with offset slicing."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._cached_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos().to(dtype=dtype)
        self._sin_cached = emb.sin().to(dtype=dtype)
        self._cached_len = seq_len

    def forward(
        self,
        seq_len: int,
        offset: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        needed = offset + seq_len
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._cached_len < needed
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._build_cache(max(needed, self.max_seq_len), device=device, dtype=dtype)
        assert self._cos_cached is not None and self._sin_cached is not None
        cos = self._cos_cached[offset:needed].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        sin = self._sin_cached[offset:needed].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class GroupedQueryAttention(nn.Module):
    """Causal Grouped-Query Attention with optional KV-Cache."""

    def __init__(self, cfg: JebrexConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        bsz, n_kv, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bsz, n_kv, self.n_rep, seq_len, head_dim)
            .reshape(bsz, n_kv * self.n_rep, seq_len, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present_kv = (k, v) if use_cache else None
        k_exp = self._repeat_kv(k)
        v_exp = self._repeat_kv(v)

        kv_len = k_exp.shape[2]
        drop_p = self.dropout if self.training else 0.0

        if past_key_value is None:
            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=None, dropout_p=drop_p, is_causal=True
            )
        elif q_len == 1:
            # Single newly decoded token attends causally to all past + current tokens.
            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=None, dropout_p=drop_p, is_causal=False
            )
        else:
            # Multi-token chunk with non-empty past cache: construct exact offset causal mask.
            past_len = kv_len - q_len
            mask = torch.ones((q_len, kv_len), dtype=torch.bool, device=x.device)
            mask = torch.tril(mask, diagonal=past_len)
            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=mask, dropout_p=drop_p, is_causal=False
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.n_heads * self.head_dim)
        return self.o_proj(attn_out), present_kv


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class JebrexBlock(nn.Module):
    """Pre-RMSNorm Transformer block with GQA + SwiGLU."""

    def __init__(self, cfg: JebrexConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.intermediate_size, dropout=cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present_kv = self.attn(
            self.attn_norm(x),
            cos=cos,
            sin=sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, present_kv


class JebrexTransformer(nn.Module):
    """Jebrex AI v2 Modern Foundation Decoder-Only Transformer."""

    def __init__(self, config: JebrexConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([JebrexBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.gradient_checkpointing: bool = bool(config.gradient_checkpointing)
        self.apply(self._init_weights)
        # Residual output projection scaling (GPT-2 / Llama depth-aware init)
        residual_std = 0.02 / math.sqrt(2.0 * config.n_layers)
        for layer in self.layers:
            if layer.attn.o_proj.weight.device.type != "meta":
                nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=residual_std)
                nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if module.weight.device.type != "meta":
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)
        self.config.gradient_checkpointing = bool(enabled)

    def count_parameters(self) -> Dict[str, int]:
        """Count unique parameters accurately (accounting for weight tying)."""
        seen: set[int] = set()
        total = 0
        trainable = 0
        for p in self.parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        return {"total_parameters": total, "trainable_parameters": trainable}

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        ignore_index: int = -100,
    ) -> JebrexModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected 2D input_ids (B, T), got shape {tuple(input_ids.shape)}")
        _, seq_len = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if past_len + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {past_len + seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(seq_len=seq_len, offset=past_len, device=x.device, dtype=x.dtype)

        new_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        use_ckpt = self.gradient_checkpointing and self.training and not use_cache

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if use_ckpt:
                def _layer_fn(hidden: torch.Tensor, _layer: JebrexBlock = layer) -> torch.Tensor:
                    out, _ = _layer(hidden, cos=cos, sin=sin, past_key_value=None, use_cache=False)
                    return out

                x = torch_checkpoint(_layer_fn, x, use_reentrant=False)
                present = None
            else:
                x, present = layer(
                    x, cos=cos, sin=sin, past_key_value=layer_past, use_cache=use_cache
                )
            if use_cache and present is not None and new_cache is not None:
                new_cache.append(present)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )

        return JebrexModelOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(new_cache) if new_cache is not None else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive decoding with KV-Cache and NaN-safe top-k/top-p sampling."""
        self.eval()
        generated = input_ids
        past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None

        for step in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_seq_len:
                break
            if use_cache and past_kv is not None:
                model_in = generated[:, -1:]
            else:
                model_in = generated[:, -self.config.max_seq_len :]
                past_kv = None

            out = self.forward(model_in, past_key_values=past_kv, use_cache=use_cache)
            next_logits = out.logits[:, -1, :].float()
            next_logits = torch.nan_to_num(next_logits, nan=-1e9, posinf=1e4, neginf=-1e9)
            if use_cache:
                past_kv = out.past_key_values

            if temperature <= 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / max(temperature, 1e-5)
                if top_k > 0 and top_k < scaled.size(-1):
                    vals, _ = torch.topk(scaled, top_k, dim=-1)
                    cutoff = vals[:, -1:]
                    scaled = torch.where(scaled < cutoff, torch.full_like(scaled, -float("inf")), scaled)
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(probs, dim=-1)
                    remove = cumprobs > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                    scaled = torch.zeros_like(scaled).scatter_(1, sorted_idx, sorted_logits)
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        return generated


def verify_preset_parameter_counts() -> Dict[str, Dict[str, object]]:
    """Verify exact parameter counts on torch.device('meta') with zero RAM overhead."""
    summary: Dict[str, Dict[str, object]] = {}
    for name, cfg in JEBBREX_MODEL_PRESETS.items():
        with torch.device("meta"):
            model = JebrexTransformer(cfg)
            counts = model.count_parameters()
        # FP32 weights + BF16 weights memory estimates (MB)
        params = counts["total_parameters"]
        summary[name] = {
            "config": cfg.to_dict(),
            "total_parameters": params,
            "parameters_millions": round(params / 1_000_000.0, 3),
            "weights_bf16_mib": round((params * 2) / (1024 * 1024), 2),
            "weights_fp32_mib": round((params * 4) / (1024 * 1024), 2),
            "adamw_training_states_fp32_mib": round((params * 16) / (1024 * 1024), 2),
        }
    return summary

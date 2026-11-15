from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

ARCHITECTURE_PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "vocab_size": 260,
        "context_length": 128,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "dropout": 0.1,
    },
    "nano": {
        "vocab_size": 260,
        "context_length": 128,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "dropout": 0.1,
    },
    "small": {
        "vocab_size": 260,
        "context_length": 256,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 6,
        "dropout": 0.1,
    },
    "base": {
        "vocab_size": 260,
        "context_length": 512,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,
        "dropout": 0.1,
    },
}


@dataclass
class GPTConfig:
    vocab_size: int = 260
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "d_model", "n_heads", "n_layers"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        return 4 * self.d_model

    @classmethod
    def from_preset(cls, preset_name: str = "default", **overrides) -> "GPTConfig":
        if preset_name not in ARCHITECTURE_PRESETS:
            raise ValueError(f"unknown architecture preset {preset_name!r}; available: {sorted(ARCHITECTURE_PRESETS)}")
        params = dict(ARCHITECTURE_PRESETS[preset_name])
        params.update(overrides)
        return cls(**params)


def causal_mask(length: int, device=None) -> torch.Tensor:
    """Boolean True entries are blocked by PyTorch MultiheadAttention."""
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.attn = nn.MultiheadAttention(c.d_model, c.n_heads, dropout=c.dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(c.d_model, 4 * c.d_model),
            nn.GELU(),
            nn.Linear(4 * c.d_model, c.d_model),
            nn.Dropout(c.dropout),
        )

    def forward(self, x):
        mask = causal_mask(x.size(1), device=x.device)
        normalized = self.ln1(x)
        attended, _ = self.attn(normalized, normalized, normalized, attn_mask=mask, need_weights=False)
        x = x + attended
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config: Optional[GPTConfig] = None):
        super().__init__()
        self.config = config if config is not None else GPTConfig()
        c = self.config
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.context_length, c.d_model)
        self.emb_dropout = nn.Dropout(c.dropout)
        self.blocks = nn.Sequential(*(Block(c) for _ in range(c.n_layers)))
        self.norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self._init_weights()
        self.head.weight = self.token.weight

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions with residual depth scaling."""
        residual_std = 0.02 / math.sqrt(2.0 * self.config.n_layers)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                nn.init.normal_(module.out_proj.weight, mean=0.0, std=residual_std)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

        for block in self.blocks:
            if isinstance(block, Block) and isinstance(block.ff[2], nn.Linear):
                nn.init.normal_(block.ff[2].weight, mean=0.0, std=residual_std)

    def count_parameters(self, *, trainable_only: bool = True) -> int:
        """Return total unique parameters (accounting for tied token/head weights)."""
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    def parameter_breakdown(self) -> dict[str, Any]:
        """Return exact parameter counts by architectural component."""
        tok_params = self.token.weight.numel()
        pos_params = self.pos.weight.numel()
        per_block = []
        blocks_total = 0
        for idx, block in enumerate(self.blocks):
            ln1_p = sum(p.numel() for p in block.ln1.parameters())
            attn_p = sum(p.numel() for p in block.attn.parameters())
            ln2_p = sum(p.numel() for p in block.ln2.parameters())
            ff_p = sum(p.numel() for p in block.ff.parameters())
            b_tot = ln1_p + attn_p + ln2_p + ff_p
            blocks_total += b_tot
            per_block.append(
                {
                    "layer_index": idx,
                    "ln1": ln1_p,
                    "self_attention": attn_p,
                    "ln2": ln2_p,
                    "feed_forward": ff_p,
                    "block_total": b_tot,
                }
            )
        final_norm_params = sum(p.numel() for p in self.norm.parameters())
        unique_total = self.count_parameters(trainable_only=False)
        trainable_total = self.count_parameters(trainable_only=True)
        return {
            "token_embedding": tok_params,
            "positional_embedding": pos_params,
            "transformer_blocks_total": blocks_total,
            "per_block": per_block[0] if per_block else {},
            "n_layers": self.config.n_layers,
            "final_layer_norm": final_norm_params,
            "lm_head_tied": self.head.weight is self.token.weight,
            "lm_head_shared_parameters": self.head.weight.numel(),
            "total_unique_parameters": unique_total,
            "total_trainable_parameters": trainable_total,
        }

    def architecture_summary(self) -> dict[str, Any]:
        """Return full specification of the finalized GPT architecture."""
        return {
            "architecture_type": "Decoder-only Pre-LN Transformer (GPT)",
            "config": asdict(self.config),
            "head_dim": self.config.head_dim,
            "ffn_hidden_dim": self.config.ffn_hidden_dim,
            "layer_norm_placement": "Pre-LN (before self-attention and before FFN, plus final LayerNorm)",
            "activation_function": "GELU",
            "positional_encoding": "Learned absolute positional embeddings (0 .. context_length - 1)",
            "causal_masking": "Strict upper-triangular boolean mask (diagonal=1)",
            "weight_initialization": "GPT-2 normal(0, 0.02) with residual projection scaling 0.02 / sqrt(2 * n_layers)",
            "weight_tying": bool(self.head.weight is self.token.weight),
            "parameters": self.parameter_breakdown(),
        }

    @staticmethod
    def _validate_ids(ids, name):
        if not isinstance(ids, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"{name} must have an integer dtype")

    def forward(self, idx, targets=None):
        self._validate_ids(idx, "idx")
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch, time]")
        b, t = idx.shape
        if b <= 0 or t <= 0:
            raise ValueError("idx batch and sequence dimensions must be non-empty")
        if t > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        if torch.any(idx < 0) or torch.any(idx >= self.config.vocab_size):
            raise ValueError("idx contains token IDs outside the configured vocabulary")
        positions = torch.arange(t, device=idx.device)
        x = self.emb_dropout(self.token(idx.long()) + self.pos(positions))
        logits = self.head(self.norm(self.blocks(x)))
        loss = None
        if targets is not None:
            self._validate_ids(targets, "targets")
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same [batch, time] shape as idx")
            if torch.any(targets < 0) or torch.any(targets >= self.config.vocab_size):
                raise ValueError("targets contains token IDs outside the configured vocabulary")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.long().reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=40, eos_id=None):
        self._validate_ids(idx, "idx")
        if idx.ndim != 2 or idx.shape[0] <= 0 or idx.shape[1] <= 0:
            raise ValueError("idx must be a non-empty [batch, time] tensor")
        if torch.any(idx < 0) or torch.any(idx >= self.config.vocab_size):
            raise ValueError("idx contains token IDs outside the configured vocabulary")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError("temperature must be positive")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if eos_id is not None and (isinstance(eos_id, bool) or not isinstance(eos_id, int) or not 0 <= eos_id < self.config.vocab_size):
            raise ValueError("eos_id must be a valid token ID")

        was_training = self.training
        self.eval()
        result = idx.long()
        finished = torch.zeros(result.size(0), device=result.device, dtype=torch.bool)
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(result[:, -self.config.context_length:])
                logits = logits[:, -1, :] / float(temperature)
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                if eos_id is not None:
                    next_token = torch.where(finished[:, None], torch.full_like(next_token, eos_id), next_token)
                result = torch.cat((result, next_token), dim=1)
                if eos_id is not None:
                    finished |= next_token[:, 0].eq(eos_id)
                    if bool(finished.all()):
                        break
            return result
        finally:
            self.train(was_training)

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "config": asdict(self.config),
            "parameter_count": self.count_parameters(trainable_only=False),
            "state_dict": self.state_dict(),
        }
        fd, temp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @classmethod
    def load_checkpoint(cls, path, map_location="cpu"):
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:  # compatibility with PyTorch versions without weights_only
            payload = torch.load(path, map_location=map_location)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("unsupported or invalid checkpoint format")
        model = cls(GPTConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model

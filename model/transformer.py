from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


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
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


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
            nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(),
            nn.Linear(4 * c.d_model, c.d_model), nn.Dropout(c.dropout),
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
        self.blocks = nn.Sequential(*(Block(c) for _ in range(c.n_layers)))
        self.norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.token.weight

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
        logits = self.head(self.norm(self.blocks(self.token(idx.long()) + self.pos(positions))))
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
        torch.save({"format_version": 1, "config": asdict(self.config), "state_dict": self.state_dict()}, path)

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

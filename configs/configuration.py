"""Validated project configuration loaded from JSON."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tokenizer.byte_tokenizer import ByteTokenizer


@dataclass(frozen=True)
class ProjectConfig:
    vocab_size: int = 260
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    batch_size: int = 8
    learning_rate: float = 0.0003
    device: str = "cpu"
    validation_fraction: float = 0.1
    random_seed: int = 1337
    max_document_chars: int = 2_000_000
    io_batch_lines: int = 512
    cleaning_unicode_normalization: str = "NFC"
    cleaning_remove_control_chars: bool = True
    cleaning_collapse_whitespace: bool = True
    cleaning_preserve_paragraphs: bool = False
    cleaning_min_chars: int = 1
    cleaning_reject_short: bool = False
    epochs: int = 3
    weight_decay: float = 0.01
    adam_betas: tuple = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    scheduler_type: str = "cosine"
    precision: str = "fp32"
    validation_interval: int = 100
    checkpoint_interval: int = 100
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    max_training_steps: int = 0
    checkpoint_keep: int = 3

    def validate(self, tokenizer=None):
        tok = tokenizer or ByteTokenizer()
        positive_ints = ("vocab_size", "context_length", "d_model", "n_heads", "n_layers", "batch_size", "max_document_chars", "io_batch_lines", "cleaning_min_chars", "epochs", "gradient_accumulation_steps", "validation_interval", "checkpoint_interval", "early_stopping_patience", "checkpoint_keep")
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab_size != tok.vocab_size:
            raise ValueError(f"vocab_size ({self.vocab_size}) does not match tokenizer ({tok.vocab_size})")
        if self.cleaning_unicode_normalization not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError("cleaning_unicode_normalization must be NFC, NFD, NFKC, or NFKD")
        for name in ("cleaning_remove_control_chars", "cleaning_collapse_whitespace", "cleaning_preserve_paragraphs", "cleaning_reject_short"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be a number in [0, 1)")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        if isinstance(self.validation_fraction, bool) or not isinstance(self.validation_fraction, (int, float)) or not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        for name in ("warmup_steps", "max_training_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("weight_decay", "adam_epsilon", "early_stopping_min_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        if self.adam_epsilon <= 0:
            raise ValueError("adam_epsilon must be positive")
        if isinstance(self.gradient_clip_norm, bool) or not isinstance(self.gradient_clip_norm, (int, float)) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be a positive number")
        if not isinstance(self.adam_betas, (list, tuple)) or len(self.adam_betas) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 < x < 1 for x in self.adam_betas):
            raise ValueError("adam_betas must contain two numbers in (0, 1)")
        if self.scheduler_type not in ("linear_warmup_cosine", "cosine", "constant"):
            raise ValueError("scheduler_type must be linear_warmup_cosine, cosine, or constant")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16, or bf16")
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be boolean")
        return self

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def load_config(path=None, tokenizer=None) -> ProjectConfig:
    if path is None:
        path = Path(__file__).with_name("default.json")
    path = Path(path)
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"configuration file not found: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON configuration at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a JSON object")
    allowed = set(ProjectConfig.__dataclass_fields__)
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
    try:
        return ProjectConfig(**payload).validate(tokenizer)
    except TypeError as exc:
        raise ValueError(f"invalid configuration fields: {exc}") from exc

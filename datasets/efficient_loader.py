"""Jebrex AI — Memory-Mapped Efficient Token Stream Data Loader (Phase C).

Packs multilingual JSONL splits into compact uint16 binary shards (.bin) with a
JSON metadata sidecar (including SHA-256) and streams fixed-length (x, y) blocks
via numpy.memmap with near-zero RAM footprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenizerProtocol(Protocol):
    vocab_size: int
    bos_id: int
    eos_id: int

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> Sequence[int]:
        ...


def pack_jsonl_to_uint16_bin(
    jsonl_path: Path,
    out_bin_path: Path,
    tokenizer: TokenizerProtocol,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Dict[str, object]:
    """Tokenize a JSONL split once into a flat uint16 binary token stream."""
    if tokenizer.vocab_size > 65536:
        raise ValueError("uint16 packing requires vocab_size <= 65536")
    out_bin_path = Path(out_bin_path)
    out_bin_path.parent.mkdir(parents=True, exist_ok=True)

    tokens: list[int] = []
    doc_count = 0
    with Path(jsonl_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not isinstance(text, str) or not text:
                continue
            doc_count += 1
            tokens.extend(tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos))

    arr = np.asarray(tokens, dtype=np.uint16)
    arr.tofile(out_bin_path)
    sha = hashlib.sha256(out_bin_path.read_bytes()).hexdigest()

    meta = {
        "source_jsonl": str(jsonl_path).replace("\\", "/"),
        "bin_path": str(out_bin_path).replace("\\", "/"),
        "dtype": "uint16",
        "vocab_size": int(tokenizer.vocab_size),
        "documents": doc_count,
        "total_tokens": int(arr.size),
        "size_bytes": int(out_bin_path.stat().st_size),
        "sha256": sha,
    }
    meta_path = out_bin_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class MemmapTokenDataset(Dataset):
    """Zero-copy memory-mapped next-token prediction dataset."""

    def __init__(self, bin_path: Path, block_size: int, stride: Optional[int] = None) -> None:
        self.bin_path = Path(bin_path)
        self.block_size = int(block_size)
        self.stride = int(stride) if stride is not None else self.block_size
        if self.block_size < 2:
            raise ValueError("block_size must be >= 2")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        self._data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = int(self._data.shape[0])
        if self.total_tokens < self.block_size + 1:
            raise ValueError(
                f"Token stream ({self.total_tokens}) is smaller than block_size + 1 ({self.block_size + 1})"
            )
        self._num_samples = ((self.total_tokens - (self.block_size + 1)) // self.stride) + 1

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(idx)
        start = idx * self.stride
        chunk = np.asarray(self._data[start : start + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def build_memmap_dataloader(
    bin_path: Path,
    block_size: int,
    batch_size: int,
    shuffle: bool = True,
    stride: Optional[int] = None,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    ds = MemmapTokenDataset(bin_path=bin_path, block_size=block_size, stride=stride)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

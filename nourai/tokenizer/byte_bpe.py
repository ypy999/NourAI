"""Pure-Python Lossless Byte-Level BPE Tokenizer for Arabic, English, and French.

Design goals:
- Zero external dependencies (pure Python standard library).
- 100% lossless UTF-8 round-trip for Arabic, English, French, and arbitrary Unicode.
- Base vocabulary 0..255 (raw UTF-8 bytes), special tokens 256..262, learned BPE merges 263..(vocab_size - 1).
- Supports evaluating 8K (8192) and 16K (16384) target vocabulary sizes as well as smaller test vocabularies.
- Compatible interface with ByteTokenizer: .encode(), .decode(), .vocab_size, .pad_id, .bos_id, .eos_id, .unk_id.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable

from tokenizer.byte_tokenizer import ByteTokenizer

# Pre-tokenization regex splitting words, Arabic/Latin runs, numbers, punctuation, and whitespace
_PRETOKEN_RE = re.compile(
    r""" ?[A-Za-zÀ-ÖØ-öø-ÿ]+| ?[\u0600-\u06FF]+| ?[0-9]+| ?[^\sA-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06FF0-9]+|\s+""",
    re.UNICODE,
)

SPECIAL_TOKENS: dict[str, int] = {
    "<PAD>": 256,
    "<BOS>": 257,
    "<EOS>": 258,
    "<UNK>": 259,
    "<|user|>": 260,
    "<|assistant|>": 261,
    "<|system|>": 262,
}

BASE_MERGE_START_ID = 263


class ByteLevelBPETokenizer:
    """Lossless Byte-Level BPE tokenizer with full Arabic/English/French support."""

    pad_id = SPECIAL_TOKENS["<PAD>"]
    bos_id = SPECIAL_TOKENS["<BOS>"]
    eos_id = SPECIAL_TOKENS["<EOS>"]
    unk_id = SPECIAL_TOKENS["<UNK>"]

    def __init__(
        self,
        merges: list[tuple[int, int]] | None = None,
        *,
        target_vocab_size: int = 8192,
    ):
        if target_vocab_size < BASE_MERGE_START_ID:
            raise ValueError(f"target_vocab_size must be >= {BASE_MERGE_START_ID}")
        self.target_vocab_size = int(target_vocab_size)
        self.merges: list[tuple[int, int]] = list(merges or [])
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._rebuild_tables()

    def _rebuild_tables(self) -> None:
        self._merge_ranks = {}
        self._id_to_bytes = {i: bytes([i]) for i in range(256)}
        for idx, pair in enumerate(self.merges):
            new_id = BASE_MERGE_START_ID + idx
            self._merge_ranks[pair] = idx
            self._id_to_bytes[new_id] = self._id_to_bytes[pair[0]] + self._id_to_bytes[pair[1]]

    @property
    def vocab_size(self) -> int:
        return BASE_MERGE_START_ID + len(self.merges)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 8192,
        min_pair_frequency: int = 2,
        max_pretokens: int = 50000,
    ) -> "ByteLevelBPETokenizer":
        """Train a Byte-Level BPE vocabulary over a multilingual corpus."""
        if vocab_size < BASE_MERGE_START_ID:
            raise ValueError(f"vocab_size must be >= {BASE_MERGE_START_ID}")
        num_merges = vocab_size - BASE_MERGE_START_ID

        word_freqs: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            if not text:
                continue
            for piece in _PRETOKEN_RE.findall(text):
                b_tuple = tuple(piece.encode("utf-8", errors="strict"))
                if b_tuple:
                    word_freqs[b_tuple] += 1
                    if len(word_freqs) >= max_pretokens:
                        break

        merges: list[tuple[int, int]] = []
        for merge_idx in range(num_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for seq, freq in word_freqs.items():
                if len(seq) < 2:
                    continue
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += freq
            if not pair_counts:
                break
            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < min_pair_frequency:
                break
            new_id = BASE_MERGE_START_ID + merge_idx
            merges.append(best_pair)
            p0, p1 = best_pair

            updated_freqs: Counter[tuple[int, ...]] = Counter()
            for seq, freq in word_freqs.items():
                if len(seq) < 2:
                    updated_freqs[seq] += freq
                    continue
                merged_seq: list[int] = []
                i = 0
                n = len(seq)
                while i < n:
                    if i < n - 1 and seq[i] == p0 and seq[i + 1] == p1:
                        merged_seq.append(new_id)
                        i += 2
                    else:
                        merged_seq.append(seq[i])
                        i += 1
                updated_freqs[tuple(merged_seq)] += freq
            word_freqs = updated_freqs

        return cls(merges=merges, target_vocab_size=vocab_size)

    def _encode_piece(self, piece: str) -> list[int]:
        ids = list(piece.encode("utf-8", errors="strict"))
        if not self._merge_ranks or len(ids) < 2:
            return ids
        while len(ids) >= 2:
            best_rank = None
            best_pair = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None or best_rank is None:
                break
            new_id = BASE_MERGE_START_ID + best_rank
            p0, p1 = best_pair
            new_ids: list[int] = []
            i = 0
            n = len(ids)
            while i < n:
                if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in _PRETOKEN_RE.findall(text):
            ids.extend(self._encode_piece(piece))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        if errors not in ("replace", "strict", "ignore"):
            raise ValueError("errors must be 'replace', 'strict', or 'ignore'")
        raw = bytearray()
        special_ids = set(SPECIAL_TOKENS.values())
        for token_id in ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if token_id in special_ids:
                continue
            chunk = self._id_to_bytes.get(token_id)
            if chunk is None:
                raise ValueError(f"token ID {token_id} is outside the vocabulary (size={self.vocab_size})")
            raw.extend(chunk)
        return bytes(raw).decode("utf-8", errors=errors)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "nourai_byte_bpe_v1",
            "target_vocab_size": self.target_vocab_size,
            "actual_vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "merges": [[a, b] for a, b in self.merges],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "nourai_byte_bpe_v1":
            raise ValueError("unsupported Byte-Level BPE tokenizer format")
        merges = [(int(a), int(b)) for a, b in payload.get("merges", [])]
        return cls(merges=merges, target_vocab_size=int(payload.get("target_vocab_size", 8192)))


def benchmark_tokenizers(
    samples_by_lang: dict[str, list[str]],
    *,
    bpe_tokenizers: dict[str, ByteLevelBPETokenizer],
) -> dict[str, Any]:
    """Compare ByteTokenizer (260) against one or more ByteLevelBPETokenizer configs (e.g. 8K, 16K)."""
    byte_tok = ByteTokenizer()
    results: dict[str, Any] = {"baseline_byte_tokenizer": {}, "bpe_comparisons": {}}

    for lang, texts in samples_by_lang.items():
        t0 = time.perf_counter()
        total_chars = sum(len(t) for t in texts)
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        total_tokens = 0
        exact_roundtrip = True
        for text in texts:
            ids = byte_tok.encode(text)
            total_tokens += len(ids)
            if byte_tok.decode(ids, errors="strict") != text:
                exact_roundtrip = False
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results["baseline_byte_tokenizer"][lang] = {
            "vocab_size": byte_tok.vocab_size,
            "total_chars": total_chars,
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
            "tokens_per_char": round(total_tokens / max(1, total_chars), 4),
            "bytes_per_token": round(total_bytes / max(1, total_tokens), 4),
            "exact_roundtrip": exact_roundtrip,
            "encode_decode_ms": round(elapsed_ms, 3),
        }

    for name, bpe in bpe_tokenizers.items():
        per_lang: dict[str, Any] = {}
        for lang, texts in samples_by_lang.items():
            t0 = time.perf_counter()
            total_chars = sum(len(t) for t in texts)
            total_bytes = sum(len(t.encode("utf-8")) for t in texts)
            total_tokens = 0
            exact_roundtrip = True
            for text in texts:
                ids = bpe.encode(text)
                total_tokens += len(ids)
                if bpe.decode(ids, errors="strict") != text:
                    exact_roundtrip = False
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            base_tokens = results["baseline_byte_tokenizer"][lang]["total_tokens"]
            compression_ratio = round(base_tokens / max(1, total_tokens), 4)
            per_lang[lang] = {
                "target_vocab_size": bpe.target_vocab_size,
                "actual_vocab_size": bpe.vocab_size,
                "total_tokens": total_tokens,
                "tokens_per_char": round(total_tokens / max(1, total_chars), 4),
                "bytes_per_token": round(total_bytes / max(1, total_tokens), 4),
                "compression_vs_byte": compression_ratio,
                "effective_128ctx_chars_gain": f"{compression_ratio:.2f}x",
                "exact_roundtrip": exact_roundtrip,
                "encode_decode_ms": round(elapsed_ms, 3),
            }
        results["bpe_comparisons"][name] = per_lang

    return results

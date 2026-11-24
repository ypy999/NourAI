"""Jebrex AI — Multilingual Byte-Level BPE Tokenizer (Phase B).

Zero-dependency, deterministic, lossless UTF-8 byte-level BPE tokenizer engineered
for Arabic, English, French, mixed-script text, numbers, symbols, and code.

Token ID layout (uint16-compatible up to 65,536):
  0..255   : Raw UTF-8 bytes (guarantees 0% UNK and 100% lossless Unicode round-trip)
  256      : <|pad|>
  257      : <|bos|>
  258      : <|eos|>
  259      : <|unk|>
  260      : <|system|>
  261      : <|user|>
  262      : <|assistant|>
  263      : <|tool|>
  264..N-1 : Learned BPE merge tokens (supports 32,768 and 65,536 production presets)
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tokenizer.byte_tokenizer import ByteTokenizer

PRODUCTION_VOCAB_PRESETS: Dict[str, int] = {
    "jebrex-bpe-32k": 32768,
    "jebrex-bpe-64k": 65536,
}

SPECIAL_TOKENS: Dict[str, int] = {
    "<|pad|>": 256,
    "<|bos|>": 257,
    "<|eos|>": 258,
    "<|unk|>": 259,
    "<|system|>": 260,
    "<|user|>": 261,
    "<|assistant|>": 262,
    "<|tool|>": 263,
}
ID_TO_SPECIAL: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
FIRST_MERGE_ID = 264
MAX_UINT16_VOCAB = 65536

# Unicode-aware lossless pre-tokenizer:
# Concatenating all matched chunks reproduces the exact original string for any
# valid Unicode input (verified by fallback `.` with DOTALL).
_PRETOKEN_PATTERN = re.compile(
    r"""
      [ ]?[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+     # Arabic words (incl. diacritics)
    | [ ]?[A-Za-z\u00C0-\u024F]+                         # English & French Latin letters
    | [0-9\u0660-\u0669]{1,3}                            # Digits in 1-3 char groups (LLM arithmetic standard)
    | [ \t]+                                             # Horizontal whitespace runs
    | \r?\n+                                             # Newline runs
    | [ ]?[^\sA-Za-z\u00C0-\u024F0-9\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+  # Punctuation/symbols
    | .                                                  # Guaranteed lossless fallback
    """,
    re.VERBOSE | re.DOTALL,
)

_SPECIAL_SPLIT_PATTERN = re.compile(
    "(" + "|".join(re.escape(tok) for tok in SPECIAL_TOKENS.keys()) + ")"
)


def pretokenize(text: str) -> List[str]:
    """Split text into lossless pre-token chunks for multilingual BPE."""
    if not text:
        return []
    return _PRETOKEN_PATTERN.findall(text)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _merge_tuple(seq: Tuple[int, ...], pair: Tuple[int, int], new_id: int) -> Tuple[int, ...]:
    a, b = pair
    out: List[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


class MultilingualBPETokenizer:
    """Lossless byte-level BPE tokenizer compatible with Jebrex AI models."""

    pad_id: int = SPECIAL_TOKENS["<|pad|>"]
    bos_id: int = SPECIAL_TOKENS["<|bos|>"]
    eos_id: int = SPECIAL_TOKENS["<|eos|>"]
    unk_id: int = SPECIAL_TOKENS["<|unk|>"]
    system_id: int = SPECIAL_TOKENS["<|system|>"]
    user_id: int = SPECIAL_TOKENS["<|user|>"]
    assistant_id: int = SPECIAL_TOKENS["<|assistant|>"]
    tool_id: int = SPECIAL_TOKENS["<|tool|>"]

    def __init__(
        self,
        merges: Optional[Sequence[Tuple[int, int]]] = None,
        target_vocab_size: int = 32768,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        if target_vocab_size < FIRST_MERGE_ID or target_vocab_size > MAX_UINT16_VOCAB:
            raise ValueError(
                f"target_vocab_size must be in [{FIRST_MERGE_ID}, {MAX_UINT16_VOCAB}], got {target_vocab_size}"
            )
        self.target_vocab_size: int = int(target_vocab_size)
        self.merges: List[Tuple[int, int]] = [(int(a), int(b)) for a, b in (merges or [])]
        if FIRST_MERGE_ID + len(self.merges) > MAX_UINT16_VOCAB:
            raise ValueError("Total vocabulary exceeds uint16 limit (65,536).")
        self.metadata: Dict[str, object] = dict(metadata or {})

        self.merge_ranks: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.id_to_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for rank, (a, b) in enumerate(self.merges):
            new_id = FIRST_MERGE_ID + rank
            self.merge_ranks[(a, b)] = (rank, new_id)
            self.id_to_bytes[new_id] = self.id_to_bytes[a] + self.id_to_bytes[b]

        self.vocab_size: int = FIRST_MERGE_ID + len(self.merges)
        self._chunk_cache: Dict[str, Tuple[int, ...]] = {}

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        target_vocab_size: int = 32768,
        max_merges: Optional[int] = None,
        min_pair_frequency: int = 2,
        metadata: Optional[Dict[str, object]] = None,
    ) -> "MultilingualBPETokenizer":
        """Train byte-level BPE merges deterministically from an iterable of texts."""
        if target_vocab_size < FIRST_MERGE_ID or target_vocab_size > MAX_UINT16_VOCAB:
            raise ValueError(
                f"target_vocab_size must be in [{FIRST_MERGE_ID}, {MAX_UINT16_VOCAB}]"
            )
        budget = target_vocab_size - FIRST_MERGE_ID
        if max_merges is not None:
            budget = min(budget, max(0, int(max_merges)))

        # Step 1: Pre-tokenize and count unique UTF-8 byte chunks.
        chunk_freq_counter: Counter[Tuple[int, ...]] = Counter()
        total_chars = 0
        total_bytes = 0
        doc_count = 0
        for text in texts:
            if not text:
                continue
            doc_count += 1
            total_chars += len(text)
            total_bytes += len(text.encode("utf-8", errors="strict"))
            for chunk in pretokenize(text):
                b = tuple(chunk.encode("utf-8", errors="strict"))
                if b:
                    chunk_freq_counter[b] += 1

        seqs: List[Tuple[int, ...]] = []
        freqs: List[int] = []
        for seq, freq in sorted(chunk_freq_counter.items()):
            seqs.append(seq)
            freqs.append(freq)

        # Step 2: Build initial pair frequencies and inverted index pair -> chunk indices.
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_to_chunks: Dict[Tuple[int, int], set[int]] = defaultdict(set)
        for idx, seq in enumerate(seqs):
            if len(seq) < 2:
                continue
            f = freqs[idx]
            for j in range(len(seq) - 1):
                p = (seq[j], seq[j + 1])
                pair_counts[p] += f
                pair_to_chunks[p].add(idx)

        # Max-heap via negated count; tie-break deterministically by pair tuple.
        heap: List[Tuple[int, Tuple[int, int]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items() if cnt >= min_pair_frequency
        ]
        heapq.heapify(heap)

        merges: List[Tuple[int, int]] = []
        t0 = time.perf_counter()

        for step in range(budget):
            best_pair: Optional[Tuple[int, int]] = None
            best_count = 0
            while heap:
                neg_cnt, cand = heapq.heappop(heap)
                actual_cnt = pair_counts.get(cand, 0)
                if actual_cnt < min_pair_frequency:
                    continue
                if -neg_cnt != actual_cnt:
                    # Stale heap entry; reinsert current count if still eligible.
                    heapq.heappush(heap, (-actual_cnt, cand))
                    continue
                best_pair = cand
                best_count = actual_cnt
                break

            if best_pair is None or best_count < min_pair_frequency:
                break

            new_id = FIRST_MERGE_ID + len(merges)
            merges.append(best_pair)

            affected_indices = list(pair_to_chunks.pop(best_pair, ()))
            touched_pairs: set[Tuple[int, int]] = set()

            for idx in affected_indices:
                old_seq = seqs[idx]
                if len(old_seq) < 2:
                    continue
                new_seq = _merge_tuple(old_seq, best_pair, new_id)
                if new_seq == old_seq:
                    continue
                f = freqs[idx]

                # Subtract old pair occurrences.
                for j in range(len(old_seq) - 1):
                    op = (old_seq[j], old_seq[j + 1])
                    pair_counts[op] -= f
                    if pair_counts[op] <= 0:
                        pair_counts.pop(op, None)
                    if op != best_pair:
                        touched_pairs.add(op)

                # Update sequence and add new pair occurrences.
                seqs[idx] = new_seq
                for j in range(len(new_seq) - 1):
                    np = (new_seq[j], new_seq[j + 1])
                    pair_counts[np] += f
                    pair_to_chunks[np].add(idx)
                    touched_pairs.add(np)

            pair_counts.pop(best_pair, None)

            for tp in touched_pairs:
                cnt = pair_counts.get(tp, 0)
                if cnt >= min_pair_frequency:
                    heapq.heappush(heap, (-cnt, tp))

        elapsed_sec = round(time.perf_counter() - t0, 4)
        meta = dict(metadata or {})
        meta.update(
            {
                "trained_documents": doc_count,
                "trained_chars": total_chars,
                "trained_utf8_bytes": total_bytes,
                "unique_pretoken_chunks": len(seqs),
                "target_vocab_size": target_vocab_size,
                "actual_vocab_size": FIRST_MERGE_ID + len(merges),
                "merges_learned": len(merges),
                "min_pair_frequency": min_pair_frequency,
                "training_elapsed_sec": elapsed_sec,
                "saturation_note": (
                    "Target vocabulary reached."
                    if (FIRST_MERGE_ID + len(merges)) >= target_vocab_size
                    else (
                        "Stopped before target_vocab_size because max_merges limit or "
                        "min_pair_frequency floor was reached on the current corpus."
                    )
                ),
            }
        )
        return cls(merges=merges, target_vocab_size=target_vocab_size, metadata=meta)

    def _encode_chunk(self, chunk: str) -> Tuple[int, ...]:
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            return cached

        ids = list(chunk.encode("utf-8", errors="strict"))
        if len(ids) >= 2 and self.merge_ranks:
            while len(ids) >= 2:
                best_rank = None
                best_pair = None
                best_new_id = -1
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    entry = self.merge_ranks.get(pair)
                    if entry is not None:
                        rank, new_id = entry
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            best_new_id = new_id
                if best_pair is None:
                    break
                a, b = best_pair
                merged: List[int] = []
                i = 0
                n = len(ids)
                while i < n:
                    if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                        merged.append(best_new_id)
                        i += 2
                    else:
                        merged.append(ids[i])
                        i += 1
                ids = merged

        res = tuple(ids)
        if len(self._chunk_cache) < 65536:
            self._chunk_cache[chunk] = res
        return res

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = False,
    ) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        out: List[int] = [self.bos_id] if add_bos else []
        if text:
            if allow_special:
                parts = _SPECIAL_SPLIT_PATTERN.split(text)
            else:
                parts = [text]
            for part in parts:
                if not part:
                    continue
                if allow_special and part in SPECIAL_TOKENS:
                    out.append(SPECIAL_TOKENS[part])
                else:
                    for chunk in pretokenize(part):
                        out.extend(self._encode_chunk(chunk))
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(
        self,
        ids: Iterable[int],
        errors: str = "replace",
        skip_special_tokens: bool = True,
    ) -> str:
        if errors not in ("replace", "strict", "ignore"):
            raise ValueError("errors must be 'replace', 'strict', or 'ignore'")
        pieces: List[str] = []
        raw = bytearray()

        def _flush() -> None:
            if raw:
                pieces.append(bytes(raw).decode("utf-8", errors=errors))
                raw.clear()

        for token_id in ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if token_id in ID_TO_SPECIAL:
                if not skip_special_tokens:
                    _flush()
                    pieces.append(ID_TO_SPECIAL[token_id])
                continue
            token_bytes = self.id_to_bytes.get(token_id)
            if token_bytes is None:
                raise ValueError(f"token ID {token_id} is outside the vocabulary [0..{self.vocab_size - 1}]")
            raw.extend(token_bytes)

        _flush()
        return "".join(pieces)

    def save(self, path: Path) -> str:
        """Save tokenizer artifact JSON and return its SHA-256 hex digest."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "tokenizer_type": "jebrex_byte_bpe",
            "special_tokens": SPECIAL_TOKENS,
            "first_merge_id": FIRST_MERGE_ID,
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "metadata": self.metadata,
            "merges": [[a, b] for a, b in self.merges],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        path.write_text(serialized, encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> "MultilingualBPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("tokenizer_type") != "jebrex_byte_bpe":
            raise ValueError(f"Unsupported tokenizer_type: {data.get('tokenizer_type')}")
        merges = [(int(a), int(b)) for a, b in data.get("merges", [])]
        return cls(
            merges=merges,
            target_vocab_size=int(data.get("target_vocab_size", 32768)),
            metadata=data.get("metadata", {}),
        )


def _iter_jsonl_texts(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            lang = obj.get("language", "unknown")
            if isinstance(text, str) and text:
                rows.append((lang, text))
    return rows


def benchmark_tokenizer(
    bpe: MultilingualBPETokenizer,
    eval_rows: Sequence[Tuple[str, str]],
) -> Dict[str, object]:
    """Compare MultilingualBPETokenizer vs ByteTokenizer on held-out + stress tests."""
    byte_tok = ByteTokenizer()

    stress_cases: List[Tuple[str, str]] = [
        ("ar_diacritics", "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ، مَرْحَباً بِكُمْ فِي جِيبْرِكْس للذَّكَاءِ الاصْطِنَاعِيِّ ٢٠٢٦!"),
        ("en_prose", "Jebrex AI is an independent multilingual language model built with reproducible engineering."),
        ("fr_accents", "L'intelligence artificielle multilingue gère les accents : été, naïve, garçon, cœur, où, déjà."),
        ("mixed_tri", "مرحبا Jebrex AI! Nous construisons un modèle 100% مستقل في عام 2026."),
        ("numbers_math", "1 + 2 = 3; 3.14159265; 1,000,000 USD; x^2 + 2x - 15 = 0; 2026-11-24T22:00:00Z"),
        ("code_symbols", "def rope_embed(x: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:\n    return x * theta"),
    ]

    def _eval_bucket(texts: Sequence[str]) -> Dict[str, object]:
        total_chars = 0
        total_utf8_bytes = 0
        byte_tokens = 0
        bpe_tokens = 0
        exact_roundtrips = 0

        t0 = time.perf_counter()
        for t in texts:
            total_chars += len(t)
            total_utf8_bytes += len(t.encode("utf-8"))
            bt = byte_tok.encode(t)
            byte_tokens += len(bt)
        byte_sec = max(time.perf_counter() - t0, 1e-9)

        t1 = time.perf_counter()
        for t in texts:
            ids = bpe.encode(t)
            bpe_tokens += len(ids)
            decoded = bpe.decode(ids, errors="strict")
            if decoded == t:
                exact_roundtrips += 1
        bpe_sec = max(time.perf_counter() - t1, 1e-9)

        return {
            "samples": len(texts),
            "total_chars": total_chars,
            "total_utf8_bytes": total_utf8_bytes,
            "byte_tokenizer_tokens": byte_tokens,
            "bpe_tokenizer_tokens": bpe_tokens,
            "compression_ratio_vs_byte": round(byte_tokens / max(bpe_tokens, 1), 4),
            "token_reduction_pct": round((1.0 - (bpe_tokens / max(byte_tokens, 1))) * 100.0, 2),
            "bpe_tokens_per_char": round(bpe_tokens / max(total_chars, 1), 4),
            "bpe_bytes_per_token": round(total_utf8_bytes / max(bpe_tokens, 1), 4),
            "roundtrip_exact_match_rate": round(exact_roundtrips / max(len(texts), 1), 6),
            "byte_chars_per_sec": round(total_chars / byte_sec, 1),
            "bpe_chars_per_sec": round(total_chars / bpe_sec, 1),
        }

    by_lang: Dict[str, List[str]] = defaultdict(list)
    all_eval_texts: List[str] = []
    for lang, text in eval_rows:
        by_lang[lang].append(text)
        all_eval_texts.append(text)

    lang_results = {lang: _eval_bucket(texts) for lang, texts in sorted(by_lang.items())}
    stress_results = {name: _eval_bucket([txt]) for name, txt in stress_cases}

    return {
        "tokenizer_vocab_size": bpe.vocab_size,
        "target_vocab_size": bpe.target_vocab_size,
        "merges_learned": len(bpe.merges),
        "held_out_overall": _eval_bucket(all_eval_texts),
        "held_out_by_language": lang_results,
        "stress_suite": stress_results,
        "production_vocab_presets_supported": PRODUCTION_VOCAB_PRESETS,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train and benchmark Jebrex Multilingual BPE Tokenizer.")
    parser.add_argument("--train", type=Path, default=Path("data/processed/jebrex_v1_clean/splits/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/processed/jebrex_v1_clean/splits/test.jsonl"))
    parser.add_argument("--target-vocab", type=int, default=32768)
    parser.add_argument("--max-merges", type=int, default=2048)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("tokenizer/artifacts/jebrex_bpe_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/production_execution/tokenizer_benchmark.json"))
    args = parser.parse_args(argv)

    train_rows = _iter_jsonl_texts(args.train)
    eval_rows = _iter_jsonl_texts(args.eval)
    train_sha = _sha256_file(args.train)
    eval_sha = _sha256_file(args.eval)

    bpe = MultilingualBPETokenizer.train(
        texts=(t for _, t in train_rows),
        target_vocab_size=args.target_vocab,
        max_merges=args.max_merges,
        min_pair_frequency=args.min_freq,
        metadata={
            "train_split_path": str(args.train).replace("\\", "/"),
            "train_split_sha256": train_sha,
            "trained_exclusively_on_train_split": True,
        },
    )
    artifact_sha = bpe.save(args.out)

    # Reload from disk to verify persistence fidelity before benchmarking.
    reloaded = MultilingualBPETokenizer.load(args.out)
    report = benchmark_tokenizer(reloaded, eval_rows)
    report["artifact_path"] = str(args.out).replace("\\", "/")
    report["artifact_sha256"] = artifact_sha
    report["artifact_size_bytes"] = args.out.stat().st_size
    report["train_split_sha256"] = train_sha
    report["eval_split_path"] = str(args.eval).replace("\\", "/")
    report["eval_split_sha256"] = eval_sha
    report["training_metadata"] = reloaded.metadata

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "vocab_size": reloaded.vocab_size,
                "merges_learned": len(reloaded.merges),
                "artifact_size_bytes": report["artifact_size_bytes"],
                "held_out_overall": report["held_out_overall"],
                "held_out_by_language": {
                    k: {
                        "compression_ratio_vs_byte": v["compression_ratio_vs_byte"],
                        "token_reduction_pct": v["token_reduction_pct"],
                        "roundtrip_exact_match_rate": v["roundtrip_exact_match_rate"],
                    }
                    for k, v in report["held_out_by_language"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

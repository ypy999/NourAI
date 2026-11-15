"""Disk-backed fixed-length next-token dataset."""
from array import array
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset

from configs.configuration import ProjectConfig, load_config
from tokenizer.byte_tokenizer import ByteTokenizer
from data.provenance import canonical_record, PROVENANCE_FIELDS
from data.source_registry import SourceRegistry


class TokenSequenceDataset(Dataset):
    def __init__(self, jsonl_path, config=None, tokenizer=None, *, cache_dir=None, preload_tokens=False):
        self.path = Path(jsonl_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"dataset file not found: {self.path}")
        self.config = config if isinstance(config, ProjectConfig) else load_config(config, tokenizer)
        self.tokenizer = tokenizer or ByteTokenizer()
        self.config.validate(self.tokenizer)
        self.preload_tokens = bool(preload_tokens)
        self._preloaded_u16 = None
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._temp = tempfile.TemporaryDirectory(prefix="nourai-token-cache-", dir=cache_dir)
        self._token_path = Path(self._temp.name) / "tokens.u16"
        self._metadata_path = Path(self._temp.name) / "provenance.sqlite3"
        source_registry = SourceRegistry()
        token_count = 0
        metadata_db = sqlite3.connect(str(self._metadata_path))
        metadata_db.execute("CREATE TABLE sequence_records(token_start INTEGER NOT NULL, token_end INTEGER NOT NULL, record_id TEXT, metadata_json TEXT NOT NULL)")
        metadata_db.execute("CREATE INDEX idx_sequence_record_range ON sequence_records(token_start,token_end)")
        try:
            with self.path.open("r", encoding="utf-8", errors="strict") as source, self._token_path.open("wb") as out:
                for line_no, line in enumerate(source, 1):
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSONL at {self.path}:{line_no}: {exc}") from exc
                    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                        raise ValueError(f"expected a JSON object with string 'text' at {self.path}:{line_no}")
                    provenance = canonical_record(item, source_file=self.path, source_line=line_no,
                                                  source_registry=source_registry)
                    tokens = self.tokenizer.encode(item["text"], add_eos=True)
                    if not tokens or min(tokens) < 0 or max(tokens) >= self.config.vocab_size or max(tokens) > 65535:
                        raise ValueError("tokenizer returned IDs outside supported vocabulary/uint16 range")
                    if sys.byteorder == "little":
                        out.write(array("H", tokens).tobytes())
                    else:
                        out.write(bytearray().join(int(token).to_bytes(2, "little") for token in tokens))
                    metadata = {name: provenance.get(name) for name in PROVENANCE_FIELDS if name not in ("text", "raw_text")}
                    metadata.update({"provenance_missing": provenance["provenance_missing"],
                                     "record_id_method": provenance.get("record_id_method"),
                                     "detected_language_method": provenance.get("detected_language_method")})
                    metadata_db.execute("INSERT INTO sequence_records VALUES(?,?,?,?)",
                                        (token_count, token_count + len(tokens), provenance["record_id"],
                                         json.dumps(metadata, ensure_ascii=False, sort_keys=True)))
                    token_count += len(tokens)
                    if line_no % 4096 == 0:
                        metadata_db.commit()
            metadata_db.commit()
        finally:
            metadata_db.close()
        self.token_count = token_count
        self.sequence_length = self.config.context_length
        self._sample_count = max(0, (token_count - 1) // self.sequence_length)
        if self._sample_count == 0:
            self._temp.cleanup()
            raise ValueError(f"dataset needs at least {self.sequence_length + 1} tokens for one training example")
        self._token_count = token_count
        if self.preload_tokens:
            raw_all = self._token_path.read_bytes()
            self._preloaded_u16 = torch.frombuffer(bytearray(raw_all), dtype=torch.uint16)

    def __len__(self):
        return self._sample_count

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError("dataset index out of range")
        start = index * self.sequence_length
        if self._preloaded_u16 is not None:
            chunk = self._preloaded_u16[start : start + self.sequence_length + 1].to(torch.int64)
            return chunk[:-1], chunk[1:]
        with self._token_path.open("rb") as tokens_file:
            tokens_file.seek(start * 2)
            raw = tokens_file.read((self.sequence_length + 1) * 2)
        if len(raw) % 2:
            raise RuntimeError("corrupt token cache: odd byte count")
        chunk = torch.frombuffer(bytearray(raw), dtype=torch.uint16).to(torch.int64)
        if chunk.numel() != self.sequence_length + 1:
            raise RuntimeError("corrupt token cache: incomplete sequence")
        return chunk[:-1], chunk[1:]

    def metadata_for_sequence(self, index):
        """Return provenance for every source record overlapping a token window.

        The established `(input_ids, target_ids)` return interface is unchanged.
        Text is represented by token IDs; original text remains in the source
        JSONL and is not duplicated into the temporary metadata index.
        """
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError("dataset index out of range")
        start = index * self.sequence_length
        end = start + self.sequence_length + 1
        db = sqlite3.connect(str(self._metadata_path))
        try:
            rows = db.execute("SELECT record_id,metadata_json FROM sequence_records "
                              "WHERE token_end>? AND token_start<? ORDER BY token_start", (start, end)).fetchall()
        finally:
            db.close()
        return [json.loads(payload) for _, payload in rows]

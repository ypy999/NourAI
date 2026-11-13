"""Streaming corpus statistics for the existing ByteTokenizer."""
from dataclasses import asdict, dataclass, field

from tokenizer.byte_tokenizer import ByteTokenizer
from .language_analysis import classify_language


BYTE_TOKENIZER_VERSION = "byte-tokenizer-utf8-v1"


@dataclass
class TokenizerLanguageStats:
    records: int = 0
    characters: int = 0
    tokens: int = 0
    unique_token_ids: int = 0
    average_tokens_per_record: float = 0.0
    tokens_per_character: float | None = None
    sequence_length_buckets: dict = field(default_factory=lambda: {"0-127": 0, "128-511": 0, "512-2047": 0, "2048+": 0})
    records_over_context: int = 0

    def as_dict(self):
        return asdict(self)


def analyze_tokenizer_records(records, tokenizer=None, *, context_length=128, chunk_characters=65536, add_eos=True):
    """Analyze records in one pass, encoding at most `chunk_characters` at once.

    The UTF-8 byte tokenizer is lossless; statistics are byte-token counts plus
    EOS when requested. This function measures storage and sequence lengths,
    not linguistic segmentation quality.
    """
    tokenizer = tokenizer or ByteTokenizer()
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if isinstance(chunk_characters, bool) or not isinstance(chunk_characters, int) or chunk_characters <= 0:
        raise ValueError("chunk_characters must be a positive integer")
    totals = {}
    total_records = total_chars = total_tokens = utf8_bytes = 0
    all_token_ids = set()
    for item in records:
        text = item if isinstance(item, str) else item.get("text") if isinstance(item, dict) else None
        if not isinstance(text, str):
            raise ValueError("each record must be a string or mapping with string 'text'")
        language = classify_language(text)
        stats = totals.setdefault(language, {"records": 0, "characters": 0, "tokens": 0,
            "token_ids": set(), "length_buckets": {"0-127": 0, "128-511": 0, "512-2047": 0, "2048+": 0},
            "over_context": 0})
        token_count = 0
        for offset in range(0, len(text), chunk_characters):
            chunk_ids = tokenizer.encode(text[offset:offset + chunk_characters])
            if any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < tokenizer.vocab_size for token in chunk_ids):
                raise ValueError("tokenizer returned an ID outside its vocabulary")
            token_count += len(chunk_ids)
            stats["token_ids"].update(chunk_ids)
            all_token_ids.update(chunk_ids)
        if add_eos:
            token_count += 1
            stats["token_ids"].add(tokenizer.eos_id)
            all_token_ids.add(tokenizer.eos_id)
        length = len(text)
        bucket = "0-127" if token_count < 128 else "128-511" if token_count < 512 else "512-2047" if token_count < 2048 else "2048+"
        stats["records"] += 1
        stats["characters"] += length
        stats["tokens"] += token_count
        stats["length_buckets"][bucket] += 1
        stats["over_context"] += token_count > context_length
        total_records += 1
        total_chars += length
        total_tokens += token_count
        utf8_bytes += len(text.encode("utf-8", errors="strict"))
    by_language = {}
    for language, item in totals.items():
        count = item["records"]
        chars = item["characters"]
        tokens = item["tokens"]
        by_language[language] = TokenizerLanguageStats(
            records=count, characters=chars, tokens=tokens,
            unique_token_ids=len(item["token_ids"]),
            average_tokens_per_record=tokens / count if count else 0.0,
            tokens_per_character=tokens / chars if chars else None,
            sequence_length_buckets=item["length_buckets"],
            records_over_context=item["over_context"],
        ).as_dict()
    return {
        "tokenizer": type(tokenizer).__name__,
        "tokenizer_version": BYTE_TOKENIZER_VERSION if isinstance(tokenizer, ByteTokenizer) else "custom-unspecified",
        "vocab_size": tokenizer.vocab_size,
        "records": total_records,
        "characters": total_chars,
        "tokens": total_tokens,
        "unique_token_ids": len(all_token_ids),
        "average_tokens_per_record": total_tokens / total_records if total_records else 0.0,
        "tokens_per_character": total_tokens / total_chars if total_chars else None,
        "utf8_source_bytes": utf8_bytes,
        "token_cache_bytes_uint16_estimate": total_tokens * 2,
        "context_length": context_length,
        "records_over_context": sum(item["over_context"] for item in totals.values()),
        "sequence_length_buckets": {bucket: sum(item["length_buckets"][bucket] for item in totals.values())
                                     for bucket in ("0-127", "128-511", "512-2047", "2048+")},
        "by_language": by_language,
        "note": "Byte counts and context overflow only; this is not a tokenizer-quality or language-understanding score.",
    }

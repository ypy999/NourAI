"""Bounded-memory synthetic throughput benchmark; never writes corpus data."""
import json
import ctypes
import tempfile
import time
from pathlib import Path

from data.domain_analysis import classify_domains
from data.language_analysis import classify_language
from data.quality import score_record
from data.splitting import split_records
from tokenizer.byte_tokenizer import ByteTokenizer


def _working_set_bytes():
    try:
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                        ("peak", ctypes.c_size_t), ("working", ctypes.c_size_t),
                        ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
                        ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
                        ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
        value = Counters(); value.cb = ctypes.sizeof(value)
        memory_api = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_api.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        memory_api.restype = ctypes.c_int
        if memory_api(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(value), value.cb):
            return {"working_set_bytes": value.working, "peak_working_set_bytes": value.peak}
    except (AttributeError, OSError):
        return None
    return None


def _synthetic_records(count):
    for index in range(count):
        # Synthetic records are explicitly marked and never enter the corpus.
        if index % 3 == 0:
            text = f"Synthetic benchmark Arabic record {index}: مرحباً بالعالم." 
        elif index % 3 == 1:
            text = f"Synthetic benchmark French record {index}: Bonjour le monde." 
        else:
            text = f"Synthetic benchmark English record {index}: science and technology." 
        yield {"text": text, "source": "SYNTHETIC_BENCHMARK_ONLY", "line_number": index + 1}


def run_scaling_benchmark(scales=(1_000, 10_000, 100_000), *, include_million=False):
    """Measure local CPU analysis and SQLite dedup/splitting at safe scales."""
    sizes = list(scales)
    if include_million:
        sizes.append(1_000_000)
    tokenizer = ByteTokenizer()
    results = []
    for count in sizes:
        memory_before = _working_set_bytes()
        start_analysis = time.perf_counter()
        token_count = 0
        for record in _synthetic_records(count):
            text = record["text"]
            # ByteTokenizer emits one ID per UTF-8 byte plus EOS; this exact
            # arithmetic avoids allocating an ID list during the benchmark.
            token_count += len(text.encode("utf-8", errors="strict")) + 1
            classify_language(text)
            classify_domains(text)
            score_record(text)
        analysis_seconds = time.perf_counter() - start_analysis
        memory_after_analysis = _working_set_bytes()
        with tempfile.TemporaryDirectory(prefix="nourai-phase9-benchmark-") as temporary:
            output_dir = Path(temporary) / "splits"
            start_split = time.perf_counter()
            split_result = split_records(_synthetic_records(count), output_dir, seed=1337, stratify=True,
                                         checkpoint_batch_size=4096)
            split_seconds = time.perf_counter() - start_split
            split_bytes = sum(path.stat().st_size for path in output_dir.glob("*.jsonl"))
        results.append({"records": count, "dataset_kind": "synthetic_benchmark_only",
                        "analysis_seconds": analysis_seconds,
                        "analysis_records_per_second": count / analysis_seconds if analysis_seconds else None,
                        "tokens_including_eos": token_count,
                        "analysis_tokens_per_second": token_count / analysis_seconds if analysis_seconds else None,
                        "split_and_deduplicate_seconds": split_seconds,
                        "split_records_per_second": count / split_seconds if split_seconds else None,
                        "deduplicated_records": split_result.unique_records,
                        "duplicate_records_removed": split_result.duplicate_records,
                        "split_counts": split_result.split_counts,
                        "output_bytes": split_bytes,
                        "process_memory_before_bytes": memory_before,
                        "process_memory_after_analysis_bytes": memory_after_analysis,
                        "split_leakage_by_construction": "split_records globally deduplicates before assignment; synthetic output not treated as corpus"})
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware": "CPU only; current host; no CUDA calls",
            "measurement_scope": "synthetic generation excluded; analysis includes tokenizer/language/domain/quality passes; splitting includes SQLite deduplication and output writing",
            "memory_metric": "Windows process working set/peak working set; None if unavailable",
            "results": results,
            "one_million_run": "executed" if include_million else "not run; default safe scale ceiling is 100,000; no one-million result is claimed"}


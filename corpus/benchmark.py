"""Bounded repeatable streaming benchmark for corpus JSONL files."""
import json
from pathlib import Path
import time

from data.reporting import memory_snapshot
from .importers import iter_source_records


def benchmark_jsonl(path, *, checkpoints=(1_000, 10_000, 100_000, 1_000_000)):
    """Measure sequential UTF-8/JSON parse throughput on a real imported file."""
    path = Path(path)
    requested = sorted(set(int(value) for value in checkpoints if int(value) > 0))
    reached = []
    start = time.perf_counter()
    bytes_read = 0
    records = 0
    invalid = 0
    next_index = 0
    with path.open("rb") as stream:
        for raw in stream:
            bytes_read += len(raw)
            try:
                row = json.loads(raw.decode("utf-8", errors="strict"))
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    raise ValueError("JSONL record requires a string text field")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                invalid += 1
            else:
                records += 1
            while next_index < len(requested) and records >= requested[next_index]:
                elapsed = time.perf_counter() - start
                reached.append({"records": requested[next_index], "elapsed_seconds": elapsed,
                                "records_per_second": requested[next_index] / elapsed if elapsed else None,
                                "bytes_read": bytes_read})
                next_index += 1
    elapsed = time.perf_counter() - start
    return {
        "path": str(path), "data_origin": "actual imported corpus file",
        "records_read": records, "invalid_records": invalid, "bytes_read": bytes_read,
        "elapsed_seconds": elapsed, "records_per_second": records / elapsed if elapsed else None,
        "requested_checkpoints": requested, "checkpoint_results": reached,
        "unreached_checkpoints": requested[next_index:],
        "memory_snapshot": memory_snapshot(),
        "notes": "Sequential JSONL disk-read/parse only; not model inference, training, or end-to-end language performance.",
    }


def benchmark_source_records(path, *, checkpoints=(1_000, 10_000, 100_000, 1_000_000)):
    """Measure record parsing from an acquired source archive without extraction."""
    path = Path(path)
    requested = sorted(set(int(value) for value in checkpoints if int(value) > 0))
    reached = []
    start = time.perf_counter()
    records = 0
    errors = 0
    next_index = 0
    for item in iter_source_records(path):
        if "error" in item:
            errors += 1
            continue
        if not isinstance(item.get("text"), str):
            continue
        records += 1
        while next_index < len(requested) and records >= requested[next_index]:
            elapsed = time.perf_counter() - start
            reached.append({"records": requested[next_index], "elapsed_seconds": elapsed,
                            "records_per_second": requested[next_index] / elapsed if elapsed else None})
            next_index += 1
    elapsed = time.perf_counter() - start
    return {"path": str(path), "data_origin": "actual acquired source archive", "records_read": records,
            "errors": errors, "elapsed_seconds": elapsed,
            "records_per_second": records / elapsed if elapsed else None,
            "requested_checkpoints": requested, "checkpoint_results": reached,
            "unreached_checkpoints": requested[next_index:], "memory_snapshot": memory_snapshot(),
            "notes": "Sequential archive decoding and record parsing only; no extraction, model execution, or training."}

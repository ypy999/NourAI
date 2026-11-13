# NourAI Dataset Engineering Report

- Created: 2026-11-11T20:21:45.047051+01:00
- Dataset: None
- Records: 0
- Corpus present: False
- Training eligible: False

## Provenance

- Source: null
- License: null

## Results

### Language distribution

```json
{
  "actual_dataset": null,
  "synthetic_test_fixture": {
    "records": 4,
    "total_characters": 102,
    "records_by_language": {
      "arabic": 1,
      "english": 1,
      "french": 1,
      "mixed": 1,
      "unclassified": 0
    },
    "characters_by_language": {
      "arabic": 27,
      "english": 25,
      "french": 26,
      "other": 24
    },
    "average_record_characters": 25.5,
    "minimum_record_characters": 19,
    "maximum_record_characters": 30,
    "length_buckets": {
      "0-99": 4,
      "100-499": 0,
      "500-1999": 0,
      "2000+": 0
    },
    "percentages": {
      "arabic": 25.0,
      "english": 25.0,
      "french": 25.0,
      "mixed": 25.0,
      "unclassified": 0.0
    }
  }
}
```

### Cleaning statistics

```json
{
  "actual_dataset": null,
  "checks": "UTF-8, NFC/NFD, Arabic diacritics, punctuation, control characters, whitespace, paragraph mode, short-record policy"
}
```

### Duplicate statistics

```json
{
  "actual_dataset": null,
  "synthetic_fixture_records": 60,
  "synthetic_fixture_duplicates": 2,
  "exact_duplicates": 1,
  "normalized_duplicates": 1,
  "profile": "NFC + casefold + whitespace collapse; punctuation/diacritics retained"
}
```

### Quality statistics

```json
{
  "actual_dataset": null,
  "synthetic_test_fixture": {
    "records": 4,
    "mean_record_score": 100.0,
    "language_balance_score": 100.0,
    "overall_dataset_score": 100.0,
    "scores_below_50": 0,
    "records_by_language": {
      "arabic": 1,
      "english": 1,
      "french": 1,
      "mixed": 1,
      "unclassified": 0
    },
    "score_formula": "record score = sum(component_percent * weight); weights: utf8_encoding=0.15, completeness=0.15, uniqueness=0.20, length=0.15, unicode_integrity=0.20, formatting=0.15",
    "dataset_formula": "overall = 0.90 * mean_record_score + 0.10 * language_balance_score; if no target-language records, overall = mean_record_score"
  }
}
```

### Tokenizer statistics

```json
{
  "actual_dataset": null,
  "synthetic_test_fixture": {
    "tokenizer": "ByteTokenizer",
    "tokenizer_version": "byte-tokenizer-utf8-v1",
    "vocab_size": 260,
    "records": 4,
    "characters": 102,
    "tokens": 134,
    "unique_token_ids": 39,
    "average_tokens_per_record": 33.5,
    "tokens_per_character": 1.3137254901960784,
    "utf8_source_bytes": 130,
    "token_cache_bytes_uint16_estimate": 268,
    "context_length": 128,
    "records_over_context": 0,
    "sequence_length_buckets": {
      "0-127": 4,
      "128-511": 0,
      "512-2047": 0,
      "2048+": 0
    },
    "by_language": {
      "arabic": {
        "records": 1,
        "characters": 23,
        "tokens": 44,
        "unique_token_ids": 17,
        "average_tokens_per_record": 44.0,
        "tokens_per_character": 1.9130434782608696,
        "sequence_length_buckets": {
          "0-127": 1,
          "128-511": 0,
          "512-2047": 0,
          "2048+": 0
        },
        "records_over_context": 0
      },
      "english": {
        "records": 1,
        "characters": 30,
        "tokens": 31,
        "unique_token_ids": 14,
        "average_tokens_per_record": 31.0,
        "tokens_per_character": 1.0333333333333334,
        "sequence_length_buckets": {
          "0-127": 1,
          "128-511": 0,
          "512-2047": 0,
          "2048+": 0
        },
        "records_over_context": 0
      },
      "french": {
        "records": 1,
        "characters": 30,
        "tokens": 32,
        "unique_token_ids": 19,
        "average_tokens_per_record": 32.0,
        "tokens_per_character": 1.0666666666666667,
        "sequence_length_buckets": {
          "0-127": 1,
          "128-511": 0,
          "512-2047": 0,
          "2048+": 0
        },
        "records_over_context": 0
      },
      "mixed": {
        "records": 1,
        "characters": 19,
        "tokens": 27,
        "unique_token_ids": 20,
        "average_tokens_per_record": 27.0,
        "tokens_per_character": 1.4210526315789473,
        "sequence_length_buckets": {
          "0-127": 1,
          "128-511": 0,
          "512-2047": 0,
          "2048+": 0
        },
        "records_over_context": 0
      }
    },
    "note": "Byte counts and context overflow only; this is not a tokenizer-quality or language-understanding score."
  }
}
```

### Train/validation/test splits

```json
{
  "actual_dataset": null,
  "synthetic_stream_benchmark": {
    "input_records": 4000,
    "empty_records": 0,
    "duplicate_records": 0,
    "exact_duplicate_records": 0,
    "normalized_duplicate_records": 0,
    "unique_records": 4000,
    "split_counts": {
      "train": 3600,
      "validation": 200,
      "test": 200
    },
    "language_counts": {
      "train": {
        "english": 3600
      },
      "validation": {
        "english": 200
      },
      "test": {
        "english": 200
      }
    },
    "requested_ratios": {
      "train": 0.9,
      "validation": 0.05,
      "test": 0.05
    },
    "actual_ratios": {
      "train": 0.9,
      "validation": 0.05,
      "test": 0.05
    },
    "seed": 19,
    "warnings": [],
    "processing_seconds": 2.164585999999872,
    "source_import_stats": {},
    "source_estimate": {}
  }
}
```

### Storage

```json
{
  "raw_inputs_modified": false,
  "actual_dataset_files": 0,
  "D_free_bytes": 35721166848,
  "D_free_GiB": 33.27,
  "RAM_total_bytes": 8475860992,
  "RAM_available_bytes": 4172759040,
  "process_working_set_bytes": null,
  "process_peak_working_set_bytes": null,
  "python_tracemalloc_peak_bytes_for_4000_synthetic_records": 451856,
  "tracemalloc_excludes_SQLite_native_and_OS_file_cache": true,
  "cpu": "Intel64 Family 6 Model 142 Stepping 12, GenuineIntel",
  "cuda_used": false
}
```

### Memory measurements

```json
{
  "total_ram_bytes": 8475860992,
  "available_ram_bytes": 4172759040,
  "process_working_set_bytes": null,
  "process_peak_working_set_bytes": null
}
```

### Test results

```json
{
  "command": ".venv\\Scripts\\python.exe -m pytest -p no:cacheprovider -q",
  "passed": 107,
  "failed": 0,
  "skipped": 0,
  "duration_seconds": 10.66,
  "by_phase": {
    "phase3": 28,
    "phase4": 15,
    "phase5": 21,
    "phase6": 19,
    "phase7": 24
  },
  "synthetic_split_benchmark": {
    "records": 4000,
    "splits": {
      "train": 3600,
      "validation": 200,
      "test": 200
    },
    "seconds": 2.166199200000392,
    "records_per_second": 1846.55,
    "python_tracemalloc_peak_bytes": 451856
  },
  "python": "3.15.2 (tags/v3.15.2:5e58eaa, Oct 25 2026, 17:27:03) [MSC v.1951 64 bit (AMD64)]",
  "pytorch": "2.15.0+cpu",
  "numpy": "2.5.4",
  "pytest": "9.2.0"
}
```

## Processing time

2.166199200000392

## Warnings and errors

- Warning: No real dataset was present or processed.
- Warning: Language labels and quality scores are heuristics, not semantic quality measures.
- Warning: All sample statistics and throughput below use synthetic test fixtures only.
- No errors recorded

## Readiness blockers

- source is missing or unverified
- license is missing or unverified
- dataset contains no accepted records
- train/validation/test split statistics are missing

## Limitations

- Heuristic language labels are approximate.
- Quality scores measure format/integrity signals, not factual accuracy or educational value.
- Tokenizer analysis measures byte counts, not linguistic segmentation quality.

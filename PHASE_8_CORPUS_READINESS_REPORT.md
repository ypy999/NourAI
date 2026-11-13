# NourAI — Phase 8 Corpus Acquisition & Dataset Readiness

Created: 2026-11-11T20:49:29.076722+00:00

Readiness: **CONDITIONALLY_READY**

## Provenance and license

Only an explicitly CC0-labeled source is eligible for automatic ingestion. This status records technical provenance; it is not legal advice.

- `tatoeba_cc0` — VERIFIED — CC0-1.0

### Registered source decisions

- `tatoeba_cc0` — VERIFIED — CC0-1.0 — https://tatoeba.org/en/downloads
- `tatoeba_general` — NEEDS_REVIEW — CC-BY-2.0-FR — https://tatoeba.org/en/terms_of_use
- `wikimedia` — NEEDS_REVIEW — CC-BY-SA-4.0/GFDL (project and revision dependent) — https://meta.wikimedia.org/wiki/Terms_of_Use
- `oscar` — UNKNOWN_LICENSE — Underlying page licenses vary / are not established by packaging license — https://huggingface.co/datasets/oscar

## Corpus results

```json
{
  "downloads": [
    {
      "source_id": "tatoeba_cc0",
      "status": "already_present_verified",
      "bytes": 7963537,
      "sha256": "7e4251d06edd949db9e47cc360b472873da6c054af5dc4b9d2c9e06d024cd710",
      "publisher_checksum_available": false,
      "url": "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2"
    }
  ],
  "imports": [
    {
      "source_id": "tatoeba_cc0",
      "records_seen": 562272,
      "accepted_records": 55719,
      "filtered_outside_target_languages": 506553,
      "empty_records": 0,
      "parse_errors": 0,
      "languages": {
        "eng": 41515,
        "fra": 14202,
        "ara": 2
      },
      "processing_seconds": 8.055174199999783,
      "output_file": "data\\processed\\phase8\\imported\\tatoeba_cc0_multilingual_v4.jsonl",
      "output_bytes": 52065889,
      "output_sha256": "00f752ca1b78d27fc717a8664bb2c7ebd0f001b2ace11685e1d36456ebb1841a",
      "normalization": {
        "records_seen": 55719,
        "accepted_records": 55719,
        "rejected_records": 0,
        "errors": [],
        "output_bytes": 81303011,
        "normalization_form": "NFC",
        "processing_seconds": 4.9864741,
        "measurement": "timed real-data normalization run into an automatically cleaned temporary folder"
      }
    }
  ],
  "splits": {
    "train": 50144,
    "validation": 2787,
    "test": 2785
  },
  "validation": {
    "deduplicated_splits": {
      "valid": true,
      "errors": [],
      "split_counts": {
        "train": 50144,
        "validation": 2787,
        "test": 2785
      },
      "duplicate_leakage": 0,
      "duplicates_within_splits": 0,
      "normalized_duplicates_removed_before_split": 3,
      "split_ratios": {
        "train": 0.9,
        "validation": 0.05,
        "test": 0.05
      },
      "random_seed": 1337,
      "processing_seconds": 25.9765702
    },
    "pre_dedup_import": {
      "records": 55719,
      "normalized_duplicates": 3,
      "valid_before_dedup": false,
      "resolved_by_deduplication": true,
      "unknown_source_records": 0,
      "unknown_license_records": 0,
      "suspicious_records": 0
    }
  },
  "quality": {
    "record_count_scored": 55719,
    "record_count_total": 55719,
    "characters": 4263825,
    "byte_tokenizer_ids_including_eos": 4361342,
    "language_counts_heuristic": {
      "english": 33141,
      "unclassified": 8790,
      "french": 13014,
      "mixed": 772,
      "arabic": 2
    },
    "source_language_metadata_counts": {
      "eng": 41515,
      "fra": 14202,
      "ara": 2
    },
    "language_confidence_mean_by_language": {
      "English": 0.99,
      "French": 0.99,
      "Arabic": 0.99
    },
    "language_classification_methods": {
      "source_language_metadata": 55719
    },
    "mean_record_score": 99.9989,
    "language_balance_score": 54.182,
    "overall_quality_score": 95.4172,
    "scores_below_50": 0,
    "score_formula": "record = 0.15 UTF-8 + 0.15 completeness + 0.20 uniqueness + 0.15 length + 0.20 Unicode integrity + 0.15 formatting; dataset = 0.90 mean record + 0.10 normalized target-language balance",
    "score_interpretation": "format/integrity heuristic only; not semantic, factual, educational quality or language understanding"
  }
}
```

## Performance

```json
{
  "source_archive_parse": {
    "records_read": 562272,
    "errors": 0,
    "elapsed_seconds": 3.3542853,
    "records_per_second": 167627.9594,
    "reached_checkpoints": [
      1000,
      10000,
      100000
    ],
    "unreached_checkpoints": [
      1000000
    ],
    "memory_snapshot": {
      "total_ram_bytes": 8475860992,
      "available_ram_bytes": 4745494528,
      "process_working_set_bytes": null,
      "process_peak_working_set_bytes": null
    },
    "scope": "archive decode and record parsing only"
  },
  "imported_jsonl_parse": {
    "records_read": 55719,
    "invalid_records": 0,
    "bytes_read": 52065889,
    "elapsed_seconds": 0.6786277,
    "records_per_second": 82105.4018,
    "reached_checkpoints": [
      1000,
      10000
    ],
    "unreached_checkpoints": [
      100000,
      1000000
    ],
    "scope": "JSONL disk read and parse only"
  },
  "normalization": {
    "records_seen": 55719,
    "accepted_records": 55719,
    "rejected_records": 0,
    "errors": [],
    "output_bytes": 81303011,
    "normalization_form": "NFC",
    "processing_seconds": 4.9864741,
    "measurement": "timed real-data normalization run into an automatically cleaned temporary folder"
  },
  "import_processing_seconds": 8.055174199999783,
  "phase7_dedup_split_seconds": 25.9765702,
  "quality_statistics_processing_seconds": 24.6406541,
  "metadata_database_bytes": 66916352,
  "metadata_catalog": {
    "sources": 1,
    "datasets": 2,
    "files": 6,
    "records_indexed": 111435,
    "by_license": {
      "CC0-1.0": 111435
    },
    "by_status": {
      "IMPORTED": 55719,
      "SPLIT": 55716
    }
  },
  "file_bytes": {
    "source_archive": 7963537,
    "imported_jsonl": 52065889,
    "normalized_jsonl": 81303011,
    "train": 11397598,
    "validation": 628496,
    "test": 630635
  },
  "ram_snapshot": {
    "total_ram_bytes": 8475860992,
    "available_ram_bytes": 4745494528,
    "process_working_set_bytes": null,
    "process_peak_working_set_bytes": null
  },
  "scope": "data I/O, parsing and cleaning only; no model inference, training or CUDA",
  "report_generation_seconds_first_pass": 0.0037986999996064696
}
```

## Quality score formula

Record score = 0.15×UTF-8 + 0.15×completeness + 0.20×uniqueness + 0.15×length + 0.20×Unicode integrity + 0.15×formatting. Dataset score = 0.90×mean record score + 0.10×normalized language-balance score (or mean score if no target-language records). Each component is a format/integrity heuristic; this does not assess factual correctness, educational value, or language understanding.

## Readiness blockers

- Only 2 Arabic records; neither validation nor test contains Arabic.
- The selected dataset is sentence-level and not a representative pretraining corpus.
- No training or trained-model evaluation was performed.

## Actual model-training decision: NOT_READY_FOR_REAL_MULTILINGUAL_LLM_TRAINING

- Only 2 Arabic records; neither validation nor test contains Arabic.
- The selected dataset is sentence-level and not a representative pretraining corpus.
- No training or trained-model evaluation was performed.

## Errors and warnings

- None
- Warning: Three normalized duplicate records were detected before split and removed; final splits have no duplicates/leakage.
- Warning: Only two Arabic records exist; Arabic validation and test sets are empty.
- Warning: Language labels are source metadata, not independently verified language identification.
- Warning: A prior failed empty generic-import artifact remains preserved with FAILED_DO_NOT_USE status and was excluded.

## Scope

No model training or CUDA work was performed. Synthetic fixtures used for automated tests are not real corpus data. The readiness report does not claim language understanding.


## Implementation files

- Created: `corpus/__init__.py`
- Created: `corpus/__main__.py`
- Created: `corpus/acquisition.py`
- Created: `corpus/benchmark.py`
- Created: `corpus/importers.py`
- Created: `corpus/licensing.py`
- Created: `corpus/pipeline.py`
- Created: `corpus/readiness.py`
- Created: `corpus/reporting.py`
- Created: `corpus/storage.py`
- Created: `corpus/validation.py`
- Created: `data/source_registry.py`
- Created: `data/sources.py`
- Created: `data/license_manager.py`
- Created: `data/license_policy.py`
- Created: `data/acquisition.py`
- Created: `data/download_manager.py`
- Created: `data/importers.py`
- Created: `data/checksum.py`
- Created: `data/raw_storage.py`
- Created: `data/normalization.py`
- Created: `data/corpus_pipeline.py`
- Created: `data/multilingual.py`
- Created: `data/storage_manager.py`
- Created: `data/validation.py`
- Created: `data/corpus_dashboard.py`
- Created: `data/readiness.py`
- Created: `nourai/__init__.py`
- Created: `nourai/__main__.py`
- Created: `configs/dataset_sources.json`
- Created: `configs/phase8_corpus.json`
- Created: `tests/test_phase8_corpus.py`
- Created: `tests/test_phase8_sources.py`
- Created: `tests/test_phase8_licenses.py`
- Created: `tests/test_phase8_acquisition.py`
- Created: `tests/test_phase8_raw_data.py`
- Created: `tests/test_phase8_pipeline.py`
- Created: `tests/test_phase8_multilingual.py`
- Created: `tests/test_phase8_storage.py`
- Created: `tests/test_phase8_validation.py`
- Created: `tests/test_phase8_dashboard.py`
- Created: `tests/test_phase8_readiness.py`
- Created: `data/approved/tatoeba_cc0_phase8_final.manifest.json`
- Created: `reports/phase8_corpus_readiness.json`
- Created: `reports/phase8_corpus_readiness.md`
- Created: `reports/phase8_corpus_dashboard.html`
- Created: `PHASE_8_CORPUS_READINESS_REPORT.md`
- None; Phase 8 files are new
- Backup handling: No existing Phase 3–7 source files were modified; all new modules were added. No pre-existing source/raw data or model checkpoints were overwritten. A new corrected final manifest was created alongside the earlier manifest.

## Automated test results

```json
{
  "command": ".venv/Scripts/python.exe -m pytest -q --basetemp data/processed/phase8/pytest-temp",
  "passed": 149,
  "failed": 0,
  "total": 149,
  "duration_seconds": 12.07,
  "by_phase": {
    "phase3": 28,
    "phase4": 15,
    "phase5": 21,
    "phase6": 19,
    "phase7": 24,
    "phase8": 42
  },
  "synthetic_fixtures_not_real_corpus_data": true
}
```

## Known limitations

- Only the explicitly CC0-labeled Tatoeba export was ingested; general Tatoeba and other candidates remain NEEDS_REVIEW/UNKNOWN_LICENSE.
- The export listing did not provide an independent publisher checksum; the recorded SHA-256 is a local integrity fingerprint.
- Parquet support remains optional; pyarrow is not installed and no dependency was added.
- Windows process RSS was unavailable; system RAM was measured.
- Arabic coverage is far too small and absent from validation/test.
- The 1,000,000-record benchmark checkpoint was not reached; performance numbers are streaming I/O/parse measurements, not model throughput.
- The early failed empty JSONL artifact is retained with an explicit do-not-use status to preserve history.

## Recommendations

- Acquire additional corpora only after source-specific license/provenance review, prioritizing a meaningful Arabic contribution and independent held-out coverage.
- Review privacy, content suitability, language distribution and downstream obligations before training approval.
- Before Phase 9 training, adopt a documented data mixing plan, use only approved manifests and review representative validation/test examples.
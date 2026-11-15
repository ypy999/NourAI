# Phase 10.3 — Data Pipeline Reproducibility Report

**Status:** `COMPLETE`  
**Date:** `2026-11-13`  
**Test Suite:** `207 passed, 0 failed` (including 6 new Phase 10.3 tests in `tests/test_phase10_3_pipeline_reproducibility.py`)

---

## 1. Objectives Completed

1. **Run Manifest Architecture (`corpus/reproducibility.py`)**:
   - Added `build_run_manifest()` and `verify_run_manifest()` to bind configuration SHA-256, raw source SHA-256 and provenance sidecars, adapter versions, imported JSONL hashes, normalized JSONL hashes, and final split hashes (`train`, `validation`, `test`) into a single verifiable `run_manifest.json`.
   - Upgraded `audit_artifacts()` and `_producer_verdict()` to recognize `.import.json`, `.normalized.json`, and `run_manifest.json` sidecars and support custom target directories (`targets=...`).

2. **Compact Traceable Normalization (`corpus/pipeline.py`)**:
   - Updated `normalize_imported_file()` to support `retain_stage_texts=True/False` while always writing a `.normalized.json` sidecar containing `input_sha256`, `output_sha256`, `unicode_form`, and `cleaning_configuration`.
   - When `retain_stage_texts=False`, per-record hashes (`raw_text_sha256`, `normalized_text_sha256`, `cleaned_text_sha256`) and the `.normalized.json` link to `imported.jsonl` preserve 100% auditability without duplicating multi-gigabyte raw strings across stages.

3. **Multi-Source `prepare_splits` (`corpus/pipeline.py`)**:
   - Upgraded `prepare_splits()` to accept multiple license-verified imported files (`tatoeba_cc0`, `qnl_arabic_ocr_v2`, `wto_public_corpus_en_fr`), verifying each source against `SourceRegistry`, `LicenseManager`, and `verify_download_provenance` before normalization and group-aware splitting.
   - Added hard write guards in `import_source_file`, `normalize_imported_file`, and `prepare_splits` that raise `PermissionError` if any output path targets `data/processed/phase10_2`.

4. **Phase 10.3 Configuration (`configs/phase10_3_corpus.json`)**:
   - Created dedicated configuration targeting `data/processed/phase10_3/` and `data/metadata/phase10_3_corpus.sqlite3` with approved split ratios `0.90 / 0.05 / 0.05` and `random_seed = 1337`.

---

## 2. Acceptance Criteria Verification

- **Deterministic Reproduction**: Verified by `test_reproducible_two_runs_and_run_manifest_verification`.
- **Tamper & Policy Detection**: Verified by `test_verify_run_manifest_detects_tampering_and_bad_split_decisions`.
- **Frozen Tree Protection**: Verified by `test_pipeline_blocks_any_write_to_frozen_phase10_2`.
- **Multi-Source Support**: Verified by `test_multi_source_prepare_splits_and_manifest`.

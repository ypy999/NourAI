# Phase 10.4 — QNL Adapter Implementation Report

**Status:** `COMPLETE`  
**Date:** `2026-11-13`  
**Adapter Version:** `qnl_arabic_ocr_v2_adapter_v1`  
**Tests:** `tests/test_phase10_4_qnl_adapter.py` (4 passed, including read-only verification on the real 1.06 GB QNL archive)

---

## 1. Implementation Details

1. **Streaming QNL Adapter (`corpus/adapters/qnl_adapter.py`)**:
   - Parses `qnl_arabic_ocr_v2_metadata.csv` (2,894 cataloged items) and maps the 9 official catalog columns (`CALL #(ITEM)`, `Repository URL`, `Catalog URL`, `AUTHOR`, `ADD AUTHOR`, `PUB INFO`, `TITLE`, `DESCRIPTION`, `VOLUME`) into `qnl_metadata`.
   - Reads `checksums.sha256` embedded inside `qnl_arabic_ocr_v2.zip` and verifies the SHA-256 of every `.txt` member before emitting a record.
   - Extracts `source_record_id`, `document_id`, and `original_document_id` from the official `RECORD #(ITEM)` (e.g., `i10027154`), sets `document_id_status = "SOURCE_PROVIDED"`, and `provenance_id_source = "QNL RECORD #(ITEM); original catalog record ID"`.
   - Computes `arabic_script_ratio` accurately as the proportion of characters in Unicode Arabic script blocks (`U+0600..U+06FF`, `U+0750..U+077F`, `U+08A0..U+08FF`, `U+FB50..U+FDFF`, `U+FE70..U+FEFF`). Note: the vanished Phase 10.2 script had inverted this metric (`1 - ratio = 0.214829` on `i10027154`); the standard adapter records the true Arabic script ratio (`0.785171`).

2. **Pipeline & Reproducibility Integration (`corpus/importers.py`, `corpus/pipeline.py`, `corpus/reproducibility.py`)**:
   - `iter_source_records()` and `import_source_file()` automatically dispatch `zip_txt_with_official_metadata_csv` to `iter_qnl_archive_records()` and preserve all QNL provenance fields and `adapter_version` in `.import.json`.
   - `build_plan()` upgrades `qnl_arabic_ocr_v2` to `SUPPORTED` and verifies both the ZIP sidecar and the metadata CSV sidecar when `--verify-hashes` is enabled.

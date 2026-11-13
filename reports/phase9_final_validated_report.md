# Phase 9 Final Validated Corpus Snapshot

Generated 2026-11-12. Corpus counts below come from the existing Phase 9 corpus report; database checks were independently run against SQLite in read-only URI mode.

## Corpus data

| Metric | Actual value | Source |
|---|---:|---|
| Imported records before deduplication | 55,719 | `reports/phase9_corpus_report.json` |
| Normalized duplicates removed | 3 | Phase 8/9 deduplication results |
| Unique records after deduplication | 55,716 | Phase 8/9 report |
| Arabic | 2 | Phase 9 source-language metadata |
| English | 41,515 | Phase 9 source-language metadata |
| French | 14,202 | Phase 9 source-language metadata |
| Train / Validation / Test | 50,144 / 2,787 / 2,785 | Phase 8 split report |
| Duplicate leakage | 0 | Phase 8 split validation |

Imported source: `tatoeba_cc0` only. Domain counts are heuristic; see the phase9 JSON for per-language estimates. They are not verified subject classifications.

## SQLite validation

Read-only connection succeeded using Windows URI `file:///D:/NourAI/data/metadata/corpus.sqlite3?mode=ro`. SQLite version: 3.53.4. `PRAGMA integrity_check` returned `ok`; `PRAGMA foreign_key_check` returned no rows.

There are 8 tables including `sqlite_sequence`: sources=1, datasets=2, files=6, records=111,435, operations=3, license_current=4, license_history=4. Records are indexed under source `tatoeba_cc0` (111,435), with no orphan record rows and no missing `source_record_id` or `raw_sha256` values.

Database language metadata counts are Arabic=4, English=74,655, French=27,216, Unknown=9,560. These counts include both imported and split metadata rows and must not be interpreted as corpus language counts. The database schema has no `document_id` or ingestion timestamp column, so those fields are not independently auditable from SQLite.

License states: VERIFIED=1, NEEDS_REVIEW=2, UNKNOWN_LICENSE=1. Only Tatoeba CC0 is represented as imported. Candidate registry entries are not automatically approved.

## Tests and synthetic data separation

The previously executed full suite reported 174 passed, 0 failed, 0 skipped. It was not rerun to create this report. Synthetic fixtures used by tests are test-only and are not included in these corpus statistics. No synthetic benchmark or new performance benchmark is represented here.

## Readiness

Arabic has only two records, with no Arabic validation or test coverage. Project readiness remains **NOT_READY**. Passing software tests do not establish language capability, semantic quality, or fitness for training.

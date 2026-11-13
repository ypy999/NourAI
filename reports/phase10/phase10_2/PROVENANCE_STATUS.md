# Phase 10.2 — Provenance Status (QNL / WTO)

**Status:** `REPAIRED / VERIFIED AGAINST THE LOCAL CATALOG`
**Audited:** 2026-11-13
**Backups (pre-edit):** `.backup/p0-provenance-20261113/`
**Tests:** `tests/test_phase10_provenance_compat.py` (10 tests), full suite `201 passed`

No raw archive was modified. No license check was relaxed. Missing metadata is
still an error, never a default.

---

## 1. Defects found

| # | Defect | Effect |
| --- | --- | --- |
| 1 | `verify_download_provenance` required the literal field `source_url`. | QNL sidecars record `official_item_url` and failed validation. |
| 2 | The verifier compared `license_id` only if present in the sidecar; a sidecar without any license field passed. | The WTO sidecar (no license at all) was accepted silently. |
| 3 | `corpus.validation.validate_jsonl` hardcoded `CC0-1.0` and `approved_source_ids=("tatoeba_cc0",)`. | Every QNL/WTO record was reported as an unknown source and an unknown license. |
| 4 | WTO sidecar carried identity and hash but **no license identity**. | Provenance was incomplete: a source without recorded terms. |

The underlying WTO license is documented in `configs/dataset_sources.json`
(`status=VERIFIED`, `checked_at=2026-11-13`, official WTO corpus page). The
defect was in the *sidecar*, not in the registry.

---

## 2. What changed

### 2.1 `corpus/acquisition.py` — strict, alias-aware verification

- Added `md5_file()` and `SOURCE_URL_FIELDS = ("source_url", "official_item_url", "landing_page_url")`.
- Identity rule: at least one identity field must be present **and** equal the
  registry `url`; nothing is filled in from the registry.
- `download_url`, when recorded, must equal the registry `download_url`.
- `license_id` is **mandatory** (`missing license_id` is a hard error) and must
  equal the registry `license_id`; a recorded `license_url` must match too.
- Size and SHA-256 are always checked; a recorded `md5` must match the file and
  must agree with `expected_md5_from_official_metadata` when that is recorded.
- The returned record names which alias was used: `verified_source_url_field`.

### 2.2 `corpus/validation.py` — registry-driven record validation

- `approved_source_ids` now defaults to every registry source with
  `status == "VERIFIED"` **and** `approved_for_ingestion is True`
  (`_approved_registry_source_ids()`).
- A record's `license_id` must equal the `license_id` registered for **its own**
  `source_id`; the `CC0-1.0` literal is gone.
- Passing an explicit allowlist cannot bypass the license check — an allowed
  source with the wrong license still fails (covered by a test).

### 2.3 WTO sidecar — metadata repaired, critical fields untouched

File: `data/raw/external/phase10_2/wto_en_fr_corpus_2019_EN-FR.zip.provenance.json`

| Field | Before | After |
| --- | --- | --- |
| `license_id` | *absent* | `WTO_PUBLIC_CORPUS_TERMS` |
| `license_url` | *absent* | `https://www.wto.org/english/res_e/corpus_e/corpus_e.htm` |
| `license_evidence` | *absent* | registry entry + `status=VERIFIED` + `checked_at` |
| `source_name` | *absent* | `WTO Public Trilingual Corpus — English/French subset` |
| `provenance_repaired_at_utc`, `provenance_repair_note` | *absent* | recorded |
| `source_id`, `source_url`, `download_url`, `bytes`, `sha256`, `retrieved_at_utc`, `official_last_modified`, `archive_validation` | — | **unchanged** (verified programmatically) |

QNL sidecars were **not** edited: they already carried identity, license and MD5
evidence. Only the verifier was made able to read them correctly.

---

## 3. Verification evidence (real artifacts, no fixtures)

```
$ python -c "verify_download_provenance(source_id, path) for the three sources"
qnl_arabic_ocr_v2      OK  official_item_url  PUBLIC_DOMAIN_TEXTS; CC0-1.0_METADATA  352d7d2e1e3c6425
wto_public_corpus_en_fr OK  source_url         WTO_PUBLIC_CORPUS_TERMS                 09f958e84a626cc2
tatoeba_cc0            OK  source_url         CC0-1.0                                 7e4251d06edd949d
```

QNL MD5 `1e23030c19b864170b6d3c8060986021` equals
`expected_md5_from_official_metadata`; the metadata CSV MD5
`d361e155b845e4a5e73445ba3b04c11f` matches as well.

Audit evidence: `reports/phase10/phase10_2/ARTIFACT_AUDIT.json` records the
post-repair sidecar SHA-256 `7deca5edc687df45…` (1,130 bytes) alongside the
unchanged archive hash.

---

## 4. Test coverage

`tests/test_phase10_provenance_compat.py` — 10 tests:

Valid cases
1. QNL sidecar with `official_item_url` only → accepted, alias reported.
2. WTO sidecar with its registered license → accepted; `download_url` echoed.
3. Records from QNL and WTO validate against the registry licenses.

Invalid cases
4. QNL `official_item_url` pointing elsewhere → rejected.
5. WTO sidecar with `license_id` removed → rejected (`missing license_id`), not defaulted.
6. Wrong `license_id` (CC0 on WTO) → rejected.
7. No identity field at all → rejected.
8. Mismatched `download_url` → rejected.
9. Recorded MD5 inconsistent with the file → rejected; inconsistent with the official MD5 → rejected.
10. Record with a wrong license, unknown source, or an explicit allowlist → rejected.

Full suite: **201 passed** (`-p no:cacheprovider`), including the 181 pre-existing
tests and the 10 reproducibility tests.

---

## 5. Remaining limits (not fixable inside P0)

1. The registry entries themselves are locally curated JSON; this change proves
   **consistency with the catalog**, not that the upstream license page still
   says the same thing. Re-verification against the live pages remains periodic
   manual work.
2. QNL `license_id` is a composite string
   (`PUBLIC_DOMAIN_TEXTS; CC0-1.0_METADATA`) because texts and metadata have
   different terms. It is matched verbatim; any future split of that identifier
   needs a registry schema change.
3. WTO terms are recorded as `WTO_PUBLIC_CORPUS_TERMS` with a source URL, not as
   an SPDX identifier; there is no SPDX equivalent.
4. The 13.4 GB payloads imported by the vanished ad-hoc script still cannot be
   regenerated (see `REPRODUCIBILITY_STATUS.md`). Provenance of the *downloads*
   is now sound; provenance of the *derivation* is not.
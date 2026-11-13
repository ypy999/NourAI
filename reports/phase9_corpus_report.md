# Phase 9 Corpus and Gap Analysis

Generated: 2026-11-12T15:23:59.581866+00:00

Status: **NOT_READY**. No training started; no new corpus imported.

## Current corpus

| Language | Records | ByteTokenizer IDs incl. EOS | UTF-8 text bytes | Mean characters |
|---|---:|---:|---:|---:|
| english | 41,515 | 3,373,668 | 3,332,153 | 80.09 |
| french | 14,202 | 987,575 | 973,373 | 66.10 |
| arabic | 2 | 99 | 97 | 28.00 |

## Readiness blockers

- arabic is absent from the corpus or an independent validation/test split
- arabic has 1 source(s); configured minimum is 2
- arabic has 0 source-verified domain(s); configured minimum is 3
- english has 1 source(s); configured minimum is 2
- english has 0 source-verified domain(s); configured minimum is 3
- french has 1 source(s); configured minimum is 2
- french has 0 source-verified domain(s); configured minimum is 3
- training is explicitly out of scope for Phase 9
- one or more legal, technical, language-evaluation, or diversity gates are not ready

## Heuristic domain distribution

- **english:** Conversations=1,202, Education=769, General Knowledge=2,920, Geography=1,052, History=683, Literature=480, Mathematics=283, News=904, Other=33,671, Programming=191, Science=388, Technology=385
- **french:** Conversations=652, Education=156, General Knowledge=534, Geography=171, History=215, Literature=279, Mathematics=70, News=120, Other=11,893, Programming=44, Science=198, Technology=163
- **arabic:** Conversations=1, Other=1

## Phase 8 text-level mixed and unclassified rates

- unclassified: 8,790 (15.78%)
- mixed: 772 (1.39%)

## Existing splits (text-derived language labels)

| Split | Records | ByteTokenizer IDs incl. EOS | UTF-8 bytes |
|---|---:|---:|---:|
| train | 50,144 | 3,931,688 | 3,881,544 |
| validation | 2,787 | 213,486 | 210,699 |
| test | 2,785 | 215,909 | 213,124 |

Normalized duplicates removed before splitting: 3; exact duplicates: 0; normalized duplicate fraction: 5.384159801862919e-05; duplicate leakage: 0; validation: True.

Phase 7 format/integrity quality mean: 99.9989; interpretation: heuristic, not semantic or factual quality.

## Source candidates and license gates

- [tatoeba_general](https://downloads.tatoeba.org/exports/sentences.tar.bz2): CC-BY-2.0-FR / NEEDS_REVIEW, approved=False. Attribution and corpus use require review.
- [wikimedia](https://dumps.wikimedia.org/): CC-BY-SA-4.0/GFDL (project and revision dependent) / NEEDS_REVIEW, approved=False. Needs project/revision-level license and attribution review.
- [oscar](https://huggingface.co/datasets/oscar): Underlying page licenses vary / are not established by packaging license / UNKNOWN_LICENSE, approved=False. Do not ingest automatically.
- [common_voice_ar_27](https://mozilladatacollective.com/datasets/cmu5wom3z00ceo107guxtg4xa): CC0-1.0 / VERIFIED, approved=False. CC0 identified; Common Voice platform terms prohibit mirroring/rehosting and speaker identification; ingestion not approved pending use/retention review. Only validated sentences; gated MDC access; ASR text not general-domain.
- [common_voice_en_au_24](https://mozilladatacollective.com/datasets/cmko7havo02f5nw07rbwwhowe): CC0-1.0 / VERIFIED, approved=False. Gated MDC; elicited ASR speech, not general-domain text. Platform terms restrict redistribution/speaker identification.
- [common_voice_fr_26](https://mozilladatacollective.com/datasets/cmrumhinj0001md07y3huuhs5): CC0-1.0 / VERIFIED, approved=False. Gated MDC; small ASR subset, not general-domain text. Platform terms restrict redistribution/speaker identification.
- [madar_parallel](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/): RESEARCH_INTERNAL_USE_ONLY / NEEDS_REVIEW, approved=False. Official terms restrict use to internal research/evaluation and prohibit redistribution/modification; excluded from training ingestion.
- [culturax](https://huggingface.co/datasets/uonlp/CulturaX): UNKNOWN_LICENSE / UNKNOWN_LICENSE, approved=False. Packaging does not resolve source-page rights; privacy/sensitive data risk flagged; excluded.
- [oscar_moscar](https://oscar-project.github.io/documentation/versions/mOSCAR/): UNKNOWN_LICENSE / UNKNOWN_LICENSE, approved=False. Packaging license is not treated as granting rights to underlying pages; excluded pending source-level review.
- [mot_voa](https://github.com/bltlab/mot): CC-BY-4.0 / NEEDS_REVIEW, approved=False. English/French candidate only; no Arabic. Requires attribution and artifact/source terms review; not downloaded.
- [hplt_v1_2](https://hplt-project.org/datasets/v1.2): UNKNOWN_LICENSE / UNKNOWN_LICENSE, approved=False. CC0 applies to packaging/artifacts as described; source-page text rights are not resolved. Arabic raw corpus is approximately 174 GB and is not suitable for current local storage. Excluded.

Heuristic domain assignment is not validated labeling; corpus size and passing software checks do not establish language capability.


## Execution and governance results

**Implementation status: PARTIAL — Phase 9 analytics and reporting are implemented, but multilingual corpus acquisition/expansion is not complete.** The truthful data readiness state is **NOT_READY**. No new source data was downloaded or imported. Training was not started and CUDA was not used.

### Corpus measurements

- Existing imported records: 55,719; unique after the Phase 8 deduplication: 55,716.
- ByteTokenizer IDs including EOS across the imported corpus: 4,361,342; UTF-8 text bytes: 4,305,623.
- Per-language source metadata counts, tokens, bytes, and domain counts are in `reports/phase9_corpus_report.json` and `reports/phase9_domain_distribution.json`.
- Phase 8 measured mixed text: 772 (1.39%); unclassified: 8,790 (15.78%).
- Phase 7 format/integrity score mean: 99.9989; this is not semantic or factual text quality.
- Normalized duplicates removed before splitting: 3; exact duplicates reported: 0; leakage: 0; remaining within-split duplicates: 0.
- Split record counts: Train 50,144, Validation 2,787, Test 2,785.

### Readiness gates

- Legal: READY for the one currently used verified source (Tatoeba CC0).
- Technical split validation: READY; zero leakage.
- Language readiness: NOT_READY. English and French have independent held-out data; Arabic has 2 records, both in Train, with 0 Arabic records in Validation and Test.
- Diversity readiness: NOT_READY; only one source is used per language and there are no source-verified domain labels.
- Training readiness: NOT_READY; no training started.

### Source review

All candidates below remain excluded from the approved corpus. A CC0 label on a dataset page does not override platform restrictions. Web-crawl packaging does not establish rights in each source page.

- [tatoeba_general](https://downloads.tatoeba.org/exports/sentences.tar.bz2): NEEDS_REVIEW, license `CC-BY-2.0-FR`, approved `False`.
- [wikimedia](https://dumps.wikimedia.org/): NEEDS_REVIEW, license `CC-BY-SA-4.0/GFDL (project and revision dependent)`, approved `False`.
- [oscar](https://huggingface.co/datasets/oscar): UNKNOWN_LICENSE, license `Underlying page licenses vary / are not established by packaging license`, approved `False`.
- [common_voice_ar_27](https://mozilladatacollective.com/datasets/cmu5wom3z00ceo107guxtg4xa): VERIFIED, license `CC0-1.0`, approved `False`.
- [common_voice_en_au_24](https://mozilladatacollective.com/datasets/cmko7havo02f5nw07rbwwhowe): VERIFIED, license `CC0-1.0`, approved `False`.
- [common_voice_fr_26](https://mozilladatacollective.com/datasets/cmrumhinj0001md07y3huuhs5): VERIFIED, license `CC0-1.0`, approved `False`.
- [madar_parallel](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/): NEEDS_REVIEW, license `RESEARCH_INTERNAL_USE_ONLY`, approved `False`.
- [culturax](https://huggingface.co/datasets/uonlp/CulturaX): UNKNOWN_LICENSE, license `UNKNOWN_LICENSE`, approved `False`.
- [oscar_moscar](https://oscar-project.github.io/documentation/versions/mOSCAR/): UNKNOWN_LICENSE, license `UNKNOWN_LICENSE`, approved `False`.
- [mot_voa](https://github.com/bltlab/mot): NEEDS_REVIEW, license `CC-BY-4.0`, approved `False`.
- [hplt_v1_2](https://hplt-project.org/datasets/v1.2): UNKNOWN_LICENSE, license `UNKNOWN_LICENSE`, approved `False`.

Official references reviewed: [Common Voice Arabic 27](https://mozilladatacollective.com/datasets/cmu5wom3z00ceo107guxtg4xa), [Common Voice terms](https://commonvoice.mozilla.org/dav/terms), [MADAR](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/), [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX), [OSCAR/mOSCAR](https://oscar-project.github.io/documentation/versions/mOSCAR/), [HPLT v1.2](https://hplt-project.org/datasets/v1.2), [MOT](https://github.com/bltlab/mot), [Wikimedia Terms](https://foundation.wikimedia.org/wiki/ToU).

### Measured CPU performance

Real corpus analysis took 47.61 seconds (1170.2 records/second). Working set was 21,053,440 bytes before and 22,814,720 bytes after; peak working set was 25,382,912 bytes.

Synthetic-only scaling tests (not corpus data):

| Synthetic records | Analysis seconds | Records/second | Dedup + split seconds | Output bytes | Working set after analysis (bytes) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.27 | 3656.1 | 0.15 | 132,788 | 23,474,176 |
| 10,000 | 2.75 | 3641.2 | 1.47 | 1,347,789 | 23,977,984 |
| 100,000 | 29.80 | 3355.3 | 28.57 | 13,677,790 | 24,162,304 |

The 1,000,000-record test was not run: the measured 100,000-record CPU deduplication/split took 28.57 seconds, so a million-scale run would exceed the bounded time budget for this pass. No million-record result is claimed.

### Tests, files, and backups

Full suite: **161 passed, 0 failed, 0 skipped** (baseline before Phase 9: 149 passed).

New files:

- `data/domain_analysis.py`
- `data/language_balancing.py`
- `data/phase9_readiness.py`
- `data/phase9_pipeline.py`
- `data/phase9_benchmark.py`
- `configs/phase9_corpus.json`
- `tests/test_phase9.py`
- `tests/test_phase9_domains.py`
- `tests/test_phase9_balancing.py`
- `tests/test_phase9_readiness.py`
- `reports/phase9_corpus_report.json`
- `reports/phase9_corpus_report.md`
- `reports/phase9_gap_analysis.json`
- `reports/phase9_gap_analysis.md`
- `reports/phase9_language_balance.json`
- `reports/phase9_domain_distribution.json`
- `reports/phase9_benchmark.json`
- `reports/phase9_corpus_dashboard.html`
- `PHASE_9_MULTILINGUAL_CORPUS_REPORT.md`

Modified existing files:

- `configs/dataset_sources.json`

Backups created before modifying the source registry:

- `.backup/phase9-before-execution/configs/dataset_sources.json`
- `.backup/phase9-before-execution/data/metadata/corpus.sqlite3`

The SQLite backup was created but the live metadata database was not changed. Raw corpus files and Phase 8 splits were not modified.

### Remaining blockers and next step

- Arabic requires a legally cleared, locally accessible corpus large enough for independent Validation and Test; Common Voice download access is gated and its storage/usage restrictions need approval.
- English and French need additional legally approved sources with broader non-ASR domain coverage and source diversity.
- Domains currently use low-confidence lexical heuristics; none are source-verified labels.
- The phase must remain NOT_READY until these artifacts are acquired through an authorized route, imported through the existing Phase 8 pipeline, deduplicated, re-split, and re-evaluated.

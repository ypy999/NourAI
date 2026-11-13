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

# JABRIX AI v3.4.1 — Reasoning Accuracy Patch Report (`REASONING_PATCH_V3_4_1.md`)

**Branch:** `dev/v3.4.1` (branched from `dev/v3.4`; all previous branches preserved intact).

## 1. Root-Cause Fixes Implemented
- **Multilingual Clause Segmentation ([clause_segmenter.py](file:///d:/NourAI/nourai/reasoning_v34/clause_segmenter.py)):**
  - Splits compound Arabic, English, and French sentences on punctuation (protecting decimal dots like `2.3x` / `3.4.1` while splitting cleanly after sentence-ending digits like `7860;` and `9999.`) and coordinating/subordinating connectors (including attached Arabic `و` before `ال` and common clause starters such as `والذاكرة`, `وعاصمة`, `ويدعم`) while preserving inter-clause relations (`root`, `conjunction`, `contrast`, `causal`, `sequence`).
  - Strips leading discourse markers (`however,`, `cependant,`, `ولكن`, `بينما`) prior to subject-predicate extraction so multiple subjects and predicates in a single sentence are parsed accurately.
- **Negation Detection & Polarity Contradictions ([self_verification.py](file:///d:/NourAI/nourai/reasoning_v34/self_verification.py)):**
  - Detects explicit AR/EN/FR negations and disabled states (`not`, `never`, `disabled`, `n'est pas`, `désactivé`, `لا`, `ليس`, `معطل`).
  - Explicitly separates `contradicted_claims` (factual & polarity contradictions) from `missing_info_claims` (unsupported claims due to absent evidence).

## 2. Measured Evaluation Results ([v3_4_1_benchmark_report.json](file:///d:/NourAI/reports/v3_4_1_benchmark_report.json))
- **Known `v3.4.0` 1/6 Failure Fixed:** **`6 / 6`** passed (including `"SQLite WAL is disabled in JABRIX AI and PrimaryPort is 1234."`).
- **36-Case Adversarial Multilingual Suite (12 EN · 12 AR · 12 FR):**
  - **Precision:** `1.0000` (`TP=21, FP=0`)
  - **Recall:** `1.0000` (`TP=21, FN=0`)
  - **F1 Score:** `1.0000`
  - **Per Language:** Arabic `12/12` (`TP=7, TN=5`) · French `12/12` (`TP=7, TN=5`) · English `12/12` (`TP=7, TN=5`)
- **Performance & Hardware (Intel Core i3-10110Y, 8 GB RAM, CPU-only):**
  - **Total 36-Case Verification Latency:** `10.246 ms` (`0.285 ms / case`)
  - **Process RSS:** `204.21 MiB` (`4` logical CPUs, `degraded_mode: false`)

> **Honest Scope Note:** These Precision, Recall, and F1 numbers apply strictly to this 36-case structured multilingual adversarial suite and do not guarantee universal 100% accuracy on arbitrary open-world prose or subtle pragmatic/implicit contradictions.

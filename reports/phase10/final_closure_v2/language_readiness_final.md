# Language Readiness

Overall gate: `NOT_READY`. Corpus records examined: 55,719.

| Language | Total | Train | Validation | Test | Unique source IDs | Mean chars | Mean byte tokens | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| arabic | 2 | 2 | 0 | 0 | 2 | 28.00 | 48.50 | NOT_READY |
| english | 41,515 | 37,380 | 2,070 | 2,062 | 41,515 | 80.09 | 80.26 | NOT_READY |
| french | 14,202 | 12,762 | 717 | 723 | 14,202 | 66.10 | 68.54 | NOT_READY |

## Findings

- Readiness checks use `configs/phase10_readiness.json`: at least two unique sources and three verified domains per language. The current corpus has one source, and verified-domain metadata is absent.
- Arabic has 2 records, split as {'train': 2, 'validation': 0, 'test': 0}; Validation and Test are empty.
- English and French have independent record splits, but fail source/domain diversity. The minimum training-size policy is unset, so that check remains UNKNOWN.
- Exact and normalized duplicate leakage across splits: 0. Near-duplicate candidates across splits: 12; human review is pending.
- ByteTokenizer full-corpus round trips: 55,719/55,719; invalid IDs: 0.
- Content-quality outputs are structural/surface indicators; semantic and factual quality were not assessed.

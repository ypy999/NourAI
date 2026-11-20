# NourAI v3.1 — Long-Term Memory Validation (`MEMORY_VALIDATION.md`)

Implementation: [advanced_memory.py](file:///d:/NourAI/nourai/memory/advanced_memory.py) · Tests: [test_v31_upgrade.py](file:///d:/NourAI/tests/test_v31_upgrade.py).

## 1. Precision & Consistency Upgrades in `v3.1`
- **Arabic Text Normalization (`_normalize_text`):** Strips diacritics/tatweel and normalizes `أ/إ/آ → ا`, `ة → ه`, `ى → ي` so queries without hamza/diacritics accurately match stored Arabic memories.
- **French Accent Folding:** NFKD diacritic stripping (`é → e`, `à → a`, `ç → c`) for Latin tokens while preserving Arabic Unicode.
- **Unigram + Bigram TF-IDF with Multilingual Stopword Filtering & Tier Weighting:** Eliminates episodic echo pollution (`1.25x` boost for `semantic`/`procedural` vs `0.85x` for `episodic`).
- **Preference Superseding (`update_preference`):** Automatically expires previous values of the same `preference_key` while logging `superseded_at` metadata, preventing false contradictions when user preferences evolve.

## 2. Verified Stress Scenarios (Automated Tests Passed)
| Scenario | Verified Behavior |
| :--- | :--- |
| **Multiple Conversations** | Separate sessions read/write shared user `SQLite WAL` memory cleanly. |
| **Application Restarts** | Fresh `AdvancedMemoryManager(db_path)` instances recover all persisted semantic, episodic, and procedural records from disk. |
| **Long-Running Tasks** | Episodic summaries persist outcomes without drowning out semantic facts. |
| **Conflicting Information** | Non-preference factual conflicts (e.g., `ServerPort is 7860` vs `9000`) are flagged by `check_consistency()`. |
| **Updated User Preferences** | `update_preference("response_language", "arabic" -> "french")` supersedes old value and maintains `consistent == True`. |
| **User Memory Deletion** | `delete_memory()` persists deletion across simulated restarts. |
| **Privacy Protection** | Rejects raw passwords, API keys, bearer tokens, and private keys. |

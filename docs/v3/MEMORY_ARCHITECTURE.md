# NourAI v3.0 — Advanced Memory & Knowledge Graph (`MEMORY_ARCHITECTURE.md`)

## 1. Four-Tier Memory Architecture (`nourai/memory/advanced_memory.py`)
1. **Working Memory (`WorkingMemory`):** Bounded in-memory scratchpad (`max_items=32`) scoped to active `task_id` and cleared on completion.
2. **Episodic Memory (`tier="episodic"`):** Summarized past task executions, outcomes, and tools used, stored with default 30-day TTL.
3. **Semantic Memory (`tier="semantic"`):** Structured persistent facts with SHA-256 content-hash deduplication and source attribution.
4. **Procedural Memory (`tier="procedural"`):** Verified reusable multi-step workflows (`verified=True` enforced).

## 2. Retrieval, Privacy, and Governance
- **Semantic Retrieval:** Pure-Python TF-IDF cosine similarity weighted by confidence score (`semantic_search()`).
- **Privacy Guard:** Rejects storage of raw passwords, API keys, bearer tokens, and PEM private keys (`contains_sensitive_secret()`).
- **Lifecycle Controls:** Per-item `ttl_seconds` expiration + `purge_expired()`, user-controlled `delete_memory()` and `clear_user_memories()`, and `check_consistency()` contradiction auditing.

## 3. Integrated Knowledge Graph (`nourai/knowledge/knowledge_graph.py`)
- Extracts multilingual `(subject, predicate, object)` triples from Arabic, English, and French text.
- Stores bidirectional relations (`predicate` + `inverse_predicate`) in SQLite (`v3_kg_entities`, `v3_kg_relations`).
- Detects contradictions on single-valued predicates (`has_capital`, `version_is`, `owned_by`, `status_is`).
- Synchronized export methods:
  - `export_obsidian_logseq_markdown()` (`[[Wikilinks]]` + Logseq properties)
  - `export_freeplane_opml()` (valid OPML 2.0 XML)
  - `export_thebrain_json()` (`thebrain_plex_v1` Thoughts & Jump links)

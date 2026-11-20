# NourAI v3.2 — Autonomous Research Agent (`RESEARCH_AGENT.md`)

Implementation: [research_agent.py](file:///d:/NourAI/nourai/research/research_agent.py) · API: `GET/POST /api/v3/research` ([v3_routes.py](file:///d:/NourAI/nourai/api/v3_routes.py)).

## 1. Research Workflow Architecture
1. **Sub-Question Decomposition:** Splits complex Arabic, English, and French research objectives into focused sub-questions.
2. **Authorized Multi-Source Retrieval:** Queries `DocumentIntelligenceEngine` (citation-indexed chunks), `KnowledgeGraph` (bidirectional triples), and `AdvancedMemoryManager` (semantic & procedural tiers).
3. **Source Reliability Scoring:** Assigns explicit reliability weights (`document: 0.95`, `knowledge_graph: 0.88`, `semantic_memory: 0.82`, `web: 0.72`, `hypothesis: 0.40`).
4. **Cross-Source Contradiction Detection:** Compares extracted subject-value claims across independent citations plus Knowledge Graph single-valued conflicts.
5. **Fact vs Hypothesis Separation:** Verified facts require traceable citations (`[doc:file#chunk_N]`, `[kg:...]`, `[mem:...]`); unresolved conflicts or unsupported queries are explicitly separated into `hypotheses` and `missing_evidence`.
6. **Structured Persistence:** Persists Markdown + JSON reports in SQLite WAL (`v32_research_reports`).

## 2. Verified Benchmark Results
- **25 / 25** research tasks passed in [v32_benchmark.py](file:///d:/NourAI/nourai/evaluation/v32_benchmark.py) (AR/EN/FR multi-source retrieval, contradiction detection, and missing-evidence identification).
- **20 / 20** citation verification tasks passed with traceable `[doc:filename#chunk_N]` citations.

# NourAI v3.0 — System Architecture (`ARCHITECTURE_V3.md`)

## 1. Architectural Overview
NourAI `v3.0.0` transforms the `v1.2.0` local-first multilingual language model into a **modular, verifiable, hybrid autonomous AI system** supporting **Arabic, English, and French** on constrained local hardware (~8 GB RAM, CPU-only, Windows/Linux).

All `v1.2.0` components (`model/`, `tokenizer/`, `training/`, `inference/`, `api/`, `corpus/`, `deployment/`, `releases/`) remain intact and backward-compatible. New `v3.0.0` capabilities live in clean, independent modules under `nourai/`:

- `nourai/core/versioning.py` — Versioned configuration (`NourAIV3Config`) and idempotent SQLite migrations (`v1.2.0 -> v1.5 -> v2.0 -> v2.5 -> v3.0`).
- `nourai/tokenizer/byte_bpe.py` — Pure-Python lossless **Byte-Level BPE Tokenizer** (8K/16K target evaluation + benchmark vs `ByteTokenizer`).
- `nourai/neural/audit_and_optimize.py` — Transformer architectural audit, KV-cache equivalence & CPU benchmark, dynamic INT8 evaluation, and `L2AnchorRegularizer` for SFT catastrophic forgetting protection.
- `nourai/memory/advanced_memory.py` — Four-tier memory (`Working`, `Episodic`, `Semantic`, `Procedural`) with SQLite WAL persistence, TF-IDF cosine retrieval, deduplication, TTL expiration, privacy filtering, and consistency checks.
- `nourai/knowledge/knowledge_graph.py` — SQLite bidirectional Knowledge Graph with AR/EN/FR triple extraction, contradiction detection, and synchronized Obsidian/Logseq Markdown, Freeplane OPML 2.0, and TheBrain JSON exports.
- `nourai/tools/` — Standardized `ToolRegistry`, least-privilege `PermissionPolicy`, timeout runner, SQLite audit log, rollback snapshots, and 9 sandboxed tools across 8 categories.
- `nourai/agent/` — 6-engine Autonomous Agent Core (`Perception`, `Reasoning`, `Planning`, `TaskManager`, `Critic`, `Recovery`) + 13-stage `AutonomousAgentOrchestrator`.
- `nourai/evaluation/v3_benchmarks.py` — Measurable evaluation harness, fixed AR/EN/FR + agent datasets, and deployment regression gate.
- `nourai/monitoring/hardware.py` — CPU/RAM monitor, `LazyModelLoader`, and memory-aware batch sizing.
- `nourai/api/v3_routes.py` — `V3APIRouter` mounted into `api/backend.py` alongside `v1` routes.

## 2. Honesty & Capability Attribution
NourAI v3.0 strictly separates:
- `response_source="neural_model"` — direct local Transformer inference (`inference/engine.py`).
- `response_source="v3_autonomous_agent_orchestrator"` — deterministic, tool-grounded autonomous execution (`nourai/agent/orchestrator.py`).
- `response_source="rule_based_fallback_assistant"` — optional `v1.2.0` fallback assistant (`nourai/capabilities.py`).

# NourAI v3.0 — Versioning & Migration Guide (`MIGRATION_GUIDE.md`)

## 1. Version Progression
| Release | Codename | Schema Version | Key Capabilities Added |
| :--- | :--- | :---: | :--- |
| `v1.2.0` | Local-First Multilingual Foundation | `1` | Pre-LN Transformer, ByteTokenizer, v1 API, TLS 1.3, SQLite WAL |
| `v1.5.0` | Smart Agent | `2` | Standardized Tool Engine + `v3_tool_audit_log` |
| `v2.0.0` | Cognitive Agent | `3` | 4-Tier Memory (`v3_memory_items`) + Knowledge Graph (`v3_kg_entities`, `v3_kg_relations`) |
| `v2.5.0` | Adaptive Agent | `4` | Measurable Evaluation & Regression History (`v3_evaluation_runs`) |
| `v3.0.0` | Autonomous AI System | `5` | 6-Engine Autonomous Agent Core + Persistent Tasks (`v3_agent_tasks`) + v3 API |

## 2. Backward Compatibility Guarantees
- All `v1.2.0` API endpoints (`/api/v1/*`), checkpoints (`releases/v1.2.0/`), and `ByteTokenizer` remain 100% intact.
- All database migrations in `MigrationManager.migrate_to_latest()` (`nourai/core/versioning.py`) are **strictly additive and idempotent** (`CREATE TABLE IF NOT EXISTS`) — zero destructive schema drops or user data deletion.

## 3. Running Migrations Programmatically
```python
from nourai.core.versioning import MigrationManager
status = MigrationManager("data/production/nourai_production.sqlite3").migrate_to_latest()
print(status["all_applied"])
```

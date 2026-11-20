# NourAI v3.0 — Standardized Tool System (`TOOL_SYSTEM.md`)

## 1. Tool Contract (`nourai/tools/tool_engine.py`)
Every tool subclasses `BaseTool` and declares an immutable `ToolSpec`:
- `name`, `category`, `description`
- `input_schema`, `output_schema`
- `required_permissions` (least-privilege scope tuple)
- `timeout_seconds` (enforced via `ThreadPoolExecutor` timeout)
- `is_destructive` (triggers mandatory human approval gate)
- `supports_rollback` (enables snapshot restoration)

## 2. Registered Built-in Tools (`nourai/tools/builtin_tools.py`)
| Tool | Category | Permissions | Destructive | Safety Controls |
| :--- | :--- | :--- | :---: | :--- |
| `file_read` | `file_operations` | `fs:read` | No | Workspace path jail, blocks `.env`/`.jwt_secret`/`*.pem`/`*.key` |
| `file_write` | `file_operations` | `fs:write` | **Yes** | Path jail, 2 MiB cap, explicit approval, rollback snapshot |
| `document_analysis` | `document_analysis` | `doc:analyze` | No | Path jail, safe stdlib `.docx`/`.pdf` extractor |
| `database_query` | `database_queries` | `db:read` | No | Read-only SQLite URI (`?mode=ro`), `SELECT`-only, blocks DML/DDL |
| `web_search` | `web_search` | `network:web` | No | Wraps SSRF-protected `nourai/web_search.py` |
| `report_generation` | `report_generation` | `report:generate` | No | Pure Markdown/JSON formatter |
| `python_sandbox` | `controlled_python` | `code:sandbox` | No | AST whitelist evaluator; zero `exec`/`eval`, zero imports/OS |
| `api_call` | `api_integrations` | `network:api` | No | `validate_public_url` SSRF guard + host allowlist |
| `knowledge_retrieval` | `knowledge_retrieval` | `knowledge:read` | No | Queries `AdvancedMemoryManager` + `KnowledgeGraph` |

All invocations are logged to SQLite (`v3_tool_audit_log`) and in-memory ring buffer.

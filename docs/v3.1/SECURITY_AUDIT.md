# NourAI v3.1 — Security Audit (`SECURITY_AUDIT.md`)

## 1. Verified Security Controls (Tested & Passing)
- **Least-Privilege Tool Execution:** Default user policy grants read-only/sandbox scopes only (`fs:read`, `doc:analyze`, `db:read`, `report:generate`, `code:sandbox`, `knowledge:read`). Destructive (`fs:write`) and network (`network:web`, `network:api`) scopes require explicit admin grants (`403` otherwise).
- **Mandatory Human Approval Gate:** Destructive tools (`file_write`) cannot run without explicit approval (`APPROVAL_REQUIRED` pauses task in `AWAITING_APPROVAL`). `RecoveryEngine` and `DAGToolExecutor` never bypass approval or permission denials.
- **Parallel Execution Safety (`DAGToolExecutor`):** Parallel `ThreadPoolExecutor` waves are strictly restricted to non-destructive, non-approval-gated steps; any destructive step forces sequential execution.
- **Sandboxed Python (`python_sandbox`):** Pure AST whitelist (`ast.parse(..., mode="eval")`); blocks `__import__`, `os.system`, attribute access, and excessive nesting/exponents. Server-side `/api/v1/code/run` remains disabled (`403`).
- **Path Jail & Secret Protection:** `_resolve_safe_path()` blocks `../` escapes and sensitive files (`.env`, `.jwt_secret`, `*.pem`, `*.key`).
- **Read-Only SQLite Queries:** `database_query` uses `mode=ro` URI and regex-blocks all DML/DDL (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `PRAGMA`, etc.).
- **SSRF & Memory Privacy:** `api_call` and `web_search` enforce `validate_public_url()` (blocking localhost/private/metadata IPs), and `AdvancedMemoryManager` rejects raw passwords/API keys/private keys.
- **No Unverified Auto-Training:** `SelfEvaluator.allow_auto_unverified_model_updates` and `NourAIV3Config.allow_auto_train_on_unverified_feedback` are strictly `False`.

## 2. Residual Risks / Notes
- `ui/control_center.html` uses inline `<script>` and `<style>` blocks; when served behind strict CSP headers in production, script/style SHA-256 hashes (matching `/` in `api/backend.py`) should be computed if inline scripts are restricted by reverse proxies.
- No external deployments or paid cloud calls were performed.

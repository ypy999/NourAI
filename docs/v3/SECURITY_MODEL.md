# NourAI v3.0 — Security Model (`SECURITY_MODEL.md`)

## 1. Core Security Principles
- **Least-Privilege Execution:** `PermissionPolicy` grants read-only/sandbox permissions by default (`fs:read`, `doc:analyze`, `db:read`, `report:generate`, `code:sandbox`, `knowledge:read`). Elevated permissions (`fs:write`, `network:web`, `network:api`) require explicit admin grants (`POST /api/v3/tools/permissions`).
- **Mandatory Human Approval:** Destructive tools (`is_destructive=True`, e.g., `file_write`) cannot execute without `approved=True` or explicit human approval via `POST /api/v3/agent/tasks/{id}/approve`. The `RecoveryEngine` never bypasses `APPROVAL_REQUIRED` or `PERMISSION_DENIED`.
- **No Unrestricted Shell / Code Execution:** `python_sandbox` uses an AST whitelist (`ast.parse(..., mode="eval")`) with zero `exec`/`eval`, zero imports, zero attribute traversal, and strict depth/node/exponent limits. Legacy `/api/v1/code/run` remains disabled (`403`).
- **Path Jail & Secret Guard:** `_resolve_safe_path()` enforces containment within `workspace_root` and blocks `.env`, `.jwt_secret`, `id_rsa`, `id_ed25519`, `*.pem`, `*.key`, `*.p12`, `*.pfx`.
- **Network & SSRF Protection:** `web_search` and `api_call` enforce `validate_public_url()` (blocks localhost, private RFC1918 IPs, and cloud metadata endpoints) plus host allowlisting for `api_call`.
- **Memory Privacy Guard:** `AdvancedMemoryManager` rejects storing raw passwords, API keys, bearer tokens, and private keys.
- **Inherited `v1.2.0` Controls:** `PBKDF2-HMAC-SHA256` (100K iterations), `HMAC-SHA256` tokens (`NOURAI_SECRET_KEY`), per-IP/per-user rate limiting, `TLS 1.3`, `SQLite WAL`, and strict CSP/security headers.

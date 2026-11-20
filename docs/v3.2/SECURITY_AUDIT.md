# NourAI v3.2 — Security & Reliability Audit (`SECURITY_AUDIT.md`)

## 1. Audited & Verified Controls
1. **Tool Permissions:** Least-privilege `PermissionPolicy` blocks unauthorized writes (`fs:write`) and network calls (`network:web`, `network:api`) by default.
2. **Document Parsing Safety:** `DocumentIntelligenceEngine` enforces `8 MiB` upload cap and `12 MiB` `.docx` XML decompression cap (zip-bomb protection) with zero unsafe `pickle`/`eval` deserialization.
3. **External URLs / SSRF:** `web_search` and `api_call` enforce `validate_public_url()` (blocking localhost, RFC1918 private networks, and metadata IPs) plus explicit host allowlisting.
4. **File Access:** `_resolve_safe_path()` enforces strict workspace path containment and blocks `.env`, `.jwt_secret`, `*.pem`, `*.key`.
5. **Persistent Memory:** Privacy guard blocks raw passwords, API keys, bearer tokens, and PEM private keys across all 4 tiers and project-scoped memories.
6. **API Authentication:** All `/api/v3/*` endpoints (except public `/api/v3/system/health`) require valid HMAC-SHA256 Bearer tokens and respect rate limiting.
7. **Execution Sandboxing:** `python_sandbox` uses a strict AST whitelist (zero `exec`/`eval`/imports); `database_query` enforces `?mode=ro` read-only `SELECT`-only SQLite access.
8. **Human Approval Preservation:** Destructive tools (`file_write`) and `requires_approval=True` project tasks pause in `AWAITING_APPROVAL` and cannot be bypassed by `RecoveryEngine` or `ProjectManager`.
9. **No Uncontrolled Self-Modification:** `SelfEvaluator.allow_auto_unverified_model_updates` remains strictly `False`.

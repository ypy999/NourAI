# 8. Security & Integrity Report (`SECURITY_REPORT.md`)

- **Audit Engine:** `security/security_audit_v2.py` (`python -m security.security_audit_v2`)
- **Gate Verdict:** **`PASS_NO_OPEN_CRITICAL_OR_HIGH`** (`all_verification_checks_passed: true`)
- **Open Findings Summary:** **`CRITICAL = 0`** | **`HIGH = 0`** | **`MEDIUM = 3`** | **`LOW = 1`**

## 1. Remediated Critical, High & Integrity Findings (Verified)

| ID | Previous Severity | Vulnerability / Defect | Remediation & Verification |
|---|:---:|---|---|
| `SEC-FIX-01` | `CRITICAL` | Remote Code Execution (`subprocess`) in `/api/v1/code/run` | Replaced with strict AST arithmetic-only evaluator; arbitrary Python rejected with `403`. |
| `SEC-FIX-02` | `CRITICAL` | Public admin privilege escalation in `/api/v1/users/register` | Hard-locked public registration to `role="user"`; privileged roles rejected with `403`. |
| `SEC-FIX-03` | `HIGH` | Prompt injection unguarded in default neural chat mode | Enforced safety guard across BOTH neural and fallback paths. |
| `SEC-FIX-04` | `HIGH` | Missing rate limiting on `/api/v1/auth/token`, `/api/v1/users/register`, `/api/v1/bug-reports` | Per-IP rate limits active (`429 Too Many Requests`). |
| `SEC-FIX-05` | `HIGH` | Plaintext API key cookie & missing `HttpOnly` flag on session cookie | Removed `nourai_gemini_key` cookie; enforced `HttpOnly; SameSite=Lax; Max-Age=<ttl>` (and `Secure` under TLS). |
| `SEC-FIX-06` | `HIGH` | Unbounded HTTP `Content-Length` DoS & missing `client_ip` forwarding | Added `1 MiB` hard limit (`413`), malformed JSON `400`, and real client IP forwarding in `deployment/production.py`. |
| `SEC-FIX-07` | `INTEGRITY` | Secret third-party LLM substitution (`text.pollinations.ai`, `image.pollinations.ai`, `generativelanguage.googleapis.com`) in `ui/index.html` | **100% eliminated** from `ui/index.html`; verified in both unit test (`test_web_ui_honesty_pages_and_bug_report_api`) and live HTTPS probe (`forbidden_external_llm_leaks: []`). |

## 2. Live Web & Container Hardening Verified

- **Live Firebase Hosting Headers Verified (`7/7` routes):**
  - `Strict-Transport-Security: max-age=31556926; includeSubDomains; preload`
  - `X-Frame-Options: DENY`
  - `X-Content-Type-Options: nosniff`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Cache-Control: no-cache, no-store, must-revalidate` on `/about.html`, `/privacy.html`, `/report-bug.html`
- **Container Hardening (`Dockerfile`):** Dedicated non-root `jebrex` runtime user + built-in `/api/v1/health` `HEALTHCHECK`.
- **Git & CI Hygiene:** `.github/workflows/tests.yml` scans tracked files to block `.env`, `.pem`, `.key`, `.safetensors`, `.gguf` and blocks deployment unless `security_gate == PASS_NO_OPEN_CRITICAL_OR_HIGH`.

## 3. Remaining Accepted / Monitored Findings (`MEDIUM` / `LOW`)

| ID | Severity | Finding | Next Action |
|---|:---:|---|---|
| `SEC-REM-01` | `MEDIUM` | Single-file `ui/index.html` uses `'unsafe-inline'` in API CSP | Split inline scripts into external hashed JS files before large-scale public API launch. |
| `SEC-REM-02` | `MEDIUM` | Local launcher `.jwt_secret` file lacks Windows DPAPI encryption | Production hosts must inject `NOURAI_SECRET_KEY` via cloud secret manager / env var. |
| `SEC-REM-03` | `MEDIUM` | Pattern-based prompt-injection guard can be bypassed by paraphrasing | Add learned safety classifier during Level-2 alignment phase. |
| `SEC-REM-04` | `LOW` | Offline host could not run online `pip-audit` CVE check | Enable `pip-audit` step in GitHub Actions once CI runs online. |

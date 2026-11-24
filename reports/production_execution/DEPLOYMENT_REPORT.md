# 9. Deployment Report (`DEPLOYMENT_REPORT.md`)

- **Deployment Timestamp (UTC):** `2026-11-24T22:18:56Z`
- **GitHub Repository:** `https://github.com/ypy999/NourAI`
- **Pushed Branch URL:** `https://github.com/ypy999/NourAI/tree/comprehensive_project_audit`
- **Pull Request Creation URL:** `https://github.com/ypy999/NourAI/pull/new/comprehensive_project_audit`
- **Live Deployed Preview Website (24/7 Global CDN HTTPS):** **`https://ypyai-95ca7--jebrex-prod-vpozlkst.web.app`**
- **Firebase Release Version:** `d7a8760f94f512f6` (expires `2026-12-24T23:17:52Z`)
- **Primary Live Channel (`https://ypyai-95ca7.web.app`):** Intentionally untouched so existing `dev/v3.7` site is preserved until explicit maintainer approval to promote.

## 1. GitHub Phased Commit History (Pushed to `origin/comprehensive_project_audit`)

| Commit SHA | Scope | Summary |
|---|---|---|
| `e56c6b5` | Phases 1–10 | Forensic audit, governance docs, clean data gate, v1 resume hardening, security remediation (`0 CRIT/HIGH`), Windows/Android apps, 12-domain benchmark (`62` files) |
| `db6d7e9` | Cloud Storage & Phases A–E | Storage strategy & SHA-256 manifests, data versioning, BPE `32K–64K` tokenizer, `125M–350M` v2 Transformer, cloud trainer, Colab notebook, SFT/DPO/RAG pipeline (`19` files) |
| `83ab5c9` | Web UI, Docker & CI/CD | Zero-secret-LLM Web UI, `/about.html`, `/privacy.html`, `/report-bug.html`, `/api/v1/bug-reports`, `/api/v1/model-info`, hardened `Dockerfile`, `docker-compose.yml`, `.github/workflows/{tests,deploy}.yml`, `README.md`, `CONTRIBUTING.md`, `docs/DEPLOYMENT.md` (`14` files) |
| *(Final Commit)* | Production Reports | All 12 mandatory reports in `reports/production_execution/` + synced governance trackers |

## 2. CI/CD Gate Design

1. **`.github/workflows/tests.yml`** runs on push/PR to `main`, `comprehensive_project_audit`, and `dev/**`:
   - Verifies zero forbidden secrets or binary models (`.env`, `.pem`, `.key`, `.safetensors`, `.gguf`) in `git ls-files`.
   - Executes all `53` self-contained Jebrex, UI, Stage B–L, security, and benchmark tests.
   - Enforces `security_gate == "PASS_NO_OPEN_CRITICAL_OR_HIGH"`.
2. **`.github/workflows/deploy.yml`** runs on push to `main` (or manual `workflow_dispatch`):
   - Strictly depends on `needs: quality_and_security_gates` passing before deploying `ui/` to Firebase Hosting (`ypyai-95ca7`).

## 3. Exact Promotion & Rollback Commands

- **Promote verified preview to primary live site (`https://ypyai-95ca7.web.app` — requires user approval):**
  ```powershell
  & "C:\Users\info\AppData\Roaming\npm\firebase.cmd" deploy --only hosting
  ```
- **Or merge Pull Request to `main`:** Triggers `.github/workflows/deploy.yml` after all CI tests and security gates pass.
- **Launch Always-On Docker API Backend:**
  ```bash
  export NOURAI_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  docker compose up --build -d
  ```
- **Rollback Frontend:** Firebase Console -> Hosting -> Release History -> Rollback (instant, zero downtime).
- **Rollback Local Model Weights:** `python scripts/rollback_release.py --version v1.2.0`.

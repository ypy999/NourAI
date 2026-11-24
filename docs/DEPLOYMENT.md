# Jebrex AI — Cloud & 24/7 Production Deployment Guide

## 1. Separated Service Architecture

Jebrex AI deliberately separates responsibilities so the public web experience remains fast, secure, and available **24/7 at \$0/month** while heavy compute scales independently:

| Layer | Responsibility | Verified Hosting Target | 24/7 Availability & Free-Tier Honesty |
|---|---|---|---|
| **Frontend Static UI** (`ui/`) | Chat UI (RTL AR / EN / FR), `/about.html`, `/privacy.html`, `/report-bug.html`, local tools & browser RAG | **Firebase Hosting** (`ypyai-95ca7`) | **True 24/7 Global CDN** (Spark Free: 10 GB storage, 360 MB/day transfer, automatic TLS/HTTPS). |
| **Backend REST API** (`api/backend.py`, `deployment/production.py`) | Auth (PBKDF2 + HMAC), SQLite sessions/memory/bug reports, rate limiting, live SSE neural streaming (`/api/v1/chat/stream`), `/api/v1/model/status` | **Docker / Self-Hosted** or **Oracle Always-Free / Cloud Run / Fly.io** | **Honest Free-Tier Warning:** Render Free (`render.yaml`) sleeps after **15m idle** (~50s cold start) and Hugging Face Spaces Free sleeps after **48h idle** — **NEITHER is true 24/7**. True 24/7 at lowest cost requires Oracle Cloud Always-Free ARM VM (`\$0/mo`), Self-Hosted Docker + Cloudflare Tunnel (`\$0/mo`), or paid Fly.io/Cloud Run (`\$5–\$25/mo`, requires user approval). |
| **Database** | Users, sessions, memories, local RAG chunks, bug reports | **SQLite WAL** (`data/production/*.sqlite3`) with verified snapshot backups | Upgrade to managed PostgreSQL / Turso / Neon if multi-replica horizontal scaling is needed. |
| **Model Storage & Datasets** | Checkpoints (`.pt`), memmap shards (`.bin`), multi-billion corpora | **Google Drive / GCS / Cloudflare R2** (outside Git) + `manifests/artifacts_manifest.json` (SHA-256) | Downloaded lazily with SHA-256 verification via `scripts/artifact_storage.py`. |
| **GPU Training Jobs** | Pretraining (125M–350M), SFT, DPO | **Google Colab / Kaggle / Cloud GPU** (`notebooks/JebrexAI_Cloud_GPU_Training.ipynb`) | Resumable portable SHA-256 bundles survive session timeouts. |

---

## 2. Live Hosting Deployment (Firebase Hosting)

### Safe Preview Channel Deploy (Non-Destructive Verification)
```bash
firebase hosting:channel:deploy jebrex-prod --expires 30d
```

### Promote to Production Live Channel (`https://ypyai-95ca7.web.app`)
Run ONLY after preview verification and explicit maintainer approval:
```bash
firebase deploy --only hosting
```

### Rollback Procedure (Frontend)
- Firebase Hosting keeps versioned release history. Roll back instantly via the Firebase Console (**Hosting -> Release History -> Rollback**) or re-deploy a tagged Git commit (`git checkout <tag> && firebase deploy --only hosting`).

---

## 3. Running the Backend API with Docker

```bash
# Set a strong random signing secret (never commit to Git)
export NOURAI_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"

# Build and launch container (non-root user, port 7860, built-in /api/v1/health check)
docker compose up --build -d

# Verify health and honest model readiness
curl -fsS http://127.0.0.1:7860/api/v1/health
curl -fsS http://127.0.0.1:7860/api/v1/model-info
```

Connect the live Firebase static UI to your running HTTPS backend via query string:
`https://<your-firebase-url>/?api=https://<your-api-domain>`

---

## 4. Security, Monitoring, Backup & Rollback Controls

- **HTTPS / TLS:** Enforced automatically by Firebase Hosting CDN and supported natively in `deployment/production.py` (`TLSv1.3` + `Strict-Transport-Security`).
- **Headers:** `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy`, and strict CSP on API responses.
- **Rate Limiting:** Enforced per IP/user on `/api/v1/auth/guest`, `/api/v1/auth/token`, `/api/v1/users/register`, `/api/v1/bug-reports`, and authenticated routes (`429 Too Many Requests`).
- **Health Monitoring:** Poll `GET /api/v1/health` (HTTP 200, JSON `status: "ok"`, memory telemetry) and `GET /` on Firebase Hosting every 5 minutes using an external uptime monitor (e.g., UptimeRobot Free / BetterStack Free).
- **Database Backup:** `deployment/production.py` (`ProductionDatabase.backup_to`) performs safe SQLite online backups; schedule daily backups to encrypted off-host storage.
- **Model Release Rollback:** Use `scripts/rollback_release.py --version <v1.x.y>` for local checkpoint rollback; `scripts/artifact_storage.py` enforces retention that **never** deletes the last verified healthy checkpoint.
- **CI/CD Gating:** `.github/workflows/deploy.yml` blocks automated deployment unless `.github/workflows/tests.yml` (unit/integration tests + `0 CRITICAL / 0 HIGH` security audit) passes first.

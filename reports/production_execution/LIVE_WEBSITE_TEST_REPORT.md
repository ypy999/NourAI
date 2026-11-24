# 10. Live Website Test Report (`LIVE_WEBSITE_TEST_REPORT.md`)

- **Tested Live URL:** **`https://ypyai-95ca7--jebrex-prod-vpozlkst.web.app`**
- **Test Timestamp (UTC):** `2026-11-24T22:19:28Z`
- **Raw Verification Artifact:** `reports/production_execution/live_website_probe.json`
- **Overall Verdict:** **`PASS (7/7 Live HTTPS Routes Verified, Average Latency 236.65 ms, 0 External LLM Leaks)`**

## 1. Live HTTPS Route Verification Matrix

| Route | HTTP Status | Latency (ms) | Payload Size | Content-Type | Security Headers (`HSTS` / `XFO` / `Nosniff`) | Content & Integrity Checks | Verdict |
|---|:---:|---:|---:|---|:---:|---|:---:|
| `/` | **`200`** | `431.26 ms` | `143,690 B` | `text/html; charset=utf-8` | All Present (`DENY`, `nosniff`, HSTS preload) | `Jebrex AI`, `dir="rtl"`, `model-readiness-banner`, page nav links, **0 secret external LLM leaks** | **PASS** |
| `/about.html` | **`200`** | `165.04 ms` | `7,845 B` | `text/html; charset=utf-8` | All Present + `no-cache` | `عن Jebrex AI`, `125M`, `350M`, `RoPE` | **PASS** |
| `/privacy.html` | **`200`** | `154.04 ms` | `6,694 B` | `text/html; charset=utf-8` | All Present + `no-cache` | `سياسة الخصوصية`, `Local-First`, `DELETE /api/v1/users/me` | **PASS** |
| `/report-bug.html` | **`200`** | `208.42 ms` | `8,274 B` | `text/html; charset=utf-8` | All Present + `no-cache` | `الإبلاغ عن خطأ`, `bug-form`, `/api/v1/bug-reports` | **PASS** |
| `/manifest.json` | **`200`** | `209.15 ms` | `389 B` | `application/json` | All Present | `Jebrex AI` PWA metadata | **PASS** |
| `/robots.txt` | **`200`** | `278.21 ms` | `89 B` | `text/plain; charset=utf-8` | All Present | `User-agent` valid | **PASS** |
| `/icon.svg` | **`200`** | `210.44 ms` | `301 B` | `image/svg+xml` | All Present + `max-age=86400` | Valid `<svg` vector icon | **PASS** |

## 2. Backend API Status & Honest Disclosure

- **Live Static Frontend:** 100% online 24/7 via Firebase Hosting Global CDN. Displays honest connection badge (`واجهة ثابتة 24/7 (بدون خادم خلفي)`) when no backend URL is linked, and supports connecting any self-hosted/cloud HTTPS backend via `?api=https://<your-api-domain>`.
- **Legacy Render URL (`https://nourai-studio.onrender.com/api/v1/health`):** Probed live at **HTTP 404** (`372 ms`) — not deployed.
- **Local Full-Stack HTTP Server (`deployment/production.py` + `api/backend.py`):** Verified live in `reports/production_execution/local_api_perf_probe.json` — all 7 endpoints (`/`, `/about.html`, `/privacy.html`, `/report-bug.html`, `/api/v1/health`, `/api/v1/model-info`, `POST /api/v1/bug-reports`) + authenticated Arabic/English/French neural chat returned **HTTP 200/201** with **`0%` error rate**.

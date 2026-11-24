# 11. Cost & Resource Report (`COST_AND_RESOURCE_REPORT.md`)

- **Reporting Month:** `2026-11`
- **Actual Monetary Spend to Date:** **`\$0.00 / month`** (zero paid cloud charges incurred)

## 1. Current Active Resource & Free-Tier Utilization

| Service | Plan / Tier | Actual Cost | Free-Tier Limits | Current Usage | Compliance |
|---|---|---:|---|---|---|
| **GitHub Repository & Actions** | Free Public Repo | `\$0.00` | Unlimited public Actions minutes, soft repo limit `< 1 GB` | Repo `~175 MiB` (0 new `.pt` added) | Compliant |
| **Firebase Hosting (`ypyai-95ca7`)** | Spark Free Plan | `\$0.00` | `10 GB` storage, `360 MB/day` data transfer, global TLS CDN | `ui/` static bundle `< 0.25 MiB` (`< 0.01%` of limit) | Compliant (True 24/7) |
| **Local Development Machine** | Existing Windows Host | `\$0.00` | Intel i3-10110Y (`2C/4T`), `7.89 GiB` RAM, `C:` `6.71 GiB` free | `314.68 MiB` indexed artifacts (`263 MiB` RSS runtime) | Within safe ceiling |
| **Render Web Service** | Not deployed | `\$0.00` | Free tier sleeps after 15m idle (750 hrs/mo) | `0` (HTTP 404) | N/A |

## 2. Transparent Cost Estimates for Future Production Milestones (Requires Explicit User Approval Before Spend)

| Milestone | Required Infrastructure | Estimated Budget (USD) | Expected Outcome |
|---|---|---:|---|
| **Always-On 24/7 API Backend (Small Scale)** | Oracle Cloud Always-Free ARM VM (`Docker`) **OR** Fly.io/VPS (`2 vCPU / 2–4 GB RAM`) | `\$0` (Oracle Free) – `\$10–\$25/mo` | True 24/7 `/api/v1/*` neural API + SQLite/Postgres without 15-min Render sleep |
| **Cloud Storage for Datasets & Checkpoints** | Cloudflare R2 (`10 GB` free, `\$0` egress) or GCS Standard (`\$0.02/GB/mo`) | `\$0 – \$5/mo` (up to `200 GB`) | Stores licensed corpora & `125M–350M` portable SHA-256 checkpoint bundles outside Git |
| **Level-1 Foundation Pretraining (`jebrex-125m`, `3B` tokens)** | Cloud GPU (`1x A100 80GB` or `L40S`, ~30–50 GPU-hours) or Colab Pro+ | `\$50 – \$150` one-time | Genuine fluent trilingual base language model (still below frontier LLMs) |
| **Level-2 Conversational Model (`jebrex-350m`, `10B–50B` tokens + Human SFT/DPO)** | Multi-GPU (`4x–8x A100/H100`) + human multilingual annotation | `\$1,500 – \$15,000+` | Coherent trilingual assistant with real instruction following & tool reliability |
| **Frontier ChatGPT / Gemini Parity** | `7B–70B+` parameters, `> 2T` tokens, dedicated H100 clusters, full RLHF/eval teams | `\$500,000+` | Frontier-class competitive parity (explicitly requires institutional scale) |

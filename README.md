# Jebrex AI — Independent Multilingual AI Platform (Arabic · English · French)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ypy999/NourAI/blob/comprehensive_project_audit/notebooks/JebrexAI_Cloud_GPU_Training.ipynb)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**Jebrex AI** (evolved transparently on top of the `NourAI` repository) is an open, local-first, cloud-optimized engineering platform for building an **independent multilingual language model** supporting **Arabic, English, and French** — with strict scientific honesty, zero secret external LLM substitution, and automated security/quality gates.

> **Honest Readiness Disclosure:**
> All engineering layers — data readiness gates & versioning, multilingual byte-level BPE tokenizer (`32K–64K` ready), modern `125M–350M` Foundation Transformer (`RoPE`, `RMSNorm`, `SwiGLU`, `GQA + KV-Cache`, `Gradient Checkpointing`), cloud-resumable trainer with SHA-256 portable bundles, response-masked SFT + DPO + cited RAG pipeline, hardened REST API, 24/7 static Web UI, Windows desktop app, and Android client scaffold — are **implemented and verified by automated tests**.
> However, currently bundled neural weights (`843K`–`4.87M` parameters) were trained on laboratory-scale splits; **Jebrex AI does NOT yet match ChatGPT or Google Gemini** until the `125M–350M` foundation architecture is trained on multi-billion licensed multilingual tokens on cloud GPUs.

---

## Key Capabilities & Architecture

| Pillar | Implementation | Verification Status |
|---|---|---|
| **Data Gate & Versioning (Phase A)** | `data/jebrex_data_gate.py`, `data/versioning.py`, `data/manifests/` | `0` cross-split contamination; explicit `DATA_NOT_READY` gate for production scale |
| **Multilingual BPE Tokenizer (Phase B)** | `tokenizer/bpe_tokenizer.py` (`32K–64K` presets, lossless UTF-8) | **64.5%** overall token reduction (`72.25%` on Arabic), **100%** lossless round-trip |
| **Foundation Model v2 (Phase C)** | `model/jebrex_transformer.py` (RoPE, RMSNorm, SwiGLU, GQA, KV-Cache, Grad Checkpointing) | Meta-verified **`124.28M`** (`jebrex-125m`) & **`351.32M`** (`jebrex-350m`); KV-Cache parity `<1e-4` |
| **Cloud Resumable Training (Phase D)** | `training/cloud_trainer.py`, `datasets/efficient_loader.py`, `notebooks/JebrexAI_Cloud_GPU_Training.ipynb` | Exact AdamW + Scheduler + RNG resume + SHA-256 portable bundles + BF16/FP16 on GPU |
| **Conversational Pipeline (Phase E)** | `training/conversational_pipeline.py` (Masked SFT, DPO loss, AST math, cited RAG, `is_synthetic` audit) | Unit-tested gradient flow, citation provenance, and sandboxed math |
| **Cloud Storage & Hygiene** | `docs/CLOUD_STORAGE_STRATEGY.md`, `scripts/artifact_storage.py`, `manifests/artifacts_manifest.json` | SHA-256 manifest verification, lazy cached downloads, safe retention policy |
| **Web UI & Pages (24/7 CDN)** | `ui/index.html`, `ui/about.html`, `ui/privacy.html`, `ui/report-bug.html`, `firebase.json` | RTL Arabic / EN / FR, honest readiness banner, zero secret external LLM calls |
| **Hardened REST API & Apps** | `api/backend.py`, `api/fastapi_app.py`, `deployment/production.py`, `apps/windows/`, `apps/android/` | `0 CRITICAL`, `0 HIGH` open findings in `security/security_audit_v2.py` |
| **CI/CD & Containers** | `.github/workflows/tests.yml`, `.github/workflows/deploy.yml`, `Dockerfile`, `docker-compose.yml` | Gated testing, security scan, non-root container with `/api/v1/health` |

---

## Quick Start

### 1. Run Local Server & Web UI
```bash
python run_nourai.py --no-browser --host 127.0.0.1 --port 7860
```
Open `http://127.0.0.1:7860/` (or `/about.html`, `/privacy.html`, `/report-bug.html`, `/api/v1/health`, `/api/v1/model-info`).

### 2. Run with Docker
```bash
docker compose up --build
```

### 3. Run Automated Tests & Security Audit
```bash
PYTHONPATH=. pytest -v \
  tests/test_jebrex_data_gate.py \
  tests/test_jebrex_storage_and_versioning.py \
  tests/test_jebrex_bpe_tokenizer.py \
  tests/test_jebrex_model_and_training.py \
  tests/test_jebrex_foundation_v2.py \
  tests/test_jebrex_inference_and_chat.py \
  tests/test_jebrex_conversational_and_web.py \
  tests/test_jebrex_apps.py \
  tests/test_phase19_user_interface.py \
  tests/test_stage_b_to_l_certification.py

PYTHONPATH=. python -m security.security_audit_v2
```

### 4. Train `125M–350M` on Cloud GPU (Colab / Kaggle)
Launch [`notebooks/JebrexAI_Cloud_GPU_Training.ipynb`](notebooks/JebrexAI_Cloud_GPU_Training.ipynb) on a T4/L4/A100 GPU runtime with Google Drive checkpoint auto-resume.

---

## Governance & Engineering Reports

- **Storage Strategy:** [`docs/CLOUD_STORAGE_STRATEGY.md`](docs/CLOUD_STORAGE_STRATEGY.md)
- **Deployment & 24/7 Architecture:** [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
- **Production Execution Reports (12 Files):** `reports/production_execution/`
- **Project Governance & Roadmap:** `docs/project_management/`
- **Contributing Guidelines:** [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **License:** [`LICENSE`](LICENSE) (MIT)

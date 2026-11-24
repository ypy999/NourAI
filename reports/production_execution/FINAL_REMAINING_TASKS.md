# 12. Final Remaining Tasks & Next Execution Priorities (`FINAL_REMAINING_TASKS.md`)

- **Audit Date (UTC):** `2026-11-24`
- **Completed Engineering Foundation:** Storage strategy & SHA-256 manifests, Phase A data gate & versioning, Phase B multilingual BPE tokenizer (`32K–64K` ready), Phase C `125M–350M` modern Transformer (`RoPE/RMSNorm/SwiGLU/GQA/KV-Cache/GradCkpt`), Phase D cloud resumable trainer + Colab notebook, Phase E SFT/DPO/RAG/Tool pipeline, hardened zero-secret-LLM Web UI (`/` + `/about.html` + `/privacy.html` + `/report-bug.html`), CI/CD workflows, Docker, live Firebase preview deployment (`7/7` HTTPS routes verified), and GitHub push (`comprehensive_project_audit`).

## 1. Decisions Requiring Explicit User Approval First

1. **Promote Live Production Website (`https://ypyai-95ca7.web.app`):**
   - Review the verified preview deployment at `https://ypyai-95ca7--jebrex-prod-vpozlkst.web.app`.
   - Approve merging PR `comprehensive_project_audit -> main` (`https://github.com/ypy999/NourAI/pull/new/comprehensive_project_audit`) and/or running `firebase deploy --only hosting` to replace the older `dev/v3.7` site.
2. **Optional Storage Cleanup (Reclaim `~43.5 MiB` Local Duplicate Checkpoints):**
   - Approve deduplicating `releases/v1.2.0/nourai_v1_2_model.pt` (`18.61 MiB`, identical SHA-256 copy of `checkpoints/v1_2/nourai_v1_2_model.pt`) and local sanity-run duplicate snapshots once archived to cloud storage.
3. **Budget & Compute Approval for Always-On API & Cloud GPU Training:**
   - Choose between `\$0` (Oracle Always-Free VM / Colab Free GPU experiments) vs paid Cloud Run / GPU budget (`\$50–\$150` for Level-1 `125M` pretraining).

## 2. Remaining Technical Milestones (Ordered by Priority)

| Priority | Task | Why Blocked Today | Verification Gate to Close |
|:---:|---|---|---|
| **P0** | **Ingest Licensed Multi-Billion Multilingual Corpus to Cloud Storage (GCS / R2)** | Local disk has `6.71 GiB` free; current clean split has `121K` tokens (`0.004%` of Level-1 target) | Run `data/jebrex_data_gate.py` & `data/versioning.py` -> `overall_production_readiness_gate == "DATA_READY"` (`>= 3B` tokens, `0` leakage) |
| **P0** | **Retrain `MultilingualBPETokenizer` to Full `32,768` Vocabulary** | Requires multi-hundred-MB licensed AR/EN/FR training corpus | `vocab_size == 32768`, `100%` lossless round-trip, `>= 3.5x` Arabic compression on held-out test |
| **P0** | **Execute Cloud GPU Pretraining for `jebrex-125m` (`124.28M`)** | Local host has no CUDA GPU (`i3-10110Y` CPU only) | Run `notebooks/JebrexAI_Cloud_GPU_Training.ipynb` on A100/L4/T4; export SHA-256 portable bundle; verify held-out test PPL & `evaluation/jebrex_benchmark.py` improvement |
| **P1** | **Deploy True 24/7 Backend API (`Docker Compose` / Cloud Run / Oracle Free VM)** | No always-on backend host provisioned (`render.yaml` Free sleeps after 15m; Render URL currently 404) | Live HTTPS `GET /api/v1/health` + `POST /api/v1/chat` monitored 24/7 with external uptime checks |
| **P1** | **Wire `JebrexTransformer v2` + `MultilingualBPETokenizer` into `inference/engine.py` Checkpoint Loader** | Wait until first trained v2 checkpoint (`125M`) is produced on GPU so v1 legacy compatibility remains untouched | Auto-detect `schema_version == 2` checkpoints and serve with KV-Cache via `/api/v1/chat/stream` |
| **P1** | **Build Signed Android APK from `apps/android/`** | Android SDK, Gradle wrapper, and JDK 17 are not installed on this Windows host | `./gradlew assembleDebug` & `./gradlew test` passing in CI or Android Studio |
| **P2** | **Curate Human-Reviewed Multilingual SFT (`100K`) + DPO (`30K`) Datasets** | Requires human annotation/review across Arabic, English, and French | `validate_conversational_dataset()["production_human_sft_ready"] == True` + independent human + automated benchmark gain vs `v1.2` |
| **P2** | **Close Remaining `MEDIUM` Security Items (`SEC-REM-01..03`)** | Single-file HTML CSP `'unsafe-inline'`; pattern-only prompt-injection filter | Externalize UI JS bundle with strict non-inline CSP + add learned safety classifier |
| **P2** | **Verify GitHub Actions Green Run on `comprehensive_project_audit` & Enable `pip-audit`** | Requires inspecting remote GitHub Actions run after push | Green badge on `.github/workflows/tests.yml` |

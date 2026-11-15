# NourAI — Installation & Operations Guide

## 1. Environment Setup
```powershell
D:\NourAI\.venv\Scripts\python.exe -m pytest tests/
```

## 2. Starting the Production Server
Use `deployment.production.start_live_http_server` with `NourAIBackend` and `ProductionDatabase`.
Database backups are written atomically to `data/production/backups/` with SHA-256 verification sidecars.

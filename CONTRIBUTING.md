# Contributing to Jebrex AI

Thank you for contributing to **Jebrex AI** — an independent, transparent, multilingual (Arabic, English, French) AI project.

## Core Engineering & Scientific Integrity Rules

1. **Zero Secret External LLM Proxying:** Never call OpenAI, Gemini, Claude, Pollinations, or any external LLM API silently to generate answers presented as Jebrex AI's independent neural model.
2. **Evidence Over Claims:** Do not mark a capability, benchmark, or training stage complete without automated tests and verifiable evidence artifacts.
3. **No Secrets or Heavy Weights in Git:**
   - Never commit `.env`, `.pem`, `.key`, service account JSONs, passwords, or API tokens.
   - Never commit new `.pt`, `.bin`, `.safetensors`, `.gguf`, or raw multi-GB corpora to standard Git. Store heavy artifacts externally with SHA-256 manifests (`scripts/artifact_storage.py`).
4. **Strict Split Hygiene:** Never train tokenizers or models on `validation.jsonl` or `test.jsonl`. All new datasets must pass `data/jebrex_data_gate.py` and `data/versioning.py` with `0` cross-split contamination.
5. **Backups & Safe Changes:** Never delete historical reports, audit trails, original datasets, or healthy checkpoints.

## Local Setup & Running Tests

```bash
# Install CPU dependencies
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/base.txt
pip install pytest

# Run Jebrex AI test suite
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

# Run automated security audit (must keep 0 CRITICAL and 0 HIGH)
PYTHONPATH=. python -m security.security_audit_v2
```

## Pull Request Checklist

- [ ] Branch created from the latest integration branch with clean, atomic commits.
- [ ] All automated tests pass locally and in `.github/workflows/tests.yml`.
- [ ] `security/security_audit_v2.py` reports `0 CRITICAL` and `0 HIGH`.
- [ ] No new secrets or unapproved large binaries tracked by `git status`.
- [ ] Documentation and reports updated honestly with observed numbers.

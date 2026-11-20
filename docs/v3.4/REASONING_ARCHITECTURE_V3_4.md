# JABRIX AI v3.4.0 — Advanced Reasoning Engine Architecture (`REASONING_ARCHITECTURE_V3_4.md`)

**Branch:** `dev/v3.4` (`dev/v3.3`, `dev/v3.2`, `dev/v3.1`, and `main` preserved intact for rollback).

## 1. Six Modular Reasoning Engines (`nourai/reasoning_v34/`)
1. **[reasoning_engine.py](file:///d:/NourAI/nourai/reasoning_v34/reasoning_engine.py) (`AdvancedReasoningOrchestrator`):** Lazy-loads all sub-engines on first call, enforces `timeout_seconds` & bounded `AdaptiveCache`, orchestrates dependency-aware stages, and integrates directly with `IntelligentAlgorithmRouter`, `HybridSearchEngine`, `SemanticMemoryEngine`, `ContextManager`, `ResultVerificationEngine`, and `ResourceAwareScheduler`.
2. **[causal_engine.py](file:///d:/NourAI/nourai/reasoning_v34/causal_engine.py) (`CausalEngine`):** Extracts multilingual (AR/EN/FR) cause-and-effect edges (including decimal numbers like `2.3x`), builds a directed causal graph, identifies `root_causes` & `terminal_effects`, and extracts cycle-safe `causal_chains`.
3. **[hypothesis_engine.py](file:///d:/NourAI/nourai/reasoning_v34/hypothesis_engine.py) (`HypothesisEngine`):** Generates hypotheses from causal chains, top evidence, or caller inputs; compares against evidence; explicitly classifies each as `SUPPORTED`, `CONTESTED`, `REJECTED_CONTRADICTED`, or `REJECTED_UNSUPPORTED`.
4. **[planning_engine.py](file:///d:/NourAI/nourai/reasoning_v34/planning_engine.py) (`ConstraintPlanningEngine`):** Decomposes compound goals into a dependency-ordered reasoning DAG, evaluates `latency_budget_ms` & `ram_budget_mb` constraints, prunes optional stages under tight latency budgets (`<45 ms`), and computes `plan_score`.
5. **[uncertainty_engine.py](file:///d:/NourAI/nourai/reasoning_v34/uncertainty_engine.py) (`UncertaintyEngine`):** Separates `epistemic_uncertainty` (missing evidence) from `conflict_uncertainty` (contradictions), calibrates confidence (`HIGH`/`MEDIUM`/`LOW`), and lists explicit `missing_evidence` gaps.
6. **[self_verification.py](file:///d:/NourAI/nourai/reasoning_v34/self_verification.py) (`SelfVerificationEngine`):** Extracts atomic claims, matches against retrieved evidence, detects evidence-vs-evidence and claim-vs-evidence contradictions, and flags unsupported/hallucinated claims (`grounded_ratio`, `hallucination_risk`).

## 2. Backward-Compatible API Integration (`nourai/api/v3_routes.py`)
- `POST /api/v3/reasoning/analyze` — Executes the full multi-stage v3.4 reasoning pipeline (lazy-initialized via `get_reasoning_v34()`).
- `POST /api/v3/reasoning/verify` — Runs direct claim extraction, contradiction detection, and hallucination risk scoring.

# NourAI v3.2 — Advanced Reasoning Engine (`REASONING_ENGINE.md`)

Implementation: [reasoning.py](file:///d:/NourAI/nourai/agent/reasoning.py).

## 1. v3.2 Reasoning Capabilities
Upgraded while maintaining 100% backward compatibility with v3.0/v3.1 callers:
- **Constraint Analysis (`constraints`):** Explicitly enumerates language, least-privilege permission, human-approval, and missing-input constraints before planning.
- **Falsifiable Hypothesis Generation (`hypotheses`):** Formulates testable strategy hypotheses with explicit `verification_criterion` and `prior_confidence`.
- **Alternative Solution Evaluation (`alternatives`):** Scores candidate strategies and records explicit rationale.
- **Memory Contradiction Detection (`contradictions`):** Flags conflicting retrieved memory values and automatically adjusts `uncertainty_score` and `confidence`.
- **Multi-Step Mathematical Pre-Verification (`math_verification`):** Counts operators and routes expressions to the deterministic AST `python_sandbox`.
- **Planning Quality Score (`planning_quality_score`):** Computes a bounded `[0.1, 1.0]` quality metric accounting for strategy score and detected contradictions.

## 2. Verification
- **25 / 25** multilingual reasoning tasks (Arabic, English, French) independently verified in [v32_benchmark.py](file:///d:/NourAI/nourai/evaluation/v32_benchmark.py) for exact mathematical outcomes, non-empty constraints, and falsifiable hypotheses.

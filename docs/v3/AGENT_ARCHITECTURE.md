# NourAI v3.0 — Autonomous Agent Architecture (`AGENT_ARCHITECTURE.md`)

## 1. Six Modular Agent Core Engines (`nourai/agent/`)
1. **Perception Engine (`perception.py`):** Detects language (`arabic`, `english`, `french`), primary intent, instruction type (`simple_query`, `tool_task`, `multi_step_workflow`, `destructive_request`), extracts math expressions and file paths, and scores complexity.
2. **Reasoning Engine (`reasoning.py`):** Performs evidence-based analysis using retrieved memory and Knowledge Graph relations, scores competing strategies (`AlternativeOption`), and computes explicit `uncertainty_score` and `confidence`.
3. **Planning Engine (`planning.py`):** Decomposes objectives into dependency-ordered `PlanStep` DAGs bounded by `max_steps` (prevents infinite loops).
4. **Task Manager (`task_manager.py`):** Persists tasks and full step-by-step execution traces in SQLite (`v3_agent_tasks`), managing lifecycle states (`PENDING`, `PLANNING`, `AWAITING_APPROVAL`, `RUNNING`, `VERIFYING`, `RECOVERING`, `COMPLETED`, `FAILED`, `CANCELLED`) and cancellation tokens.
5. **Critic Engine (`critic.py`):** Validates plan DAGs before execution, verifies tool outputs after execution, detects Knowledge Graph contradictions, and computes measurable `quality_score`.
6. **Recovery Engine (`recovery.py`):** Diagnoses failures by error code; enforces hard abort on security violations, pauses at `AWAITING_APPROVAL` for unapproved destructive ops, executes bounded retries on transient timeouts, or switches to alternative `knowledge_retrieval` fallback.

## 2. 13-Stage Autonomous Workflow (`orchestrator.py`)
`User Request -> Perception -> Context Retrieval -> Planning -> Task Decomposition -> Tool Selection -> Authorization -> Execution -> Verification -> Recovery (if needed) -> Final Result -> Evaluation -> Selective Memory Update`.

- **No infinite loops:** Hard `max_steps` and `max_retries` caps enforced at every iteration.
- **Human approval checkpoints:** Destructive tools pause in `AWAITING_APPROVAL` until explicitly resumed via `approve_task()` or `POST /api/v3/agent/tasks/{id}/approve`.
- **Selective memory update:** Episodes are stored only on verified task completion and pass through the privacy filter.

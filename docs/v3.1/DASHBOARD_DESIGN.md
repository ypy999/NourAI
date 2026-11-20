# NourAI v3.1 — Control Center Dashboard Design (`DASHBOARD_DESIGN.md`)

Frontend: [control_center.html](file:///d:/NourAI/ui/control_center.html) · Route: `GET /control-center` ([backend.py](file:///d:/NourAI/api/backend.py)) · Summary API: `GET /api/v3/control-center/summary` ([v3_routes.py](file:///d:/NourAI/nourai/api/v3_routes.py)).

## 1. Design Principles
- **Zero External Dependencies:** Pure HTML5, CSS Grid, and vanilla JavaScript — works 100% offline/local-first without CDN calls.
- **Multilingual RTL/LTR Support:** Live direction switcher (`LTR` for EN/FR, `RTL` for Arabic).
- **Full `v1.2.0` Compatibility:** Mounted at `/control-center` alongside the existing v1 chat interface at `/`, reusing `/api/v1/auth/guest` and Bearer token authentication.

## 2. Implemented Dashboard Panels (All 10 Modules)
1. **Agent Task Monitor:** Submit new AR/EN/FR objectives and inspect recent task statuses.
2. **Task Execution Timeline:** Displays full step-by-step execution trace JSON for any selected task.
3. **Planning Visualization:** Displays the generated DAG plan (`steps`, `depends_on`, `priority`, `tool_name`).
4. **Human Approval Queue:** Lists `AWAITING_APPROVAL` destructive tasks with one-click **Approve** (`POST /api/v3/agent/tasks/{id}/approve`) and **Cancel** (`POST /api/v3/agent/tasks/{id}/cancel`) controls.
5. **CPU & RAM Hardware Monitor:** Live RSS MiB, CPU core count, PyTorch thread count, and degraded-mode badge.
6. **Neural Model Metrics:** Parameter count, weights size in MiB, context length limit, and execution device.
7. **Tool Activity & Permissions:** Live audit log of recent tool invocations with status and latency.
8. **4-Tier Memory Inspector:** Lists stored memories across tiers with consistency badge and one-click user deletion (`DELETE /api/v3/memory/{id}`).
9. **Evaluation & Self-Evaluation:** One-click benchmark trigger (`POST /api/v3/evaluation/run`) + failure diagnostics.
10. **Error & Recovery Logs:** Displays classified failure causes and actionable recommendations from `SelfEvaluator`.

# NourAI v3.2 — Long-Term Project Management (`PROJECT_MANAGER.md`)

Implementation: [project_manager.py](file:///d:/NourAI/nourai/projects/project_manager.py) · API: `GET/POST /api/v3/projects`, `POST /api/v3/projects/{id}/resume`, `POST /api/v3/projects/{id}/approve-task`.

## 1. Persistent Data Model (`SQLite WAL`)
- `v32_projects`: project metadata, objective, status (`ACTIVE`, `IN_PROGRESS`, `AWAITING_APPROVAL`, `COMPLETED`).
- `v32_milestones`: ordered project phases linked to `project_id`.
- `v32_project_tasks`: individual tasks with `depends_on_json`, `requires_approval`, `approved`, `agent_task_id`, `result_json`, and `error`.
- **Dedicated Project Memory:** Automatically scopes project facts and completed task episodes under `project:<project_id>` in `AdvancedMemoryManager`.

## 2. Restart Resilience & Authorization
- `resume_project()` evaluates completed dependency IDs from SQLite, executes ready unblocked tasks via `AutonomousAgentOrchestrator`, pauses any `requires_approval=True` task in `AWAITING_APPROVAL`, and updates milestone and project completion percentages (`progress.completion_pct`).
- Verified across **25 / 25** project management checks in [v32_benchmark.py](file:///d:/NourAI/nourai/evaluation/v32_benchmark.py), including creating a brand-new `ProjectManager` + `AdvancedMemoryManager` instance on the same SQLite database to simulate complete application restarts.

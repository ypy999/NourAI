"""Mission 2 — Persistent Long-Term Project Manager with Milestones, Task Dependencies,
Project Memory, Blocked-Task Detection, Approval Gates, Progress Reports, and Restart Resumption."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Any

from nourai.agent.orchestrator import AutonomousAgentOrchestrator
from nourai.memory.advanced_memory import AdvancedMemoryManager


class ProjectManager:
    """SQLite WAL-persisted long-term project execution manager surviving complete application restarts."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        memory_manager: AdvancedMemoryManager,
        orchestrator: AutonomousAgentOrchestrator | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = memory_manager
        self.orchestrator = orchestrator
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v32_projects (
                    project_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v32_milestones (
                    milestone_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    order_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v32_project_tasks (
                    ptask_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    milestone_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    depends_on_json TEXT NOT NULL,
                    requires_approval INTEGER NOT NULL,
                    approved INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    agent_task_id TEXT,
                    result_json TEXT,
                    error TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )

    def create_project(self, user_id: str, name: str, objective: str) -> dict[str, Any]:
        pid = f"prj_{secrets.token_hex(6)}"
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO v32_projects(project_id, user_id, name, objective, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (pid, user_id, name.strip(), objective.strip(), "ACTIVE", now, now),
            )
        # Seed dedicated project memory scope
        self.memory.store(
            f"project:{pid}",
            "semantic",
            f"Project '{name.strip()}' objective: {objective.strip()}",
            source=f"project:{pid}",
            confidence=0.95,
        )
        return self.get_project(pid, user_id=user_id) or {}

    def add_milestone(self, project_id: str, title: str, *, order_index: int = 1) -> dict[str, Any]:
        mid = f"ms_{secrets.token_hex(5)}"
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO v32_milestones(milestone_id, project_id, title, order_index, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (mid, project_id, title.strip(), int(order_index), "PENDING", now),
            )
        return {"milestone_id": mid, "project_id": project_id, "title": title.strip(), "order_index": int(order_index), "status": "PENDING"}

    def add_task(
        self,
        project_id: str,
        milestone_id: str,
        title: str,
        objective: str,
        *,
        depends_on: list[str] | None = None,
        requires_approval: bool = False,
    ) -> dict[str, Any]:
        tid = f"pt_{secrets.token_hex(5)}"
        now = time.time()
        deps = list(depends_on or [])
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO v32_project_tasks(
                    ptask_id, project_id, milestone_id, title, objective, depends_on_json,
                    requires_approval, approved, status, agent_task_id, result_json, error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tid,
                    project_id,
                    milestone_id,
                    title.strip(),
                    objective.strip(),
                    json.dumps(deps),
                    1 if requires_approval else 0,
                    0,
                    "PENDING",
                    None,
                    None,
                    None,
                    now,
                ),
            )
        return {
            "ptask_id": tid,
            "project_id": project_id,
            "milestone_id": milestone_id,
            "title": title.strip(),
            "objective": objective.strip(),
            "depends_on": deps,
            "requires_approval": bool(requires_approval),
            "status": "PENDING",
        }

    def approve_project_task(self, project_id: str, ptask_id: str) -> bool:
        with self._lock, closing(self._connect()) as conn, conn:
            cur = conn.execute(
                """
                UPDATE v32_project_tasks SET approved = 1, status = 'PENDING', updated_at = ?
                WHERE project_id = ? AND ptask_id = ?
                """,
                (time.time(), project_id, ptask_id),
            )
            return cur.rowcount > 0

    def get_project(self, project_id: str, *, user_id: str | None = None) -> dict[str, Any] | None:
        with closing(self._connect()) as conn, conn:
            prow = conn.execute(
                "SELECT project_id, user_id, name, objective, status, created_at, updated_at FROM v32_projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            if not prow:
                return None
            if user_id and prow[1] != user_id:
                return None
            mrows = conn.execute(
                "SELECT milestone_id, title, order_index, status FROM v32_milestones WHERE project_id = ? ORDER BY order_index ASC",
                (project_id,),
            ).fetchall()
            trows = conn.execute(
                """
                SELECT ptask_id, milestone_id, title, objective, depends_on_json, requires_approval,
                       approved, status, agent_task_id, result_json, error, updated_at
                FROM v32_project_tasks WHERE project_id = ? ORDER BY updated_at ASC
                """,
                (project_id,),
            ).fetchall()

        tasks = [
            {
                "ptask_id": r[0],
                "milestone_id": r[1],
                "title": r[2],
                "objective": r[3],
                "depends_on": json.loads(r[4]),
                "requires_approval": bool(r[5]),
                "approved": bool(r[6]),
                "status": r[7],
                "agent_task_id": r[8],
                "result": json.loads(r[9]) if r[9] else None,
                "error": r[10],
                "updated_at": float(r[11]),
            }
            for r in trows
        ]
        completed_ids = {t["ptask_id"] for t in tasks if t["status"] == "COMPLETED"}
        blocked_tasks = [
            t["ptask_id"]
            for t in tasks
            if t["status"] in ("PENDING", "AWAITING_APPROVAL")
            and (
                any(d not in completed_ids for d in t["depends_on"])
                or (t["requires_approval"] and not t["approved"])
            )
        ]
        total_t = len(tasks)
        done_t = len(completed_ids)
        return {
            "project_id": prow[0],
            "user_id": prow[1],
            "name": prow[2],
            "objective": prow[3],
            "status": prow[4],
            "created_at": float(prow[5]),
            "updated_at": float(prow[6]),
            "milestones": [{"milestone_id": m[0], "title": m[1], "order_index": int(m[2]), "status": m[3]} for m in mrows],
            "tasks": tasks,
            "blocked_tasks": blocked_tasks,
            "progress": {
                "total_tasks": total_t,
                "completed_tasks": done_t,
                "blocked_count": len(blocked_tasks),
                "completion_pct": round((done_t / max(1, total_t)) * 100.0, 1) if total_t else 0.0,
            },
            "project_memories": self.memory.list_memories(f"project:{project_id}", limit=20),
        }

    def list_projects(self, user_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT project_id FROM v32_projects WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [p for (pid,) in rows if (p := self.get_project(pid, user_id=user_id)) is not None]

    def resume_project(
        self,
        project_id: str,
        *,
        user_id: str,
        max_tasks_to_run: int = 20,
    ) -> dict[str, Any]:
        """Execute all ready, unblocked tasks in dependency order and persist progress for restartability."""
        if self.orchestrator is None:
            raise RuntimeError("ProjectManager requires an AutonomousAgentOrchestrator to resume execution")
        proj = self.get_project(project_id, user_id=user_id)
        if not proj:
            raise KeyError(f"project not found: {project_id}")

        executed = 0
        while executed < max_tasks_to_run:
            proj = self.get_project(project_id, user_id=user_id)
            if not proj:
                break
            tasks = proj["tasks"]
            completed_ids = {t["ptask_id"] for t in tasks if t["status"] == "COMPLETED"}

            ready = [
                t for t in tasks
                if t["status"] == "PENDING"
                and all(d in completed_ids for d in t["depends_on"])
            ]
            if not ready:
                break

            made_progress = False
            for pt in ready:
                if executed >= max_tasks_to_run:
                    break
                if pt["requires_approval"] and not pt["approved"]:
                    with self._lock, closing(self._connect()) as conn, conn:
                        conn.execute(
                            "UPDATE v32_project_tasks SET status = 'AWAITING_APPROVAL', updated_at = ? WHERE ptask_id = ?",
                            (time.time(), pt["ptask_id"]),
                        )
                    continue

                agent_task = self.orchestrator.run_task(
                    user_id,
                    pt["objective"],
                    approved=bool(pt["approved"]),
                )
                status = agent_task.get("status", "FAILED")
                with self._lock, closing(self._connect()) as conn, conn:
                    conn.execute(
                        """
                        UPDATE v32_project_tasks
                        SET status = ?, agent_task_id = ?, result_json = ?, error = ?, updated_at = ?
                        WHERE ptask_id = ?
                        """,
                        (
                            status,
                            agent_task.get("task_id"),
                            json.dumps(agent_task.get("result"), ensure_ascii=False),
                            agent_task.get("error"),
                            time.time(),
                            pt["ptask_id"],
                        ),
                    )
                if status == "COMPLETED":
                    resp = (agent_task.get("result") or {}).get("response", "")
                    try:
                        self.memory.store(
                            f"project:{project_id}",
                            "episodic",
                            f"Completed task '{pt['title']}': {resp[:240]}",
                            source=f"ptask:{pt['ptask_id']}",
                        )
                    except ValueError:
                        pass
                executed += 1
                made_progress = True

            if not made_progress:
                break

        # Recompute milestone and project statuses
        proj = self.get_project(project_id, user_id=user_id)
        if proj:
            with self._lock, closing(self._connect()) as conn, conn:
                for ms in proj["milestones"]:
                    ms_tasks = [t for t in proj["tasks"] if t["milestone_id"] == ms["milestone_id"]]
                    if ms_tasks and all(t["status"] == "COMPLETED" for t in ms_tasks):
                        m_stat = "COMPLETED"
                    elif any(t["status"] == "AWAITING_APPROVAL" for t in ms_tasks):
                        m_stat = "AWAITING_APPROVAL"
                    elif any(t["status"] == "COMPLETED" for t in ms_tasks):
                        m_stat = "IN_PROGRESS"
                    else:
                        m_stat = "PENDING"
                    conn.execute("UPDATE v32_milestones SET status = ? WHERE milestone_id = ?", (m_stat, ms["milestone_id"]))

                all_tasks = proj["tasks"]
                if all_tasks and all(t["status"] == "COMPLETED" for t in all_tasks):
                    p_stat = "COMPLETED"
                elif any(t["status"] == "AWAITING_APPROVAL" for t in all_tasks):
                    p_stat = "AWAITING_APPROVAL"
                else:
                    p_stat = "IN_PROGRESS"
                conn.execute("UPDATE v32_projects SET status = ?, updated_at = ? WHERE project_id = ?", (p_stat, time.time(), project_id))

        return self.get_project(project_id, user_id=user_id) or {}

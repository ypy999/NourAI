"""Specialized Agent 4: Memory Agent (Strictly User-Isolated SemanticMemoryEngine)."""
from __future__ import annotations

from nourai.algorithm_hub import SemanticMemoryEngine
from .base import AgentOutput, AgentRoleSpec, AgentTaskRequest, BaseSpecializedAgent


class MemorySpecialistAgent(BaseSpecializedAgent):
    spec = AgentRoleSpec(
        name="memory_agent",
        role="memory",
        description="Retrieves and audits strictly user-isolated semantic, episodic, and procedural memories.",
        allowed_permissions=("knowledge:read",),
        timeout_seconds=4.0,
    )

    def __init__(self, *, semantic_memory: SemanticMemoryEngine):
        self.semantic_memory = semantic_memory

    def run(self, request: AgentTaskRequest) -> AgentOutput:
        hits = self.semantic_memory.retrieve_contextual(
            request.user_id,
            request.objective,
            session_key=request.session_key,
            top_k=5,
        )
        consistency = self.semantic_memory.memory.check_consistency(request.user_id)
        facts = [{"claim": m["content"], "memory_id": m["memory_id"], "tier": m["tier"]} for m in hits]
        summary = " ".join(m["content"].rstrip(".") + "." for m in hits[:2]) if hits else "No matching user memories."
        conf = 0.85 if (hits and consistency["consistent"]) else (0.45 if hits else 0.2)
        return AgentOutput(
            agent_name=self.spec.name,
            ok=True,
            summary=summary,
            facts=facts,
            evidence=[{"text": m["content"], "source": f"memory:{m['tier']}"} for m in hits],
            confidence=conf,
            metadata={"consistency": consistency, "user_id_isolated": request.user_id},
        )

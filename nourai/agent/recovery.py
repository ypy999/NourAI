"""F. Recovery Engine — Failure diagnosis, safe bounded retry, alternative planning, and recovery reporting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from nourai.tools.tool_engine import ToolResult
from .planning import PlanStep


@dataclass
class RecoveryDecision:
    diagnosed_cause: str
    action: str  # "retry", "alternative_step", "await_approval", "abort"
    reason: str
    alternative_step: PlanStep | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.alternative_step is not None:
            d["alternative_step"] = self.alternative_step.to_dict()
        return d


class RecoveryEngine:
    """Diagnoses tool failures and decides between safe retry, alternative fallback, or controlled abort."""

    RETRYABLE_CODES = {"TIMEOUT", "RUNTIME_ERROR"}

    def diagnose_and_recover(
        self,
        failed_step: PlanStep,
        result: ToolResult,
        *,
        attempt: int,
        max_retries: int,
        objective: str,
    ) -> RecoveryDecision:
        code = result.error_code or "UNKNOWN"

        if code == "APPROVAL_REQUIRED":
            return RecoveryDecision(
                diagnosed_cause="destructive_operation_unapproved",
                action="await_approval",
                reason="Human approval is mandatory before executing destructive operations.",
            )

        if code in ("PERMISSION_DENIED", "PATH_VIOLATION", "SECRET_ACCESS_DENIED", "SQL_POLICY_VIOLATION", "SANDBOX_VIOLATION", "SSRF_BLOCKED"):
            return RecoveryDecision(
                diagnosed_cause=f"security_policy_block:{code}",
                action="abort",
                reason="Security or authorization boundary cannot be bypassed.",
            )

        if code in self.RETRYABLE_CODES and attempt < max_retries:
            return RecoveryDecision(
                diagnosed_cause=f"transient_failure:{code}",
                action="retry",
                reason=f"Safe bounded retry {attempt + 1}/{max_retries}.",
            )

        # Fallback alternative strategy when a non-security read/search tool fails (e.g., NOT_FOUND or offline web search)
        if failed_step.tool_name in ("file_read", "web_search", "document_analysis"):
            alt = PlanStep(
                step_id=f"{failed_step.step_id}_fallback",
                description="Fallback to local knowledge & memory retrieval",
                tool_name="knowledge_retrieval",
                tool_input={"query": objective, "top_k": 5},
                priority=failed_step.priority,
            )
            return RecoveryDecision(
                diagnosed_cause=f"primary_source_unavailable:{code}",
                action="alternative_step",
                reason=f"Primary tool '{failed_step.tool_name}' failed ({code}); switching to local knowledge_retrieval.",
                alternative_step=alt,
            )

        return RecoveryDecision(
            diagnosed_cause=f"unrecoverable:{code}",
            action="abort",
            reason=f"Exhausted retries ({attempt}/{max_retries}) or non-recoverable error: {result.error}",
        )

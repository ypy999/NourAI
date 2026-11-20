"""JABRIX AI v3.5.0 — Unified Multi-Agent Interface, Role Permissions, and Lazy Agent Registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import threading
from typing import Any, Callable


@dataclass(frozen=True)
class AgentRoleSpec:
    name: str
    role: str
    description: str
    allowed_permissions: tuple[str, ...]
    timeout_seconds: float = 5.0
    parallel_safe: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["allowed_permissions"] = list(self.allowed_permissions)
        return d


@dataclass
class AgentTaskRequest:
    user_id: str
    objective: str
    language: str = "english"
    session_key: str = "default"
    documents: list[str] = field(default_factory=list)
    shared_evidence: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    agent_name: str
    ok: bool
    summary: str
    facts: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.75
    requires_approval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseSpecializedAgent:
    spec: AgentRoleSpec

    def run(self, request: AgentTaskRequest) -> AgentOutput:
        raise NotImplementedError


class LazyAgentRegistry:
    """Thread-safe lazy registry so specialized agents consume zero RAM until first invoked."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._factories: dict[str, tuple[AgentRoleSpec, Callable[[], BaseSpecializedAgent]]] = {}
        self._instances: dict[str, BaseSpecializedAgent] = {}

    def register_lazy(self, spec: AgentRoleSpec, factory: Callable[[], BaseSpecializedAgent]) -> None:
        with self._lock:
            self._factories[spec.name] = (spec, factory)

    def get(self, name: str) -> BaseSpecializedAgent | None:
        inst = self._instances.get(name)
        if inst is not None:
            return inst
        with self._lock:
            inst = self._instances.get(name)
            if inst is not None:
                return inst
            entry = self._factories.get(name)
            if entry is None:
                return None
            _, factory = entry
            inst = factory()
            self._instances[name] = inst
            return inst

    def loaded_count(self) -> int:
        return len(self._instances)

    def catalog(self) -> list[dict[str, Any]]:
        out = []
        for name, (spec, _) in self._factories.items():
            d = spec.to_dict()
            d["loaded"] = name in self._instances
            out.append(d)
        return out

"""JABRIX AI v3.5.0 — Multi-Agent Cognitive Architecture Package."""
from .base import AgentOutput, AgentRoleSpec, AgentTaskRequest, BaseSpecializedAgent, LazyAgentRegistry
from .coding_agent import CodingSpecialistAgent
from .collaboration_engine import CollaborationEngine
from .learning_agent import LearningSpecialistAgent
from .master_controller import MasterController
from .memory_agent import MemorySpecialistAgent
from .planning_agent import PlanningSpecialistAgent
from .prediction_agent import PredictionSpecialistAgent
from .reasoning_agent import ReasoningSpecialistAgent
from .research_agent import ResearchSpecialistAgent
from .verification_agent import VerificationSpecialistAgent

__all__ = [
    "AgentOutput",
    "AgentRoleSpec",
    "AgentTaskRequest",
    "BaseSpecializedAgent",
    "LazyAgentRegistry",
    "ResearchSpecialistAgent",
    "ReasoningSpecialistAgent",
    "CodingSpecialistAgent",
    "MemorySpecialistAgent",
    "PlanningSpecialistAgent",
    "VerificationSpecialistAgent",
    "PredictionSpecialistAgent",
    "LearningSpecialistAgent",
    "CollaborationEngine",
    "MasterController",
]

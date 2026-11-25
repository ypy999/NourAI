"""Jebrex AI Official Identity, Ownership & Self-Knowledge Subsystem."""

from knowledge.identity.guard import (
    IdentityConsistencyGuard,
    build_official_system_prompt,
    classify_identity_query,
    compose_grounded_identity_answer,
    detect_identity_language,
)
from knowledge.identity.manager import (
    DEFAULT_IDENTITY_PATH,
    IdentityVersionManager,
    load_identity_registry,
    validate_identity_record,
)

__all__ = [
    "DEFAULT_IDENTITY_PATH",
    "IdentityConsistencyGuard",
    "IdentityVersionManager",
    "build_official_system_prompt",
    "classify_identity_query",
    "compose_grounded_identity_answer",
    "detect_identity_language",
    "load_identity_registry",
    "validate_identity_record",
]

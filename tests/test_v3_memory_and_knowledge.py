"""Comprehensive tests for NourAI v3.0 Four-Tier Memory Architecture and Integrated Knowledge Graph."""
from __future__ import annotations

from pathlib import Path
import time
import xml.etree.ElementTree as ET
import pytest

from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager


def test_four_tier_memory_lifecycle_dedup_ttl_privacy_and_consistency(tmp_path: Path):
    mm = AdvancedMemoryManager(tmp_path / "mem.sqlite3")
    uid = "user_v3"

    # 1. Working memory
    mm.working.put("t1", "intent", "analysis")
    assert len(mm.working.get_context("t1")) == 1
    mm.working.clear("t1")
    assert mm.working.get_context("t1") == []

    # 2. Semantic memory + deduplication
    m1 = mm.store(uid, "semantic", "NourAI uses SQLite WAL for local persistence.", source="docs")
    m2 = mm.store(uid, "semantic", "  nourai uses sqlite wal for local persistence. ", source="docs_v2")
    assert m1.memory_id == m2.memory_id

    # 3. Episodic & Procedural memory
    ep = mm.store_episode(uid, "Calculate 10 * 5", "50", success=True, tools_used=["python_sandbox"])
    proc = mm.store_procedure(uid, "Safe Release", ["Run tests", "Verify SHA-256", "Tag release"], verified=True)
    assert ep.tier == "episodic"
    assert proc.tier == "procedural"

    # 4. Privacy protection blocks secrets
    with pytest.raises(ValueError, match="privacy policy"):
        mm.store(uid, "semantic", "password=SuperSecretPassword123!")
    with pytest.raises(ValueError, match="privacy policy"):
        mm.store(uid, "semantic", "api_key=sk-1234567890abcdef123456")

    # 5. Semantic TF-IDF retrieval
    hits = mm.semantic_search(uid, "SQLite WAL persistence", top_k=3)
    assert len(hits) >= 1
    assert "SQLite WAL" in hits[0]["content"]

    # 6. TTL expiration & purge
    mm.store(uid, "semantic", "Temporary note expiring immediately", ttl_seconds=0.01)
    time.sleep(0.03)
    purged = mm.purge_expired()
    assert purged >= 1

    # 7. Contradiction / consistency check
    mm.store(uid, "semantic", "ProjectVersion is 1.2.0")
    mm.store(uid, "semantic", "ProjectVersion is 3.0.0")
    check = mm.check_consistency(uid)
    assert check["consistent"] is False
    assert len(check["contradictions"]) >= 1

    # 8. User-controlled deletion
    assert mm.delete_memory(uid, m1.memory_id) is True
    assert mm.clear_user_memories(uid) >= 1


def test_knowledge_graph_bidirectional_contradictions_and_exports(tmp_path: Path):
    kg = KnowledgeGraph(tmp_path / "kg.sqlite3")
    added = kg.ingest_text(
        "NourAI uses PyTorch. The capital of Qatar is Doha. عاصمة فرنسا هي باريس.",
        source="unit_test",
    )
    assert len(added) >= 3

    # Bidirectional traversal
    neighbors = kg.get_neighbors("PyTorch")
    assert neighbors["found"] is True
    assert any(inc["inverse_predicate"] == "used_by" and inc["source_entity"] == "NourAI" for inc in neighbors["incoming"])

    # Contradiction detection on single-valued predicate
    kg.add_relation("Qatar", "has_capital", "Al Khor", source="conflicting_source")
    contradictions = kg.detect_contradictions()
    assert len(contradictions) == 1
    assert contradictions[0]["predicate"] == "has_capital"

    # Synchronized exports: Obsidian/Logseq Markdown, Freeplane OPML 2.0, TheBrain JSON
    md = kg.export_obsidian_logseq_markdown()
    assert "[[NourAI]]" in md and "[[PyTorch]]" in md

    opml_xml = kg.export_freeplane_opml()
    root = ET.fromstring(opml_xml)
    assert root.tag == "opml"

    tb = kg.export_thebrain_json()
    assert tb["format"] == "thebrain_plex_v1"
    assert len(tb["thoughts"]) >= 4
    assert len(tb["links"]) >= 3

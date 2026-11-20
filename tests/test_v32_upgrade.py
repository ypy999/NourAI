"""Comprehensive automated test suite for NourAI v3.2:
Research Agent, Project Persistence across Restarts, Advanced Reasoning, Multimodal Document Intelligence,
7-Domain Zero-Leakage Datasets, Self-Monitoring, Control Center 2.0 API/UI, and 205-Task Benchmark."""
from __future__ import annotations

from pathlib import Path

from api.backend import NourAIBackend
from inference.engine import InferenceEngine
from model.transformer import GPT, GPTConfig
from nourai.documents.multimodal import DocumentIntelligenceEngine, detect_ocr_capabilities
from nourai.evaluation.self_eval import SelfEvaluator
from nourai.evaluation.v32_benchmark import run_205_task_v32_benchmark
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.neural.v32_datasets import build_v32_multidomain_datasets
from nourai.research.research_agent import AutonomousResearchAgent


def test_research_agent_citations_contradictions_and_missing_evidence(tmp_path: Path):
    db = tmp_path / "res.sqlite3"
    docs = DocumentIntelligenceEngine(db)
    mm = AdvancedMemoryManager(db)
    kg = KnowledgeGraph(db)
    agent = AutonomousResearchAgent(db_path=db, documents=docs, memory=mm, knowledge_graph=kg)

    docs.ingest_bytes("u1", "source_a.txt", b"ServerPort is 7860. NourAI uses SQLite WAL.")
    docs.ingest_bytes("u1", "source_b.txt", b"ServerPort is 8080.")

    rep = agent.conduct_research("u1", "ServerPort and SQLite WAL")
    assert len(rep["verified_facts"]) >= 2
    assert len(rep["citations"]) >= 2
    assert len(rep["contradictions"]) >= 1
    assert len(rep["hypotheses"]) >= 1
    assert "## Verified Facts" in rep["markdown_report"]

    rep_missing = agent.conduct_research("u1", "unknown_dark_matter_drive_xyz")
    assert len(rep_missing["missing_evidence"]) >= 1


def test_multimodal_document_intelligence_and_honest_ocr_detection(tmp_path: Path):
    caps = detect_ocr_capabilities()
    assert caps["vision_model_available"] is False

    docs = DocumentIntelligenceEngine(tmp_path / "docs.sqlite3")
    md_doc = docs.ingest_bytes("u1", "spec.md", b"# Spec\n| ColA | ColB |\n| --- | --- |\n| 1 | 2 |\nNourAI v3.2")
    assert md_doc.doc_type == "markdown"
    assert len(md_doc.tables) == 1
    hits = docs.retrieve_with_citations("u1", "NourAI v3.2", top_k=3)
    assert len(hits) == 1 and hits[0]["citation"] == "[doc:spec.md#chunk_1]"


def test_v32_7domain_datasets_zero_leakage(tmp_path: Path):
    info = build_v32_multidomain_datasets(tmp_path / "ds32")
    assert len(info["domain_counts"]) == 7
    assert info["total_unique_records"] >= 200
    assert info["cross_split_leakage"] == 0


def test_control_center_2_api_research_projects_and_monitoring(tmp_path: Path):
    model = GPT(GPTConfig.from_preset("nano"))
    backend = NourAIBackend(engine=InferenceEngine(model), db_path=tmp_path / "api32.sqlite3", rate_limit_per_minute=60)

    st_cc, _, cc_html = backend.handle_request("GET", "/control-center")
    assert st_cc == 200 and "Control Center 2.0" in cc_html

    _, _, guest = backend.handle_request("POST", "/api/v1/auth/guest", body={})
    h = {"Authorization": f"Bearer {guest['access_token']}"}

    # Ingest multimodal doc -> Run Research -> Create & Resume Project -> Inspect Control Center Summary
    s_doc, _, _ = backend.handle_request(
        "POST", "/api/v3/documents/multimodal", headers=h,
        body={"filename": "brief.md", "content": "# Brief\nNourAI v3.2 supports autonomous research."},
    )
    assert s_doc == 201

    s_res, _, p_res = backend.handle_request(
        "POST", "/api/v3/research", headers=h,
        body={"objective": "NourAI v3.2 autonomous research"},
    )
    assert s_res == 201 and len(p_res["report"]["verified_facts"]) >= 1

    s_prj, _, p_prj = backend.handle_request(
        "POST", "/api/v3/projects", headers=h,
        body={
            "name": "Alpha",
            "objective": "Run math milestone",
            "milestones": [{"title": "M1", "tasks": [{"title": "T1", "objective": "Calculate 15 * 4"}]}],
        },
    )
    assert s_prj == 201
    pid = p_prj["project"]["project_id"]

    s_run, _, p_run = backend.handle_request("POST", f"/api/v3/projects/{pid}/resume", headers=h, body={})
    assert s_run == 200 and p_run["project"]["status"] == "COMPLETED"

    s_sum, _, p_sum = backend.handle_request("GET", "/api/v3/control-center/summary", headers=h)
    assert s_sum == 200 and p_sum["version"] == "3.2.0"
    assert p_sum["self_monitoring"]["allow_auto_unverified_model_updates"] is False
    assert len(p_sum["research_reports"]) >= 1 and len(p_sum["projects"]) >= 1


def test_205_task_v32_benchmark(tmp_path: Path):
    rep = run_205_task_v32_benchmark(tmp_path / "b205")
    assert rep["total_tasks"] >= 200
    assert rep["total_tasks"] == 205
    assert rep["success_rate"] == 1.0
    assert rep["total_failed"] == 0

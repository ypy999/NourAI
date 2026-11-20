"""Run real NourAI v3.0 neural, tokenizer, hardware, and autonomous agent benchmarks on local hardware."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.transformer import GPT, GPTConfig
from nourai.agent.orchestrator import AutonomousAgentOrchestrator
from nourai.agent.task_manager import TaskManager
from nourai.core.versioning import CURRENT_VERSION
from nourai.evaluation.v3_benchmarks import FIXED_MULTILINGUAL_SAMPLES, V3EvaluationSuite
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.monitoring.hardware import HardwareMonitor
from nourai.neural.audit_and_optimize import audit_transformer_model, benchmark_kv_cache_and_quantization
from nourai.tokenizer.byte_bpe import ByteLevelBPETokenizer, benchmark_tokenizers
from nourai.tools.builtin_tools import register_default_tools
from nourai.tools.tool_engine import PermissionPolicy, ToolRegistry


def main() -> None:
    ckpt = REPO_ROOT / "releases" / "v1.2.0" / "nourai_v1_2_model.pt"
    if ckpt.is_file():
        model = GPT.load_checkpoint(ckpt, map_location="cpu")
        model_source = str(ckpt.relative_to(REPO_ROOT))
    else:
        model = GPT(GPTConfig.from_preset("nano"))
        model_source = "fresh_nano_init"

    # 1. Neural Audit & CPU KV-Cache / INT8 Benchmark
    neural_audit = audit_transformer_model(model)
    cpu_bench = benchmark_kv_cache_and_quantization(model, prompt_len=16, gen_steps=24)

    # 2. Byte-Level BPE (8K & 16K target) vs ByteTokenizer on multilingual samples
    train_texts = (
        FIXED_MULTILINGUAL_SAMPLES["arabic"]
        + FIXED_MULTILINGUAL_SAMPLES["english"]
        + FIXED_MULTILINGUAL_SAMPLES["french"]
    ) * 6
    bpe_8k = ByteLevelBPETokenizer.train(train_texts, vocab_size=8192, min_pair_frequency=2)
    bpe_16k = ByteLevelBPETokenizer.train(train_texts, vocab_size=16384, min_pair_frequency=2)
    tok_bench = benchmark_tokenizers(
        FIXED_MULTILINGUAL_SAMPLES,
        bpe_tokenizers={"bpe_8k_target": bpe_8k, "bpe_16k_target": bpe_16k},
    )

    # 3. Full Autonomous Agent + Memory + Recovery + Multilingual Neural Evaluation
    with tempfile.TemporaryDirectory(prefix="nourai_v3_eval_") as tmpdir:
        tmp = Path(tmpdir)
        db = tmp / "v3_eval.sqlite3"
        mm = AdvancedMemoryManager(db)
        kg = KnowledgeGraph(db)
        policy = PermissionPolicy()
        tools = ToolRegistry(db_path=db, policy=policy)
        register_default_tools(tools, workspace_root=tmp, db_path=db, memory_manager=mm, knowledge_graph=kg)
        tm = TaskManager(db)
        orch = AutonomousAgentOrchestrator(
            memory_manager=mm,
            knowledge_graph=kg,
            tool_registry=tools,
            task_manager=tm,
        )
        suite = V3EvaluationSuite(db_path=db)
        eval_report = suite.run_full_evaluation(orch, model=model, version=CURRENT_VERSION)

    output = {
        "version": CURRENT_VERSION,
        "model_source": model_source,
        "hardware_snapshot": HardwareMonitor().snapshot().to_dict(),
        "neural_audit": {
            "audit_passed": neural_audit["audit_passed"],
            "weight_tied": neural_audit["weight_tied"],
            "has_pre_ln": neural_audit["has_pre_ln"],
            "kv_cache_equivalent": neural_audit["kv_cache_equivalent"],
            "kv_cache_max_abs_diff": neural_audit["kv_cache_max_abs_diff"],
            "total_params": neural_audit["architecture_summary"]["parameters"]["total_unique_parameters"],
        },
        "cpu_inference_benchmark": cpu_bench,
        "tokenizer_benchmark": tok_bench,
        "agent_evaluation": eval_report,
    }
    out_path = REPO_ROOT / "reports" / "v3_evaluation_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

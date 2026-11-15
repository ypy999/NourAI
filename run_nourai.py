"""Quick interactive launcher for NourAI v1.2.0 (Web UI + CLI)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import webbrowser

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.backend import NourAIBackend
from deployment.production import ProductionDatabase, SystemMonitor, start_live_http_server
from inference.engine import GenerationConfig, InferenceEngine
from nourai.capabilities import NourAICapabilityEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="NourAI v1.2.0 Launcher")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal chat instead of Web UI")
    parser.add_argument(
        "--fallback-rules",
        action="store_true",
        help="Explicitly use the rule-based fallback assistant instead of direct neural generation",
    )
    parser.add_argument("--https", action="store_true", help="Enable TLS 1.3 HTTPS server")
    args = parser.parse_args()

    if args.cli:
        if args.fallback_rules:
            eng = NourAICapabilityEngine()
            history: list[dict[str, str]] = []
            print("=== NourAI v1.2.0 Rule-Based Fallback Assistant CLI (type 'exit' or 'خروج' to quit) ===")
            while True:
                try:
                    q = input("\nأنت > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "خروج"):
                    break
                res = eng.respond(q, history=history)
                history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": res["reply"]}])
                print(f"NourAI [rule_fallback | {res['language']}] > {res['reply']}")
            return

        engine = InferenceEngine.from_checkpoint("checkpoints/v1_2/nourai_v1_2_model.pt")
        cfg = GenerationConfig(
            max_new_tokens=48,
            strategy="greedy",
            temperature=0.7,
            stop_sequences=("<|user|>", "<|endoftext|>"),
        )
        print("=== NourAI v1.2.0 Direct Neural Model CLI (type 'exit' or 'خروج' to quit) ===")
        while True:
            try:
                q = input("\nأنت > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "خروج"):
                break
            formatted_prompt = f"<|user|>\n{q}\n<|assistant|>\n"
            out = engine.generate(formatted_prompt, cfg)
            print(
                f"NourAI [neural_model | {out['generated_token_count']} tokens | {out['tokens_per_second']:.1f} tok/s] > "
                f"{out['generated_text'].strip()}"
            )
        return

    db = ProductionDatabase()
    backend = NourAIBackend(db_path=db.db_path, use_capabilities=args.fallback_rules)
    monitor = SystemMonitor(db)
    server, thread, url = start_live_http_server(backend, monitor, enable_tls=args.https)
    print("============================================================")
    print(f"NourAI v1.2.0 Web UI & API Server is running at:\n  {url}")
    print(f"Default Mode: {'rule_based_fallback_assistant' if args.fallback_rules else 'neural_model'}")
    print("Press Ctrl+C to stop the server.")
    print("============================================================")
    try:
        webbrowser.open(url)
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down NourAI server...")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

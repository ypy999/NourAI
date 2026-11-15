"""Quick interactive launcher for NourAI v1.2.0 (Web UI + Live Web Search + CLI)."""
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
from nourai.web_search import NourAIWebAndSmartEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="NourAI v1.2.0 Launcher")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal chat instead of Web UI")
    parser.add_argument(
        "--web-search",
        action="store_true",
        default=True,
        help="Enable live Google News + Wikipedia + Web Search in CLI mode (default: True)",
    )
    parser.add_argument(
        "--fallback-rules",
        action="store_true",
        help="Explicitly use the offline rule-based fallback assistant in CLI mode",
    )
    parser.add_argument(
        "--raw-neural",
        action="store_true",
        help="Use raw 843K neural generation in CLI mode without web search",
    )
    parser.add_argument("--https", action="store_true", help="Enable local TLS 1.3 HTTPS server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (use 0.0.0.0 for cloud/LAN)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "0")),
        help="Port to bind (0 selects an available local port automatically)",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser window")
    args = parser.parse_args()

    if args.cli:
        if args.raw_neural:
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

        if args.fallback_rules:
            eng = NourAICapabilityEngine()
            history: list[dict[str, str]] = []
            print("=== NourAI v1.2.0 Offline Rule-Based Assistant CLI (type 'exit' or 'خروج' to quit) ===")
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

        web_eng = NourAIWebAndSmartEngine()
        history = []
        print("=== NourAI v1.2.0 Live Web Search & Universal AI CLI (type 'exit' or 'خروج' to quit) ===")
        while True:
            try:
                q = input("\nأنت > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "خروج"):
                break
            res = web_eng.answer_with_web(q, history=history, enable_web_search=True)
            history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": res["reply"]}])
            print(f"\nNourAI [{res['response_source']} | {res['language']}] >\n{res['reply']}")
        return

    db = ProductionDatabase()
    secret_file = db.db_path.parent / ".jwt_secret"
    if not secret_file.is_file():
        import secrets as _sec
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        secret_file.write_text(_sec.token_hex(32), encoding="utf-8")
    persisted_secret = secret_file.read_text(encoding="utf-8").strip()
    backend = NourAIBackend(db_path=db.db_path, secret_key=persisted_secret, use_capabilities=args.fallback_rules)
    monitor = SystemMonitor(db)
    server, thread, url = start_live_http_server(
        backend,
        monitor,
        enable_tls=args.https,
        host=args.host,
        port=args.port,
    )
    print("============================================================")
    print(f"NourAI v1.2.0 Studio (ChatGPT & Gemini Style + Live Web Search) is running at:\n  {url}")
    print("Press Ctrl+C to stop the server.")
    print("============================================================")
    try:
        if not args.no_browser and args.host == "127.0.0.1":
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

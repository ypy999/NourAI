"""Jebrex AI — Windows Native Desktop Application (Tkinter + Local Jebrex AI REST API).

Connects exclusively to the local Jebrex AI core backend (`NourAIBackend`),
displaying explicit `response_source` telemetry (`neural_model` by default)
without delegating to any external cloud LLM provider.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class JebrexDesktopApiClient:
    """Zero-dependency HTTP client talking strictly to a local Jebrex AI REST backend."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.token: str | None = None
        self.session_id: str | None = None

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                err_obj = json.loads(raw)
                msg = err_obj.get("error", raw)
            except Exception:
                msg = raw
            raise RuntimeError(f"HTTP {exc.code}: {msg}") from exc
        except URLError as exc:
            raise RuntimeError(f"Connection failed: {exc.reason}") from exc

    def health(self) -> dict[str, Any]:
        return self._call("GET", "/api/v1/health")

    def authenticate_guest(self) -> str:
        res = self._call("POST", "/api/v1/auth/guest", {})
        self.token = str(res["access_token"])
        return self.token

    def create_session(self, language: str = "arabic", title: str = "Jebrex Desktop Session") -> dict[str, Any]:
        if not self.token:
            self.authenticate_guest()
        res = self._call("POST", "/api/v1/sessions", {"language": language, "title": title})
        self.session_id = str(res["session"]["session_id"])
        return res["session"]

    def list_sessions(self) -> list[dict[str, Any]]:
        if not self.token:
            self.authenticate_guest()
        return list(self._call("GET", "/api/v1/sessions").get("sessions", []))

    def send_chat(
        self,
        message: str,
        *,
        language: str = "arabic",
        strategy: str = "top_k_top_p",
        temperature: float = 0.7,
        top_k: int = 16,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        max_new_tokens: int = 96,
    ) -> dict[str, Any]:
        if not self.token:
            self.authenticate_guest()
        if not self.session_id:
            self.create_session(language=language)
        return self._call(
            "POST",
            "/api/v1/chat",
            {
                "session_id": self.session_id,
                "message": message,
                "language": language,
                "strategy": strategy,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": max_new_tokens,
                "use_fallback_assistant": False,
                "enable_web_search": False,
            },
        )


class JebrexDesktopWindow:
    """Tkinter desktop window for Jebrex AI on Windows."""

    def __init__(self, root: tk.Tk, client: JebrexDesktopApiClient):
        self.root = root
        self.client = client
        self.root.title("Jebrex AI — Windows Desktop (Independent Local Core)")
        self.root.geometry("920x640")
        self.root.configure(bg="#111827")

        self._build_ui()
        self._connect_initial()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#1f2937", padx=10, pady=8)
        top.pack(fill=tk.X)

        tk.Label(top, text="Jebrex AI Desktop", fg="#f9fafb", bg="#1f2937", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Connecting to local Jebrex AI core...")
        tk.Label(top, textvariable=self.status_var, fg="#93c5fd", bg="#1f2937", font=("Segoe UI", 9)).pack(side=tk.RIGHT)

        controls = tk.Frame(self.root, bg="#111827", padx=10, pady=6)
        controls.pack(fill=tk.X)

        tk.Label(controls, text="Language:", fg="#d1d5db", bg="#111827").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="arabic")
        ttk.Combobox(controls, textvariable=self.lang_var, values=("arabic", "english", "french"), width=10, state="readonly").pack(side=tk.LEFT, padx=6)

        tk.Label(controls, text="Strategy:", fg="#d1d5db", bg="#111827").pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar(value="greedy")
        ttk.Combobox(controls, textvariable=self.strategy_var, values=("greedy", "top_k_top_p", "top_p", "top_k", "temperature"), width=12, state="readonly").pack(side=tk.LEFT, padx=6)

        tk.Label(controls, text="Temp:", fg="#d1d5db", bg="#111827").pack(side=tk.LEFT)
        self.temp_var = tk.DoubleVar(value=0.7)
        tk.Spinbox(controls, from_=0.1, to=2.0, increment=0.1, textvariable=self.temp_var, width=5).pack(side=tk.LEFT, padx=4)

        tk.Label(controls, text="Max Tokens:", fg="#d1d5db", bg="#111827").pack(side=tk.LEFT)
        self.max_tok_var = tk.IntVar(value=96)
        tk.Spinbox(controls, from_=16, to=256, increment=16, textvariable=self.max_tok_var, width=5).pack(side=tk.LEFT, padx=4)

        tk.Button(controls, text="New Chat", command=self._new_session, bg="#374151", fg="#f9fafb", relief=tk.FLAT, padx=10).pack(side=tk.RIGHT)

        self.transcript = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#f8fafc",
            font=("Segoe UI", 10),
            state=tk.DISABLED,
        )
        self.transcript.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        bottom = tk.Frame(self.root, bg="#111827", padx=10, pady=8)
        bottom.pack(fill=tk.X)

        self.input_entry = tk.Entry(bottom, bg="#1f2937", fg="#f9fafb", insertbackground="#f9fafb", font=("Segoe UI", 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self.input_entry.bind("<Return>", lambda _e: self._send_clicked())

        self.send_btn = tk.Button(bottom, text="Send", command=self._send_clicked, bg="#2563eb", fg="#ffffff", relief=tk.FLAT, padx=16, pady=6)
        self.send_btn.pack(side=tk.RIGHT, padx=(8, 0))

    def _append_line(self, speaker: str, text: str) -> None:
        self.transcript.configure(state=tk.NORMAL)
        self.transcript.insert(tk.END, f"[{speaker}]\n{text}\n\n")
        self.transcript.see(tk.END)
        self.transcript.configure(state=tk.DISABLED)

    def _connect_initial(self) -> None:
        try:
            h = self.client.health()
            self.client.authenticate_guest()
            self.client.create_session(language=self.lang_var.get())
            params = h.get("memory", {}).get("parameter_count", "?")
            self.status_var.set(f"Connected | Local Core ({params:,} params)" if isinstance(params, int) else "Connected")
            self._append_line("SYSTEM", "Connected to local Jebrex AI neural model (independent mode — no external LLM APIs).")
        except Exception as exc:
            self.status_var.set("Backend unreachable")
            self._append_line("ERROR", str(exc))

    def _new_session(self) -> None:
        try:
            self.client.create_session(language=self.lang_var.get())
            self.transcript.configure(state=tk.NORMAL)
            self.transcript.delete("1.0", tk.END)
            self.transcript.configure(state=tk.DISABLED)
            self._append_line("SYSTEM", f"New session created ({self.client.session_id}).")
        except Exception as exc:
            messagebox.showerror("Session Error", str(exc))

    def _send_clicked(self) -> None:
        msg = self.input_entry.get().strip()
        if not msg:
            return
        self.input_entry.delete(0, tk.END)
        self._append_line("YOU", msg)
        self.send_btn.configure(state=tk.DISABLED)

        def _worker() -> None:
            try:
                res = self.client.send_chat(
                    msg,
                    language=self.lang_var.get(),
                    strategy=self.strategy_var.get(),
                    temperature=float(self.temp_var.get()),
                    max_new_tokens=int(self.max_tok_var.get()),
                )
                src = res.get("response_source", "unknown")
                m = res.get("metrics", {})
                header = f"JEBREX AI ({src} | {m.get('generated_tokens', 0)} tok | {m.get('tokens_per_second', 0)} tok/s)"
                self.root.after(0, lambda: self._append_line(header, res.get("reply", "")))
            except Exception as exc:
                self.root.after(0, lambda: self._append_line("ERROR", str(exc)))
            finally:
                self.root.after(0, lambda: self.send_btn.configure(state=tk.NORMAL))

        threading.Thread(target=_worker, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Jebrex AI Windows Desktop Client")
    parser.add_argument("--api-url", default="", help="Existing local Jebrex AI API URL (starts embedded server if omitted)")
    args = parser.parse_args()

    embedded_server = None
    base_url = args.api_url.strip()
    if not base_url:
        from api.backend import NourAIBackend
        from deployment.production import ProductionDatabase, SystemMonitor, start_live_http_server

        db = ProductionDatabase()
        backend = NourAIBackend(db_path=db.db_path, use_capabilities=False)
        monitor = SystemMonitor(db)
        embedded_server, _thread, base_url = start_live_http_server(backend, monitor, host="127.0.0.1", port=0)

    root = tk.Tk()
    client = JebrexDesktopApiClient(base_url)
    JebrexDesktopWindow(root, client)
    try:
        root.mainloop()
    finally:
        if embedded_server is not None:
            embedded_server.shutdown()
            embedded_server.server_close()


if __name__ == "__main__":
    main()

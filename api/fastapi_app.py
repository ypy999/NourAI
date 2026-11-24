"""Optional FastAPI ASGI adapter wrapping Jebrex AI's core NourAIBackend.

The local environment runs on zero-dependency Python stdlib ThreadingHTTPServer
(`deployment/production.py`) to keep disk and memory overhead minimal. When
`fastapi` and `uvicorn` are installed in a production/cloud environment, this
module exposes a drop-in FastAPI ASGI application backed by the exact same
hardened `NourAIBackend` dispatcher and local Jebrex AI neural model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from api.backend import NourAIBackend
from ui.chat_app import load_ui_html


def create_fastapi_app(
    backend: NourAIBackend | None = None,
    *,
    db_path: str | Path | None = None,
    secret_key: str | None = None,
) -> Any:
    """Create a FastAPI ASGI app delegating every route to NourAIBackend."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is not installed in the current Python environment. "
            "Use deployment.production.start_live_http_server() for zero-dependency local serving, "
            "or install fastapi and uvicorn when deploying behind an ASGI server."
        ) from exc

    core = backend or NourAIBackend(db_path=db_path, secret_key=secret_key, use_capabilities=False)
    ui_html = load_ui_html()

    app = FastAPI(
        title="Jebrex AI Multilingual Language Model API",
        version="1.3.0",
        description="Local-first independent Arabic/English/French LLM API (FastAPI adapter).",
    )

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"])
    async def dispatch_all(full_path: str, request: Request):
        route = "/" + full_path.lstrip("/")
        raw_bytes = await request.body()
        if len(raw_bytes) > 1_048_576:
            return JSONResponse({"error": "request payload exceeds 1 MiB limit"}, status_code=413)
        body: dict[str, Any] = {}
        if raw_bytes.strip():
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid JSON request body"}, status_code=400)
        client_ip = request.client.host if request.client else "127.0.0.1"
        status, headers, payload = core.handle_request(
            request.method,
            route,
            headers=dict(request.headers),
            body=body if isinstance(body, dict) else {},
            client_ip=client_ip,
            ui_html_provider=lambda: ui_html,
        )
        if isinstance(payload, str):
            return HTMLResponse(content=payload, status_code=status, headers=headers)
        return JSONResponse(content=payload, status_code=status, headers=headers)

    return app

# NourAI — REST API Reference (`v1.0.0`)

## Endpoints
- `GET /api/v1/health` — System health, supported languages, and memory status.
- `GET /api/v1/openapi.json` — OpenAPI 3.1.0 specification.
- `POST /api/v1/users/register` — Register a user account (PBKDF2-HMAC-SHA256).
- `POST /api/v1/auth/token` — Issue an HMAC-SHA256 Bearer token.
- `GET /api/v1/sessions` / `POST /api/v1/sessions` — List or create multi-turn chat sessions.
- `POST /api/v1/chat` — Synchronous multilingual chat completion.
- `POST /api/v1/chat/stream` — Progressive streaming token generation.
- `GET /api/v1/admin/users` — Admin RBAC user directory (`role=admin`).

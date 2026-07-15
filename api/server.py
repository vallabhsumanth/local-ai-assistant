"""JARVIS web layer — FastAPI.

Same core `Assistant`, exposed over HTTP so it can run both locally and on
Railway (which needs a process listening on $PORT).

Endpoints:
  GET  /            -> serves the chat UI (frontend/index.html)
  GET  /health      -> {"status": "ok"}  (Railway healthcheck)
  POST /chat        -> streamed plain-text reply (real live typing + a
                        working Stop button, via an aborted fetch)
                        body: {"message": str, "session": str,
                               "deep_think": bool = False}

Run modes:
  - Local dev:  uvicorn api.server:app --reload   (or ./serve.sh)
  - Railway:    the Dockerfile launches uvicorn on $PORT

Safety: in web/cloud mode there's no interactive terminal, so destructive file
operations (delete/move) are auto-DENIED. They only work in the local REPL,
where the user can confirm at the prompt.
"""

from __future__ import annotations

import os
from pathlib import Path

from core.assistant import Assistant
from core.deps import ensure_package
from tools import files
from utils.logger import get_logger

log = get_logger("jarvis.api")

fastapi = ensure_package("fastapi")
if fastapi is None:  # pragma: no cover
    raise SystemExit("fastapi not installed. Run: pip install -r requirements.txt")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

# In cloud/web mode we cannot prompt interactively, so deny destructive actions.
files.set_confirm_handler(lambda msg: (log.warning("Auto-denied (web): %s", msg), False)[1])

app = FastAPI(title="JARVIS", version="1.0")

# One assistant instance; conversations are separated by `session` in memory.
_assistant = Assistant()


class ChatIn(BaseModel):
    message: str
    session: str = "web"
    deep_think: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": _assistant.provider.name}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if FRONTEND.exists():
        return FRONTEND.read_text(encoding="utf-8")
    return "<h1>JARVIS</h1><p>POST /chat with {\"message\": \"...\"}</p>"


@app.post("/chat")
def chat(body: ChatIn) -> StreamingResponse:
    _assistant.session = body.session
    _assistant.deep_think = body.deep_think
    # A plain streamed text body: the browser reads it incrementally for real
    # live typing, and aborting the fetch (Stop button) ends the generator —
    # closing the underlying Ollama connection so it stops generating too.
    return StreamingResponse(
        _assistant.handle_stream(body.message), media_type="text/plain; charset=utf-8"
    )


def main() -> None:
    """Entry point used by `python -m api.server` and the Dockerfile."""
    uvicorn = ensure_package("uvicorn")
    if uvicorn is None:  # pragma: no cover
        raise SystemExit("uvicorn not installed.")
    port = int(os.environ.get("PORT", "8000"))
    log.info("Starting JARVIS web on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

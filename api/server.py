"""JARVIS web layer — FastAPI.

Same core `Assistant`, exposed over HTTP so it can run both locally and on
Railway (which needs a process listening on $PORT).

Endpoints:
  GET  /            -> serves the chat UI (frontend/index.html)
  GET  /health      -> {"status": "ok"}  (Railway healthcheck)
  GET  /chats       -> [{"session","title","created_at","last_active"}, ...]
                        the ChatGPT-style chat list, most recent first
  GET  /chats/{session}/history -> [{"role","content"}, ...] full history
                        for that chat, used when switching back into it
  DELETE /chats/{session} -> removes the chat AND its messages from Supabase
                        (the sidebar's red delete button; permanent, not a
                        hide-from-list — matches deleting the chat for real)
  POST /chat        -> streamed plain-text reply (real live typing + a
                        working Stop button, via an aborted fetch)
                        body: {"message": str, "session": str,
                               "deep_think": bool = False}
  POST /upload      -> the attach-file button. Accepts a .xlsx/.xls/.csv/.txt
                        file (multipart/form-data), analyzes it, saves the
                        analysis to the permanent knowledge base, and returns
                        {"filename": str, "analysis": str}.

On startup, any chat inactive for more than 20 days is deleted (registry
entry + its messages) — see Assistant._touch_chat / memory cleanup_expired_chats.

Run modes:
  - Local dev:  uvicorn api.server:app --reload   (or ./serve.sh)
  - Railway:    the Dockerfile launches uvicorn on $PORT

Safety: in web/cloud mode there's no interactive terminal, so destructive file
operations (delete/move) are auto-DENIED. They only work in the local REPL,
where the user can confirm at the prompt.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from core.assistant import Assistant
from core.deps import ensure_package
from tools import files, sheets
from utils.logger import get_logger

log = get_logger("jarvis.api")

fastapi = ensure_package("fastapi")
if fastapi is None:  # pragma: no cover
    raise SystemExit("fastapi not installed. Run: pip install -r requirements.txt")
ensure_package("multipart")  # python-multipart — needed for file uploads

from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB
UPLOAD_EXTENSIONS = {".xlsx", ".xls", ".csv", ".txt"}

# In cloud/web mode we cannot prompt interactively, so deny destructive actions.
files.set_confirm_handler(lambda msg: (log.warning("Auto-denied (web): %s", msg), False)[1])

app = FastAPI(title="Nap Bot", version="1.0")

# One assistant instance; conversations are separated by `session` in memory.
_assistant = Assistant()

# Sweep out chats nobody has touched in 20+ days, once per server start.
try:
    _removed = _assistant.memory.cleanup_expired_chats(20)
    if _removed:
        log.info("Cleaned up %d expired chat(s)", _removed)
except Exception as exc:  # noqa: BLE001 - never block startup over cleanup
    log.warning("Chat cleanup failed: %s", exc)


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
    return "<h1>Nap Bot</h1><p>POST /chat with {\"message\": \"...\"}</p>"


@app.get("/chats")
def chats() -> list[dict]:
    return _assistant.memory.list_chats()


@app.get("/chats/{session}/history")
def chat_history(session: str) -> list[dict]:
    return _assistant.memory.recent_turns(session, limit=500)


@app.delete("/chats/{session}")
def remove_chat(session: str) -> dict:
    _assistant.memory.delete_chat(session)
    return {"deleted": session}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session: str = Form("web")) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in UPLOAD_EXTENSIONS:
        return {"error": f"Unsupported file type {ext!r}. Supported: "
                          f"{', '.join(sorted(UPLOAD_EXTENSIONS))}"}

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        return {"error": f"File too large ({len(contents) / 1e6:.1f}MB) — "
                          f"max {MAX_UPLOAD_BYTES / 1e6:.0f}MB."}

    # Strip any path components from the name (safety) and make it unique on
    # disk so a same-named re-upload never silently overwrites an old one.
    safe_name = Path(file.filename or "upload").name
    stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    dest.write_bytes(contents)

    if ext in (".xlsx", ".xls", ".csv"):
        analysis = sheets.read_spreadsheet(str(dest))
    else:  # .txt
        analysis = contents.decode("utf-8", errors="replace")[:8000]

    # Persisted in two places: the knowledge base (so it's searchable and
    # informs later conversations, even new ones) AND this chat's own history
    # (so reopening this specific chat still shows the upload + analysis).
    # The analysis itself already succeeded at this point — a knowledge-base
    # hiccup (e.g. the napbot_knowledge table not created yet) shouldn't throw
    # away a perfectly good result, so it's isolated in its own try/except.
    knowledge_saved = True
    try:
        _assistant.memory.save_knowledge(
            topic=f"Uploaded file: {safe_name}", content=analysis, source=f"upload:{safe_name}",
        )
    except Exception as exc:  # noqa: BLE001 - degrade, don't fail the whole upload
        knowledge_saved = False
        log.warning("save_knowledge failed for %s (%s) — returning the analysis anyway. "
                    "Have you run the napbot_knowledge SQL in memory/schema.sql yet?",
                    safe_name, exc)

    note = f"[Uploaded file: {safe_name}]"
    _assistant.memory.add_turn(session, "user", note)
    _assistant.memory.add_turn(session, "assistant", analysis)
    _assistant.memory.touch_chat(session, title=note)
    log.info("Uploaded %s (%d bytes) -> chat %s (knowledge base saved: %s)",
             safe_name, len(contents), session, knowledge_saved)
    return {"filename": safe_name, "analysis": analysis, "knowledge_saved": knowledge_saved}


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
    log.info("Starting Nap Bot web on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

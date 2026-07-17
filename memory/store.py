"""JARVIS memory — storage abstraction.

All user data (conversation history + durable facts) is stored in the cloud
(Supabase), never on local disk. This module defines:

- `MemoryBackend`  : the interface every backend implements.
- `InMemoryMemory` : an ephemeral RAM fallback used only when Supabase isn't
                     configured yet. It is NOT persisted to disk — restarting
                     JARVIS clears it — so we never silently write user data
                     locally, which is exactly what you asked for.
- `get_memory()`   : factory that returns the Supabase backend when configured,
                     otherwise the ephemeral one (with a loud warning).

The Supabase backend lives in `memory/supabase_store.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from config.settings import settings
from utils.logger import get_logger

log = get_logger(__name__)

Turn = dict[str, str]
ChatInfo = dict[str, str]  # {session, title, created_at, last_active}
KnowledgeItem = dict[str, str]  # {topic, content, source, created_at}


class MemoryBackend(ABC):
    name: str = "base"

    @abstractmethod
    def add_turn(self, session: str, role: str, content: str) -> None: ...

    @abstractmethod
    def recent_turns(self, session: str, limit: int = 20) -> list[Turn]: ...

    @abstractmethod
    def remember(self, key: str, value: str) -> None: ...

    @abstractmethod
    def recall(self, key: str) -> str | None: ...

    @abstractmethod
    def all_facts(self) -> dict[str, str]: ...

    # --- chats (ChatGPT-style chat list + 20-day auto-cleanup) ---
    @abstractmethod
    def touch_chat(self, session: str, title: str | None = None) -> None:
        """Register a chat the first time it's used (with `title`), or just
        bump its last-active time on every later use. Never overwrites an
        already-set title, so it's safe to call on every message."""

    @abstractmethod
    def list_chats(self) -> list[ChatInfo]:
        """All chats, most recently active first."""

    @abstractmethod
    def delete_chat(self, session: str) -> None:
        """Remove a chat's registry entry and all its stored messages."""

    @abstractmethod
    def cleanup_expired_chats(self, days: int = 20) -> int:
        """Delete chats inactive for more than `days`. Returns how many."""

    # --- persistent knowledge base (research that compounds over time) ---
    @abstractmethod
    def save_knowledge(self, topic: str, content: str, source: str | None = None) -> None:
        """Persist a research finding so future conversations can build on it
        instead of re-researching the same thing from scratch."""

    @abstractmethod
    def search_knowledge(self, query: str, limit: int = 5) -> list[KnowledgeItem]:
        """Search previously saved knowledge. Most relevant first."""


class InMemoryMemory(MemoryBackend):
    """Ephemeral, process-local store. Lost on exit. No disk writes."""

    name = "in-memory (ephemeral)"

    def __init__(self) -> None:
        self._turns: list[tuple[str, str, str]] = []  # (session, role, content)
        self._facts: dict[str, str] = {}
        self._chats: dict[str, dict] = {}  # session -> {title, created_at, last_active}
        self._knowledge: list[dict] = []  # [{topic, content, source, created_at}]

    def add_turn(self, session: str, role: str, content: str) -> None:
        self._turns.append((session, role, content))

    def recent_turns(self, session: str, limit: int = 20) -> list[Turn]:
        rows = [t for t in self._turns if t[0] == session][-limit:]
        return [{"role": r, "content": c} for _, r, c in rows]

    def remember(self, key: str, value: str) -> None:
        self._facts[key] = value

    def recall(self, key: str) -> str | None:
        return self._facts.get(key)

    def all_facts(self) -> dict[str, str]:
        return dict(self._facts)

    def touch_chat(self, session: str, title: str | None = None) -> None:
        now = datetime.now(timezone.utc)
        chat = self._chats.get(session)
        if chat is None:
            self._chats[session] = {
                "session": session,
                "title": title or "New chat",
                "created_at": now,
                "last_active": now,
            }
        else:
            chat["last_active"] = now

    def list_chats(self) -> list[ChatInfo]:
        chats = sorted(self._chats.values(), key=lambda c: c["last_active"], reverse=True)
        return [
            {**c, "created_at": c["created_at"].isoformat(),
             "last_active": c["last_active"].isoformat()}
            for c in chats
        ]

    def delete_chat(self, session: str) -> None:
        self._chats.pop(session, None)
        self._turns = [t for t in self._turns if t[0] != session]

    def cleanup_expired_chats(self, days: int = 20) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        expired = [s for s, c in self._chats.items() if c["last_active"] < cutoff]
        for s in expired:
            self.delete_chat(s)
        return len(expired)

    def save_knowledge(self, topic: str, content: str, source: str | None = None) -> None:
        self._knowledge.append({
            "topic": topic, "content": content, "source": source or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    def search_knowledge(self, query: str, limit: int = 5) -> list[KnowledgeItem]:
        words = [w for w in query.lower().split() if w]
        scored = []
        for item in self._knowledge:
            hay = (item["topic"] + " " + item["content"]).lower()
            # Count occurrences, not just presence — a document mentioning a
            # term repeatedly is a stronger match than one mentioning it once.
            score = sum(hay.count(w) for w in words)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]


def get_memory() -> MemoryBackend:
    """Return the configured storage backend."""
    if settings.supabase_configured:
        try:
            from memory.supabase_store import SupabaseMemory

            backend = SupabaseMemory()
            log.info("Memory backend: Supabase (cloud)")
            return backend
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash
            log.warning("Supabase init failed (%s); using ephemeral memory.", exc)
    else:
        log.warning(
            "SUPABASE_URL / SUPABASE_KEY not set — using EPHEMERAL memory "
            "(data is lost on exit). Add credentials to .env for cloud storage."
        )
    return InMemoryMemory()

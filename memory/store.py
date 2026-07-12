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

from config.settings import settings
from utils.logger import get_logger

log = get_logger(__name__)

Turn = dict[str, str]


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


class InMemoryMemory(MemoryBackend):
    """Ephemeral, process-local store. Lost on exit. No disk writes."""

    name = "in-memory (ephemeral)"

    def __init__(self) -> None:
        self._turns: list[tuple[str, str, str]] = []  # (session, role, content)
        self._facts: dict[str, str] = {}

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

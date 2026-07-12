"""Supabase-backed memory for JARVIS.

Stores conversation history and durable facts in the cloud. The `supabase`
Python client is installed on demand by the dependency manager the first time
this backend is used.

Tables (see memory/schema.sql):
  - jarvis_conversations (id, session, role, content, ts)
  - jarvis_facts         (key, value, updated)
"""

from __future__ import annotations

from core.deps import ensure_package
from config.settings import settings
from memory.store import MemoryBackend, Turn
from utils.logger import get_logger

log = get_logger(__name__)

CONV_TABLE = "jarvis_conversations"
FACT_TABLE = "jarvis_facts"


class SupabaseMemory(MemoryBackend):
    name = "supabase"

    def __init__(self) -> None:
        supabase = ensure_package("supabase")
        if supabase is None:
            raise RuntimeError("supabase client unavailable")
        if not settings.supabase_configured:
            raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not set")
        self._client = supabase.create_client(
            settings.supabase_url, settings.supabase_key
        )

    def add_turn(self, session: str, role: str, content: str) -> None:
        self._client.table(CONV_TABLE).insert(
            {"session": session, "role": role, "content": content}
        ).execute()

    def recent_turns(self, session: str, limit: int = 20) -> list[Turn]:
        resp = (
            self._client.table(CONV_TABLE)
            .select("role, content")
            .eq("session", session)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        rows = list(reversed(resp.data or []))
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def remember(self, key: str, value: str) -> None:
        # upsert on primary key `key`
        self._client.table(FACT_TABLE).upsert(
            {"key": key, "value": value}, on_conflict="key"
        ).execute()

    def recall(self, key: str) -> str | None:
        resp = (
            self._client.table(FACT_TABLE)
            .select("value")
            .eq("key", key)
            .limit(1)
            .execute()
        )
        data = resp.data or []
        return data[0]["value"] if data else None

    def all_facts(self) -> dict[str, str]:
        resp = self._client.table(FACT_TABLE).select("key, value").execute()
        return {r["key"]: r["value"] for r in (resp.data or [])}

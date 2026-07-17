"""Supabase-backed memory for Nap Bot.

Stores conversation history and durable facts in the cloud. The `supabase`
Python client is installed on demand by the dependency manager the first time
this backend is used.

Tables (see memory/schema.sql):
  - napbot_conversations (id, session, role, content, ts)
  - napbot_facts         (key, value, updated)
  - napbot_chats         (session, title, created_at, last_active)
  - napbot_knowledge     (id, topic, content, source, created_at, search_vector)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.deps import ensure_package
from config.settings import settings
from memory.store import ChatInfo, KnowledgeItem, MemoryBackend, Turn
from utils.logger import get_logger

log = get_logger(__name__)

CONV_TABLE = "napbot_conversations"
FACT_TABLE = "napbot_facts"
CHATS_TABLE = "napbot_chats"
KNOWLEDGE_TABLE = "napbot_knowledge"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def touch_chat(self, session: str, title: str | None = None) -> None:
        # Try updating an existing chat's last-active time first; only
        # insert a new row (with the title) if none existed. This never
        # overwrites a title that's already set.
        resp = (
            self._client.table(CHATS_TABLE)
            .update({"last_active": _now_iso()})
            .eq("session", session)
            .execute()
        )
        if not resp.data:
            self._client.table(CHATS_TABLE).insert({
                "session": session,
                "title": title or "New chat",
            }).execute()

    def list_chats(self) -> list[ChatInfo]:
        resp = (
            self._client.table(CHATS_TABLE)
            .select("session, title, created_at, last_active")
            .order("last_active", desc=True)
            .execute()
        )
        return resp.data or []

    def delete_chat(self, session: str) -> None:
        self._client.table(CONV_TABLE).delete().eq("session", session).execute()
        self._client.table(CHATS_TABLE).delete().eq("session", session).execute()

    def cleanup_expired_chats(self, days: int = 20) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        resp = (
            self._client.table(CHATS_TABLE)
            .select("session")
            .lt("last_active", cutoff)
            .execute()
        )
        expired = [row["session"] for row in (resp.data or [])]
        for session in expired:
            self.delete_chat(session)
        return len(expired)

    def save_knowledge(self, topic: str, content: str, source: str | None = None) -> None:
        self._client.table(KNOWLEDGE_TABLE).insert({
            "topic": topic, "content": content, "source": source,
        }).execute()

    def search_knowledge(self, query: str, limit: int = 5) -> list[KnowledgeItem]:
        cols = "topic, content, source, created_at"
        try:
            resp = (
                self._client.table(KNOWLEDGE_TABLE)
                .select(cols)
                .text_search("search_vector", query, {"type": "web_search"})
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception as exc:  # noqa: BLE001 - fall back to plain substring match
            log.warning("Full-text search failed (%s); using substring fallback.", exc)
            resp = (
                self._client.table(KNOWLEDGE_TABLE)
                .select(cols)
                .ilike("content", f"%{query}%")
                .limit(limit)
                .execute()
            )
            return resp.data or []

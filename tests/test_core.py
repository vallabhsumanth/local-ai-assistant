"""Phase 1 smoke tests. Run: myenv/bin/python -m pytest  (or unittest below)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.assistant import Assistant  # noqa: E402
from core.llm import EchoProvider, get_provider  # noqa: E402
from memory.store import InMemoryMemory, get_memory  # noqa: E402
from tools import files  # noqa: E402


class TestMemory(unittest.TestCase):
    def test_facts_roundtrip(self):
        mem = InMemoryMemory()
        mem.remember("editor", "VS Code")
        self.assertEqual(mem.recall("editor"), "VS Code")
        self.assertIn("editor", mem.all_facts())

    def test_conversation_history(self):
        mem = InMemoryMemory()
        mem.add_turn("s", "user", "hi")
        mem.add_turn("s", "assistant", "hello")
        turns = mem.recent_turns("s")
        self.assertEqual([t["role"] for t in turns], ["user", "assistant"])

    def test_factory_returns_backend(self):
        # Type depends on ambient .env (Supabase if configured, else ephemeral).
        from memory.store import MemoryBackend
        self.assertIsInstance(get_memory(), MemoryBackend)

    def test_touch_chat_creates_then_keeps_title(self):
        mem = InMemoryMemory()
        mem.touch_chat("s1", title="First message here")
        mem.touch_chat("s1", title="Should be ignored")
        chats = mem.list_chats()
        self.assertEqual(len(chats), 1)
        self.assertEqual(chats[0]["title"], "First message here")

    def test_list_chats_orders_by_recency(self):
        mem = InMemoryMemory()
        mem.touch_chat("first", title="First")
        mem.touch_chat("second", title="Second")
        mem.touch_chat("first")  # re-touch -> should move back to the top
        chats = mem.list_chats()
        self.assertEqual(chats[0]["session"], "first")

    def test_delete_chat_removes_registry_and_messages(self):
        mem = InMemoryMemory()
        mem.touch_chat("s1", title="Chat 1")
        mem.add_turn("s1", "user", "hi")
        mem.delete_chat("s1")
        self.assertEqual(mem.recent_turns("s1"), [])
        self.assertEqual(mem.list_chats(), [])

    def test_cleanup_expired_chats(self):
        from datetime import datetime, timedelta, timezone
        mem = InMemoryMemory()
        mem.touch_chat("old", title="Old chat")
        mem.touch_chat("recent", title="Recent chat")
        mem._chats["old"]["last_active"] = datetime.now(timezone.utc) - timedelta(days=25)
        removed = mem.cleanup_expired_chats(days=20)
        self.assertEqual(removed, 1)
        sessions = [c["session"] for c in mem.list_chats()]
        self.assertNotIn("old", sessions)
        self.assertIn("recent", sessions)

    def test_save_and_search_knowledge(self):
        mem = InMemoryMemory()
        mem.save_knowledge("Nauti Nati pricing", "Uses discount-heavy pricing, avg ₹600.", "nautinati.com")
        mem.save_knowledge("Hopscotch marketing", "Focuses on influencer partnerships.", "hopscotch.in")
        results = mem.search_knowledge("pricing")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["topic"], "Nauti Nati pricing")

    def test_search_knowledge_ranks_by_relevance(self):
        mem = InMemoryMemory()
        mem.save_knowledge("A", "mentions kidswear once")
        mem.save_knowledge("B", "kidswear kidswear kidswear market trends")
        results = mem.search_knowledge("kidswear")
        self.assertEqual(results[0]["topic"], "B")  # more matching words -> ranked first

    def test_search_knowledge_no_match(self):
        mem = InMemoryMemory()
        mem.save_knowledge("A", "something unrelated")
        self.assertEqual(mem.search_knowledge("zzz_nomatch"), [])


class TestLLM(unittest.TestCase):
    def test_get_provider_returns_provider(self):
        # Type depends on ambient .env; just assert we get a usable provider.
        from core.llm import LLMProvider
        self.assertIsInstance(get_provider(), LLMProvider)

    def test_echo_reply(self):
        reply = EchoProvider().chat([{"role": "user", "content": "ping"}])
        self.assertIn("ping", reply)

    # --- message/tool-spec conversion logic for Anthropic/OpenAI tool-calling ---
    # These are pure data transforms (static methods, no API key/network needed)
    # so they're fully verifiable without a live Anthropic/OpenAI key.

    def test_anthropic_tool_spec_conversion(self):
        from core.llm import AnthropicProvider
        openai_shaped = [{"type": "function", "function": {
            "name": "get_weather", "description": "d",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
        }}]
        out = AnthropicProvider.to_anthropic_tools(openai_shaped)
        self.assertEqual(out, [{
            "name": "get_weather", "description": "d",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
        }])

    def test_anthropic_message_conversion_system_and_tool_result(self):
        from core.llm import AnthropicProvider
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_name": "get_weather", "tool_call_id": "abc123", "content": "sunny"},
        ]
        system, convo = AnthropicProvider.to_anthropic_messages(messages)
        self.assertEqual(system, "You are helpful.")
        self.assertEqual(convo[0], {"role": "user", "content": "hi"})
        self.assertEqual(convo[1]["role"], "user")  # tool results ride on a user message
        block = convo[1]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "abc123")
        self.assertEqual(block["content"], "sunny")

    def test_anthropic_message_conversion_passes_through_native_assistant(self):
        from core.llm import AnthropicProvider
        native = {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "f", "input": {}}]}
        _, convo = AnthropicProvider.to_anthropic_messages([native])
        self.assertEqual(convo, [native])

    def test_openai_message_conversion_tool_result(self):
        from core.llm import OpenAIProvider
        messages = [{"role": "tool", "tool_name": "get_weather", "tool_call_id": "call_1", "content": "sunny"}]
        out = OpenAIProvider.to_openai_messages(messages)
        self.assertEqual(out, [{"role": "tool", "tool_call_id": "call_1", "content": "sunny"}])

    def test_openai_message_conversion_passes_through_native_assistant(self):
        from core.llm import OpenAIProvider
        native = {"role": "assistant", "content": None,
                  "tool_calls": [{"id": "call_1", "type": "function",
                                   "function": {"name": "f", "arguments": "{}"}}]}
        out = OpenAIProvider.to_openai_messages([native])
        self.assertEqual(out, [native])

    def test_echo_supports_tools_fallback(self):
        # A non-tool provider still answers via chat_with_tools (no tool_calls).
        out = EchoProvider().chat_with_tools([{"role": "user", "content": "hi"}], [])
        self.assertEqual(out["tool_calls"], [])
        self.assertIn("hi", out["content"])


class TestFiles(unittest.TestCase):
    def test_write_read_search(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "note.txt"
            files.write_text(p, "hello world")
            self.assertEqual(files.read_text(p), "hello world")
            self.assertTrue(files.search(d, "*.txt"))


class TestSheets(unittest.TestCase):
    def test_read_csv_reports_shape_and_stats(self):
        from tools import sheets
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "orders.csv"
            p.write_text("sku,qty,price\nA,2,100\nB,1,50\nC,3,150\n")
            out = sheets.read_spreadsheet(str(p))
            self.assertIn("Rows: 3", out)
            self.assertIn("Columns: 3", out)
            self.assertIn("sku, qty, price", out)
            self.assertIn("100", out)  # a real value made it into the sample rows

    def test_read_missing_file(self):
        from tools import sheets
        out = sheets.read_spreadsheet("/no/such/file.xlsx")
        self.assertIn("not found", out.lower())

    def test_row_cap_keeps_full_stats(self):
        from tools import sheets
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "big.csv"
            lines = ["qty\n"] + [f"{i}\n" for i in range(1, 301)]  # 300 rows, 1..300
            p.write_text("".join(lines))
            out = sheets.read_spreadsheet(str(p))
            self.assertIn("Rows: 300", out)
            self.assertIn("more rows exist", out)
            self.assertIn("300.00", out)  # max=300 present in the full-data summary stats


class TestAssistant(unittest.TestCase):
    def test_help_and_provider_commands(self):
        a = Assistant(session="test")
        self.assertIn("JARVIS commands", a.handle("/help"))
        self.assertIn("provider", a.handle("/provider").lower())
        self.assertEqual(a.handle("/quit"), "__quit__")

    def test_touch_chat_registers_with_truncated_title(self):
        # Uses InMemoryMemory directly so this doesn't need a live LLM call.
        a = Assistant(session="chattest")
        a.memory = InMemoryMemory()
        a._touch_chat("Hello there, this is my first message")
        chats = a.memory.list_chats()
        self.assertEqual(len(chats), 1)
        self.assertEqual(chats[0]["session"], "chattest")
        self.assertIn("Hello there", chats[0]["title"])


class TestAgent(unittest.TestCase):
    def test_agent_invokes_tool_then_answers(self):
        from core.agent import Agent
        from core.llm import LLMProvider
        from memory.store import InMemoryMemory
        from tools.registry import build_registry

        # Fake provider: first call asks for a tool, second returns final text.
        class FakeProvider(LLMProvider):
            name = "fake"
            supports_tools = True
            def __init__(self): self.calls = 0
            def chat(self, messages): return "unused"
            def chat_with_tools(self, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    return {"content": "", "raw": {"role": "assistant", "content": ""},
                            "tool_calls": [{"name": "remember_fact",
                                            "args": {"key": "color", "value": "blue"}}]}
                return {"content": "Done — I stored that.", "tool_calls": [], "raw": None}

        mem = InMemoryMemory()
        agent = Agent(FakeProvider(), mem, build_registry(mem))
        reply = agent.run([{"role": "user", "content": "remember my color is blue"}])
        self.assertEqual(reply, "Done — I stored that.")
        self.assertEqual(mem.recall("color"), "blue")   # tool really ran

    def test_registry_specs_are_wellformed(self):
        from memory.store import InMemoryMemory
        from tools.registry import build_registry
        for tool in build_registry(InMemoryMemory()):
            spec = tool.spec()
            self.assertEqual(spec["type"], "function")
            self.assertIn("name", spec["function"])
            self.assertIn("parameters", spec["function"])


class TestCache(unittest.TestCase):
    def test_similar_question_hits(self):
        from core.cache import ResponseCache
        c = ResponseCache()
        c.put("What is the capital of France?", "Paris.")
        # exact
        self.assertEqual(c.get("What is the capital of France?"), "Paris.")
        # rephrased but similar
        self.assertEqual(c.get("what is the capital of france"), "Paris.")

    def test_unrelated_question_misses(self):
        from core.cache import ResponseCache
        c = ResponseCache()
        c.put("What is the capital of France?", "Paris.")
        self.assertIsNone(c.get("How do I bake sourdough bread at home?"))

    def test_volatile_not_cached(self):
        from core.cache import ResponseCache
        c = ResponseCache()
        c.put("What is the weather in Mumbai today?", "Sunny.")
        self.assertIsNone(c.get("What is the weather in Mumbai today?"))

    def test_personal_not_cached(self):
        from core.cache import ResponseCache
        c = ResponseCache()
        c.put("What is my name?", "Asim.")
        self.assertIsNone(c.get("What is my name?"))

    def test_brand_strategy_not_cached(self):
        from core.cache import ResponseCache
        c = ResponseCache()
        c.put("How can Nap Chief beat Nauti Nati?", "Insight A.")
        self.assertIsNone(c.get("How can Nap Chief beat Nauti Nati?"))
        c.put("What's our growth strategy against Hopscotch?", "Insight B.")
        self.assertIsNone(c.get("What's our growth strategy against Hopscotch?"))

    def test_expiry(self):
        from core.cache import ResponseCache
        c = ResponseCache(ttl=100)
        c.put("What is the capital of France?", "Paris.", now=0)
        self.assertEqual(c.get("What is the capital of France?", now=50), "Paris.")
        self.assertIsNone(c.get("What is the capital of France?", now=200))


class TestMailer(unittest.TestCase):
    def test_stage_requires_valid_email(self):
        from tools.mailer import Emailer
        m = Emailer()
        out = m.stage("not-an-email", "Hi", "Body")
        self.assertIn("valid email", out.lower())
        self.assertFalse(m.has_pending())

    def test_stage_then_cancel(self):
        from tools.mailer import Emailer
        m = Emailer()
        m.stage("a@b.com", "Subj", "Body")
        self.assertTrue(m.has_pending())
        self.assertIn("discarded", m.cancel().lower())
        self.assertFalse(m.has_pending())

    def test_confirm_sends_once(self):
        from tools.mailer import Emailer
        m = Emailer()
        sent = []
        m._send = lambda to, subj, body: sent.append((to, subj, body))  # mock SMTP
        # pretend configured
        import config.settings as cfg
        orig = cfg.settings.__class__.email_configured
        cfg.settings.__class__.email_configured = property(lambda self: True)
        try:
            m.stage("a@b.com", "Subj", "Body")
            out = m.confirm()
        finally:
            cfg.settings.__class__.email_configured = orig
        self.assertIn("sent", out.lower())
        self.assertEqual(sent, [("a@b.com", "Subj", "Body")])
        self.assertFalse(m.has_pending())  # cleared after send

    def test_confirm_without_pending(self):
        from tools.mailer import Emailer
        self.assertIn("no pending", Emailer().confirm().lower())


class TestDesktop(unittest.TestCase):
    def test_not_mac_message(self):
        from desktop import apps
        orig = apps.IS_MAC
        apps.IS_MAC = False
        try:
            self.assertIn("locally on your mac", apps.open_app("Spotify").lower())
            self.assertIn("locally on your mac", apps.control_music("play").lower())
        finally:
            apps.IS_MAC = orig

    def test_invalid_music_action(self):
        from desktop import apps
        if not apps.IS_MAC:
            self.skipTest("mac-only")
        self.assertIn("play, pause", apps.control_music("boogie").lower())

    def test_open_app_success_mocked(self):
        from desktop import apps
        import subprocess
        orig_mac, orig_run = apps.IS_MAC, apps._run
        apps.IS_MAC = True
        apps._run = lambda args, timeout=10: subprocess.CompletedProcess(args, 0, "", "")
        try:
            self.assertIn("opening spotify", apps.open_app("Spotify").lower())
        finally:
            apps.IS_MAC, apps._run = orig_mac, orig_run


class TestUpload(unittest.TestCase):
    """The /upload endpoint (attach-file button + drag-and-drop)."""

    def _client(self):
        from fastapi.testclient import TestClient
        import api.server as srv
        srv._assistant.memory = InMemoryMemory()  # isolate from real Supabase
        return TestClient(srv.app), srv

    def test_upload_csv_saves_to_knowledge_and_history(self):
        client, srv = self._client()
        csv_bytes = b"sku,qty,price\nDRESS,2,499\nSET,1,899\n"
        resp = client.post(
            "/upload",
            files={"file": ("orders.csv", csv_bytes, "text/csv")},
            data={"session": "up1"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["filename"], "orders.csv")
        self.assertIn("Rows: 2", body["analysis"])

        history = srv._assistant.memory.recent_turns("up1")
        self.assertEqual(len(history), 2)
        self.assertIn("Uploaded file", history[0]["content"])

        found = srv._assistant.memory.search_knowledge("orders")
        self.assertTrue(any("orders.csv" in k["topic"] for k in found))

    def test_upload_rejects_unsupported_extension(self):
        client, _ = self._client()
        resp = client.post("/upload", files={"file": ("bad.exe", b"x", "application/octet-stream")})
        self.assertIn("error", resp.json())
        self.assertIn("Unsupported", resp.json()["error"])

    def test_upload_rejects_oversized_file(self):
        client, _ = self._client()
        too_big = b"x" * (21 * 1024 * 1024)
        resp = client.post("/upload", files={"file": ("big.txt", too_big, "text/plain")})
        self.assertIn("error", resp.json())
        self.assertIn("too large", resp.json()["error"].lower())

    def test_upload_plain_text_file(self):
        client, _ = self._client()
        resp = client.post("/upload", files={"file": ("notes.txt", b"hello world", "text/plain")})
        self.assertEqual(resp.json()["analysis"], "hello world")


if __name__ == "__main__":
    unittest.main()

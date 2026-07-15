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


class TestLLM(unittest.TestCase):
    def test_get_provider_returns_provider(self):
        # Type depends on ambient .env; just assert we get a usable provider.
        from core.llm import LLMProvider
        self.assertIsInstance(get_provider(), LLMProvider)

    def test_echo_reply(self):
        reply = EchoProvider().chat([{"role": "user", "content": "ping"}])
        self.assertIn("ping", reply)

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


class TestAssistant(unittest.TestCase):
    def test_help_and_provider_commands(self):
        a = Assistant(session="test")
        self.assertIn("JARVIS commands", a.handle("/help"))
        self.assertIn("provider", a.handle("/provider").lower())
        self.assertEqual(a.handle("/quit"), "__quit__")


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


if __name__ == "__main__":
    unittest.main()

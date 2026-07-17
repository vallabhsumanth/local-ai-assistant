"""The JARVIS orchestrator.

Phase 1 keeps this deliberately simple: it wires together the LLM provider and
memory, exposes a handful of built-in slash-commands for file operations (so
the assistant is useful even with the echo provider), and otherwise forwards
the conversation to the LLM.

Later phases will replace the hand-written command routing with the multi-agent
planner/executor described in the original spec.
"""

from __future__ import annotations

import shlex
from typing import Iterator

from config.settings import settings
from core.agent import Agent
from core.cache import ResponseCache
from core.llm import LLMProvider, get_deep_provider, get_provider
from memory.store import get_memory
from tools import files
from tools.mailer import Emailer
from tools.registry import build_registry
from utils.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are JARVIS, a personal assistant running on the user's Mac.\n"
    "Your capabilities:\n"
    "• Natural conversation and answering questions\n"
    "• Long-term memory — remember and recall facts about the user\n"
    "• File management — list, read, write, and search files\n"
    "• Web — search the web and read/summarise pages\n"
    "• Live info — current weather and news headlines\n"
    "• Mac control — open apps and websites, control music (play/pause/next)\n"
    "• Email — draft emails and send them after you confirm\n"
    "\n"
    "IMPORTANT: When the user asks what you can do, your features, or how you "
    "can help, describe these capabilities in your own words. Do NOT call any "
    "tool for that — just answer.\n"
    "Memory is shared across every chat, not just this one. Whenever the user "
    "states a durable fact about themselves in normal conversation — their "
    "name, role, company, preferences, tools they use — call remember_fact "
    "right away, even if they never say the word 'remember'. Don't wait to be "
    "asked; if it's the kind of thing worth knowing next time, store it. Don't "
    "store one-off requests, moods, or anything temporary.\n"
    "For real tasks, call the matching tool and answer from its result. Only "
    "call get_news when the user explicitly asks about news, headlines, or "
    "current events. Only call get_weather for weather/temperature questions.\n"
    "Report tool results faithfully — never invent files, contents, or entries "
    "beyond what a tool returned. Prefer acting over asking to clarify when a "
    "tool has sensible defaults. Before deleting files, changing settings, or "
    "anything irreversible, explain it and ask first."
)

BRAND_CONTEXT_FILE = settings.root / "config" / "brand_context.txt"


def _load_brand_context() -> str:
    """Optional, editable company/brand context — kept OUT of source code so
    Nap Bot stays portable: to hand this system to someone else, just edit or
    empty this file (no code changes). Empty/missing file = no brand-specific
    behavior at all, a fully generic assistant."""
    if BRAND_CONTEXT_FILE.exists():
        text = BRAND_CONTEXT_FILE.read_text(encoding="utf-8").strip()
        if text:
            return "\n\n" + text
    return ""


BRAND_CONTEXT = _load_brand_context()
SYSTEM_PROMPT += BRAND_CONTEXT

DEEP_THINK_INSTRUCTIONS = (
    "\n\nDEEP RESEARCH MODE IS ON for this question. Take your time: make at "
    "least 2-3 separate tool calls covering different angles or sources "
    "before answering (e.g. more than one search query, or a search plus "
    "fetching a specific page). Cross-check what you find rather than "
    "settling on the first result. Only give your final answer once you've "
    "gathered enough to be thorough."
)
DEEP_THINK_MAX_STEPS = 12

HELP = """\
JARVIS commands (Phase 1)
  /help                     show this help
  /ls [path]                list a directory
  /read <path>              print a text file
  /find <root> <pattern>    recursive search, e.g. /find ~/Documents *.pdf
  /remember <key> <value>   store a durable fact
  /recall <key>             retrieve a fact
  /facts                    list everything remembered
  /provider                 show the active LLM provider
  /quit                     exit
Anything else is sent to the LLM as a chat message."""


class Assistant:
    def __init__(self, session: str = "default") -> None:
        self.session = session
        self.memory = get_memory()
        self.provider: LLMProvider = get_provider()
        self.mailer = Emailer()
        self.agent = (
            Agent(self.provider, self.memory, build_registry(self.memory, self.mailer))
            if self.provider.supports_tools
            else None
        )
        # Optional stronger model used ONLY when Deep Think is on. None if not
        # configured (DEEP_LLM_PROVIDER unset) — deep_think then falls back to
        # the everyday agent, just with a bigger step budget, as before.
        self.deep_provider: LLMProvider | None = get_deep_provider()
        self.deep_agent = (
            Agent(self.deep_provider, self.memory, build_registry(self.memory, self.mailer))
            if self.deep_provider is not None
            else None
        )
        self.cache = ResponseCache()
        self.deep_think = False

    def _active_agent(self) -> Agent | None:
        """Which agent handles the current turn: the frontier Deep Think
        agent if it's on and configured, otherwise the everyday one."""
        if self.deep_think and self.deep_agent is not None:
            return self.deep_agent
        return self.agent

    _AFFIRM = {"confirm", "yes", "send", "send it", "yes send", "confirm send",
               "ok", "okay", "yep", "go ahead", "/confirm", "do it"}
    _DENY = {"cancel", "no", "discard", "stop", "nevermind", "never mind",
             "/cancel", "don't", "dont"}

    def _touch_chat(self, text: str) -> None:
        """Register this chat on its first message, or bump its last-active
        time on every later one — drives the chat list + 20-day cleanup."""
        title = text.strip()[:60] + ("…" if len(text.strip()) > 60 else "")
        self.memory.touch_chat(self.session, title=title)

    # --- main entry ---
    def handle(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        # A pending email must be resolved (confirm/cancel) before anything else.
        if self.mailer.has_pending():
            low = text.lower().rstrip(".!")
            if low in self._AFFIRM:
                return self.mailer.confirm()
            if low in self._DENY:
                return self.mailer.cancel()
            return self.mailer.pending_reminder()

        if text.startswith("/"):
            return self._command(text)
        return self._chat(text)

    # --- LLM path ---
    def _chat(self, text: str) -> str:
        self.memory.add_turn(self.session, "user", text)
        self._touch_chat(text)

        # Fast path: a similar question answered recently → skip the LLM.
        # Deep-think mode always researches fresh, so it skips the cache
        # entirely (both reading and writing).
        if not self.deep_think:
            cached = self.cache.get(text)
            if cached is not None:
                self.memory.add_turn(self.session, "assistant", cached)
                return cached

        history = self.memory.recent_turns(self.session, limit=20)
        system = SYSTEM_PROMPT + (DEEP_THINK_INSTRUCTIONS if self.deep_think else "")
        messages = [{"role": "system", "content": system}, *history]
        try:
            active_agent = self._active_agent()
            if active_agent is not None:
                max_steps = DEEP_THINK_MAX_STEPS if self.deep_think else None
                reply = active_agent.run(messages, max_steps=max_steps)
            else:
                reply = self.provider.chat(messages)
        except Exception as exc:  # noqa: BLE001
            log.exception("LLM call failed")
            reply = f"[error talking to LLM: {exc}]"
        self.memory.add_turn(self.session, "assistant", reply)
        if not self.deep_think:
            self.cache.put(text, reply)
        return reply

    # --- streaming path (web UI only — real live typing + a working Stop
    # button). The terminal REPL keeps using handle()/_chat() above, untouched. ---
    def handle_stream(self, text: str) -> Iterator[str]:
        """Generator version of handle(): yields text chunks as they arrive.

        Non-chat paths (slash commands, pending email confirm/cancel) yield
        their single reply as one chunk; only real LLM chat truly streams.
        """
        text = text.strip()
        if not text:
            return

        if self.mailer.has_pending():
            low = text.lower().rstrip(".!")
            if low in self._AFFIRM:
                yield self.mailer.confirm()
            elif low in self._DENY:
                yield self.mailer.cancel()
            else:
                yield self.mailer.pending_reminder()
            return

        if text.startswith("/"):
            result = self._command(text)
            if result == "__quit__":
                result = "(quit is a terminal-only command; just close the tab in the web UI)"
            yield result
            return

        yield from self.stream_chat(text)

    def stream_chat(self, text: str) -> Iterator[str]:
        """Generator version of _chat(): yields text deltas as they're
        generated, then does the same memory/cache bookkeeping as _chat()
        once the full reply is known.
        """
        self.memory.add_turn(self.session, "user", text)
        self._touch_chat(text)

        if not self.deep_think:
            cached = self.cache.get(text)
            if cached is not None:
                self.memory.add_turn(self.session, "assistant", cached)
                yield cached
                return

        history = self.memory.recent_turns(self.session, limit=20)
        system = SYSTEM_PROMPT + (DEEP_THINK_INSTRUCTIONS if self.deep_think else "")
        messages = [{"role": "system", "content": system}, *history]
        full_text = ""
        try:
            active_agent = self._active_agent()
            if active_agent is not None:
                max_steps = DEEP_THINK_MAX_STEPS if self.deep_think else None
                for chunk in active_agent.run_stream(messages, max_steps=max_steps):
                    if chunk["type"] == "delta":
                        full_text += chunk["text"]
                        yield chunk["text"]
                    else:
                        full_text = chunk["text"]
            else:
                full_text = self.provider.chat(messages)
                yield full_text
        except Exception as exc:  # noqa: BLE001
            log.exception("LLM call failed")
            full_text = f"[error talking to LLM: {exc}]"
            yield full_text

        self.memory.add_turn(self.session, "assistant", full_text)
        if not self.deep_think:
            self.cache.put(text, full_text)

    # --- built-in commands ---
    def _command(self, text: str) -> str:
        try:
            parts = shlex.split(text)
        except ValueError:
            parts = text.split()
        cmd, args = parts[0].lower(), parts[1:]

        if cmd in ("/help", "/?"):
            return HELP
        if cmd == "/quit":
            return "__quit__"
        if cmd == "/provider":
            return f"Active LLM provider: {self.provider.name}"
        if cmd == "/ls":
            return "\n".join(files.list_dir(args[0] if args else "."))
        if cmd == "/read":
            if not args:
                return "usage: /read <path>"
            return files.read_text(args[0])
        if cmd == "/find":
            if len(args) < 2:
                return "usage: /find <root> <pattern>"
            hits = files.search(args[0], args[1])
            return "\n".join(hits) if hits else "(no matches)"
        if cmd == "/remember":
            if len(args) < 2:
                return "usage: /remember <key> <value>"
            self.memory.remember(args[0], " ".join(args[1:]))
            return f"Remembered {args[0]!r}."
        if cmd == "/recall":
            if not args:
                return "usage: /recall <key>"
            val = self.memory.recall(args[0])
            return val if val is not None else f"(nothing stored for {args[0]!r})"
        if cmd == "/facts":
            facts = self.memory.all_facts()
            if not facts:
                return "(no facts stored yet)"
            return "\n".join(f"{k} = {v}" for k, v in facts.items())
        return f"Unknown command {cmd!r}. Try /help."

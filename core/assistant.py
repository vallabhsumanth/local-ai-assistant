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

from core.agent import Agent
from core.cache import ResponseCache
from core.llm import LLMProvider, get_provider
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
    "For real tasks, call the matching tool and answer from its result. Only "
    "call get_news when the user explicitly asks about news, headlines, or "
    "current events. Only call get_weather for weather/temperature questions.\n"
    "Report tool results faithfully — never invent files, contents, or entries "
    "beyond what a tool returned. Prefer acting over asking to clarify when a "
    "tool has sensible defaults. Before deleting files, changing settings, or "
    "anything irreversible, explain it and ask first."
)

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
        self.cache = ResponseCache()

    _AFFIRM = {"confirm", "yes", "send", "send it", "yes send", "confirm send",
               "ok", "okay", "yep", "go ahead", "/confirm", "do it"}
    _DENY = {"cancel", "no", "discard", "stop", "nevermind", "never mind",
             "/cancel", "don't", "dont"}

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

        # Fast path: a similar question answered recently → skip the LLM.
        cached = self.cache.get(text)
        if cached is not None:
            self.memory.add_turn(self.session, "assistant", cached)
            return cached

        history = self.memory.recent_turns(self.session, limit=20)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        try:
            if self.agent is not None:
                reply = self.agent.run(messages)
            else:
                reply = self.provider.chat(messages)
        except Exception as exc:  # noqa: BLE001
            log.exception("LLM call failed")
            reply = f"[error talking to LLM: {exc}]"
        self.memory.add_turn(self.session, "assistant", reply)
        self.cache.put(text, reply)
        return reply

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

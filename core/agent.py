"""The tool-calling agent loop.

Given a conversation and a set of tools, it asks the model what to do. If the
model decides to call a tool, the agent runs it, feeds the result back, and
asks again — repeating until the model produces a final text answer (or a step
limit is hit). This is what turns JARVIS from a chatbot into an assistant that
*acts*.

Only used when the active provider advertises `supports_tools`. Otherwise the
assistant falls back to plain chat.
"""

from __future__ import annotations

from tools.registry import Tool
from utils.logger import get_logger

log = get_logger(__name__)

MAX_STEPS = 6            # safety cap on tool-call rounds
MAX_TOOL_OUTPUT = 4000   # chars of tool result fed back to the model


class Agent:
    def __init__(self, provider, memory, tools: list[Tool]) -> None:
        self.provider = provider
        self.memory = memory
        self.tools = {t.name: t for t in tools}
        self._specs = [t.spec() for t in tools]

    def run(self, messages: list[dict], max_steps: int | None = None) -> str:
        """Drive the tool loop. `messages` is mutated with intermediate turns.

        `max_steps` overrides the default budget — used by deep-research mode
        to allow more rounds of research before forcing a final answer.
        """
        for step in range(max_steps if max_steps is not None else MAX_STEPS):
            resp = self.provider.chat_with_tools(messages, self._specs)
            calls = resp.get("tool_calls") or []

            if not calls:
                return resp.get("content", "") or "(no response)"

            # Keep the assistant's tool-call message in history verbatim.
            messages.append(resp.get("raw") or {"role": "assistant", "content": ""})

            for call in calls:
                name, args = call["name"], call.get("args") or {}
                result = self._invoke(name, args)
                log.info("tool %s(%s) -> %d chars", name, args, len(result))
                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": result[:MAX_TOOL_OUTPUT],
                })

        # Ran out of steps — ask for a plain summary without tools.
        final = self.provider.chat_with_tools(messages, [])
        return final.get("content", "") or "(stopped after too many tool calls)"

    def _invoke(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'."
        try:
            out = tool.func(**args)
            return "" if out is None else str(out)
        except TypeError as exc:
            return f"Error: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface to the model, don't crash
            return f"Error running {name}: {exc}"

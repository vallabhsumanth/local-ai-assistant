"""Pluggable LLM provider layer.

You picked "decide later", so JARVIS ships with a provider *interface* and a
zero-dependency `EchoProvider` that works out of the box. Wiring a real model
later is just setting env vars — no code changes:

    JARVIS_LLM_PROVIDER=anthropic  ANTHROPIC_API_KEY=sk-...  JARVIS_LLM_MODEL=claude-sonnet-5
    JARVIS_LLM_PROVIDER=openai     OPENAI_API_KEY=sk-...     JARVIS_LLM_MODEL=gpt-4o
    JARVIS_LLM_PROVIDER=ollama                               JARVIS_LLM_MODEL=llama3

The `anthropic`/`openai` SDKs are installed on demand by the dependency
manager the first time you select that provider.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Iterator

from config.settings import settings
from core.deps import ensure_package
from utils.logger import get_logger

log = get_logger(__name__)

Message = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}


class LLMProvider(ABC):
    name: str = "base"
    supports_tools: bool = False

    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        """Return the assistant's reply for the given conversation."""

    def chat_with_tools(self, messages: list[Message], tools: list[dict]) -> dict:
        """Tool-calling variant. Returns {content, tool_calls, raw}.

        `tool_calls` is a list of {"name": str, "args": dict}. Providers that
        don't support tools fall back to plain chat (no tool_calls).
        """
        return {"content": self.chat(messages), "tool_calls": [], "raw": None}

    def chat_stream(self, messages: list[Message], tools: list[dict]) -> Iterator[dict]:
        """Streaming variant, used for live typing + a real Stop button.

        Yields {"delta": str} chunks as text becomes available, then a final
        {"done": True, "tool_calls": [...], "raw": ...}. Providers without
        real streaming support fall back to yielding the whole reply as one
        chunk (still correct, just not incremental).
        """
        result = self.chat_with_tools(messages, tools)
        if result.get("content"):
            yield {"delta": result["content"]}
        yield {"done": True, "tool_calls": result.get("tool_calls") or [],
               "raw": result.get("raw")}


class EchoProvider(LLMProvider):
    """No-network fallback. Confirms the pipeline works before you add a key."""

    name = "echo"

    def __init__(self, model: str | None = None) -> None:
        pass  # accepts `model` for a uniform constructor signature; unused

    def chat(self, messages: list[Message]) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        return (
            "[echo provider — no LLM configured] "
            f"You said: {last_user!r}. "
            "Set JARVIS_LLM_PROVIDER + an API key to enable real responses."
        )


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supports_tools = True

    def __init__(self, model: str | None = None) -> None:
        anthropic = ensure_package("anthropic")
        if anthropic is None:
            raise RuntimeError("anthropic SDK unavailable")
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.llm_model or "claude-sonnet-5"

    def chat(self, messages: list[Message]) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system or None,
            messages=convo,  # type: ignore[arg-type]
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )

    @staticmethod
    def to_anthropic_tools(tools: list[dict]) -> list[dict]:
        """Our Tool.spec() is OpenAI-shaped: {"type":"function","function":
        {name, description, parameters}}. Anthropic wants the inner dict
        directly, with "parameters" renamed to "input_schema"."""
        out = []
        for t in tools:
            fn = t.get("function", t)
            out.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        return out

    @staticmethod
    def to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict]]:
        """Converts our generic message list (incl. {"role":"tool",...}
        entries) into Anthropic's system-string + content-block convention.
        Anthropic has no "tool" role — a tool result is a "user" message
        containing a tool_result block referencing the original tool_use id.
        """
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                system_parts.append(m.get("content", ""))
            elif role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id"),
                        "content": m.get("content", ""),
                    }],
                })
            elif role == "assistant" and isinstance(m.get("content"), list):
                out.append(m)  # already Anthropic-native, from a prior turn this call
            else:
                out.append({"role": role, "content": m.get("content", "")})
        return "\n".join(p for p in system_parts if p), out

    def chat_with_tools(self, messages: list[Message], tools: list[dict]) -> dict:
        system, convo = self.to_anthropic_messages(messages)
        kwargs: dict = {"model": self._model, "max_tokens": 4096, "messages": convo}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self.to_anthropic_tools(tools)
        resp = self._client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        calls, raw_content = [], []
        for b in resp.content:
            btype = getattr(b, "type", "")
            if btype == "text":
                raw_content.append({"type": "text", "text": b.text})
            elif btype == "tool_use":
                raw_content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                calls.append({"name": b.name, "args": b.input or {}, "id": b.id})
        return {"content": text, "tool_calls": calls,
                "raw": {"role": "assistant", "content": raw_content}}


class OpenAIProvider(LLMProvider):
    name = "openai"
    supports_tools = True

    def __init__(self, model: str | None = None) -> None:
        openai = ensure_package("openai")
        if openai is None:
            raise RuntimeError("openai SDK unavailable")
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.llm_model or "gpt-4o"

    def chat(self, messages: list[Message]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def to_openai_messages(messages: list[Message]) -> list[dict]:
        """Our Tool.spec()/tool-role convention already matches OpenAI's own
        closely — this just strips our internal-only 'tool_name' key and
        passes native OpenAI assistant tool_calls messages through as-is."""
        out = []
        for m in messages:
            role = m.get("role")
            if role == "tool":
                out.append({"role": "tool", "tool_call_id": m.get("tool_call_id"),
                            "content": m.get("content", "")})
            elif role == "assistant" and "tool_calls" in m:
                out.append(m)  # already OpenAI-native, from a prior turn this call
            else:
                out.append({"role": role, "content": m.get("content", "")})
        return out

    def chat_with_tools(self, messages: list[Message], tools: list[dict]) -> dict:
        kwargs: dict = {"model": self._model, "messages": self.to_openai_messages(messages)}
        if tools:
            kwargs["tools"] = tools  # Tool.spec() is already OpenAI's exact shape
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        calls: list[dict] = []
        raw_tool_calls = None
        if msg.tool_calls:
            raw_tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"name": tc.function.name, "args": args, "id": tc.id})
                raw_tool_calls.append({"id": tc.id, "type": "function", "function":
                                        {"name": tc.function.name, "arguments": tc.function.arguments}})
        raw = {"role": "assistant", "content": msg.content}
        if raw_tool_calls:
            raw["tool_calls"] = raw_tool_calls
        return {"content": msg.content or "", "tool_calls": calls, "raw": raw}


class OllamaProvider(LLMProvider):
    name = "ollama"
    supports_tools = True

    def __init__(self, model: str | None = None) -> None:
        requests = ensure_package("requests")
        if requests is None:
            raise RuntimeError("requests unavailable")
        self._requests = requests
        self._model = model or settings.llm_model or "llama3"

    def _post(self, payload: dict) -> dict:
        resp = self._requests.post(
            f"{settings.ollama_host}/api/chat",
            json={"model": self._model, "stream": False, **payload},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()

    def chat(self, messages: list[Message]) -> str:
        return self._post({"messages": messages})["message"]["content"]

    def chat_with_tools(self, messages: list[Message], tools: list[dict]) -> dict:
        payload: dict = {"messages": messages}
        if tools:
            payload["tools"] = tools
        msg = self._post(payload)["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            calls.append({"name": fn.get("name", ""), "args": fn.get("arguments") or {}})
        return {"content": msg.get("content", ""), "tool_calls": calls, "raw": msg}

    def chat_stream(self, messages: list[Message], tools: list[dict]) -> Iterator[dict]:
        payload: dict = {"messages": messages}
        if tools:
            payload["tools"] = tools
        resp = self._requests.post(
            f"{settings.ollama_host}/api/chat",
            json={"model": self._model, "stream": True, **payload},
            stream=True, timeout=180,
        )
        resp.raise_for_status()
        content = ""
        raw_tool_calls = None
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message") or {}
                delta = msg.get("content") or ""
                if delta:
                    content += delta
                    yield {"delta": delta}
                if msg.get("tool_calls"):
                    raw_tool_calls = msg["tool_calls"]
                if chunk.get("done"):
                    break
        finally:
            resp.close()
        calls = []
        for tc in raw_tool_calls or []:
            fn = tc.get("function", {})
            calls.append({"name": fn.get("name", ""), "args": fn.get("arguments") or {}})
        raw = {"role": "assistant", "content": content}
        if raw_tool_calls:
            raw["tool_calls"] = raw_tool_calls
        yield {"done": True, "tool_calls": calls, "raw": raw}


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "echo": EchoProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def get_provider() -> LLMProvider:
    """Instantiate the configured provider, falling back to echo on failure."""
    key = settings.llm_provider.lower()
    cls = _PROVIDERS.get(key)
    if cls is None:
        log.warning("Unknown provider %r; using echo.", key)
        return EchoProvider()
    try:
        provider = cls()
        log.info("LLM provider ready: %s", provider.name)
        return provider
    except Exception as exc:  # noqa: BLE001 - want to degrade gracefully
        log.warning("Provider %r failed to init (%s); using echo.", key, exc)
        return EchoProvider()


def get_deep_provider() -> LLMProvider | None:
    """Instantiate the optional stronger model used only for Deep Think.

    Returns None if DEEP_LLM_PROVIDER isn't set, or if it fails to init —
    the caller falls back to the everyday provider (still boosted with more
    steps) rather than silently downgrading to echo, which would be a
    confusing regression specifically for the "smarter analysis" mode.
    """
    if not settings.deep_think_configured:
        return None
    key = settings.deep_llm_provider.lower()
    cls = _PROVIDERS.get(key)
    if cls is None:
        log.warning("Unknown DEEP_LLM_PROVIDER %r; Deep Think will use the "
                     "everyday model instead.", key)
        return None
    try:
        provider = cls(model=settings.deep_llm_model or None)
        if not provider.supports_tools:
            log.warning("Deep Think provider %r doesn't support tools; "
                         "using the everyday model instead.", provider.name)
            return None
        log.info("Deep Think provider ready: %s", provider.name)
        return provider
    except Exception as exc:  # noqa: BLE001 - degrade, don't crash
        log.warning("Deep Think provider %r failed to init (%s); using the "
                     "everyday model instead.", key, exc)
        return None

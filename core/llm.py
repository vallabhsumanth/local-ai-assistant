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

from abc import ABC, abstractmethod

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


class EchoProvider(LLMProvider):
    """No-network fallback. Confirms the pipeline works before you add a key."""

    name = "echo"

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

    def __init__(self) -> None:
        anthropic = ensure_package("anthropic")
        if anthropic is None:
            raise RuntimeError("anthropic SDK unavailable")
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model or "claude-sonnet-5"

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


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        openai = ensure_package("openai")
        if openai is None:
            raise RuntimeError("openai SDK unavailable")
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model or "gpt-4o"

    def chat(self, messages: list[Message]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        return resp.choices[0].message.content or ""


class OllamaProvider(LLMProvider):
    name = "ollama"
    supports_tools = True

    def __init__(self) -> None:
        requests = ensure_package("requests")
        if requests is None:
            raise RuntimeError("requests unavailable")
        self._requests = requests
        self._model = settings.llm_model or "llama3"

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

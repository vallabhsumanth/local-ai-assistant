"""Tool registry — the functions JARVIS's LLM is allowed to call.

Each Tool bundles: a name, a natural-language description (the model reads this
to decide when to use it), a JSON-schema for its arguments, and the Python
callable that actually runs.

`build_registry(memory)` wires the file tools and the (instance-bound) memory
tools into a list the agent can offer to the model.

Safety: only non-destructive operations are exposed for autonomous calling.
Deleting/moving files is deliberately NOT here — those stay behind explicit
slash-commands with confirmation, so the model can't erase data on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from browser import web
from desktop import apps
from tools import files, sheets
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON schema (object)
    func: Callable[..., Any]

    def spec(self) -> dict:
        """OpenAI/Ollama-style function schema the model sees."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def build_registry(memory, mailer=None) -> list[Tool]:
    """Return the tools available to the agent, bound to this memory instance."""
    tools = [
        Tool(
            "list_directory",
            "List the files and folders in a directory on the user's Mac. "
            "Use for questions like 'what's in my Documents folder'.",
            _obj({"path": {"type": "string", "description": "Directory path, e.g. '~/Documents'. Defaults to current dir."}}, []),
            lambda path=".": "\n".join(files.list_dir(path)),
        ),
        Tool(
            "read_file",
            "Read and return the text contents of a file.",
            _obj({"path": {"type": "string", "description": "Path to the file."}}, ["path"]),
            lambda path: files.read_text(path),
        ),
        Tool(
            "write_file",
            "Create or overwrite a text file with the given content.",
            _obj({
                "path": {"type": "string", "description": "Path to write to."},
                "content": {"type": "string", "description": "Text to write."},
            }, ["path", "content"]),
            lambda path, content: str(files.write_text(path, content)),
        ),
        Tool(
            "search_files",
            "Recursively search a folder for files matching a glob pattern "
            "(e.g. '*.pdf'). Returns matching paths.",
            _obj({
                "root": {"type": "string", "description": "Folder to search under."},
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.txt'."},
            }, ["root", "pattern"]),
            lambda root, pattern: "\n".join(files.search(root, pattern)) or "(no matches)",
        ),
        Tool(
            "remember_fact",
            "Store a durable fact about the user for later — their name, role, "
            "company, preferences, or common folders/tools. Call this whenever "
            "the user states something like this in normal conversation, even "
            "if they never say the word 'remember' — e.g. 'I'm Asim' or 'I run "
            "Nap Chief' should be stored proactively, not just explicit "
            "'remember X' requests. This is shared across every chat, not just "
            "the current one, so it's worth capturing. Don't store one-off "
            "requests, moods, or anything temporary — only durable facts.",
            _obj({
                "key": {"type": "string", "description": "Short identifier, e.g. 'name'."},
                "value": {"type": "string", "description": "The value to store."},
            }, ["key", "value"]),
            lambda key, value: (memory.remember(key, value) or f"Remembered {key} = {value}"),
        ),
        Tool(
            "recall_fact",
            "Look up a previously stored fact by its key.",
            _obj({"key": {"type": "string", "description": "The fact's key."}}, ["key"]),
            lambda key: (memory.recall(key) or f"(nothing stored for '{key}')"),
        ),
        Tool(
            "list_facts",
            "List everything currently remembered about the user.",
            _obj({}, []),
            lambda: "\n".join(f"{k} = {v}" for k, v in memory.all_facts().items()) or "(nothing remembered yet)",
        ),
        Tool(
            "web_search",
            "Search the web for current information and return the top results "
            "(title, url, snippet). Use for questions about recent events, "
            "facts, products, or anything you don't already know.",
            _obj({"query": {"type": "string", "description": "The search query."}}, ["query"]),
            lambda query: "\n\n".join(
                f"{r['title']}\n{r['url']}\n{r['snippet']}" for r in web.web_search(query)
            ) or "(no results)",
        ),
        Tool(
            "fetch_page",
            "Open a web page in a headless browser and return its title and "
            "visible text. Use to read or summarize a specific URL.",
            _obj({"url": {"type": "string", "description": "The page URL."}}, ["url"]),
            lambda url: (lambda p: f"{p['title']}\n\n{p['text']}")(web.fetch_page(url)),
        ),
        Tool(
            "get_weather",
            "Get the current weather and temperature for a city or place. "
            "Use whenever the user asks about weather or temperature.",
            _obj({"location": {"type": "string", "description": "City or place, e.g. 'Mumbai'."}}, ["location"]),
            lambda location: web.get_weather(location),
        ),
        Tool(
            "get_news",
            "Get current news headlines. Use ONLY when the user explicitly asks "
            "about news, headlines, or current events. Do NOT use it for "
            "questions about your own capabilities or general chat. Leave topic "
            "empty for top world news, or pass a topic (e.g. 'technology').",
            _obj({"topic": {"type": "string", "description": "Optional topic; empty = top headlines."}}, []),
            lambda topic="": "\n".join(f"- {h['title']}" for h in web.get_news(topic)) or "(no headlines)",
        ),
        Tool(
            "open_app",
            "Open an application on the user's Mac by name (e.g. 'Spotify', "
            "'Safari', 'Notes'). Also handles web apps like 'Shopify'. Use when "
            "the user says 'open X'. Announce what you're opening.",
            _obj({"name": {"type": "string", "description": "App or site name, e.g. 'Spotify' or 'Shopify'."}}, ["name"]),
            lambda name: apps.open_app(name),
        ),
        Tool(
            "open_website",
            "Open a website/URL in the user's default browser.",
            _obj({"url": {"type": "string", "description": "URL or known site name."}}, ["url"]),
            lambda url: apps.open_website(url),
        ),
        Tool(
            "control_music",
            "Control music playback on the Mac: play, pause, resume, next, or "
            "previous. Works with Spotify or Apple Music.",
            _obj({
                "action": {"type": "string", "description": "One of: play, pause, resume, next, previous."},
                "app": {"type": "string", "description": "Optional: 'Spotify' or 'Music'."},
            }, ["action"]),
            lambda action, app="": apps.control_music(action, app),
        ),
        Tool(
            "play_song",
            "Search for a specific song and start playing it immediately "
            "(Spotify by default). Use whenever the user asks to play a song, "
            "artist, or album by name, e.g. 'play Blinding Lights'.",
            _obj({"query": {"type": "string", "description": "Song / artist to search for."}}, ["query"]),
            lambda query, app="Spotify": apps.play_song(query, app),
        ),
        Tool(
            "read_spreadsheet",
            "Read an Excel (.xlsx/.xls) or CSV file — orders, inventory, ad "
            "performance exports, any tabular data. Returns column names, "
            "full summary statistics computed over every row, and a sample "
            "of the actual rows. Use whenever the user references a "
            "spreadsheet file or asks you to analyze order/sales data.",
            _obj({"path": {"type": "string", "description": "Path to the .xlsx/.csv file."}}, ["path"]),
            lambda path: sheets.read_spreadsheet(path),
        ),
        Tool(
            "save_knowledge",
            "Save a research finding to the permanent knowledge base — a "
            "summary of what you learned about a topic (a competitor, a "
            "market trend, anything researched via web_search/fetch_page). "
            "Call this after researching something non-trivial, so future "
            "conversations can build on it instead of re-researching cold. "
            "Don't save trivial or one-off facts — only real findings.",
            _obj({
                "topic": {"type": "string", "description": "Short topic label, e.g. 'Nauti Nati pricing'."},
                "content": {"type": "string", "description": "What you found — a clear, factual summary."},
                "source": {"type": "string", "description": "Optional: where this came from, e.g. a URL."},
            }, ["topic", "content"]),
            lambda topic, content, source=None: (
                memory.save_knowledge(topic, content, source)
                or f"Saved to knowledge base: {topic}"
            ),
        ),
        Tool(
            "search_knowledge",
            "Search the permanent knowledge base for research saved in past "
            "conversations. Call this BEFORE doing a fresh web search on a "
            "topic you might already have researched — avoids redundant work "
            "and lets you build on what's already known.",
            _obj({"query": {"type": "string", "description": "What to search for."}}, ["query"]),
            lambda query: "\n\n".join(
                f"{k['topic']}: {k['content']}" for k in memory.search_knowledge(query)
            ) or "(nothing saved on this yet)",
        ),
    ]

    if mailer is not None:
        tools.append(Tool(
            "send_email",
            "Draft an email to send on the user's behalf. This does NOT send "
            "immediately — it prepares the email for the user to review and "
            "confirm. Use when the user asks to email or mail someone.",
            _obj({
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Subject line."},
                "body": {"type": "string", "description": "Body of the email."},
            }, ["to", "subject", "body"]),
            lambda to, subject, body: mailer.stage(to, subject, body),
        ))

    return tools

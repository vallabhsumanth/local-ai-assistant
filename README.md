# JARVIS — personal AI assistant (macOS)

A modular, extensible personal assistant. **This is Phase 1: the core foundation.**
It runs today with zero external dependencies and a built-in fallback provider,
so you can use it before adding an API key.

## Quick start

```bash
cd jarvis
./install.sh      # creates myenv, seeds .env
```

**Two ways to run — same core:**

```bash
./run.sh          # terminal REPL   (type /help)
./serve.sh        # web UI, opens FULLSCREEN in Chrome app mode
```

`./serve.sh` opens JARVIS in a chromeless, fullscreen Chrome window (falls back
to your default browser + an in-app ⛶ fullscreen button).

### Voice (free — browser built-in, no model)

- **🔊 speaks replies** aloud (Web Speech synthesis) — toggle in the top bar.
- **🎤 talk to JARVIS** — click the mic, speak, it transcribes and sends.

Voice uses the browser's free Web Speech API (no key, no model, no cost).
Speech-to-text works best in **Chrome** (it uses Google's engine and sends
audio there for transcription); text-to-speech works everywhere.

## Run modes: local vs cloud (Railway)

One codebase, two environments:

- **Local (your Mac):** full access — files, and (later phases) desktop/apps/voice.
- **Railway (cloud container):** the cloud-safe core — chat, Supabase storage,
  web API, and (later) headless browser + code execution. It **cannot** touch
  your Mac's files/desktop; destructive file ops are auto-denied in web mode.

### Deploy to Railway

1. Push this repo to GitHub.
2. On [railway.app](https://railway.app): **New Project → Deploy from GitHub**.
   Railway auto-detects the `Dockerfile` / `railway.json`.
3. Add environment variables (**Variables** tab): `SUPABASE_URL`, `SUPABASE_KEY`,
   and your LLM vars (`JARVIS_LLM_PROVIDER`, key, `JARVIS_LLM_MODEL`).
   Railway provides `PORT` automatically.
4. Deploy. Healthcheck hits `/health`; the web UI is at your Railway URL.

Files involved: [`Dockerfile`](Dockerfile), [`railway.json`](railway.json),
[`Procfile`](Procfile), [`.dockerignore`](.dockerignore).

## What Phase 1 includes

| Area | Status |
|------|--------|
| Project scaffold + `myenv` venv | ✅ |
| Auto dependency manager (allow-listed, safe) | ✅ |
| Pluggable LLM provider (echo / Anthropic / OpenAI / Ollama) | ✅ |
| Long-term memory in the **cloud (Supabase)**: conversations + facts | ✅ |
| File management (read/write/list/search/copy/move/rename/delete/zip/unzip) | ✅ |
| Confirmation gate on destructive actions | ✅ |
| Logging (console + rotating file) | ✅ |
| **Web layer** (FastAPI API + browser chat UI) | ✅ |
| **Railway deploy** config (Dockerfile, railway.json) | ✅ |
| **Real LLM** via Ollama (local) — pluggable to hosted APIs | ✅ |
| **Tool-calling agent** — acts on natural language | ✅ (Phase 2) |
| Smoke tests | ✅ |

## How JARVIS acts (tool-calling)

JARVIS isn't just a chatbot — the LLM can call functions. When you type
natural language, the agent ([`core/agent.py`](core/agent.py)) offers the model
a set of tools ([`tools/registry.py`](tools/registry.py)); the model decides
which to call, JARVIS runs them, and answers from the results.

Available tools: `list_directory`, `read_file`, `write_file`, `search_files`,
`remember_fact`, `recall_fact`, `list_facts`, `web_search`, `fetch_page`,
`get_weather`, `get_news`, `send_email`. Destructive ops (delete/move) are
intentionally kept out of autonomous calling — use slash-commands with
confirmation for those.

**Email** (`send_email`, Gmail SMTP): drafts an email and **always waits for
your confirmation** — it never auto-sends. JARVIS shows To/Subject/Body; reply
`confirm` to send or `cancel` to discard. Setup: create a Gmail **app password**
(Google Account → Security → App passwords) and set `SMTP_USER` +
`SMTP_PASSWORD` in `.env`.

**Mac control** (`open_app`, `open_website`, `control_music`, `play_song`):
LOCAL ONLY — works when JARVIS runs on your Mac, not on Railway (cloud has no
desktop). Open any app ("open Spotify", "open Shopify"), browse, and control
playback (play/pause/resume/next/previous). JARVIS announces what it's doing
("Opening Shopify…") — and with voice on, says it aloud. Uses `open` +
AppleScript; scoped to launching/playback, not arbitrary scripting.

Weather (`get_weather`, via wttr.in) and news (`get_news`, via Google News RSS)
work with **no API key**. General web search uses the keyless DuckDuckGo API by
default; add `BRAVE_API_KEY` for ranked results.

**Web / browser** (headless Chromium via Playwright): `fetch_page` reads any
URL; `web_search` searches the web. Search uses the keyless DuckDuckGo API by
default — set `BRAVE_API_KEY` in `.env` for product-grade ranked results
(free key at brave.com/search/api). Read-only by design: no autonomous form
filling, clicking, or logins. Local setup: `playwright install chromium` (the
Docker image does this automatically for Railway).

Example: *"Remember my company is Nap Chief"* → JARVIS calls `remember_fact` and
stores it in Supabase. Later, *"What company do I work at?"* → it recalls it.

> Tool-calling quality scales with the model. `qwen2.5:1.5b` (local) works but
> can embellish; a larger local model or a hosted API (on Railway) is crisper.

## Memory & cache

JARVIS has four layers:
1. **Short-term** — the last ~20 turns of the session are fed to the model as
   context each reply.
2. **Long-term facts** — durable key/value memory the LLM reads/writes via
   `remember_fact` / `recall_fact` tools.
3. **Storage** — both live in Supabase (`napbot_conversations`, `napbot_facts`).
4. **Response cache** ([`core/cache.py`](core/cache.py)) — similar/rephrased
   questions return instantly without an LLM call. Uses lexical similarity (no
   model), TTL + LRU, and skips volatile (weather/news/"now") and personal
   ("what's my name") questions so it never serves stale answers.

## Cloud storage (Supabase)

All JARVIS data is stored in Supabase — **nothing user-facing is written to
local disk** (only operational logs stay in `logs/`). Until you add credentials,
JARVIS runs on an ephemeral in-RAM store that clears on exit.

**Setup:**
1. Create a project at [supabase.com](https://supabase.com).
2. Open **SQL Editor**, paste [`memory/schema.sql`](memory/schema.sql), and Run.
3. In **Settings → API**, copy the Project URL and a key.
4. Add them to `.env`:
   ```dotenv
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-service-role-or-anon-key
   ```

That's it — on next launch JARVIS uses Supabase automatically.

## Enabling a real LLM

Edit `.env`:

```dotenv
JARVIS_LLM_PROVIDER=anthropic
JARVIS_LLM_MODEL=claude-sonnet-5
ANTHROPIC_API_KEY=sk-ant-...
```

The matching SDK (`anthropic`, `openai`, or `requests` for Ollama) installs
automatically the first time you select that provider. No code changes needed.

## Built-in commands

```
/help                     show help
/ls [path]                list a directory
/read <path>              print a text file
/find <root> <pattern>    recursive search, e.g. /find ~/Documents *.pdf
/remember <key> <value>   store a durable fact
/recall <key>             retrieve a fact
/facts                    list everything remembered
/provider                 show the active LLM provider
/quit                     exit
```

Anything not starting with `/` is sent to the LLM.

## Project layout

```
jarvis/
├── app.py            # REPL entry point
├── install.sh        # installer (creates myenv)
├── run.sh            # launcher (uses myenv's python)
├── config/settings.py
├── core/
│   ├── assistant.py  # orchestrator
│   ├── llm.py        # pluggable providers
│   └── deps.py       # safe dependency manager
├── memory/
│   ├── store.py          # storage abstraction + factory + RAM fallback
│   ├── supabase_store.py # cloud backend
│   └── schema.sql        # run this in Supabase once
├── tools/files.py    # file operations
├── utils/logger.py
├── logs/
└── tests/test_core.py
```

## Roadmap (later phases)

- **Phase 2:** code generation + sandboxed execution, terminal/zsh automation, browser automation (Playwright).
- **Phase 3:** voice (wake word, STT, TTS), OCR + screenshot analysis.
- **Phase 4:** macOS app control (AppleScript/Automator), plugin system, full multi-agent planner/executor.

## Safety

JARVIS is for your own machine. Destructive actions (delete, move) require
confirmation. Runtime auto-install is limited to an explicit allow-list rather
than installing arbitrary imports.

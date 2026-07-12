#!/usr/bin/env bash
# Launch the JARVIS web server locally and open it FULLSCREEN.
# Chrome (if present) opens in chromeless app + fullscreen mode for a native
# feel; otherwise your default browser opens the URL.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "myenv" ]; then
  echo "myenv not found — running installer first."
  ./install.sh
fi

PORT="${PORT:-8000}"
URL="http://localhost:${PORT}"

# Ensure Ollama (local LLM) is running.
if ! curl -sf http://localhost:11434/ >/dev/null 2>&1; then
  OLLAMA_BIN="$(command -v ollama || echo /opt/homebrew/opt/ollama/bin/ollama)"
  if [ -x "$OLLAMA_BIN" ]; then
    echo "==> Starting Ollama..."
    "$OLLAMA_BIN" serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 20); do
      curl -sf http://localhost:11434/ >/dev/null 2>&1 && break
      sleep 0.3
    done
  else
    echo "!! Ollama not found — install it or set JARVIS_LLM_PROVIDER in .env"
  fi
fi

# Start the server in the background.
PORT="$PORT" myenv/bin/python -m api.server &
SERVER_PID=$!

# Stop the server when this script exits (Ctrl+C).
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait for it to come up.
for _ in $(seq 1 30); do
  if curl -sf "${URL}/health" >/dev/null 2>&1; then break; fi
  sleep 0.3
done

# Open fullscreen.
CHROME="/Applications/Google Chrome.app"
if [ -d "$CHROME" ]; then
  echo "==> Opening JARVIS fullscreen in Chrome app mode"
  open -na "Google Chrome" --args --app="$URL" --start-fullscreen >/dev/null 2>&1 || open "$URL"
else
  echo "==> Opening JARVIS in your default browser (press the ⛶ button for fullscreen)"
  open "$URL"
fi

echo "==> JARVIS running at ${URL}  (Ctrl+C to stop)"
wait $SERVER_PID

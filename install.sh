#!/usr/bin/env bash
# JARVIS installer (macOS). Idempotent: safe to re-run.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> JARVIS installer"

# 1. Ensure myenv exists
if [ ! -d "myenv" ]; then
  echo "==> Creating virtual environment 'myenv'"
  python3 -m venv myenv
else
  echo "==> Reusing existing 'myenv'"
fi

# 2. Activate + upgrade pip
# shellcheck disable=SC1091
source myenv/bin/activate
python -m pip install --quiet --upgrade pip

# 3. Install requirements (Phase 1 has none required, but this is future-proof)
if [ -s requirements.txt ] && grep -qvE '^\s*#|^\s*$' requirements.txt; then
  echo "==> Installing requirements"
  pip install -r requirements.txt
else
  echo "==> No required packages (Phase 1 runs on the stdlib)"
fi

# 4. Seed .env if missing
if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env from template (edit it to add an LLM key)"
fi

echo "==> Done. Start JARVIS with:  ./run.sh"

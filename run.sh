#!/usr/bin/env bash
# Launch JARVIS using myenv's interpreter, creating the venv if needed.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "myenv" ]; then
  echo "myenv not found — running installer first."
  ./install.sh
fi

exec myenv/bin/python app.py "$@"

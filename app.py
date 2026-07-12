#!/usr/bin/env python3
"""JARVIS entry point — interactive REPL.

Run it after activating the venv:

    source myenv/bin/activate
    python app.py

Or use the wrapper that guarantees the right interpreter:

    ./run.sh
"""

from __future__ import annotations

import sys

# Make sure the project root is importable no matter where we're launched from.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.assistant import Assistant  # noqa: E402
from core.deps import verify_environment  # noqa: E402
from tools import files  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger("jarvis.app")

BANNER = r"""
   _   _   ___  _   _ ___ ____
  | | / \ | _ \| | | |_ _/ ___|    JARVIS  (Phase 1 core)
 _| |/ _ \|   /| |_| || |\___ \    macOS personal assistant
|__/_/ \_\_|_\ \___/|___|____/     type /help  ·  /quit to exit
"""


def _confirm(message: str) -> bool:
    try:
        answer = input(f"\n⚠️  {message}  [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def main() -> int:
    files.set_confirm_handler(_confirm)

    for warning in verify_environment():
        print(f"⚠️  {warning}\n")

    assistant = Assistant()
    print(BANNER)
    print(f"LLM provider: {assistant.provider.name}\n")

    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye 👋")
            return 0

        if not user:
            continue

        reply = assistant.handle(user)
        if reply == "__quit__":
            print("bye 👋")
            return 0
        if reply:
            print(f"\njarvis › {reply}\n")


if __name__ == "__main__":
    raise SystemExit(main())

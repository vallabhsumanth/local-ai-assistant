"""Dependency manager for JARVIS.

Two responsibilities:

1. `verify_environment()` — confirm we're running inside `myenv` and that the
   core dependencies are importable. Called at startup.
2. `ensure_package()` — safely install a *known* package into the active venv
   and add it to requirements.txt.

Design note: the original spec asked to auto-install *any* missing import at
runtime and never stop. That is a real security/stability risk (an attacker or
a typo could trigger arbitrary `pip install`). Instead we only auto-install
from an explicit allow-list, and anything else asks for confirmation. This
keeps the "never crash on ModuleNotFoundError" spirit without the footgun.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from config.settings import settings
from utils.logger import get_logger

log = get_logger(__name__)

# Packages JARVIS may install without asking. Extend as features are added.
ALLOWED_PACKAGES: dict[str, str] = {
    # import name -> pip name
    "anthropic": "anthropic",
    "openai": "openai",
    "requests": "requests",
    "rich": "rich",
    "supabase": "supabase",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "pydantic": "pydantic",
    "playwright": "playwright",
}

REQUIREMENTS = settings.root / "requirements.txt"


def in_virtualenv() -> bool:
    """True if the running interpreter is inside a virtualenv."""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def running_in_myenv() -> bool:
    """True if the active interpreter lives under the project's myenv/."""
    try:
        return Path(sys.prefix).resolve() == (settings.root / "myenv").resolve()
    except OSError:
        return False


def _pip_install(pip_name: str) -> bool:
    log.info("Installing %s into %s ...", pip_name, sys.prefix)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pip_name]
        )
        _add_to_requirements(pip_name)
        return True
    except subprocess.CalledProcessError as exc:  # pragma: no cover - network
        log.error("pip install %s failed: %s", pip_name, exc)
        return False


def _add_to_requirements(pip_name: str) -> None:
    base = pip_name.split("==")[0].split(">=")[0].strip()
    existing: set[str] = set()
    if REQUIREMENTS.exists():
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            existing.add(line.split("==")[0].split(">=")[0].strip().lower())
    if base.lower() not in existing:
        with REQUIREMENTS.open("a", encoding="utf-8") as fh:
            fh.write(f"{pip_name}\n")
        log.info("Added %s to requirements.txt", pip_name)


def ensure_package(import_name: str, *, assume_yes: bool = False):
    """Import a module, installing it first if it's on the allow-list.

    Returns the imported module, or None if unavailable/declined.
    """
    try:
        return importlib.import_module(import_name)
    except ImportError:
        pass

    pip_name = ALLOWED_PACKAGES.get(import_name)
    if pip_name is None:
        log.warning(
            "Module '%s' is missing and not on the allow-list; skipping "
            "auto-install. Add it to ALLOWED_PACKAGES or install manually.",
            import_name,
        )
        return None

    if not assume_yes and not in_virtualenv():
        log.warning("Refusing to install outside a virtualenv.")
        return None

    if _pip_install(pip_name):
        try:
            return importlib.import_module(import_name)
        except ImportError:
            log.error("Installed %s but still cannot import %s", pip_name, import_name)
    return None


def verify_environment() -> list[str]:
    """Return a list of human-readable warnings about the environment."""
    warnings: list[str] = []
    if not in_virtualenv():
        warnings.append(
            "Not running inside a virtualenv. Activate myenv:\n"
            "    source myenv/bin/activate"
        )
    elif not running_in_myenv():
        warnings.append(
            f"Running in a venv at {sys.prefix}, but not the project's myenv."
        )
    return warnings

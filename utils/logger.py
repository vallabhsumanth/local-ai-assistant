"""Logging setup for JARVIS.

Provides `get_logger(name)` which returns a logger that writes to both the
console and a rotating file in `logs/`. Import and call once at startup via
`setup_logging()`; individual modules just call `get_logger(__name__)`.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config.settings import settings

_CONFIGURED = False


def setup_logging() -> None:
    """Configure the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.ensure_dirs()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(level)

    file_handler = RotatingFileHandler(
        settings.log_dir / "jarvis.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)  # file always keeps detail

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)

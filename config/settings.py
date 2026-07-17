"""Central configuration for JARVIS.

Loads values from environment variables (optionally via a `.env` file) and
exposes them as a single `settings` object. Nothing here is macOS-specific yet,
but paths default to sensible locations inside the project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = the directory that contains this `config/` package.
ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't require python-dotenv for Phase 1.

    Lines look like KEY=value. Existing environment variables win, so real
    shell exports always override the file.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    root: Path = ROOT

    # --- LLM provider (pluggable; pick the actual key/model later) ---
    llm_provider: str = os.environ.get("JARVIS_LLM_PROVIDER", "echo")
    llm_model: str = os.environ.get("JARVIS_LLM_MODEL", "")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # --- Deep Think: an optional stronger model used only when the Deep
    # Research toggle is on. Falls back to the everyday provider above (still
    # boosted with more steps + deeper instructions) if this isn't set. ---
    deep_llm_provider: str = os.environ.get("DEEP_LLM_PROVIDER", "")
    deep_llm_model: str = os.environ.get("DEEP_LLM_MODEL", "")

    # --- Cloud storage (Supabase) — all user data lives here, not on disk ---
    supabase_url: str = os.environ.get("SUPABASE_URL", "")
    supabase_key: str = os.environ.get("SUPABASE_KEY", "")

    # --- Spotify (optional; lets JARVIS resolve a named song to an exact
    # track for instant playback — free app at developer.spotify.com) ---
    spotify_client_id: str = os.environ.get("SPOTIFY_CLIENT_ID", "")
    spotify_client_secret: str = os.environ.get("SPOTIFY_CLIENT_SECRET", "")

    # --- Email (Gmail SMTP; sending always requires user confirmation) ---
    smtp_host: str = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.environ.get("SMTP_PORT", "465"))
    smtp_user: str = os.environ.get("SMTP_USER", "")
    smtp_password: str = os.environ.get("SMTP_PASSWORD", "")
    email_from: str = os.environ.get("EMAIL_FROM", "")

    # Local logs are operational diagnostics only, not user data.
    log_dir: Path = ROOT / "logs"

    # --- Behaviour ---
    log_level: str = os.environ.get("JARVIS_LOG_LEVEL", "INFO")
    confirm_destructive: bool = os.environ.get("JARVIS_CONFIRM", "1") != "0"

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    @property
    def deep_think_configured(self) -> bool:
        return bool(self.deep_llm_provider)

    @property
    def spotify_configured(self) -> bool:
        return bool(self.spotify_client_id and self.spotify_client_secret)

    @property
    def email_configured(self) -> bool:
        return bool(self.smtp_user and self.smtp_password)

    @property
    def email_sender(self) -> str:
        return self.email_from or self.smtp_user

    def ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

"""Email sending for JARVIS — Gmail SMTP, with mandatory confirmation.

Sending email is irreversible and outward-facing, so it's a TWO-STEP flow:

1. `stage(to, subject, body)` — the agent calls this; it does NOT send. It
   stores a pending draft and returns a preview for the user to review.
2. The user replies "confirm" (handled in `Assistant.handle`) → `confirm()`
   actually sends via SMTP. "cancel" → `cancel()` discards it.

This works identically in the web UI and the terminal — no mid-request prompt.
Credentials come from `.env` (SMTP_USER + an app password); JARVIS never sees
them typed in chat.
"""

from __future__ import annotations

import re
import smtplib
from email.message import EmailMessage

from config.settings import settings
from utils.logger import get_logger

log = get_logger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Emailer:
    def __init__(self) -> None:
        self._pending: dict | None = None

    def has_pending(self) -> bool:
        return self._pending is not None

    def stage(self, to: str, subject: str, body: str) -> str:
        """Prepare a draft for confirmation. Does not send."""
        to = (to or "").strip()
        if not _EMAIL_RE.match(to):
            return f"That doesn't look like a valid email address: {to!r}."
        self._pending = {
            "to": to,
            "subject": (subject or "").strip() or "(no subject)",
            "body": (body or "").strip(),
        }
        note = ""
        if not settings.email_configured:
            note = ("\n\n⚠️ Email isn't configured yet — add SMTP_USER and "
                    "SMTP_PASSWORD (a Gmail app password) to .env to actually send.")
        p = self._pending
        log.info("Email staged to %s", p["to"])
        return (
            "📧 Draft ready — please review before it goes out:\n"
            f"To: {p['to']}\n"
            f"Subject: {p['subject']}\n\n"
            f"{p['body']}\n\n"
            "Reply 'confirm' to send, or 'cancel' to discard." + note
        )

    def pending_reminder(self) -> str:
        p = self._pending or {}
        return (f"You have a pending email to {p.get('to')} "
                f"(subject: {p.get('subject')}). Reply 'confirm' to send "
                "or 'cancel' to discard.")

    def cancel(self) -> str:
        self._pending = None
        return "Email discarded — nothing was sent."

    def confirm(self) -> str:
        if not self._pending:
            return "There's no pending email to send."
        if not settings.email_configured:
            return ("Can't send: email isn't configured. Add SMTP_USER and "
                    "SMTP_PASSWORD (Gmail app password) to .env, then try again.")
        p = self._pending
        try:
            self._send(p["to"], p["subject"], p["body"])
        except Exception as exc:  # noqa: BLE001 - report cleanly to the user
            log.exception("Email send failed")
            return f"❌ Failed to send: {exc}"
        self._pending = None
        log.info("Email sent to %s", p["to"])
        return f"✅ Email sent to {p['to']}."

    def _send(self, to: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"] = settings.email_sender
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=30) as s:
            s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)

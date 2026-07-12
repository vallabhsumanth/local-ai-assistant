"""macOS app & media control for JARVIS (LOCAL ONLY).

Opens applications, opens websites, and controls music playback via `open`
and AppleScript (`osascript`). These only work when JARVIS runs on the Mac
itself — in a cloud container (Railway) there's no desktop, so every function
returns a clear "local only" message instead of failing.

Scope is deliberately narrow and non-destructive: launch apps, open URLs,
transport controls (play/pause/next/previous). It is NOT a general
"run any AppleScript" escape hatch — that would be arbitrary code execution.
"""

from __future__ import annotations

import subprocess
import sys

from utils.logger import get_logger

log = get_logger(__name__)

IS_MAC = sys.platform == "darwin"

# Friendly names → URLs, for things that live on the web (no native Mac app).
KNOWN_SITES = {
    "shopify": "https://admin.shopify.com",
    "gmail": "https://mail.google.com",
    "youtube": "https://youtube.com",
    "github": "https://github.com",
    "supabase": "https://supabase.com/dashboard",
    "google": "https://google.com",
    "maps": "https://maps.google.com",
    "whatsapp": "https://web.whatsapp.com",
}

# Music transport verbs → AppleScript command.
_MUSIC_CMDS = {
    "play": "play", "resume": "play", "start": "play",
    "pause": "pause", "stop": "pause",
    "playpause": "playpause", "toggle": "playpause",
    "next": "next track", "skip": "next track",
    "previous": "previous track", "back": "previous track", "prev": "previous track",
}

_NOT_MAC = ("This only works when JARVIS is running locally on your Mac "
            "(not in the cloud).")


def _run(args: list[str], timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def open_app(name: str) -> str:
    """Launch a Mac application by name (e.g. 'Spotify', 'Safari').

    Falls back to opening a known website if there's no native app of that name
    (e.g. 'shopify' → the Shopify admin in the browser).
    """
    if not IS_MAC:
        return _NOT_MAC
    name = name.strip()
    try:
        r = _run(["open", "-a", name])
        if r.returncode == 0:
            log.info("Opened app %r", name)
            return f"Opening {name}…"
    except Exception as exc:  # noqa: BLE001
        log.warning("open -a failed: %s", exc)
    # fall back to a known website
    site = KNOWN_SITES.get(name.lower())
    if site:
        _run(["open", site])
        log.info("Opened site for %r -> %s", name, site)
        return f"Opening {name} in your browser…"
    return (f"I couldn't find an app called {name!r} on this Mac. "
            "Try the exact app name, or ask me to open a website.")


def open_website(url: str) -> str:
    """Open a URL (or a known site name) in the default browser."""
    if not IS_MAC:
        return _NOT_MAC
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = KNOWN_SITES.get(url.lower(), "https://" + url)
    _run(["open", url])
    log.info("Opened website %s", url)
    return f"Opening {url}…"


def _running_music_app() -> str | None:
    for app in ("Spotify", "Music"):
        try:
            r = _run(["osascript", "-e",
                      f'tell application "System Events" to (name of processes) contains "{app}"'])
            if r.returncode == 0 and "true" in r.stdout.lower():
                return app
        except Exception:  # noqa: BLE001
            pass
    return None


def control_music(action: str, app: str = "") -> str:
    """Control music playback: play, pause, resume, next, or previous.

    Targets Spotify or Apple Music (whichever is running; defaults to Music).
    """
    if not IS_MAC:
        return _NOT_MAC
    cmd = _MUSIC_CMDS.get(action.strip().lower())
    if not cmd:
        return (f"I can play, pause, resume, skip, or go back — not {action!r}.")
    target = app.strip() or _running_music_app() or "Music"
    try:
        r = _run(["osascript", "-e", f'tell application "{target}" to {cmd}'])
    except Exception as exc:  # noqa: BLE001
        return f"Couldn't control {target}: {exc}"
    if r.returncode != 0:
        return (f"Couldn't control {target}. Is it installed and open? "
                f"({r.stderr.strip()})")
    verb = {"play": "Playing", "pause": "Paused", "playpause": "Toggled playback",
            "next track": "Skipped to the next track",
            "previous track": "Went to the previous track"}.get(cmd, cmd)
    log.info("Music: %s on %s", cmd, target)
    return f"{verb} on {target}."


def play_song(query: str, app: str = "Spotify") -> str:
    """Search for the song and start playing it in Spotify (or Apple Music).

    Resolves the query to an exact track (via tools/spotify.py), then tells
    the Spotify app to play that URI. If the track can't be resolved, falls
    back to opening the in-app search so the user can tap the top result.
    """
    if not IS_MAC:
        return _NOT_MAC
    query = query.strip()
    if app.lower() != "spotify":
        _run(["open", f"https://music.apple.com/search?term={query.replace(' ', '+')}"])
        return f"Opened Apple Music search for “{query}”."

    from tools import spotify
    track = spotify.search_track(query)
    if track:
        script = ('tell application "Spotify"\n'
                  'activate\n'
                  f'play track "{track["uri"]}"\n'
                  'end tell')
        try:
            r = _run(["osascript", "-e", script], timeout=25)
            if r.returncode == 0:
                log.info("Playing %s — %s", track["name"], track["artist"])
                by = f" by {track['artist']}" if track["artist"] else ""
                return f"Now playing “{track['name']}”{by} on Spotify."
            log.warning("osascript play failed: %s", r.stderr.strip())
        except Exception as exc:  # noqa: BLE001 - fall through to search
            log.warning("Spotify playback failed: %s", exc)

    _run(["open", f"spotify:search:{query}"])
    return (f"I couldn't auto-play “{query}”, so I opened the Spotify search "
            "for it — tap the top result.")

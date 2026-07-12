"""Spotify track search for JARVIS.

Resolves "a song the user named" → a concrete Spotify track URI that
desktop/apps.py can hand to the Mac's Spotify app for instant playback.

Two strategies, tried in order:
  1. Official Web API (Client Credentials flow — no user login). Enabled by
     setting SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET in .env; create a free
     app at https://developer.spotify.com/dashboard to get them. Fast (~0.5s).
  2. Keyless fallback: render open.spotify.com's search in the same headless
     Chromium used by browser/web.py and grab the top track result. Slower
     (~5s) and depends on Spotify's markup, but needs zero setup.

Search only — nothing here plays audio or touches the user's account.
"""

from __future__ import annotations

import time
from urllib.parse import quote

from config.settings import settings
from core.deps import ensure_package
from utils.logger import get_logger

log = get_logger(__name__)

SEARCH_TIMEOUT = 15_000   # ms to wait for results in the scrape fallback

# Client-credentials token cache (module-level; tokens last ~1h).
_token: str = ""
_token_expiry: float = 0.0


def _api_token() -> str | None:
    """Return a cached app token, or None when credentials aren't configured."""
    global _token, _token_expiry
    if not settings.spotify_configured:
        return None
    if _token and time.time() < _token_expiry - 60:
        return _token
    requests = ensure_package("requests")
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(settings.spotify_client_id, settings.spotify_client_secret),
        timeout=10,
    )
    r.raise_for_status()
    j = r.json()
    _token = j["access_token"]
    _token_expiry = time.time() + int(j.get("expires_in", 3600))
    return _token


def _api_search(query: str) -> dict | None:
    token = _api_token()
    if token is None:
        return None
    requests = ensure_package("requests")
    r = requests.get(
        "https://api.spotify.com/v1/search",
        params={"q": query, "type": "track", "limit": 1},
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    r.raise_for_status()
    items = r.json().get("tracks", {}).get("items", [])
    if not items:
        return None
    t = items[0]
    return {
        "uri": t["uri"],
        "name": t["name"],
        "artist": ", ".join(a["name"] for a in t["artists"]),
    }


def _scrape_search(query: str) -> dict | None:
    """Keyless fallback: read the top result off open.spotify.com's search page."""
    from browser.web import _Session

    url = f"https://open.spotify.com/search/{quote(query)}/tracks"
    with _Session() as page:
        page.goto(url, wait_until="domcontentloaded")
        link = page.wait_for_selector('a[href^="/track/"]', timeout=SEARCH_TIMEOUT)
        href = link.get_attribute("href") or ""
        row_text = link.evaluate(
            "el => (el.closest('[role=\"row\"]') || el).innerText"
        ) or ""
    track_id = href.split("/track/")[-1].split("?")[0]
    if not track_id:
        return None
    # Row text looks like: "1\nSong Name\nE\nArtist" — skip row numbers/badges.
    lines = [ln.strip() for ln in row_text.split("\n")
             if ln.strip() and not ln.strip().isdigit() and len(ln.strip()) > 1]
    return {
        "uri": f"spotify:track:{track_id}",
        "name": lines[0] if lines else query,
        "artist": lines[1] if len(lines) > 1 else "",
    }


def search_track(query: str) -> dict | None:
    """Top matching track as {'uri', 'name', 'artist'}, or None if not found."""
    query = query.strip()
    if not query:
        return None
    try:
        track = _api_search(query)
        if track:
            log.info("spotify[api] %r -> %s", query, track["uri"])
            return track
    except Exception as exc:  # noqa: BLE001 - fall through to the scrape
        log.warning("Spotify API search failed (%s); trying scrape.", exc)
    try:
        track = _scrape_search(query)
        if track:
            log.info("spotify[scrape] %r -> %s", query, track["uri"])
        return track
    except Exception as exc:  # noqa: BLE001 - caller falls back to opening search
        log.warning("Spotify scrape search failed: %s", exc)
        return None

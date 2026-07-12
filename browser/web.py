"""Headless browser automation for JARVIS (Playwright + Chromium).

Phase 2 exposes read-only web capabilities — the safe, high-value set:
  - `fetch_page(url)`  : load a page and return its title + visible text
  - `web_search(query)`: DuckDuckGo search, returns top results
  - `screenshot(url)`  : save a PNG of a page into screenshots/

These run against a fresh headless Chromium per call (simple + thread-safe;
no shared state to leak between requests). Works locally and on Railway.

Deliberately NOT exposed for autonomous calling: form filling, clicking,
logins, downloads — those take actions on the user's behalf and belong behind
explicit confirmation, which we'll add when needed.
"""

from __future__ import annotations

from urllib.parse import quote, quote_plus

from config.settings import settings
from core.deps import ensure_package
from utils.logger import get_logger

log = get_logger(__name__)

NAV_TIMEOUT = 30_000          # ms
MAX_TEXT = 6_000              # chars of page text returned to the model
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def _sync_playwright():
    pw = ensure_package("playwright")
    if pw is None:
        raise RuntimeError(
            "playwright not installed. Run: pip install playwright && "
            "playwright install chromium"
        )
    from playwright.sync_api import sync_playwright
    return sync_playwright


class _Session:
    """Context manager yielding a ready page on a fresh headless browser."""

    def __enter__(self):
        self._pw = _sync_playwright()().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._ctx = self._browser.new_context(user_agent=UA)
        self.page = self._ctx.new_page()
        self.page.set_default_navigation_timeout(NAV_TIMEOUT)
        return self.page

    def __exit__(self, *exc):
        try:
            self._browser.close()
        finally:
            self._pw.stop()
        return False


def fetch_page(url: str) -> dict:
    """Load `url`, return {'url', 'title', 'text'} with visible text truncated."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    with _Session() as page:
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(500)
        title = page.title()
        try:
            text = page.inner_text("body")
        except Exception:  # noqa: BLE001
            text = ""
    text = " ".join(text.split())
    log.info("fetch_page %s -> %d chars", url, len(text))
    return {"url": url, "title": title, "text": text[:MAX_TEXT]}


def web_search(query: str, limit: int = 5) -> list[dict]:
    """Search the web, return [{title, url, snippet}].

    Uses the Brave Search API when BRAVE_API_KEY is set (real ranked web
    results); otherwise falls back to DuckDuckGo's keyless Instant Answer API
    (reliable, no CAPTCHA, best for factual/entity queries). Note: the fallback
    is lighter than full web ranking — set BRAVE_API_KEY for product-grade
    results. Get a free key at https://brave.com/search/api/.
    """
    import os
    requests = ensure_package("requests")
    ua = {"User-Agent": UA}

    brave_key = os.environ.get("BRAVE_API_KEY", "")
    if brave_key:
        try:
            r = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": limit},
                headers={"X-Subscription-Token": brave_key, "Accept": "application/json"},
                timeout=20,
            )
            r.raise_for_status()
            hits = (r.json().get("web") or {}).get("results") or []
            out = [{"title": h.get("title", ""), "url": h.get("url", ""),
                    "snippet": h.get("description", "")} for h in hits[:limit]]
            log.info("web_search[brave] %r -> %d", query, len(out))
            return out
        except Exception as exc:  # noqa: BLE001 - fall through to DDG
            log.warning("Brave search failed (%s); using DDG fallback.", exc)

    # Keyless fallback: DuckDuckGo Instant Answer JSON API
    j = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        headers=ua, timeout=20,
    ).json()

    results: list[dict] = []
    if j.get("AbstractText"):
        results.append({
            "title": j.get("Heading") or query,
            "url": j.get("AbstractURL", ""),
            "snippet": j["AbstractText"],
        })

    def _walk(topics):
        for t in topics:
            if "Topics" in t:
                _walk(t["Topics"])
            elif t.get("Text"):
                results.append({
                    "title": t["Text"].split(" - ")[0][:80],
                    "url": t.get("FirstURL", ""),
                    "snippet": t["Text"],
                })
            if len(results) >= limit:
                return
    _walk(j.get("RelatedTopics") or [])
    log.info("web_search[ddg] %r -> %d", query, len(results))
    return results[:limit]


def get_weather(location: str) -> str:
    """Current weather for a place, via wttr.in (keyless, reliable plain text)."""
    requests = ensure_package("requests")
    fmt = "%l: %C, %t (feels %f), humidity %h, wind %w"
    url = f"https://wttr.in/{quote(location)}?format={quote(fmt)}"
    # wttr.in returns plain text only to curl-like agents (browsers get HTML).
    r = requests.get(url, headers={"User-Agent": "curl/8.0"}, timeout=15)
    r.raise_for_status()
    text = r.text.strip()
    log.info("get_weather %r -> %s", location, text[:80])
    return text or f"(no weather found for {location!r})"


def get_news(topic: str = "", limit: int = 8) -> list[dict]:
    """Latest headlines via Google News RSS (keyless, no CAPTCHA).

    topic="" returns top world stories; otherwise searches that topic.
    Returns [{title, url}].
    """
    import xml.etree.ElementTree as ET

    requests = ensure_package("requests")
    if topic.strip():
        url = ("https://news.google.com/rss/search?q=" + quote_plus(topic)
               + "&hl=en-US&gl=US&ceid=US:en")
    else:
        url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"

    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out: list[dict] = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if title:
            out.append({"title": title, "url": link})
    log.info("get_news %r -> %d headlines", topic, len(out))
    return out


def screenshot(url: str, name: str = "page.png") -> str:
    """Save a full-page PNG of `url` into screenshots/ and return the path."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    out_dir = settings.root / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".png"):
        name += ".png"
    path = out_dir / name
    with _Session() as page:
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(600)
        page.screenshot(path=str(path), full_page=True)
    log.info("screenshot %s -> %s", url, path)
    return str(path)

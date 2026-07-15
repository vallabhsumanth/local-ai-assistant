"""Similarity response cache — fast answers for repeated / rephrased questions.

Before JARVIS calls the (slow) LLM, the assistant checks this cache. If the
current question is similar enough to one answered recently, the stored answer
is returned instantly — no model call.

Design choices:
- **No embedding model.** Similarity is computed with lightweight lexical
  measures (word-set Jaccard + difflib ratio) — free, instant, good enough for
  "same question, different words".
- **TTL + LRU.** Entries expire (default 6h) and the cache is size-bounded, so
  it never serves ancient answers or grows unbounded.
- **Skips volatile / personal questions.** Weather, news, "now/today", prices,
  and memory questions ("what's my name") are never cached — those must stay
  live, otherwise we'd serve stale info.

It's in-process (fast, per running server). Derived data, so it doesn't need to
live in Supabase — restart just starts with a cold cache.
"""

from __future__ import annotations

import re
import time
from difflib import SequenceMatcher

from utils.logger import get_logger

log = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9]+")

# Filler/question words stripped before comparing, so "what is X" and
# "tell me about X" match on the meaningful word (X).
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "to", "in", "on",
    "for", "and", "or", "what", "whats", "which", "who", "whom", "how", "why",
    "when", "where", "do", "does", "did", "can", "could", "would", "will",
    "you", "i", "me", "my", "please", "tell", "give", "show", "about", "explain",
    "know", "want", "need", "get", "some", "any", "there", "it", "that", "this",
}

# Questions containing these are never cached (answers change over time).
_VOLATILE = (
    "weather", "temperature", "temp", "forecast", "news", "headline",
    "today", "tonight", "now", "current", "currently", "latest", "recent",
    "time", "date", "price", "stock", "score", "live",
)
# Personal / memory questions — answers depend on mutable stored facts.
_PERSONAL = (
    "remember", "recall", "forget", "my name", "who am i",
    "what is my", "what's my", "whats my", "do i have",
)
# Brand/strategy/competitor questions — must give a fresh angle every time,
# never a canned repeat (see BRAND_CONTEXT in core/assistant.py).
_STRATEGY = (
    "strategy", "strategies", "compete", "competitor", "competitors",
    "competition", "beat", "scale", "scaling", "positioning", "insight",
    "insights", "growth", "market share", "nauti nati", "hopscotch",
    "klinkara", "nap chief", "brand",
)


def _words(text: str) -> list[str]:
    """Meaningful (content) words: lowercased tokens minus stopwords.

    Falls back to all tokens if stripping stopwords leaves nothing.
    """
    tokens = _WORD.findall(text.lower())
    content = [t for t in tokens if t not in _STOPWORDS]
    return content or tokens


class ResponseCache:
    def __init__(self, ttl: float = 6 * 3600, max_size: int = 300,
                 threshold: float = 0.84) -> None:
        self.ttl = ttl
        self.max_size = max_size
        self.threshold = threshold
        # each entry: {"wordset": set, "norm": str, "answer": str, "ts": float}
        self._entries: list[dict] = []

    def cacheable(self, question: str) -> bool:
        q = question.strip().lower()
        if len(q) < 6:
            return False
        if q.startswith("/"):          # slash-commands are handled elsewhere
            return False
        if any(k in q for k in _VOLATILE):
            return False
        if any(k in q for k in _PERSONAL):
            return False
        if any(k in q for k in _STRATEGY):
            return False
        return True

    def get(self, question: str, now: float | None = None) -> str | None:
        """Return a cached answer for a similar recent question, or None."""
        if not self.cacheable(question):
            return None
        now = time.time() if now is None else now
        words = _words(question)
        wset = set(words)
        if not wset:
            return None
        norm = " ".join(words)

        # Drop expired entries lazily.
        self._entries = [e for e in self._entries if now - e["ts"] <= self.ttl]

        best, best_score = None, 0.0
        for e in self._entries:
            inter = len(wset & e["wordset"])
            union = len(wset | e["wordset"]) or 1
            jaccard = inter / union
            seq = SequenceMatcher(None, norm, e["norm"]).ratio()
            score = max(jaccard, seq)
            if score > best_score:
                best, best_score = e, score

        if best is not None and best_score >= self.threshold:
            log.info("cache HIT (score=%.2f) for %r", best_score, question)
            return best["answer"]
        return None

    def put(self, question: str, answer: str, now: float | None = None) -> None:
        if not self.cacheable(question) or not answer:
            return
        now = time.time() if now is None else now
        words = _words(question)
        self._entries.append({
            "wordset": set(words),
            "norm": " ".join(words),
            "answer": answer,
            "ts": now,
        })
        # LRU-ish bound: drop oldest.
        if len(self._entries) > self.max_size:
            self._entries.pop(0)

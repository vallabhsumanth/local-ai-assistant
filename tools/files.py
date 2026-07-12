"""File-management tools for JARVIS (Phase 1, macOS).

Pure-Python operations with a safety gate on destructive actions. Every
function that can lose data (`delete`, `move`) routes through `confirm()` when
`settings.confirm_destructive` is on.

Paths are expanded (`~`) and resolved so relative input works from anywhere.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Callable

from config.settings import settings
from utils.logger import get_logger

log = get_logger(__name__)

# Injected by the app so we can prompt in the REPL; defaults to auto-deny.
ConfirmFn = Callable[[str], bool]
_confirm: ConfirmFn = lambda _msg: False


def set_confirm_handler(fn: ConfirmFn) -> None:
    global _confirm
    _confirm = fn


def _p(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _guard(action: str) -> bool:
    if not settings.confirm_destructive:
        return True
    return _confirm(action)


def read_text(path: str | Path, max_bytes: int = 200_000) -> str:
    p = _p(path)
    data = p.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def write_text(path: str | Path, content: str) -> Path:
    p = _p(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    log.info("Wrote %d chars to %s", len(content), p)
    return p


def list_dir(path: str | Path = ".") -> list[str]:
    p = _p(path)
    return sorted(
        f"{'d' if c.is_dir() else 'f'}  {c.name}" for c in p.iterdir()
    )


def search(root: str | Path, pattern: str, limit: int = 200) -> list[str]:
    """Recursive glob search, e.g. pattern='*.pdf'."""
    base = _p(root)
    hits: list[str] = []
    for match in base.rglob(pattern):
        hits.append(str(match))
        if len(hits) >= limit:
            break
    return hits


def copy(src: str | Path, dst: str | Path) -> Path:
    s, d = _p(src), _p(dst)
    if s.is_dir():
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    log.info("Copied %s -> %s", s, d)
    return d


def move(src: str | Path, dst: str | Path) -> Path | None:
    s, d = _p(src), _p(dst)
    if not _guard(f"Move {s} -> {d}?"):
        log.info("Move cancelled by user.")
        return None
    shutil.move(str(s), str(d))
    log.info("Moved %s -> %s", s, d)
    return d


def rename(path: str | Path, new_name: str) -> Path:
    p = _p(path)
    target = p.with_name(new_name)
    p.rename(target)
    log.info("Renamed %s -> %s", p, target)
    return target


def delete(path: str | Path) -> bool:
    p = _p(path)
    if not _guard(f"DELETE {p}? This cannot be undone."):
        log.info("Delete cancelled by user.")
        return False
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    log.info("Deleted %s", p)
    return True


def zip_dir(src: str | Path, archive: str | Path) -> Path:
    s, a = _p(src), _p(archive)
    with zipfile.ZipFile(a, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in s.rglob("*"):
            zf.write(item, item.relative_to(s.parent))
    log.info("Zipped %s -> %s", s, a)
    return a


def unzip(archive: str | Path, dest: str | Path) -> Path:
    a, d = _p(archive), _p(dest)
    d.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a) as zf:
        zf.extractall(d)
    log.info("Extracted %s -> %s", a, d)
    return d

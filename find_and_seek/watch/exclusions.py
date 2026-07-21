"""User-defined path exclusions, persisted to ~/.findandseek/exclusions.json.

A watched root (e.g. ~/Documents) is a coarse grant — it often contains large
subtrees the user does NOT want searchable: a code workspace, a scratch/test
corpus, an archive of someone else's files. Roots grant access at the folder
level; exclusions carve specific path prefixes back out, so search stays about
the user's actual documents rather than developer noise.

Honored by both the startup scan (`watch/scan.py`) and the live watcher
(`watch/watcher.py`). Companion to `watch/roots.py` — same data-dir, same shape.
Reversible: removing an exclusion and re-scanning re-indexes the subtree; the
files on disk are never touched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

EXCLUSIONS_PATH = Path(os.environ.get("FINDANDSEEK_EXCLUSIONS_PATH", Path.home() / ".findandseek" / "exclusions.json"))


def excluded_prefixes() -> list[str]:
    """Absolute path prefixes the user has chosen to skip. Empty if none set."""
    if not EXCLUSIONS_PATH.exists():
        return []
    try:
        data = json.loads(EXCLUSIONS_PATH.read_text(encoding="utf-8"))
        return [str(Path(p).expanduser()).rstrip("/") for p in data.get("excluded_prefixes", [])]
    except Exception:
        return []


def is_user_excluded(path: str, prefixes: list[str] | None = None) -> bool:
    """True if ``path`` is at or under any excluded prefix. Pass ``prefixes`` to
    avoid re-reading the file in a hot loop (e.g. a full-tree scan)."""
    pres = excluded_prefixes() if prefixes is None else prefixes
    p = str(path)
    return any(p == pre or p.startswith(pre + "/") for pre in pres)


def _save(prefixes: list[str]) -> None:
    EXCLUSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    out: list[str] = []
    for x in prefixes:
        s = str(Path(x).expanduser()).rstrip("/")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    EXCLUSIONS_PATH.write_text(json.dumps({"excluded_prefixes": out}, indent=2), encoding="utf-8")


def add_exclusion(path: str | Path) -> list[str]:
    prefixes = excluded_prefixes()
    s = str(Path(path).expanduser()).rstrip("/")
    if s not in prefixes:
        prefixes.append(s)
    _save(prefixes)
    return prefixes


def remove_exclusion(path: str | Path) -> list[str]:
    s = str(Path(path).expanduser()).rstrip("/")
    prefixes = [x for x in excluded_prefixes() if x != s]
    _save(prefixes)
    return prefixes

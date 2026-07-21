"""Watched-folder configuration, persisted to ~/.findandseek/roots.json.

The user grants folders explicitly (via the API/app); the engine only ever reads
what's listed here. There is **no implicit Full-Disk-style default scope** — an
absent or empty config means the engine indexes nothing until a folder is added.
This is the per-folder-permissions model (see MAC_APP_STORE_AND_PERMISSIONS.md):
least privilege, bounded ingest scope, and a clean path to the macOS sandbox.
The watcher reads this and picks up changes live.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

ROOTS_PATH = Path(os.environ.get("FINDANDSEEK_ROOTS_PATH", Path.home() / ".findandseek" / "roots.json"))


def suggested_roots() -> list[Path]:
    """Folders to *offer* during onboarding — never auto-granted. The user still
    adds each one explicitly. Just the common document homes that exist."""
    # Lazy import avoids a circular import with watcher.py.
    from find_and_seek.watch.watcher import DEFAULT_ROOTS

    return [r for r in DEFAULT_ROOTS if r.exists()]


def list_roots() -> list[Path]:
    """The folders the user has explicitly granted. Empty until one is added."""
    if not ROOTS_PATH.exists():
        return []
    try:
        data = json.loads(ROOTS_PATH.read_text(encoding="utf-8"))
        return [Path(p).expanduser() for p in data.get("roots", [])]
    except Exception:
        return []


def _save(roots: list[Path]) -> None:
    ROOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    out: list[str] = []
    for r in roots:
        s = str(Path(r).expanduser())
        if s not in seen:
            seen.add(s)
            out.append(s)
    ROOTS_PATH.write_text(json.dumps({"roots": out}, indent=2), encoding="utf-8")


def add_root(path: str | Path) -> list[Path]:
    p = Path(path).expanduser()
    roots = list_roots()
    if p not in roots:
        roots.append(p)
    _save(roots)
    return roots


def remove_root(path: str | Path) -> list[Path]:
    p = Path(path).expanduser()
    roots = [r for r in list_roots() if r != p]
    _save(roots)
    return roots


# Standard home folders we *prefer* to surface as a seed root when they actually
# contain indexed files — these match the onboarding suggestions (watcher.DEFAULT_ROOTS).
_PREFERRED_HOME_DIRS = ("Documents", "Desktop", "Downloads")


def seed_roots_from_index(conn: sqlite3.Connection) -> list[Path]:
    """One-time upgrade migration: reconstruct granted roots from the existing index.

    The app moved from Full-Disk-Access to explicit per-folder grants (D32). A user
    who upgrades has an indexed corpus but NO roots.json, so list_roots() returns []
    and onboarding gates on "no granted folders" — silently resetting their whole
    scope to nothing. To avoid that, on first boot after upgrade we derive a small
    set of sensible top-level roots from what's already indexed and persist them.

    Idempotency is the file itself: if ROOTS_PATH already exists we no-op, so this
    never overrides a user's real choices and never re-runs. A fresh install has an
    empty index, so nothing is written (new users onboard explicitly) — the "empty
    means empty" property is preserved for the normal case.

    Returns the roots written (possibly empty).
    """
    # roots.json already exists → never override the user's real choices, never re-run.
    if ROOTS_PATH.exists():
        return list_roots()

    try:
        rows = conn.execute(
            "SELECT DISTINCT path FROM files WHERE status='indexed'"
        ).fetchall()
    except sqlite3.DatabaseError:
        # A corrupt/unusable index must not derail boot — leave roots absent.
        return []

    paths = [Path(r[0]) for r in rows if r and r[0]]
    if not paths:
        # Empty index (fresh install) → write nothing; onboarding stays explicit.
        return []

    roots = _derive_roots(paths)
    if roots:
        _save(roots)
    return roots


def _derive_roots(paths: list[Path]) -> list[Path]:
    """Reduce a flat list of indexed file paths to a SHORT set of ancestor roots.

    Each indexed file collapses to its immediate top-level directory under the
    user's home (e.g. anything in ~/Documents/** → ~/Documents) — deduped, so we
    never emit one root per file. The standard home folders (~/Documents,
    ~/Desktop, ~/Downloads) fall out of this naturally and are surfaced FIRST so
    the seeded list matches the onboarding suggestions. Files indexed outside the
    user's home collapse to their own parent directory.
    """
    home = Path.home()

    seen: set[Path] = set()
    roots: list[Path] = []

    def _add(root: Path) -> None:
        if root not in seen:
            seen.add(root)
            roots.append(root)

    for p in paths:
        try:
            rel = p.relative_to(home)
        except ValueError:
            # Indexed outside the user's home — use its own top-level directory.
            if p.parent != p:
                _add(p.parent)
            continue
        if not rel.parts:
            continue
        _add(home / rel.parts[0])

    # Surface the standard home folders first (stable, matches onboarding offer).
    preferred = [home / d for d in _PREFERRED_HOME_DIRS]
    ordered = [r for r in preferred if r in seen]
    ordered += [r for r in roots if r not in set(ordered)]
    return ordered

"""Opt-in macOS Finder-tag sync (ORGANIZE_FEATURE_DESIGN.md §7.2).

Makes a chosen subset of FindandSeek's native tags visible in **Finder & Spotlight**
by writing them to each file's ``com.apple.metadata:_kMDItemUserTags`` extended
attribute (a binary-plist array of ``"Name"`` / ``"Name\\n<coloridx>"`` strings).

Like every Organize write it is **reversible and journalled**: it runs as an
``add_finder_tag`` plan action through the existing apply/undo machinery — apply
writes the merged tag set (preserving the user's own Finder tags) and journals the
*exact prior* set; undo restores it byte-for-byte. Off by default (see settings).
"""

from __future__ import annotations

import ctypes
import os
import plistlib
import sys
from pathlib import Path
from typing import Any
import sqlite3

from find_and_seek.organize.naming import _title
from find_and_seek.organize.tags import get_file_tags

_XATTR = "com.apple.metadata:_kMDItemUserTags"

# ── low-level xattr (macOS via libc; Linux via os.*) ─────────────────
# CPython only exposes os.getxattr/setxattr on Linux. macOS needs libc, whose
# get/set/removexattr take extra (position, options) args vs Linux.
_ENOATTR_DARWIN = 93  # macOS ENOATTR


def _darwin_libc() -> ctypes.CDLL:
    libc = ctypes.CDLL("libc.dylib", use_errno=True)
    cp, vp, sz, u32, ci = (ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t,
                           ctypes.c_uint32, ctypes.c_int)
    libc.getxattr.restype = ctypes.c_ssize_t
    libc.getxattr.argtypes = [cp, cp, vp, sz, u32, ci]
    libc.setxattr.restype = ci
    libc.setxattr.argtypes = [cp, cp, vp, sz, u32, ci]
    libc.removexattr.restype = ci
    libc.removexattr.argtypes = [cp, cp, ci]
    return libc


_LIBC = _darwin_libc() if sys.platform == "darwin" else None


def xattr_supported() -> bool:
    return sys.platform == "darwin" and (_LIBC is not None or hasattr(os, "getxattr"))


def _xattr_get(path: str) -> bytes:
    if _LIBC is not None:
        pb, nb = path.encode(), _XATTR.encode()
        size = _LIBC.getxattr(pb, nb, None, 0, 0, 0)
        if size < 0:
            return b""  # ENOATTR / missing
        buf = ctypes.create_string_buffer(size)
        n = _LIBC.getxattr(pb, nb, buf, size, 0, 0)
        return buf.raw[:n] if n >= 0 else b""
    try:
        return os.getxattr(path, _XATTR)  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        return b""


def _xattr_set(path: str, data: bytes) -> None:
    if _LIBC is not None:
        if _LIBC.setxattr(path.encode(), _XATTR.encode(), data, len(data), 0, 0) < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), path)
        return
    os.setxattr(path, _XATTR, data, 0)  # type: ignore[attr-defined]


def _xattr_remove(path: str) -> None:
    if _LIBC is not None:
        _LIBC.removexattr(path.encode(), _XATTR.encode(), 0)  # ignore ENOATTR
        return
    try:
        os.removexattr(path, _XATTR)  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        pass

# Finder colour index per tag kind:
# 0 none, 1 gray, 2 green, 3 purple, 4 blue, 5 yellow, 6 red, 7 orange.
_KIND_COLOR = {"type": 4, "year": 1, "party": 3, "status": 5, "project": 2}

DEFAULT_KINDS = ("type", "year", "party", "status")


# ── display names ────────────────────────────────────────────────────
def display_name(tag_name: str) -> str:
    """``"party:acme-corp"`` → ``"Acme Corp"`` (strip kind prefix, title-case)."""
    slug = tag_name.split(":", 1)[1] if ":" in tag_name else tag_name
    return _title(slug)


def _strip(entry: str) -> str:
    """Drop the trailing ``\\n<coloridx>`` from a raw Finder-tag entry."""
    return entry.split("\n", 1)[0]


# ── xattr read/write ─────────────────────────────────────────────────
def read_raw(path: str | Path) -> list[str]:
    """Current raw Finder-tag entries (with colour suffixes), or ``[]``."""
    raw = _xattr_get(os.fspath(path))
    if not raw:
        return []
    try:
        items = plistlib.loads(raw)
    except Exception:  # noqa: BLE001 — corrupt/foreign value → treat as none
        return []
    return [str(i) for i in items]


def read_finder_tags(path: str | Path) -> list[str]:
    """Current Finder-tag display names (colour stripped)."""
    return [_strip(e) for e in read_raw(path)]


def write_raw(path: str | Path, entries: list[str]) -> None:
    """Write raw entries verbatim (already ``"Name"`` / ``"Name\\n<idx>"``).

    An empty list removes the attribute entirely so the file shows no tags.
    """
    p = os.fspath(path)
    if not entries:
        _xattr_remove(p)
        return
    data = plistlib.dumps(entries, fmt=plistlib.FMT_BINARY)
    _xattr_set(p, data)


# ── sync computation ─────────────────────────────────────────────────
def _managed_universe(conn: sqlite3.Connection, kinds: tuple[str, ...]) -> set[str]:
    """Every display name FindandSeek *could* manage for these kinds — so we can
    drop a stale managed tag without touching the user's own Finder tags."""
    if not kinds:
        return set()
    ph = ",".join("?" for _ in kinds)
    rows = conn.execute(f"SELECT DISTINCT name FROM tags WHERE kind IN ({ph})", list(kinds)).fetchall()
    return {display_name(r[0]) for r in rows}


def findandseek_entries_for_file(
    conn: sqlite3.Connection, file_id: int, kinds: tuple[str, ...]
) -> list[str]:
    """The raw Finder entries (name + colour) FindandSeek wants present on a file."""
    out: list[str] = []
    for t in get_file_tags(conn, file_id):
        if t["kind"] in kinds:
            name = display_name(t["name"])
            color = _KIND_COLOR.get(t["kind"])
            out.append(f"{name}\n{color}" if color is not None else name)
    return out


def compute_sync(
    conn: sqlite3.Connection, file_id: int, path: str | Path, kinds: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """Return ``(before_raw, after_raw)``: preserve the user's own Finder tags,
    drop stale FindandSeek-managed ones, add the file's current FindandSeek tags."""
    before_raw = read_raw(path)
    universe = _managed_universe(conn, kinds)
    preserved = [e for e in before_raw if _strip(e) not in universe]
    preserved_names = {_strip(e) for e in preserved}
    additions = [e for e in findandseek_entries_for_file(conn, file_id, kinds)
                 if _strip(e) not in preserved_names]
    return before_raw, preserved + additions


# ── plan builder (opt-in backfill as a reversible plan) ──────────────
def build_finder_sync_plan(
    conn: sqlite3.Connection,
    roots: list[str] | None = None,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
) -> int:
    """Create an accepted plan of ``add_finder_tag`` actions for indexed files in
    scope. Apply it to write Finder tags; undo it to remove the whole sync."""
    from find_and_seek.organize import plan_store

    sql = "SELECT id, path FROM files WHERE status='indexed'"
    params: list[Any] = []
    if roots:
        sql += " AND (" + " OR ".join("path LIKE ?" for _ in roots) + ")"
        params += [f"{r}%" for r in roots]
    files = conn.execute(sql, params).fetchall()

    actions = [
        {
            "action_type": "add_finder_tag",
            "file_id": f["id"],
            "payload": {"from_path": f["path"], "kinds": list(kinds),
                        "reason": "sync FindandSeek tags to Finder"},
        }
        for f in files
    ]
    plan_id = plan_store.create_plan(
        conn, strategy="finder_sync", scope=roots,
        summary={"finder_sync": len(actions), "kinds": list(kinds)},
    )
    plan_store.add_actions(conn, plan_id, actions)
    plan_store.set_decision(conn, plan_id, "accepted")  # explicit opt-in
    conn.commit()
    return plan_id

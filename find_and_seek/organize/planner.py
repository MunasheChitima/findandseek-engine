"""The proposal engine: turn the catalog into a reorganization Plan (design §8).

Layered, cheap-to-expensive, safe-rules-first (per the locked decision):
1. **Rules** — canonical type + year → `…/<TypeFolder>/<Year>/` (the bulk).
2. **Entity grouping** — strongest org becomes the {Party} in the canonical name.
3. **Duplicate detection** — exact `content_hash` groups; keep the best-located
   copy, flag the rest for quarantine (never hard-deleted).
4. Semantic embedding clusters are returned as *optional suggestions* only — they
   never become moves in v1.

Pure read + Plan rows: this never writes to the filesystem. Every action carries
a human ``reason`` so the preview is explainable, not magic.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import PurePosixPath
from typing import Any

from find_and_seek.organize import plan_store
from find_and_seek.organize.naming import canonical_filename, destination_folder, year_for
from find_and_seek.organize.tags import _party_slugs, derive_tags
from find_and_seek.organize.taxonomy import canonicalize_type
from find_and_seek.watch.roots import list_roots

# Folders we treat as "junk drawers" — files here are prime relocation targets.
JUNK_NAMES = {
    "misc", "miscellaneous", "untitled", "new folder", "new folder with items",
    "downloads", "desktop", "stuff", "temp", "tmp", "unsorted", "to sort",
}

# Never propose touching these — opaque packages and system trees.
EXCLUDE_SUBSTRINGS = ("/Library/", "/System/", "/.Trash/", "/node_modules/")
PACKAGE_EXTS = (".app", ".rtfd", ".bundle", ".framework", ".photoslibrary")


def _excluded(path: str) -> bool:
    if any(s in path for s in EXCLUDE_SUBSTRINGS):
        return True
    low = path.lower()
    return any(f".{e.lstrip('.')}/" in low or low.endswith(e) for e in PACKAGE_EXTS)


def _root_for(path: str, roots: list[str]) -> str:
    """The watched root that contains `path` (longest matching prefix)."""
    best = ""
    for r in roots:
        rr = r.rstrip("/")
        if (path == rr or path.startswith(rr + "/")) and len(rr) > len(best):
            best = rr
    return best or str(PurePosixPath(path).parent)


def _in_junk_drawer(path: str) -> bool:
    parent = PurePosixPath(path).parent
    return any(part.lower() in JUNK_NAMES for part in parent.parts)


def _locatedness(path: str) -> tuple[int, int, str]:
    """Sort key: lower is *better located* (kept on dedup)."""
    penalty = 0
    low = path.lower()
    if _in_junk_drawer(path):
        penalty += 3
    if "/downloads/" in low:
        penalty += 2
    if "/desktop/" in low:
        penalty += 1
    depth = len(PurePosixPath(path).parts)
    return (penalty, depth, path)


def _scope_clause(roots: list[str]) -> tuple[str, list[str]]:
    if not roots:
        return "0", []
    clauses = " OR ".join("f.path LIKE ?" for _ in roots)
    params = [f"{r.rstrip('/')}/%" for r in roots]
    return f"({clauses})", params


def generate_plan(
    conn: sqlite3.Connection,
    roots: list[str] | None = None,
    strategy: str = "by_type",
) -> dict[str, Any]:
    """Build, persist and return a draft Plan for the given roots.

    `roots=None` → every watched root (the default "whatever is indexed" scope).
    """
    if not roots:
        roots = [str(r) for r in list_roots()]
    roots = [str(r) for r in roots]

    where, params = _scope_clause(roots)
    rows = conn.execute(
        f"""
        SELECT f.id, f.path, f.filename, f.content_hash, f.modified_at,
               fs.document_type, fs.key_facts, fs.suggested_filename
        FROM files f
        LEFT JOIN file_summaries fs ON fs.file_id = f.id
        WHERE f.status='indexed' AND {where}
        ORDER BY f.path
        """,
        params,
    ).fetchall()

    files = [r for r in rows if not _excluded(r["path"])]

    # ── duplicate groups (exact content_hash) ───────────────────────
    by_hash: dict[str, list[sqlite3.Row]] = {}
    for r in files:
        by_hash.setdefault(r["content_hash"], []).append(r)
    quarantine_ids: dict[int, int] = {}  # dup file_id -> kept file_id
    for h, group in by_hash.items():
        if len(group) < 2:
            continue
        keeper = min(group, key=lambda r: _locatedness(r["path"]))
        for r in group:
            if r["id"] != keeper["id"]:
                quarantine_ids[r["id"]] = keeper["id"]

    actions: list[dict[str, Any]] = []
    needed_dirs: set[str] = set()
    counts = {"move": 0, "rename": 0, "create_dir": 0, "quarantine_duplicate": 0, "tag": 0}

    for r in files:
        fid = r["id"]
        from_path = r["path"]

        # Duplicate → quarantine action, no move/rename for this copy.
        if fid in quarantine_ids:
            actions.append({
                "action_type": "quarantine_duplicate",
                "file_id": fid,
                "payload": {
                    "from_path": from_path,
                    "dup_of_file_id": quarantine_ids[fid],
                    "reason": f"exact duplicate (same content_hash) of file {quarantine_ids[fid]}; "
                              f"keeping the better-located copy",
                },
            })
            counts["quarantine_duplicate"] += 1
            continue

        root = _root_for(from_path, roots)
        ct = canonicalize_type(r["document_type"])
        key_facts_raw = r["key_facts"]
        try:
            kf = json.loads(key_facts_raw) if key_facts_raw else {}
        except (json.JSONDecodeError, TypeError):
            kf = {}
        year = year_for(kf, r["modified_at"])

        party_slugs = _party_slugs(conn, fid)
        party = party_slugs[0] if party_slugs else None

        dest_folder = destination_folder(root, ct, year)
        # Don't relocate files we can't confidently classify. Dumping them into an
        # "Other/<year>" bucket isn't organising — it's just moving the mess into a
        # new drawer (design panel: kill the "Other" bucket). Leave them where they
        # are; a rename-in-place is still allowed if it clarifies the name.
        if ct == "other":
            dest_folder = str(PurePosixPath(from_path).parent)
        new_name = canonical_filename(
            document_type=r["document_type"],
            key_facts_raw=key_facts_raw,
            suggested_filename=r["suggested_filename"],
            original_filename=r["filename"],
            party=party,
        )
        to_path = f"{dest_folder}/{new_name}"

        cur_folder = str(PurePosixPath(from_path).parent)
        renamed = new_name != r["filename"]
        moved = dest_folder != cur_folder

        # Build an explainable reason.
        bits = [f"type={ct}"]
        if year:
            bits.append(f"date→{year}")
        if party:
            bits.append(f"party={party}")
        if _in_junk_drawer(from_path):
            bits.append("out of junk drawer")
        reason = ", ".join(bits) + f" → {dest_folder}"

        tags = [f"{k}:{s}" for k, s, _ in derive_tags(conn, fid)]
        counts["tag"] += len(tags)

        if moved:
            needed_dirs.add(dest_folder)
            actions.append({
                "action_type": "move",
                "file_id": fid,
                "payload": {
                    "from_path": from_path,
                    "to_path": to_path,
                    "from_name": r["filename"],
                    "to_name": new_name,
                    "renamed": renamed,
                    "reason": reason,
                    "tags": tags,
                },
            })
            counts["move"] += 1
        elif renamed:
            actions.append({
                "action_type": "rename",
                "file_id": fid,
                "payload": {
                    "from_path": from_path,
                    "to_path": to_path,
                    "from_name": r["filename"],
                    "to_name": new_name,
                    "reason": f"canonical name; {reason}",
                    "tags": tags,
                },
            })
            counts["rename"] += 1

    # create_dir actions for destination folders not already on disk (read-only stat).
    dir_actions: list[dict[str, Any]] = []
    for d in sorted(needed_dirs):
        if not os.path.isdir(d):
            dir_actions.append({
                "action_type": "create_dir",
                "file_id": None,
                "payload": {"path": d, "reason": "destination folder for the moves above"},
            })
            counts["create_dir"] += 1

    # create_dir first (apply order), then moves/renames/quarantine.
    ordered = dir_actions + actions
    summary = {
        "roots": roots,
        "files_in_scope": len(files),
        "counts": counts,
        "duplicate_groups": sum(1 for g in by_hash.values() if len(g) > 1),
    }

    plan_id = plan_store.create_plan(conn, strategy, {"roots": roots}, summary)
    plan_store.add_actions(conn, plan_id, ordered)
    conn.commit()

    return {"plan_id": plan_id, "summary": summary}

"""Apply engine — execute an approved Plan transactionally, with full undo.

The first code in Organize that writes to the filesystem. It honours the design's
non-negotiables (§6.5, §9):

- Only ``accepted``/``edited`` actions run, in ``seq`` order.
- Every op re-verifies its preconditions (source still present, ``content_hash``
  unchanged, destination free) and writes a `migration_journal` row **before**
  touching disk, so a crash leaves a resumable, fully-undoable state.
- Atomic ``rename`` within a volume; copy → verify-hash → remove across volumes.
- Never overwrites (name collisions get a ` (2)` suffix) and never hard-deletes
  (exact duplicates are moved to a per-plan quarantine, restorable).
- A failing action is logged and skipped; the rest of the plan continues.

Catalog stays consistent: `db.store.update_path` (which also re-syncs the FTS
filename) is reused for every move/rename.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any

from find_and_seek.db.store import iso_now, sha256_file, update_path
from find_and_seek.organize import finder_tags, journal, plan_store

# decisions that mean "do it"
_ACTIONABLE = ("accepted", "edited")


def quarantine_root() -> Path:
    """Where quarantined duplicates go. Override with FINDANDSEEK_QUARANTINE_DIR
    (tests redirect it into a temp dir so they never touch the real ~/.findandseek)."""
    override = os.environ.get("FINDANDSEEK_QUARANTINE_DIR")
    return Path(override) if override else Path.home() / ".findandseek" / "quarantine"


def _unique_dest(path: str) -> str:
    """Return `path`, or `name (2).ext`, `name (3).ext`… if it already exists."""
    if not os.path.exists(path):
        return path
    p = Path(path)
    stem, suffix, parent = p.stem, p.suffix, p.parent
    n = 2
    while True:
        cand = parent / f"{stem} ({n}){suffix}"
        if not cand.exists():
            return str(cand)
        n += 1


def _move(src: str, dest: str) -> None:
    """Atomic within a volume; copy→verify→remove across volumes (never lossy)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        os.rename(src, dest)
        return
    except OSError as e:
        import errno
        if e.errno != errno.EXDEV:
            raise
    # Cross-volume: copy, verify the bytes survived, then drop the source.
    before = sha256_file(src)
    shutil.copy2(src, dest)
    if sha256_file(dest) != before:
        os.remove(dest)
        raise OSError(f"cross-volume copy hash mismatch for {src!r}")
    os.remove(src)


def _mark(conn: sqlite3.Connection, action_id: int, status: str, error: str | None = None) -> None:
    conn.execute(
        "UPDATE plan_actions SET status=?, applied_at=?, error=? WHERE id=?",
        (status, iso_now() if status == "done" else None, error, action_id),
    )
    conn.commit()


def _verify_source(conn: sqlite3.Connection, action: dict[str, Any]) -> str | None:
    """Return a skip-reason if the source is unsafe to act on, else None."""
    from_path = action["payload"].get("from_path")
    if not from_path or not os.path.exists(from_path):
        return "source missing since indexing"
    fid = action["file_id"]
    if fid is not None:
        row = conn.execute("SELECT content_hash FROM files WHERE id=?", (fid,)).fetchone()
        if row and row[0] and sha256_file(from_path) != row[0]:
            return "file changed since indexing"
    return None


def apply_plan(conn: sqlite3.Connection, plan_id: int) -> dict[str, Any] | None:
    """Run a Plan's accepted actions. Returns a result summary, or None if unknown."""
    plan = plan_store.get_plan(conn, plan_id)
    if not plan:
        return None

    plan_store.set_status(conn, plan_id, "applying")
    conn.commit()

    applied = skipped = failed = 0
    errors: list[dict[str, Any]] = []

    for action in plan_store.get_actions(conn, plan_id):
        aid = action["id"]
        atype = action["action_type"]
        pl = action["payload"]

        if action["decision"] not in _ACTIONABLE:
            continue  # leave pending/rejected actions untouched
        if action["status"] == "done":
            continue  # resume-safe: already applied

        try:
            if atype == "create_dir":
                path = pl["path"]
                if os.path.isdir(path):
                    _mark(conn, aid, "skipped", "already exists")
                    skipped += 1
                else:
                    journal.record(conn, plan_id=plan_id, action_id=aid,
                                   op="create_dir", after_path=path)
                    os.makedirs(path, exist_ok=True)
                    _mark(conn, aid, "done")
                    applied += 1
                continue

            if atype == "add_finder_tag":
                # Tagging is non-destructive: require only that the file exists
                # (no content-hash gate — a since-edited file should still get its
                # catalog tags). Reversible via the journalled prior tag set.
                path = pl.get("from_path")
                if not path or not os.path.exists(path):
                    _mark(conn, aid, "skipped", "source missing since indexing")
                    skipped += 1
                    continue
                kinds = tuple(pl.get("kinds") or finder_tags.DEFAULT_KINDS)
                before_raw, after_raw = finder_tags.compute_sync(
                    conn, action["file_id"], path, kinds
                )
                journal.record(conn, plan_id=plan_id, action_id=aid, op="tag",
                               before_path=path,
                               after_path=json.dumps(after_raw),
                               before_hash=json.dumps(before_raw))
                finder_tags.write_raw(path, after_raw)
                conn.commit()
                _mark(conn, aid, "done")
                applied += 1
                continue

            reason = _verify_source(conn, action)
            if reason:
                _mark(conn, aid, "skipped", reason)
                skipped += 1
                continue

            from_path = pl["from_path"]

            if atype in ("move", "rename"):
                dest = _unique_dest(pl["to_path"])
                journal.record(conn, plan_id=plan_id, action_id=aid, op=atype,
                               before_path=from_path, after_path=dest,
                               before_hash=_hash_of(conn, action["file_id"]))
                _move(from_path, dest)
                update_path(conn, action["file_id"], dest)
                conn.commit()
                _mark(conn, aid, "done")
                applied += 1

            elif atype == "quarantine_duplicate":
                qdir = quarantine_root() / f"plan-{plan_id}"
                dest = _unique_dest(str(qdir / PurePosixPath(from_path).name))
                journal.record(conn, plan_id=plan_id, action_id=aid, op="quarantine",
                               before_path=from_path, after_path=dest,
                               before_hash=_hash_of(conn, action["file_id"]))
                _move(from_path, dest)
                update_path(conn, action["file_id"], dest)
                conn.commit()
                _mark(conn, aid, "done")
                applied += 1

            else:
                _mark(conn, aid, "skipped", f"unknown action_type {atype}")
                skipped += 1

        except Exception as e:  # noqa: BLE001 - see below
            # Not just OSError. update_path writes a UNIQUE column, so a stale
            # `files` row still claiming the destination raises IntegrityError —
            # and _unique_dest only stats the disk, so it hands back that path
            # happily. Letting a non-OSError escape skips the terminal status
            # update below, stranding the plan in "applying" forever: the
            # remaining accepted actions never run, and the caller gets a 500
            # with no summary, so the user cannot even see which files moved.
            # One bad action must cost that action, never the whole run.
            _mark(conn, aid, "failed", str(e)[:500])
            errors.append({"action_id": aid, "error": str(e)})
            failed += 1

    status = "failed" if failed else "applied"
    conn.execute(
        "UPDATE plans SET status=?, applied_at=? WHERE id=?",
        (status, iso_now(), plan_id),
    )
    conn.commit()

    summary = {"applied": applied, "skipped": skipped, "failed": failed, "errors": errors}
    return {"plan_id": plan_id, "status": status, **summary}


def _hash_of(conn: sqlite3.Connection, file_id: int | None) -> str | None:
    if file_id is None:
        return None
    row = conn.execute("SELECT content_hash FROM files WHERE id=?", (file_id,)).fetchone()
    return row[0] if row else None

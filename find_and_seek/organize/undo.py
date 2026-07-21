"""Undo — walk a Plan back to its exact prior on-disk state (design §6.6).

Replays the `migration_journal` in reverse: moves/renames/quarantines back to
their `before_path`, and removes created dirs if still empty. Each reversal first
checks the current file's hash against the journalled `before_hash`, so undo never
clobbers a file the user has changed since the apply — that case is reported and
skipped, not forced. Available persistently: a plan stays undoable until its
history is cleared.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from find_and_seek.db.store import iso_now, sha256_file, update_path
from find_and_seek.organize import finder_tags, journal, plan_store
from find_and_seek.organize.apply import _move, quarantine_root


def _file_id_for_action(conn: sqlite3.Connection, action_id: int) -> int | None:
    row = conn.execute("SELECT file_id FROM plan_actions WHERE id=?", (action_id,)).fetchone()
    return row[0] if row else None


def undo_plan(conn: sqlite3.Connection, plan_id: int) -> dict[str, Any] | None:
    """Reverse an applied Plan. Returns a result summary, or None if unknown."""
    plan = plan_store.get_plan(conn, plan_id)
    if not plan:
        return None

    undone = skipped = failed = 0
    errors: list[dict[str, Any]] = []

    for entry in journal.for_plan(conn, plan_id, reverse=True):
        op = entry["op"]
        before_path = entry["before_path"]
        after_path = entry["after_path"]
        action_id = entry["action_id"]

        try:
            if op == "tag":
                # Restore the exact prior Finder-tag set (no hash guard: tag
                # writes don't change file bytes).
                if not before_path or not os.path.exists(before_path):
                    skipped += 1
                    continue
                prior = json.loads(entry["before_hash"]) if entry["before_hash"] else []
                finder_tags.write_raw(before_path, prior)
                if action_id is not None:
                    conn.execute("UPDATE plan_actions SET status='undone' WHERE id=?", (action_id,))
                    conn.commit()
                undone += 1
                continue

            if op == "create_dir":
                # Remove the dir we created, but only if it's still empty.
                if after_path and os.path.isdir(after_path) and not os.listdir(after_path):
                    os.rmdir(after_path)
                    undone += 1
                else:
                    skipped += 1
                continue

            # move | rename | quarantine — restore the file to before_path.
            if not after_path or not os.path.exists(after_path):
                skipped += 1  # nothing at the destination to move back
                continue
            if entry["before_hash"] and sha256_file(after_path) != entry["before_hash"]:
                skipped += 1  # changed since apply — don't clobber the user's edit
                errors.append({"action_id": action_id, "error": "changed since apply; left in place"})
                continue

            # The original location may have been refilled since the apply — the
            # user re-downloads invoice.pdf into Downloads weeks after a plan
            # moved the old one out. os.rename replaces the destination silently
            # on POSIX, so restoring here would destroy a file this plan never
            # touched, with no journal entry and no way back. Undo already
            # refuses to clobber a *changed* file (above); refusing to clobber a
            # *different* one is the same promise. Skip and report — the moved
            # file stays where it is and the user can resolve it by hand.
            if os.path.exists(before_path):
                skipped += 1
                errors.append({
                    "action_id": action_id,
                    "error": f"{before_path} is occupied by another file; left in place",
                })
                continue

            os.makedirs(os.path.dirname(before_path), exist_ok=True)
            # _move, not os.rename: apply may have crossed a filesystem boundary
            # (external drive, network mount), and os.rename raises EXDEV there —
            # which would make every cross-volume move permanently un-undoable.
            _move(after_path, before_path)
            fid = _file_id_for_action(conn, action_id)
            if fid is not None:
                update_path(conn, fid, before_path)
            conn.execute("UPDATE plan_actions SET status='undone' WHERE id=?", (action_id,))
            conn.commit()
            undone += 1

        except OSError as e:
            failed += 1
            errors.append({"action_id": action_id, "error": str(e)})

    # Clean up an empty per-plan quarantine dir.
    qdir = quarantine_root() / f"plan-{plan_id}"
    if qdir.is_dir() and not os.listdir(qdir):
        try:
            qdir.rmdir()
        except OSError:
            pass

    # Only claim the plan is undone if it actually is. A run where entries were
    # skipped (destination occupied, file edited since apply) or failed leaves
    # files at their applied locations; recording "undone" there tells the user
    # the plan was reversed while their files sit where the plan put them, and
    # removes the obvious affordance to try again. Leaving it "applied" keeps
    # undo retryable — already-restored entries no-op on a second run, because
    # their after_path no longer exists.
    fully_undone = failed == 0 and skipped == 0
    if fully_undone:
        conn.execute(
            "UPDATE plans SET status='undone', undone_at=? WHERE id=?",
            (iso_now(), plan_id),
        )
        conn.commit()

    return {"plan_id": plan_id,
            "status": "undone" if fully_undone else "partially_undone",
            "undone": undone, "skipped": skipped, "failed": failed, "errors": errors}

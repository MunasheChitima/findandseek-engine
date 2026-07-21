"""The undo ledger — the literal on-disk truth before each applied action.

Every filesystem op an apply performs writes one row here *before* it runs, so
an interrupted apply (crash/quit) is resumable and the partial result is fully
undoable (design §5.3, §9). Undo replays these in reverse.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from find_and_seek.db.store import iso_now


def record(
    conn: sqlite3.Connection,
    *,
    plan_id: int,
    action_id: int,
    op: str,
    before_path: str | None = None,
    after_path: str | None = None,
    before_hash: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO migration_journal
            (plan_id, action_id, op, before_path, after_path, before_hash, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (plan_id, action_id, op, before_path, after_path, before_hash, iso_now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def for_plan(conn: sqlite3.Connection, plan_id: int, reverse: bool = False) -> list[dict[str, Any]]:
    order = "DESC" if reverse else "ASC"
    rows = conn.execute(
        f"SELECT * FROM migration_journal WHERE plan_id=? ORDER BY id {order}",
        (plan_id,),
    ).fetchall()
    return [
        {
            "id": r["id"], "plan_id": r["plan_id"], "action_id": r["action_id"],
            "op": r["op"], "before_path": r["before_path"],
            "after_path": r["after_path"], "before_hash": r["before_hash"], "ts": r["ts"],
        }
        for r in rows
    ]

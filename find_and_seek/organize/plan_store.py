"""CRUD for Organize plans, actions and the (Phase-2) migration journal.

A Plan is the unit of proposal + approval + undo (design §5.1). This module only
persists/reads plan rows — it never applies anything. `plan_actions.status` stays
``staged`` for the whole of Phase 1; the journal is created but unwritten until
Apply (Phase 2) exists.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from find_and_seek.db.store import iso_now

# Status values used by Phase 1 (draft/previewed only).
PLAN_DRAFT = "draft"
PLAN_PREVIEWED = "previewed"

# Statuses that are safe to garbage-collect: disposable analysis artifacts,
# regenerated on the next background refresh.
#
# Everything NOT listed here is load-bearing history and must never be pruned:
# "applied" and "undone" plans own the plan_actions and the migration_journal
# rows that undo_plan replays, and "applying" is an in-flight apply. Delete any
# of those and the apply becomes irreversible, with the user's files already
# moved on disk.
#
# "failed" is deliberately absent even though it sounds disposable: apply_plan
# sets it when *any* action fails, so a plan that moved nine hundred files and
# hit one permission error lands here with a full, valid undo journal.
#
# This list is only the first of two gates — _prune also refuses any plan with
# migration_journal rows, which is the property that actually matters. Callers
# choose how many artifacts to keep, never which statuses are eligible.
PRUNABLE_STATUSES = frozenset({PLAN_DRAFT, PLAN_PREVIEWED})


def create_plan(
    conn: sqlite3.Connection,
    strategy: str,
    scope: dict[str, Any] | list[str] | None,
    summary: dict[str, Any] | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO plans (created_at, status, strategy, scope_json, summary_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            iso_now(),
            PLAN_DRAFT,
            strategy,
            json.dumps(scope) if scope is not None else None,
            json.dumps(summary) if summary is not None else None,
        ),
    )
    return int(cur.lastrowid)


def add_actions(conn: sqlite3.Connection, plan_id: int, actions: list[dict[str, Any]]) -> int:
    """Persist a list of action dicts. Each dict: action_type, file_id?, payload(dict)."""
    seq = 0
    for a in actions:
        conn.execute(
            """
            INSERT INTO plan_actions (plan_id, seq, action_type, file_id, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                seq,
                a["action_type"],
                a.get("file_id"),
                json.dumps(a.get("payload", {})),
            ),
        )
        seq += 1
    return seq


def set_summary(conn: sqlite3.Connection, plan_id: int, summary: dict[str, Any]) -> None:
    conn.execute(
        "UPDATE plans SET summary_json=? WHERE id=?", (json.dumps(summary), plan_id)
    )


def set_status(conn: sqlite3.Connection, plan_id: int, status: str) -> None:
    conn.execute("UPDATE plans SET status=? WHERE id=?", (status, plan_id))


def _plan_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "applied_at": row["applied_at"],
        "undone_at": row["undone_at"],
        "status": row["status"],
        "strategy": row["strategy"],
        "scope": json.loads(row["scope_json"]) if row["scope_json"] else None,
        "summary": json.loads(row["summary_json"]) if row["summary_json"] else None,
    }


def get_plan(conn: sqlite3.Connection, plan_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone()
    return _plan_to_dict(row) if row else None


def list_plans(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM plans ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_plan_to_dict(r) for r in rows]


def latest_plan_id(conn: sqlite3.Connection, status: str | None = None) -> int | None:
    """The most recent plan's id (optionally of a given status), or None.

    Lets the UI load a *precomputed* plan instantly instead of generating one on
    open. We order by id (monotonic) so it's stable even at sub-second cadence.
    """
    sql = "SELECT id FROM plans"
    params: list[Any] = []
    if status:
        sql += " WHERE status=?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    return int(row["id"]) if row else None


def _prune(conn: sqlite3.Connection, statuses: frozenset[str], keep: int) -> int:
    """Delete all but the newest `keep` plans whose status is in *statuses* **and
    which never touched the filesystem**.

    The journal check is the real guard; the status list is a convenience on top
    of it. Classifying by status alone has now failed twice: first because
    ``applied`` was prunable, then because ``failed`` is *also* a post-apply
    status — ``apply_plan`` sets it when a single action fails, so a plan that
    moved nine hundred files and hit one permission error is marked ``failed``
    while its journal describes nine hundred reversible moves.

    ``migration_journal`` is written before each filesystem operation, so "has a
    journal row" is exactly "something happened on disk that undo can reverse".
    That is the property worth protecting, and unlike a status list it cannot be
    invalidated by a caller elsewhere choosing a new status.

    Ordering and the keep-window are computed *within* the eligible set, so an
    interleaved applied plan neither gets deleted nor consumes a keep slot.
    """
    eligible = statuses & PRUNABLE_STATUSES  # never prunable-by-caller-request
    if not eligible:
        return 0
    ph = ",".join("?" for _ in eligible)
    old = [
        r["id"]
        for r in conn.execute(
            f"SELECT id FROM plans WHERE status IN ({ph}) "
            f"  AND id NOT IN (SELECT DISTINCT plan_id FROM migration_journal"
            f"                 WHERE plan_id IS NOT NULL) "
            f"ORDER BY id DESC LIMIT -1 OFFSET ?",
            (*sorted(eligible), keep),
        ).fetchall()
    ]
    if not old:
        return 0
    ph_old = ",".join("?" for _ in old)
    # plan_actions is deleted explicitly (the FK cascade is only enforced when
    # foreign_keys=ON, which we don't rely on here).
    conn.execute(f"DELETE FROM plan_actions WHERE plan_id IN ({ph_old})", old)
    conn.execute(f"DELETE FROM plans WHERE id IN ({ph_old})", old)
    conn.commit()
    return len(old)


def prune_plans(conn: sqlite3.Connection, keep: int = 5) -> int:
    """Drop all but the newest `keep` *disposable* plans so background
    re-analysis doesn't accumulate stale rows. Returns plans removed.

    Only :data:`PRUNABLE_STATUSES` are eligible. Applied, undone, and in-flight
    plans are never touched: this runs on the ingest worker's idle tick, so a
    status-blind prune would silently delete a plan the user applied — and with
    it the journal that undo replays — a few background refreshes after the
    apply, leaving moved files with no way back.
    """
    return _prune(conn, PRUNABLE_STATUSES, keep)


def prune_draft_plans(conn: sqlite3.Connection, keep: int = 3) -> int:
    """Drop all but the newest `keep` *draft* plans. Returns plans removed.

    Narrower than :func:`prune_plans` — it leaves even previewed plans alone —
    for callers that generate a plan per invocation and should only clean up
    after themselves (notably the MCP tool, which an agent can loop).
    """
    return _prune(conn, frozenset({PLAN_DRAFT}), keep)


def _action_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "plan_id": row["plan_id"],
        "seq": row["seq"],
        "action_type": row["action_type"],
        "file_id": row["file_id"],
        "payload": json.loads(row["payload_json"]) if row["payload_json"] else {},
        "decision": row["decision"],
        "status": row["status"],
    }


def get_actions(
    conn: sqlite3.Connection, plan_id: int, decision: str | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM plan_actions WHERE plan_id=?"
    params: list[Any] = [plan_id]
    if decision:
        sql += " AND decision=?"
        params.append(decision)
    sql += " ORDER BY seq"
    return [_action_to_dict(r) for r in conn.execute(sql, params).fetchall()]


def set_decision(
    conn: sqlite3.Connection,
    plan_id: int,
    decision: str,
    action_ids: list[int] | None = None,
) -> int:
    """Accept/reject/edit actions, in bulk or by id. Returns rows updated.

    Phase 1: this only records the user's intent — nothing is applied.
    """
    if decision not in ("pending", "accepted", "rejected", "edited"):
        raise ValueError(f"invalid decision: {decision}")
    if action_ids:
        placeholders = ",".join("?" for _ in action_ids)
        cur = conn.execute(
            f"UPDATE plan_actions SET decision=? WHERE plan_id=? AND id IN ({placeholders})",
            [decision, plan_id, *action_ids],
        )
    else:
        cur = conn.execute(
            "UPDATE plan_actions SET decision=? WHERE plan_id=?", (decision, plan_id)
        )
    conn.commit()
    return cur.rowcount

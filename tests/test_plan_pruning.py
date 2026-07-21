"""Bounding the one MCP tool that writes.

``propose_organize_plan`` is read-shaped — an agent will call it in a loop — but
it persists a draft plan so the user can open it by id in the app. Unbounded,
that is an agent-triggered unbounded write to the index. ``prune_draft_plans``
caps it, and must never reach a plan that has been applied or undone: that
history is what makes an apply reversible.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from find_and_seek.db.connection import init_db
from find_and_seek.organize import plan_store


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setenv("FINDANDSEEK_TEST", "1")
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "t.db")
        yield conn
        conn.close()


def make_plan(conn, status: str = plan_store.PLAN_DRAFT) -> int:
    pid = plan_store.create_plan(conn, "by_type", None)
    plan_store.add_actions(conn, pid, [{"action_type": "move", "payload": {"to": "/x"}}])
    if status != plan_store.PLAN_DRAFT:
        plan_store.set_status(conn, pid, status)
    conn.commit()
    return pid


def plan_ids(conn) -> list[int]:
    return [r["id"] for r in conn.execute("SELECT id FROM plans ORDER BY id")]


def test_keeps_only_the_newest_drafts(db):
    ids = [make_plan(db) for _ in range(10)]
    removed = plan_store.prune_draft_plans(db, keep=3)
    assert removed == 7
    assert plan_ids(db) == ids[-3:]


def test_repeated_calls_do_not_grow_the_index(db):
    """The actual agent-loop scenario."""
    for _ in range(50):
        make_plan(db)
        plan_store.prune_draft_plans(db, keep=3)
    assert len(plan_ids(db)) == 3


def test_never_prunes_applied_or_undone_plans(db):
    applied = make_plan(db, "applied")
    undone = make_plan(db, "undone")
    drafts = [make_plan(db) for _ in range(10)]

    plan_store.prune_draft_plans(db, keep=2)

    surviving = plan_ids(db)
    assert applied in surviving
    assert undone in surviving
    assert [d for d in drafts if d in surviving] == drafts[-2:]


def test_pruning_cascades_to_actions(db):
    doomed = make_plan(db)
    for _ in range(3):
        make_plan(db)
    plan_store.prune_draft_plans(db, keep=1)

    orphans = db.execute(
        "SELECT COUNT(*) FROM plan_actions WHERE plan_id = ?", (doomed,)
    ).fetchone()[0]
    assert orphans == 0


def test_noop_when_under_the_limit(db):
    ids = [make_plan(db) for _ in range(2)]
    assert plan_store.prune_draft_plans(db, keep=5) == 0
    assert plan_ids(db) == ids


# ── prune_plans: the background-precompute path ───────────────────


def test_background_precompute_never_destroys_an_applied_plan(db):
    """The regression that mattered most.

    ``refresh_artifacts`` runs on the ingest worker's idle tick and calls
    ``prune_plans(keep=5)``. When that pruned by recency alone, a plan the user
    had *applied* was deleted a few background refreshes later — taking its
    plan_actions and journal rows with it, so ``undo_plan`` returned None and
    files that had already moved on disk could not be put back.
    """
    from find_and_seek.organize.undo import undo_plan

    applied = make_plan(db, "applied")
    assert undo_plan(db, applied) is not None, "undo should work immediately after apply"

    for _ in range(20):  # background ticks, far more than keep=5
        make_plan(db, plan_store.PLAN_PREVIEWED)
        plan_store.prune_plans(db, keep=5)

    assert plan_store.get_plan(db, applied) is not None
    assert undo_plan(db, applied) is not None, "undo must survive background pruning"


def test_prune_plans_still_bounds_disposable_artifacts(db):
    """The fix must not turn the pruner into a no-op."""
    for _ in range(30):
        make_plan(db, plan_store.PLAN_PREVIEWED)
    plan_store.prune_plans(db, keep=5)
    assert len(plan_ids(db)) == 5


def test_applied_plans_do_not_consume_keep_slots(db):
    """An interleaved applied plan must not crowd out disposable artifacts —
    the keep-window is computed within the eligible set."""
    for _ in range(3):
        make_plan(db, plan_store.PLAN_PREVIEWED)
    applied = make_plan(db, "applied")
    previewed = [make_plan(db, plan_store.PLAN_PREVIEWED) for _ in range(5)]

    plan_store.prune_plans(db, keep=2)

    surviving = plan_ids(db)
    assert applied in surviving
    assert [p for p in previewed if p in surviving] == previewed[-2:]


@pytest.mark.parametrize("status", ["applied", "undone", "applying"])
def test_load_bearing_statuses_are_never_prunable(db, status):
    protected = make_plan(db, status)
    for _ in range(20):
        make_plan(db, plan_store.PLAN_DRAFT)
        plan_store.prune_plans(db, keep=1)
        plan_store.prune_draft_plans(db, keep=1)
    assert protected in plan_ids(db)
    assert status not in plan_store.PRUNABLE_STATUSES


def test_mcp_tool_declares_a_bound():
    """If the cap constant disappears, the tool is unbounded again."""
    from find_and_seek.mcp import server

    assert isinstance(server._MAX_AGENT_DRAFT_PLANS, int)
    assert 1 <= server._MAX_AGENT_DRAFT_PLANS <= 10


def test_mcp_tool_does_not_claim_to_be_read_only():
    """It writes a draft plan row. Saying otherwise in the docstring is the
    kind of thing an agent believes and then loops on."""
    from find_and_seek.mcp import server

    doc = server.propose_organize_plan.__doc__ or ""
    assert "read-only" not in doc.lower()
    assert "no files are touched" in doc.lower()

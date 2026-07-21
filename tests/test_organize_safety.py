"""Apply/undo safety properties — the ones whose failure costs a user their files.

Every test here corresponds to a defect that was live in this codebase. They are
written to fail against the code as it was, not merely to pass against the code
as it is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from find_and_seek.db.connection import init_db
from find_and_seek.db.store import upsert_file
from find_and_seek.organize import journal, plan_store
from find_and_seek.organize.undo import undo_plan


@pytest.fixture
def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FINDANDSEEK_TEST", "1")
    monkeypatch.setenv("FINDANDSEEK_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    conn = init_db(tmp_path / "t.db")
    yield conn
    conn.close()


def applied_plan_with_move(conn, before: Path, after: Path) -> int:
    """A plan that has already moved `before` -> `after` on disk."""
    pid = plan_store.create_plan(conn, "by_type", None)
    fid = upsert_file(conn, str(after), "hash-of-moved-file", "document")
    plan_store.add_actions(
        conn, pid,
        [{"action_type": "move", "file_id": fid,
          "payload": {"from_path": str(before), "to_path": str(after)}}],
    )
    aid = conn.execute("SELECT id FROM plan_actions WHERE plan_id=?", (pid,)).fetchone()[0]
    journal.record(conn, plan_id=pid, action_id=aid, op="move",
                   before_path=str(before), after_path=str(after))
    plan_store.set_status(conn, pid, "applied")
    conn.commit()
    return pid


# ── undo must never destroy a file the plan did not touch ─────────


def test_undo_refuses_to_overwrite_a_refilled_original_location(db, tmp_path):
    """The scenario: a plan moves ~/Downloads/invoice.pdf out. Weeks later the
    user downloads a *different* invoice.pdf to the same place. Then they undo.

    os.rename replaces the destination silently on POSIX, so the naive restore
    destroys the new file — untouched by the plan, unjournalled, unrecoverable.
    """
    downloads, docs = tmp_path / "Downloads", tmp_path / "Docs"
    downloads.mkdir(), docs.mkdir()
    before, after = downloads / "invoice.pdf", docs / "invoice.pdf"

    after.write_text("the file the plan moved")
    pid = applied_plan_with_move(db, before, after)

    before.write_text("a DIFFERENT invoice the user downloaded later")

    result = undo_plan(db, pid)

    assert before.read_text() == "a DIFFERENT invoice the user downloaded later", (
        "undo destroyed a user file the plan never touched"
    )
    assert after.exists(), "the moved file should be left in place, not lost"
    assert result["skipped"] == 1
    assert result["undone"] == 0
    assert any("occupied" in e["error"] for e in result["errors"])


def test_undo_restores_normally_when_the_original_location_is_free(db, tmp_path):
    """The guard above must not break the ordinary case."""
    downloads, docs = tmp_path / "Downloads", tmp_path / "Docs"
    downloads.mkdir(), docs.mkdir()
    before, after = downloads / "invoice.pdf", docs / "invoice.pdf"

    after.write_text("the file the plan moved")
    pid = applied_plan_with_move(db, before, after)

    result = undo_plan(db, pid)

    assert before.read_text() == "the file the plan moved"
    assert not after.exists()
    assert result["undone"] == 1
    assert result["status"] == "undone"


def test_undo_does_not_claim_success_when_nothing_was_undone(db, tmp_path):
    """Recording "undone" for a run that restored nothing tells the user their
    plan was reversed while the files sit where the plan put them — and removes
    the affordance to retry."""
    downloads, docs = tmp_path / "Downloads", tmp_path / "Docs"
    downloads.mkdir(), docs.mkdir()
    before, after = downloads / "invoice.pdf", docs / "invoice.pdf"

    after.write_text("moved")
    pid = applied_plan_with_move(db, before, after)
    before.write_text("occupant")

    result = undo_plan(db, pid)

    assert result["status"] != "undone"
    assert plan_store.get_plan(db, pid)["status"] == "applied", (
        "a plan that could not be undone must stay applied, so undo is retryable"
    )


# ── pruning must never reach a plan that touched the filesystem ───


@pytest.mark.parametrize("status", ["applied", "undone", "applying", "failed"])
def test_pruning_never_deletes_a_plan_with_a_journal(db, tmp_path, status):
    """`failed` is the one that bit us: apply_plan sets it when a *single* action
    fails, so a plan that moved hundreds of files and hit one permission error
    lands here holding a full, valid undo journal.
    """
    before, after = tmp_path / "a.pdf", tmp_path / "Docs" / "a.pdf"
    (tmp_path / "Docs").mkdir()
    after.write_text("moved")
    pid = applied_plan_with_move(db, before, after)
    plan_store.set_status(db, pid, status)
    db.commit()

    for _ in range(30):  # background precompute ticks, far beyond keep
        p = plan_store.create_plan(db, "by_type", None)
        plan_store.set_status(db, p, plan_store.PLAN_PREVIEWED)
        db.commit()
        plan_store.prune_plans(db, keep=2)

    assert plan_store.get_plan(db, pid) is not None, (
        f"a {status!r} plan with journal rows was pruned — its files are moved "
        "and undo is now impossible"
    )
    assert undo_plan(db, pid) is not None


def test_pruning_still_reclaims_plans_that_never_ran(db):
    """The guard must not turn the pruner into a no-op: a plan with no journal
    touched no files and is safe to reclaim."""
    for _ in range(30):
        p = plan_store.create_plan(db, "by_type", None)
        plan_store.set_status(db, p, plan_store.PLAN_PREVIEWED)
        db.commit()
    plan_store.prune_plans(db, keep=3)
    remaining = db.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
    assert remaining == 3


def test_failed_is_not_in_the_prunable_status_list(db):
    """Belt to the journal check's braces — `failed` is reached after the
    filesystem has been written, so it is not a disposable artifact."""
    assert "failed" not in plan_store.PRUNABLE_STATUSES
    assert "applied" not in plan_store.PRUNABLE_STATUSES
    assert "undone" not in plan_store.PRUNABLE_STATUSES
    assert "applying" not in plan_store.PRUNABLE_STATUSES


# ── apply must always reach a terminal status ─────────────────────


def test_apply_records_a_terminal_status_even_on_an_unexpected_error(db, tmp_path, monkeypatch):
    """A non-OSError escaping the per-action handler used to skip the terminal
    UPDATE, stranding the plan in "applying" forever: remaining actions never
    run and the caller gets a 500 with no summary."""
    from find_and_seek.db.store import sha256_file
    from find_and_seek.organize import apply as apply_mod

    src = tmp_path / "a.pdf"
    src.write_text("x")
    dst = tmp_path / "Docs" / "a.pdf"

    pid = plan_store.create_plan(db, "by_type", None)
    # The real hash, or _verify_source skips the action as "changed since
    # indexing" and the code path under test never runs.
    fid = upsert_file(db, str(src), sha256_file(src), "document")
    plan_store.add_actions(
        db, pid,
        [{"action_type": "move", "file_id": fid,
          "payload": {"from_path": str(src), "to_path": str(dst)}}],
    )
    db.execute("UPDATE plan_actions SET decision='accepted' WHERE plan_id=?", (pid,))
    db.commit()

    def boom(*a, **k):
        raise ValueError("not an OSError")

    monkeypatch.setattr(apply_mod, "update_path", boom)

    result = apply_mod.apply_plan(db, pid)

    assert result is not None, "apply_plan must return a summary, not raise"
    assert plan_store.get_plan(db, pid)["status"] in ("failed", "applied")
    assert plan_store.get_plan(db, pid)["status"] != "applying", (
        "plan stranded mid-flight — the terminal status update was skipped"
    )

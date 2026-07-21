"""Tests for the max-file-size guard (FIX #2) and power-aware ingest (FIX #3).

FIX #2: a file larger than FINDANDSEEK_MAX_FILE_MB is skipped gracefully — it
leaves the queue ('done', not 'failed') and is never indexed, so a single giant
file (log/DB-dump) can't OOM the worker mid-parse — while normal files still index.

FIX #3 (D45 — smooth/full power): the default "smooth" mode is a good citizen —
it pauses on Low-Power Mode / battery, yields to an actively-used machine, and on
AC indexes gently (a reduced batch + cool-down). "full" mode blitzes: a full
batch regardless of power state.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import find_and_seek.ingest.worker as worker
from find_and_seek.db.connection import init_db
from find_and_seek.db.store import enqueue


@pytest.fixture
def test_db(monkeypatch):
    monkeypatch.setenv("FINDANDSEEK_TEST", "1")
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.db")
        yield conn
        conn.close()


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _drain(conn) -> int:
    total = 0
    while True:
        n = worker.process_batch(conn)
        total += n
        if n == 0:
            pending = conn.execute(
                "SELECT COUNT(*) FROM ingest_queue WHERE status='pending'"
            ).fetchone()[0]
            if pending == 0:
                break
    return total


def _status(conn, path: str) -> str:
    return conn.execute(
        "SELECT status FROM ingest_queue WHERE path=?", (path,)
    ).fetchone()[0]


# ── FIX #2: max-file-size guard ──────────────────────────────────────

def test_oversized_file_is_skipped_not_failed(test_db, workspace, monkeypatch):
    """A file over the cap is marked 'done' (left the queue), never indexed, and
    NOT dead-lettered — while a normal file in the same batch still indexes."""
    monkeypatch.setattr(worker, "BATCH_SIZE", 16)
    monkeypatch.setattr(worker, "MAX_FILE_MB", 1)  # tiny cap for the test

    big = workspace / "huge.txt"
    big.write_text("A" * (2 * 1024 * 1024))  # ~2 MB > 1 MB cap
    small = workspace / "normal.txt"
    small.write_text(
        "Quarterly status report. The plumbing invoice for March was paid and "
        "the council letter about the park renovation was filed — enough text."
    )
    big_path, small_path = str(big.resolve()), str(small.resolve())
    for path in (big_path, small_path):
        enqueue(test_db, path, "created")
    test_db.commit()

    _drain(test_db)  # must not raise

    # Oversized file: skipped (queue done, but NOT a failure, and no files row).
    assert _status(test_db, big_path) == "done"
    big_rows = test_db.execute(
        "SELECT COUNT(*) FROM files WHERE path=?", (big_path,)
    ).fetchone()[0]
    assert big_rows == 0, "oversized file must not be indexed"

    # Normal file still indexes.
    assert _status(test_db, small_path) == "done"
    indexed = test_db.execute(
        "SELECT COUNT(*) FROM files WHERE status='indexed'"
    ).fetchone()[0]
    assert indexed == 1


# ── FIX #3: power-aware ingest ───────────────────────────────────────

def _seed_one(conn, workspace: Path) -> str:
    p = workspace / "doc.txt"
    p.write_text(
        "Quarterly status report. The plumbing invoice for March was paid and "
        "the council letter about the park renovation was filed — enough text."
    )
    path = str(p.resolve())
    enqueue(conn, path, "created")
    conn.commit()
    return path


def _seed_many(conn, workspace: Path, count: int) -> list[str]:
    paths = []
    for i in range(count):
        p = workspace / f"doc_{i}.txt"
        p.write_text(
            f"Status report {i}. The plumbing invoice for March was paid and the "
            f"council letter about the park renovation was filed — enough text."
        )
        path = str(p.resolve())
        enqueue(conn, path, "created")
        paths.append(path)
    conn.commit()
    return paths


def test_smooth_low_power_mode_pauses(test_db, workspace, monkeypatch):
    """Smooth mode + Low-Power Mode pauses ingest regardless of AC: returns 0."""
    path = _seed_one(test_db, workspace)
    monkeypatch.setattr(worker, "get_mode", lambda: "smooth")
    monkeypatch.setattr(worker, "low_power_mode", lambda: True)
    monkeypatch.setattr(worker, "on_ac_power", lambda: True)
    monkeypatch.setattr(worker.time, "sleep", lambda *_: None)  # don't wait

    assert worker.process_batch(test_db) == 0
    assert _status(test_db, path) == "pending"  # nothing claimed


def test_smooth_on_battery_pauses(test_db, workspace, monkeypatch):
    """Smooth mode on battery pauses to protect the charge: returns 0, no work."""
    path = _seed_one(test_db, workspace)
    monkeypatch.setattr(worker, "get_mode", lambda: "smooth")
    monkeypatch.setattr(worker, "low_power_mode", lambda: False)
    monkeypatch.setattr(worker, "on_ac_power", lambda: False)
    monkeypatch.setattr(worker.time, "sleep", lambda *_: None)

    assert worker.process_batch(test_db) == 0
    assert _status(test_db, path) == "pending"


def test_smooth_on_ac_busy_yields_to_user(test_db, workspace, monkeypatch):
    """Smooth mode, plugged in but the machine is busy → pause this cycle so we
    don't fight the active user (keeps the fans down while they work)."""
    path = _seed_one(test_db, workspace)
    monkeypatch.setattr(worker, "get_mode", lambda: "smooth")
    monkeypatch.setattr(worker, "low_power_mode", lambda: False)
    monkeypatch.setattr(worker, "on_ac_power", lambda: True)
    monkeypatch.setattr(worker, "machine_is_busy", lambda *a, **k: True)
    monkeypatch.setattr(worker.time, "sleep", lambda *_: None)

    assert worker.process_batch(test_db) == 0
    assert _status(test_db, path) == "pending"


def test_smooth_on_ac_idle_uses_reduced_batch(test_db, workspace, monkeypatch):
    """Smooth mode, plugged in and idle → gentle: a REDUCED batch (still makes
    progress) rather than the full BATCH_SIZE."""
    monkeypatch.setattr(worker, "BATCH_SIZE", 16)
    monkeypatch.setattr(worker, "_SMOOTH_BATCH_SIZE", 2)
    monkeypatch.setattr(worker, "get_mode", lambda: "smooth")
    monkeypatch.setattr(worker, "low_power_mode", lambda: False)
    monkeypatch.setattr(worker, "on_ac_power", lambda: True)
    monkeypatch.setattr(worker, "machine_is_busy", lambda *a, **k: False)
    monkeypatch.setattr(worker.time, "sleep", lambda *_: None)  # skip cooldown

    _seed_many(test_db, workspace, 6)

    n = worker.process_batch(test_db)  # one gentle batch
    assert n <= 2, f"smooth batch must be small, got {n}"
    assert n >= 1, "smooth must still make progress"


def test_full_mode_uses_full_batch_ignoring_power(test_db, workspace, monkeypatch):
    """Full power blitzes: a full batch even on battery + Low-Power Mode (the user
    explicitly opted into the cost)."""
    monkeypatch.setattr(worker, "BATCH_SIZE", 16)
    monkeypatch.setattr(worker, "get_mode", lambda: "full")
    # Hostile power state — full mode must ignore it entirely.
    monkeypatch.setattr(worker, "low_power_mode", lambda: True)
    monkeypatch.setattr(worker, "on_ac_power", lambda: False)
    monkeypatch.setattr(worker.time, "sleep", lambda *_: None)

    paths = _seed_many(test_db, workspace, 3)

    n = worker.process_batch(test_db)
    assert n == 3, "full power: full batch processed regardless of power state"
    for path in paths:
        assert _status(test_db, path) == "done"

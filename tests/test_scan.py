"""Tests for the startup/full-folder scan (find_and_seek.watch.scan).

The cold scan hashes every file in the watched roots. It must NOT do that inside
a single open write transaction: that would hold the WAL write lock for the whole
(multi-minute) scan and starve the ingest worker — the lock-contention startup
crash-loop. These tests pin the batched-commit behaviour (the lock is released
between batches) and that the path filters still hold.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import find_and_seek.watch.scan as scan
from find_and_seek.db.connection import init_db


class _CountingConn:
    """Forwards everything to the real connection but counts commits, so a test
    can prove the scan commits in batches rather than once for the whole walk."""

    def __init__(self, real):
        self._real = real
        self.commits = 0

    def commit(self):
        self.commits += 1
        return self._real.commit()

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("FINDANDSEEK_TEST", "1")
    # tempfile lives under /var on macOS, which is in EXCLUDED_PREFIXES; neutralise
    # the system/user exclusion gates so the temp workspace is actually scanned.
    monkeypatch.setattr(scan, "EXCLUDED_PREFIXES", ())
    monkeypatch.setattr(scan, "excluded_prefixes", lambda: set())
    monkeypatch.setattr(scan, "is_user_excluded", lambda *a, **k: False)
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.db")
        with tempfile.TemporaryDirectory() as ws:
            yield conn, Path(ws)
        conn.close()


def _queued_paths(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT path FROM ingest_queue WHERE status='pending'")}


def test_scan_commits_in_batches_not_one_transaction(env, monkeypatch):
    conn, ws = env
    monkeypatch.setattr(scan, "SCAN_COMMIT_BATCH", 2)
    for i in range(5):
        (ws / f"doc_{i}.txt").write_text(f"healthy document number {i} with text")

    counting = _CountingConn(conn)
    enqueued = scan.scan_roots(counting, [ws])

    assert enqueued == 5
    assert len(_queued_paths(conn)) == 5
    # 5 files, batch of 2 → flush at 2, at 4, then a final flush of the last 1:
    # three short write transactions, NOT one held across the whole scan.
    assert counting.commits == 3


def test_scan_skips_unwatched_hidden_and_excluded_dirs(env):
    conn, ws = env
    (ws / "keep.txt").write_text("real document text")
    (ws / "ignore.bin").write_text("unwatched extension")
    (ws / ".hidden.txt").write_text("dotfile should be skipped")
    nm = ws / "node_modules"
    nm.mkdir()
    (nm / "dep.txt").write_text("dependency junk should be skipped")

    enqueued = scan.scan_roots(conn, [ws])

    queued = _queued_paths(conn)
    assert enqueued == 1
    assert str((ws / "keep.txt").resolve()) in {str(Path(p).resolve()) for p in queued}
    assert not any("node_modules" in p for p in queued)
    assert not any(Path(p).name == "ignore.bin" for p in queued)
    assert not any(Path(p).name == ".hidden.txt" for p in queued)


def test_scan_skips_renamed_node_modules_variants(env):
    """A node_modules folder renamed to dodge *npm* (e.g. with an ``_IGNORE``
    suffix, or a space) must still be skipped — the exact-name match alone let
    thousands of dependency files flood the queue in the field."""
    conn, ws = env
    (ws / "keep.txt").write_text("real document text")
    for name in ("node_modules node_modules_IGNORE", "node_modules_old", "node_modules.bak"):
        d = ws / name
        d.mkdir()
        (d / "dep.txt").write_text("dependency junk should be skipped")

    scan.scan_roots(conn, [ws])

    queued = _queued_paths(conn)
    assert not any("node_modules" in p for p in queued)
    assert str((ws / "keep.txt").resolve()) in {str(Path(p).resolve()) for p in queued}

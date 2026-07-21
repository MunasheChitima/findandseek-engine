"""Regression tests for ingest-worker batch-stage crash isolation.

A failure in one of the heavy batch-level MLX stages (vision / embed / summarise)
used to escape ``process_batch`` and kill the worker, dead-lettering every
healthy file that merely shared the poison file's batch (AAR-031). These tests
prove a single poison file is dead-lettered as an individual while its
batch-mates still index — and the worker never crashes.
"""

from __future__ import annotations

import errno
import multiprocessing
import tempfile
import time
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


def _seed(conn, workspace: Path, good: int = 3) -> tuple[list[str], str]:
    """Write `good` healthy .txt files + one POISON file, enqueue them all."""
    good_paths = []
    for i in range(good):
        p = workspace / f"good_{i}.txt"
        p.write_text(
            f"Quarterly status report number {i}. The plumbing invoice for March "
            f"was paid and the council letter about the park renovation was filed. "
            f"This is healthy document {i} with enough text to chunk and embed."
        )
        good_paths.append(str(p.resolve()))
    poison = workspace / "poison.txt"
    poison.write_text("POISON " * 80)
    poison_path = str(poison.resolve())
    for path in good_paths + [poison_path]:
        enqueue(conn, path, "created")
    conn.commit()
    return good_paths, poison_path


def _drain(conn) -> int:
    """Run process_batch to completion; return total files processed. Must NOT raise."""
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


def test_embed_poison_is_dead_lettered_not_the_batch(test_db, workspace, monkeypatch):
    """A file whose embed stage raises is dead-lettered alone; batch-mates index."""
    monkeypatch.setattr(worker, "BATCH_SIZE", 16)  # one batch, all files together
    good_paths, poison_path = _seed(test_db, workspace)

    real_embed = worker.embed_texts

    def fake_embed(texts):
        if any("POISON" in t for t in texts):
            raise RuntimeError("simulated MLX OOM on poison chunk")
        return real_embed(texts)

    monkeypatch.setattr(worker, "embed_texts", fake_embed)

    # Crucially: this returns normally — the stage failure does not escape.
    _drain(test_db)

    assert _status(test_db, poison_path) == "failed"
    for gp in good_paths:
        assert _status(test_db, gp) == "done", gp
    indexed = test_db.execute(
        "SELECT COUNT(*) FROM files WHERE status='indexed'"
    ).fetchone()[0]
    assert indexed == len(good_paths)


def test_summarise_batch_failure_falls_back_per_file(test_db, workspace, monkeypatch):
    """If the batched summarise forward pass blows up, per-file fallback still
    indexes every good file — the batch is the fast path, not a single point of
    failure."""
    monkeypatch.setattr(worker, "BATCH_SIZE", 16)
    good_paths, poison_path = _seed(test_db, workspace)

    # Force every file through the LLM stage (skip fast_classify) so the batch
    # call is actually exercised, then make the *batch* call explode.
    monkeypatch.setattr(worker, "fast_classify", lambda *a, **k: None)

    def boom(_items):
        raise RuntimeError("simulated batch forward-pass OOM")

    monkeypatch.setattr(worker, "summarise_files", boom)

    _drain(test_db)  # must not raise

    # All files (including poison.txt — it's only "poison" to embed, not summary)
    # index via the per-file fallback; the batch failure took nothing down.
    for path in good_paths + [poison_path]:
        assert _status(test_db, path) == "done", path


def test_summarise_per_file_poison_is_isolated(test_db, workspace, monkeypatch):
    """Batch fails → per-file fallback; one file still fails there → it alone is
    dead-lettered, the rest index."""
    monkeypatch.setattr(worker, "BATCH_SIZE", 16)
    good_paths, poison_path = _seed(test_db, workspace)
    monkeypatch.setattr(worker, "fast_classify", lambda *a, **k: None)

    def batch_boom(_items):
        raise RuntimeError("batch pass failed")

    real_single = worker.summarise_file

    def single(chunks, filename, *a, **k):
        if filename == "poison.txt":
            raise RuntimeError("poison file fails per-file summarise too")
        return real_single(chunks, filename, *a, **k)

    monkeypatch.setattr(worker, "summarise_files", batch_boom)
    monkeypatch.setattr(worker, "summarise_file", single)

    _drain(test_db)

    assert _status(test_db, poison_path) == "failed"
    for gp in good_paths:
        assert _status(test_db, gp) == "done", gp


def test_transient_edeadlk_is_retried_not_failed(test_db, workspace, monkeypatch):
    """A file whose parse hits a one-shot EDEADLK (a read losing a lock-ordering
    race against concurrent Metal/MLX work) is retried in-loop and indexes — it
    is NOT dead-lettered. This was 76% of production ingest failures (AAR-031)."""
    monkeypatch.setattr(worker, "BATCH_SIZE", 16)
    monkeypatch.setattr(worker, "_PARSE_BACKOFF", 0)  # keep the test fast
    good_paths, poison_path = _seed(test_db, workspace)

    real_extract = worker.extract_file
    # extract_file now runs in a forked child process (see extract/guard.py) so
    # a plain dict closed over here wouldn't see the child's mutations — each
    # fork gets its own copy of memory, not a shared one. multiprocessing.Value
    # backs onto real shared memory, so increments in the child are visible here.
    poison_calls = multiprocessing.Value("i", 0)

    def flaky_extract(p):
        if p.name == "poison.txt":
            with poison_calls.get_lock():
                poison_calls.value += 1
                n = poison_calls.value
            if n == 1:
                raise OSError(errno.EDEADLK, "Resource deadlock avoided")
        return real_extract(p)

    monkeypatch.setattr(worker, "extract_file", flaky_extract)

    _drain(test_db)

    # The transient error was retried — every file, poison included, indexes.
    for path in good_paths + [poison_path]:
        assert _status(test_db, path) == "done", path
    assert poison_calls.value >= 2  # proves the retry actually happened


def test_persistent_edeadlk_requeues_then_deadletters(test_db, workspace, monkeypatch):
    """A file that keeps hitting EDEADLK is requeued for later passes (not
    immediately dropped), but is bounded by MAX_ATTEMPTS so it can't loop
    forever — eventually dead-lettered while every healthy file still indexes."""
    monkeypatch.setattr(worker, "BATCH_SIZE", 16)
    monkeypatch.setattr(worker, "_PARSE_BACKOFF", 0)
    good_paths, poison_path = _seed(test_db, workspace)

    real_extract = worker.extract_file

    def always_deadlock(p):
        if p.name == "poison.txt":
            raise OSError(errno.EDEADLK, "Resource deadlock avoided")
        return real_extract(p)

    monkeypatch.setattr(worker, "extract_file", always_deadlock)

    _drain(test_db)  # must terminate (bounded by MAX_ATTEMPTS) and not raise

    assert _status(test_db, poison_path) == "failed"
    assert test_db.execute(
        "SELECT attempts FROM ingest_queue WHERE path=?", (poison_path,)
    ).fetchone()[0] == worker.MAX_ATTEMPTS
    for gp in good_paths:
        assert _status(test_db, gp) == "done", gp


def test_extract_hang_is_bounded_and_dead_lettered(test_db, workspace, monkeypatch):
    """A file whose extraction truly hangs (no exception, ever — the real-world
    failure mode is PyMuPDF blocking forever in a native call on a malformed
    large PDF) is bounded by EXTRACT_TIMEOUT_S and eventually dead-lettered,
    exactly like persistent EDEADLK above. Before the timeout guard existed,
    this poison file would wedge the worker forever, on every restart."""
    monkeypatch.setattr(worker, "BATCH_SIZE", 16)
    monkeypatch.setattr(worker, "EXTRACT_TIMEOUT_S", 0.2)  # keep the test fast
    monkeypatch.setattr(worker, "_PARSE_BACKOFF", 0)
    monkeypatch.setattr(worker, "get_mode", lambda: "full")  # no smooth-mode cooldown
    good_paths, poison_path = _seed(test_db, workspace)

    real_extract = worker.extract_file

    def hanging_extract(p):
        if p.name == "poison.txt":
            time.sleep(5)  # far longer than the 0.2s timeout — must be killed
        return real_extract(p)

    monkeypatch.setattr(worker, "extract_file", hanging_extract)

    start = time.perf_counter()
    _drain(test_db)  # must terminate (bounded by MAX_ATTEMPTS x timeout), not hang
    elapsed = time.perf_counter() - start

    assert elapsed < 10, f"drain took {elapsed:.1f}s — the timeout guard did not bound the hang"
    assert _status(test_db, poison_path) == "failed"
    assert test_db.execute(
        "SELECT attempts FROM ingest_queue WHERE path=?", (poison_path,)
    ).fetchone()[0] == worker.MAX_ATTEMPTS
    for gp in good_paths:
        assert _status(test_db, gp) == "done", gp

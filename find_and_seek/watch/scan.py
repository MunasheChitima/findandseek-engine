"""Startup full scan of watched roots."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from find_and_seek.db.store import enqueue, get_stored_hash, sha256_file
from find_and_seek.ingest.extract.router import WATCHED_EXTENSIONS
from find_and_seek.watch.exclusions import excluded_prefixes, is_user_excluded
from find_and_seek.watch.watcher import EXCLUDED_PREFIXES, is_excluded_dir

# Commit the cold scan in bounded batches rather than one transaction for the
# whole walk. A full scan hashes every file (slow I/O); with sqlite3's default
# deferred isolation, doing that inside a single open transaction would hold the
# WAL write lock for the entire scan and starve the ingest worker (its claim/
# mark writes would block until busy_timeout and then raise "database is locked"
# — the startup crash-loop guarded in run_worker). Hashing happens with NO write
# transaction held; only the short burst of INSERTs per batch takes the lock, so
# the worker gets a write window between every batch. Every first run (and every
# large folder-add) goes through here, so this is a shipping-path fix.
SCAN_COMMIT_BATCH = int(os.environ.get("FINDANDSEEK_SCAN_COMMIT_BATCH") or 200)


def scan_roots(conn: sqlite3.Connection, roots: list[Path] | None = None) -> int:
    # No implicit default scope: when no roots are passed, use exactly what the
    # user granted (which may be nothing). Never fall back to a whole-home crawl.
    if roots is None:
        from find_and_seek.watch.roots import list_roots

        roots = [r for r in list_roots() if r.exists()]
    # User-excluded subtrees (e.g. a code workspace inside ~/Documents). Read once
    # — a full-tree rglob can visit many thousands of paths.
    user_excl = excluded_prefixes()
    enqueued = 0
    pending: list[tuple[str, str]] = []

    def _flush() -> None:
        nonlocal enqueued
        if not pending:
            return
        for path_str, event_type in pending:
            enqueue(conn, path_str, event_type)
        conn.commit()  # short write txn — just these INSERTs, no hashing inside it
        enqueued += len(pending)
        pending.clear()

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if is_excluded_dir(path.parts):
                continue
            if any(str(path).startswith(x) for x in EXCLUDED_PREFIXES):
                continue
            if is_user_excluded(str(path), user_excl):
                continue
            if path.suffix.lower() not in WATCHED_EXTENSIONS:
                continue
            try:
                h = sha256_file(path)
            except OSError:
                continue
            stored = get_stored_hash(conn, str(path))
            if stored != h:
                pending.append((str(path), "created" if stored is None else "modified"))
                if len(pending) >= SCAN_COMMIT_BATCH:
                    _flush()
    _flush()
    return enqueued

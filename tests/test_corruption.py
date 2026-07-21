"""F-4.2.3 — corrupt index detection + lossless rebuild.

Unit-level coverage of the detection crux (is_corruption_error / quick_integrity_ok).
The full HTTP flow (corrupt DB → 503 index_corrupt → /rebuild → ok) is exercised
end-to-end by the audit harness `audit/scripts/phase4.sh` (4.2.3).
"""

from __future__ import annotations

import os
import sqlite3

from find_and_seek.db.connection import (
    get_connection,
    is_corruption_error,
    quick_integrity_ok,
)


def test_is_corruption_error_classifies_malformed():
    malformed = sqlite3.DatabaseError("database disk image is malformed")
    not_a_db = sqlite3.DatabaseError("file is not a database")
    assert is_corruption_error(malformed)
    assert is_corruption_error(not_a_db)


def test_is_corruption_error_ignores_unrelated_errors():
    # A locked DB or a programming error is NOT corruption — must not trigger a
    # destructive rebuild offer.
    assert not is_corruption_error(sqlite3.OperationalError("database is locked"))
    assert not is_corruption_error(sqlite3.DatabaseError("no such table: files"))
    assert not is_corruption_error(ValueError("nope"))


def test_quick_integrity_ok_on_healthy_db(tmp_path):
    db = tmp_path / "ok.db"
    conn = get_connection(db)
    conn.execute("CREATE TABLE t(x)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    assert quick_integrity_ok(conn) is True
    conn.close()


def test_quick_integrity_ok_false_on_corrupt_db(tmp_path):
    db = tmp_path / "bad.db"
    conn = get_connection(db)
    # Make it a real DB with a few pages, then close so WAL is checkpointed.
    conn.execute("CREATE TABLE t(x)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(500)])
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    # Scribble garbage across the b-tree pages (preserve nothing past the header).
    size = os.path.getsize(db)
    with open(db, "r+b") as f:
        f.seek(1024)
        f.write(os.urandom(min(size - 1024, 200_000)))
    # Corruption must surface as a corruption-classified error — either when the
    # connection is opened (the WAL pragma touches the file) or on quick_check.
    # Both are paths the API/lifespan guard against; assert one of them fires.
    try:
        conn2 = get_connection(db)
        healthy = quick_integrity_ok(conn2)
        conn2.close()
        assert healthy is False
    except sqlite3.DatabaseError as exc:
        assert is_corruption_error(exc)

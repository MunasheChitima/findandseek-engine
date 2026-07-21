"""SQLite connection with sqlite-vec extension and pragmas."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import sqlite_vec

from find_and_seek.db.migrations import apply_schema

# Allow an explicit override (tests, alternate indexes) before falling back to
# the per-user default. Lets the API/MCP/worker all point at the same non-default DB.
DEFAULT_DB_PATH = Path(os.environ.get("FINDANDSEEK_DB_PATH", Path.home() / ".findandseek" / "index.db"))
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


# Substrings SQLite uses when the file is structurally unusable. Matched
# case-insensitively against the error message — these are the conditions a user
# can only recover from by rebuilding the index (the source files are intact, so
# a rebuild is always lossless). See F-4.2.3.
_CORRUPTION_MARKERS = (
    "malformed",
    "not a database",
    "file is encrypted",
    "disk image is malformed",
)


def is_corruption_error(exc: BaseException) -> bool:
    """True if `exc` indicates the index file is structurally corrupt/unusable."""
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    return any(m in str(exc).lower() for m in _CORRUPTION_MARKERS)


def quick_integrity_ok(conn: sqlite3.Connection) -> bool:
    """Cheap structural check (`PRAGMA quick_check`); False on any corruption.
    quick_check is far faster than integrity_check and enough to decide
    'serve vs offer rebuild'."""
    try:
        row = conn.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError:
        return False
    return bool(row) and str(row[0]).lower() == "ok"


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # Set busy_timeout first so every subsequent PRAGMA that needs a write lock
    # (e.g. journal_mode=WAL) waits instead of immediately raising OperationalError.
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    _configure_connection(conn)
    return conn


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    conn = get_connection(db_path)
    apply_schema(conn, SCHEMA_PATH)
    # Self-heal a corrupt/legacy keyword index so search can never crash on it.
    # Skipped under tests (fresh temp DBs) and via opt-out for fast tooling.
    if os.environ.get("FINDANDSEEK_TEST") != "1" and os.environ.get("FINDANDSEEK_SKIP_FTS_CHECK") != "1":
        from find_and_seek.db.fts_repair import ensure_fts_healthy

        ensure_fts_healthy(conn)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise

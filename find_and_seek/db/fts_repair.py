"""FTS5 self-heal: detect a corrupt/legacy chunk_fts and rebuild it.

The keyword index can desync (historically: an external-content table that
declared a `filename` column the content table lacked, plus delete triggers that
fed a mismatched filename). That surfaces only on multi-term bm25 ranking as
``database disk image is malformed`` — and SQLite's own ``integrity-check`` does
NOT flag it — so we probe with a real bm25 query and, on failure, drop and
rebuild the index from file_chunks. The rebuilt table matches schema.sql exactly.
"""

from __future__ import annotations

import logging
import sqlite3

log = logging.getLogger(__name__)

# Must stay in sync with schema.sql (chunk_fts + triggers).
_FTS_DDL = [
    "DROP TRIGGER IF EXISTS chunks_ai",
    "DROP TRIGGER IF EXISTS chunks_ad",
    "DROP TRIGGER IF EXISTS chunks_au",
    "DROP TABLE IF EXISTS chunk_fts",
    """CREATE VIRTUAL TABLE chunk_fts USING fts5(
         text, content='file_chunks', content_rowid='id', tokenize='porter unicode61'
       )""",
    "INSERT INTO chunk_fts(chunk_fts) VALUES ('rebuild')",
    """CREATE TRIGGER chunks_ai AFTER INSERT ON file_chunks BEGIN
         INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
       END""",
    """CREATE TRIGGER chunks_ad AFTER DELETE ON file_chunks BEGIN
         INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.id, old.text);
       END""",
    """CREATE TRIGGER chunks_au AFTER UPDATE ON file_chunks BEGIN
         INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.id, old.text);
         INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
       END""",
]


def _is_legacy_schema(conn: sqlite3.Connection) -> bool:
    """True if chunk_fts still has the old `filename` column (the root defect).

    Schema-based detection is deterministic: it catches every pre-fix DB in one
    pass, regardless of which terms happen to land on a corrupt bm25 segment.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunk_fts'"
    ).fetchone()
    return bool(row and row[0] and "filename" in row[0].lower())


def fts_is_healthy(conn: sqlite3.Connection) -> bool:
    """True if chunk_fts is the current schema AND multi-term bm25 ranking works."""
    if _is_legacy_schema(conn):
        log.warning("chunk_fts uses the legacy (filename) schema")
        return False
    try:
        conn.execute(
            "SELECT bm25(chunk_fts) FROM chunk_fts "
            "WHERE chunk_fts MATCH ? ORDER BY 1 LIMIT 1",
            ['"the" OR "and" OR "of" OR "report" OR "invoice"'],
        ).fetchone()
        return True
    except sqlite3.DatabaseError as e:
        log.warning("chunk_fts health probe failed: %s", e)
        return False


def repair_fts(conn: sqlite3.Connection) -> None:
    """Drop and rebuild chunk_fts + triggers from file_chunks. Idempotent."""
    log.warning("rebuilding chunk_fts ...")
    for stmt in _FTS_DDL:
        conn.execute(stmt)
    conn.commit()
    log.warning("chunk_fts rebuilt")


def ensure_fts_healthy(conn: sqlite3.Connection) -> bool:
    """Probe chunk_fts; rebuild if broken. Returns True if a repair was performed."""
    if fts_is_healthy(conn):
        return False
    repair_fts(conn)
    return True

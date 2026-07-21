"""Apply schema.sql idempotently, plus additive column migrations.

schema.sql uses ``CREATE TABLE IF NOT EXISTS``, so a column added to an existing
table is *not* picked up on a live DB (the table already exists). Additive
``ALTER TABLE … ADD COLUMN`` migrations below fill that gap; each is guarded so it
is safe to run repeatedly and on fresh DBs (where the column already exists).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# (table, column, definition) — applied after the base schema. ADD COLUMN can't
# use a non-constant default, so these use constants (matching schema.sql).
_ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("ingest_queue", "priority", "INTEGER NOT NULL DEFAULT 0"),
    ("ingest_queue", "recency", "REAL NOT NULL DEFAULT 0"),
    # JSON {page: anchor} for composite (multi-document) files; NULL otherwise.
    ("file_summaries", "section_anchors", "TEXT"),
    # Classifier confidence in document_type ('high'|'medium'|'low'|'none').
    # First-class so the UI can hedge the badge and MCP can hand agents a
    # trust signal alongside the type, instead of asserting every guess as fact.
    ("file_summaries", "classification_confidence", "TEXT"),
    # The classifier's own short name for the doc's kind — seeds emergent categories.
    ("file_summaries", "suggested_category", "TEXT"),
)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    for table, column, ddl in _ADDITIVE_COLUMNS:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not cols:
            continue  # table doesn't exist yet (bare DB) — nothing to alter
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    # Created here (not in schema.sql) because it references priority/recency,
    # which on a live DB don't exist until the ALTERs above have run.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_queue_order "
        "ON ingest_queue(status, priority DESC, recency DESC)"
    )


def apply_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    from find_and_seek.config.profiles import EMBED_DIM
    sql = schema_path.read_text(encoding="utf-8")
    if EMBED_DIM != 768:
        sql = sql.replace("FLOAT[768]", f"FLOAT[{EMBED_DIM}]")
    conn.executescript(sql)
    _ensure_columns(conn)
    conn.commit()

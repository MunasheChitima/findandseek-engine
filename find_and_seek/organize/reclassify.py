"""Taxonomy mutations + re-classification — the "engage with the change" half.

Adding a category isn't a cosmetic label: it expands the taxonomy the classifier
reasons against, and then we *re-run classification* over the documents most
likely to belong to it (the unsorted/other pile), so the new category actually
fills. Correcting a single file is the trust loop. Both write only DB rows.
"""

from __future__ import annotations

import logging
import sqlite3

from find_and_seek.organize.categories import load_taxonomy
from find_and_seek.organize.classify import classify_document
from find_and_seek.organize.taxonomy import slugify

logger = logging.getLogger(__name__)

# Files in these buckets are the natural candidates to re-home when the taxonomy
# grows or a reclassify is requested — we don't disturb confident placements.
RECLASSIFY_SCOPES = ("needs-review", "other")


def add_category(conn: sqlite3.Connection, label: str, definition: str, excludes: str = "") -> str:
    """Add a user category to the taxonomy. Returns its slug."""
    slug = slugify(label)
    if not slug:
        raise ValueError("category needs a name")
    conn.execute(
        "INSERT INTO categories (slug, label, definition, excludes, builtin) "
        "VALUES (?,?,?,?,0) ON CONFLICT(slug) DO UPDATE SET "
        "label=excluded.label, definition=excluded.definition, excludes=excluded.excludes",
        (slug, label.strip(), definition.strip(), excludes.strip()),
    )
    conn.commit()
    logger.info("added user category: %s", slug)
    return slug


def set_file_category(conn: sqlite3.Connection, file_id: int, slug: str) -> bool:
    """User correction: pin a file's type. Validated against the taxonomy."""
    valid = {c.slug for c in load_taxonomy(conn)} | {"needs-review"}
    if slug not in valid:
        return False
    # A user-asserted type is the most trustworthy signal we have — pin confidence
    # to 'high' so the UI/MCP stop hedging a category the human confirmed.
    cur = conn.execute(
        "UPDATE file_summaries SET document_type=?, confidence_note='user', "
        "classification_confidence='high' WHERE file_id=?",
        (slug, file_id),
    )
    conn.commit()
    return cur.rowcount > 0


def _file_text(conn: sqlite3.Connection, file_id: int, limit: int = 6) -> str:
    rows = conn.execute(
        "SELECT text FROM file_chunks WHERE file_id=? ORDER BY chunk_index LIMIT ?",
        (file_id, limit),
    ).fetchall()
    return "\n".join((r[0] if isinstance(r, tuple) else r["text"]) for r in rows)


def reclassify(conn: sqlite3.Connection, scopes: tuple[str, ...] = RECLASSIFY_SCOPES,
               limit: int | None = None) -> dict:
    """Re-run classification over the files in `scopes` against the *current*
    taxonomy. Returns {scanned, changed}. Confident placements aren't touched."""
    taxonomy = load_taxonomy(conn)
    q = ("SELECT f.id, f.path, f.filename FROM file_summaries s JOIN files f ON f.id=s.file_id "
         f"WHERE s.document_type IN ({','.join('?' * len(scopes))})")
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = conn.execute(q, scopes).fetchall()
    scanned = changed = 0
    for r in rows:
        fid = r[0] if isinstance(r, tuple) else r["id"]
        path = r[1] if isinstance(r, tuple) else r["path"]
        name = r[2] if isinstance(r, tuple) else r["filename"]
        scanned += 1
        res = classify_document(path, name, _file_text(conn, fid), conn=conn, taxonomy=taxonomy)
        old = conn.execute("SELECT document_type FROM file_summaries WHERE file_id=?", (fid,)).fetchone()
        old_t = (old[0] if isinstance(old, tuple) else old["document_type"]) if old else None
        # Refresh the confidence on every pass (cheap, and it may have been NULL
        # on a pre-confidence-column DB); change the type only when it actually moves.
        conn.execute(
            "UPDATE file_summaries SET classification_confidence=?, suggested_category=? WHERE file_id=?",
            (res.confidence, res.suggested, fid))
        if res.slug != old_t:
            conn.execute("UPDATE file_summaries SET document_type=? WHERE file_id=?", (res.slug, fid))
            changed += 1
    conn.commit()
    logger.info("reclassify: scanned %d, changed %d", scanned, changed)
    return {"scanned": scanned, "changed": changed}

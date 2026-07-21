"""Database helpers — purge, upsert, queue."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec

from find_and_seek.db.connection import transaction
from find_and_seek.ingest.chunk import Chunk


# Version of the ingest pipeline's derived data (chunks/entities/summaries/facts).
# Bump when an ingest-side change must apply to already-indexed files: the worker
# re-processes any file whose stored index_version is older, even if its content
# hash is unchanged. v2: letterspace normalization, entity quality gate,
# key_facts entities, money decimal precision (AAR-021).
# v3: spaCy PERSON/ORG/LOC removed; key_facts (Qwen3-4B) is sole source for names.
# v4: Apple Foundation Models (AFM) replaces Qwen2.5-3B as the summary backend.
INDEX_VERSION = 4


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def purge_file(conn: sqlite3.Connection, file_id: int) -> None:
    chunk_ids = [r[0] for r in conn.execute("SELECT id FROM file_chunks WHERE file_id=?", (file_id,))]
    for cid in chunk_ids:
        conn.execute("DELETE FROM chunk_vectors WHERE chunk_id=?", (cid,))
    conn.execute("DELETE FROM summary_vectors WHERE file_id=?", (file_id,))
    conn.execute("DELETE FROM files WHERE id=?", (file_id,))


def purge_file_by_path(conn: sqlite3.Connection, path: str) -> None:
    row = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
    if row:
        purge_file(conn, row[0])


def purge_root(conn: sqlite3.Connection, root: str | Path) -> int:
    """Forget everything indexed under ``root`` — the security half of removing a
    watched folder (per MAC_APP_STORE_AND_PERMISSIONS.md): revoking access must
    revoke searchability.

    Deletes each file under the root (cascading to chunks/summaries/entities/tags/
    facts via FK, and to ``chunk_fts`` via trigger), its vec0 vectors (which FK
    cascade can't reach — same reason ``purge_file`` clears them by hand), and any
    pending ``ingest_queue`` rows beneath the prefix. Returns the files purged.

    Matches the root itself and anything beneath it; the trailing ``/`` on the LIKE
    prefix keeps a sibling like ``/a/Docs2`` from being swept when removing
    ``/a/Docs``.
    """
    base = str(Path(root).expanduser()).rstrip("/")
    like = base + "/%"
    file_ids = [r[0] for r in conn.execute(
        "SELECT id FROM files WHERE path = ? OR path LIKE ?", (base, like)
    )]
    for fid in file_ids:
        purge_file(conn, fid)
    conn.execute("DELETE FROM ingest_queue WHERE path = ? OR path LIKE ?", (base, like))
    return len(file_ids)


def purge_unwatched(conn: sqlite3.Connection) -> dict[str, int]:
    """Forget files (and queue rows) whose extension is no longer in the watched
    set — e.g. after narrowing scope to documents-only, where a dev workspace's
    code/config files become pure search noise. Auto-adapts to the current
    ``WATCHED_EXTENSIONS`` (so it also cleans up if code indexing is later toggled
    off again). Returns counts of purged files and dropped queue rows.

    Reversible: the files on disk are never touched — re-enabling the type and
    re-scanning re-indexes them. ``files.extension`` is stored without the leading
    dot, so the watched set is compared dot-stripped.
    """
    from find_and_seek.ingest.extract.router import WATCHED_EXTENSIONS

    watched = {e.lstrip(".").lower() for e in WATCHED_EXTENSIONS}
    placeholders = ",".join("?" * len(watched))
    args = tuple(sorted(watched))

    file_ids = [r[0] for r in conn.execute(
        f"SELECT id FROM files WHERE lower(extension) NOT IN ({placeholders})", args
    )]
    for fid in file_ids:
        purge_file(conn, fid)

    # Queue rows are keyed by path; there's no SQL suffix helper, so filter in
    # Python (the queue is small relative to a full corpus).
    bad_paths = [
        p for (p,) in conn.execute("SELECT path FROM ingest_queue")
        if Path(p).suffix.lstrip(".").lower() not in watched
    ]
    for p in bad_paths:
        conn.execute("DELETE FROM ingest_queue WHERE path = ?", (p,))

    return {"files": len(file_ids), "queue": len(bad_paths)}


def get_stored_hash(conn: sqlite3.Connection, path: str) -> str | None:
    row = conn.execute("SELECT content_hash FROM files WHERE path=?", (path,)).fetchone()
    return row[0] if row else None


def file_with_hash(conn: sqlite3.Connection, content_hash: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM files WHERE content_hash=?", (content_hash,)).fetchone()


def update_path(conn: sqlite3.Connection, file_id: int, new_path: str) -> None:
    p = Path(new_path)
    conn.execute(
        "UPDATE files SET path=?, filename=? WHERE id=?",
        (str(new_path), p.name, file_id),
    )
    # FTS no longer stores filename (see schema.sql); the filename-token boost in
    # search.hybrid reads files.filename live, so a rename needs no FTS re-sync.


def upsert_file(
    conn: sqlite3.Connection,
    path: str,
    content_hash: str,
    file_type: str,
    status: str = "indexed",
    page_count: int | None = None,
) -> int:
    p = Path(path)
    stat = p.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO files (path, filename, extension, content_hash, size_bytes, modified_at,
                           indexed_at, index_version, file_type, status, page_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          content_hash=excluded.content_hash,
          size_bytes=excluded.size_bytes,
          modified_at=excluded.modified_at,
          indexed_at=excluded.indexed_at,
          index_version=excluded.index_version,
          file_type=excluded.file_type,
          status=excluded.status,
          page_count=excluded.page_count
        """,
        (
            str(path),
            p.name,
            p.suffix.lower().lstrip("."),
            content_hash,
            stat.st_size,
            modified,
            iso_now(),
            INDEX_VERSION,
            file_type,
            status,
            page_count,
        ),
    )
    row = conn.execute("SELECT id FROM files WHERE path=?", (str(path),)).fetchone()
    return int(row[0])


def replace_chunks(
    conn: sqlite3.Connection,
    file_id: int,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> list[int]:
    old_ids = [r[0] for r in conn.execute("SELECT id FROM file_chunks WHERE file_id=?", (file_id,))]
    for cid in old_ids:
        conn.execute("DELETE FROM chunk_vectors WHERE chunk_id=?", (cid,))
    conn.execute("DELETE FROM file_entities WHERE file_id=?", (file_id,))
    conn.execute("DELETE FROM file_chunks WHERE file_id=?", (file_id,))

    chunk_ids: list[int] = []
    for chunk, vec in zip(chunks, vectors):
        cur = conn.execute(
            """
            INSERT INTO file_chunks (file_id, chunk_index, text, source_type, location_ref, token_estimate)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                chunk.chunk_index,
                chunk.text,
                chunk.source_type,
                chunk.location_ref,
                chunk.token_estimate,
            ),
        )
        cid = cur.lastrowid
        chunk_ids.append(cid)
        conn.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (cid, sqlite_vec.serialize_float32(np.array(vec, dtype=np.float32))),
        )
    return chunk_ids


def indexed_twin(conn: sqlite3.Connection, content_hash: str, exclude_path: str) -> int | None:
    """Return the id of an already-indexed file with identical content, if any.

    Only twins at the current INDEX_VERSION qualify — during a version-bump
    re-index, cloning from a stale twin would copy old derived data. The
    candidate's extension must also match: a content-hash collision across
    extensions (e.g. a text file and an image sharing bytes, or more
    realistically a duplicate-content synthetic fixture saved under two
    extensions) means two different extractors ran — cloning across that
    boundary silently inherits the wrong file_type and whatever chunk count
    that extractor produced, including zero if OCR/parsing was disabled.
    """
    exclude_ext = Path(exclude_path).suffix.lower()
    rows = conn.execute(
        """
        SELECT f.id, f.path, COUNT(fc.id) AS chunk_count
        FROM files f LEFT JOIN file_chunks fc ON fc.file_id = f.id
        WHERE f.content_hash=? AND f.status='indexed' AND f.index_version=? AND f.path<>?
        GROUP BY f.id
        """,
        (content_hash, INDEX_VERSION, exclude_path),
    ).fetchall()
    for row in rows:
        if Path(row[1]).suffix.lower() == exclude_ext and row[2] > 0:
            return int(row[0])
    return None


def clone_indexed_file(
    conn: sqlite3.Connection,
    src_file_id: int,
    dest_path: str,
    content_hash: str,
    file_type: str,
    page_count: int | None = None,
) -> int:
    """Copy chunks/vectors/summary/entities from an identical file instead of
    re-running embedding + summarisation for byte-identical duplicates."""
    src_chunk_count = conn.execute(
        "SELECT COUNT(*) FROM file_chunks WHERE file_id=?", (src_file_id,)
    ).fetchone()[0]
    if src_chunk_count == 0:
        raise ValueError(
            f"refusing to clone file_id={src_file_id} into {dest_path!r}: source has zero chunks"
        )

    dest_id = upsert_file(conn, dest_path, content_hash, file_type, status="indexed", page_count=page_count)

    old_ids = [r[0] for r in conn.execute("SELECT id FROM file_chunks WHERE file_id=?", (dest_id,))]
    for cid in old_ids:
        conn.execute("DELETE FROM chunk_vectors WHERE chunk_id=?", (cid,))
    conn.execute("DELETE FROM file_entities WHERE file_id=?", (dest_id,))
    conn.execute("DELETE FROM file_chunks WHERE file_id=?", (dest_id,))

    id_map: dict[int, int] = {}
    src_chunks = conn.execute(
        """
        SELECT id, chunk_index, text, source_type, location_ref, token_estimate
        FROM file_chunks WHERE file_id=? ORDER BY chunk_index
        """,
        (src_file_id,),
    ).fetchall()
    for r in src_chunks:
        cur = conn.execute(
            """
            INSERT INTO file_chunks (file_id, chunk_index, text, source_type, location_ref, token_estimate)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (dest_id, r["chunk_index"], r["text"], r["source_type"], r["location_ref"], r["token_estimate"]),
        )
        new_cid = cur.lastrowid
        id_map[r["id"]] = new_cid
        vec = conn.execute("SELECT embedding FROM chunk_vectors WHERE chunk_id=?", (r["id"],)).fetchone()
        if vec:
            conn.execute("INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)", (new_cid, vec[0]))

    summary = conn.execute("SELECT * FROM file_summaries WHERE file_id=?", (src_file_id,)).fetchone()
    if summary:
        conn.execute(
            """
            INSERT INTO file_summaries (file_id, summary_text, one_line_anchor, document_type,
                                        key_facts, suggested_filename, confidence_note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_id) DO UPDATE SET
              summary_text=excluded.summary_text, one_line_anchor=excluded.one_line_anchor,
              document_type=excluded.document_type, key_facts=excluded.key_facts,
              suggested_filename=excluded.suggested_filename, confidence_note=excluded.confidence_note
            """,
            (
                dest_id, summary["summary_text"], summary["one_line_anchor"], summary["document_type"],
                summary["key_facts"], summary["suggested_filename"], summary["confidence_note"],
            ),
        )
    svec = conn.execute("SELECT embedding FROM summary_vectors WHERE file_id=?", (src_file_id,)).fetchone()
    if svec:
        conn.execute("DELETE FROM summary_vectors WHERE file_id=?", (dest_id,))
        conn.execute("INSERT INTO summary_vectors (file_id, embedding) VALUES (?, ?)", (dest_id, svec[0]))

    for e in conn.execute(
        "SELECT chunk_id, entity_type, entity_value, entity_raw FROM file_entities WHERE file_id=?",
        (src_file_id,),
    ):
        conn.execute(
            """
            INSERT INTO file_entities (file_id, chunk_id, entity_type, entity_value, entity_raw)
            VALUES (?, ?, ?, ?, ?)
            """,
            (dest_id, id_map.get(e["chunk_id"]), e["entity_type"], e["entity_value"], e["entity_raw"]),
        )
    return dest_id


def write_summary(
    conn: sqlite3.Connection,
    file_id: int,
    summary: dict[str, Any],
    summary_vector: list[float] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO file_summaries (file_id, summary_text, one_line_anchor, document_type,
                                    key_facts, suggested_filename, confidence_note, section_anchors,
                                    classification_confidence, suggested_category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
          summary_text=excluded.summary_text,
          one_line_anchor=excluded.one_line_anchor,
          document_type=excluded.document_type,
          key_facts=excluded.key_facts,
          suggested_filename=excluded.suggested_filename,
          confidence_note=excluded.confidence_note,
          section_anchors=excluded.section_anchors,
          classification_confidence=excluded.classification_confidence,
          suggested_category=excluded.suggested_category
        """,
        (
            file_id,
            summary.get("summary_text"),
            summary.get("one_line_anchor"),
            summary.get("document_type"),
            json.dumps(summary.get("key_facts", {})),
            summary.get("suggested_filename"),
            summary.get("confidence_note"),
            json.dumps(summary["section_anchors"]) if summary.get("section_anchors") else None,
            summary.get("classification_confidence"),
            summary.get("suggested_category"),
        ),
    )
    if summary_vector:
        conn.execute("DELETE FROM summary_vectors WHERE file_id=?", (file_id,))
        conn.execute(
            "INSERT INTO summary_vectors (file_id, embedding) VALUES (?, ?)",
            (file_id, sqlite_vec.serialize_float32(np.array(summary_vector, dtype=np.float32))),
        )


def write_entities(conn: sqlite3.Connection, file_id: int, entities: list[dict[str, Any]]) -> None:
    for ent in entities:
        conn.execute(
            """
            INSERT INTO file_entities (file_id, chunk_id, entity_type, entity_value, entity_raw)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                file_id,
                ent.get("chunk_id"),
                ent["entity_type"],
                ent["entity_value"],
                ent.get("entity_raw"),
            ),
        )


def write_facts(conn: sqlite3.Connection, file_id: int, facts: list[dict[str, Any]]) -> None:
    """Replace the typed facts for a file (idempotent on re-index)."""
    conn.execute("DELETE FROM facts WHERE file_id=?", (file_id,))
    for f in facts:
        conn.execute(
            """
            INSERT INTO facts (file_id, chunk_id, fact_type, key, value_text,
                               value_number, value_date, unit, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                f.get("chunk_id"),
                f["fact_type"],
                f.get("key"),
                f.get("value_text"),
                f.get("value_number"),
                f.get("value_date"),
                f.get("unit"),
                f.get("confidence"),
                f.get("source"),
            ),
        )


def backfill_facts(conn: sqlite3.Connection, limit: int | None = None) -> int:
    """Populate the facts table from already-indexed summaries + entities.

    For the existing catalog (the ingest hook only covers newly-indexed files).
    Reads summaries/entities, normalizes via ingest.facts, writes facts. Returns
    the number of files processed.
    """
    from find_and_seek.ingest.facts import extract_facts

    rows = conn.execute(
        "SELECT file_id, key_facts FROM file_summaries"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    n = 0
    for row in rows:
        file_id = row["file_id"]
        try:
            key_facts = json.loads(row["key_facts"]) if row["key_facts"] else {}
        except (ValueError, TypeError):
            key_facts = {}
        ents = conn.execute(
            "SELECT entity_type, entity_value, entity_raw, chunk_id "
            "FROM file_entities WHERE file_id=?",
            (file_id,),
        ).fetchall()
        entities = [dict(e) for e in ents]
        facts = extract_facts({"key_facts": key_facts}, entities)
        with transaction(conn):
            write_facts(conn, file_id, facts)
        n += 1
    return n


def _recency_of(path: str) -> float:
    """Recency score for queue ordering: most-recently modified OR opened wins.
    max(mtime, atime) honours 'recently opened' where the OS tracks atime, and
    falls back to mtime where it doesn't (macOS often disables atime updates).
    Missing/unreadable files (e.g. delete events) score 0 → processed last."""
    try:
        st = os.stat(path)
        return max(st.st_mtime, st.st_atime)
    except OSError:
        return 0.0


def enqueue(conn: sqlite3.Connection, path: str, event_type: str, priority: int = 0) -> None:
    conn.execute(
        """
        INSERT INTO ingest_queue (path, event_type, queued_at, status, attempts, priority, recency)
        VALUES (?, ?, ?, 'pending', 0, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          event_type=excluded.event_type,
          queued_at=excluded.queued_at,
          status='pending',
          last_error=NULL,
          priority=excluded.priority,
          recency=excluded.recency
        """,
        (path, event_type, iso_now(), priority, _recency_of(path)),
    )


MAX_ATTEMPTS = 3


def claim_queue(conn: sqlite3.Connection, limit: int = 16) -> list[sqlite3.Row]:
    """Atomically claim up to ``limit`` pending rows.

    A single ``UPDATE ... RETURNING`` flips pending→processing and returns the
    claimed rows in one statement, so two workers can't claim the same path
    (the SELECT-then-UPDATE race). Rows that have already exhausted their
    attempts are skipped here and dead-lettered by ``requeue_stale_processing``.

    The inner SELECT orders by ``priority DESC, recency DESC`` so the highest-
    priority, most-recently modified/opened files are *selected* first (recent
    files land in the first batches → understood first). NB: the RETURNING row
    order within one claim is unspecified in SQLite; that's fine — a batch is
    processed as a unit, so only cross-batch selection order matters.
    """
    rows = conn.execute(
        """
        UPDATE ingest_queue SET status='processing', attempts=attempts+1
        WHERE path IN (
            SELECT path FROM ingest_queue
            WHERE status='pending' AND attempts < ?
            ORDER BY priority DESC, recency DESC, queued_at LIMIT ?
        )
        RETURNING *
        """,
        (MAX_ATTEMPTS, limit),
    ).fetchall()
    conn.commit()
    return rows


def requeue_stale_processing(conn: sqlite3.Connection, max_attempts: int = MAX_ATTEMPTS) -> tuple[int, int]:
    """Crash recovery: rows left in 'processing' by a dead worker are either
    requeued (attempts left) or dead-lettered to 'failed' (attempts exhausted).
    Call once on worker startup. Returns (requeued, dead_lettered).

    Also dead-letters any 'pending' row that has already hit the attempts ceiling.
    ``claim_queue`` only ever selects ``attempts < max_attempts``, so a pending row
    at the ceiling can never be claimed again — it is stuck by definition and
    belongs in 'failed'. Rows can land there if a crash or a failed DB write (e.g.
    a transient 'disk full') interrupts recovery midway; without this they sit in
    'pending' forever, inflating the "waiting to be indexed" count with ghosts that
    are never retried and never resolved. COALESCE preserves any recorded error."""
    dead = conn.execute(
        "UPDATE ingest_queue SET status='failed', "
        "last_error=COALESCE(last_error, 'max attempts exceeded') "
        "WHERE status IN ('processing', 'pending') AND attempts >= ?",
        (max_attempts,),
    ).rowcount
    requeued = conn.execute(
        "UPDATE ingest_queue SET status='pending' WHERE status='processing' AND attempts < ?",
        (max_attempts,),
    ).rowcount
    conn.commit()
    return requeued, dead


def mark_done(conn: sqlite3.Connection, path: str) -> None:
    conn.execute(
        "UPDATE ingest_queue SET status='done', last_error=NULL WHERE path=?",
        (path,),
    )


def mark_failed(conn: sqlite3.Connection, path: str, error: str) -> None:
    conn.execute(
        "UPDATE ingest_queue SET status='failed', last_error=? WHERE path=?",
        (error[:2000], path),
    )


def requeue_pending(conn: sqlite3.Connection, path: str, note: str | None = None) -> None:
    """Return a row to the pending lane for a later, less-contended retry.

    Unlike ``mark_failed`` (which permanently dead-letters), this keeps the file
    in the queue: a *transient* failure (e.g. an EDEADLK the syscall lost to a
    Metal/MLX lock-ordering race — the file itself is fine) shouldn't drop an
    indexable file from the index. The ``attempts`` counter (bumped at every
    ``claim_queue``) still bounds how many times a row can be requeued, so a row
    that keeps failing is eventually dead-lettered rather than looping forever."""
    conn.execute(
        "UPDATE ingest_queue SET status='pending', last_error=? WHERE path=?",
        ((note or "transient error — will retry")[:2000], path),
    )


def estimated_tokens_for_file(conn: sqlite3.Connection, file_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(token_estimate), 0) FROM file_chunks WHERE file_id=?",
        (file_id,),
    ).fetchone()
    total = int(row[0])
    if total:
        return total
    row = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(text)), 0) FROM file_chunks WHERE file_id=?",
        (file_id,),
    ).fetchone()
    return int(row[0]) // 4

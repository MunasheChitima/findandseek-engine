"""Re-summarise an existing catalog in place — regenerate the derived anchor /
summary / key_facts over already-stored chunks, WITHOUT re-extracting or
re-embedding.

This is the missing half of a catalog refresh. `tools/refresh_catalog.py` already
re-types (document_type) and backfills the facts table, but it deliberately keeps
the *old* anchors/summaries — which is why a catalog indexed before the
extraction fixes still shows weak anchors ("Invoice for MRI services" with no
amount). This pass re-runs the summariser (improved prompt + token budget) over
the stored chunks and updates `one_line_anchor`, `summary_text`, `key_facts`,
leaving `document_type` (already fixed by the classifier) untouched.

Reads stored chunks only; never touches the original files or the network.
Idempotent, batched, commits per batch. WRITES THE TARGET DB — back it up first.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Callable

logger = logging.getLogger(__name__)


def _bound_mlx_cache() -> None:
    """Cap MLX's Metal buffer cache so batched inference can't balloon unified
    memory (the shipped service sets FINDANDSEEK_MLX_CACHE_MB=512; ad-hoc runs must
    too, or a re-summarise will eat the machine)."""
    try:
        import mlx.core as mx
        cap = int(os.environ.get("FINDANDSEEK_MLX_CACHE_MB", "512")) * 1024 * 1024
        setter = getattr(mx, "set_cache_limit", None) or getattr(getattr(mx, "metal", None), "set_cache_limit", None)
        if setter:
            setter(cap)
    except Exception:  # noqa: BLE001
        pass


def _free_mlx_cache() -> None:
    try:
        import mlx.core as mx
        (getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None) or (lambda: None))()
    except Exception:  # noqa: BLE001
        pass


def resummarize(conn: sqlite3.Connection, batch_size: int = 8, limit: int | None = None,
                progress: Callable[[int, int], None] | None = None) -> dict:
    from find_and_seek.ingest.chunk import Chunk
    from find_and_seek.ingest.summarise import summarise_files

    _bound_mlx_cache()
    q = "SELECT id, filename FROM files WHERE status='indexed' ORDER BY id"
    if limit:
        q += f" LIMIT {int(limit)}"
    files = conn.execute(q).fetchall()
    total = len(files)
    done = changed = 0

    for start in range(0, total, batch_size):
        batch = files[start:start + batch_size]
        items, ids = [], []
        for row in batch:
            fid = row[0] if isinstance(row, tuple) else row["id"]
            name = row[1] if isinstance(row, tuple) else row["filename"]
            crows = conn.execute(
                "SELECT text FROM file_chunks WHERE file_id=? ORDER BY chunk_index LIMIT 8", (fid,)
            ).fetchall()
            if not crows:
                continue
            chunks = [Chunk(text=(c[0] if isinstance(c, tuple) else c["text"]),
                            source_type="text", location_ref="", chunk_index=j,
                            token_estimate=len((c[0] if isinstance(c, tuple) else c["text"])) // 4)
                      for j, c in enumerate(crows)]
            items.append((chunks, name))
            ids.append(fid)
        if items:
            summaries = summarise_files(items)
            for fid, s in zip(ids, summaries):
                conn.execute(
                    "UPDATE file_summaries SET summary_text=?, one_line_anchor=?, key_facts=?, "
                    "section_anchors=? WHERE file_id=?",
                    (s.get("summary_text", "") or "", s.get("one_line_anchor", "") or "",
                     json.dumps(s.get("key_facts", {})),
                     json.dumps(s["section_anchors"]) if s.get("section_anchors") else None,
                     fid),
                )
                changed += 1
            conn.commit()
            _free_mlx_cache()   # release the batch's Metal buffers before the next
        done += len(batch)
        if progress:
            progress(done, total)
    logger.info("resummarize: %d/%d files refreshed", changed, total)
    return {"scanned": total, "changed": changed}

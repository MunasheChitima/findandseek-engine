"""Background precomputation of Organize artifacts.

The Organize screen must feel *already done* when the user opens it — the catalog
analysis is a background activity, not an on-click wait. This module builds the
tags + reorganization plan once (after indexing settles) and stores them, so the
UI just loads the latest precomputed plan instantly.

Pure catalog work (SQL + heuristics, no model inference) — safe to run on the
ingest worker's idle tick. Never touches the filesystem.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from find_and_seek.organize import plan_store
from find_and_seek.organize.planner import generate_plan
from find_and_seek.organize.tags import auto_tag_scope

logger = logging.getLogger(__name__)


def refresh_artifacts(
    conn: sqlite3.Connection, strategy: str = "by_type", keep_plans: int = 5
) -> dict[str, Any]:
    """Re-tag the catalog and regenerate the reorganization plan, then prune old
    plans. Returns the new plan_id + counts. Idempotent and cheap to re-run."""
    t0 = time.perf_counter()
    tag_counts = auto_tag_scope(conn)
    result = generate_plan(conn, roots=None, strategy=strategy)
    plan_store.set_status(conn, result["plan_id"], plan_store.PLAN_PREVIEWED)
    conn.commit()
    plan_store.prune_plans(conn, keep=keep_plans)
    dt = time.perf_counter() - t0
    logger.info(
        "Organize artifacts refreshed: plan #%s, %s tags, %.2fs",
        result["plan_id"], tag_counts.get("tags", 0), dt,
    )
    return {"plan_id": result["plan_id"], "tags": tag_counts, "seconds": round(dt, 2)}

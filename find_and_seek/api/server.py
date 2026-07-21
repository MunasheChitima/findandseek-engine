"""FastAPI sidecar — localhost:8775 (DAM uses 8765)."""

from __future__ import annotations

import errno
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from find_and_seek.config.settings import API_HOST, API_PORT
from find_and_seek.db.connection import (
    DEFAULT_DB_PATH,
    get_connection,
    init_db,
    is_corruption_error,
    transaction,
)
from find_and_seek.db.facts_query import aggregate_facts, query_facts
from find_and_seek.db.scope import scope_and_clause, scope_clause
from find_and_seek.db.store import estimated_tokens_for_file, purge_root
from find_and_seek.organize import finder_tags, plan_store, preview as preview_mod
from find_and_seek.organize import settings as organize_settings_mod
from find_and_seek.config import power_settings as power_settings_mod
from find_and_seek.organize.apply import apply_plan
from find_and_seek.organize.planner import generate_plan
from find_and_seek.organize.tags import auto_tag_scope, get_file_tags, list_tag_facets
from find_and_seek.organize.undo import undo_plan
from find_and_seek.search.hybrid import hit_to_dict, search
from find_and_seek.watch.roots import add_root, list_roots, remove_root, seed_roots_from_index
from find_and_seek.watch.scan import scan_roots

import threading

# FastAPI runs sync routes in a threadpool; sharing one SQLite connection across
# threads risks interleaved transactions. Give each worker thread its own
# connection (WAL makes concurrent readers cheap).
_local = threading.local()

# Bumped by /rebuild so every thread drops its cached (possibly corrupt or
# pointing-at-a-deleted-file) connection and reconnects to the fresh DB.
_db_generation = 0
_db_gen_lock = threading.Lock()

# Serializes /rebuild — the unlink -> init_db -> scan_roots sequence is not
# safe under concurrent invocation (two callers can race the file unlink
# against another's init_db, or double-enqueue every watched root).
_rebuild_lock = threading.Lock()


def db() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "gen", -1) != _db_generation:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 — closing a stale/broken handle is best-effort
            pass
        conn = None
    if conn is None:
        conn = get_connection()
        _local.conn = conn
        _local.gen = _db_generation
    return conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    from find_and_seek.config.logging_setup import configure as _log_configure
    from find_and_seek.config.settings import role_backend
    from find_and_seek import diagnostics
    import logging as _logging

    _log_configure()
    _log = _logging.getLogger(__name__)
    _log.info("[api] sidecar starting on %s:%s", API_HOST, API_PORT)

    # Ensure the schema exists once at startup — but a *corrupt* index here must
    # NOT take down the sidecar (F-4.2.3). If we crash on boot, /health and
    # /rebuild are never reachable and the app can only crash-loop. Instead stay
    # up: /health then reports index_corrupt and the user can trigger /rebuild.
    # We don't auto-delete the user's file on boot — recovery is offered, not forced.
    try:
        init_db().close()
    except sqlite3.DatabaseError as exc:
        if not is_corruption_error(exc):
            raise

    # One-time upgrade migration (D32 follow-up): users who upgrade from the old
    # Full-Disk-Access model have an indexed corpus but no roots.json, which would
    # make onboarding gate on "no granted folders" and silently reset their scope.
    # Seed roots from the existing index on the first boot after upgrade. Best-effort
    # — a seeding failure must never block boot — and a no-op once roots.json exists.
    try:
        _seed_conn = get_connection()
        try:
            seeded = seed_roots_from_index(_seed_conn)
        finally:
            _seed_conn.close()
        if seeded:
            _log.info("[api] seeded %d root(s) from existing index (upgrade migration)", len(seeded))
    except Exception:  # noqa: BLE001 — seeding is best-effort; never block startup
        _log.warning("[api] seed_roots_from_index failed; continuing without seeded roots", exc_info=True)
    # Warm the embedding model AND the cross-encoder reranker in the background so
    # the first user query doesn't pay the ~3s MLX cold-load or the ~1-2s ONNX
    # session-load + first-inference cost. Daemon thread → never blocks startup.
    # State is exposed in /health so the app can show "warming up" on the first
    # search instead of an unexplained multi-second pause.
    def _warm() -> None:
        try:
            from find_and_seek.ingest.embed import embed_one

            embed_one("warm up the embedder")
            _WARMUP["embedder"] = "ready"
        except Exception:  # noqa: BLE001 — best-effort warmup
            _WARMUP["embedder"] = "failed"
        try:
            # Load the ONNX session/tokenizer and run one real inference so the
            # first /search doesn't eat the cross-encoder cold start.
            from find_and_seek.search.rerank import rerank

            rerank("warm up the reranker", [(0, "warm up the reranker")])
            _WARMUP["reranker"] = "ready"
        except Exception:  # noqa: BLE001 — best-effort warmup
            _WARMUP["reranker"] = "failed"

    threading.Thread(target=_warm, daemon=True).start()

    summary_backend = role_backend("summary")
    embed_backend = role_backend("embed")
    _log.info("[api] backends: summary=%s embed=%s", summary_backend, embed_backend)
    diagnostics.record("api_start", stage="ready")

    yield

    _log.info("[api] sidecar shutting down")


app = FastAPI(title="Find and Seek", lifespan=lifespan)

# Warmup state, reported in /health: "pending" until the background warm thread
# finishes each stage. The first search is slow while pending — the app shows
# a "warming up" state instead of an unexplained pause.
_WARMUP: dict[str, str] = {"embedder": "pending", "reranker": "pending"}

# MCP client liveness, fed by the beacon's POST /mcp/heartbeat. "Connected"
# in the tray means observed tool calls — not "a config file exists". Persisted
# so the tray still shows last-seen across sidecar restarts.
_MCP_CLIENTS: dict[str, dict[str, Any]] = {}
_MCP_CLIENTS_PATH = Path.home() / ".findandseek" / "mcp_clients.json"
_mcp_clients_lock = threading.Lock()


def _load_mcp_clients() -> None:
    try:
        _MCP_CLIENTS.update(json.loads(_MCP_CLIENTS_PATH.read_text()))
    except Exception:  # noqa: BLE001 — absent/corrupt state starts fresh
        pass


_load_mcp_clients()


@app.exception_handler(sqlite3.DatabaseError)
async def _sqlite_error_handler(request: Request, exc: sqlite3.DatabaseError) -> JSONResponse:
    """A corrupt/unusable index used to escape as an opaque HTTP 500 (F-4.2.3).
    Surface a typed, actionable error instead: corruption is recoverable by a
    lossless rebuild (POST /rebuild), so the app can show that path rather than
    a dead end."""
    if is_corruption_error(exc):
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": {
                    "code": "index_corrupt",
                    "message": "The search index is corrupted. It can be rebuilt from your files with no data loss.",
                    "recoverable": True,
                },
            },
        )
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": {"code": "db_error", "message": str(exc)}},
    )


class SearchRequest(BaseModel):
    query: str
    scope: str = "all"
    limit: int = 10
    type_filter: str | None = None


class FolderRequest(BaseModel):
    path: str
    # On remove, also forget everything indexed under the folder (default). Set
    # true to stop watching but keep the existing index entries searchable.
    keep_index: bool = False


class ScopeRequest(BaseModel):
    # None / empty → every watched root (the default "whatever is indexed" scope).
    roots: list[str] | None = None


class PlanRequest(BaseModel):
    roots: list[str] | None = None
    strategy: str = "by_type"


class DecisionRequest(BaseModel):
    decision: str               # accepted|rejected|edited|pending
    action_ids: list[int] | None = None   # None → all actions in the plan


class CategoryRequest(BaseModel):
    label: str
    definition: str
    excludes: str = ""


class FileCategoryRequest(BaseModel):
    slug: str


def _err(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _reclassify_bg(scopes: tuple[str, ...], limit: int = 500) -> None:
    """Background re-classification — opens its own connection (the request's is
    thread-local). Heavy (loads the classifier model); runs off the request and
    is capped so it can't become a runaway job."""
    from find_and_seek.db.connection import get_connection
    from find_and_seek.organize.reclassify import reclassify
    conn = get_connection()
    try:
        reclassify(conn, scopes=scopes, limit=limit)
    finally:
        conn.close()


@app.get("/health")
def health() -> dict[str, Any]:
    # Health must never itself 500 on corruption — it's exactly how the app polls
    # to discover an unusable index and surface the Rebuild path.
    try:
        indexed = db().execute("SELECT COUNT(*) FROM files WHERE status='indexed'").fetchone()[0]
        pending = db().execute(
            "SELECT COUNT(*) FROM ingest_queue WHERE status IN ('pending','processing')"
        ).fetchone()[0]
    except sqlite3.DatabaseError as exc:
        if is_corruption_error(exc):
            return {"ok": False, "status": "index_corrupt", "files_indexed": 0,
                    "queue_pending": 0, "recoverable": True}
        raise
    status = "indexing" if pending else "up_to_date"
    return {"ok": True, "status": status, "files_indexed": indexed, "queue_pending": pending,
            "warmup": dict(_WARMUP)}


class HeartbeatBody(BaseModel):
    client: str = "unknown"
    event: str = "tool"
    tool: str | None = None
    tokens_est: int = 0
    escalation: bool = False


@app.post("/mcp/heartbeat")
def mcp_heartbeat(body: HeartbeatBody) -> dict[str, Any]:
    """Beacon sink: MCP server processes report spawn + every tool call here.
    Rolling per-client counters give the tray ground truth on which AI clients
    are actually routing through the engine (and the Tier-1 savings meter its
    tokens-served number)."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    with _mcp_clients_lock:
        c = _MCP_CLIENTS.setdefault(
            body.client,
            {"first_seen": now, "tool_calls": 0, "tokens_est": 0, "escalations": 0},
        )
        c["last_seen"] = now
        if body.event == "tool":
            c["tool_calls"] += 1
            c["tokens_est"] += max(0, body.tokens_est)
            if body.escalation:
                c["escalations"] += 1
            c["last_tool"] = body.tool
        try:
            _MCP_CLIENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _MCP_CLIENTS_PATH.write_text(json.dumps(_MCP_CLIENTS))
        except OSError:
            pass
    return {"ok": True}


@app.get("/mcp/clients")
def mcp_clients() -> dict[str, Any]:
    with _mcp_clients_lock:
        return {"ok": True, "clients": json.loads(json.dumps(_MCP_CLIENTS))}


@app.post("/rebuild")
def rebuild_route() -> dict[str, Any]:
    """Lossless index rebuild for a corrupt/unusable DB (F-4.2.3). The source
    files are intact, so we drop the index file (+ WAL/SHM), recreate the schema,
    and re-enqueue every watched root for re-ingestion by the worker."""
    global _db_generation
    if not _rebuild_lock.acquire(blocking=False):
        return {"ok": False, "rebuilt": False, "already_rebuilding": True}
    try:
        base = Path(DEFAULT_DB_PATH)
        for p in (base, Path(f"{base}-wal"), Path(f"{base}-shm")):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        # Invalidate every thread's cached connection before recreating the file.
        with _db_gen_lock:
            _db_generation += 1
        conn = init_db()
        try:
            enqueued = scan_roots(conn)
        finally:
            conn.close()
        return {"ok": True, "rebuilt": True, "enqueued": enqueued}
    finally:
        _rebuild_lock.release()


@app.get("/diagnostics")
def diagnostics_route() -> dict[str, Any]:
    """Local-only diagnostics payload (no file contents, paths, or queries) for
    the in-app 'Export Diagnostics' button — see find_and_seek/diagnostics.py."""
    from find_and_seek import diagnostics

    return {"ok": True, "diagnostics": diagnostics.collect(db())}


@app.get("/diagnostics/export")
def diagnostics_export_route() -> Any:
    """Build and return the full diagnostics bundle as a zip file.

    Reads log files from disk, bundles them with the structured JSON payload,
    and returns the zip bytes. The Swift 'Export Diagnostics' button saves the
    bytes to a user-chosen path — nothing is sent externally.
    """
    from fastapi.responses import Response
    import tempfile
    from find_and_seek import diagnostics

    with tempfile.TemporaryDirectory() as tmp:
        path = diagnostics.export(db(), dest_dir=tmp)
        data = path.read_bytes()

    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )


@app.get("/logs")
def logs_route(tail: int = Query(200, ge=1, le=2000)) -> dict[str, Any]:
    """Tail of the rotating sidecar log + boot log + structured event stream.

    Designed for the in-app live log viewer (Preferences → Diagnostics).
    Returns plain-text lines from findandseek.log and app-boot.log, plus the
    structured metrics.jsonl events. Nothing here contains file contents,
    paths, or query text — callers at the logging sites are responsible for
    that discipline (same rule as diagnostics.py).

    `tail` caps the number of lines returned per source (default 200, max 2000).
    """
    from find_and_seek import diagnostics
    from find_and_seek.config.logging_setup import LOG_FILE, LOG_DIR

    def _tail_file(path: Path, n: int) -> list[str]:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return lines[-n:]
        except OSError:
            return []

    boot_log = LOG_DIR / "app-boot.log"
    return {
        "ok": True,
        "sidecar_log": _tail_file(LOG_FILE, tail),
        "boot_log": _tail_file(boot_log, tail),
        "events": diagnostics.recent_events(tail),
    }


@app.post("/search")
def search_route(body: SearchRequest) -> dict[str, Any]:
    try:
        hits, ms = search(db(), body.query, body.scope, body.limit, body.type_filter)
    except sqlite3.DatabaseError:
        from find_and_seek.db.fts_repair import ensure_fts_healthy
        ensure_fts_healthy(db())
        try:
            hits, ms = search(db(), body.query, body.scope, body.limit, body.type_filter)
        except sqlite3.DatabaseError:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "error": "search_index_rebuilding",
                    "message": "Search index is rebuilding — try again in a moment",
                },
            )
    from find_and_seek import diagnostics
    diagnostics.record("search", count=len(hits), seconds=round(ms / 1000, 3))
    return {"ok": True, "results": [hit_to_dict(h) for h in hits], "latency_ms": round(ms, 2)}


@app.get("/file/{file_id}/summary")
def get_summary(file_id: int) -> dict[str, Any]:
    row = db().execute(
        "SELECT * FROM file_summaries WHERE file_id=?",
        (file_id,),
    ).fetchone()
    if not row:
        return _err("not_found", f"No summary for file {file_id}")
    key_facts = json.loads(row["key_facts"]) if row["key_facts"] else {}
    card = {
        "file_id": file_id,
        "summary_text": row["summary_text"],
        "one_line_anchor": row["one_line_anchor"],
        "document_type": row["document_type"],
        "classification_confidence": row["classification_confidence"],
        "suggested_category": row["suggested_category"],
        "key_facts": key_facts,
        "confidence_note": row["confidence_note"],
    }
    # User corrections outrank the model layer at read time; drift (file changed
    # after the correction) is surfaced, never silently resolved.
    from find_and_seek.db import edits as edits_mod

    card = edits_mod.apply_edits(db(), file_id, card)
    return {"ok": True, "summary": card}


class CardEditBody(BaseModel):
    field: str
    value: str | None = None


@app.get("/file/{file_id}/card_edits")
def get_card_edits(file_id: int) -> dict[str, Any]:
    from find_and_seek.db import edits as edits_mod

    return {"ok": True, "edits": edits_mod.edits_for_file(db(), file_id)}


@app.put("/file/{file_id}/card_edits")
def put_card_edit(file_id: int, body: CardEditBody) -> dict[str, Any]:
    """Record a user correction (value=null deletes the field/fact). The edit
    lives in the overlay table — re-summarise and model upgrades never touch it."""
    from find_and_seek.db import edits as edits_mod

    try:
        edits_mod.set_edit(db(), file_id, body.field, body.value)
    except ValueError as e:
        return _err("bad_field", str(e))
    return {"ok": True}


@app.delete("/file/{file_id}/card_edits/{field:path}")
def delete_card_edit(file_id: int, field: str) -> dict[str, Any]:
    from find_and_seek.db import edits as edits_mod

    edits_mod.clear_edit(db(), file_id, field)
    return {"ok": True}


@app.get("/facts")
def get_facts(
    fact_type: str | None = None,
    key: str | None = None,
    min_number: float | None = None,
    max_number: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    contains: str | None = None,
    scope: str = "all",
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    facts = query_facts(
        db(),
        fact_type=fact_type,
        key=key,
        min_number=min_number,
        max_number=max_number,
        start_date=start_date,
        end_date=end_date,
        contains=contains,
        scope=scope,
        limit=limit,
    )
    return {"ok": True, "facts": facts}


@app.get("/facts/aggregate")
def get_facts_aggregate(
    op: str = "count",
    fact_type: str | None = None,
    key: str | None = None,
    min_number: float | None = None,
    max_number: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    scope: str = "all",
) -> dict[str, Any]:
    return {
        "ok": True,
        "result": aggregate_facts(
            db(),
            op=op,
            fact_type=fact_type,
            key=key,
            min_number=min_number,
            max_number=max_number,
            start_date=start_date,
            end_date=end_date,
            scope=scope,
        ),
    }


@app.get("/chunk/{chunk_id}")
def get_chunk(chunk_id: int, neighbours: int = Query(0, ge=0, le=5)) -> dict[str, Any]:
    row = db().execute(
        "SELECT id, file_id, text, location_ref, source_type, chunk_index FROM file_chunks WHERE id=?",
        (chunk_id,),
    ).fetchone()
    if not row:
        return _err("not_found", f"Chunk {chunk_id} not found")

    chunk = {
        "id": row["id"],
        "file_id": row["file_id"],
        "text": row["text"],
        "location_ref": row["location_ref"],
        "source_type": row["source_type"],
    }
    neighbour_rows: list[dict] = []
    if neighbours:
        for offset in range(-neighbours, neighbours + 1):
            if offset == 0:
                continue
            n = db().execute(
                """
                SELECT id, file_id, text, location_ref, source_type
                FROM file_chunks
                WHERE file_id=? AND chunk_index=?
                """,
                (row["file_id"], row["chunk_index"] + offset),
            ).fetchone()
            if n:
                neighbour_rows.append(
                    {
                        "id": n["id"],
                        "file_id": n["file_id"],
                        "text": n["text"],
                        "location_ref": n["location_ref"],
                        "source_type": n["source_type"],
                    }
                )
    return {"ok": True, "chunk": chunk, "neighbours": neighbour_rows}


@app.get("/entity")
def find_entity(
    type: str = Query(..., alias="type"),
    value: str = Query(...),
    scope: str = "all",
) -> dict[str, Any]:
    norm = value.strip().lower()
    params: list[Any] = [type, norm]
    scope_sql, scope_params = scope_and_clause(scope)
    params.extend(scope_params)

    rows = db().execute(
        f"""
        SELECT f.id AS file_id, f.filename, fs.one_line_anchor, COUNT(*) AS occurrences
        FROM file_entities e
        JOIN files f ON f.id = e.file_id
        LEFT JOIN file_summaries fs ON fs.file_id = f.id
        WHERE e.entity_type=? AND LOWER(e.entity_value)=?{scope_sql}
        GROUP BY f.id
        ORDER BY occurrences DESC
        """,
        params,
    ).fetchall()
    return {
        "ok": True,
        "files": [
            {
                "file_id": r["file_id"],
                "filename": r["filename"],
                "one_line_anchor": r["one_line_anchor"],
                "occurrences": r["occurrences"],
            }
            for r in rows
        ],
    }


@app.get("/recent")
def list_recent(
    days: int = 7,
    type_filter: str | None = None,
    scope: str = "all",
) -> dict[str, Any]:
    params: list[Any] = [f"-{days} days"]
    scope_sql, scope_params = scope_and_clause(scope)
    params.extend(scope_params)
    type_sql = ""
    if type_filter:
        type_sql = " AND fs.document_type = ?"
        params.append(type_filter)

    rows = db().execute(
        f"""
        SELECT f.id AS file_id, f.filename, fs.one_line_anchor, fs.document_type,
               fs.classification_confidence, f.modified_at
        FROM files f
        LEFT JOIN file_summaries fs ON fs.file_id = f.id
        WHERE f.modified_at >= datetime('now', ?){scope_sql}{type_sql}
        ORDER BY f.modified_at DESC
        LIMIT 50
        """,
        params,
    ).fetchall()
    return {
        "ok": True,
        "files": [
            {
                "file_id": r["file_id"],
                "filename": r["filename"],
                "one_line_anchor": r["one_line_anchor"],
                "document_type": r["document_type"],
                "classification_confidence": r["classification_confidence"],
                "modified_at": (r["modified_at"] or "")[:10],
            }
            for r in rows
        ],
    }


@app.get("/browse")
def browse(
    type: str | None = None,
    scope: str = "all",
    limit: int = Query(150, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """List indexed files (newest first), optionally filtered to a document type
    (e.g. ?type=invoice) or a folder path prefix (?scope=/Users/…/Downloads).

    Keep this query cheap: no per-row correlated subqueries on ``facts`` /
    ``file_chunks``. Those made every folder click take ~20s on large catalogues.
    Row preview uses ``one_line_anchor``; full chunk text loads on demand in the
    preview pane.
    """
    where = ["f.status='indexed'"]
    params: list[Any] = []
    if type:
        where.append("fs.document_type = ?")
        params.append(type)
    scope_sql, scope_params = scope_clause(scope)
    if scope_sql:
        where.append(scope_sql)
        params.extend(scope_params)
    params += [limit, offset]
    rows = db().execute(
        f"""
        SELECT f.id AS file_id, f.filename, f.path, f.modified_at,
               fs.document_type, fs.classification_confidence, fs.one_line_anchor
        FROM files f
        LEFT JOIN file_summaries fs ON fs.file_id = f.id
        WHERE {' AND '.join(where)}
        ORDER BY f.modified_at DESC
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()
    return {
        "ok": True,
        "files": [
            {
                "file_id": r["file_id"],
                "filename": r["filename"],
                "path": r["path"],
                "document_type": r["document_type"],
                "classification_confidence": r["classification_confidence"],
                "one_line_anchor": r["one_line_anchor"],
                "modified_at": (r["modified_at"] or "")[:10] if r["modified_at"] else None,
                "headline_fact": None,
                # Browse list preview — anchor only. Opening chunk text is loaded
                # when the user opens a file's preview pane.
                "opening_text": (r["one_line_anchor"] or "")[:300],
            }
            for r in rows
        ],
    }


def _scan_folder(path: str) -> None:
    """Backfill an added folder in the background (own connection for the thread)."""
    conn = get_connection()
    try:
        scan_roots(conn, [Path(path)])
    finally:
        conn.close()


def _root_needs_permission(path: Path) -> bool:
    """True if the engine is blocked from READING a granted folder — a macOS TCC
    denial of the background sidecar. The user added the folder in onboarding, but
    the OS hasn't authorized the launchd-run engine to read it, so it would index
    nothing and silently show 0 files. Surfacing this lets the UI prompt the user
    to allow access instead of looking broken.

    The probe runs in the api process, which is the SAME signed sidecar binary as
    the worker (D28 consolidated entry) — so its read access reflects the worker's.
    A bare directory listing is enough: TCC denies enumerating a protected folder's
    contents with EPERM/EACCES even though stat() on the folder itself succeeds.
    """
    try:
        with os.scandir(path) as it:
            next(it, None)
    except PermissionError:
        return True
    except OSError as e:  # FileNotFoundError (ENOENT) etc. → not a permission issue
        return e.errno in (errno.EPERM, errno.EACCES)
    return False


@app.get("/folders")
def get_folders() -> dict[str, Any]:
    folders = []
    for r in list_roots():
        prefix = f"{r}/%"
        indexed = db().execute(
            "SELECT COUNT(*) FROM files WHERE status='indexed' AND path LIKE ?",
            (prefix,),
        ).fetchone()[0]
        # pending + processing = all unfinished work (processing rows are stuck
        # if no worker is running; requeue_stale_processing resets them on startup).
        queue_row = db().execute(
            "SELECT COUNT(*) FILTER (WHERE status='pending'),"
            "       COUNT(*) FILTER (WHERE status='processing')"
            " FROM ingest_queue WHERE path LIKE ?",
            (prefix,),
        ).fetchone()
        pending, processing = queue_row[0], queue_row[1]
        unfinished = pending + processing
        # Show the file at the front of the work queue (pending or processing).
        current_row = db().execute(
            "SELECT path FROM ingest_queue"
            " WHERE status IN ('processing','pending') AND path LIKE ?"
            " ORDER BY CASE status WHEN 'processing' THEN 0 ELSE 1 END, priority DESC, recency DESC"
            " LIMIT 1",
            (prefix,),
        ).fetchone()
        current_file = Path(current_row[0]).name if current_row else None
        folders.append({
            "path": str(r),
            "exists": Path(r).is_dir(),
            "files_indexed": indexed,
            "files_pending": unfinished,
            "files_total": indexed + unfinished,
            "current_file": current_file,
            # macOS may have granted the folder in onboarding but not authorized
            # the background engine to read it (TCC). Flag it so the UI can prompt
            # rather than silently showing an empty folder.
            "needs_permission": _root_needs_permission(Path(r)),
        })
    return {"ok": True, "folders": folders}


@app.post("/folders/resync")
def resync_folder(body: FolderRequest, background: BackgroundTasks) -> dict[str, Any]:
    """Re-scan a watched folder and enqueue any new or changed files."""
    p = Path(body.path).expanduser()
    if not p.is_dir():
        return _err("not_found", f"Not a directory: {p}")
    if p not in list_roots():
        return _err("not_watched", f"Not a watched folder: {p}")
    background.add_task(_scan_folder, str(p))
    return {"ok": True, "folder": str(p), "status": "scanning"}


@app.post("/folders")
def add_folder(body: FolderRequest, background: BackgroundTasks) -> dict[str, Any]:
    p = Path(body.path).expanduser()
    if not p.is_dir():
        return _err("not_found", f"Not a directory: {p}")
    add_root(p)
    background.add_task(_scan_folder, str(p))  # backfill without blocking the response
    return {"ok": True, "folder": str(p), "status": "scanning", "note": "watcher picks it up within ~5s"}


@app.delete("/folders")
def delete_folder(body: FolderRequest) -> dict[str, Any]:
    p = Path(body.path).expanduser()
    remove_root(p)
    # Revoking access revokes searchability: purge everything indexed under the
    # folder unless the caller explicitly opts to keep it. Own connection +
    # transaction so the multi-table purge is atomic on the request thread.
    purged = 0
    if not body.keep_index:
        conn = get_connection()
        try:
            with transaction(conn):
                purged = purge_root(conn, p)
        finally:
            conn.close()
    return {"ok": True, "removed": str(p), "purged_files": purged, "kept_index": body.keep_index}


# ── Organize: tagging (Phase 0) + reorganize (Phase 1 propose/preview) ──
# Tagging writes only `tags`/`file_tags`; planning/preview write only plan rows —
# none of the routes below touch the filesystem. The filesystem-mutating Apply +
# Undo (Phase 2) live further down, separated out deliberately.


@app.get("/tags")
def get_tags(kind: str | None = None) -> dict[str, Any]:
    # Type facets are the single source of truth with the row badges + the
    # /browse filter: derive them from file_summaries.document_type (what the
    # badges show), NOT the tags/file_tags tables (the Organize tagging step,
    # which is partial/stale and drifts apart). This keeps the rail honest —
    # including a navigable `needs-review` bucket — instead of mirroring tags.
    if kind == "type":
        return {"ok": True, "tags": _document_type_facets(db())}
    return {"ok": True, "tags": list_tag_facets(db(), kind)}


def _document_type_facets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Type facets sourced from file_summaries.document_type (the badge field).

    Shape matches what the Swift sidebar decodes (`type:`-prefixed `name` +
    `file_count`); its `slug` strips the prefix and feeds it back as the
    /browse `?type=` filter, which also matches on document_type."""
    rows = conn.execute(
        """
        SELECT fs.document_type AS dtype, COUNT(*) AS file_count
        FROM files f
        JOIN file_summaries fs ON fs.file_id = f.id
        WHERE f.status='indexed' AND fs.document_type IS NOT NULL
              AND fs.document_type != ''
        GROUP BY fs.document_type
        ORDER BY file_count DESC, fs.document_type
        """
    ).fetchall()
    return [{"name": f"type:{r['dtype']}", "file_count": r["file_count"]} for r in rows]


@app.get("/file/{file_id}/tags")
def file_tags(file_id: int) -> dict[str, Any]:
    return {"ok": True, "file_id": file_id, "tags": get_file_tags(db(), file_id)}


# ── Classification taxonomy (user-extensible) ────────────────────────
@app.get("/categories")
def list_categories() -> dict[str, Any]:
    from find_and_seek.organize.categories import load_taxonomy
    cats = [{"slug": c.slug, "label": c.label, "definition": c.definition, "builtin": c.builtin}
            for c in load_taxonomy(db())]
    return {"ok": True, "categories": cats}


@app.get("/categories/suggestions")
def category_suggestions(min_count: int = Query(2, ge=1, le=50)) -> dict[str, Any]:
    """Emergent categories: cluster the classifier's own `suggested_category` across
    the docs it COULDN'T confidently place (needs-review / low / none), and surface
    the recurring ones as "you have N documents that look like X — add this type?".
    The low-confidence pile is the signal for which categories this corpus needs."""
    rows = db().execute(
        """
        SELECT LOWER(TRIM(suggested_category)) AS label,
               COUNT(*) AS n,
               GROUP_CONCAT(file_id) AS file_ids
        FROM file_summaries
        WHERE suggested_category IS NOT NULL AND TRIM(suggested_category) <> ''
          AND (document_type = 'needs-review'
               OR classification_confidence IN ('low','none')
               OR document_type IS NULL)
        GROUP BY label
        HAVING n >= ?
        ORDER BY n DESC, label
        """,
        (min_count,),
    ).fetchall()
    # Don't offer something we already have as a category.
    from find_and_seek.organize.categories import load_taxonomy
    have = {c.label.lower() for c in load_taxonomy(db())} | {c.slug for c in load_taxonomy(db())}
    out = [
        {
            "label": r["label"],
            "count": r["n"],
            "file_ids": [int(x) for x in (r["file_ids"] or "").split(",") if x],
        }
        for r in rows
        if r["label"] not in have
    ]
    return {"ok": True, "suggestions": out}


@app.post("/categories")
def add_category(body: CategoryRequest, background: BackgroundTasks) -> dict[str, Any]:
    """Add a user category, then re-engage: re-classify the unsorted pile against
    the expanded taxonomy in the background (so the new category actually fills)."""
    from find_and_seek.organize.reclassify import add_category as _add
    try:
        slug = _add(db(), body.label, body.definition, body.excludes)
    except ValueError as e:
        return _err("bad_request", str(e))
    # Re-engage from the genuinely-unsorted pile only (bounded). A full re-type of
    # the 'other' bucket is the explicit /reclassify (or a re-index).
    background.add_task(_reclassify_bg, ("needs-review",))
    return {"ok": True, "slug": slug, "reclassifying": True}


@app.post("/file/{file_id}/category")
def correct_category(file_id: int, body: FileCategoryRequest) -> dict[str, Any]:
    from find_and_seek.organize.reclassify import set_file_category
    ok = set_file_category(db(), file_id, body.slug)
    return {"ok": ok} if ok else _err("bad_request", f"unknown category {body.slug!r}")


@app.post("/reclassify")
def reclassify_route(background: BackgroundTasks) -> dict[str, Any]:
    """Re-type the unsorted/other pile against the current taxonomy (background)."""
    from find_and_seek.organize.reclassify import RECLASSIFY_SCOPES
    background.add_task(_reclassify_bg, RECLASSIFY_SCOPES)
    return {"ok": True, "queued": True}


@app.post("/organize/autotag")
def organize_autotag(body: ScopeRequest) -> dict[str, Any]:
    # Synchronous: pure SQL over the catalog, no model inference. Writes native
    # tags only — never the filesystem.
    counts = auto_tag_scope(db(), body.roots)
    return {"ok": True, **counts}


@app.get("/organize/latest")
def organize_latest() -> dict[str, Any]:
    """Load the most recent *precomputed* plan's preview — instant, no generation.
    The worker refreshes this in the background after indexing, so the UI never
    waits on analysis. Returns preview=None if nothing has been computed yet."""
    pid = plan_store.latest_plan_id(db())
    if pid is None:
        return {"ok": True, "preview": None}
    return {"ok": True, "preview": preview_mod.build_preview(db(), pid, diff_limit=150, include_trees=False)}


@app.post("/organize/refresh")
def organize_refresh(body: PlanRequest) -> dict[str, Any]:
    """Force a re-analysis now (the 'Re-analyze' button). Normally unnecessary —
    the background precompute keeps the latest plan fresh."""
    from find_and_seek.organize.precompute import refresh_artifacts

    result = refresh_artifacts(db(), strategy=body.strategy or "by_type")
    return {"ok": True, "preview": preview_mod.build_preview(db(), result["plan_id"], diff_limit=150, include_trees=False)}


@app.post("/organize/plan")
def organize_plan(body: PlanRequest) -> dict[str, Any]:
    result = generate_plan(db(), body.roots, body.strategy)
    return {"ok": True, **result}


@app.get("/organize/plans")
def organize_plans() -> dict[str, Any]:
    return {"ok": True, "plans": plan_store.list_plans(db())}


@app.get("/organize/plan/{plan_id}/preview")
def organize_preview(plan_id: int) -> dict[str, Any]:
    view = preview_mod.build_preview(db(), plan_id)
    if view is None:
        return _err("not_found", f"No plan {plan_id}")
    return {"ok": True, "preview": view}


@app.post("/organize/plan/{plan_id}/decision")
def organize_decision(plan_id: int, body: DecisionRequest) -> dict[str, Any]:
    if not plan_store.get_plan(db(), plan_id):
        return _err("not_found", f"No plan {plan_id}")
    try:
        updated = plan_store.set_decision(db(), plan_id, body.decision, body.action_ids)
    except ValueError as e:
        return _err("bad_request", str(e))
    return {"ok": True, "updated": updated}


# ── Organize Phase 2: transactional Apply + Undo (filesystem-mutating) ──
# These are the only routes that move/rename/quarantine files. They act solely
# on actions the user marked accepted/edited, journal every op before it runs,
# and are fully reversible via /undo.


@app.post("/organize/plan/{plan_id}/apply")
def organize_apply(plan_id: int) -> dict[str, Any]:
    result = apply_plan(db(), plan_id)
    if result is None:
        return _err("not_found", f"No plan {plan_id}")
    return {"ok": True, **result}


@app.post("/organize/plan/{plan_id}/undo")
def organize_undo(plan_id: int) -> dict[str, Any]:
    result = undo_plan(db(), plan_id)
    if result is None:
        return _err("not_found", f"No plan {plan_id}")
    return {"ok": True, **result}


# ── Organize: opt-in Finder-tag sync (reversible, journalled) ──────────


class FinderSyncRequest(BaseModel):
    enable: bool = True
    kinds: list[str] | None = None       # None → settings default
    roots: list[str] | None = None       # None → all watched roots


@app.get("/organize/settings")
def organize_settings() -> dict[str, Any]:
    return {"ok": True, "settings": organize_settings_mod.get_settings()}


@app.post("/organize/finder-sync")
def organize_finder_sync(body: FinderSyncRequest, background: BackgroundTasks) -> dict[str, Any]:
    """Flip the opt-in Finder-sync flag and (on enable) run a one-time backfill as
    a reversible plan, so the whole sync can be undone from History."""
    settings = organize_settings_mod.get_settings()
    kinds = tuple(body.kinds or settings.get("synced_kinds") or finder_tags.DEFAULT_KINDS)
    organize_settings_mod.update_settings(finder_sync=body.enable, synced_kinds=list(kinds))
    if not body.enable:
        return {"ok": True, "enabled": False}
    plan_id = finder_tags.build_finder_sync_plan(db(), body.roots, kinds)
    result = apply_plan(db(), plan_id)
    return {"ok": True, "enabled": True, "plan_id": plan_id, **(result or {})}


class PowerSettingsRequest(BaseModel):
    mode: str  # "smooth" | "full" (anything else clamps to smooth)


@app.get("/settings/power")
def get_power_settings() -> dict[str, Any]:
    """Current performance mode. 'smooth' (default) sips power and keeps the fans
    quiet; 'full' blitzes when the user opts in."""
    return {"ok": True, "settings": power_settings_mod.get_settings()}


@app.post("/settings/power")
def set_power_settings(body: PowerSettingsRequest) -> dict[str, Any]:
    """Set the performance mode. Takes effect on the worker's next batch — no
    restart needed (the worker hot-reads the setting each cycle)."""
    return {"ok": True, "settings": power_settings_mod.set_mode(body.mode)}


def main() -> None:
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")


if __name__ == "__main__":
    main()

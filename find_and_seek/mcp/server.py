"""MCP server — localhost:8776, wraps search API logic."""

from __future__ import annotations

import os

# Runtime offline hardening (agent feedback: zero-egress product should never
# touch the network at query time). Models are fetched at setup; runtime is
# boringly offline. Override with HF_HUB_OFFLINE=0 if you must fetch live.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Recall floor for the agent surface (this server IS a generation-agent client —
# Claude Desktop / IDEs drive it). When the precision gate returns zero hits on a
# vocabulary-gap query, surface a small ranked shortlist of the best sub-threshold
# candidates instead of nothing, so the agent doesn't re-search — the dominant
# hidden token cost. Mirrors the facade MCP server; must be set before
# `search.hybrid` is imported (it reads SEARCH_MIN_RESULTS at module load).
# Confident hits are untouched; a query that genuinely matches nothing still
# returns empty. Override with FINDANDSEEK_SEARCH_MIN_RESULTS=0 to restore the
# strict precision-first behaviour.
os.environ.setdefault("FINDANDSEEK_SEARCH_MIN_RESULTS", "3")

import json
import re
import sqlite3
from typing import Any

from mcp.server.fastmcp import FastMCP

from find_and_seek.config.settings import MCP_HOST, MCP_PORT
from find_and_seek.db.connection import init_db
from find_and_seek.db.facts_query import aggregate_facts, query_facts
from find_and_seek.db.scope import scope_and_clause
from find_and_seek.mcp.beacon import tracked
from find_and_seek.search.hybrid import hit_to_dict, search

# propose_organize_plan persists a draft plan per call and an agent can loop it,
# so we keep only the newest few drafts. Enough that a user can compare a couple
# of proposals in the app; bounded so the index can't grow without limit.
_MAX_AGENT_DRAFT_PLANS = 3

# The usage doctrine rides IN the server (returned to every client at
# initialize) so it needs no user setup and updates with the sidecar. Its
# fail-loud counterpart — "if these tools are ABSENT, stop and say so" — cannot
# live here (a server that failed to load carries nothing) and is installed
# client-side by Connect to AI.
INSTRUCTIONS = (
    "Find and Seek is this machine's local document intelligence engine. "
    "ALWAYS search with search_files first and work from the returned triage "
    "cards (anchor, summary, key facts, match_confidence) — they are usually "
    "sufficient and keep your context small. Escalate to get_chunk or "
    "get_file_context only when a card is genuinely insufficient for the task. "
    "Cite the file path (and location where given) for any fact you use. "
    "Trust signals are calibrated: match_confidence and "
    "classification_confidence are set by code, confidence_note lists which "
    "facts were verified against the source, 'degraded' marks a fallback card, "
    "and anything under user_verified was corrected by the user and outranks "
    "model output."
)

mcp = FastMCP("find-and-seek", host=MCP_HOST, port=MCP_PORT, instructions=INSTRUCTIONS)
DB: sqlite3.Connection | None = None


def _get_db() -> sqlite3.Connection:
    global DB
    if DB is None:
        DB = init_db()
    return DB


def _hashes_for(conn: sqlite3.Connection, file_ids: list[int]) -> dict[int, str]:
    if not file_ids:
        return {}
    ph = ",".join("?" for _ in file_ids)
    return {
        int(r["id"]): r["content_hash"]
        for r in conn.execute(f"SELECT id, content_hash FROM files WHERE id IN ({ph})", file_ids)
    }


def _duplicate_paths(conn: sqlite3.Connection, key: str, exclude_path: str, cap: int = 10) -> list[str]:
    """Other indexed paths holding identical content (same content_hash)."""
    if key.startswith("id:"):  # no usable hash → can't know about copies
        return []
    rows = conn.execute(
        "SELECT path FROM files WHERE content_hash=? AND path<>? LIMIT ?",
        (key, exclude_path, cap),
    ).fetchall()
    return [r["path"] for r in rows]


def _parse_location(ref: str | None) -> dict[str, Any] | None:
    """Best-effort structured form of the free-text location_ref.

    location_ref is denormalised at ingest in several string shapes
    ("page 26 image 3", "page 1 – page 43", "slide 5", "document"). Agents asked
    for a consistent object; we parse at query time without touching the index.
    """
    if not ref:
        return None
    s = ref.strip().lower()
    nums = re.findall(r"\d+", s)
    if "slide" in s:
        kind = "slide"
    elif ("–" in s or "-" in s) and "page" in s and len(nums) >= 2:
        kind = "page_range"
    elif "page" in s:
        kind = "page"
    else:
        kind = "document"
    out: dict[str, Any] = {"kind": kind, "raw": ref}
    if kind in ("page", "slide") and nums:
        out["number"] = int(nums[0])
        if "image" in s and len(nums) >= 2:
            out["image"] = int(nums[1])
    elif kind == "page_range" and len(nums) >= 2:
        out["start"], out["end"] = int(nums[0]), int(nums[1])
    return out


def _enrich(d: dict[str, Any]) -> dict[str, Any]:
    """Add a structured `top_chunk_location` alongside the raw string."""
    loc = _parse_location(d.get("top_chunk_location"))
    if loc:
        d["top_chunk_location_parsed"] = loc
    return d


@mcp.tool()
@tracked
def search_files(
    query: str,
    scope: str = "all",
    limit: int = 5,
    offset: int = 0,
    type_filter: str | None = None,
    group_by_file: bool = True,
) -> dict[str, Any]:
    """Semantic + keyword hybrid search over indexed files.

    scope: "all", or a path PREFIX to restrict results (e.g.
        "/Users/me/Documents/Work"). type_filter: a document_type to filter by
        (see index_status -> document_types for valid values, e.g. "invoice",
        "contract", "report"). offset: skip N results for pagination (step through
        large result sets). group_by_file (default true): collapse to one hit per
        UNIQUE CONTENT, exposing other copies' paths in `duplicate_paths`; set
        false to allow multiple chunks of the same file.

    Each hit carries:
      relevance: blended score (0.65·sigmoid(cross-encoder) + 0.35·fused-norm).
        Meaningful range ~0.35–0.90; hits below 0.40 are filtered unless the
        query's own words appear in the text (lexical grounding) — those return
        as "weak". Use `match_confidence` rather than raw relevance for branching.
      match_confidence: "strong" (≥0.55, CE clearly fired) | "moderate" (above gate,
        likely CE-supported) | "weak" (lexical rescue only). Parallels
        classification_confidence — acts as the trust signal for this result.
    """
    conn = _get_db()
    want = offset + limit
    if group_by_file:
        # Over-fetch then collapse by CONTENT HASH (not just file_id) so neither
        # several chunks of one file NOR the same PDF stored in multiple folders
        # eats the result window. The collapsed copies are surfaced as
        # `duplicate_paths` so the agent still knows they exist (agent feedback).
        hits, _ = search(conn, query, scope, max(want * 4, want), type_filter)
        hash_by_id = _hashes_for(conn, [h.file_id for h in hits])
        seen: set[str] = set()
        deduped = []
        for h in hits:
            key = hash_by_id.get(h.file_id) or f"id:{h.file_id}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append((h, key))
            if len(deduped) >= want:
                break
        page = deduped[offset:want]
        results = []
        for h, key in page:
            d = _enrich(hit_to_dict(h))
            dupes = _duplicate_paths(conn, key, exclude_path=h.path)
            if dupes:
                d["duplicate_paths"] = dupes
                d["duplicate_count"] = len(dupes)
            results.append(d)
    else:
        hits, _ = search(conn, query, scope, want, type_filter)
        results = [_enrich(hit_to_dict(h)) for h in hits[offset:want]]
    return {"results": results, "count": len(results), "offset": offset}


@mcp.tool()
@tracked
def get_summary(file_id: int | None = None, path: str | None = None) -> dict[str, Any]:
    """Return stored summary and key_facts for a file (by file_id OR path).
    Errors if not found / not indexed.
    """
    conn = _get_db()
    if file_id is None and path is None:
        raise ValueError("provide file_id or path")
    if file_id is None:
        r = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
        if not r:
            raise ValueError(f"not indexed: {path!r} — try search_files, or see index_status for coverage")
        file_id = int(r["id"])
    row = conn.execute("SELECT * FROM file_summaries WHERE file_id=?", (file_id,)).fetchone()
    if not row:
        # Raise so MCP marks isError=true — agents branch on that, not on parsing
        # an {"error": ...} body (agent feedback).
        raise ValueError(f"no summary for file_id={file_id}")
    return {
        "file_id": file_id,
        "summary_text": row["summary_text"],
        "one_line_anchor": row["one_line_anchor"],
        "document_type": row["document_type"],
        # How sure the classifier was about document_type — agents should weight a
        # 'low'/'none' type accordingly rather than acting on it as ground truth.
        "classification_confidence": row["classification_confidence"],
        # The model's own name for this kind of doc — a hint when document_type is
        # 'needs-review'/low. A side-channel suggestion, NOT a stable type to act on.
        "suggested_type": row["suggested_category"],
        "key_facts": json.loads(row["key_facts"]) if row["key_facts"] else {},
        "confidence_note": row["confidence_note"],
    }


@mcp.tool()
@tracked
def find_entity(entity_type: str, value: str, scope: str = "all") -> dict[str, Any]:
    """Known-item lookup against extracted entities.

    Tries exact match first (LOWER(entity_value) = query), then falls back to
    token-containment match (LOWER(entity_value) LIKE %query%). Returns
    `match_type` ("exact" | "partial" | null) so the caller knows what fired.

    The entity layer is NER/LLM-derived and not exhaustive. When the result is
    empty, an `entity_layer_note` explains this and suggests search_files as a
    more complete fallback. An empty `files` list means no entity was found by
    this lookup — not that the entity is absent from the corpus.
    """
    conn = _get_db()
    norm = value.strip().lower()

    scope_sql, scope_params = scope_and_clause(scope)

    _ENTITY_SELECT = """
        SELECT f.id AS file_id, f.path, f.filename, fs.one_line_anchor,
               COUNT(*) AS occurrences
        FROM file_entities e
        JOIN files f ON f.id = e.file_id
        LEFT JOIN file_summaries fs ON fs.file_id = f.id
        WHERE e.entity_type=? AND {value_clause}{scope_sql}
        GROUP BY f.id
        ORDER BY occurrences DESC
    """

    # Tier 1 — exact match.
    rows = conn.execute(
        _ENTITY_SELECT.format(value_clause="LOWER(e.entity_value)=?", scope_sql=scope_sql),
        [entity_type, norm, *scope_params],
    ).fetchall()
    match_type: str | None = "exact" if rows else None

    # Tier 2 — token-containment (LIKE) fallback.
    if not rows:
        rows = conn.execute(
            _ENTITY_SELECT.format(value_clause="LOWER(e.entity_value) LIKE ?", scope_sql=scope_sql),
            [entity_type, f"%{norm}%", *scope_params],
        ).fetchall()
        match_type = "partial" if rows else None

    files = [
        {
            "file_id": r["file_id"],
            "path": r["path"],
            "filename": r["filename"],
            "one_line_anchor": r["one_line_anchor"],
            "occurrences": r["occurrences"],
            "match_type": match_type,
        }
        for r in rows
    ]

    result: dict[str, Any] = {"files": files, "match_type": match_type}
    if not files:
        result["entity_layer_note"] = (
            "No entity found via exact or partial lookup. "
            "The entity layer is NER/LLM-derived and may not cover all named items. "
            "Use search_files for full-text coverage, or check index_status for entity_types."
        )
    return result


@mcp.tool()
@tracked
def get_chunk(chunk_id: int, neighbours: int = 0) -> dict[str, Any]:
    """Return one chunk's full text, optionally with adjacent chunks."""
    conn = _get_db()
    row = conn.execute(
        "SELECT id, file_id, text, location_ref, source_type, chunk_index FROM file_chunks WHERE id=?",
        (chunk_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"no chunk with id={chunk_id}")
    chunk = {
        "id": row["id"],
        "file_id": row["file_id"],
        "text": row["text"],
        "location_ref": row["location_ref"],
        "location": _parse_location(row["location_ref"]),
        "source_type": row["source_type"],
    }
    neighbour_rows = []
    for offset in range(-neighbours, neighbours + 1):
        if offset == 0:
            continue
        n = conn.execute(
            "SELECT id, file_id, text, location_ref, source_type FROM file_chunks WHERE file_id=? AND chunk_index=?",
            (row["file_id"], row["chunk_index"] + offset),
        ).fetchone()
        if n:
            neighbour_rows.append(dict(n))
    return {"chunk": chunk, "neighbours": neighbour_rows}


@mcp.tool()
@tracked
def list_recent(days: int = 7, type_filter: str | None = None, scope: str = "all") -> dict[str, Any]:
    """List recently modified indexed files."""
    conn = _get_db()
    params: list[Any] = [f"-{days} days"]
    scope_sql, scope_params = scope_and_clause(scope)
    params.extend(scope_params)
    type_sql = ""
    if type_filter:
        type_sql = " AND fs.document_type = ?"
        params.append(type_filter)
    rows = conn.execute(
        f"""
        SELECT f.id AS file_id, f.filename, fs.one_line_anchor, fs.document_type,
               fs.classification_confidence, f.modified_at
        FROM files f
        LEFT JOIN file_summaries fs ON fs.file_id = f.id
        WHERE f.modified_at >= datetime('now', ?){scope_sql}{type_sql}
        ORDER BY f.modified_at DESC LIMIT 50
        """,
        params,
    ).fetchall()
    files = [
        {
            "file_id": r["file_id"],
            "filename": r["filename"],
            "one_line_anchor": r["one_line_anchor"],
            "document_type": r["document_type"],
            "classification_confidence": r["classification_confidence"],
            "modified_at": (r["modified_at"] or "")[:10],
        }
        for r in rows
    ]
    result: dict[str, Any] = {"files": files}
    # When empty, tell the agent what the most-recent indexed modification actually
    # was so it can answer "nothing changed since February" rather than "not found".
    if not files:
        fresh = conn.execute("SELECT MAX(modified_at) AS m FROM files").fetchone()
        result["newest_file_modified_at"] = (fresh["m"] or "")[:10] if fresh else None
        result["note"] = (
            f"No files modified in the last {days} days. "
            f"Newest indexed modification: {result['newest_file_modified_at'] or 'unknown'}."
        )
    return result


@mcp.tool()
@tracked
def query_typed_facts(
    fact_type: str | None = None,
    key: str | None = None,
    min_number: float | None = None,
    max_number: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    contains: str | None = None,
    scope: str = "all",
    limit: int = 25,
    offset: int = 0,
) -> dict[str, Any]:
    """Query normalized facts (money/date/quantity/person/org/…).

    Examples: invoices over an amount (fact_type='money', min_number=500);
    documents in a date range (fact_type='date', start_date='2025-01-01',
    end_date='2025-12-31'). offset paginates large result sets. Each result cites
    its file and location_ref.
    """
    conn = _get_db()
    facts = query_facts(
        conn,
        fact_type=fact_type,
        key=key,
        min_number=min_number,
        max_number=max_number,
        start_date=start_date,
        end_date=end_date,
        contains=contains,
        scope=scope,
        limit=limit,
        offset=offset,
    )
    total = _total_facts(conn)
    # Distinguish "no matching facts" from "facts layer not populated for this
    # index" so an agent can explain uncertainty accurately (agent feedback).
    return {
        "facts": facts,
        "match_count": len(facts),
        "facts_available": total > 0,
        "total_facts_indexed": total,
    }


@mcp.tool()
@tracked
def aggregate_typed_facts(
    op: str = "count",
    fact_type: str | None = None,
    key: str | None = None,
    min_number: float | None = None,
    max_number: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    scope: str = "all",
) -> dict[str, Any]:
    """Aggregate facts: count / sum / avg / min / max of the numeric value.

    Example: total of all money facts (op='sum', fact_type='money').
    """
    conn = _get_db()
    result = aggregate_facts(
        conn,
        op=op,
        fact_type=fact_type,
        key=key,
        min_number=min_number,
        max_number=max_number,
        start_date=start_date,
        end_date=end_date,
        scope=scope,
    )
    total = _total_facts(conn)
    result["facts_available"] = total > 0
    result["total_facts_indexed"] = total
    return result


def _total_facts(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])


@mcp.tool()
@tracked
def index_status() -> dict[str, Any]:
    """Report what the index covers, how fresh it is, and which tools have data.

    Use this first so a null/empty result is trustworthy: it tells you the
    indexed file count, last-indexed time, watched roots (scope), the valid
    document_type / entity_type / fact_type values to pass elsewhere, and whether
    the typed-facts layer is populated.
    """
    conn = _get_db()

    def rows(sql: str) -> list[sqlite3.Row]:
        return conn.execute(sql).fetchall()

    counts = {
        r["status"]: r["n"]
        for r in rows("SELECT status, COUNT(*) n FROM files GROUP BY status")
    }
    fresh = conn.execute(
        "SELECT MAX(indexed_at) AS i, MAX(modified_at) AS m FROM files"
    ).fetchone()
    try:
        from find_and_seek.watch.roots import list_roots

        roots = [str(r) for r in list_roots()]
    except Exception:
        roots = []
    total_facts = _total_facts(conn)
    return {
        "files_indexed": counts.get("indexed", 0),
        "files_by_status": counts,
        "summaries": int(conn.execute("SELECT COUNT(*) FROM file_summaries").fetchone()[0]),
        "chunks": int(conn.execute("SELECT COUNT(*) FROM file_chunks").fetchone()[0]),
        "entities": int(conn.execute("SELECT COUNT(*) FROM file_entities").fetchone()[0]),
        "last_indexed_at": fresh["i"],
        "newest_file_modified_at": fresh["m"],
        "watched_roots": roots,
        "document_types": [
            r["document_type"]
            for r in rows(
                "SELECT document_type FROM file_summaries "
                "WHERE document_type IS NOT NULL GROUP BY document_type ORDER BY COUNT(*) DESC"
            )
        ],
        "entity_types": [
            r["entity_type"]
            for r in rows("SELECT entity_type FROM file_entities GROUP BY entity_type ORDER BY COUNT(*) DESC")
        ],
        "facts_available": total_facts > 0,
        "total_facts_indexed": total_facts,
        "fact_types": [
            r["fact_type"]
            for r in rows("SELECT fact_type FROM facts GROUP BY fact_type ORDER BY COUNT(*) DESC")
        ],
    }


@mcp.tool()
@tracked
def get_file_context(
    file_id: int | None = None, path: str | None = None, max_chunks: int = 3,
) -> dict[str, Any]:
    """One compact call: a file's summary, top entities, tags, fact counts, and
    first chunks — enough to reason about a file without several round-trips.

    Identify the file by `file_id` OR `path` (agents often arrive with a path
    from search results or the user). If the path isn't indexed, this errors with
    a clear message — run search_files or check index_status for coverage.
    """
    conn = _get_db()
    if file_id is None and path is None:
        raise ValueError("provide file_id or path")
    if file_id is None:
        row = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
        if not row:
            raise ValueError(f"not indexed: {path!r} — try search_files, or see index_status for coverage")
        file_id = int(row["id"])
    f = conn.execute(
        "SELECT id, path, filename, file_type, modified_at FROM files WHERE id=?",
        (file_id,),
    ).fetchone()
    if not f:
        raise ValueError(f"no file with id={file_id}")
    s = conn.execute("SELECT * FROM file_summaries WHERE file_id=?", (file_id,)).fetchone()
    entities = [
        {"entity_type": r["entity_type"], "value": r["entity_value"], "occurrences": r["n"]}
        for r in conn.execute(
            "SELECT entity_type, entity_value, COUNT(*) n FROM file_entities "
            "WHERE file_id=? GROUP BY entity_type, LOWER(entity_value) ORDER BY n DESC LIMIT 15",
            (file_id,),
        )
    ]
    raw_tags = [
        r["name"]
        for r in conn.execute(
            "SELECT t.name FROM file_tags ft JOIN tags t ON t.id=ft.tag_id WHERE ft.file_id=?",
            (file_id,),
        )
    ]
    # Derive type: tag from document_type at read time so it always matches the
    # authoritative classifier output. Stored type: tags may be stale (written by
    # the Phase-0 organize pass before the classifier overhaul).
    doc_type = s["document_type"] if s else None
    authoritative_type_tag = f"type:{doc_type}" if doc_type else None
    tags = [t for t in raw_tags if not t.startswith("type:")]
    if authoritative_type_tag:
        tags = [authoritative_type_tag] + tags
    fact_counts = {
        r["fact_type"]: r["n"]
        for r in conn.execute(
            "SELECT fact_type, COUNT(*) n FROM facts WHERE file_id=? GROUP BY fact_type",
            (file_id,),
        )
    }
    chunks = [
        {"chunk_id": r["id"], "location_ref": r["location_ref"], "preview": (r["text"] or "")[:280]}
        for r in conn.execute(
            "SELECT id, location_ref, text FROM file_chunks WHERE file_id=? ORDER BY chunk_index LIMIT ?",
            (file_id, max_chunks),
        )
    ]
    return {
        "file_id": file_id,
        "path": f["path"],
        "filename": f["filename"],
        "file_type": f["file_type"],
        "modified_at": (f["modified_at"] or "")[:10],
        "document_type": s["document_type"] if s else None,
        "classification_confidence": s["classification_confidence"] if s else None,
        "suggested_type": s["suggested_category"] if s else None,
        "one_line_anchor": s["one_line_anchor"] if s else None,
        "summary_text": s["summary_text"] if s else None,
        "key_facts": json.loads(s["key_facts"]) if s and s["key_facts"] else {},
        "entities": entities,
        "tags": tags,
        "fact_counts": fact_counts,
        "chunks": chunks,
    }


@mcp.tool()
@tracked
def propose_organize_plan(
    scope: str = "all",
    strategy: str = "by_type",
) -> dict[str, Any]:
    """Generate a preview plan for organizing files safely.

    **No files are touched.** The plan is a proposal: it records what *would*
    move, rename, or be quarantined, and returns a plan_id plus a sample of the
    actions. Applying is user-gated — there is deliberately no MCP tool that
    moves files.

    Not a pure read, though: the draft plan is persisted to the index so the
    user can open it by id in the app. Only the newest few drafts are kept, so
    calling this repeatedly is safe and does not grow the database. Applied and
    undone plans are never pruned — that history is what makes an apply
    reversible.

    scope: "all" or a folder path to restrict the plan to (e.g.
        "/Users/me/Downloads"). Matches that folder and everything beneath it.
    strategy: "by_type" (default) or other engine-supported strategy.
    """
    conn = _get_db()
    from find_and_seek.organize import plan_store
    from find_and_seek.organize.planner import generate_plan

    roots = [scope] if scope and scope != "all" else None
    result = generate_plan(conn, roots=roots, strategy=strategy)

    # Bound the write this tool performs. An agent can call a read-shaped tool
    # in a loop; without this, each call would leave another draft plan (and its
    # action rows) behind indefinitely.
    plan_store.prune_draft_plans(conn, keep=_MAX_AGENT_DRAFT_PLANS)

    actions = plan_store.get_actions(conn, result["plan_id"])
    # Lead the preview with actions that actually change something — a plan's
    # first rows are create_dir scaffolding, which tells the agent nothing.
    moves = [a for a in actions if a["action_type"] in ("move", "rename", "quarantine_duplicate")]

    return {
        "plan_id": result["plan_id"],
        "summary": result["summary"],
        "actions_preview": [dict(a) for a in (moves[:5] if moves else actions[:5])],
        "total_actions": len(actions),
        # Applying is USER-GATED — there is intentionally no MCP tool that moves
        # files. The agent proposes; only the human applies, in the Find and Seek app.
        "to_apply": (
            f"This is a preview only — no files were changed. Applying moves files and is "
            f"USER-GATED: agents cannot execute it. Ask the user to open the Find and Seek app and "
            f"review/apply plan {result['plan_id']} (Organize → review → Apply), where every "
            f"change is journaled and reversible."
        ),
    }


def main() -> None:
    from find_and_seek.mcp.beacon import ping

    ping("spawn")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

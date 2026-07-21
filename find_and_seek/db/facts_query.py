"""Read-side queries over the typed ``facts`` table.

Shared by the MCP tools and the FastAPI sidecar so the filter/aggregate logic
lives in one place. All queries are read-only.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from find_and_seek.db.scope import scope_clause

_NUMERIC_OPS = {"sum", "avg", "min", "max", "count"}


def _where(
    fact_type: str | None,
    key: str | None,
    min_number: float | None,
    max_number: float | None,
    start_date: str | None,
    end_date: str | None,
    contains: str | None,
    scope: str | None,
) -> tuple[str, list[Any]]:
    clauses = ["+f.status = 'indexed'"]
    params: list[Any] = []
    if fact_type:
        clauses.append("ft.fact_type = ?")
        params.append(fact_type)
    if key:
        clauses.append("ft.key = ?")
        params.append(key)
    if min_number is not None:
        clauses.append("ft.value_number >= ?")
        params.append(min_number)
    if max_number is not None:
        clauses.append("ft.value_number <= ?")
        params.append(max_number)
    if start_date:
        clauses.append("ft.value_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("ft.value_date <= ?")
        params.append(end_date)
    if contains:
        clauses.append("ft.value_text LIKE ?")
        params.append(f"%{contains}%")
    scope_sql, scope_params = scope_clause(scope)
    if scope_sql:
        clauses.append(scope_sql)
        params.extend(scope_params)
    return " AND ".join(clauses), params


def query_facts(
    conn: sqlite3.Connection,
    *,
    fact_type: str | None = None,
    key: str | None = None,
    min_number: float | None = None,
    max_number: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    contains: str | None = None,
    scope: str | None = None,
    order_by: str = "value_number",
    descending: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Filter facts and return them with their file + chunk provenance."""
    where, params = _where(
        fact_type, key, min_number, max_number, start_date, end_date, contains, scope
    )
    order_col = {
        "value_number": "ft.value_number",
        "value_date": "ft.value_date",
        "confidence": "ft.confidence",
    }.get(order_by, "ft.value_number")
    direction = "DESC" if descending else "ASC"
    params.extend([int(limit), int(offset)])
    rows = conn.execute(
        f"""
        SELECT ft.file_id, ft.fact_type, ft.key, ft.value_text, ft.value_number,
               ft.value_date, ft.unit, ft.confidence, ft.source,
               f.filename, f.path, fs.one_line_anchor, c.location_ref
        FROM facts ft
        JOIN files f ON f.id = ft.file_id
        LEFT JOIN file_summaries fs ON fs.file_id = ft.file_id
        LEFT JOIN file_chunks c ON c.id = ft.chunk_id
        WHERE {where}
        ORDER BY {order_col} {direction} NULLS LAST
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()
    return [
        {
            "file_id": r["file_id"],
            "filename": r["filename"],
            "one_line_anchor": r["one_line_anchor"],
            "fact_type": r["fact_type"],
            "key": r["key"],
            "value_text": r["value_text"],
            "value_number": r["value_number"],
            "value_date": r["value_date"],
            "unit": r["unit"],
            "confidence": r["confidence"],
            "source": r["source"],
            "location_ref": r["location_ref"],
        }
        for r in rows
    ]


def aggregate_facts(
    conn: sqlite3.Connection,
    *,
    op: str = "count",
    fact_type: str | None = None,
    key: str | None = None,
    min_number: float | None = None,
    max_number: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    """Aggregate over matching facts: count / sum / avg / min / max of value_number.

    Deduplicates by (content_hash, value_number, value_text) so the same fact
    extracted from multiple copies of a document (e.g. a PDF and a duplicate .docx)
    is counted only once. Returns `distinct_files` (number of unique source files
    after deduplication) alongside the aggregate value.
    """
    op = op.lower()
    if op not in _NUMERIC_OPS:
        return {"error": f"unsupported op '{op}'", "supported": sorted(_NUMERIC_OPS)}
    where, params = _where(
        fact_type, key, min_number, max_number, start_date, end_date, None, scope
    )
    # Dedupe by (content_hash, value_number, value_text) to avoid counting the
    # same amount from N copies of the same document N times. The inner query
    # emits one row per distinct (hash, value) combination; the outer aggregates
    # those deduplicated rows. DISTINCT on three columns is cheap at this scale.
    agg_expr = "COUNT(*)" if op == "count" else f"{op.upper()}(value_number)"
    row = conn.execute(
        f"""
        SELECT {agg_expr} AS result,
               COUNT(*) AS n,
               COUNT(DISTINCT content_hash) AS distinct_files
        FROM (
            SELECT DISTINCT f.content_hash, ft.value_number, ft.value_text
            FROM facts ft
            JOIN files f ON f.id = ft.file_id
            WHERE {where}
        ) _dedup
        """,
        params,
    ).fetchone()
    return {
        "op": op,
        "fact_type": fact_type,
        "key": key,
        "value": row["result"],
        "n": row["n"],
        "distinct_files": row["distinct_files"],
    }

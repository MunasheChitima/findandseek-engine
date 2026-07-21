"""User corrections to summary cards — the field-level overlay.

The model's card lives in ``file_summaries`` and is regenerated freely; a
human's correction lives here and outranks it forever. ``apply_edits`` merges
the overlay at read time, marks the card ``user_verified`` (the one confidence
tier not earned by code — it is earned by a person), and reports drift when the
file's content changed after the correction was made, so the user stays the
authority on what they actually saw.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

# Card fields a user may override directly. key_facts entries use the
# 'key_facts.<key>' form and are validated separately.
EDITABLE_FIELDS = frozenset(
    {"summary_text", "one_line_anchor", "document_type", "suggested_filename"}
)

_FACT_PREFIX = "key_facts."


def set_edit(
    conn: sqlite3.Connection,
    file_id: int,
    field: str,
    value: str | None,
) -> None:
    """Record a user's correction (value=None deletes the field/fact).

    Raises ValueError for a field outside the card contract — the overlay must
    never become a junk drawer of arbitrary keys.
    """
    if field not in EDITABLE_FIELDS and not (
        field.startswith(_FACT_PREFIX) and len(field) > len(_FACT_PREFIX)
    ):
        raise ValueError(f"not an editable card field: {field!r}")
    row = conn.execute("SELECT content_hash FROM files WHERE id=?", (file_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown file_id: {file_id}")
    conn.execute(
        "INSERT INTO summary_edits (file_id, field, value, edited_at, content_hash) "
        "VALUES (?,?,?,?,?) "
        "ON CONFLICT(file_id, field) DO UPDATE SET "
        "value=excluded.value, edited_at=excluded.edited_at, content_hash=excluded.content_hash",
        (file_id, field, value, datetime.now(timezone.utc).isoformat(), row[0]),
    )
    conn.commit()


def clear_edit(conn: sqlite3.Connection, file_id: int, field: str) -> None:
    """Remove a correction entirely (back to the model's value)."""
    conn.execute(
        "DELETE FROM summary_edits WHERE file_id=? AND field=?", (file_id, field)
    )
    conn.commit()


def edits_for_file(conn: sqlite3.Connection, file_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT field, value, edited_at, content_hash FROM summary_edits WHERE file_id=?",
        (file_id,),
    ).fetchall()
    return [
        {"field": r[0], "value": r[1], "edited_at": r[2], "content_hash": r[3]}
        for r in rows
    ]


def apply_edits(
    conn: sqlite3.Connection, file_id: int, card: dict[str, Any]
) -> dict[str, Any]:
    """Merge the user overlay into a model card at read time.

    Returns the card with:
      * edited fields/facts replaced (or removed, for value=None),
      * ``user_verified`` = list of overridden field names (empty list absent
        any edits — consumers can key trust off it),
      * ``edit_drift`` = True when the file's current content_hash differs from
        the hash recorded at edit time (the UI says "document changed since
        your correction"; the edit still applies — the human stays authoritative
        until they revisit it).
    """
    edits = edits_for_file(conn, file_id)
    if not edits:
        return card
    current = conn.execute(
        "SELECT content_hash FROM files WHERE id=?", (file_id,)
    ).fetchone()
    current_hash = current[0] if current else None

    out = dict(card)
    facts = dict(out.get("key_facts") or {})
    verified: list[str] = []
    drift = False
    for e in edits:
        field, value = e["field"], e["value"]
        if e["content_hash"] and current_hash and e["content_hash"] != current_hash:
            drift = True
        if field.startswith(_FACT_PREFIX):
            key = field[len(_FACT_PREFIX):]
            if value is None:
                facts.pop(key, None)
            else:
                facts[key] = value
            verified.append(field)
        else:
            if value is None:
                out[field] = None
            else:
                out[field] = value
            verified.append(field)
    out["key_facts"] = facts
    out["user_verified"] = sorted(verified)
    if drift:
        out["edit_drift"] = True
    return out


def dump_edits(conn: sqlite3.Connection, file_id: int) -> str:
    """JSON form of the overlay — used by the API surface and by the local
    golden-set export (each correction is a labelled model mistake)."""
    return json.dumps(edits_for_file(conn, file_id), ensure_ascii=False)

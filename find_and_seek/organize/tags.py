"""Native tag store + auto-tagging from the catalog (design §7).

Native (`file_tags`) tags only — instant, never touch the filesystem, power
search facets. macOS Finder-tag sync is Phase 2.

Tag names are namespaced ``<kind>:<slug>`` (e.g. ``type:invoice``, ``year:2024``,
``party:acme-corp``) so the UNIQUE name can't collide across kinds (a party
literally named "report" must not clash with ``type:report``). The UI strips the
prefix for display.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from find_and_seek.organize.taxonomy import canonicalize_type, slugify

# Org names below this length, or that are pure noise, make poor party tags.
_MIN_PARTY_LEN = 3
_MAX_PARTY_TAGS = 2
# Generic words and roles NER picks up as "orgs" but that make useless parties.
_PARTY_STOP = {
    "inc", "ltd", "llc", "the", "and", "pty", "co", "corp", "company",
    "digital", "commission", "applicant", "respondent", "appellant", "page",
    "email", "subject", "attachment", "document", "draft", "final", "copy",
    "csv", "xlsx", "pdf", "board", "chair", "respondents", "vce", "opened",
    "viewed", "sent", "from",
}
# Email-tracker / pagination artifacts that leak in as entities — never real
# parties. Includes Mailsuite tracking-pixel strings ("opened in <place>") whose
# slugs end in tracking verbs like ``…-from`` / ``…-opened``.
_PARTY_NOISE_RE = re.compile(
    r"(mailsuite|tracking|pixel|page-\d+|unsubscribe|-(from|opened|viewed|clicked|sent)$)",
    re.I,
)

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _looks_like_id(slug: str) -> bool:
    """True for extracted IDs/hashes (e.g. ``lstwslh4dbhuskhedm6g1szjd``) rather
    than human-readable names — judged by length, vowel scarcity, digit mixing."""
    s = slug.replace("-", "")
    if len(s) < 6:
        return False
    vowels = sum(c in "aeiou" for c in s) / len(s)
    has_digit = any(c.isdigit() for c in s)
    if len(s) >= 12 and vowels < 0.22:   # long & vowel-starved → random token
        return True
    if has_digit and vowels < 0.30:       # digit-laced & barely pronounceable → id
        return True
    return False


def tag_name(kind: str, slug: str) -> str:
    return f"{kind}:{slug}"


def ensure_tag(conn: sqlite3.Connection, kind: str, slug: str, source: str = "auto") -> int:
    """Idempotently get-or-create a tag, return its id."""
    name = tag_name(kind, slug)
    row = conn.execute("SELECT id FROM tags WHERE name=?", (name,)).fetchone()
    if row:
        return int(row[0])
    cur = conn.execute(
        "INSERT INTO tags (name, kind, source) VALUES (?, ?, ?)",
        (name, kind, source),
    )
    return int(cur.lastrowid)


def get_file_tags(conn: sqlite3.Connection, file_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT t.id, t.name, t.kind, ft.source, ft.confidence
        FROM file_tags ft JOIN tags t ON t.id = ft.tag_id
        WHERE ft.file_id = ?
        ORDER BY t.kind, t.name
        """,
        (file_id,),
    ).fetchall()
    return [
        {"id": r["id"], "name": r["name"], "kind": r["kind"],
         "source": r["source"], "confidence": r["confidence"]}
        for r in rows
    ]


def list_tag_facets(conn: sqlite3.Connection, kind: str | None = None) -> list[dict[str, Any]]:
    """Tags with the count of files carrying each — for search facets/filters."""
    sql = """
        SELECT t.id, t.name, t.kind, COUNT(ft.file_id) AS file_count
        FROM tags t LEFT JOIN file_tags ft ON ft.tag_id = t.id
    """
    params: list[Any] = []
    if kind:
        sql += " WHERE t.kind = ?"
        params.append(kind)
    sql += " GROUP BY t.id ORDER BY file_count DESC, t.name"
    rows = conn.execute(sql, params).fetchall()
    return [
        {"id": r["id"], "name": r["name"], "kind": r["kind"], "file_count": r["file_count"]}
        for r in rows
    ]


def clear_auto_tags(conn: sqlite3.Connection, file_id: int) -> None:
    """Drop a file's auto tags (keeps user tags) so re-tagging is idempotent."""
    conn.execute("DELETE FROM file_tags WHERE file_id=? AND source='auto'", (file_id,))


def _party_slugs(conn: sqlite3.Connection, file_id: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT entity_value, COUNT(*) AS n
        FROM file_entities
        WHERE file_id=? AND entity_type='org'
        GROUP BY LOWER(entity_value)
        ORDER BY n DESC
        """,
        (file_id,),
    ).fetchall()
    out: list[str] = []
    for r in rows:
        slug = slugify(r["entity_value"])
        if len(slug) < _MIN_PARTY_LEN or slug in _PARTY_STOP:
            continue
        if _PARTY_NOISE_RE.search(slug) or _looks_like_id(slug):
            continue
        if slug not in out:
            out.append(slug)
        if len(out) >= _MAX_PARTY_TAGS:
            break
    return out


def derive_tags(conn: sqlite3.Connection, file_id: int) -> list[tuple[str, str, float]]:
    """Compute (kind, slug, confidence) tags for a file from the catalog.

    Pure read — does not write. Order: type, year, party, status.
    """
    summary = conn.execute(
        "SELECT document_type, key_facts FROM file_summaries WHERE file_id=?",
        (file_id,),
    ).fetchone()
    filerow = conn.execute("SELECT modified_at FROM files WHERE id=?", (file_id,)).fetchone()

    raw_type = summary["document_type"] if summary else None
    key_facts: dict[str, Any] = {}
    if summary and summary["key_facts"]:
        try:
            key_facts = json.loads(summary["key_facts"]) or {}
        except (json.JSONDecodeError, TypeError):
            key_facts = {}

    tags: list[tuple[str, str, float]] = []

    # type — canonical, high confidence when the source was a known label.
    ct = canonicalize_type(raw_type)
    tags.append(("type", ct, 0.9 if raw_type else 0.4))

    # year — prefer a date in key_facts, fall back to file mtime.
    year, conf = None, 0.0
    kf_date = str(key_facts.get("date") or "")
    m = _YEAR_RE.search(kf_date)
    if m:
        year, conf = m.group(0), 0.9
    elif filerow and filerow["modified_at"]:
        m = _YEAR_RE.search(filerow["modified_at"])
        if m:
            year, conf = m.group(0), 0.6
    if year:
        tags.append(("year", year, conf))

    # party — strongest org entities.
    for slug in _party_slugs(conn, file_id):
        tags.append(("party", slug, 0.7))

    # status — only when explicitly present in key_facts (heuristics are thin
    # without re-reading content; keep v1 conservative).
    status = key_facts.get("status")
    if isinstance(status, str) and status.strip():
        tags.append(("status", slugify(status), 0.6))

    return tags


def auto_tag_file(conn: sqlite3.Connection, file_id: int) -> int:
    """Replace a file's auto tags with freshly-derived ones. Returns tag count."""
    clear_auto_tags(conn, file_id)
    n = 0
    for kind, slug, conf in derive_tags(conn, file_id):
        if not slug:
            continue
        tag_id = ensure_tag(conn, kind, slug, source="auto")
        conn.execute(
            """
            INSERT INTO file_tags (file_id, tag_id, source, confidence)
            VALUES (?, ?, 'auto', ?)
            ON CONFLICT(file_id, tag_id) DO UPDATE SET confidence=excluded.confidence
            """,
            (file_id, tag_id, conf),
        )
        n += 1
    return n


def auto_tag_scope(
    conn: sqlite3.Connection, scope_prefixes: list[str] | None = None
) -> dict[str, int]:
    """Auto-tag every indexed file under the given path prefixes (all if None).

    Writes only `tags`/`file_tags`. Returns {files, tags} counts.
    """
    if scope_prefixes:
        clauses = " OR ".join("path LIKE ?" for _ in scope_prefixes)
        params = [f"{p.rstrip('/')}/%" for p in scope_prefixes]
        rows = conn.execute(
            f"SELECT id FROM files WHERE status='indexed' AND ({clauses})", params
        ).fetchall()
    else:
        rows = conn.execute("SELECT id FROM files WHERE status='indexed'").fetchall()

    files = 0
    tags = 0
    for r in rows:
        tags += auto_tag_file(conn, int(r[0]))
        files += 1
    conn.commit()
    return {"files": files, "tags": tags}

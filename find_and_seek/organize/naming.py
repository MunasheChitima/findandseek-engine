"""Canonical filename + destination-folder derivation (design §8.1).

Pattern: ``{Party} — {Type} {Id?} — {Date}.{ext}`` — e.g.
``Acme Corp — Invoice 4471 — 2025-03-03.pdf``. Fields come from the catalog and
the template degrades gracefully when any are missing, ultimately falling back to
the file's existing `suggested_filename` then its original name. Nothing here
touches disk — it only proposes strings the user edits in preview.
"""

from __future__ import annotations

import json
import re
from typing import Any

from find_and_seek.organize.taxonomy import canonicalize_type, type_folder

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_BAD_FS = re.compile(r'[/\\:*?"<>|]+')


def _clean(part: str) -> str:
    """Trim a path component to something filesystem-safe and tidy."""
    return _BAD_FS.sub("", part).strip().strip("-—").strip()


def _title(slug_or_name: str) -> str:
    return " ".join(w.capitalize() for w in re.split(r"[\s_-]+", slug_or_name) if w)


def year_for(key_facts: dict[str, Any], modified_at: str | None) -> str:
    """Best-effort 4-digit year from key_facts.date, else file mtime, else ''."""
    m = _YEAR_RE.search(str(key_facts.get("date") or ""))
    if m:
        return m.group(0)
    if modified_at:
        m = _YEAR_RE.search(modified_at)
        if m:
            return m.group(0)
    return ""


def destination_folder(root: str, canonical_type: str, year: str) -> str:
    """`<root>/<TypeFolder>[/<Year>]` — the rules-layer home for a file."""
    parts = [root.rstrip("/"), type_folder(canonical_type)]
    if year:
        parts.append(year)
    return "/".join(parts)


def _parse_key_facts(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw) or {}
    except (json.JSONDecodeError, TypeError):
        return {}


def canonical_filename(
    *,
    document_type: str | None,
    key_facts_raw: str | None,
    suggested_filename: str | None,
    original_filename: str,
    party: str | None = None,
) -> str:
    """Build ``{Party} — {Type} {Id?} — {Date}.{ext}`` with graceful fallback.

    Falls back to `suggested_filename`, then `original_filename`, when too few
    fields are known to make the canonical pattern meaningful.
    """
    ext = ""
    dot = original_filename.rfind(".")
    if dot > 0:
        ext = original_filename[dot:].lower()

    key_facts = _parse_key_facts(key_facts_raw)
    ct = canonicalize_type(document_type)
    type_label = _title(ct) if ct != "other" else ""

    party_label = _title(_clean(party)) if party else ""

    ref = key_facts.get("id") or key_facts.get("ref") or key_facts.get("number") or ""
    ref = _clean(str(ref)) if ref else ""

    date = ""
    m = _DATE_RE.search(str(key_facts.get("date") or ""))
    if m:
        date = m.group(0)

    # Need at least a meaningful type plus one of party/date to beat the fallback.
    if type_label and (party_label or date):
        middle = type_label + (f" {ref}" if ref else "")
        segments = [s for s in (party_label, middle, date) if s]
        stem = " — ".join(segments)
        return _clean(stem) + ext

    if suggested_filename:
        return suggested_filename
    return original_filename

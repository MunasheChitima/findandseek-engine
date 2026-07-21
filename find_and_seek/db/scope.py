"""Canonical path-scope matching for SQL filters.

Every "restrict this query to a folder" filter in the engine routes through
here. Hand-rolling it produces two bugs that both look fine until real paths
hit them:

  • ``path LIKE 'scope%'`` matches siblings that merely share a prefix — a
    scope of ``/a/Docs`` silently pulls in ``/a/Docs-old``. Anchoring on a
    trailing slash fixes that, but then the scope folder's own files stop
    matching, so the clause has to accept the exact directory too.
  • ``_`` and ``%`` are LIKE wildcards. Unescaped, a scope of ``/a/jane_doe``
    also matches ``/a/janexdoe``, and a folder with ``%`` in its name matches
    very nearly everything.

Scope is a trust boundary as much as a filter — it's what an agent passes to
say "only look in Downloads" — so over-matching leaks files the caller meant
to exclude.
"""

from __future__ import annotations

from typing import Any

# Values meaning "no restriction", accepted so callers can pass user input through.
_UNSCOPED = {"", "all", "/"}


def scope_clause(scope: str | None, column: str = "f.path") -> tuple[str, list[Any]]:
    """Return ``(sql, params)`` restricting *column* to *scope* and its subtree.

    The SQL is a self-contained parenthesised boolean expression with no
    leading connective — callers join it themselves. For the unscoped cases
    (``None``, ``""``, ``"all"``, ``"/"``) it returns ``("", [])``, so callers
    can build clause lists unconditionally.
    """
    norm = (scope or "").strip().rstrip("/")
    if not norm or (scope or "").strip() in _UNSCOPED:
        return "", []
    # Backslash first — escaping it after the wildcards would double-escape them.
    like = norm.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"({column} = ? OR {column} LIKE ? ESCAPE '\\')", [norm, like + "/%"]


def scope_and_clause(scope: str | None, column: str = "f.path") -> tuple[str, list[Any]]:
    """:func:`scope_clause` prefixed with ``" AND "``, for appending to an
    existing WHERE body. Empty string when unscoped."""
    sql, params = scope_clause(scope, column)
    return (f" AND {sql}" if sql else ""), params

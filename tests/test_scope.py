"""Folder-scope matching.

Scope is a trust boundary, not just a filter — it's what a caller passes to say
"only look in Downloads" — so these assert the *negative* cases hardest: what a
scope must NOT match. Every one of these was a real over-match under the
hand-rolled ``LIKE 'scope%'`` this replaced.

The clause is exercised through sqlite rather than by asserting on parameter
lists, because the bugs live in LIKE semantics, not in string building.
"""

from __future__ import annotations

import sqlite3

import pytest

from find_and_seek.db.scope import scope_and_clause, scope_clause

PATHS = [
    "/home/me/Docs/a.pdf",
    "/home/me/Docs/nested/deep/b.pdf",
    "/home/me/Docs",              # the scope directory itself, as a row
    "/home/me/Docs-old/c.pdf",    # sibling sharing a prefix
    "/home/me/Docsy/d.pdf",       # sibling sharing a prefix, no separator
    "/home/me/Downloads/e.pdf",
    "/home/me/jane_doe/f.pdf",    # "_" is a LIKE wildcard
    "/home/me/janexdoe/g.pdf",    # what an unescaped "_" would also match
    "/home/me/100%/h.pdf",        # "%" is a LIKE wildcard
    "/home/me/100pct/i.pdf",      # what an unescaped "%" would also match
]


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE f (path TEXT)")
    c.executemany("INSERT INTO f (path) VALUES (?)", [(p,) for p in PATHS])
    return c


def matched(conn: sqlite3.Connection, scope: str | None) -> set[str]:
    sql, params = scope_clause(scope, column="path")
    where = f"WHERE {sql}" if sql else ""
    return {r["path"] for r in conn.execute(f"SELECT path FROM f {where}", params)}


def test_matches_subtree_and_the_folder_itself(conn):
    got = matched(conn, "/home/me/Docs")
    assert got == {
        "/home/me/Docs",
        "/home/me/Docs/a.pdf",
        "/home/me/Docs/nested/deep/b.pdf",
    }


def test_does_not_match_prefix_siblings(conn):
    got = matched(conn, "/home/me/Docs")
    assert "/home/me/Docs-old/c.pdf" not in got
    assert "/home/me/Docsy/d.pdf" not in got


def test_underscore_is_literal_not_a_wildcard(conn):
    got = matched(conn, "/home/me/jane_doe")
    assert got == {"/home/me/jane_doe/f.pdf"}
    assert "/home/me/janexdoe/g.pdf" not in got


def test_percent_is_literal_not_a_wildcard(conn):
    got = matched(conn, "/home/me/100%")
    assert got == {"/home/me/100%/h.pdf"}
    assert "/home/me/100pct/i.pdf" not in got


def test_trailing_slash_is_normalised(conn):
    assert matched(conn, "/home/me/Docs/") == matched(conn, "/home/me/Docs")


@pytest.mark.parametrize("scope", [None, "", "  ", "all", "/"])
def test_unscoped_values_disable_the_clause(conn, scope):
    assert scope_clause(scope, column="path") == ("", [])
    assert matched(conn, scope) == set(PATHS)


def test_backslash_escaping_does_not_double_escape(conn):
    # A backslash in a folder name must stay literal. Escaping "\" after the
    # wildcards (rather than before) would corrupt the escapes themselves.
    conn.execute("INSERT INTO f (path) VALUES (?)", (r"/home/me/back\slash/j.pdf",))
    assert matched(conn, r"/home/me/back\slash") == {r"/home/me/back\slash/j.pdf"}


def test_and_variant_is_appendable(conn):
    sql, params = scope_and_clause("/home/me/Docs", column="path")
    assert sql.startswith(" AND ")
    rows = conn.execute(f"SELECT path FROM f WHERE 1=1{sql}", params).fetchall()
    assert len(rows) == 3
    assert scope_and_clause("all", column="path") == ("", [])


def test_all_scoped_query_paths_agree():
    """Every scoped surface must delegate here, or scopes mean different things
    depending on which tool the caller reached for."""
    from find_and_seek.search.hybrid import _scope_clause

    assert _scope_clause("/home/me/Docs") == scope_and_clause("/home/me/Docs")

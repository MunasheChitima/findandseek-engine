"""Tests for the one-time upgrade migration that seeds roots.json from the index.

Background (D32 follow-up): the app moved from Full-Disk-Access to explicit
per-folder grants. A user who upgrades has an indexed corpus but no roots.json, so
list_roots() returns [] → onboarding gates on "no granted folders" and silently
resets their whole scope. seed_roots_from_index() reconstructs a small set of
sensible roots from the existing index, exactly once (guarded by the file existing).

All temp-DB / temp-path; never touches the live index or roots.json.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import find_and_seek.watch.roots as roots_mod
from find_and_seek.db.connection import init_db


def _insert_indexed(conn, path: str, status: str = "indexed") -> None:
    p = Path(path)
    conn.execute(
        """
        INSERT INTO files (path, filename, extension, content_hash, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (str(p), p.name, p.suffix, f"hash-{p.name}", status),
    )
    conn.commit()


@pytest.fixture
def env(monkeypatch):
    """Temp DB + temp roots.json path + a fake home directory."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        roots_path = tmpdir / "config" / "roots.json"
        fake_home = tmpdir / "home"
        fake_home.mkdir()
        monkeypatch.setenv("FINDANDSEEK_TEST", "1")
        monkeypatch.setattr(roots_mod, "ROOTS_PATH", roots_path)
        monkeypatch.setattr(roots_mod.Path, "home", staticmethod(lambda: fake_home))
        conn = init_db(tmpdir / "test.db")
        try:
            yield conn, roots_path, fake_home
        finally:
            conn.close()


def test_seed_writes_deduped_roots_from_index(env):
    conn, roots_path, home = env
    # Several files across two standard home folders + many in one of them: must
    # collapse to the top-level folders, deduped (NOT one root per file).
    _insert_indexed(conn, str(home / "Documents" / "a.pdf"))
    _insert_indexed(conn, str(home / "Documents" / "sub" / "b.pdf"))
    _insert_indexed(conn, str(home / "Documents" / "sub" / "c.pdf"))
    _insert_indexed(conn, str(home / "Downloads" / "d.pdf"))

    seeded = roots_mod.seed_roots_from_index(conn)

    assert roots_path.exists()
    assert set(seeded) == {home / "Documents", home / "Downloads"}
    # Persisted + readable via the normal API, and deduped to 2 roots.
    persisted = roots_mod.list_roots()
    assert set(persisted) == {home / "Documents", home / "Downloads"}
    assert len(persisted) == 2


def test_seed_prefers_standard_home_folders_first(env):
    conn, roots_path, home = env
    _insert_indexed(conn, str(home / "Projects" / "x.txt"))
    _insert_indexed(conn, str(home / "Desktop" / "y.txt"))

    seeded = roots_mod.seed_roots_from_index(conn)

    # Standard home folder surfaces first; the other top-level dir follows.
    assert seeded[0] == home / "Desktop"
    assert set(seeded) == {home / "Desktop", home / "Projects"}


def test_seed_is_noop_when_roots_already_present(env):
    conn, roots_path, home = env
    _insert_indexed(conn, str(home / "Documents" / "a.pdf"))
    # A pre-existing user choice — must never be overwritten.
    roots_mod._save([home / "MyChosenFolder"])
    assert roots_path.exists()

    seeded = roots_mod.seed_roots_from_index(conn)

    assert seeded == [home / "MyChosenFolder"]
    assert roots_mod.list_roots() == [home / "MyChosenFolder"]


def test_seed_writes_nothing_for_empty_index(env):
    conn, roots_path, home = env
    # No indexed files at all.
    seeded = roots_mod.seed_roots_from_index(conn)

    assert seeded == []
    assert not roots_path.exists()
    assert roots_mod.list_roots() == []


def test_seed_ignores_non_indexed_rows(env):
    conn, roots_path, home = env
    # Pending/failed rows are not granted scope evidence — only status='indexed'.
    _insert_indexed(conn, str(home / "Documents" / "pending.pdf"), status="pending")

    seeded = roots_mod.seed_roots_from_index(conn)

    assert seeded == []
    assert not roots_path.exists()

"""Opt-in Finder-tag sync — xattr round-trip + reversible apply/undo.

Exercises real extended-attribute writes on throwaway temp files (macOS). Never
touches the live ~/.findandseek or the live index.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from find_and_seek.db.connection import init_db
from find_and_seek.db.store import iso_now
from find_and_seek.organize import finder_tags
from find_and_seek.organize.apply import apply_plan
from find_and_seek.organize.undo import undo_plan

pytestmark = pytest.mark.skipif(
    not finder_tags.xattr_supported(), reason="needs xattr support (macOS/Linux)"
)


@pytest.fixture
def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FINDANDSEEK_TEST", "1")
    monkeypatch.setenv("FINDANDSEEK_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    conn = init_db(tmp_path / "t.db")
    yield conn, str(tmp_path)
    conn.close()


def _seed_file_with_tags(db, name="invoice.pdf", tags=(("type", "invoice"),
                                                       ("year", "2025"),
                                                       ("party", "acme-corp"))):
    """Create a real temp file + a catalog row + namespaced tags on it."""
    conn, tmp = db
    path = str(Path(tmp) / name)
    Path(path).write_text("hello")
    conn.execute(
        """INSERT INTO files (path, filename, extension, content_hash, size_bytes,
                              modified_at, indexed_at, file_type, status)
           VALUES (?, ?, 'pdf', 'h', 5, ?, ?, 'pdf', 'indexed')""",
        (path, name, "2025-01-01T00:00:00+00:00", iso_now()),
    )
    fid = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]
    for kind, slug in tags:
        full = f"{kind}:{slug}"
        conn.execute("INSERT INTO tags (name, kind, source) VALUES (?, ?, 'auto')", (full, kind))
        tid = conn.execute("SELECT id FROM tags WHERE name=?", (full,)).fetchone()[0]
        conn.execute("INSERT INTO file_tags (file_id, tag_id, source) VALUES (?, ?, 'auto')", (fid, tid))
    conn.commit()
    return fid, path


def test_display_name():
    assert finder_tags.display_name("party:acme-corp") == "Acme Corp"
    assert finder_tags.display_name("type:invoice") == "Invoice"
    assert finder_tags.display_name("2025") == "2025"


def test_xattr_round_trip(db):
    conn, tmp = db
    path = str(Path(tmp) / "f.txt")
    Path(path).write_text("x")
    finder_tags.write_raw(path, ["Invoice\n4", "2025\n1"])
    assert finder_tags.read_finder_tags(path) == ["Invoice", "2025"]
    finder_tags.write_raw(path, [])  # removal
    assert finder_tags.read_finder_tags(path) == []


def test_apply_writes_finder_tags_then_undo_clears(db):
    conn, _ = db
    fid, path = _seed_file_with_tags(db)
    plan_id = finder_tags.build_finder_sync_plan(conn, kinds=("type", "year", "party"))
    result = apply_plan(conn, plan_id)
    assert result["applied"] == 1
    assert set(finder_tags.read_finder_tags(path)) == {"Invoice", "2025", "Acme Corp"}

    undo_plan(conn, plan_id)
    assert finder_tags.read_finder_tags(path) == []


def test_sync_preserves_user_tags_and_undo_restores_exactly(db):
    conn, _ = db
    fid, path = _seed_file_with_tags(db)
    # The user already has their own Finder tag with a colour.
    finder_tags.write_raw(path, ["Important\n6"])

    plan_id = finder_tags.build_finder_sync_plan(conn, kinds=("type", "year", "party"))
    apply_plan(conn, plan_id)
    after = set(finder_tags.read_finder_tags(path))
    assert "Important" in after                       # user tag preserved
    assert {"Invoice", "2025", "Acme Corp"} <= after  # findandseek tags added

    undo_plan(conn, plan_id)
    # Exact prior state restored, colour and all.
    assert finder_tags.read_raw(path) == ["Important\n6"]


def test_compute_sync_drops_stale_managed_tag(db):
    conn, _ = db
    fid, path = _seed_file_with_tags(db, tags=(("type", "invoice"),))
    # Simulate a previously-synced (now stale) FindandSeek tag from the same universe.
    conn.execute("INSERT INTO tags (name, kind, source) VALUES ('type:receipt', 'type', 'auto')")
    conn.commit()
    finder_tags.write_raw(path, ["Receipt\n4", "Important\n6"])

    before, after = finder_tags.compute_sync(conn, fid, path, ("type",))
    names = {finder_tags._strip(e) for e in after}
    assert "Receipt" not in names      # stale managed tag dropped
    assert "Important" in names        # user tag preserved
    assert "Invoice" in names          # current managed tag present

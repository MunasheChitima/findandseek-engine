"""Organize feature — Phase 0 (tagging) + Phase 1 (preview-only reorganize).

Everything runs against a temp DB with synthetic catalog rows — no model
inference, and (critically) never the live ~/.findandseek/index.db. The whole
point of Phase 1 is that nothing touches the filesystem, so the tests assert
that too: file paths on disk are unchanged after planning/preview.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from find_and_seek.db.connection import init_db
from find_and_seek.db.store import iso_now, sha256_file
from find_and_seek.organize import journal, plan_store, preview as preview_mod
from find_and_seek.organize.apply import apply_plan
from find_and_seek.organize.naming import canonical_filename, destination_folder
from find_and_seek.organize.planner import generate_plan
from find_and_seek.organize.undo import undo_plan
from find_and_seek.organize.tags import (
    auto_tag_file,
    auto_tag_scope,
    derive_tags,
    get_file_tags,
    list_tag_facets,
)
from find_and_seek.organize.taxonomy import canonicalize_type, slugify, type_folder


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setenv("FINDANDSEEK_TEST", "1")
    with tempfile.TemporaryDirectory() as tmp:
        # Keep quarantined duplicates inside the temp dir — never the real ~/.findandseek.
        monkeypatch.setenv("FINDANDSEEK_QUARANTINE_DIR", str(Path(tmp) / "quarantine"))
        conn = init_db(Path(tmp) / "t.db")
        yield conn
        conn.close()


def _add_file(conn, *, path, content_hash="h", doc_type="invoice",
              key_facts=None, suggested=None, orgs=(), modified="2025-03-03T00:00:00+00:00"):
    p = Path(path)
    conn.execute(
        """INSERT INTO files (path, filename, extension, content_hash, size_bytes,
                              modified_at, indexed_at, file_type, status)
           VALUES (?, ?, ?, ?, 100, ?, ?, 'pdf', 'indexed')""",
        (path, p.name, p.suffix.lstrip("."), content_hash, modified, iso_now()),
    )
    fid = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]
    conn.execute(
        """INSERT INTO file_summaries (file_id, document_type, key_facts, suggested_filename)
           VALUES (?, ?, ?, ?)""",
        (fid, doc_type, json.dumps(key_facts or {}), suggested),
    )
    for org in orgs:
        conn.execute(
            "INSERT INTO file_entities (file_id, entity_type, entity_value) VALUES (?, 'org', ?)",
            (fid, org),
        )
    conn.commit()
    return fid


# ── Taxonomy ─────────────────────────────────────────────────────────


class TestTaxonomy:
    def test_drift_collapses(self):
        assert canonicalize_type("slide_deck") == "slide-deck"
        assert canonicalize_type("slide deck") == "slide-deck"
        assert canonicalize_type("product_page") == "product"
        assert canonicalize_type("product_description") == "product"
        assert canonicalize_type("csv") == "spreadsheet"
        assert canonicalize_type("cover correspondence") == "letter"

    def test_known_passthrough_and_unknown(self):
        assert canonicalize_type("invoice") == "invoice"
        assert canonicalize_type("totally-made-up") == "other"
        assert canonicalize_type(None) == "other"
        assert canonicalize_type("") == "other"

    def test_slugify(self):
        assert slugify("Acme Corp, Inc.") == "acme-corp-inc"
        assert slugify("") == ""

    def test_type_folder(self):
        assert type_folder("invoice") == "Invoices"
        assert type_folder("slide-deck") == "Presentations"


# ── Tagging (Phase 0) ────────────────────────────────────────────────


class TestTagging:
    def test_derives_type_year_party(self, db):
        fid = _add_file(
            db, path="/U/Documents/Misc/x.pdf", doc_type="invoice",
            key_facts={"date": "2025-03-03"}, orgs=["Acme Corp", "Acme Corp"],
        )
        kinds = {k: s for k, s, _ in derive_tags(db, fid)}
        assert kinds["type"] == "invoice"
        assert kinds["year"] == "2025"
        assert kinds["party"] == "acme-corp"

    def test_year_falls_back_to_mtime(self, db):
        fid = _add_file(db, path="/U/Documents/a.pdf", key_facts={}, modified="2023-01-01T00:00:00+00:00")
        kinds = {k: s for k, s, _ in derive_tags(db, fid)}
        assert kinds["year"] == "2023"

    def test_auto_tag_idempotent(self, db):
        fid = _add_file(db, path="/U/Documents/b.pdf", orgs=["Northwind Health"])
        n1 = auto_tag_file(db, fid)
        n2 = auto_tag_file(db, fid)
        assert n1 == n2
        # no duplicate file_tags
        assert db.execute("SELECT COUNT(*) FROM file_tags WHERE file_id=?", (fid,)).fetchone()[0] == n1

    def test_facets_count(self, db):
        _add_file(db, path="/U/Documents/c1.pdf", content_hash="a", doc_type="invoice")
        _add_file(db, path="/U/Documents/c2.pdf", content_hash="b", doc_type="invoice")
        auto_tag_scope(db, ["/U/Documents"])
        facets = {f["name"]: f["file_count"] for f in list_tag_facets(db, kind="type")}
        assert facets["type:invoice"] == 2

    def test_party_noise_is_filtered(self, db):
        # NER false-positives seen in the real catalog: extracted IDs, email
        # tracking-pixel strings, generic words — none should become party tags.
        fid = _add_file(
            db, path="/U/Documents/noisy.pdf",
            orgs=[
                "lstwslh4dbhuskhedm6g1szjd",   # random id/hash
                "Mailsuite Page 1 1",          # tracking pixel
                "Barcelona España opened",     # "opened in <place>" tracker
                "csv",                          # spreadsheet header noise
                "Northwind Health",            # the one real party
            ],
        )
        parties = [s for k, s, _ in derive_tags(db, fid) if k == "party"]
        assert parties == ["northwind-health"]


# ── Naming ───────────────────────────────────────────────────────────


class TestNaming:
    def test_canonical_pattern(self):
        name = canonical_filename(
            document_type="invoice",
            key_facts_raw=json.dumps({"date": "2025-03-03", "id": "4471"}),
            suggested_filename=None, original_filename="Untitled.pdf", party="acme-corp",
        )
        assert name == "Acme Corp — Invoice 4471 — 2025-03-03.pdf"

    def test_fallback_to_suggested(self):
        name = canonical_filename(
            document_type="other", key_facts_raw="{}",
            suggested_filename="My Doc.pdf", original_filename="x.pdf", party=None,
        )
        assert name == "My Doc.pdf"

    def test_destination_folder(self):
        assert destination_folder("/U/Documents", "invoice", "2025") == "/U/Documents/Invoices/2025"
        assert destination_folder("/U/Documents", "other", "") == "/U/Documents/Other"


# ── Planner (Phase 1) ────────────────────────────────────────────────


class TestPlanner:
    def test_invoice_moves_with_reason(self, db):
        _add_file(db, path="/U/Documents/Misc/Untitled.pdf", doc_type="invoice",
                  key_facts={"date": "2025-03-03"}, orgs=["Acme Corp"])
        res = generate_plan(db, ["/U/Documents"])
        actions = plan_store.get_actions(db, res["plan_id"])
        moves = [a for a in actions if a["action_type"] == "move"]
        assert len(moves) == 1
        to_path = moves[0]["payload"]["to_path"]
        assert "/Invoices/2025/" in to_path
        assert moves[0]["payload"]["reason"]  # explainable
        assert "out of junk drawer" in moves[0]["payload"]["reason"]

    def test_exact_duplicate_quarantined(self, db):
        # Same content_hash; the deeper/junk copy should be quarantined, the
        # better-located one kept (moved/renamed, not quarantined).
        _add_file(db, path="/U/Documents/Invoices/keep.pdf", content_hash="dup", doc_type="invoice")
        _add_file(db, path="/U/Documents/Downloads/dup.pdf", content_hash="dup", doc_type="invoice")
        res = generate_plan(db, ["/U/Documents"])
        actions = plan_store.get_actions(db, res["plan_id"])
        quarantined = [a for a in actions if a["action_type"] == "quarantine_duplicate"]
        assert len(quarantined) == 1
        assert "Downloads" in quarantined[0]["payload"]["from_path"]
        assert res["summary"]["duplicate_groups"] == 1

    def test_excludes_packages_and_system(self, db):
        _add_file(db, path="/U/Documents/Notes.rtfd/TXT.rtf", doc_type="note")
        _add_file(db, path="/U/Library/Caches/x.pdf", doc_type="invoice")
        res = generate_plan(db, ["/U/Documents", "/U/Library"])
        assert res["summary"]["files_in_scope"] == 0


# ── Preview (Phase 1) ────────────────────────────────────────────────


class TestPreview:
    def test_before_after_and_counts(self, db):
        _add_file(db, path="/U/Documents/Misc/Untitled.pdf", doc_type="invoice",
                  key_facts={"date": "2025-03-03"}, orgs=["Acme Corp"])
        res = generate_plan(db, ["/U/Documents"])
        view = preview_mod.build_preview(db, res["plan_id"])
        assert view is not None
        assert view["before_tree"] and view["after_tree"]
        assert view["diffs"][0]["from_path"] == "/U/Documents/Misc/Untitled.pdf"
        assert "moves" in view["summary_line"]

    def test_unknown_plan_returns_none(self, db):
        assert preview_mod.build_preview(db, 9999) is None


# ── Decisions + the no-filesystem-writes guarantee ───────────────────


class TestDecisionsAndSafety:
    def test_bulk_accept_persists_but_stays_staged(self, db):
        _add_file(db, path="/U/Documents/Misc/Untitled.pdf", doc_type="invoice",
                  key_facts={"date": "2025-03-03"}, orgs=["Acme Corp"])
        res = generate_plan(db, ["/U/Documents"])
        updated = plan_store.set_decision(db, res["plan_id"], "accepted")
        assert updated >= 1
        for a in plan_store.get_actions(db, res["plan_id"]):
            assert a["decision"] == "accepted"
            assert a["status"] == "staged"  # apply disabled — nothing executed

    def test_planning_touches_no_files(self, db, tmp_path):
        # Real files on disk; planning must not move/rename/delete any of them.
        f = tmp_path / "Untitled.pdf"
        f.write_bytes(b"hello")
        before = {p.name for p in tmp_path.iterdir()}
        _add_file(db, path=str(f), doc_type="invoice", key_facts={"date": "2025-03-03"}, orgs=["Acme"])
        res = generate_plan(db, [str(tmp_path)])
        preview_mod.build_preview(db, res["plan_id"])
        plan_store.set_decision(db, res["plan_id"], "accepted")
        after = {p.name for p in tmp_path.iterdir()}
        assert before == after  # disk untouched
        assert f.read_bytes() == b"hello"


# ── Apply + Undo (Phase 2 — the only filesystem-mutating code) ───────


def _write_indexed(conn, path, *, content=b"INVOICE BODY", doc_type="invoice",
                   key_facts=None, orgs=("Acme Corp",)):
    """Write a real file on disk and index it with the *matching* content_hash."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    h = sha256_file(p)
    conn.execute(
        """INSERT INTO files (path, filename, extension, content_hash, size_bytes,
                              modified_at, indexed_at, file_type, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'pdf', 'indexed')""",
        (str(p), p.name, p.suffix.lstrip("."), h, len(content),
         "2025-03-03T00:00:00+00:00", iso_now()),
    )
    fid = conn.execute("SELECT id FROM files WHERE path=?", (str(p),)).fetchone()[0]
    conn.execute(
        "INSERT INTO file_summaries (file_id, document_type, key_facts) VALUES (?, ?, ?)",
        (fid, doc_type, json.dumps(key_facts or {"date": "2025-03-03"})),
    )
    for org in orgs:
        conn.execute(
            "INSERT INTO file_entities (file_id, entity_type, entity_value) VALUES (?, 'org', ?)",
            (fid, org),
        )
    conn.commit()
    return fid


def _accept_and_apply(conn, root):
    res = generate_plan(conn, [str(root)])
    plan_store.set_decision(conn, res["plan_id"], "accepted")
    return res["plan_id"], apply_plan(conn, res["plan_id"])


class TestApplyUndo:
    def test_move_then_undo_round_trip(self, db, tmp_path):
        src = tmp_path / "Misc" / "Untitled.pdf"
        fid = _write_indexed(db, src)

        plan_id, result = _accept_and_apply(db, tmp_path)
        assert result["applied"] >= 1 and result["failed"] == 0
        assert not src.exists()                      # moved off the junk drawer
        new_path = db.execute("SELECT path FROM files WHERE id=?", (fid,)).fetchone()[0]
        assert "/Invoices/2025/" in new_path and Path(new_path).exists()
        assert journal.for_plan(db, plan_id)         # op was journalled

        undo = undo_plan(db, plan_id)
        assert undo["undone"] >= 1
        assert src.exists() and src.read_bytes() == b"INVOICE BODY"   # exact restore
        restored = db.execute("SELECT path FROM files WHERE id=?", (fid,)).fetchone()[0]
        assert restored == str(src)
        assert db.execute("SELECT status FROM plans WHERE id=?", (plan_id,)).fetchone()[0] == "undone"

    def test_exact_duplicate_quarantined_and_restored(self, db, tmp_path):
        keep = tmp_path / "Invoices" / "keep.pdf"
        dup = tmp_path / "Misc" / "dup.pdf"
        _write_indexed(db, keep, content=b"SAME BYTES")
        dup_fid = _write_indexed(db, dup, content=b"SAME BYTES")

        plan_id, result = _accept_and_apply(db, tmp_path)
        assert result["failed"] == 0
        dup_now = db.execute("SELECT path FROM files WHERE id=?", (dup_fid,)).fetchone()[0]
        assert "/quarantine/" in dup_now and Path(dup_now).exists()
        assert not dup.exists()

        undo_plan(db, plan_id)
        assert dup.exists() and dup.read_bytes() == b"SAME BYTES"

    def test_stale_source_is_skipped_not_acted_on(self, db, tmp_path):
        src = tmp_path / "Misc" / "Untitled.pdf"
        _write_indexed(db, src)
        src.write_bytes(b"CHANGED SINCE INDEXING")   # hash no longer matches catalog

        plan_id, result = _accept_and_apply(db, tmp_path)
        # The move is skipped (stale hash); only the harmless create_dir may run.
        assert result["skipped"] >= 1
        assert not list((tmp_path / "Invoices").rglob("*.pdf"))   # nothing relocated
        assert src.exists() and src.read_bytes() == b"CHANGED SINCE INDEXING"  # untouched
        assert db.execute("SELECT path FROM files WHERE path=?", (str(src),)).fetchone()  # catalog unchanged

    def test_name_collision_gets_suffix_never_overwrites(self, db, tmp_path):
        src = tmp_path / "Misc" / "Untitled.pdf"
        fid = _write_indexed(db, src)
        # Pre-create the canonical destination so the move would collide.
        res = generate_plan(db, [str(tmp_path)])
        dest = [a for a in plan_store.get_actions(db, res["plan_id"])
                if a["action_type"] == "move"][0]["payload"]["to_path"]
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"PRE-EXISTING - DO NOT CLOBBER")

        plan_store.set_decision(db, res["plan_id"], "accepted")
        apply_plan(db, res["plan_id"])
        assert Path(dest).read_bytes() == b"PRE-EXISTING - DO NOT CLOBBER"  # never overwritten
        moved = db.execute("SELECT path FROM files WHERE id=?", (fid,)).fetchone()[0]
        assert moved != dest and " (2)" in Path(moved).name

    def test_rejected_actions_do_not_run(self, db, tmp_path):
        src = tmp_path / "Misc" / "Untitled.pdf"
        _write_indexed(db, src)
        res = generate_plan(db, [str(tmp_path)])
        plan_store.set_decision(db, res["plan_id"], "rejected")   # reject everything
        result = apply_plan(db, res["plan_id"])
        assert result["applied"] == 0
        assert src.exists()   # nothing moved

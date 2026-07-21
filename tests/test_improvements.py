"""Regression tests for the post-review improvements.

These are deliberately lightweight (no corpus indexing, no spaCy/Ollama) so they
run in seconds and lock in the behaviour the fixes introduced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sqlite_vec

from find_and_seek.db.connection import init_db
from find_and_seek.db.fts_repair import _is_legacy_schema, ensure_fts_healthy
from find_and_seek.db.store import (
    backfill_facts,
    clone_indexed_file,
    enqueue,
    indexed_twin,
    purge_root,
    update_path,
    upsert_file,
    write_summary,
)
from find_and_seek.ingest.facts import extract_facts
from find_and_seek.safe_log import safe_error
from find_and_seek.search.hybrid import _fts_query, _scope_clause, search
from find_and_seek.search.rerank import _blended_scores
from find_and_seek.watch import roots as roots_mod


# ── scope prefix (#3) ─────────────────────────────────────────────


def test_scope_clause_anchors_on_trailing_slash():
    # The scope folder itself, plus everything under it — and nothing that
    # merely shares a prefix ("/a/Docs" must not match "/a/Docs-old").
    _, params = _scope_clause("/Users/me/Docs")
    assert params == ["/Users/me/Docs", "/Users/me/Docs/%"]
    # A trailing slash on the input is normalised away, not doubled.
    _, params2 = _scope_clause("/Users/me/Docs/")
    assert params2 == params
    assert _scope_clause("all") == ("", [])
    assert _scope_clause("") == ("", [])


# ── FTS query sanitisation (crash-on-punctuation) ─────────────────


def test_fts_query_sanitises_punctuation():
    # Raw punctuation must not leak into FTS5 syntax.
    q = _fts_query("what's the cost? (urgent) levy: $5,400 a.b.c *")
    assert q  # non-empty
    assert "*" not in q and "(" not in q and ":" not in q
    assert q.count('"') % 2 == 0  # balanced quotes
    assert _fts_query("!!! ??? .") == ""  # no usable terms


def test_search_survives_punctuation_query(tmp_path):
    """End-to-end: a query full of FTS metacharacters must not raise."""
    db = init_db(tmp_path / "p.db")
    fid = (tmp_path / "doc.txt"); fid.write_text("x")
    file_id = upsert_file(db, str(fid), "h", "document", status="indexed")
    db.execute(
        "INSERT INTO file_chunks (file_id, chunk_index, text, source_type, location_ref, token_estimate)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (file_id, 0, "kitchen burst pipe repair invoice", "text", "p1", 5),
    )
    db.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (db.execute("SELECT id FROM file_chunks").fetchone()[0],
         sqlite_vec.serialize_float32(np.ones(768, dtype=np.float32))),
    )
    db.commit()
    for q in ['what\'s the "cost"?', "levy: $5,400 *", "a.b.c (urgent)", "???"]:
        hits, _ = search(db, q, limit=5)  # must not raise
        assert isinstance(hits, list)


def test_search_survives_pathologically_long_query(tmp_path):
    """A pasted paragraph (1000+ tokens) must not raise. Each filename-boost
    term becomes an OR'd LIKE clause and SQLite caps expression-tree depth at
    1000, so the term list has to be deduped + capped before the query."""
    db = init_db(tmp_path / "long.db")
    fid = (tmp_path / "doc.txt"); fid.write_text("x")
    file_id = upsert_file(db, str(fid), "h", "document", status="indexed")
    db.execute(
        "INSERT INTO file_chunks (file_id, chunk_index, text, source_type, location_ref, token_estimate)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (file_id, 0, "kitchen burst pipe repair invoice", "text", "p1", 5),
    )
    db.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (db.execute("SELECT id FROM file_chunks").fetchone()[0],
         sqlite_vec.serialize_float32(np.ones(768, dtype=np.float32))),
    )
    db.commit()
    long_q = " ".join(["contract"] * 1500)            # 1500 repeated tokens
    distinct_q = " ".join(f"term{i}" for i in range(1500))  # 1500 distinct tokens
    for q in (long_q, distinct_q):
        hits, _ = search(db, q, limit=5)  # must not raise OperationalError
        assert isinstance(hits, list)


# ── watched-folder config (add/remove persistence) ───────────────


def test_roots_add_remove_persist(tmp_path, monkeypatch):
    # Redirect the config path so the real ~/.findandseek/roots.json is untouched.
    monkeypatch.setattr(roots_mod, "ROOTS_PATH", tmp_path / "roots.json")

    # Per-folder permissions: no implicit default scope — empty means empty.
    assert roots_mod.list_roots() == []
    roots_mod.add_root(tmp_path / "Inbox")
    assert (tmp_path / "Inbox") in roots_mod.list_roots()
    roots_mod.add_root(tmp_path / "Inbox")  # idempotent
    assert roots_mod.list_roots().count(tmp_path / "Inbox") == 1
    roots_mod.remove_root(tmp_path / "Inbox")
    assert (tmp_path / "Inbox") not in roots_mod.list_roots()
    assert roots_mod.list_roots() == []  # back to no scope, not a default crawl


# ── reranker honesty (#1) ─────────────────────────────────────────


def test_blended_rerank_breaks_fused_ties_with_lexical_overlap():
    cands = [(1, "the quick brown fox"), (2, "completely unrelated content")]
    fused = {1: 0.5, 2: 0.5}  # tie on the semantic prior
    ranked = _blended_scores("quick brown fox", cands, fused)
    assert ranked[0][0] == 1  # lexical overlap is the tie-breaker


def test_blended_rerank_respects_fused_prior_without_overlap():
    cands = [(1, "alpha"), (2, "beta")]
    fused = {1: 0.1, 2: 0.9}
    ranked = _blended_scores("zzz no overlap", cands, fused)
    assert ranked[0][0] == 2  # strong semantic prior still wins


# ── privacy: log scrubbing (#7) ───────────────────────────────────


def test_safe_error_keeps_type_drops_extra_lines_and_caps_length():
    exc = ValueError("first line leak\nSECOND LINE document body " + "x" * 600)
    msg = safe_error(exc)
    assert msg.startswith("ValueError:")
    assert "SECOND LINE" not in msg  # only the first line survives
    assert len(msg) <= 320  # bounded


# ── dedup by content hash (#7 / clone) ────────────────────────────


def test_clone_indexed_file_copies_chunks_vectors_summary(tmp_path):
    db = init_db(tmp_path / "t.db")
    src = tmp_path / "a.txt"
    src.write_text("hello world")
    dst = tmp_path / "b.txt"
    dst.write_text("hello world")

    fid = upsert_file(db, str(src), "samehash", "document", status="indexed")
    cur = db.execute(
        "INSERT INTO file_chunks (file_id, chunk_index, text, source_type, location_ref, token_estimate)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (fid, 0, "hello world", "text", "p1", 3),
    )
    db.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (cur.lastrowid, sqlite_vec.serialize_float32(np.ones(768, dtype=np.float32))),
    )
    write_summary(db, fid, {"summary_text": "s", "document_type": "other", "key_facts": {}})
    db.commit()

    # Duplicate is detected and cloned without re-running the pipeline.
    assert indexed_twin(db, "samehash", str(dst)) == fid
    new_id = clone_indexed_file(db, fid, str(dst), "samehash", "document")
    db.commit()
    assert new_id != fid

    texts = [r[0] for r in db.execute("SELECT text FROM file_chunks WHERE file_id=?", (new_id,))]
    assert texts == ["hello world"]
    nvec = db.execute(
        "SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id IN (SELECT id FROM file_chunks WHERE file_id=?)",
        (new_id,),
    ).fetchone()[0]
    assert nvec == 1
    assert db.execute("SELECT summary_text FROM file_summaries WHERE file_id=?", (new_id,)).fetchone()[0] == "s"


def test_indexed_twin_rejects_extension_mismatch(tmp_path):
    # A duplicate-content injector can copy a text file's bytes into a
    # differently-extensioned file (e.g. a .jpg). Different extensions mean
    # different extractors ran, so a content-hash match alone must not be
    # treated as the same file — see AAR on the twin-cloning bug.
    db = init_db(tmp_path / "t.db")
    src = tmp_path / "incident.txt"
    src.write_text("near-miss in Zone B")
    dst = tmp_path / "doc_488_2972.jpg"
    dst.write_bytes(b"near-miss in Zone B")

    fid = upsert_file(db, str(src), "samehash", "document", status="indexed")
    cur = db.execute(
        "INSERT INTO file_chunks (file_id, chunk_index, text, source_type, location_ref, token_estimate)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (fid, 0, "near-miss in Zone B", "text", "p1", 4),
    )
    db.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (cur.lastrowid, sqlite_vec.serialize_float32(np.ones(768, dtype=np.float32))),
    )
    db.commit()

    assert indexed_twin(db, "samehash", str(dst)) is None

    # Same extension still matches.
    dst_txt = tmp_path / "incident_copy.txt"
    dst_txt.write_text("near-miss in Zone B")
    assert indexed_twin(db, "samehash", str(dst_txt)) == fid


def test_indexed_twin_skips_zero_chunk_candidates(tmp_path):
    # A prior clone (or a failed extraction) can leave an indexed file with
    # zero chunks. It must never be offered as a twin — cloning it would
    # propagate the emptiness to every future duplicate.
    db = init_db(tmp_path / "t.db")
    src = tmp_path / "empty.txt"
    src.write_text("content")
    dst = tmp_path / "empty_copy.txt"
    dst.write_text("content")

    upsert_file(db, str(src), "samehash", "document", status="indexed")
    db.commit()

    assert indexed_twin(db, "samehash", str(dst)) is None


def test_clone_indexed_file_refuses_zero_chunk_source(tmp_path):
    db = init_db(tmp_path / "t.db")
    src = tmp_path / "empty.txt"
    src.write_text("content")
    dst = tmp_path / "empty_copy.txt"
    dst.write_text("content")

    fid = upsert_file(db, str(src), "samehash", "document", status="indexed")
    db.commit()

    with pytest.raises(ValueError, match="zero chunks"):
        clone_indexed_file(db, fid, str(dst), "samehash", "document")


# ── per-folder permissions: removing a folder forgets its index ───────────


def test_purge_root_forgets_only_in_scope(tmp_path):
    db = init_db(tmp_path / "t.db")
    root = tmp_path / "Docs"
    inside = str(root / "a.txt")
    nested = str(root / "sub" / "b.txt")
    outside = str(tmp_path / "Other" / "c.txt")
    sibling = str(tmp_path / "Docs-old" / "d.txt")  # must NOT be swept by the prefix

    for p in (inside, nested, outside, sibling):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("x")
        upsert_file(db, p, "h", "document", status="indexed")
    # Give the in-scope file a chunk + vector + summary so we prove vec0 + cascade.
    fid = db.execute("SELECT id FROM files WHERE path=?", (inside,)).fetchone()[0]
    cur = db.execute(
        "INSERT INTO file_chunks (file_id, chunk_index, text, source_type, location_ref, token_estimate)"
        " VALUES (?, 0, 'x', 'text', 'p1', 1)",
        (fid,),
    )
    db.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (cur.lastrowid, sqlite_vec.serialize_float32(np.ones(768, dtype=np.float32))),
    )
    write_summary(db, fid, {"summary_text": "s", "document_type": "other", "key_facts": {}})
    enqueue(db, str(root / "queued.txt"), "created")  # pending work under the root
    enqueue(db, outside, "created")
    db.commit()

    removed = purge_root(db, root)
    db.commit()

    assert removed == 2  # inside + nested only
    remaining = {r[0] for r in db.execute("SELECT path FROM files")}
    assert remaining == {outside, sibling}
    # vec0 + cascade + FTS-shadow cleared for the purged file.
    assert db.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM file_summaries").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM file_chunks").fetchone()[0] == 0
    # Queue rows under the root are dropped; the out-of-scope one survives.
    queued = {r[0] for r in db.execute("SELECT path FROM ingest_queue")}
    assert queued == {outside}


# ── rename updates files.filename; FTS indexes body text only (#2) ────────
# chunk_fts no longer denormalises filename (that schema broke bm25 + self-repair,
# see db/fts_repair.py). Filename matching is served by the live filename-token
# boost in search.hybrid, which reads files.filename — so a rename only needs the
# files row updated, and the FTS keeps indexing chunk text regardless of name.


def test_update_path_updates_filename_not_fts(tmp_path):
    db = init_db(tmp_path / "t.db")
    orig = tmp_path / "orig.txt"
    orig.write_text("x")
    fid = upsert_file(db, str(orig), "h", "document", status="indexed")
    db.execute(
        "INSERT INTO file_chunks (file_id, chunk_index, text, source_type, location_ref, token_estimate)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (fid, 0, "renovation budget", "text", "p1", 2),
    )
    db.commit()

    def fts_match(term: str) -> int:
        return len(db.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?", (term,)).fetchall())

    # Body text is searchable; filename is not in FTS (by design).
    assert fts_match("renovation") == 1
    assert fts_match("orig") == 0

    update_path(db, fid, str(tmp_path / "council_letter.txt"))
    db.commit()

    # Rename is reflected in files.filename (what the search filename-boost reads).
    assert db.execute("SELECT filename FROM files WHERE id=?", (fid,)).fetchone()[0] == "council_letter.txt"
    # Body text still indexed; FTS still doesn't carry the filename token.
    assert fts_match("renovation") == 1
    assert fts_match("council") == 0


# ── FTS self-heal: legacy schema is detected and migrated (#bm25-malformed) ──


def test_ensure_fts_healthy_migrates_legacy_schema(tmp_path):
    db = init_db(tmp_path / "t.db")  # FINDANDSEEK_TEST skips auto-heal at init
    # Recreate the legacy external-content FTS with the phantom `filename` column.
    db.executescript(
        """
        DROP TRIGGER IF EXISTS chunks_ai;
        DROP TRIGGER IF EXISTS chunks_ad;
        DROP TRIGGER IF EXISTS chunks_au;
        DROP TABLE chunk_fts;
        CREATE VIRTUAL TABLE chunk_fts USING fts5(
          text, filename, content='file_chunks', content_rowid='id', tokenize='porter unicode61'
        );
        """
    )
    src = tmp_path / "a.txt"
    src.write_text("x")
    fid = upsert_file(db, str(src), "h", "document", status="indexed")
    db.execute(
        "INSERT INTO file_chunks (file_id, chunk_index, text, source_type, location_ref, token_estimate)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (fid, 0, "annual report invoice budget summary", "text", "p1", 5),
    )
    db.commit()
    assert _is_legacy_schema(db) is True

    repaired = ensure_fts_healthy(db)

    assert repaired is True
    assert _is_legacy_schema(db) is False
    # Multi-term bm25 (the operation legacy corruption broke) now works and finds the row.
    rows = db.execute(
        "SELECT rowid, bm25(chunk_fts) FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY 2",
        ['"annual" OR "report" OR "invoice" OR "budget"'],
    ).fetchall()
    assert len(rows) == 1
    # Idempotent: a healthy index is left alone.
    assert ensure_fts_healthy(db) is False


# ── facts backfill + entity quality gate (typed-facts were empty) ─────────


def test_backfill_facts_populates_typed_facts(tmp_path):
    db = init_db(tmp_path / "t.db")
    src = tmp_path / "inv.pdf"
    src.write_text("x")
    fid = upsert_file(db, str(src), "h", "document", status="indexed")
    cur = db.execute(
        "INSERT INTO file_chunks (file_id, chunk_index, text, source_type, location_ref, token_estimate)"
        " VALUES (?, 0, 'body', 'text', 'p1', 1)",
        (fid,),
    )
    cid = cur.lastrowid
    write_summary(db, fid, {"summary_text": "s", "document_type": "invoice",
                            "key_facts": {"amount": "$500.00", "due date": "2025-03-01"}})
    # NER entities: a real org kept; a money parsed; a digit-only "person" dropped.
    for et, val in [("org", "Acme Pty Ltd"), ("money", "$1,200"), ("person", "0404538881")]:
        db.execute(
            "INSERT INTO file_entities (file_id, chunk_id, entity_type, entity_value, entity_raw)"
            " VALUES (?, ?, ?, ?, ?)",
            (fid, cid, et, val, val),
        )
    db.commit()

    assert backfill_facts(db) == 1
    facts = db.execute("SELECT fact_type, value_number, value_date, unit FROM facts WHERE file_id=?", (fid,)).fetchall()
    by_type = {f["fact_type"] for f in facts}
    assert "money" in by_type and "date" in by_type and "org" in by_type
    money = [f for f in facts if f["fact_type"] == "money"]
    assert any(f["value_number"] == 500.0 for f in money)
    assert all(f["fact_type"] != "person" for f in facts)  # digit-only person filtered out
    # Idempotent: re-running doesn't duplicate.
    n_before = db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    backfill_facts(db)
    assert db.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == n_before


def test_extract_facts_entity_quality_gate():
    ents = [
        {"entity_type": "person", "entity_value": "Jane Doe"},
        {"entity_type": "person", "entity_value": "12 345"},     # numeric → drop
        {"entity_type": "location", "entity_value": "#"},         # junk → drop
        {"entity_type": "org", "entity_value": "X"},              # too short → drop
    ]
    out = extract_facts({"key_facts": {}}, ents)
    names = {(f["fact_type"], f["value_text"]) for f in out}
    assert ("person", "Jane Doe") in names
    assert all(v != "12 345" and v != "#" and v != "X" for _, v in names)


# ── organize apply is user-gated: NO execute tool on the agent surface ────


def test_mcp_has_no_apply_tool():
    """The agent surface must not be able to move files — applying is user-only
    (done in the app via the API). Only the read-only proposer is exposed."""
    import find_and_seek.mcp.server as srv

    assert hasattr(srv, "propose_organize_plan")
    assert not hasattr(srv, "apply_organize_plan")
    assert not hasattr(srv, "_check_apply_guardrail")


# ── queue robustness (atomic claim, crash recovery, dead-lettering) ──


def _enqueue_n(db, n):
    from find_and_seek.db.store import enqueue
    for i in range(n):
        enqueue(db, f"/tmp/f{i}.txt", "created")
    db.commit()


def test_claim_queue_is_atomic_and_no_double_claim(tmp_path):
    from find_and_seek.db.store import claim_queue
    db = init_db(tmp_path / "q.db")
    _enqueue_n(db, 5)
    first = claim_queue(db, limit=3)
    second = claim_queue(db, limit=3)
    assert len(first) == 3
    assert len(second) == 2  # only the 2 remaining pending, never re-claimed
    claimed = {r["path"] for r in first} | {r["path"] for r in second}
    assert len(claimed) == 5  # all distinct, no overlap
    assert claim_queue(db, limit=3) == []  # nothing left


def test_claim_queue_skips_exhausted_attempts(tmp_path):
    from find_and_seek.db.store import MAX_ATTEMPTS, claim_queue
    db = init_db(tmp_path / "q.db")
    _enqueue_n(db, 1)
    # Simulate repeated crash-during-processing: claim, reset to pending, repeat.
    for _ in range(MAX_ATTEMPTS):
        assert len(claim_queue(db, limit=1)) == 1
        db.execute("UPDATE ingest_queue SET status='pending' WHERE path='/tmp/f0.txt'")
        db.commit()
    # Attempts now == MAX_ATTEMPTS → no longer claimable.
    assert claim_queue(db, limit=1) == []


def test_requeue_stale_processing_recovers_and_dead_letters(tmp_path):
    from find_and_seek.db.store import MAX_ATTEMPTS, requeue_stale_processing
    db = init_db(tmp_path / "q.db")
    _enqueue_n(db, 2)
    # One crashed mid-flight with attempts left, one already exhausted.
    db.execute("UPDATE ingest_queue SET status='processing', attempts=1 WHERE path='/tmp/f0.txt'")
    db.execute("UPDATE ingest_queue SET status='processing', attempts=? WHERE path='/tmp/f1.txt'", (MAX_ATTEMPTS,))
    db.commit()
    requeued, dead = requeue_stale_processing(db)
    assert (requeued, dead) == (1, 1)
    assert db.execute("SELECT status FROM ingest_queue WHERE path='/tmp/f0.txt'").fetchone()[0] == "pending"
    assert db.execute("SELECT status FROM ingest_queue WHERE path='/tmp/f1.txt'").fetchone()[0] == "failed"


def test_requeue_stale_processing_dead_letters_stuck_pending(tmp_path):
    """A row left 'pending' at the attempts ceiling is unclaimable (claim_queue
    requires attempts < MAX) so it can never make progress or resolve — it just
    inflates the "waiting" count forever. Recovery must dead-letter it too, not
    only 'processing' rows. Repro of the real ghost-queue seen in the field, where
    an interrupted/disk-full recovery left thousands of pending rows at attempts=3
    with a NULL last_error (so COALESCE fills a reason)."""
    from find_and_seek.db.store import MAX_ATTEMPTS, claim_queue, requeue_stale_processing
    db = init_db(tmp_path / "q.db")
    _enqueue_n(db, 2)
    # One healthy pending row with attempts left; one stuck pending at the ceiling.
    db.execute("UPDATE ingest_queue SET status='pending', attempts=1 WHERE path='/tmp/f0.txt'")
    db.execute("UPDATE ingest_queue SET status='pending', attempts=?, last_error=NULL "
               "WHERE path='/tmp/f1.txt'", (MAX_ATTEMPTS,))
    db.commit()
    requeued, dead = requeue_stale_processing(db)
    assert dead == 1  # the stuck-pending row was dead-lettered
    assert db.execute("SELECT status FROM ingest_queue WHERE path='/tmp/f0.txt'").fetchone()[0] == "pending"
    row = db.execute("SELECT status, last_error FROM ingest_queue WHERE path='/tmp/f1.txt'").fetchone()
    assert row[0] == "failed"
    assert row[1]  # COALESCE filled a non-null reason
    # The healthy row is still claimable; the dead-lettered one is gone from pending.
    assert [r["path"] for r in claim_queue(db, limit=10)] == ["/tmp/f0.txt"]


# ── summariser JSON robustness (batch must not crash on one bad item) ──


def test_parse_summary_survives_malformed_json():
    """A malformed/non-dict model response must return None (so the caller falls
    back per-item), never raise — a raise here used to crash the whole batched
    summariser and silently demote it to the slow per-file path, and could mark
    files as failed. Genuinely unparseable input still returns None."""
    from find_and_seek.ingest.summarise import parse_summary

    # Outright malformed (stray commas) — unrecoverable.
    assert parse_summary('{"document_type": invoice,,}', "f") is None
    # Valid JSON but not an object.
    assert parse_summary("[1, 2, 3]", "f") is None
    # No JSON at all.
    assert parse_summary("I cannot help with that.", "f") is None
    # Well-formed JSON still parses and normalises document_type casing.
    ok = parse_summary('{"document_type": "Invoice", "one_line_anchor": "x"}', "f")
    assert ok is not None and ok["document_type"] == "invoice"


def test_parse_summary_repairs_truncated_json():
    """The lean batch path caps output tokens, which can sever the closing brace
    mid-object. Rather than discard an otherwise-good anchor/type, parse_summary
    repairs the truncation (balancing/trimming) and recovers what completed."""
    from find_and_seek.ingest.summarise import parse_summary

    # Cut mid-value with no closing brace — keep the type and the partial anchor.
    out = parse_summary('{"document_type": "invoice", "one_line_anchor": "Acme inv', "f")
    assert out is not None and out["document_type"] == "invoice"
    assert out["one_line_anchor"].startswith("Acme")
    # Cut right after a complete pair (no trailing comma) — the anchor survives.
    out = parse_summary('{"document_type": "letter", "one_line_anchor": "Application letter"', "f")
    assert out is not None and out["one_line_anchor"] == "Application letter"
    # Cut inside a nested key_facts object — outer type still recovers.
    out = parse_summary('{"document_type": "report", "key_facts": {"date": "2023', "f")
    assert out is not None and out["document_type"] == "report"


# ── cascade fast-path (skip the LLM for confidently-typed files) ──


def _mk_chunks(text):
    from find_and_seek.ingest.chunk import Chunk
    return [Chunk(text=text, source_type="text", location_ref="p1", chunk_index=0, token_estimate=5)]


def test_fast_classify_covers_easy_types_and_defers_ambiguous():
    from find_and_seek.ingest.summarise import fast_classify

    # Container alone fixes the type — no LLM needed.
    assert fast_classify("email", _mk_chunks("hi"), "thread.eml")["document_type"] == "email"
    # Canonicalised to the fixed taxonomy ("slide deck" → "slide-deck").
    assert fast_classify("presentation", _mk_chunks("Q3"), "deck.pptx")["document_type"] == "slide-deck"
    # Spreadsheets/CSVs deliberately DEFER to the LLM (None): the heuristic anchor
    # for tabular data is a useless column-name dump. The type is set independently
    # downstream by classify_document, so deferring costs nothing.
    assert fast_classify("spreadsheet", _mk_chunks("a,b"), "x.xlsx") is None
    assert fast_classify("data", _mk_chunks("a,b,c"), "contacts.csv") is None
    # Title-line signal on an otherwise-opaque document.
    assert fast_classify("document", _mk_chunks("INVOICE\nAcme Co"), "scan001.pdf")["document_type"] == "invoice"
    # Genuinely ambiguous → defer to the LLM (None).
    assert fast_classify("document", _mk_chunks("Lorem ipsum dolor"), "notes.txt") is None
    assert fast_classify("image", _mk_chunks("blob"), "photo.jpg") is None
    assert fast_classify("data", _mk_chunks("{}"), "data.json") is None


def test_clean_facts_drops_table_dump_artifacts():
    """Spreadsheet summaries sometimes echo the table structure as 'facts' — a
    'Columns' list or per-row samples. Those aren't decision-useful; drop them
    while keeping genuine aggregate facts."""
    from find_and_seek.ingest.summarise import _clean_facts

    out = _clean_facts({
        "Columns": ["brand", "name", "price"],
        "Sample Row 2": {"price": "$1.50"},
        "row 4": {"price": "$3.30"},
        "Price Range": "$1.50 - $22.50",
        "Category": "T-Shirts",
    })
    assert out == {"Price Range": "$1.50 - $22.50", "Category": "T-Shirts"}


def test_search_confidence_gate():
    """A hit is trusted only if its score clears the fusion floor OR the query's
    words actually appear in it — so a query with no real match returns nothing
    instead of vague vector neighbours (the absent-multi-word-agency case)."""
    from find_and_seek.search.hybrid import _content_terms, _is_confident, SearchHit

    def mk(rel, filename="x.pdf", anchor="", summary="", preview=""):
        return SearchHit(file_id=1, filename=filename, path="/x", document_type=None,
                         one_line_anchor=anchor, summary=summary, relevance=rel,
                         modified_at=None, top_chunk_preview=preview, top_chunk_id=1,
                         top_chunk_location=None, estimated_tokens_if_opened=0)

    # Cross-encoder fired (score above floor) → confident regardless of overlap.
    assert _is_confident(mk(0.55), _content_terms("anything at all"))
    # Floor-only score AND no query words present → the junk case, not confident.
    assert not _is_confident(
        mk(0.35, filename="Redarc catalogue.pdf", preview="vehicle electronics"),
        _content_terms("Northwind Compensation Authority"))
    # Floor score but the query words appear → a real match the CE wrongly floored.
    assert _is_confident(
        mk(0.35, filename="Resume.docx", preview="resume of work history"),
        _content_terms("resume"))
    # Only a fraction of a multi-word query matches → below the lexical floor.
    assert not _is_confident(
        mk(0.35, filename="bilingual-dictionary.pdf", preview="seed of a plant"),
        _content_terms("blockchain crypto wallet seed phrase"))
    # Grounding is WHOLE-WORD, not substring: "ACT" must NOT match inside
    # "CONTRACT" (the regression that surfaced unrelated docs for a short acronym).
    assert not _is_confident(
        mk(0.35, filename="Contract A Supplier Code of Conduct.docx",
           preview="CONTRACT A SUPPLIER CODE OF CONDUCT Commitment Letter"),
        _content_terms("ACT"))
    # But a real whole-word occurrence at the genuine ~0.35 floor still grounds it.
    assert _is_confident(
        mk(0.35, filename="ACT claim.pdf", preview="Accident Compensation Tribunal claim"),
        _content_terms("ACT"))
    # A coincidental whole-word hit on a short token at rock-bottom relevance
    # (e.g. "act" as a dictionary headword) is dropped by the min-relevance floor.
    assert not _is_confident(
        mk(0.146, filename="bilingual-dictionary.pdf", preview="act : a unit of ..."),
        _content_terms("ACT"))
    # Stopwords and 1–2 char tokens are dropped from the lexical signal.
    assert _content_terms("what is my ACT") == ["act"]


def test_heuristic_anchor_skips_boilerplate():
    """When the heuristic must pick an anchor (LLM unavailable/failed), it should
    skip letterhead/contact/boilerplate and lead with the real subject line."""
    from find_and_seek.ingest.summarise import _heuristic_summary

    text = (
        "CONFIDENTIAL\n"
        "Phone: +1 (555) 123-4567\n"
        "www.example.com\n"
        "Application for Talent Acquisition Partner role\n"
        "Dear Hiring Manager"
    )
    out = _heuristic_summary(_mk_chunks(text), "cover.docx")
    assert out["one_line_anchor"] == "Application for Talent Acquisition Partner role"


def test_parse_summary_canonicalizes_document_type():
    """The model drifts ("Invoice") and free-forms past the enum ("decision",
    "job_posting") — parse_summary must canonicalise so type_filter matches and the
    browse rail doesn't fragment into un-filterable pseudo-types."""
    from find_and_seek.ingest.summarise import parse_summary
    assert parse_summary('{"document_type":"Invoice"}', "x")["document_type"] == "invoice"
    assert parse_summary('{"document_type":"slide deck"}', "x")["document_type"] == "slide-deck"
    assert parse_summary('{"document_type":"decision"}', "x")["document_type"] == "other"
    assert parse_summary('{"document_type":"job_posting"}', "x")["document_type"] == "other"
    # A "statement" (bank/claim/witness) must NOT become an invoice.
    assert parse_summary('{"document_type":"statement"}', "x")["document_type"] == "other"
    # Missing type → "other", never a crash.
    assert parse_summary('{"summary_text":"hi"}', "x")["document_type"] == "other"


def test_classification_signals_and_fusion():
    """Evidence-based classifier: hard signals decide deterministically; the
    model's answer is validated against the taxonomy; uncertainty abstains."""
    from find_and_seek.organize.signals import format_signal, hard_signal, path_prior, gather_evidence
    from find_and_seek.organize.classify import classify_document, _parse
    from find_and_seek.organize.categories import NEEDS_REVIEW, BUILTIN_SLUGS

    # Format is decisive, no model needed.
    assert format_signal("data.csv").slug == "spreadsheet"
    assert format_signal("deck.pptx").slug == "presentation"
    assert format_signal("note.txt") is None
    assert hard_signal("/x/data.csv", "data.csv").slug == "spreadsheet"

    # Path priors match whole folder components only — "vce" must NOT fire "cv".
    assert path_prior("/Users/me/Invoices/x.pdf")[0] == "invoice"
    assert path_prior("/Users/me/vce-english-app/x.pdf") is None
    assert path_prior("/Users/me/Random/x.pdf") is None

    # Browser origin is evidence, not a verdict — surfaced for the model to weigh.
    ev = gather_evidence("/x/Costco/page.pdf", "page.pdf")  # no real pdf → no web_origin
    assert ev.path_hint is None  # 'Costco' isn't a type folder; stays a hint-free

    # The model's answer is validated against the taxonomy.
    valid = set(BUILTIN_SLUGS)
    assert _parse('{"category":"invoice","confidence":"high","reason":"bill"}', valid).slug == "invoice"
    assert _parse('{"category":"decision","confidence":"high"}', valid).slug == NEEDS_REVIEW   # off-taxonomy
    assert _parse('{"category":"invoice","confidence":"low"}', valid).slug == NEEDS_REVIEW     # unsure
    assert _parse("not json", valid).slug == NEEDS_REVIEW

    # Hard signal wins even in test mode (no model call): a .csv IS a spreadsheet.
    r = classify_document("/x/products.csv", "products.csv", "a,b,c\n1,2,3")
    assert r.slug == "spreadsheet" and r.raw == "hard-signal"


def test_fast_classify_ignores_body_keywords_regression():
    """A contract whose body says 'acknowledge receipt of' must NOT be classified
    as a receipt — only the title line + filename are trusted (precision)."""
    from find_and_seek.ingest.summarise import fast_classify

    body = "EMPLOYMENT AGREEMENT\n\nThe parties hereby acknowledge receipt of the deposit."
    assert fast_classify("document", _mk_chunks(body), "contract_long.pdf")["document_type"] == "contract"


# ── recency-first queue ordering ──────────────────────────────────


def test_claim_queue_recency_first(tmp_path):
    """The worker should process the most recently modified/opened files first."""
    import os
    from find_and_seek.db.store import enqueue, claim_queue

    db = init_db(tmp_path / "q.db")
    for i, name in enumerate(["old.txt", "mid.txt", "new.txt"]):
        f = tmp_path / name
        f.write_text("x")
        os.utime(f, (1_000_000 + i * 1000, 1_000_000 + i * 1000))  # (atime, mtime)
        enqueue(db, str(f), "created")
    db.commit()
    # Claim one at a time — tests the cross-batch guarantee (recent selected first).
    # (RETURNING order within a single multi-row claim is unspecified in SQLite.)
    claimed = [claim_queue(db, 1)[0]["path"] for _ in range(3)]
    assert claimed == [str(tmp_path / "new.txt"), str(tmp_path / "mid.txt"), str(tmp_path / "old.txt")]


def test_claim_queue_priority_beats_recency(tmp_path):
    """A flagged (high-priority) folder's files jump ahead of newer low-priority ones."""
    import os
    from find_and_seek.db.store import enqueue, claim_queue

    db = init_db(tmp_path / "q.db")
    old = tmp_path / "old.txt"; old.write_text("x"); os.utime(old, (1_000_000, 1_000_000))
    new = tmp_path / "new.txt"; new.write_text("x"); os.utime(new, (9_000_000, 9_000_000))
    enqueue(db, str(old), "created", priority=10)  # user-flagged
    enqueue(db, str(new), "created", priority=0)
    db.commit()
    assert claim_queue(db, 1)[0]["path"] == str(old)  # priority beats recency


def test_ensure_columns_upgrades_legacy_queue(tmp_path):
    """A pre-existing DB without priority/recency gains them idempotently."""
    from find_and_seek.db.connection import get_connection
    from find_and_seek.db.migrations import _ensure_columns

    conn = get_connection(tmp_path / "legacy.db")
    conn.execute(
        "CREATE TABLE ingest_queue (path TEXT PRIMARY KEY, event_type TEXT NOT NULL, "
        "queued_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', "
        "attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT)"
    )
    _ensure_columns(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ingest_queue)")}
    assert {"priority", "recency"} <= cols
    _ensure_columns(conn)  # idempotent — second run must not raise

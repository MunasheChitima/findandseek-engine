"""User-correction overlay: field-level edits outrank the model card, survive
re-summarise, and report drift when the file changed after the correction."""

from __future__ import annotations

import pytest

from find_and_seek.db import edits
from find_and_seek.db.connection import init_db


@pytest.fixture()
def conn(tmp_path):
    c = init_db(tmp_path / "t.db")
    c.execute(
        "INSERT INTO files (path, filename, extension, content_hash, status) "
        "VALUES ('/x/a.pdf', 'a.pdf', 'pdf', 'hash-1', 'indexed')"
    )
    c.execute(
        "INSERT INTO file_summaries (file_id, summary_text, one_line_anchor, "
        "document_type, key_facts) VALUES (1, 'model summary', 'anchor', "
        "'report', '{\"amount\": \"$1\"}')"
    )
    c.commit()
    yield c
    c.close()


CARD = {
    "summary_text": "model summary",
    "one_line_anchor": "anchor",
    "document_type": "report",
    "key_facts": {"amount": "$1", "party": "Kestrel"},
}


def test_edit_overrides_field_and_marks_user_verified(conn):
    edits.set_edit(conn, 1, "summary_text", "the human's better summary")
    out = edits.apply_edits(conn, 1, dict(CARD))
    assert out["summary_text"] == "the human's better summary"
    assert out["user_verified"] == ["summary_text"]
    assert "edit_drift" not in out


def test_fact_level_edit_and_delete(conn):
    edits.set_edit(conn, 1, "key_facts.amount", "$1.2 billion")
    edits.set_edit(conn, 1, "key_facts.party", None)   # user deleted a wrong fact
    out = edits.apply_edits(conn, 1, dict(CARD))
    assert out["key_facts"]["amount"] == "$1.2 billion"
    assert "party" not in out["key_facts"]
    assert out["user_verified"] == ["key_facts.amount", "key_facts.party"]


def test_reapplying_model_card_never_clobbers_edit(conn):
    """Re-summarise regenerates the model layer; the overlay still wins."""
    edits.set_edit(conn, 1, "one_line_anchor", "human anchor")
    regenerated = dict(CARD, one_line_anchor="new model anchor v2")
    out = edits.apply_edits(conn, 1, regenerated)
    assert out["one_line_anchor"] == "human anchor"


def test_drift_reported_when_content_hash_changes(conn):
    edits.set_edit(conn, 1, "summary_text", "verified against hash-1")
    conn.execute("UPDATE files SET content_hash='hash-2' WHERE id=1")
    conn.commit()
    out = edits.apply_edits(conn, 1, dict(CARD))
    # The human's edit still applies — but the drift is surfaced, not silent.
    assert out["summary_text"] == "verified against hash-1"
    assert out["edit_drift"] is True


def test_clear_edit_restores_model_value(conn):
    edits.set_edit(conn, 1, "summary_text", "temp")
    edits.clear_edit(conn, 1, "summary_text")
    out = edits.apply_edits(conn, 1, dict(CARD))
    assert out["summary_text"] == "model summary"
    assert "user_verified" not in out


def test_rejects_non_card_fields_and_unknown_files(conn):
    with pytest.raises(ValueError):
        edits.set_edit(conn, 1, "status", "indexed")     # not a card field
    with pytest.raises(ValueError):
        edits.set_edit(conn, 1, "key_facts.", "x")       # empty fact key
    with pytest.raises(ValueError):
        edits.set_edit(conn, 999, "summary_text", "x")   # unknown file


def test_no_edits_is_a_passthrough(conn):
    out = edits.apply_edits(conn, 1, dict(CARD))
    assert out == CARD

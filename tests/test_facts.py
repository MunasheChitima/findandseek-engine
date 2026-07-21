"""Typed facts layer — normalization, extraction, write/backfill, and queries.

Runs against a temp DB with synthetic catalog rows (no inference, never the live
~/.findandseek/index.db).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from find_and_seek.db.connection import init_db
from find_and_seek.db.facts_query import aggregate_facts, query_facts
from find_and_seek.db.store import backfill_facts, iso_now, write_facts
from find_and_seek.ingest.facts import extract_facts, parse_date, parse_money, parse_number


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setenv("FINDANDSEEK_TEST", "1")
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "t.db")
        yield conn
        conn.close()


def _add_file(conn, *, path, key_facts=None, doc_type="invoice",
              entities=(), modified="2025-03-03T00:00:00+00:00"):
    p = Path(path)
    conn.execute(
        """INSERT INTO files (path, filename, extension, content_hash, size_bytes,
                              modified_at, indexed_at, file_type, status)
           VALUES (?, ?, ?, 'h', 100, ?, ?, 'pdf', 'indexed')""",
        (path, p.name, p.suffix.lstrip("."), modified, iso_now()),
    )
    fid = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]
    conn.execute(
        "INSERT INTO file_summaries (file_id, document_type, key_facts) VALUES (?, ?, ?)",
        (fid, doc_type, json.dumps(key_facts or {})),
    )
    for et, val in entities:
        conn.execute(
            "INSERT INTO file_entities (file_id, entity_type, entity_value, entity_raw) "
            "VALUES (?, ?, ?, ?)",
            (fid, et, val.strip().lower() if et != "money" else val, val),
        )
    conn.commit()
    return fid


# ── normalizers ──────────────────────────────────────────────────────


class TestNormalize:
    def test_money(self):
        # Bare ``$`` is ambiguous; this AU deployment defaults it to AUD.
        assert parse_money("$1,234.50") == (1234.5, "AUD")
        assert parse_money("£5") == (5.0, "GBP")
        assert parse_money("500 AUD") == (500.0, "AUD")
        assert parse_money("$118.93") == (118.93, "AUD")

    def test_money_explicit_code_wins(self):
        # An explicit 3-letter code is authoritative even with the AUD default.
        assert parse_money("$100 USD") == (100.0, "USD")
        assert parse_money("100 USD") == (100.0, "USD")

    def test_money_default_currency_env_override(self, monkeypatch):
        # FINDANDSEEK_DEFAULT_CURRENCY changes the bare-``$`` fallback; re-import
        # the module so the import-time constant is recomputed.
        import importlib

        import find_and_seek.ingest.facts as facts_mod

        monkeypatch.setenv("FINDANDSEEK_DEFAULT_CURRENCY", "USD")
        importlib.reload(facts_mod)
        try:
            assert facts_mod.parse_money("$50") == (50.0, "USD")
            # Explicit symbols/codes still authoritative.
            assert facts_mod.parse_money("£5") == (5.0, "GBP")
            assert facts_mod.parse_money("100 AUD") == (100.0, "AUD")
        finally:
            monkeypatch.delenv("FINDANDSEEK_DEFAULT_CURRENCY", raising=False)
            importlib.reload(facts_mod)

    def test_money_bad_env_falls_back_to_aud(self, monkeypatch):
        import importlib

        import find_and_seek.ingest.facts as facts_mod

        monkeypatch.setenv("FINDANDSEEK_DEFAULT_CURRENCY", "bitcoin")
        importlib.reload(facts_mod)
        try:
            assert facts_mod.parse_money("$50") == (50.0, "AUD")
        finally:
            monkeypatch.delenv("FINDANDSEEK_DEFAULT_CURRENCY", raising=False)
            importlib.reload(facts_mod)

    def test_money_rejects_bare_number(self):
        # No currency signal → not money (avoids treating every number as cash).
        assert parse_money("42") is None
        assert parse_money("hello") is None

    def test_number(self):
        assert parse_number(5) == 5.0
        assert parse_number("1,000") == 1000.0
        assert parse_number(True) is None
        assert parse_number("five") is None

    def test_date_iso_and_named(self):
        assert parse_date("2025-10-28") == "2025-10-28"
        assert parse_date("August 31, 2022") == "2022-08-31"
        assert parse_date("31 August 2022") == "2022-08-31"

    def test_date_day_first_convention(self):
        # 28/10/2025 is unambiguously day-first; 13/01/2026 too.
        assert parse_date("28/10/2025") == "2025-10-28"
        assert parse_date("13/01/2026") == "2026-01-13"
        # Ambiguous 03/04/2025 resolves day-first (AU/UK).
        assert parse_date("03/04/2025") == "2025-04-03"

    def test_date_rejects_garbage(self):
        assert parse_date("sometime next week") is None
        assert parse_date("99/99/9999") is None


# ── extraction ───────────────────────────────────────────────────────


class TestExtract:
    def test_money_and_date_from_key_facts(self):
        facts = extract_facts({"key_facts": {"amount": "$10", "date": "August 31, 2022"}})
        by_type = {f["fact_type"]: f for f in facts}
        assert by_type["money"]["value_number"] == 10.0
        assert by_type["money"]["unit"] == "AUD"  # bare ``$`` → configured default
        assert by_type["date"]["value_date"] == "2022-08-31"

    def test_nested_dict_recurses_one_level(self):
        facts = extract_facts({"key_facts": {
            "court": "Magistrates' Court",
            "details": {"case_nr": "R12248401", "hearing": "13/01/2026"},
        }})
        keys = {f["key"] for f in facts}
        assert "court" in keys
        assert "details.hearing" in keys
        hearing = next(f for f in facts if f["key"] == "details.hearing")
        assert hearing["fact_type"] == "date" and hearing["value_date"] == "2026-01-13"

    def test_list_becomes_attribute(self):
        facts = extract_facts({"key_facts": {"skills": ["python", "sql"]}})
        attr = next(f for f in facts if f["key"] == "skills")
        assert attr["fact_type"] == "attribute"
        assert "python" in attr["value_text"] and "sql" in attr["value_text"]

    def test_entities_normalized_and_deduped(self):
        ents = [
            {"entity_type": "money", "entity_value": "$50", "entity_raw": "$50", "chunk_id": 1},
            {"entity_type": "money", "entity_value": "$50", "entity_raw": "$50", "chunk_id": 1},
            {"entity_type": "person", "entity_value": "jane doe", "entity_raw": "Jane Doe", "chunk_id": 2},
            {"entity_type": "date", "entity_value": "28/10/2025", "entity_raw": "28/10/2025", "chunk_id": 3},
        ]
        facts = extract_facts(None, ents)
        money = [f for f in facts if f["fact_type"] == "money"]
        assert len(money) == 1 and money[0]["value_number"] == 50.0  # deduped
        assert any(f["fact_type"] == "person" and f["value_text"] == "jane doe" for f in facts)
        assert any(f["fact_type"] == "date" and f["value_date"] == "2025-10-28" for f in facts)


# ── write / backfill / query ─────────────────────────────────────────


class TestWriteAndQuery:
    def test_write_is_idempotent(self, db):
        fid = _add_file(db, path="/x/a.pdf", key_facts={"amount": "$10"})
        facts = extract_facts({"key_facts": {"amount": "$10"}}, [])
        write_facts(db, fid, facts)
        write_facts(db, fid, facts)  # again — must replace, not append
        db.commit()
        n = db.execute("SELECT COUNT(*) FROM facts WHERE file_id=?", (fid,)).fetchone()[0]
        assert n == len(facts)

    def test_backfill_then_query_money_threshold(self, db):
        _add_file(db, path="/x/small.pdf", key_facts={"amount": "$10"})
        _add_file(db, path="/x/big.pdf", key_facts={"total": "$5,000.00"})
        processed = backfill_facts(db)
        assert processed == 2
        big = query_facts(db, fact_type="money", min_number=1000)
        assert len(big) == 1 and big[0]["filename"] == "big.pdf"

    def test_query_date_range(self, db):
        _add_file(db, path="/x/old.pdf", key_facts={"date": "2020-01-01"})
        _add_file(db, path="/x/new.pdf", key_facts={"date": "2025-06-01"})
        backfill_facts(db)
        hits = query_facts(db, fact_type="date", start_date="2025-01-01", end_date="2025-12-31")
        assert len(hits) == 1 and hits[0]["filename"] == "new.pdf"

    def test_aggregate_sum(self, db):
        _add_file(db, path="/x/a.pdf", key_facts={"amount": "$10"})
        _add_file(db, path="/x/b.pdf", key_facts={"amount": "$15.50"})
        backfill_facts(db)
        agg = aggregate_facts(db, op="sum", fact_type="money")
        assert agg["value"] == pytest.approx(25.5)
        assert agg["n"] == 2

    def test_aggregate_rejects_bad_op(self, db):
        assert "error" in aggregate_facts(db, op="median", fact_type="money")

    def test_query_respects_scope(self, db):
        _add_file(db, path="/Users/me/Documents/a.pdf", key_facts={"amount": "$10"})
        _add_file(db, path="/Users/me/Downloads/b.pdf", key_facts={"amount": "$20"})
        backfill_facts(db)
        docs = query_facts(db, fact_type="money", scope="/Users/me/Documents")
        assert len(docs) == 1 and docs[0]["filename"] == "a.pdf"

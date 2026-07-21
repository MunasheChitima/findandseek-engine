"""Diagnostics capture exactly what's needed — and nothing sensitive.

The privacy contract: no document contents, no file paths, no search queries.
These tests lock that in (a leak here would be a real privacy regression)."""

from __future__ import annotations

import json

import pytest

from find_and_seek import diagnostics
from find_and_seek.db.connection import init_db


@pytest.fixture(autouse=True)
def _tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(diagnostics, "METRICS_PATH", tmp_path / "logs" / "metrics.jsonl")


def test_record_drops_non_allowlisted_fields():
    # Allow-listed scalars are kept; anything that could carry a path/query/text
    # is silently dropped.
    diagnostics.record("ingest_batch", files=3, seconds=1.2,
                       path="/Users/secret/Taxes.pdf", query="my divorce settlement")
    ev = diagnostics.recent_events()[-1]
    assert ev["files"] == 3 and ev["seconds"] == 1.2
    assert "path" not in ev and "query" not in ev


def test_record_error_stores_type_not_message():
    diagnostics.record_error("parse", ValueError("/Users/me/private/Will.pdf is corrupt"), ext=".pdf")
    ev = diagnostics.recent_events()[-1]
    assert ev["error_type"] == "ValueError"
    assert ev["stage"] == "parse" and ev["ext"] == ".pdf"
    assert "private" not in json.dumps(ev) and "Will" not in json.dumps(ev)


def test_collect_payload_has_no_paths_or_content(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    diagnostics.record("worker_start", tier="full")
    payload = diagnostics.collect(conn)
    assert payload["schema"].startswith("find-and-seek-diagnostics")
    assert payload["system"]["arch"] and payload["system"]["summary_model"]
    assert "ram_total_gb" in payload["system"]
    assert payload["runtime"]["files_indexed"] >= 0
    # Whole payload must not contain a home path.
    blob = json.dumps(payload)
    assert "/Users/" not in blob and "/home/" not in blob


def test_metrics_log_roundtrips():
    diagnostics.record("mem_pause", free_gb=1.1, threshold=1.5)
    diagnostics.record("ingest_batch", files=4, seconds=2.0)
    events = diagnostics.recent_events()
    assert [e["kind"] for e in events][-2:] == ["mem_pause", "ingest_batch"]

"""MCP beacon: tool-call tracking must never alter results, never raise, and
must classify escalation tools — the ground truth behind the tray's
"connected" state and the savings meter."""

from __future__ import annotations

from find_and_seek.mcp import beacon


def test_tracked_preserves_result_name_and_doc(monkeypatch):
    events = []
    monkeypatch.setattr(beacon, "ping", lambda *a, **k: events.append((a, k)))

    @beacon.tracked
    def search_files(query: str) -> dict:
        """the docstring FastMCP shows to agents"""
        return {"hits": [1, 2, 3]}

    assert search_files("q") == {"hits": [1, 2, 3]}
    assert search_files.__name__ == "search_files"
    assert "FastMCP" in search_files.__doc__
    assert events and events[0][1]["tool"] == "search_files"


def test_ping_records_metrics_and_survives_dead_sidecar(monkeypatch):
    recorded = []
    from find_and_seek import diagnostics

    monkeypatch.setattr(diagnostics, "record", lambda kind, **f: recorded.append((kind, f)))
    # Point the beacon at a port nothing listens on — must not raise or block.
    monkeypatch.setattr(beacon, "_api_url", lambda: "http://127.0.0.1:1/mcp/heartbeat")
    beacon.ping("tool", tool="get_chunk", payload={"text": "x" * 400})
    kind, fields = recorded[0]
    assert kind == "mcp_beacon"
    assert fields["tool"] == "get_chunk"
    assert fields["escalation"] is True          # full-context pull, not a card
    assert fields["tokens_est"] > 0


def test_ping_spawn_event_without_payload(monkeypatch):
    recorded = []
    from find_and_seek import diagnostics

    monkeypatch.setattr(diagnostics, "record", lambda kind, **f: recorded.append(f))
    monkeypatch.setattr(beacon, "_post", lambda payload: None)
    beacon.ping("spawn")
    assert recorded[0]["event"] == "spawn"
    assert recorded[0]["tokens_est"] == 0
    assert recorded[0]["escalation"] is False

"""MCP client heartbeat + savings counters.

The silent-failure problem: when an AI client stops routing through the engine,
nothing on our side executes — the fallback (the agent grepping files itself)
works, just expensively, so the user only finds out at their usage limit. The
beacon gives the tray app ground truth: every tool call (and the server spawn)
is recorded locally to metrics.jsonl AND reported to the sidecar's
``/mcp/heartbeat``, so "connected" in the UI means *observed tool calls*, not
"a config file exists".

The same events are the Tier-1 savings meter: tokens-served estimates per call
accumulate into the "context kept out of your sessions" number.

Client identity comes from ``FINDANDSEEK_MCP_CLIENT`` — set per client in the MCP
config blocks Connect-to-AI writes. Absent (older configs), "unknown".
Everything here is best-effort and must never slow or break a tool call.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from typing import Any

CLIENT = os.environ.get("FINDANDSEEK_MCP_CLIENT", "unknown")

# Escalation tools pull full context rather than triage cards — counted
# separately so the meter can show cards-vs-escalation balance honestly.
_ESCALATION_TOOLS = frozenset({"get_chunk", "get_file_context"})


def _api_url() -> str:
    from find_and_seek.config.settings import API_HOST, API_PORT

    host = "127.0.0.1" if API_HOST in ("0.0.0.0", "") else API_HOST
    return f"http://{host}:{API_PORT}/mcp/heartbeat"


def _post(payload: dict[str, Any]) -> None:
    try:
        req = urllib.request.Request(
            _api_url(),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2).close()
    except Exception:  # noqa: BLE001 — sidecar down is exactly what /health shows
        pass


def ping(event: str, tool: str | None = None, payload: Any = None) -> None:
    """Record one beacon event (spawn or tool call). Never raises, never blocks
    the caller — the HTTP report runs on a daemon thread."""
    tokens_est = 0
    if payload is not None:
        try:
            tokens_est = len(json.dumps(payload, default=str)) // 4
        except Exception:  # noqa: BLE001
            tokens_est = 0
    record: dict[str, Any] = {
        "client": CLIENT,
        "event": event,
        "tool": tool,
        "tokens_est": tokens_est,
        "escalation": tool in _ESCALATION_TOOLS,
    }
    try:
        from find_and_seek import diagnostics

        diagnostics.record("mcp_beacon", **record)
    except Exception:  # noqa: BLE001
        pass
    threading.Thread(target=_post, args=(record,), daemon=True).start()


def tracked(fn):
    """Decorator for MCP tool functions: report the call + result size, then
    return the result untouched. Signature/docstring preserved for FastMCP's
    introspection."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        ping("tool", tool=fn.__name__, payload=result)
        return result

    return wrapper

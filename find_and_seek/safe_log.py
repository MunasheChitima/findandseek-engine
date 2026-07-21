"""Log/error sanitisation.

The product promises "no file content in logs". Parser exceptions can embed
snippets of document text in their messages, and those messages get persisted
to ``ingest_queue.last_error`` and emitted to logs. ``safe_error`` keeps the
diagnostic signal (exception type + a short, single-line message) while
bounding how much arbitrary content can leak through.
"""

from __future__ import annotations

_MAX_MESSAGE = 300


def safe_error(exc: BaseException) -> str:
    """Return a compact, content-bounded description of an exception."""
    name = type(exc).__name__
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    if len(message) > _MAX_MESSAGE:
        message = message[:_MAX_MESSAGE] + "…"
    return f"{name}: {message}" if message else name

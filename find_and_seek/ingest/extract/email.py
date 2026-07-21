"""EML and MSG email extraction."""

from __future__ import annotations

from pathlib import Path

import mailparser

from find_and_seek.ingest.extract.base import ExtractResult, TextBlock


def _format_header(val) -> str:
    """mailparser returns from/to as a list of (display_name, address) tuples
    (email.utils.getaddresses format), not a plain string — stringify those
    into "Name <addr>" (or bare addr) instead of leaking the raw Python
    list-of-tuples repr into extracted text."""
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                name, addr = item
                parts.append(f"{name} <{addr}>" if name else addr)
            else:
                parts.append(str(item))
        return ", ".join(parts)
    return str(val)


def extract_eml(path: Path) -> ExtractResult:
    parsed = mailparser.parse_from_file(str(path))
    headers = []
    for key in ("from", "to", "subject", "date"):
        val = getattr(parsed, key, None)
        if val:
            headers.append(f"{key.title()}: {_format_header(val)}")

    body = parsed.text_plain[0] if parsed.text_plain else (parsed.body or "")
    attachments = [a.get("filename", "") for a in (parsed.attachments or [])]
    att_line = f"Attachments: {', '.join(attachments)}" if attachments else ""

    text = "\n".join(headers + [body, att_line]).strip()
    return ExtractResult(
        blocks=[TextBlock(text=text, source_type="text", location_ref="email body")],
        file_type="email",
    )


def extract_msg(path: Path) -> ExtractResult:
    try:
        import extract_msg
    except ImportError as e:
        raise RuntimeError("extract-msg not installed") from e

    msg = extract_msg.Message(str(path))
    parts = [
        f"From: {msg.sender}",
        f"To: {msg.to}",
        f"Subject: {msg.subject}",
        f"Date: {msg.date}",
        msg.body or "",
    ]
    if msg.attachments:
        names = [getattr(a, "longFilename", None) or getattr(a, "shortFilename", "") for a in msg.attachments]
        parts.append(f"Attachments: {', '.join(names)}")
    msg.close()
    return ExtractResult(
        blocks=[TextBlock(text="\n".join(parts), source_type="text", location_ref="email body")],
        file_type="email",
    )

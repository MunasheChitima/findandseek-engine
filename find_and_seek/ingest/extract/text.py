"""Plain text, markdown, CSV, JSON, XML."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import chardet

from find_and_seek.ingest.extract.base import ExtractResult, TextBlock


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    return raw.decode(encoding, errors="replace")


def extract_text_file(path: Path) -> ExtractResult:
    text = _read_text(path)
    return ExtractResult(
        blocks=[TextBlock(text=text, source_type="text", location_ref="document")],
        file_type="document",
    )


def extract_csv(path: Path) -> ExtractResult:
    text = _read_text(path)
    reader = csv.reader(text.splitlines())
    rows = list(reader)
    if not rows:
        return ExtractResult(blocks=[], file_type="data")

    header = rows[0]
    body = rows[1:6]
    summary_lines = [f"CSV with columns: {', '.join(header)}"]
    for i, row in enumerate(body, start=2):
        summary_lines.append(f"row {i}: {', '.join(row)}")
    block = TextBlock(
        text="\n".join(summary_lines),
        source_type="table",
        location_ref=f"rows 1-{min(len(rows), 6)}",
    )
    return ExtractResult(blocks=[block], file_type="data")


def extract_json(path: Path) -> ExtractResult:
    data = json.loads(path.read_text(encoding="utf-8"))

    def flatten(obj, prefix: str = "") -> list[str]:
        lines: list[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, (dict, list)):
                    lines.extend(flatten(v, key))
                else:
                    lines.append(f"{key}: {v}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:20]):
                lines.extend(flatten(item, f"{prefix}[{i}]"))
        else:
            lines.append(f"{prefix}: {obj}")
        return lines

    text = "\n".join(flatten(data))
    return ExtractResult(
        blocks=[TextBlock(text=text, source_type="text", location_ref="document")],
        file_type="data",
    )


def extract_xml(path: Path) -> ExtractResult:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    parts: list[str] = []

    def walk(el, depth=0):
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if el.text and el.text.strip():
            parts.append(f"{'  ' * depth}{tag}: {el.text.strip()}")
        for child in el:
            walk(child, depth + 1)

    walk(root)
    return ExtractResult(
        blocks=[
            TextBlock(
                text="\n".join(parts) or root.tag,
                source_type="text",
                location_ref="document",
            )
        ],
        file_type="data",
    )

"""Chunking engine — 512/1024 token windows with overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass

from find_and_seek.ingest.extract.base import TextBlock

DEFAULT_WINDOW = 512
DEFAULT_OVERLAP = 64
LONG_WINDOW = 1024

# Display-tuned PDFs sometimes emit per-glyph spacing: "E X P R E S S I O N".
# This corrupts FTS tokenisation (each letter is its own token), NER, embeddings,
# and chunk previews. We collapse runs of ≥4 space-separated uppercase single
# letters back into words before anything downstream sees the text.
# Conservative: uppercase-only avoids false positives on "I am a good person".
_LETTERSPACED_RE = re.compile(r'(?<!\w)([A-Z] (?:[A-Z] ){2,}[A-Z])(?!\w)')
# Collapse any resulting multiple spaces to a single space.
_MULTI_SPACE_RE = re.compile(r'  +')


def normalize_extracted_text(text: str) -> str:
    """Collapse letterspaced uppercase runs and normalize whitespace."""
    text = _LETTERSPACED_RE.sub(lambda m: m.group(0).replace(' ', ''), text)
    return _MULTI_SPACE_RE.sub(' ', text)


@dataclass
class Chunk:
    text: str
    source_type: str
    location_ref: str
    chunk_index: int
    token_estimate: int


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _is_long_form(blocks: list[TextBlock], path_hint: str = "") -> bool:
    total = sum(len(b.text) for b in blocks)
    if total > 20000:
        return True
    if "contract" in path_hint.lower():
        return True
    for b in blocks:
        if re.search(r"\b(whereas|herein|hereby|indemnif)\b", b.text, re.I):
            return True
    return False


def _split_text(text: str, window: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + window, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _prepare_blocks(blocks: list[TextBlock], use_long: bool) -> list[TextBlock]:
    if not use_long or len(blocks) <= 1:
        return blocks
    tables = [b for b in blocks if b.source_type == "table"]
    texts = [b for b in blocks if b.source_type != "table"]
    if not texts:
        return blocks
    merged = TextBlock(
        text="\n\n".join(b.text for b in texts),
        source_type="text",
        location_ref=f"{texts[0].location_ref} – {texts[-1].location_ref}",
    )
    return [merged] + tables


def chunk_blocks(
    blocks: list[TextBlock],
    path_hint: str = "",
    window: int | None = None,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    use_long = _is_long_form(blocks, path_hint)
    win = window or (LONG_WINDOW if use_long else DEFAULT_WINDOW)
    prepared = _prepare_blocks(blocks, use_long)
    chunks: list[Chunk] = []
    idx = 0

    for block in prepared:
        block = TextBlock(
            text=normalize_extracted_text(block.text),
            source_type=block.source_type,
            location_ref=block.location_ref,
        )
        if block.source_type == "table":
            chunks.append(
                Chunk(
                    text=block.text,
                    source_type=block.source_type,
                    location_ref=block.location_ref,
                    chunk_index=idx,
                    token_estimate=estimate_tokens(block.text),
                )
            )
            idx += 1
            continue

        parts = _split_text(block.text, win, overlap)
        if not parts and block.text.strip():
            parts = [block.text.strip()]

        for part in parts:
            chunks.append(
                Chunk(
                    text=part,
                    source_type=block.source_type,
                    location_ref=block.location_ref,
                    chunk_index=idx,
                    token_estimate=estimate_tokens(part),
                )
            )
            idx += 1

    return chunks

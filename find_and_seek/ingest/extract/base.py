"""Shared types for extraction output."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextBlock:
    text: str
    source_type: str  # text|ocr|image_caption|table
    location_ref: str


@dataclass
class ImageRegion:
    data: bytes
    mime: str
    location_ref: str


@dataclass
class ExtractResult:
    blocks: list[TextBlock] = field(default_factory=list)
    images: list[ImageRegion] = field(default_factory=list)
    page_count: int | None = None
    file_type: str = "document"

"""Image-only files routed to Florence."""

from __future__ import annotations

from pathlib import Path

from find_and_seek.ingest.extract.base import ExtractResult, ImageRegion


def extract_image(path: Path) -> ExtractResult:
    data = path.read_bytes()
    # Sniff content first: router.py can route here via a magic-byte sniff
    # (e.g. a JPEG saved under a .txt/.md suffix), in which case path.suffix
    # is the wrong, misleading extension — trust the bytes over the name.
    if data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    else:
        ext = path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, f"image/{ext}")
    return ExtractResult(
        blocks=[],
        images=[ImageRegion(data=data, mime=mime, location_ref="image")],
        file_type="image",
    )

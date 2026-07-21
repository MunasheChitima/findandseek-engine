"""Tesseract OCR backend — the production OCR path off-Mac.

Same contract as ``vision_macos``: returns an ``ocr`` chunk with the recognised
text (no image caption). Tesseract is invoked as a subprocess per image with a
hard timeout so a pathological bitmap can never freeze the ingest worker.

Requires the ``tesseract`` binary (apt install tesseract-ocr). A missing binary
is reported once and images simply aren't OCR'd — extraction of the rest of the
file is unaffected.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

from find_and_seek.ingest.chunk import Chunk

logger = logging.getLogger(__name__)

_TESSERACT_BIN = os.environ.get("FINDANDSEEK_TESSERACT_BIN", "tesseract")
_OCR_TIMEOUT = float(os.environ.get("FINDANDSEEK_OCR_TIMEOUT", "20"))

_missing_logged = False


def _available() -> bool:
    global _missing_logged
    if shutil.which(_TESSERACT_BIN):
        return True
    if not _missing_logged:
        logger.warning(
            "tesseract backend: %s not found on PATH "
            "(image OCR disabled until it is installed)",
            _TESSERACT_BIN,
        )
        _missing_logged = True
    return False


def _normalise(data: bytes, tmp: str) -> str:
    """Write image bytes to a file tesseract can read.

    Tesseract handles png/jpeg/tiff/bmp natively; anything else (heic, webp…)
    is converted through Pillow. Returns the path to feed tesseract.
    """
    raw = tmp + ".img"
    with open(raw, "wb") as f:
        f.write(data)
    if data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff" or data[:4] in (b"II*\x00", b"MM\x00*", b"BM\x00\x00"):
        return raw
    try:
        from PIL import Image

        png = tmp + ".png"
        with Image.open(raw) as img:
            img.convert("RGB").save(png, "PNG")
        return png
    except Exception:  # noqa: BLE001 — let tesseract try the raw bytes
        return raw


def process_image(data: bytes, location_ref: str) -> list[Chunk]:
    if not _available():
        return []
    with tempfile.TemporaryDirectory(prefix="fas-ocr-") as td:
        path = _normalise(data, os.path.join(td, "page"))
        try:
            proc = subprocess.run(
                [_TESSERACT_BIN, path, "stdout"],
                capture_output=True,
                timeout=_OCR_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning("tesseract backend: OCR timed out on %s", location_ref)
            return []
        except OSError as e:
            logger.warning("tesseract backend: failed to run: %s", e)
            return []
    text = proc.stdout.decode("utf-8", "replace").strip()
    if not text:
        return []
    return [
        Chunk(
            text=text,
            source_type="ocr",
            location_ref=location_ref,
            chunk_index=0,
            token_estimate=max(1, len(text) // 4),
        )
    ]

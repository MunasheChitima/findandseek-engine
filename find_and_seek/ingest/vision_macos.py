"""macOS-native OCR backend (Apple Vision via a long-lived Swift helper).

~100x faster than a vision LLM for printed/scanned text, on-device and
zero-egress. The helper (`tools/ocr/ocrtool`) is launched once in --server mode
and fed one image path per line; we read back a delimited text block per image.

This backend returns only an ``ocr`` chunk (no image caption). For document
indexing the recognised text is what matters; captions were a marginal extra
from the VLM path and are dropped here in exchange for the huge speedup.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from find_and_seek.ingest.chunk import Chunk

logger = logging.getLogger(__name__)

_END = "\x01FINDANDSEEK_OCR_END\x01"
# The committed `ocrtool` binary predates the FINDANDSEEK rename and still emits
# the legacy sentinel. Accept both so OCR works whether or not the helper has been
# rebuilt from the (renamed) source in tools/ocr/ — the value is a private wire
# token, only its match matters.
_END_LEGACY = "\x01ARCHIVIST_OCR_END\x01"
_ENDS = (_END, _END_LEGACY)
_TAGS = "\x02TAGS\x02"
_OCR_BIN = os.environ.get("FINDANDSEEK_OCR_BIN") or str(
    Path(__file__).resolve().parent.parent.parent / "tools" / "ocr" / "ocrtool"
)

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_missing_logged = False

# Per-image OCR budget. If the helper doesn't return the end sentinel within this
# many seconds (huge bitmap, undecodable image, or a silent helper crash) we kill
# it and move on — a stuck OCR call must never freeze the whole ingest worker.
_OCR_TIMEOUT = float(os.environ.get("FINDANDSEEK_OCR_TIMEOUT", "20"))


def _kill_proc() -> None:
    """Kill the OCR helper and clear the handle so the next call restarts it."""
    global _proc
    p = _proc
    _proc = None
    if p is not None:
        try:
            p.kill()
        except OSError:
            pass


def _server() -> subprocess.Popen | None:
    """Lazily start (and keep) the OCR helper in server mode."""
    global _proc, _missing_logged
    if _proc is not None and _proc.poll() is None:
        return _proc
    if not Path(_OCR_BIN).exists():
        if not _missing_logged:
            logger.warning(
                "macos vision backend: ocrtool not found at %s "
                "(image OCR disabled until the helper is present)",
                _OCR_BIN,
            )
            _missing_logged = True
        return None
    try:
        _proc = subprocess.Popen(
            [_OCR_BIN, "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except OSError as e:
        logger.warning("macos vision backend: failed to start ocrtool: %s", e)
        _proc = None
    return _proc


def _ocr_path(path: str) -> tuple[str, list[str]]:
    """Return (ocr_text, tags). Tags are only emitted for low-text images."""
    proc = _server()
    if proc is None or proc.stdin is None or proc.stdout is None:
        return "", []
    with _lock:
        # Watchdog: if the helper goes silent, kill it so the read below hits EOF
        # and returns instead of blocking the worker forever.
        watchdog = threading.Timer(_OCR_TIMEOUT, _kill_proc)
        watchdog.start()
        try:
            proc.stdin.write(path + "\n")
            proc.stdin.flush()
            text_lines: list[str] = []
            tags: list[str] = []
            completed = False
            for line in proc.stdout:
                stripped = line.rstrip("\n")
                if stripped in _ENDS:
                    completed = True
                    break
                if stripped.startswith(_TAGS):
                    tags = [t for t in stripped[len(_TAGS):].split(",") if t]
                else:
                    text_lines.append(stripped)
            if not completed:
                # EOF without a sentinel => watchdog killed a hung/slow helper.
                logger.warning("macos vision backend: OCR timed out/helper died on %s", path)
                _kill_proc()
                return "", []
            return "\n".join(text_lines).strip(), tags
        except (BrokenPipeError, OSError) as e:
            logger.warning("macos vision backend: helper communication failed: %s", e)
            _kill_proc()
            return "", []
        finally:
            watchdog.cancel()


def process_image(data: bytes, location_ref: str) -> list[Chunk]:
    # The helper reads from a file path; write the (already-encoded) image bytes
    # to a temp file. NSImage handles png/jpeg/tiff/heic transparently.
    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tf:
        tf.write(data)
        tmp = tf.name
    try:
        text, tags = _ocr_path(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    chunks: list[Chunk] = []
    if text:
        chunks.append(
            Chunk(
                text=text,
                source_type="ocr",
                location_ref=location_ref,
                chunk_index=0,
                token_estimate=max(1, len(text) // 4),
            )
        )
    if tags:
        tag_text = "Image contents: " + ", ".join(tags)
        chunks.append(
            Chunk(
                text=tag_text,
                source_type="image_caption",
                location_ref=location_ref,
                chunk_index=len(chunks),
                token_estimate=max(1, len(tag_text) // 4),
            )
        )
    return chunks

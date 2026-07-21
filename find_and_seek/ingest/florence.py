"""Image OCR — dispatcher over the vision backends (macOS Vision, hardcoded).

The production OCR engine is Apple's Vision framework via the ``ocrtool`` helper
(``vision_macos``): ~60× faster than the VLM per scanned page and recovers more
text verbatim (owner decision 2026-06-11 — the only OCR path; not overridable).
The MLX VLM (``vision_mlx``, Qwen2.5-VL) remains in-tree for bench/eval use only.
The ``florence`` name survives as the model-manager role label only.
``VISION_PROMPT`` and ``build_vision_chunks`` are the shared contract the
backends produce against.
"""

from __future__ import annotations

import logging

from find_and_seek.ingest.chunk import Chunk

logger = logging.getLogger(__name__)


VISION_PROMPT = (
    "Extract all visible text from this image (OCR). "
    "Then on a new line starting with CAPTION:, describe what the image shows in one sentence."
)


def build_vision_chunks(text: str, location_ref: str) -> list[Chunk]:
    """Split a vision model's 'OCR ... CAPTION: ...' reply into ocr/caption chunks.
    Shared by the vision backends so the chunk contract is identical."""
    text = (text or "").strip()
    text = text.split("<|channel|>")[-1] if "<|channel|>" in text else text
    chunks: list[Chunk] = []
    if not text:
        return chunks

    ocr_part, caption_part = text, ""
    if "CAPTION:" in text:
        ocr_part, _, caption_part = text.partition("CAPTION:")
        ocr_part, caption_part = ocr_part.strip(), caption_part.strip()
    elif "\n" in text:
        lines = text.split("\n", 1)
        ocr_part = lines[0].strip()
        caption_part = lines[1].strip() if len(lines) > 1 else ""

    if ocr_part:
        chunks.append(Chunk(text=ocr_part, source_type="ocr", location_ref=location_ref,
                            chunk_index=0, token_estimate=max(1, len(ocr_part) // 4)))
    if caption_part:
        chunks.append(Chunk(text=caption_part, source_type="image_caption", location_ref=location_ref,
                            chunk_index=1, token_estimate=max(1, len(caption_part) // 4)))
    return chunks


def process_image(data: bytes, location_ref: str) -> list[Chunk]:
    from find_and_seek.config.settings import role_backend

    backend = role_backend("vision")
    if backend == "test":
        return []

    if backend == "tesseract":
        # The production OCR path off-Mac (see role_backend).
        from find_and_seek.ingest import vision_tesseract

        try:
            return vision_tesseract.process_image(data, location_ref)
        except Exception as e:  # noqa: BLE001
            logger.warning("tesseract OCR failed (%s) — image at %s not OCR'd.",
                           e, location_ref)
            return []

    # macOS Vision OCR is the only production OCR path on Macs (see role_backend).
    # A missing/broken ocrtool helper is a packaging bug, not a degrade case —
    # warn loudly so it gets fixed instead of silently indexing scans textless.
    from find_and_seek.ingest import vision_macos

    try:
        return vision_macos.process_image(data, location_ref)
    except Exception as e:  # noqa: BLE001
        logger.warning("macOS Vision OCR unavailable (%s) — image at %s not OCR'd. "
                       "Check the ocrtool helper is present and executable.", e, location_ref)
        return []

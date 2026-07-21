"""PDF text + embedded images."""

from __future__ import annotations

from pathlib import Path

import fitz

from find_and_seek.ingest.extract.base import ExtractResult, ImageRegion, TextBlock


def extract_pdf(path: Path) -> ExtractResult:
    import os
    from find_and_seek.config.settings import ocr_enabled_default
    _ocr_env = os.environ.get("FINDANDSEEK_ENABLE_OCR")
    enable_ocr = ocr_enabled_default() if _ocr_env is None else _ocr_env.lower() in ("1", "true", "yes")

    doc = fitz.open(str(path))
    blocks: list[TextBlock] = []
    images: list[ImageRegion] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        loc = f"page {page_num + 1}"
        if text:
            blocks.append(TextBlock(text=text, source_type="text", location_ref=loc))

        if enable_ocr:
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                    images.append(
                        ImageRegion(
                            data=base["image"],
                            mime=f"image/{base['ext']}",
                            location_ref=f"page {page_num + 1} image {img_index + 1}",
                        )
                    )
                except (ValueError, RuntimeError):
                    continue

    return ExtractResult(
        blocks=blocks,
        images=images,
        page_count=len(doc),
        file_type="document",
    )


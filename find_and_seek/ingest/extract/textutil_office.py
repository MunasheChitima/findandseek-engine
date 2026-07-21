"""Text extraction for legacy / ODF office formats via macOS `textutil`.

The router watches `.rtf .doc .odt .odp .ods` but had no extractor for them, so
every such file ingested to *nothing* — the engine was blind to them (real bug
found on the live index: a 64k-char tender chat saved as `.rtf`, a website list
as `.odt`). macOS ships `/usr/bin/textutil`, which converts all of these to plain
text with no extra dependency.

`textutil` also *sniffs the real format*, so it doubles as a rescue for files with
a lying extension — e.g. a document saved as `.docx`/`.xlsx` that is actually RTF
(python-docx/openpyxl reject those with "not a zip"). It must NOT be used for PDFs
(it returns raw PDF bytes), so it is only wired to the office formats and used as a
fallback for OOXML container failures.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from find_and_seek.ingest.extract.base import ExtractResult, TextBlock
from find_and_seek.ingest.extract.router import UnsupportedFormatError

_TEXTUTIL = "/usr/bin/textutil"


def textutil_available() -> bool:
    return (
        Path(_TEXTUTIL).exists()
        or shutil.which("textutil") is not None
        or shutil.which("soffice") is not None
        or shutil.which("libreoffice") is not None
    )


def _extract_via_soffice(soffice: str, path: Path, file_type: str) -> ExtractResult:
    """Convert via LibreOffice headless. Writes <stem>.txt into a temp dir —
    soffice has no stdout mode and must not write next to the source file."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="fas-soffice-") as td:
        proc = subprocess.run(
            [soffice, "--headless", "--convert-to", "txt:Text",
             "--outdir", td, str(path)],
            capture_output=True,
            timeout=120,
        )
        out = Path(td) / (path.stem + ".txt")
        text = out.read_text("utf-8", errors="replace").strip() if out.exists() else ""
    if not text:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise UnsupportedFormatError(f"soffice produced no text for {path.name}: {err[:80]}")
    return ExtractResult(
        blocks=[TextBlock(text=text, source_type="text", location_ref="document")],
        file_type=file_type,
    )


def extract_via_textutil(path: Path, file_type: str = "document") -> ExtractResult:
    """Convert a legacy/ODF office file to text via macOS textutil."""
    exe = _TEXTUTIL if Path(_TEXTUTIL).exists() else shutil.which("textutil")
    if not exe:
        # Fallback for RTF on Linux/Windows where textutil is missing
        try:
            content_bytes = path.read_bytes()
            if content_bytes.startswith(b"{\\rtf"):
                from striprtf.striprtf import rtf_to_text
                try:
                    content_str = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content_str = content_bytes.decode("cp1252", errors="replace")
                text = rtf_to_text(content_str).strip()
                if not text:
                    raise UnsupportedFormatError(f"striprtf produced no text for {path.name}")
                return ExtractResult(
                    blocks=[TextBlock(text=text, source_type="text", location_ref="document")],
                    file_type=file_type,
                )
        except UnsupportedFormatError:
            raise
        except Exception as e:
            raise UnsupportedFormatError(f"no textutil and striprtf failed: {e}") from e

        # Non-macOS: LibreOffice converts the same legacy/ODF formats to text.
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if soffice:
            return _extract_via_soffice(soffice, path, file_type)

        # Neither converter present: honestly unsupported rather than empty.
        raise UnsupportedFormatError(f"no textutil or soffice to read {path.suffix}")
    proc = subprocess.run(
        [exe, "-convert", "txt", "-stdout", str(path)],
        capture_output=True,
        timeout=120,
    )
    text = proc.stdout.decode("utf-8", "replace").strip()
    if not text:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise UnsupportedFormatError(f"textutil produced no text for {path.name}: {err[:80]}")
    return ExtractResult(
        blocks=[TextBlock(text=text, source_type="text", location_ref="document")],
        file_type=file_type,
    )

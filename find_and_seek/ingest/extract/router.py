"""Extension → extractor dispatch."""

from __future__ import annotations

import zipfile
from pathlib import Path

from find_and_seek.ingest.extract.base import ExtractResult
from find_and_seek.ingest.extract.docx import extract_docx
from find_and_seek.ingest.extract.email import extract_eml, extract_msg
from find_and_seek.ingest.extract.image import extract_image
from find_and_seek.ingest.extract.pdf import extract_pdf
from find_and_seek.ingest.extract.pptx import extract_pptx
from find_and_seek.ingest.extract.text import (
    extract_csv,
    extract_json,
    extract_text_file,
    extract_xml,
)
from find_and_seek.ingest.extract.xlsx import extract_xlsx

# What v1 indexes: real user documents — office formats, text/markdown, mail,
# spreadsheets, slides, images.
DOCUMENT_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".txt", ".md",
    ".xlsx", ".xls", ".csv", ".tsv", ".ods",
    ".pptx", ".ppt", ".odp",
    ".eml", ".msg",
    ".jpg", ".jpeg", ".png", ".tiff", ".heic",
})

# Source code, config and structured-data formats. DEFERRED: a documents product
# shouldn't flood the index with a dev workspace's .js/.ts/.json (that was ~38% of
# the live index — pure search noise). Code indexing is a planned opt-in (v1.5);
# enabling it is just unioning this set back in — the EXTRACTORS below already
# handle every one — gated on the index-code setting.
CODE_EXTENSIONS = frozenset({
    ".py", ".swift", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".cs", ".rb", ".sh",
    ".toml", ".yaml", ".yml", ".json", ".xml",
})


def _code_indexing_enabled() -> bool:
    """Opt-in code indexing (v1.5). Off by default; ``FINDANDSEEK_INDEX_CODE=1`` turns
    it on today, and this gets a user-facing toggle in v1.5."""
    import os

    return os.environ.get("FINDANDSEEK_INDEX_CODE", "0").strip().lower() in ("1", "true", "yes")


# v1 = documents only. Resolved at import; code extensions join only when opted in.
WATCHED_EXTENSIONS = DOCUMENT_EXTENSIONS | (
    CODE_EXTENSIONS if _code_indexing_enabled() else frozenset()
)


def _extract_textutil(p: Path) -> ExtractResult:
    # Imported lazily to avoid a circular import (textutil_office imports this module).
    from find_and_seek.ingest.extract.textutil_office import extract_via_textutil

    return extract_via_textutil(p)


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".pptx": extract_pptx,
    ".xlsx": extract_xlsx,
    ".xls": extract_xlsx,
    ".csv": extract_csv,
    ".tsv": extract_csv,
    ".txt": extract_text_file,
    ".md": extract_text_file,
    ".json": extract_json,
    ".xml": extract_xml,
    ".eml": extract_eml,
    ".msg": extract_msg,
    # Legacy / ODF office formats — read via macOS textutil (see textutil_office).
    # These were watched but had no extractor, so they ingested to nothing.
    ".rtf": _extract_textutil,
    ".doc": _extract_textutil,
    ".odt": _extract_textutil,
    ".odp": _extract_textutil,
    ".ods": _extract_textutil,
    ".ppt": _extract_textutil,
    ".jpg": extract_image,
    ".jpeg": extract_image,
    ".png": extract_image,
    ".tiff": extract_image,
    ".heic": extract_image,
    # Source code + config — plain text read is correct for all of these
    ".py": extract_text_file,
    ".swift": extract_text_file,
    ".js": extract_text_file,
    ".ts": extract_text_file,
    ".tsx": extract_text_file,
    ".jsx": extract_text_file,
    ".go": extract_text_file,
    ".rs": extract_text_file,
    ".java": extract_text_file,
    ".c": extract_text_file,
    ".cpp": extract_text_file,
    ".h": extract_text_file,
    ".cs": extract_text_file,
    ".rb": extract_text_file,
    ".sh": extract_text_file,
    ".toml": extract_text_file,
    ".yaml": extract_text_file,
    ".yml": extract_text_file,
}


class UnsupportedFormatError(Exception):
    pass


class ExtractError(Exception):
    pass


# A failed OOXML container almost always means a mislabeled file (e.g. RTF saved as
# .docx) — python-docx/openpyxl reject it with these. textutil sniffs the real
# format, so it's a good rescue. Matched against the exception text.
_OOXML_CONTAINER_ERRORS = ("not a zip file", "package not found", "bad zip", "file is not a zip")
_OOXML_EXTS = {".docx", ".xlsx", ".pptx"}

_ODF_MIMETYPES = {
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.oasis.opendocument.presentation": ".odp",
}


def sniff_format(path: Path) -> str | None:
    """Canonical extension from leading bytes, or None if unrecognized."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if head.startswith(b"%PDF"):
            return ".pdf"
        if head.startswith(b"{\\rtf"):
            return ".rtf"
        if head.startswith(b"\xd0\xcf\x11\xe0"):
            return ".doc"
        if head.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if head.startswith(b"PK\x03\x04"):
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                if any(n.startswith("word/") for n in names):
                    return ".docx"
                if any(n.startswith("xl/") for n in names):
                    return ".xlsx"
                if any(n.startswith("ppt/") for n in names):
                    return ".pptx"
                if "mimetype" in names or "content.xml" in names:
                    mimetype = zf.read("mimetype").decode("ascii", "ignore").strip()
                    return _ODF_MIMETYPES.get(mimetype)
        return None
    except Exception:  # noqa: BLE001 — sniffing is best-effort
        return None


def extract_file(path: Path | str) -> ExtractResult:
    p = Path(path)
    ext = p.suffix.lower()
    # Sniff unconditionally, not just when the extension is unrecognized. A
    # binary container (docx/xlsx/pptx/pdf/rtf/doc — anything sniff_format
    # knows) saved under a .txt/.md/.csv/.json/.xml suffix IS a recognized
    # extension, so the old "only sniff if unrecognized" check never fired —
    # and extract_text_file's errors="replace" decode never raises, so the
    # binary payload silently "succeeded" as garbage text (tens of thousands
    # of mangled chars) instead of being routed to the extractor that could
    # actually read it. Confirmed live: several corpus files with .csv/.txt/
    # .md suffixes are valid .docx files under the hood.
    sniffed = sniff_format(p)
    if sniffed and sniffed in EXTRACTORS and sniffed != ext:
        ext = sniffed
    elif ext not in EXTRACTORS:
        if sniffed in EXTRACTORS:
            ext = sniffed
        else:
            raise UnsupportedFormatError(f"Unsupported extension: {ext}")

    try:
        return EXTRACTORS[ext](p)
    except UnsupportedFormatError:
        raise
    except Exception as e:
        # The extension is the usual liar — sniff the real format before giving up,
        # so a real document isn't lost to a wrong suffix.
        sniffed = sniff_format(p)
        if sniffed and sniffed != ext and sniffed in EXTRACTORS:
            try:
                return EXTRACTORS[sniffed](p)
            except Exception:  # noqa: BLE001 — fall through to the original error
                pass
        if ext in _OOXML_EXTS and any(s in str(e).lower() for s in _OOXML_CONTAINER_ERRORS):
            try:
                return _extract_textutil(p)
            except Exception:  # noqa: BLE001 — fall through to the original error
                pass
        raise ExtractError(str(e)) from e

"""Evidence-based classification signals.

Real data taught us two things:
  1. The strongest type signals are often NOT in the extracted text — they're in
     the format, the embedded metadata, and the path.
  2. But some metadata is ORIGIN, not TYPE. A measured precision check showed
     "printed from a web browser" (Skia/PDF) fires on ~20% of PDFs — including
     receipts, medical records and articles, not just store pages. So browser
     origin is *evidence we hand the classifier*, never a verdict on its own.

Hard signals (`Signal`) are trusted to set the type directly. Soft evidence
(`Evidence`) is passed to the LLM so it decides with more context (a browser
-printed store page → web-page; a browser-printed receipt → receipt).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    slug: str
    strength: str   # "decisive" (format) | "strong" (precise metadata)
    reason: str


@dataclass(frozen=True)
class Evidence:
    """Soft context for the classifier — hints, not verdicts."""
    web_origin: bool = False
    path_hint: str | None = None        # a likely slug from the folder name
    path_reason: str | None = None
    notes: list[str] = None             # human-readable context lines

    def as_prompt(self) -> str:
        lines = list(self.notes or [])
        if self.web_origin:
            lines.append("This PDF was printed/saved from a web browser — it may be a "
                         "saved web page or online listing, OR an online receipt/record "
                         "you viewed in a browser. Judge by the content which it is.")
        if self.path_hint:
            lines.append(f"It is stored in a folder suggesting '{self.path_hint}' ({self.path_reason}).")
        return "\n".join(f"- {ln}" for ln in lines)


# ── Hard signal 1: format fixes the type ─────────────────────────────
_FORMAT_SLUG = {
    "csv": "spreadsheet", "tsv": "spreadsheet", "xlsx": "spreadsheet",
    "xls": "spreadsheet", "numbers": "spreadsheet",
    "pptx": "presentation", "ppt": "presentation", "key": "presentation",
    "eml": "email", "msg": "email",
}


def format_signal(filename: str) -> Signal | None:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    slug = _FORMAT_SLUG.get(ext)
    return Signal(slug, "decisive", f".{ext} file") if slug else None


# ── Hard signal 2: a fillable XFA/LiveCycle PDF is a form ────────────
def _pdf_meta(path: str) -> dict:
    if not path.lower().endswith(".pdf") or not os.path.exists(path):
        return {}
    try:
        import fitz
        doc = fitz.open(path)
        m = doc.metadata or {}
        doc.close()
        return m
    except Exception:  # noqa: BLE001
        return {}


def form_signal(path: str) -> Signal | None:
    m = _pdf_meta(path)
    producer = (m.get("producer") or "").lower()
    creator = (m.get("creator") or "").lower()
    if "livecycle" in creator or "xfa" in producer or "xml form" in (producer + creator):
        return Signal("form", "strong", "an Adobe XML (XFA) form")
    return None


def hard_signal(path: str, filename: str) -> Signal | None:
    """A type we trust without the model: format, then precise metadata."""
    return format_signal(filename) or form_signal(path)


# ── Soft evidence: origin + path, handed to the classifier ───────────
_BROWSER_PRODUCERS = ("skia/pdf",)
_BROWSER_CREATORS = ("mozilla/", "chrome", "safari", "chromium", "headlesschrome")

_PATH_PRIORS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"invoices", "invoice"}), "invoice"),
    (frozenset({"receipts", "receipt"}), "receipt"),
    (frozenset({"contracts", "contract", "agreements"}), "contract"),
    (frozenset({"resumes", "resume", "cvs"}), "cv"),
    (frozenset({"study-designs", "study designs", "curriculum", "syllabus"}), "reference"),
)


def web_origin(path: str) -> bool:
    m = _pdf_meta(path)
    producer = (m.get("producer") or "").lower()
    creator = (m.get("creator") or "").lower()
    return any(p in producer for p in _BROWSER_PRODUCERS) or any(c in creator for c in _BROWSER_CREATORS)


def path_prior(path: str) -> tuple[str, str] | None:
    parts = {p.strip().lower() for p in path.replace("\\", "/").split("/") if p.strip()}
    for names, slug in _PATH_PRIORS:
        hit = names & parts
        if hit:
            return slug, f"a '{next(iter(hit))}' folder"
    return None


def gather_evidence(path: str, filename: str) -> Evidence:
    pp = path_prior(path)
    return Evidence(
        web_origin=web_origin(path),
        path_hint=pp[0] if pp else None,
        path_reason=pp[1] if pp else None,
        notes=[],
    )

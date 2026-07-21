"""Canonical document-type taxonomy and tag-name normalisation.

The live catalog shows real `document_type` drift (`slide_deck` vs `slide deck`,
freeform `product`/`product_page`/`product_description`, `csv`, ...). Organize
canonicalises types against a fixed taxonomy *before* using them to drive folder
structure or tags, or the proposed tree fragments (design §4, §7.1).
"""

from __future__ import annotations

import re

# Fixed taxonomy (design §7.1). Everything maps into one of these.
CANONICAL_TYPES: tuple[str, ...] = (
    "invoice",
    "receipt",
    "payslip",
    "contract",
    "report",
    "cv",
    "letter",
    "email",
    "chat",
    "spreadsheet",
    "slide-deck",
    "note",
    "product",
    "other",
)

# Explicit collapse map for the drift seen in the catalog. Keys are compared
# after lower-casing and trimming; everything not covered falls through to the
# slug/sanitise step below, then to `other` if still unknown.
_NORMALIZATION: dict[str, str] = {
    "slide_deck": "slide-deck",
    "slide deck": "slide-deck",
    "slidedeck": "slide-deck",
    "presentation": "slide-deck",
    "deck": "slide-deck",
    "csv": "spreadsheet",
    "excel": "spreadsheet",
    "xlsx": "spreadsheet",
    "table": "spreadsheet",
    "product_page": "product",
    "product page": "product",
    "product_description": "product",
    "product description": "product",
    "cover correspondence": "letter",
    "cover_letter": "letter",
    "correspondence": "letter",
    "resume": "cv",
    "curriculum vitae": "cv",
    "job_posting": "other",
    "job posting": "other",
    "assessment_plan": "report",
    "template": "other",
    "memo": "note",
    "pay slip": "payslip",
    "pay_slip": "payslip",
    "salary statement": "payslip",
    "salary summary": "payslip",
    "chat_export": "chat",
    "chat export": "chat",
    "transcript": "chat",
    "conversation": "chat",
    "message": "chat",
    "instant message": "chat",
    "agreement": "contract",
    "bill": "invoice",
    # NB: a "statement" is ambiguous (bank/claim/witness statement) and is NOT an
    # invoice — let it fall through to "other" rather than mislabel it.
}


def canonicalize_type(raw: str | None) -> str:
    """Map a freeform `document_type` to one of CANONICAL_TYPES.

    Unknown or empty values collapse to ``other`` so the folder tree never
    fragments on a one-off label.
    """
    if not raw:
        return "other"
    key = raw.strip().lower()
    if key in _NORMALIZATION:
        return _NORMALIZATION[key]
    # Already canonical (e.g. "invoice", or "slide-deck" written directly).
    if key in CANONICAL_TYPES:
        return key
    # Try a hyphen/underscore-normalised form before giving up.
    norm = key.replace("_", "-").replace(" ", "-")
    if norm in CANONICAL_TYPES:
        return norm
    return "other"


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str | None) -> str:
    """Lower-case, hyphenated, filesystem/tag-safe slug. '' for empty input."""
    if not text:
        return ""
    s = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return s


# Title-case folder name for each canonical type (used by the planner).
TYPE_FOLDER: dict[str, str] = {
    "invoice": "Invoices",
    "receipt": "Receipts",
    "payslip": "Payslips",
    "contract": "Contracts",
    "report": "Reports",
    "cv": "CVs & Resumes",
    "letter": "Letters",
    "email": "Email",
    "chat": "Chats & Transcripts",
    "spreadsheet": "Spreadsheets",
    "slide-deck": "Presentations",
    "note": "Notes",
    "product": "Product",
    "other": "Other",
}


def type_folder(canonical_type: str) -> str:
    """Human folder name for a canonical type."""
    return TYPE_FOLDER.get(canonical_type, "Other")

"""Composite-document detection.

Some PDFs are not one document but a *stack* of unrelated ones scanned into a
single file (e.g. a tray of receipts: an MRI claim on page 1, a shop invoice on
page 2, a pharmacy receipt on page 3 …). Indexed naively, the whole file gets a
single anchor describing only page 1 — every other document is misrepresented
and hard to recognise in results.

This module detects that case from already-extracted, page-tagged chunks, with no
re-reading of the file. The signal: a genuine multi-page document (a lease, a
report) reuses vocabulary across its pages, so consecutive pages overlap heavily;
a stack of unrelated scans shares almost nothing page-to-page.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from find_and_seek.ingest.chunk import Chunk

_PAGE_RE = re.compile(r"page\s+(\d+)", re.I)
# Below this average consecutive-page word overlap (Jaccard), the pages are
# mutually unrelated → treat the file as a bundle of separate documents. Kept
# conservative (precision over recall): wrongly splitting a coherent document is
# worse than missing a bundle. Env-tunable.
COMPOSITE_OVERLAP_MAX = float(os.environ.get("FINDANDSEEK_COMPOSITE_OVERLAP_MAX", "0.08"))
# Need at least this many pages before "looks like a stack" is meaningful.
MIN_PAGES = 3
# A real scanned stack of admin documents is small; a 50/180-page file with low
# page overlap is a long single document (dictionary, glossary, report), not a
# bundle. Above this many pages we never treat it as a stack.
MAX_PAGES = 25
# The overlap signal needs enough Latin words per page to be meaningful. A
# sparse or non-Latin (e.g. CJK) page yields a near-empty word set and a
# spuriously low overlap — don't judge those.
MIN_MEDIAN_WORDS = 12
# Low overlap alone is NOT enough — dictionaries, web-page captures and articles
# also have low page-to-page overlap yet are ONE document. A genuine bundle is a
# stack of documents, so most pages must *start like a new document* (a title /
# letterhead / doc-type keyword). This fraction must do so.
MIN_DOC_START_FRACTION = 0.5

# Header keywords that signal the top of a distinct document (invoice, letter,
# statement, notice, …). Matched against each page's first few lines.
_DOC_START_RE = re.compile(
    r"\b(tax invoice|invoice|statement of (claim|account)|account statement|receipt|"
    r"remittance|purchase order|penalty|infringement|reminder notice|notice of|"
    r"prescription|pharmacy|medical centre|medical center|clinic|certificate|"
    r"dear\s+\w|payslip|pay advice|claim|abn[:\s]|policy number|reference number|"
    r"to the operator|from:|subject:|re:)\b", re.I)
# A letterhead/title is often an ALL-CAPS line of a few words.
_CAPS_TITLE_RE = re.compile(r"^[A-Z][A-Z &.,'\-]{6,}$")


def _looks_like_doc_start(text: str) -> bool:
    """Does this page begin like its own document (title / letterhead / doc-type
    keyword)? Checks the first handful of non-empty lines."""
    seen = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _DOC_START_RE.search(line) or _CAPS_TITLE_RE.match(line):
            return True
        seen += 1
        if seen >= 5:
            break
    return False


@dataclass
class Section:
    page: int
    text: str


def page_of(location_ref: str | None) -> int | None:
    """Page number parsed from a chunk's location_ref ('page 2 image 1' → 2)."""
    m = _PAGE_RE.search(location_ref or "")
    return int(m.group(1)) if m else None


_page_of = page_of  # internal alias


def _content_words(text: str) -> set[str]:
    # 4+ letter tokens only — ignores numbers/punctuation/short glue words so the
    # overlap reflects real shared subject matter, not "the/and/2024".
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def page_sections(chunks: list[Chunk]) -> list[Section]:
    """Group page-tagged chunks into one Section per page, in page order."""
    pages: dict[int, list[str]] = {}
    for c in chunks:
        p = _page_of(c.location_ref)
        if p is None:
            continue
        pages.setdefault(p, []).append(c.text)
    return [Section(page=p, text="\n".join(t)) for p, t in sorted(pages.items())]


def mean_consecutive_overlap(sections: list[Section]) -> float:
    """Average Jaccard word-overlap between consecutive pages (0..1)."""
    sets = [_content_words(s.text) for s in sections]
    sims: list[float] = []
    for a, b in zip(sets, sets[1:]):
        if not a or not b:
            sims.append(0.0)
            continue
        sims.append(len(a & b) / len(a | b))
    return sum(sims) / len(sims) if sims else 1.0


# Furniture/boilerplate lines that don't make a useful per-page label.
_BOILERPLATE_RE = re.compile(
    r"(confidential|all rights reserved|copyright|©|page \d+|www\.|https?://|"
    r"binding margin|do not write|@[\w.-]+\.\w+)", re.I)


def section_anchor(text: str, max_words: int = 18) -> str:
    """A one-line label for a single page/section — its first line with real
    content, skipping furniture. Deterministic (no model)."""
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if len(re.findall(r"[A-Za-z]{2,}", line)) < 3:
            continue
        if _BOILERPLATE_RE.search(line):
            continue
        return " ".join(line.split()[:max_words])
    for line in lines:  # fallback: first non-empty line
        if line.strip():
            return " ".join(line.strip().split()[:max_words])
    return ""


def composite_summary(chunks: list[Chunk]) -> dict | None:
    """If the file is a bundle of unrelated documents, return an honest doc-level
    summary plus a per-page ``section_anchors`` map. Else None. No model — works
    over already-extracted chunks, so it costs nothing at re-summarise time."""
    secs = detect_composite(chunks)
    if not secs:
        return None
    anchors = {s.page: section_anchor(s.text) for s in secs}
    labels = [a for a in anchors.values() if a]
    n = len(secs)
    shown = "; ".join(labels[:3])
    extra = f" (+{n - 3} more)" if n > 3 else ""
    one_line = f"Scanned bundle — {n} documents: {shown}{extra}" if labels else f"Scanned bundle of {n} documents"
    body = "\n".join(f"- p{p}: {a}" for p, a in anchors.items() if a)
    return {
        "one_line_anchor": one_line[:200],
        "summary_text": f"A scanned file containing {n} separate documents:\n{body}",
        "section_anchors": anchors,
    }


def _median(xs: list[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def detect_composite(chunks: list[Chunk]) -> list[Section] | None:
    """Return the per-page sections if the file is a bundle of unrelated
    documents, else None. Conservative: needs several pages, enough text per page
    to judge, AND low page-to-page vocabulary overlap — so a normal multi-page
    document is never split, and sparse/non-Latin pages aren't misjudged."""
    sections = page_sections(chunks)
    if not (MIN_PAGES <= len(sections) <= MAX_PAGES):
        return None
    if _median([len(_content_words(s.text)) for s in sections]) < MIN_MEDIAN_WORDS:
        return None
    if mean_consecutive_overlap(sections) > COMPOSITE_OVERLAP_MAX:
        return None
    # The decisive signal: most pages must start like their own document. This is
    # what separates a true scanned stack from a dictionary / web capture / article
    # that merely has low page-to-page word overlap.
    starts = sum(1 for s in sections if _looks_like_doc_start(s.text))
    if starts / len(sections) < MIN_DOC_START_FRACTION:
        return None
    return sections

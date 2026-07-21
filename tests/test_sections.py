"""Tests for composite-document detection (a scanned stack of unrelated docs)."""

from __future__ import annotations

from find_and_seek.ingest.chunk import Chunk
from find_and_seek.ingest.sections import (
    composite_summary,
    detect_composite,
    mean_consecutive_overlap,
    page_sections,
    section_anchor,
)


def _mk(page_texts):
    """One chunk per page, location_ref 'page N image 1'."""
    return [
        Chunk(text=t, source_type="ocr", location_ref=f"page {i+1} image 1",
              chunk_index=i, token_estimate=len(t) // 4)
        for i, t in enumerate(page_texts)
    ]


def test_detects_scanned_bundle_of_unrelated_docs():
    # Each page is a different document with distinct vocabulary (realistically
    # sized — real scanned pages have plenty of words).
    chunks = _mk([
        "Statement claim benefit payment Northwind Health imaging services reimbursement "
        "member provider rebate schedule item assessed benefits office electronic location reference",
        "Invoice Kestrel computer store laptop monitor keyboard purchase receipt total amount "
        "salesperson customer delivery freight warranty subtotal payment EFTPOS terminal approved",
        "Pharmacy prescription dispensed medication patient chemist tablets dosage refill repeat "
        "pharmacist directions quantity strength generic substitution dispensing label expiry",
        "Penalty reminder notice infringement vehicle offence registration demerit council "
        "operator obligation enforcement camera location speed fine demerit overdue payable reference",
    ])
    sections = detect_composite(chunks)
    assert sections is not None
    assert [s.page for s in sections] == [1, 2, 3, 4]


def test_coherent_multipage_document_is_not_split():
    # A lease: pages reuse the same vocabulary, so overlap is high.
    base = "tenant landlord premises rent lease agreement term notice property bond"
    chunks = _mk([
        base + " clause one commencement of the lease term and rent payable",
        base + " clause two rent review and the tenant obligations for the premises",
        base + " clause three termination notice by the landlord or the tenant",
        base + " clause four bond and condition of the premises at lease end",
    ])
    assert detect_composite(chunks) is None
    assert mean_consecutive_overlap(page_sections(chunks)) > 0.2


def test_too_few_pages_not_composite():
    assert detect_composite(_mk(["alpha beta gamma delta", "epsilon zeta eta theta"])) is None


def test_sparse_or_non_latin_pages_not_judged():
    # Near-empty Latin word sets (e.g. CJK or image-only pages) must not be
    # spuriously flagged just because overlap computes to ~0.
    chunks = _mk(["天线模块 868", "标签 测试 LED", "射频 识别 系统"])
    assert detect_composite(chunks) is None


def test_section_anchor_picks_first_real_line_skips_furniture():
    text = "BINDING MARGIN - DO NOT WRITE IN THIS AREA\nwww.example.com\nTax Invoice #10020030 Kestrel Computers\nmore body text"
    assert section_anchor(text) == "Tax Invoice #10020030 Kestrel Computers"


def test_composite_summary_builds_bundle_anchor_and_sections():
    chunks = _mk([
        "Statement claim benefit payment Northwind Health imaging services reimbursement "
        "member provider rebate schedule item assessed benefits office electronic location reference",
        "Tax Invoice number 10020030 Kestrel Computers computer store laptop monitor keyboard receipt total "
        "salesperson customer delivery freight warranty subtotal payment EFTPOS terminal approved",
        "Pharmacy prescription dispensed medication patient chemist tablets dosage refill repeat "
        "pharmacist directions quantity strength generic substitution dispensing label expiry",
        "Penalty reminder notice infringement vehicle offence registration demerit council "
        "operator obligation enforcement camera location speed fine demerit overdue payable reference",
    ])
    out = composite_summary(chunks)
    assert out is not None
    assert out["one_line_anchor"].startswith("Scanned bundle — 4 documents")
    sa = out["section_anchors"]
    assert set(sa.keys()) == {1, 2, 3, 4}
    assert "Invoice" in sa[2] and "Kestrel" in sa[2]      # page 2 = the invoice
    assert "Penalty" in sa[4] or "infringement" in sa[4]      # page 4 = the notice
    assert "separate documents" in out["summary_text"]


def test_composite_summary_none_for_coherent_doc():
    base = "tenant landlord premises rent lease agreement term notice property bond clause"
    assert composite_summary(_mk([base + " one", base + " two", base + " three"])) is None


def test_page_sections_groups_and_orders():
    chunks = [
        Chunk(text="b", source_type="ocr", location_ref="page 2 image 1", chunk_index=1, token_estimate=1),
        Chunk(text="a1", source_type="ocr", location_ref="page 1 image 1", chunk_index=0, token_estimate=1),
        Chunk(text="a2", source_type="ocr", location_ref="page 1 image 2", chunk_index=2, token_estimate=1),
    ]
    secs = page_sections(chunks)
    assert [s.page for s in secs] == [1, 2]
    assert "a1" in secs[0].text and "a2" in secs[0].text

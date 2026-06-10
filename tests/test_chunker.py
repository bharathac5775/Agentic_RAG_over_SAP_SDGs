"""Phase 1 chunker correctness tests.

These pin down the expected behavior on real SDG content. Run with:
    python -m pytest tests/test_chunker.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.chunker import chunk_pdf

PDFS = {
    "rise_s4hana_private": "rise with sap s4hana cloud priva.pdf",
    "sap_cloud_erp_private": "SAP Cloud ERP Private, RISE.pdf",
    "sap_erp_pce": "SAP ERP, private cloud edition.pdf",
}


@pytest.fixture(scope="module")
def chunks_for_cloud_erp() -> list:
    """The largest, most-structured PDF — used for most assertions."""
    return chunk_pdf(
        Path("Data") / PDFS["sap_cloud_erp_private"],
        doc_id="sap_cloud_erp_private",
        doc_title="SAP Cloud ERP Private, RISE",
    )


# ---- Bug A: title must not eat the body --------------------------------------


def test_section_1_3_body_contains_defined_term(chunks_for_cloud_erp):
    """§1.3 body must include the full definition prose, not just the trailing
    fragment. The defined term ('"API Call"' or '"Active User"') and the
    explanatory clause must both appear in the chunk text.
    """
    sect_chunks = [c for c in chunks_for_cloud_erp if c.section_number == "1.3"]
    assert sect_chunks, "no chunks for section 1.3"
    full_text = " ".join(c.text for c in sect_chunks)
    # Real §1.3 in this PDF is "API Call". We assert structure, not exact word,
    # to be tolerant of doc revisions: the body must contain a quoted defined
    # term AND a copula verb ("is" or "means" or "are") indicating definition prose.
    assert '"' in full_text or '“' in full_text, "no quoted defined term in §1.3 body"
    assert any(w in full_text for w in [" is ", " means ", " are "]), (
        f"§1.3 body looks truncated: {full_text!r}"
    )
    # The body must be more than ~40 chars — runt fragments mean the title ate it.
    assert len(full_text) > 100, f"§1.3 body is suspiciously short: {full_text!r}"


# ---- Bug B / F: no false-positive top-level sections -------------------------


def test_top_level_section_numbers_are_bounded(chunks_for_cloud_erp):
    """Real top-level section numbers in this SDG go up to ~30. Anything
    above that (e.g., 99, 346, 347) is body prose getting misclassified.
    """
    top_levels = set()
    for c in chunks_for_cloud_erp:
        sn = c.section_number
        if sn and "." not in sn and sn.isdigit():
            top_levels.add(int(sn))
    # Allow a generous ceiling — but '99' and '346' should NOT appear.
    too_big = {n for n in top_levels if n > 50}
    assert not too_big, f"false-positive top-level section numbers: {sorted(too_big)}"


def test_chunk_ids_are_unique(chunks_for_cloud_erp):
    ids = [c.chunk_id for c in chunks_for_cloud_erp]
    dupes = {x for x in ids if ids.count(x) > 1}
    assert not dupes, f"duplicate chunk_ids: {sorted(dupes)[:5]}"


# ---- Bug C / G: page tracking must reflect real positions --------------------


def test_pages_progress_through_document(chunks_for_cloud_erp):
    """Chunks should walk through the doc — late chunks should be on late
    pages. If everything reports page 1, the page-marker logic is broken.
    """
    pages = [c.page_start for c in chunks_for_cloud_erp]
    assert max(pages) >= 100, f"max page_start={max(pages)}; markers not propagating"
    # The progression should be roughly monotonic: median of late-half >> median of early-half
    half = len(pages) // 2
    early_median = sorted(pages[:half])[half // 2]
    late_median = sorted(pages[half:])[half // 2]
    assert late_median > early_median, (
        f"pages not progressing: early_median={early_median}, late_median={late_median}"
    )


def test_no_chunk_has_zero_or_negative_pages(chunks_for_cloud_erp):
    for c in chunks_for_cloud_erp:
        assert c.page_start >= 1, f"{c.chunk_id} page_start={c.page_start}"
        assert c.page_end >= c.page_start, f"{c.chunk_id} pages reversed"


# ---- General sanity ----------------------------------------------------------


def test_chunk_sizes_are_in_range(chunks_for_cloud_erp):
    """No chunks should be useless runts (< 15 chars — those are zero-info
    leftovers like a section title alone). Some sub-bullet sections like
    '"SAP HANA, enterprise edition;"' are legitimately short (~25 chars)
    and should be retained. Very few should exceed 2.5x the hard max.
    """
    sizes = [len(c.text) for c in chunks_for_cloud_erp]
    runts = [s for s in sizes if s < 15]
    giants = [s for s in sizes if s > 2500]
    assert len(runts) == 0, f"useless runt chunks remain: {len(runts)}/{len(sizes)}"
    assert len(giants) == 0, f"oversized chunks: {giants[:3]}"
    # Sanity: median should still be in the target zone (~150-500 chars)
    median = sorted(sizes)[len(sizes) // 2]
    assert 80 <= median <= 600, f"unexpected median chunk size: {median}"


def test_unnumbered_pdf_uses_fallback_strategy():
    """The 'SAP ERP, private cloud edition.pdf' lacks numbered structure.
    The chunker should fall back to all-caps headings or fixed windows,
    NOT crash, NOT produce zero chunks.
    """
    chunks = chunk_pdf(
        Path("Data") / PDFS["sap_erp_pce"],
        doc_id="sap_erp_pce",
        doc_title="SAP ERP, private cloud edition",
    )
    assert len(chunks) >= 10, f"unnumbered PDF produced only {len(chunks)} chunks"
    # All chunks must have SOME section_number (synthetic is fine)
    assert all(c.section_number for c in chunks), "chunks missing section_number"

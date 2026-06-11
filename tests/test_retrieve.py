"""Phase 2 retrieval correctness tests. Requires ./index/ to be built.

Run with:  python -m pytest tests/test_retrieve.py -v
"""

from __future__ import annotations

import pytest

from app.retrieve import get_retriever


@pytest.fixture(scope="module")
def retr():
    """One Retriever shared across all tests in this module."""
    return get_retriever()


# ---- Smoke ------------------------------------------------------------------


def test_retriever_loads(retr):
    """Cold start should produce a non-empty corpus."""
    assert len(retr.chunks) > 100, f"too few chunks: {len(retr.chunks)}"
    assert len(retr.chunks_by_id) == len(retr.chunks)


def test_returns_requested_k(retr):
    chunks, _dbg = retr.search("Active User", k=5)
    assert len(chunks) == 5
    chunks, _dbg = retr.search("Active User", k=3)
    assert len(chunks) == 3


# ---- Hybrid beats pure vector on the canonical defined-term query -----------


def test_active_user_query_hits_section_1_2_or_1_3(retr):
    """The defining test from Phase 1: 'What is an Active User?' should
    retrieve the canonical §1.X "Active User" definition near the top.
    Phase 2 dev showed pure BM25 buries this under high-TF distractors;
    boosting VECTOR (not BM25) for intent=definition fixes it. The
    Phase-7 heading-anchor boost may also surface "Usage Metric: Active
    User" sections (which are equally canonical defining sections).
    """
    chunks, dbg = retr.search(
        "What is an Active User?",
        docs=None,
        intent="definition",
        k=5,
    )
    # Within the top 5, at least one chunk must be a canonical "Active
    # User" definition. We accept the bare-title glossary chunk
    # (section_title == "Active User") OR any "Usage Metric: Active
    # User" section — both define the term identically.
    found = [
        c for c in chunks
        if "active user" in (c.section_title or "").lower()
    ]
    assert found, (
        "no chunk whose title contains 'Active User' in top-5. Top results: "
        + ", ".join(f"{c.chunk_id}({c.section_title!r})" for c in chunks)
    )
    # Confirm definition intent boosts vector, not BM25.
    assert dbg.vector_weight > dbg.bm25_weight, (
        "intent=definition should boost VECTOR weight"
    )


def test_paraphrased_query_finds_api_call_definition(retr):
    """A paraphrased question with no exact keyword overlap with the SDG
    should still retrieve the canonical definition. This exercises the
    vector retriever specifically.
    """
    chunks, _dbg = retr.search(
        "how does my application talk to the cloud service?",
        intent="general",
        k=5,
    )
    top_text = " | ".join(c.text.lower() for c in chunks[:3])
    # We expect EITHER 'API Call' or related connectivity terms surfaced.
    assert any(
        kw in top_text
        for kw in ("api call", "application programming interface", "connection")
    ), f"paraphrased query missed semantic match: {top_text[:300]}"


# ---- Doc filter ---------------------------------------------------------------


def test_filter_to_sap_erp_pce_only(retr):
    chunks, _dbg = retr.search(
        "FUE entitlement",
        docs=["sap_erp_pce"],
        intent="specific_clause",
        k=5,
    )
    assert chunks, "filter produced empty result"
    bad = [c for c in chunks if c.doc_id != "sap_erp_pce"]
    assert not bad, f"filter leaked: {[c.doc_id for c in bad]}"


def test_no_filter_returns_chunks_from_multiple_docs(retr):
    """A broad cross-product term ('subscription term') should naturally
    pull from at least two doc_ids — proves the unfiltered path doesn't
    accidentally collapse to one doc.

    Note: very generic terms like "Cloud Service" actually surface only
    one doc because the embedding cluster for that term sits inside the
    largest doc (sap_cloud_erp_private has ~3× the chunks of either RISE
    doc). This is a known retrieval bias documented in the README.
    Picking a less generic, more cross-cutting term avoids the bias for
    this test's purpose.
    """
    chunks, _dbg = retr.search("subscription term", docs=None, k=10)
    doc_ids_seen = {c.doc_id for c in chunks}
    assert len(doc_ids_seen) >= 2, f"only one doc surfaced: {doc_ids_seen}"


# ---- Intent affects result shape ---------------------------------------------


def test_comparison_intent_returns_more_chunks(retr):
    """intent='comparison' should default to COMPARISON_TOP_K (8) when k
    is not specified, producing more chunks than 'general' (default 5).
    """
    cmp_chunks, _ = retr.search("How does PCE differ from RISE?", intent="comparison")
    gen_chunks, _ = retr.search("Tell me about cloud services", intent="general")
    assert len(cmp_chunks) > len(gen_chunks), (
        f"comparison ({len(cmp_chunks)}) should exceed general ({len(gen_chunks)})"
    )


# ---- RRF correctness ---------------------------------------------------------


def test_returned_chunks_have_score_and_ranks(retr):
    """Every chunk returned by search() must carry the diagnostic fields:
    score, and at least one of rank_bm25/rank_vector (chunks ranked by both
    will have both).
    """
    chunks, _dbg = retr.search("Active User definition", k=5)
    for c in chunks:
        assert c.score is not None and c.score > 0
        assert (c.rank_bm25 is not None) or (c.rank_vector is not None)


def test_scores_are_descending(retr):
    chunks, _dbg = retr.search("data residency", k=5)
    for i in range(len(chunks) - 1):
        assert (chunks[i].score or 0) >= (chunks[i + 1].score or 0)


# ---- Empty/edge cases --------------------------------------------------------


def test_empty_query_does_not_crash(retr):
    chunks, _dbg = retr.search("", k=3)
    # We allow either: empty list OR non-empty (vector embedding may still
    # return SOMETHING for an empty string). Just don't crash.
    assert isinstance(chunks, list)


def test_nonsense_query_returns_low_quality_but_valid(retr):
    """A truly off-topic query should still return SOMETHING (valid Chunks)
    rather than crashing. Quality will be low — the verifier will catch
    it downstream.
    """
    chunks, _dbg = retr.search("xqzy nonsense gibberish", k=3)
    assert isinstance(chunks, list)
    for c in chunks:
        assert c.text  # they're real chunks

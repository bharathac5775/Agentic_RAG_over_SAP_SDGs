"""Phase 6 pipeline tests.

  - TestPipelineFlow: mocked tests covering the full route → retrieve →
    generate → verify chain. Fast.
  - TestPipelineLive: 2 real end-to-end queries via the actual LLMs.

Run:
    python -m pytest tests/test_pipeline.py -v
    python -m pytest tests/test_pipeline.py -m "not live" -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from app import pipeline
from app.retrieve import RetrievalDebug
from app.schemas import (
    Chunk,
    Citation,
    GeneratedAnswer,
    GuardrailDecision,
    RouteDecision,
    VerifierVerdict,
)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeRetriever:
    """Drop-in retriever that returns canned chunks. Bypasses Ollama / Chroma.
    Kept simple — only the .search() method is used by the pipeline.
    """

    def __init__(self, chunks: list[Chunk] | None = None):
        self._chunks = chunks or _sample_chunks()

    def search(self, query: str, *, docs=None, k=None, intent=None):
        # Return all stored chunks, capped at k if given.
        out = self._chunks if k is None else self._chunks[:k]
        debug = RetrievalDebug(
            bm25_ranks={c.chunk_id: i + 1 for i, c in enumerate(out)},
            vector_ranks={c.chunk_id: i + 1 for i, c in enumerate(out)},
            vector_top1_score=0.85,
            fallback_triggered=False,
            final_top_k=len(out),
            bm25_weight=1.0,
            vector_weight=1.5,
        )
        return out, debug


def _sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="sap_cloud_erp_private#1.2_0",
            doc_id="sap_cloud_erp_private",
            doc_title="SAP Cloud ERP Private, RISE",
            section_number="1.2",
            section_title="Active User",
            page_start=1,
            page_end=1,
            text='"Active User" is any individual who accesses the Cloud Service.',
        ),
    ]


def _patch_route(decision: RouteDecision):
    return patch(
        "app.pipeline.router.route",
        return_value=(decision, {"latency_ms": 10, "fallback_used": False, "overrides_applied": False}),
    )


def _patch_guardrail(decision: GuardrailDecision):
    return patch(
        "app.pipeline.guardrails.classify_input",
        return_value=(decision, {"stage_used": "stage1", "latency_ms": 0}),
    )


def _patch_generate(answer: GeneratedAnswer):
    return patch(
        "app.pipeline.agent.generate",
        return_value=(answer, {"latency_ms": 100, "fallback_used": False, "retries_used": 0, "dropped_citations": 0}),
    )


def _patch_verify(verdict: VerifierVerdict):
    return patch(
        "app.pipeline.agent.verify",
        return_value=(verdict, {"latency_ms": 50, "skipped_refusal": False, "fallback_used": False, "retries_used": 0}),
    )


def _patch_check_output(answer: GeneratedAnswer, debug: dict[str, Any] | None = None):
    return patch(
        "app.pipeline.guardrails.check_output",
        return_value=(answer, debug or {"downgraded_to_refusal": False, "pii_redacted": False}),
    )


# ---------------------------------------------------------------------------
# Pipeline flow tests
# ---------------------------------------------------------------------------


class TestPipelineFlow:

    def test_happy_path_grounded_answer(self):
        gd = GuardrailDecision(in_scope=True, category="in_scope", reason="ok", refusal_message=None)
        rd = RouteDecision(products=["all"], intent="definition", rewritten_query="x", reasoning="y")
        ans = GeneratedAnswer(
            answer="An Active User is any individual who accesses the Cloud Service.",
            citations=[Citation(doc="SAP Cloud ERP Private, RISE", section="1.2", page=1, quote="x")],
        )
        verdict = VerifierVerdict(grounded=True, unsupported_claims=[], missing_citations=[])

        with _patch_guardrail(gd), _patch_route(rd), _patch_generate(ans), \
             _patch_check_output(ans), _patch_verify(verdict):
            resp = pipeline.answer("What is an Active User?",
                                   retriever=_FakeRetriever())

        assert resp.verified is True
        assert resp.refused is False
        assert resp.warning is None
        assert "Active User" in resp.answer
        assert len(resp.citations) == 1

    def test_guardrail_refusal_short_circuits(self):
        """If the input guardrail rejects, NO downstream calls happen."""
        gd = GuardrailDecision(
            in_scope=False, category="code_request",
            reason="matched code", refusal_message="I won't write code.",
        )
        with _patch_guardrail(gd), \
             patch("app.pipeline.router.route") as m_route, \
             patch("app.pipeline.agent.generate") as m_gen, \
             patch("app.pipeline.agent.verify") as m_ver:
            resp = pipeline.answer("write me a Python script",
                                   retriever=_FakeRetriever())
            m_route.assert_not_called()
            m_gen.assert_not_called()
            m_ver.assert_not_called()

        assert resp.refused is True
        assert resp.refusal_reason == "matched code"
        assert "won't write code" in resp.answer

    def test_ungrounded_triggers_one_retry(self):
        """If the verifier says ungrounded, retrieval+generation+verify run again."""
        gd = GuardrailDecision(in_scope=True, category="in_scope", reason="ok", refusal_message=None)
        rd = RouteDecision(products=["all"], intent="general", rewritten_query="x", reasoning="y")
        ans = GeneratedAnswer(
            answer="Something might be true.",
            citations=[Citation(doc="X", section="1", page=1, quote="x")],
        )
        ungrounded = VerifierVerdict(grounded=False,
                                      unsupported_claims=["something"],
                                      missing_citations=[])
        grounded = VerifierVerdict(grounded=True, unsupported_claims=[], missing_citations=[])

        # First verify call returns ungrounded; second returns grounded.
        verify_returns = [
            (ungrounded, {"latency_ms": 50, "skipped_refusal": False, "fallback_used": False, "retries_used": 0}),
            (grounded,   {"latency_ms": 50, "skipped_refusal": False, "fallback_used": False, "retries_used": 0}),
        ]
        with _patch_guardrail(gd), _patch_route(rd), _patch_generate(ans), \
             _patch_check_output(ans), \
             patch("app.pipeline.agent.verify", side_effect=verify_returns) as m_ver:
            resp = pipeline.answer("anything", retriever=_FakeRetriever(), debug=True)

        assert m_ver.call_count == 2, "verify should run twice (initial + retry)"
        assert resp.verified is True
        assert resp.warning is None
        assert resp.trace is not None
        assert "retry" in resp.trace

    def test_retry_exhausted_returns_warning(self):
        """If verify is ungrounded twice, return the answer with verified=false + warning."""
        gd = GuardrailDecision(in_scope=True, category="in_scope", reason="ok", refusal_message=None)
        rd = RouteDecision(products=["all"], intent="general", rewritten_query="x", reasoning="y")
        ans = GeneratedAnswer(
            answer="Possibly wrong claim.",
            citations=[Citation(doc="X", section="1", page=1, quote="x")],
        )
        ungrounded = VerifierVerdict(grounded=False,
                                      unsupported_claims=["claim X"],
                                      missing_citations=[])

        with _patch_guardrail(gd), _patch_route(rd), _patch_generate(ans), \
             _patch_check_output(ans), _patch_verify(ungrounded):
            resp = pipeline.answer("anything", retriever=_FakeRetriever())

        assert resp.verified is False
        assert resp.refused is False
        assert resp.warning is not None
        assert "unsupported" in resp.warning.lower()

    def test_output_guardrail_downgrade_skips_verifier(self):
        """If check_output downgrades to refusal, the verifier sees a refusal and auto-passes."""
        gd = GuardrailDecision(in_scope=True, category="in_scope", reason="ok", refusal_message=None)
        rd = RouteDecision(products=["all"], intent="general", rewritten_query="x", reasoning="y")
        # Generator returns an empty-citation claim
        bad_ans = GeneratedAnswer(answer="some claim", citations=[])
        # Output guardrail downgrades to refusal
        refusal = GeneratedAnswer(answer="The provided SDGs do not specify this.", citations=[])
        # Verifier sees the refusal — its real implementation auto-passes refusals,
        # so we mock it returning grounded=True (which is what the real code does).
        grounded = VerifierVerdict(grounded=True, unsupported_claims=[], missing_citations=[])

        with _patch_guardrail(gd), _patch_route(rd), _patch_generate(bad_ans), \
             _patch_check_output(refusal, {"downgraded_to_refusal": True, "pii_redacted": False}), \
             _patch_verify(grounded):
            resp = pipeline.answer("anything", retriever=_FakeRetriever())

        assert resp.answer == "The provided SDGs do not specify this."
        assert resp.verified is True
        assert resp.refused is False  # technically not a guardrail-refusal

    def test_debug_trace_populated_on_demand(self):
        gd = GuardrailDecision(in_scope=True, category="in_scope", reason="ok", refusal_message=None)
        rd = RouteDecision(products=["all"], intent="general", rewritten_query="x", reasoning="y")
        ans = GeneratedAnswer(answer="x", citations=[Citation(doc="X", section="1", page=1, quote="x")])
        verdict = VerifierVerdict(grounded=True, unsupported_claims=[], missing_citations=[])

        with _patch_guardrail(gd), _patch_route(rd), _patch_generate(ans), \
             _patch_check_output(ans), _patch_verify(verdict):
            no_trace = pipeline.answer("x", retriever=_FakeRetriever(), debug=False)
            with_trace = pipeline.answer("x", retriever=_FakeRetriever(), debug=True)

        assert no_trace.trace is None
        assert with_trace.trace is not None
        # Must include the major sections
        assert "guardrail" in with_trace.trace
        assert "route" in with_trace.trace
        assert "retrieval" in with_trace.trace
        assert "generator" in with_trace.trace
        assert "verifier" in with_trace.trace
        assert "latency_ms" in with_trace.trace

    def test_doc_filter_resolves_correctly(self):
        """The router's products list must be translated to a doc_id list
        the retriever understands.
        """
        from app import config
        # Single family
        assert pipeline._resolve_doc_filter(["rise_family"]) == config.PRODUCT_DOCS["rise_family"]
        # All
        assert set(pipeline._resolve_doc_filter(["all"])) == set(config.PRODUCT_DOCS["all"])
        # Empty / unknown → fallback to all
        assert set(pipeline._resolve_doc_filter([])) == set(config.PRODUCT_DOCS["all"])
        assert set(pipeline._resolve_doc_filter(["nonsense"])) == set(config.PRODUCT_DOCS["all"])


# ---------------------------------------------------------------------------
# Live tests — full pipeline against real models
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestPipelineLive:

    def test_active_user_question_end_to_end(self):
        resp = pipeline.answer("What is an Active User?", debug=True)
        assert resp.refused is False
        # Either grounded or refused-with-warning. Both acceptable; what we
        # require is a structured response with citations OR a refusal.
        if not resp.answer.startswith("The provided SDGs do not"):
            assert "active user" in resp.answer.lower()
        # Trace must have all the expected keys
        assert resp.trace is not None
        assert all(k in resp.trace for k in
                   ("guardrail", "route", "retrieval", "generator", "verifier", "latency_ms"))

    def test_code_request_refused_end_to_end(self):
        resp = pipeline.answer("Write me a Python script to call SAP", debug=True)
        assert resp.refused is True
        assert resp.refusal_reason
        # Should NOT have run the LLM-heavy stages
        assert resp.trace is not None
        assert "route" not in resp.trace  # short-circuited at guardrail

"""Phase 4 generator + verifier tests.

  - TestGenerateLogic / TestVerifyLogic: mocked Ollama, fast (<0.1s each).
    Exercise fallback paths, citation post-validation, refusal handling.
  - TestAgentLive: real Ollama call. End-to-end on the actual index.
    Marked @pytest.mark.live.

Run:
    python -m pytest tests/test_agent.py -v                  # all
    python -m pytest tests/test_agent.py -m "not live" -v   # fast only
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app import agent
from app.schemas import Chunk, Citation, GeneratedAnswer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_chat_response(json_str: str):
    """Return the bare content string — that's what llm.chat() returns
    after the provider abstraction (no longer wrapped in {message: {...}})."""
    return json_str


@pytest.fixture
def sample_chunks() -> list[Chunk]:
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
        Chunk(
            chunk_id="sap_cloud_erp_private#1.3_0",
            doc_id="sap_cloud_erp_private",
            doc_title="SAP Cloud ERP Private, RISE",
            section_number="1.3",
            section_title="API Call",
            page_start=1,
            page_end=1,
            text='"API Call" is the communication of an action to or from the Cloud Service.',
        ),
    ]


# ---------------------------------------------------------------------------
# generate() logic tests — mocked
# ---------------------------------------------------------------------------


class TestGenerateLogic:

    def test_valid_json_passes_through(self, sample_chunks):
        canned = json.dumps({
            "answer": "An Active User is any individual who accesses the Cloud Service.",
            "citations": [{
                "doc": "SAP Cloud ERP Private, RISE",
                "section": "1.2",
                "page": 1,
                "quote": '"Active User" is any individual who accesses the Cloud Service.',
            }],
        })
        with patch("app.agent.llm.chat", return_value=_mock_chat_response(canned)):
            answer, debug = agent.generate("What is an Active User?", sample_chunks)
        assert "Active User" in answer.answer
        assert len(answer.citations) == 1
        assert answer.citations[0].section == "1.2"
        assert not debug["fallback_used"]
        assert debug["dropped_citations"] == 0

    def test_invented_citation_is_dropped(self, sample_chunks):
        # LLM returns a citation pointing to §99.99 which isn't in our chunks.
        canned = json.dumps({
            "answer": "Some claim.",
            "citations": [
                # This one IS in the chunks — should be kept
                {"doc": "SAP Cloud ERP Private, RISE", "section": "1.2", "page": 1, "quote": "x"},
                # This one is INVENTED — should be dropped
                {"doc": "SAP Cloud ERP Private, RISE", "section": "99.99", "page": 42, "quote": "y"},
            ],
        })
        with patch("app.agent.llm.chat", return_value=_mock_chat_response(canned)):
            answer, debug = agent.generate("anything", sample_chunks)
        assert len(answer.citations) == 1, "invented citation should have been dropped"
        assert answer.citations[0].section == "1.2"
        assert debug["dropped_citations"] == 1

    def test_empty_chunks_returns_refusal_without_llm_call(self):
        with patch("app.agent.llm.chat") as mock_chat:
            answer, debug = agent.generate("anything", [])
            mock_chat.assert_not_called()
        assert answer.answer == "The provided SDGs do not specify this."
        assert answer.citations == []
        assert debug["fallback_used"]

    def test_invalid_json_falls_back_to_refusal(self, sample_chunks):
        with patch("app.agent.llm.chat",
                   return_value=_mock_chat_response("totally not json")):
            answer, debug = agent.generate("anything", sample_chunks)
        assert answer.answer == "The provided SDGs do not specify this."
        assert debug["fallback_used"]
        assert debug["retries_used"] >= 1

    def test_validation_error_falls_back_to_refusal(self, sample_chunks):
        # JSON is valid but missing required field "answer"
        canned = json.dumps({"citations": []})
        with patch("app.agent.llm.chat", return_value=_mock_chat_response(canned)):
            answer, debug = agent.generate("anything", sample_chunks)
        assert answer.answer == "The provided SDGs do not specify this."
        assert debug["fallback_used"]

    def test_ollama_exception_triggers_retry_then_refusal(self, sample_chunks):
        with patch("app.agent.llm.chat",
                   side_effect=RuntimeError("connection refused")):
            answer, debug = agent.generate("test", sample_chunks)
        assert answer.answer == "The provided SDGs do not specify this."
        assert debug["fallback_used"]
        assert debug["retries_used"] >= 1

    def test_refusal_with_chunks_accepted(self, sample_chunks):
        # LLM correctly chose to refuse despite having chunks (chunks didn't match).
        canned = json.dumps({
            "answer": "The provided SDGs do not specify this.",
            "citations": [],
        })
        with patch("app.agent.llm.chat", return_value=_mock_chat_response(canned)):
            answer, debug = agent.generate("unrelated question", sample_chunks)
        assert answer.answer == "The provided SDGs do not specify this."
        assert answer.citations == []
        assert not debug["fallback_used"]


# ---------------------------------------------------------------------------
# verify() logic tests — mocked
# ---------------------------------------------------------------------------


class TestVerifyLogic:

    def test_grounded_answer_passes(self, sample_chunks):
        canned = json.dumps({
            "grounded": True,
            "unsupported_claims": [],
            "missing_citations": [],
        })
        good_answer = GeneratedAnswer(
            answer="Active User is any individual who accesses the Cloud Service.",
            citations=[
                Citation(doc="SAP Cloud ERP Private, RISE", section="1.2", page=1, quote="x"),
            ],
        )
        with patch("app.agent.llm.chat", return_value=_mock_chat_response(canned)):
            verdict, debug = agent.verify(good_answer, sample_chunks)
        assert verdict.grounded
        assert not debug["skipped_refusal"]

    def test_ungrounded_answer_flags_claims(self, sample_chunks):
        canned = json.dumps({
            "grounded": False,
            "unsupported_claims": ["Active User is paid $50/month"],
            "missing_citations": [],
        })
        bad_answer = GeneratedAnswer(
            answer="Active User is paid $50/month and gets a car.",
            citations=[],
        )
        with patch("app.agent.llm.chat", return_value=_mock_chat_response(canned)):
            verdict, _ = agent.verify(bad_answer, sample_chunks)
        assert not verdict.grounded
        assert "Active User is paid $50/month" in verdict.unsupported_claims

    def test_refusal_skips_llm_and_returns_grounded(self, sample_chunks):
        refusal = GeneratedAnswer(
            answer="The provided SDGs do not specify this.",
            citations=[],
        )
        with patch("app.agent.llm.chat") as mock_chat:
            verdict, debug = agent.verify(refusal, sample_chunks)
            mock_chat.assert_not_called()
        assert verdict.grounded
        assert debug["skipped_refusal"]

    def test_invalid_json_defaults_to_ungrounded(self, sample_chunks):
        ans = GeneratedAnswer(answer="some claim", citations=[])
        with patch("app.agent.llm.chat",
                   return_value=_mock_chat_response("not json")):
            verdict, debug = agent.verify(ans, sample_chunks)
        # Conservative: when verifier breaks, mark as ungrounded.
        assert not verdict.grounded
        assert debug["fallback_used"]

    def test_ollama_exception_triggers_retry_then_ungrounded(self, sample_chunks):
        ans = GeneratedAnswer(answer="some claim", citations=[])
        with patch("app.agent.llm.chat",
                   side_effect=RuntimeError("connection refused")):
            verdict, debug = agent.verify(ans, sample_chunks)
        assert not verdict.grounded
        assert debug["fallback_used"]
        assert debug["retries_used"] >= 1


# ---------------------------------------------------------------------------
# Citation post-validation
# ---------------------------------------------------------------------------


class TestCitationValidation:

    def test_doc_page_match_without_section_passes(self, sample_chunks):
        # Some LLMs forget the section number. Doc + page should suffice.
        canned = json.dumps({
            "answer": "x",
            "citations": [
                {"doc": "SAP Cloud ERP Private, RISE", "section": None, "page": 1, "quote": "x"},
            ],
        })
        with patch("app.agent.llm.chat", return_value=_mock_chat_response(canned)):
            answer, debug = agent.generate("anything", sample_chunks)
        assert len(answer.citations) == 1
        assert debug["dropped_citations"] == 0

    def test_section_with_paragraph_marker_normalized(self, sample_chunks):
        # LLM may include the § symbol; we should still match.
        canned = json.dumps({
            "answer": "x",
            "citations": [
                {"doc": "SAP Cloud ERP Private, RISE", "section": "§1.2", "page": 1, "quote": "x"},
            ],
        })
        with patch("app.agent.llm.chat", return_value=_mock_chat_response(canned)):
            answer, debug = agent.generate("anything", sample_chunks)
        assert len(answer.citations) == 1
        assert debug["dropped_citations"] == 0


# ---------------------------------------------------------------------------
# Live tests — actual Ollama
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestAgentLive:
    """End-to-end against real models using the live retriever index.

    These are smoke tests on observable behavior, NOT contract tests on model
    output (small models drift across versions).
    """

    def test_active_user_question_produces_grounded_answer(self):
        from app.retrieve import get_retriever
        retr = get_retriever()
        chunks, _ = retr.search("What is an Active User?", intent="definition", k=5)
        assert chunks, "retrieval returned nothing"

        answer, gen_debug = agent.generate("What is an Active User?", chunks)
        # Answer must mention the defined term (or refuse — both acceptable).
        is_refusal = answer.answer.startswith("The provided SDGs do not")
        assert is_refusal or "active user" in answer.answer.lower()
        # Loose latency cap: live LLM tests share the host with other work
        # and a cold 8B model can take 30–60 s. We only assert generation
        # completed within a generous bound, not that it was fast.
        assert gen_debug["latency_ms"] < 90000, f"generator took {gen_debug['latency_ms']} ms"

        if not is_refusal:
            verdict, ver_debug = agent.verify(answer, chunks)
            # We don't assert grounded=True (verifier might be conservative on
            # small model). We only assert it returns a valid verdict.
            assert isinstance(verdict.grounded, bool)
            assert ver_debug["latency_ms"] < 60000

    def test_unanswerable_question_produces_refusal(self):
        from app.retrieve import get_retriever
        retr = get_retriever()
        # Pricing is explicitly NOT in the SDGs — should refuse.
        chunks, _ = retr.search("What is the price of S/4HANA private?", k=5)
        answer, _ = agent.generate("What is the price of S/4HANA private in USD?", chunks)
        # Either refuses outright or doesn't claim a price. We accept both.
        if not answer.answer.startswith("The provided SDGs do not"):
            # Check no $ amount or specific price was hallucinated
            txt = answer.answer.lower()
            assert "$" not in txt or "price" not in txt or "do not specify" in txt

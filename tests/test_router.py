"""Phase 3 router tests.

Two test classes:
  - TestRouterLogic: mocked Ollama. Fast (~10 ms each). Tests fallback,
    overrides, validation. Run on every change.
  - TestRouterLive: real Ollama call to llama3.2:3b. Slow (~1-2s each).
    Tests that the actual model produces sensible decisions on realistic
    questions. Marked with `@pytest.mark.live` — skip with `-m "not live"`.

Run:
    python -m pytest tests/test_router.py -v
    python -m pytest tests/test_router.py -m "not live" -v   # fast only
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app import router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_ollama_response(json_str: str):
    """llm.chat() returns the bare content string (provider abstraction).
    Name kept for historical readability; despite the name it's not
    Ollama-specific — works for any provider."""
    return json_str


# ---------------------------------------------------------------------------
# Mocked logic tests — fast, run always
# ---------------------------------------------------------------------------


class TestRouterLogic:

    def test_valid_json_passes_through(self):
        canned = json.dumps({
            "products": ["rise_family"],
            "intent": "specific_clause",
            "rewritten_query": "data residency in RISE",
            "reasoning": "specific RISE clause",
        })
        with patch("app.router.llm.chat", return_value=_mock_ollama_response(canned)):
            decision, debug = router.route("Where is data stored in RISE?")
        assert list(decision.products) == ["rise_family"]
        assert decision.intent == "specific_clause"
        assert not debug["fallback_used"]
        assert not debug["overrides_applied"]

    def test_invalid_json_falls_back_after_retry(self):
        with patch("app.router.llm.chat",
                   return_value=_mock_ollama_response("totally not json")):
            decision, debug = router.route("anything")
        assert list(decision.products) == ["all"]
        assert decision.intent == "general"
        assert debug["fallback_used"]
        assert debug["retries_used"] >= 1

    def test_validation_error_falls_back(self):
        # LLM returns valid JSON but with an invalid intent value.
        canned = json.dumps({
            "products": ["all"],
            "intent": "wibble",                         # ← not in the enum
            "rewritten_query": "x",
            "reasoning": "y",
        })
        with patch("app.router.llm.chat", return_value=_mock_ollama_response(canned)):
            decision, debug = router.route("anything")
        assert debug["fallback_used"]
        assert decision.intent == "general"

    def test_comparison_intent_forces_products_all(self):
        canned = json.dumps({
            "products": ["rise_family"],                # LLM wrong; comparison needs all
            "intent": "comparison",
            "rewritten_query": "compare",
            "reasoning": "comparing",
        })
        with patch("app.router.llm.chat", return_value=_mock_ollama_response(canned)):
            decision, debug = router.route("compare RISE and PCE")
        assert list(decision.products) == ["all"], (
            "comparison intent must override products to ['all']"
        )
        assert debug["overrides_applied"]

    def test_definition_intent_forces_products_all(self):
        canned = json.dumps({
            "products": ["rise_family"],                # LLM wrong; definitions span the corpus
            "intent": "definition",
            "rewritten_query": "what is X",
            "reasoning": "def",
        })
        with patch("app.router.llm.chat", return_value=_mock_ollama_response(canned)):
            decision, debug = router.route("what is an Active User?")
        assert list(decision.products) == ["all"]
        assert debug["overrides_applied"]

    def test_specific_clause_intent_does_not_override_products(self):
        canned = json.dumps({
            "products": ["sap_erp_pce"],
            "intent": "specific_clause",
            "rewritten_query": "SLA in PCE",
            "reasoning": "PCE-specific SLA",
        })
        with patch("app.router.llm.chat", return_value=_mock_ollama_response(canned)):
            decision, debug = router.route("What's the SLA for SAP ERP PCE?")
        assert list(decision.products) == ["sap_erp_pce"]
        assert not debug["overrides_applied"]

    def test_empty_query_falls_back_without_llm_call(self):
        # Should NOT call ollama.chat at all for empty string
        with patch("app.router.llm.chat") as mock_chat:
            decision, debug = router.route("   ")
            mock_chat.assert_not_called()
        assert decision.intent == "general"
        assert debug["fallback_used"]

    def test_empty_rewritten_query_falls_back_to_original(self):
        canned = json.dumps({
            "products": ["all"],
            "intent": "general",
            "rewritten_query": "",                      # ← LLM left it blank
            "reasoning": "vague",
        })
        with patch("app.router.llm.chat", return_value=_mock_ollama_response(canned)):
            decision, _ = router.route("Tell me stuff")
        assert decision.rewritten_query == "Tell me stuff", (
            "empty rewritten_query should fall back to the original question"
        )

    def test_ollama_exception_triggers_retry_then_fallback(self):
        with patch("app.router.llm.chat",
                   side_effect=RuntimeError("connection refused")):
            decision, debug = router.route("test")
        assert debug["fallback_used"]
        assert debug["retries_used"] >= 1


# ---------------------------------------------------------------------------
# Live tests — actually call llama3.2:3b
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestRouterLive:
    """Realistic questions against the actual Ollama model.

    These are NOT contract tests of model output (small models drift across
    versions). Instead we assert weaker invariants — the schema is valid,
    the override logic activates correctly, latency is sane.
    """

    def test_pce_specific_question_routes_to_pce(self):
        decision, debug = router.route(
            "What's the SLA for SAP ERP, private cloud edition?"
        )
        # Strong signal — PCE-specific. Should be sap_erp_pce OR all (acceptable).
        assert "sap_erp_pce" in decision.products or decision.products == ["all"]
        # Cold-start latency can hit ~15s when llama3.2 first loads into RAM;
        # warm calls are <2s. Test the "warm enough to be useful" envelope.
        assert debug["latency_ms"] < 20000, (
            f"router took {debug['latency_ms']} ms — even cold start should be <20s"
        )

    def test_comparison_question_routes_to_all(self):
        decision, debug = router.route(
            "What are the differences between RISE S/4HANA private and SAP ERP PCE?"
        )
        # Comparison override forces products=["all"] regardless of model output
        assert list(decision.products) == ["all"]

    def test_definition_question_routes_to_all(self):
        decision, _ = router.route("What is an Active User?")
        # Definition override forces products=["all"]
        assert list(decision.products) == ["all"]
        assert decision.intent == "definition"

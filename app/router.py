"""
Agentic step #1 — the (product, intent) router + query rewriter.

Given a natural-language question, decides:
    products:         which product family to search (RISE family, SAP ERP PCE,
                      or all). Drives Chroma's doc_id filter in retrieve.py.
    intent:           definition / specific_clause / comparison / general.
                      Drives RRF weights and top-K in retrieve.py.
    rewritten_query:  optionally cleaned-up wording (acronyms expanded, etc.).
    reasoning:        one short sentence — surfaced via /query?debug=true.

Why this is the FIRST agentic step (the interview pitch):
  - Corpus has 2 products in 3 PDFs (the two RISE docs overlap).
  - A user asks "what's the SLA for RISE?" — the LLM needs to know that
    BOTH RISE docs are candidates, not just the one whose filename matches.
  - Routing on (product, intent) instead of filenames matches the user's
    mental model.
  - intent steers downstream: comparison → top-8 chunks across all docs;
    definition → boost vector weight in RRF.

Architecture decisions:
  - Uses MODEL_SMALL (llama3.2:3b) — classification, not synthesis. 3B is
    fine and ~5× faster than 8B.
  - format="json" + Pydantic validation. If JSON is malformed or fields
    don't match the enum, ValidationError → fallback (one retry).
  - Heuristic overrides AFTER the LLM call:
      intent=comparison  →  products=[all]   (can't compare without both)
      intent=definition  →  products=[all]   (defined terms span docs)
    The router is a JSON classifier; downstream code enforces invariants.
  - Doc summaries (index/doc_summaries.json) are loaded once at startup
    and pinned into the system prompt so the LLM has accurate descriptors
    for the actual corpus, not its training-time knowledge of SAP.

Performance envelope:
  Each route() call = 1 LLM chat to MODEL_SMALL (llama3.2:3b by default).
  Typical latency on M2 + Ollama: 0.5-1.5s. Cold start of the small model:
  +2s once. Cloud providers (OpenAI, Claude, Gemini) are typically faster
  per call but add ~50-200ms of network round-trip. See app/llm.py for
  the provider abstraction.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from app import config, llm
from app.schemas import RouteDecision


# ---------------------------------------------------------------------------
# Doc-summary loading (used to render the router prompt)
# ---------------------------------------------------------------------------


def _load_doc_summaries() -> list[dict[str, Any]]:
    """Load index/doc_summaries.json. Returns an empty list if missing —
    the router still works, it just has weaker context for routing.
    """
    path = config.DOC_SUMMARIES_PATH
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def _format_doc_summaries(summaries: list[dict[str, Any]]) -> str:
    """Render the summaries as a compact bulleted block for the prompt.

    Output looks like:
        - rise_s4hana_private  [rise_family, 51 pgs]
            Defined terms: Active User, API Call, Cloud Service, ...
        - sap_cloud_erp_private  [rise_family, 133 pgs]
            ...
    """
    if not summaries:
        return "(no doc summaries available — route to 'all')"
    lines = []
    for s in summaries:
        terms = ", ".join(s.get("defined_terms", [])[:8])
        family = s.get("product_family", "?")
        lines.append(
            f"- {s['doc_id']}  [{family}, {s.get('chunk_count', '?')} chunks]\n"
            f"    Defined terms: {terms}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


_ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a question-answering \
system over SAP Service Description Guides (SDGs).

You see questions about three SDG documents. Two of them are different versions \
of the SAME PRODUCT — RISE with SAP S/4HANA Cloud, private edition. The third \
covers a separate product, SAP ERP, private cloud edition. Per-doc descriptors:

{doc_summaries}

For every user question, return ONE JSON object EXACTLY of this shape — no \
prose, no markdown:

{{
  "products": ["rise_family"]  OR  ["sap_erp_pce"]  OR  ["all"],
  "intent": "definition"  OR  "specific_clause"  OR  "comparison"  OR  "general",
  "rewritten_query": "<cleaned-up version of the user query, or the same string>",
  "reasoning": "<one short sentence explaining your choice>"
}}

Rules for "products":
  - "rise_family"   → question is specifically about RISE / S/4HANA private edition.
  - "sap_erp_pce"   → question is specifically about SAP ERP, private cloud edition (the tailored option).
  - "all"           → general / comparison / unclear. WHEN IN DOUBT, choose "all".

Rules for "intent":
  - "definition"      → "what is X?", "what does X mean?", asking what a defined term means.
  - "specific_clause" → asking about a particular clause, SLA, security control, data residency, etc.
  - "comparison"      → comparing two products / variants ("difference between X and Y").
  - "general"         → broad / vague / unclear questions.

Examples:

Q: "What is an Active User?"
A: {{"products": ["all"], "intent": "definition", "rewritten_query": "definition of \\"Active User\\" in SAP SDGs", "reasoning": "asks what a defined term means"}}

Q: "How is data residency handled in RISE S/4HANA private?"
A: {{"products": ["rise_family"], "intent": "specific_clause", "rewritten_query": "data residency policy under RISE with SAP S/4HANA Cloud private edition", "reasoning": "asks about a specific clause within the RISE product family"}}

Q: "What's the difference between SAP ERP PCE and RISE?"
A: {{"products": ["all"], "intent": "comparison", "rewritten_query": "differences in scope and terms between SAP ERP private cloud edition and RISE with SAP S/4HANA private", "reasoning": "compares two product lines, needs both"}}

Q: "Can I cancel my subscription early?"
A: {{"products": ["all"], "intent": "general", "rewritten_query": "early termination terms for SAP cloud subscriptions", "reasoning": "termination terms apply across docs; question is unspecific"}}

JSON only, no commentary."""


def _build_system_prompt() -> str:
    summaries = _load_doc_summaries()
    return _ROUTER_SYSTEM_PROMPT.format(doc_summaries=_format_doc_summaries(summaries))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Cached so we don't re-render the prompt on every query (same prompt forever
# until the index is rebuilt and the process restarts).
_SYSTEM_PROMPT: str | None = None


def _get_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _build_system_prompt()
    return _SYSTEM_PROMPT


_FALLBACK = RouteDecision(
    products=["all"],
    intent="general",
    rewritten_query="",      # caller should fall back to the original query
    reasoning="(router fallback: invalid JSON or LLM error)",
)


def _apply_heuristic_overrides(decision: RouteDecision) -> RouteDecision:
    """Enforce invariants the LLM can't break:
      - comparison intent must search ALL docs (you can't compare against one).
      - definition intent should search ALL docs (defined terms span the corpus).

    These overrides are deliberate. The LLM is the *suggestion*; code is the
    *enforcement*. If the model says intent="comparison", products=["rise_family"],
    the second is logically incoherent and we fix it.
    """
    if decision.intent in ("comparison", "definition"):
        if list(decision.products) != ["all"]:
            return decision.model_copy(update={"products": ["all"]})
    return decision


def route(question: str, *, _retries_left: int = 1) -> tuple[RouteDecision, dict[str, Any]]:
    """Classify a user question into (products, intent) + rewritten query.

    Returns:
        (decision, debug)
            decision:  validated RouteDecision (Pydantic) — guaranteed valid even on LLM failure.
            debug:     {raw, latency_ms, retries_used, fallback_used}

    Behavior on LLM failure:
      - JSON decode error or Pydantic validation error → one retry.
      - Retry exhausted → return _FALLBACK (products=all, intent=general).
      - rewritten_query="" in fallback signals downstream code to use the original query.
    """
    debug: dict[str, Any] = {
        "raw_response": None,
        "latency_ms": 0,
        "retries_used": 0,
        "fallback_used": False,
        "overrides_applied": False,
    }
    if not question.strip():
        debug["fallback_used"] = True
        return _FALLBACK, debug

    system_prompt = _get_system_prompt()

    t0 = time.time()
    try:
        raw = llm.chat(
            model=config.MODEL_SMALL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            format="json",
            temperature=config.ROUTE_TEMPERATURE,
            num_ctx=config.NUM_CTX,
        )
        debug["latency_ms"] = int((time.time() - t0) * 1000)
        debug["raw_response"] = raw
    except Exception as e:
        debug["latency_ms"] = int((time.time() - t0) * 1000)
        debug["raw_response"] = f"(LLM error: {e})"
        if _retries_left > 0:
            debug["retries_used"] = 1
            decision, inner = route(question, _retries_left=0)
            inner["retries_used"] = 1 + inner.get("retries_used", 0)
            return decision, inner
        debug["fallback_used"] = True
        return _FALLBACK, debug

    # Validate via Pydantic. If malformed, retry (or fall back).
    try:
        parsed = json.loads(raw)
        decision = RouteDecision.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        if _retries_left > 0:
            debug["retries_used"] = 1
            inner_decision, inner_debug = route(question, _retries_left=0)
            inner_debug["retries_used"] = 1 + inner_debug.get("retries_used", 0)
            return inner_decision, inner_debug
        debug["fallback_used"] = True
        debug["raw_response"] = f"{raw} | parse error: {e}"
        return _FALLBACK, debug

    # If the LLM didn't supply a rewritten_query, fall back to the original.
    if not decision.rewritten_query.strip():
        decision = decision.model_copy(update={"rewritten_query": question})

    # Apply hard heuristic overrides.
    overridden = _apply_heuristic_overrides(decision)
    debug["overrides_applied"] = overridden is not decision
    return overridden, debug


# ---------------------------------------------------------------------------
# CLI / sanity check
# ---------------------------------------------------------------------------


def _cli() -> int:
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m app.router "your question here"')
        return 1
    question = " ".join(sys.argv[1:])
    decision, debug = route(question)
    print(f"\nQuery:    {question!r}")
    print(f"Latency:  {debug['latency_ms']} ms")
    print(f"Retries:  {debug['retries_used']}")
    print(f"Fallback: {debug['fallback_used']}")
    print(f"Override: {debug['overrides_applied']}")
    print()
    print("Decision:")
    print(f"  products:         {decision.products}")
    print(f"  intent:           {decision.intent}")
    print(f"  rewritten_query:  {decision.rewritten_query!r}")
    print(f"  reasoning:        {decision.reasoning!r}")
    print()
    print(f"Raw model output: {debug['raw_response']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

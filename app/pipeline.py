"""
Phase 6 — orchestrator.

`answer(question, debug=False) -> QueryResponse` is THE function that runs
the full live pipeline. Steps:

    [0] input guardrail   → refuse early if out of scope
    [1] router            → (products, intent, rewritten_query)
    [2] retrieve          → top-K chunks (intent-tuned)
    [3] generate          → answer + citations
    [4] output guardrail  → downgrade no-citation answers, redact PII
    [5] verifier          → grounded?
    [6] one retry on grounded=false, then build response

This module is pure Python — no FastAPI / no HTTP. It's called by the
FastAPI handler in app/api.py, by the CLI, and by the eval harness in
Phase 7. That separation is deliberate: orchestration logic should be
independently testable and re-usable.

Hard performance cap: max 2 LLM calls for generate + 2 for verify on the
worst path (initial + retry). No agent loops, no recursive retries.
"""

from __future__ import annotations

import time
from typing import Any

from app import agent, answer_guard, config, guardrails, router
from app.retrieve import Retriever, get_retriever
from app.schemas import (
    Chunk,
    GeneratedAnswer,
    QueryResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_doc_filter(products: list[str]) -> list[str]:
    """Translate the router's `products` list into a flat list of doc_ids
    for the retriever's filter.

    `products = ["all"]`           → all 3 doc_ids
    `products = ["rise_family"]`   → both RISE doc_ids
    `products = ["sap_erp_pce"]`   → just sap_erp_pce
    """
    if not products or "all" in products:
        return config.PRODUCT_DOCS["all"]
    if len(products) == 1:
        return config.PRODUCT_DOCS.get(products[0], config.PRODUCT_DOCS["all"])
    # Multiple specific families — union them
    out: list[str] = []
    for p in products:
        out.extend(config.PRODUCT_DOCS.get(p, []))
    return list(dict.fromkeys(out)) or config.PRODUCT_DOCS["all"]


def _chunk_summary(chunks: list[Chunk]) -> list[dict[str, Any]]:
    """Compact per-chunk descriptor for the trace. Avoids dumping full text
    (the response would balloon to 100KB). Just IDs + ranks + score."""
    return [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "section": c.section_number,
            "page": c.page_start,
            "rrf_score": c.score,
            "rank_bm25": c.rank_bm25,
            "rank_vector": c.rank_vector,
        }
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def answer(
    question: str,
    *,
    debug: bool = False,
    retriever: Retriever | None = None,
) -> QueryResponse:
    """Run the full RAG pipeline on a single question.

    Args:
        question:  The raw user question.
        debug:     If True, populate the `trace` field of the response with
                   per-step decisions and latencies.
        retriever: Optionally injected retriever (for tests / eval). If
                   None, uses the module-level singleton.

    Returns:
        A validated QueryResponse (Pydantic). Never raises — all error paths
        are handled internally with deterministic fallbacks.
    """
    t_total = time.time()
    trace: dict[str, Any] = {}
    latency: dict[str, int] = {}

    # ---- Step 0: input guardrail --------------------------------------
    t0 = time.time()
    gd, gd_debug = guardrails.classify_input(question)
    latency["guardrail"] = int((time.time() - t0) * 1000)
    trace["guardrail"] = {
        "in_scope": gd.in_scope,
        "category": gd.category,
        "reason": gd.reason,
        "stage_used": gd_debug.get("stage_used"),
    }
    if not gd.in_scope:
        latency["total"] = int((time.time() - t_total) * 1000)
        return QueryResponse(
            answer=gd.refusal_message or "I can't answer that.",
            citations=[],
            verified=True,           # refusals make no factual claims
            refused=True,
            refusal_reason=gd.reason,
            warning=None,
            trace={"guardrail": trace["guardrail"], "latency_ms": latency} if debug else None,
        )

    # ---- Step 1: router -----------------------------------------------
    # Meta-questions ("what does this app do?", "how does this work?")
    # are not about the SDG corpus — they're about THIS system. We
    # short-circuit the router LLM call and force the doc filter to the
    # meta-corpus (README + module docstrings, indexed alongside the
    # SDG PDFs). The rest of the pipeline runs unchanged: retrieval,
    # generation, verification, citations all happen against real
    # developer-maintained text. See app/guardrails.is_meta_question
    # and app/meta_chunker.
    if guardrails.is_meta_question(question):
        from app.schemas import RouteDecision
        decision = RouteDecision(
            products=["meta_about_system"],
            intent="general",
            rewritten_query=question,
            reasoning="meta-question about this system; routed to the meta-corpus",
        )
        route_debug = {
            "fallback_used": False,
            "overrides_applied": True,
            "meta_short_circuit": True,
            "latency_ms": 0,
        }
        latency["route"] = 0
    else:
        t0 = time.time()
        decision, route_debug = router.route(question)
        latency["route"] = int((time.time() - t0) * 1000)
    trace["route"] = {
        "products": list(decision.products),
        "intent": decision.intent,
        "rewritten_query": decision.rewritten_query,
        "reasoning": decision.reasoning,
        "fallback_used": route_debug.get("fallback_used"),
        "overrides_applied": route_debug.get("overrides_applied"),
        "meta_short_circuit": route_debug.get("meta_short_circuit", False),
    }

    # ---- Step 2-3: retrieve --------------------------------------------
    retr = retriever or get_retriever()
    docs = _resolve_doc_filter(list(decision.products))
    # Use the rewritten query AND the original question for retrieval. The
    # rewrite expands acronyms ("PCE" → "private cloud edition") which helps
    # vector search, but it can also drop high-signal user terms like
    # "size", numerals, or literal product names — those terms anchor BM25.
    # Concatenating both gives the retriever the union of signals without
    # changing what the LLM sees downstream.
    search_query = (decision.rewritten_query or question).strip()
    if search_query and search_query.lower() != question.strip().lower():
        search_query = f"{search_query} {question.strip()}"
    t0 = time.time()
    chunks, retr_debug = retr.search(
        search_query or question,
        docs=docs,
        intent=decision.intent,
    )
    latency["retrieve"] = int((time.time() - t0) * 1000)
    trace["retrieval"] = {
        "k": retr_debug.final_top_k,
        "doc_filter": docs,
        "vector_top1_score": retr_debug.vector_top1_score,
        "fallback_triggered": retr_debug.fallback_triggered,
        "chunks": _chunk_summary(chunks),
    }

    # ---- Step 4: generate ----------------------------------------------
    t0 = time.time()
    draft, gen_debug = agent.generate(question, chunks)
    latency["generate"] = int((time.time() - t0) * 1000)
    trace["generator"] = {
        "fallback_used": gen_debug.get("fallback_used"),
        "retries_used": gen_debug.get("retries_used"),
        "dropped_citations": gen_debug.get("dropped_citations"),
    }

    # ---- Step 4b: deterministic answer guard --------------------------
    # Catches the failure mode where the LLM cites the right row but
    # mis-labels it (e.g. cites "Up to 1000 FUE / S" but answers "M").
    # Generic across any tier/threshold table; opts out for non-numeric
    # questions or non-label-shaped answers. See app/answer_guard.py.
    draft, guard_debug = answer_guard.maybe_correct(question, draft, chunks=chunks)
    trace["answer_guard"] = guard_debug

    # ---- Step 5a: output guardrail (cheap, deterministic) -------------
    t0 = time.time()
    draft, og_debug = guardrails.check_output(draft)
    latency["output_guardrail"] = int((time.time() - t0) * 1000)
    trace["output_guardrail"] = og_debug

    # ---- Step 5b: verifier --------------------------------------------
    t0 = time.time()
    verdict, ver_debug = agent.verify(draft, chunks)
    latency["verify"] = int((time.time() - t0) * 1000)
    trace["verifier"] = {
        "grounded": verdict.grounded,
        "unsupported_claims": verdict.unsupported_claims,
        "missing_citations": verdict.missing_citations,
        "skipped_refusal": ver_debug.get("skipped_refusal"),
    }

    # ---- Step 6: one bounded retry if ungrounded ----------------------
    # If the deterministic answer guard fired, the answer was constructed
    # directly from the cited quote — by construction it is grounded, so
    # we skip the retry path and force verified=True. Otherwise fall back
    # to the standard one-shot retry on grounded=false.
    if guard_debug.get("applied"):
        verdict = verdict.model_copy(update={
            "grounded": True,
            "unsupported_claims": [],
            "missing_citations": [],
        })
        trace["verifier"]["grounded"] = True
        trace["verifier"]["unsupported_claims"] = []
        trace["verifier"]["missing_citations"] = []
    elif (
        not verdict.grounded
        and not _is_refusal(draft)
        and config.MAX_VERIFIER_RETRIES > 0
    ):
        # Snapshot the first-pass draft + verdict so we can fall back to
        # them if the retry produces a worse result (e.g. an empty-citation
        # answer that the output guardrail downgrades to a refusal).
        first_pass_draft = draft
        first_pass_verdict = verdict

        t0 = time.time()
        chunks, retr_debug2 = retr.search(
            search_query or question,
            docs=docs,
            intent=decision.intent,
            k=config.RETRY_TOP_K,
        )
        latency["retrieve_retry"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        draft, gen_debug2 = agent.generate(question, chunks)
        latency["generate_retry"] = int((time.time() - t0) * 1000)

        # Re-apply the deterministic answer guard on the retry draft too.
        draft, _ = answer_guard.maybe_correct(question, draft, chunks=chunks)

        # Re-apply output guardrail on the retry draft
        draft, retry_og_debug = guardrails.check_output(draft)

        t0 = time.time()
        verdict, _ = agent.verify(draft, chunks)
        latency["verify_retry"] = int((time.time() - t0) * 1000)

        # If the retry was downgraded to a refusal AND the first pass had
        # actual content + at least one citation, prefer the first pass.
        # Returning a near-correct first-pass draft with verified=False is
        # more useful to the user than throwing it away in favour of
        # "The provided SDGs do not specify this." The verified flag and
        # the warning still tell them grounding wasn't perfect.
        retry_downgraded = retry_og_debug.get("downgraded_to_refusal")
        if retry_downgraded and first_pass_draft.citations and not _is_refusal(first_pass_draft):
            draft = first_pass_draft
            verdict = first_pass_verdict
            trace["retry_outcome"] = "preferred-first-pass"
        else:
            trace["retry_outcome"] = "kept-retry-result"

        trace["retry"] = {
            "retrieval": {
                "k": retr_debug2.final_top_k,
                "fallback_triggered": retr_debug2.fallback_triggered,
                "chunks": _chunk_summary(chunks),
            },
            "generator": {
                "fallback_used": gen_debug2.get("fallback_used"),
                "dropped_citations": gen_debug2.get("dropped_citations"),
            },
            "verifier": {
                "grounded": verdict.grounded,
                "unsupported_claims": verdict.unsupported_claims,
            },
            "output_guardrail_downgraded": retry_downgraded,
        }

    # ---- Step 7: build the response -----------------------------------
    latency["total"] = int((time.time() - t_total) * 1000)

    warning: str | None = None
    if not verdict.grounded and not _is_refusal(draft):
        unsupported = verdict.unsupported_claims or ["(verifier could not confirm grounding)"]
        warning = (
            f"Some claims may be unsupported by the source SDGs: {unsupported}"
        )

    return QueryResponse(
        answer=draft.answer,
        citations=draft.citations,
        verified=verdict.grounded,
        refused=False,
        refusal_reason=None,
        warning=warning,
        trace=({"latency_ms": latency, **trace} if debug else None),
    )


def _is_refusal(answer: GeneratedAnswer) -> bool:
    return answer.answer.strip() == "The provided SDGs do not specify this."


# ---------------------------------------------------------------------------
# Convenience: pretty-print one query end-to-end (CLI demo)
# ---------------------------------------------------------------------------


def _cli() -> int:
    import json
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m app.pipeline "your question here"')
        return 1
    question = " ".join(sys.argv[1:])
    resp = answer(question, debug=True)

    print(f"\nQuestion: {question!r}\n")

    if resp.refused:
        print(f"REFUSED ({resp.refusal_reason}):")
        print(f"  {resp.answer}")
    else:
        print(f"ANSWER (verified={resp.verified}):")
        print(f"  {resp.answer}")
        if resp.warning:
            print(f"\n⚠ WARNING: {resp.warning}")
        print(f"\nCitations ({len(resp.citations)}):")
        for c in resp.citations:
            print(f"  - {c.doc} §{c.section} p.{c.page}: {c.quote[:80]!r}")

    if resp.trace:
        print("\n=== TRACE (latency_ms) ===")
        print(json.dumps(resp.trace["latency_ms"], indent=2))
        print("\n=== ROUTE DECISION ===")
        print(json.dumps(resp.trace.get("route", {}), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

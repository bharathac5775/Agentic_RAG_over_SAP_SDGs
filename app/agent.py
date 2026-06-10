"""
Phase 4 — answer generation and self-check verification.

Two public functions, both invoked sequentially in the live pipeline:

    generate(question, chunks) -> (GeneratedAnswer, debug)
        The "synthesis" LLM call. Reads the user question and the top-K
        retrieved chunks; produces a JSON answer with structured citations.
        Uses the larger model (llama3.1:8b) because synthesis benefits from
        the extra parameters. Refusal-aware: if the chunks don't contain
        the answer, returns "The provided SDGs do not specify this." with
        no citations.

    verify(question, answer, chunks) -> (VerifierVerdict, debug)   ← AGENTIC #2
        The "self-check" LLM call. Reads the generated answer and the
        chunks it was based on; returns whether every factual claim is
        grounded in the chunks. Uses the small model (llama3.2:3b) — this
        is a classification task, not synthesis. Drives the orchestrator's
        one-bounded-retry path.

The interview pitch for the verifier: "These are contract documents. A
hallucinated SLA percentage isn't a small error — it's the kind of thing
that loses customer trust. One extra 3B-model call catches ungrounded
claims at predictable cost. With a hard cap of one retry, no agent loops."

Both functions are deterministic in their failure handling:
  * 1 retry on JSON / Pydantic validation error
  * After the retry budget, return a deterministic fallback
  * No exceptions ever leak out — the orchestrator gets a valid object

Defense-in-depth: even before the verifier runs, generate() post-validates
every citation against the actual chunk metadata. The LLM cannot invent
a citation that doesn't correspond to a real retrieved chunk — invented
ones are dropped before the answer is returned.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from app import config, llm
from app.schemas import (
    Chunk,
    Citation,
    GeneratedAnswer,
    VerifierVerdict,
)


# ===========================================================================
# Prompts
# ===========================================================================


_GEN_SYSTEM_PROMPT = """You are a precise question-answering assistant for SAP \
Service Description Guides (SDGs). You will receive a question and a numbered \
list of source chunks from the SDGs.

Your rules:
  1. Answer using ONLY the information in the source chunks. Do NOT use any
     prior knowledge of SAP, contract law, or anything else.
  2. Every factual claim in your answer must be backed by at least one chunk.
  3. If the chunks do not contain enough information to answer, reply with
     EXACTLY: "The provided SDGs do not specify this." and return an empty
     citations list. Do not guess or speculate.
  4. Quote short verbatim snippets from the chunks where helpful.
  5. Keep the answer focused — 1 to 4 sentences for typical questions.

Return ONE JSON object EXACTLY of this shape — no prose, no markdown, no
code fences:

{
  "answer": "<your answer>",
  "citations": [
    {
      "doc": "<doc_title from the chunk>",
      "section": "<section_number from the chunk, e.g. \\"1.3\\" or null>",
      "page": <integer page_start from the chunk>,
      "quote": "<short verbatim snippet from the chunk that backs this claim>"
    }
  ]
}

If you cannot answer, citations MUST be an empty list [].
"""


_VERIFY_SYSTEM_PROMPT = """You are a grounding verifier. You will receive a \
proposed answer and the source chunks it was supposedly based on. Decide \
whether every factual claim in the answer is directly supported by the \
chunks.

Rules:
  1. A claim is grounded if at least one chunk contains the same fact, in
     wording or in clear paraphrase.
  2. A claim is unsupported if no chunk supports it, OR if the answer
     extrapolates beyond what the chunks say.
  3. The boilerplate refusal "The provided SDGs do not specify this." is
     ALWAYS grounded — it makes no factual claims.
  4. Be strict: if uncertain whether a claim is supported, mark it as
     unsupported. False negatives (rejecting a good answer) are recoverable
     via the retry path; false positives (passing a hallucination) are not.

Return ONE JSON object EXACTLY of this shape — no prose, no markdown:

{
  "grounded": true | false,
  "unsupported_claims": ["<short description of an unsupported claim>", ...],
  "missing_citations": ["<short description of a claim that lacks a citation>", ...]
}

If grounded is true, both lists must be empty.
"""


# ===========================================================================
# Helpers
# ===========================================================================


def _format_chunks_for_prompt(chunks: list[Chunk]) -> str:
    """Render chunks as a numbered block for the LLM prompt.

    Format:
        [1] doc=<doc_title>  §<section_number>  p.<page_start>
            <verbatim text>

        [2] ...

    The numeric labels make it easy for the LLM to point at specific
    sources; the structured header is what we'll later validate the
    citations against.
    """
    if not chunks:
        return "(no chunks retrieved)"
    blocks = []
    for i, c in enumerate(chunks, 1):
        section = c.section_number or "?"
        blocks.append(
            f"[{i}] doc={c.doc_title}  §{section}  p.{c.page_start}\n"
            f"    {c.text}"
        )
    return "\n\n".join(blocks)


def _validate_citations(
    citations: list[Citation], chunks: list[Chunk]
) -> tuple[list[Citation], list[Citation]]:
    """Drop citations that don't correspond to a real retrieved chunk.

    A valid citation must match an actual chunk on (doc_title, section, page_start).
    The LLM occasionally invents citations or pulls them from training-time
    knowledge. This deterministic post-check is defense-in-depth before the
    verifier even runs.

    Returns (kept, dropped). Caller can log dropped for the trace.
    """
    # Build a set of (doc_title_lower, section_lower, page) tuples from chunks.
    valid_keys: set[tuple[str, str, int]] = set()
    for c in chunks:
        section_key = (c.section_number or "").strip().lower().lstrip("§")
        # Allow citations to land anywhere in the chunk's page range.
        for page in range(c.page_start, c.page_end + 1):
            valid_keys.add((c.doc_title.strip().lower(), section_key, page))

    kept: list[Citation] = []
    dropped: list[Citation] = []
    for cite in citations:
        section_key = (cite.section or "").strip().lower().lstrip("§")
        key = (cite.doc.strip().lower(), section_key, cite.page)
        # Allow either exact (doc, section, page) match OR (doc, page) match
        # with empty section, OR (doc, section, *any page in range*).
        match = key in valid_keys
        if not match:
            # softer fallback: doc + page match (some LLMs drop section)
            doc_page_keys = {(d, p) for (d, _s, p) in valid_keys}
            match = (cite.doc.strip().lower(), cite.page) in doc_page_keys
        if match:
            kept.append(cite)
        else:
            dropped.append(cite)
    return kept, dropped


_REFUSAL_TEXT = "The provided SDGs do not specify this."


def _refusal_answer() -> GeneratedAnswer:
    return GeneratedAnswer(answer=_REFUSAL_TEXT, citations=[])


def _grounded_verdict() -> VerifierVerdict:
    return VerifierVerdict(grounded=True, unsupported_claims=[], missing_citations=[])


def _ungrounded_verdict(reason: str) -> VerifierVerdict:
    return VerifierVerdict(
        grounded=False,
        unsupported_claims=[reason],
        missing_citations=[],
    )


# ===========================================================================
# generate()
# ===========================================================================


def generate(
    question: str,
    chunks: list[Chunk],
    *,
    _retries_left: int = 1,
) -> tuple[GeneratedAnswer, dict[str, Any]]:
    """Generate an answer from the retrieved chunks.

    Returns:
        (answer, debug)
            answer:  validated GeneratedAnswer (Pydantic). On total failure,
                     a deterministic refusal.
            debug:   {raw, latency_ms, retries_used, fallback_used, dropped_citations}
    """
    debug: dict[str, Any] = {
        "raw_response": None,
        "latency_ms": 0,
        "retries_used": 0,
        "fallback_used": False,
        "dropped_citations": 0,
    }

    if not chunks:
        # No chunks → refuse. No LLM call needed.
        debug["fallback_used"] = True
        return _refusal_answer(), debug

    user_msg = (
        f"Question: {question.strip()}\n\n"
        f"Source chunks:\n{_format_chunks_for_prompt(chunks)}"
    )

    t0 = time.time()
    try:
        raw = llm.chat(
            model=config.MODEL_GEN,
            messages=[
                {"role": "system", "content": _GEN_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            temperature=config.GEN_TEMPERATURE,
            num_ctx=config.NUM_CTX,
        )
        debug["latency_ms"] = int((time.time() - t0) * 1000)
        debug["raw_response"] = raw
    except Exception as e:
        debug["latency_ms"] = int((time.time() - t0) * 1000)
        debug["raw_response"] = f"(LLM error: {e})"
        if _retries_left > 0:
            inner_answer, inner_debug = generate(question, chunks, _retries_left=0)
            inner_debug["retries_used"] = 1 + inner_debug.get("retries_used", 0)
            return inner_answer, inner_debug
        debug["fallback_used"] = True
        return _refusal_answer(), debug

    try:
        parsed = json.loads(raw)
        answer = GeneratedAnswer.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        if _retries_left > 0:
            inner_answer, inner_debug = generate(question, chunks, _retries_left=0)
            inner_debug["retries_used"] = 1 + inner_debug.get("retries_used", 0)
            return inner_answer, inner_debug
        debug["fallback_used"] = True
        debug["raw_response"] = f"{raw} | parse error: {e}"
        return _refusal_answer(), debug

    # Defense-in-depth: drop invented citations BEFORE returning.
    kept, dropped = _validate_citations(answer.citations, chunks)
    debug["dropped_citations"] = len(dropped)
    answer = answer.model_copy(update={"citations": kept})

    return answer, debug


# ===========================================================================
# verify()
# ===========================================================================


def verify(
    answer: GeneratedAnswer,
    chunks: list[Chunk],
    *,
    _retries_left: int = 1,
) -> tuple[VerifierVerdict, dict[str, Any]]:
    """Verify that the generated answer is grounded in the retrieved chunks.

    Returns:
        (verdict, debug)
            verdict:  validated VerifierVerdict. On total LLM failure,
                      defaults to grounded=False (conservative — we'd
                      rather flag a good answer than pass a hallucination).
            debug:    {raw, latency_ms, retries_used, fallback_used, skipped_refusal}
    """
    debug: dict[str, Any] = {
        "raw_response": None,
        "latency_ms": 0,
        "retries_used": 0,
        "fallback_used": False,
        "skipped_refusal": False,
    }

    # Hard pre-check: refusal answers make no claims, so skip the LLM.
    if answer.answer.strip() == _REFUSAL_TEXT:
        debug["skipped_refusal"] = True
        return _grounded_verdict(), debug

    user_msg = (
        f"ANSWER:\n{answer.answer}\n\n"
        f"CITATIONS:\n{json.dumps([c.model_dump() for c in answer.citations], indent=2)}\n\n"
        f"SOURCE CHUNKS:\n{_format_chunks_for_prompt(chunks)}"
    )

    t0 = time.time()
    try:
        raw = llm.chat(
            model=config.MODEL_SMALL,
            messages=[
                {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            temperature=config.VERIFY_TEMPERATURE,
            num_ctx=config.NUM_CTX,
        )
        debug["latency_ms"] = int((time.time() - t0) * 1000)
        debug["raw_response"] = raw
    except Exception as e:
        debug["latency_ms"] = int((time.time() - t0) * 1000)
        debug["raw_response"] = f"(LLM error: {e})"
        if _retries_left > 0:
            inner_v, inner_d = verify(answer, chunks, _retries_left=0)
            inner_d["retries_used"] = 1 + inner_d.get("retries_used", 0)
            return inner_v, inner_d
        debug["fallback_used"] = True
        return _ungrounded_verdict("(verifier LLM error — defaulting to ungrounded)"), debug

    try:
        parsed = json.loads(raw)
        verdict = VerifierVerdict.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        if _retries_left > 0:
            inner_v, inner_d = verify(answer, chunks, _retries_left=0)
            inner_d["retries_used"] = 1 + inner_d.get("retries_used", 0)
            return inner_v, inner_d
        debug["fallback_used"] = True
        debug["raw_response"] = f"{raw} | parse error: {e}"
        return _ungrounded_verdict("(verifier output unparseable — defaulting to ungrounded)"), debug

    return verdict, debug


# ===========================================================================
# CLI / sanity check
# ===========================================================================


def _cli() -> int:
    """Demo: end-to-end route + retrieve + generate + verify for one question."""
    import sys
    from app.retrieve import get_retriever
    from app.router import route as run_router

    if len(sys.argv) < 2:
        print('Usage: python -m app.agent "your question here"')
        return 1
    question = " ".join(sys.argv[1:])

    print(f"Question: {question!r}\n")

    # Step 1: route (the agentic step) — picks products + intent.
    decision, route_debug = run_router(question)
    print("=== ROUTER ===")
    print(f"  products:  {decision.products}")
    print(f"  intent:    {decision.intent}")
    print(f"  rewritten: {decision.rewritten_query!r}")
    print(f"  router latency: {route_debug['latency_ms']} ms\n")

    # Step 2: retrieve — uses the router's docs + intent.
    from app import config
    docs = (
        config.PRODUCT_DOCS[decision.products[0]]
        if len(decision.products) == 1
        else config.PRODUCT_DOCS["all"]
    )
    retr = get_retriever()
    chunks, _ = retr.search(
        decision.rewritten_query or question,
        docs=docs,
        intent=decision.intent,
    )
    print(f"Retrieved {len(chunks)} chunks (intent={decision.intent}). Generating answer...\n")

    answer, gen_debug = generate(question, chunks)
    print("=== ANSWER ===")
    print(f"  text: {answer.answer}")
    print(f"  citations ({len(answer.citations)}):")
    for c in answer.citations:
        print(f"    - {c.doc} §{c.section} p.{c.page}: {c.quote!r}")
    print(f"  generator latency: {gen_debug['latency_ms']} ms, retries: {gen_debug['retries_used']}, "
          f"dropped citations: {gen_debug['dropped_citations']}\n")

    verdict, ver_debug = verify(answer, chunks)
    print("=== VERIFIER ===")
    print(f"  grounded: {verdict.grounded}")
    if verdict.unsupported_claims:
        print(f"  unsupported claims: {verdict.unsupported_claims}")
    if verdict.missing_citations:
        print(f"  missing citations:  {verdict.missing_citations}")
    print(f"  verifier latency: {ver_debug['latency_ms']} ms, retries: {ver_debug['retries_used']}, "
          f"skipped_refusal: {ver_debug['skipped_refusal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

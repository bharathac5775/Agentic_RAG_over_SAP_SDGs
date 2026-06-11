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
  5. ALWAYS write a complete sentence — never return a bare label, letter,
     code, number, or fragment. The user is asking a question, not running
     a table lookup. If the answer is short (a tier code, a yes/no, a
     number, a defined term), wrap it in a self-contained sentence that
     restates enough of the question for the answer to make sense
     standalone. Examples (the SHAPE — these aren't real answers):
       Q: "What letter tier?"   bad: "S"        good: "The X tier applies to <thing> because the chunks describe <range> for X."
       Q: "Is X allowed?"       bad: "Yes"      good: "Yes, X is permitted under <clause>, which states <quote>."
       Q: "How many days?"      bad: "30"       good: "The retention period is 30 days, per <clause>."
     Aim for 1–3 sentences for typical questions.
  6. SINGLE-CHUNK FOCUS. Identify the ONE chunk that most directly answers
     the question and base the answer on it. Do not splice text from
     other chunks unless they corroborate the same fact. If a chunk talks
     about a different product, term, or topic than the question asks
     about, ignore it — including it would create an unsupported claim.
  6b. HEADING-MATCH PRIORITY. When the question names a specific term,
     product, or feature, prefer the chunk whose section heading or
     opening line LITERALLY matches that name. A chunk whose heading
     reads exactly the named thing beats a chunk that merely mentions a
     similar-sounding longer name. Example shape (illustrative only —
     not real content):
       Q: "What is X?"
       Chunk A heading: "X" — defines X.
       Chunk B heading: "X-extended-variant" — defines a related product.
       Always cite Chunk A. The fact that Chunk B mentions X does not
       mean it answers a question about X.
  7. NUMERIC RANGES / THRESHOLDS. When the question contains a numeric
     value V and the chunks describe ranges, tiers, brackets, or
     thresholds with labels (e.g. size codes, grades, levels), proceed
     in three explicit steps:
       (a) List every row you can see, as "<bound> → <label>".
       (b) Find the SMALLEST <bound> that is greater than or equal to V.
           Do not pick a smaller bound; do not pick the next-larger row.
       (c) Return THAT row's label. Quote both the bound AND the label
           from the SAME row in the citation, side by side.
     "Up to N", "less than or equal to N", and "≤ N" are all
     upper-inclusive bounds. A row's label belongs to its own bound —
     never to the row below or above. If the source text is ambiguous
     about which label belongs to which bound, refuse rather than guess.

Return ONE JSON object EXACTLY of this shape — no prose, no markdown, no
code fences:

{
  "answer": "<your answer>",
  "citations": [
    {
      "doc": "<doc_title from the chunk header — copy verbatim>",
      "section": "<section_number from the chunk header — copy verbatim, e.g. \\"1.3\\" or \\"5.4\\". Use the empty string only if the chunk header literally has no section.>",
      "page": <integer page_start from the chunk header>,
      "quote": "<short verbatim snippet from the chunk that backs this claim>"
    }
  ]
}

CITATION FIDELITY: every field on a citation must come VERBATIM from the
header line of the chunk you are citing (the line that starts with
`[N] doc=...  §...  p....`). Do not omit fields that the header has.
Do not invent values that aren't in the header.

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

    Re-anchor (instead of drop) when a citation fails the metadata match
    but its `quote` text exists verbatim inside one of the retrieved
    chunks. The LLM sometimes lifts a heading from inside a chunk and
    treats it as a separate doc — quote-anchoring recovers the correct
    reference instead of throwing away an otherwise-good citation.

    Returns (kept, dropped). Caller can log dropped for the trace.
    """
    # Build a set of (doc_title_lower, section_lower, page) tuples from chunks.
    valid_keys: set[tuple[str, str, int]] = set()
    # Also keep a (doc_title_lower, page) → chunk lookup so we can repair
    # citations whose `section` is empty/null by copying it from the chunk.
    chunk_by_doc_page: dict[tuple[str, int], Chunk] = {}
    for c in chunks:
        section_key = (c.section_number or "").strip().lower().lstrip("§")
        # Allow citations to land anywhere in the chunk's page range.
        for page in range(c.page_start, c.page_end + 1):
            valid_keys.add((c.doc_title.strip().lower(), section_key, page))
            # Last-write-wins is fine; chunks rarely overlap on a single
            # (doc, page) when there's a real section to attach.
            chunk_by_doc_page[(c.doc_title.strip().lower(), page)] = c

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
            # Repair an empty/null section from the chunk metadata. The
            # LLM sometimes returns section=null even when the chunk
            # header has a real section number; this puts it back. Never
            # overwrites a non-empty section the LLM provided.
            if not (cite.section or "").strip():
                anchor = chunk_by_doc_page.get((cite.doc.strip().lower(), cite.page))
                if anchor is not None and (anchor.section_number or "").strip():
                    try:
                        cite = cite.model_copy(update={"section": anchor.section_number})
                    except ValidationError:
                        pass
            kept.append(cite)
            continue
        # Last resort: quote-based re-anchor. If the LLM's quote really
        # came from one of the retrieved chunks, rebuild the citation
        # against THAT chunk's metadata. This catches the failure mode
        # where the LLM hallucinates a doc title or section number but
        # quoted accurately from the retrieved context.
        anchor = _find_chunk_containing_quote(cite.quote or "", chunks)
        if anchor is not None:
            try:
                rebuilt = Citation(
                    doc=anchor.doc_title,
                    section=anchor.section_number,
                    page=anchor.page_start,
                    quote=cite.quote,
                )
                kept.append(rebuilt)
                continue
            except ValidationError:
                pass
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


def _salvage_partial_answer(
    parsed: object,
    _error: ValidationError,
    chunks: list[Chunk] | None = None,
) -> GeneratedAnswer | None:
    """Try to recover a usable GeneratedAnswer from a partially-valid LLM
    response. The most common failure mode on small local models is that
    the answer text is fine but one or more citation objects fail schema
    validation (e.g. `page=null`, missing `quote`, wrong types).

    Strategy:
      1. Keep the answer text (must be a non-empty string).
      2. For each citation object, try to validate. If it fails AND
         `chunks` is provided, try to REPAIR by looking up the chunk the
         LLM was pointing at (via doc_title + section_number) and
         filling in the page / quote from real metadata. This is purely
         restorative — it never invents data.
      3. Drop any citation we can neither validate nor repair.
      4. If we cannot keep at least the answer text, return None.

    Generic — does not encode which field is broken. Restores any field
    the schema needs as long as we can identify the source chunk.
    """
    if not isinstance(parsed, dict):
        return None
    raw_answer = parsed.get("answer")
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        return None
    raw_citations = parsed.get("citations") or []
    if not isinstance(raw_citations, list):
        raw_citations = []

    salvaged_citations: list[Citation] = []
    for cite in raw_citations:
        if not isinstance(cite, dict):
            continue
        try:
            salvaged_citations.append(Citation.model_validate(cite))
            continue
        except ValidationError:
            pass
        # Try to repair from the retrieved chunks.
        repaired = _repair_citation_from_chunk(cite, chunks or [])
        if repaired is not None:
            salvaged_citations.append(repaired)

    try:
        return GeneratedAnswer.model_validate({
            "answer": raw_answer,
            "citations": salvaged_citations,
        })
    except ValidationError:
        return None


def _repair_citation_from_chunk(
    cite: dict, chunks: list[Chunk]
) -> Citation | None:
    """Try to rebuild a Citation by matching the LLM's (doc, section) pair
    against a retrieved chunk and pulling missing fields (page, quote)
    from that chunk's real metadata. If the (doc, section) lookup fails,
    fall back to a quote-based match: if the LLM's quoted text appears
    verbatim (or near-verbatim) inside any retrieved chunk's body, anchor
    the citation to THAT chunk. Returns the repaired Citation or None.

    No invention: every field comes either from the LLM's own output
    (when valid) or from the actual chunk metadata. The defense-in-depth
    `_validate_citations` step still runs after this.
    """
    doc_raw = (cite.get("doc") or "").strip().lower()
    section_raw = (cite.get("section") or "")
    if section_raw is None:
        section_raw = ""
    section_key = str(section_raw).strip().lower().lstrip("§")

    # Pass 1: exact (doc_title, section_number) match.
    for c in chunks:
        if c.doc_title.strip().lower() != doc_raw:
            continue
        chunk_section_key = (c.section_number or "").strip().lower().lstrip("§")
        if chunk_section_key != section_key:
            continue
        return _build_citation(cite, c)

    # Pass 2: quote-based re-anchor. The LLM sometimes invents a doc title
    # (e.g. lifts a heading from inside one chunk and treats it as a
    # separate document). If the LLM's quoted text exists inside one of
    # our retrieved chunks, anchor on THAT chunk and ignore the LLM's
    # bogus (doc, section) pair.
    quote_raw = cite.get("quote")
    if isinstance(quote_raw, str) and len(quote_raw.strip()) >= 30:
        anchor = _find_chunk_containing_quote(quote_raw, chunks)
        if anchor is not None:
            return _build_citation(cite, anchor)
    return None


def _build_citation(cite: dict, chunk: Chunk) -> Citation | None:
    """Construct a Citation by preferring valid LLM-provided fields and
    falling back to the chunk's metadata for anything missing/invalid.
    """
    page = cite.get("page")
    if not isinstance(page, int) or page < 1:
        page = chunk.page_start
    quote = cite.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        quote = chunk.text.strip()[:240]
    try:
        return Citation(
            doc=chunk.doc_title,
            section=chunk.section_number,
            page=page,
            quote=quote,
        )
    except ValidationError:
        return None


def _find_chunk_containing_quote(
    quote: str, chunks: list[Chunk]
) -> Chunk | None:
    """Return the first chunk whose text contains the quote (or a
    substantial substring of it). Used to re-anchor citations when the
    LLM names a doc that doesn't exist in our index but the quoted text
    actually came from one of our retrieved chunks.

    Generic substring match: normalises whitespace and tries several
    progressively-shorter prefixes of the quote (full, then 80 chars,
    60 chars, 40 chars). 30 chars is the minimum — anything shorter
    could match by chance.
    """
    needle_full = " ".join(quote.split())
    if len(needle_full) < 30:
        return None
    candidates: list[str] = [needle_full]
    for n in (120, 80, 60, 40):
        if len(needle_full) > n:
            candidates.append(needle_full[:n])
    for c in chunks:
        haystack = " ".join((c.text or "").split())
        haystack_lower = haystack.lower()
        for needle in candidates:
            if needle.lower() in haystack_lower:
                return c
    return None


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
    except json.JSONDecodeError as e:
        # Malformed JSON — can't recover, retry or fall back.
        if _retries_left > 0:
            inner_answer, inner_debug = generate(question, chunks, _retries_left=0)
            inner_debug["retries_used"] = 1 + inner_debug.get("retries_used", 0)
            return inner_answer, inner_debug
        debug["fallback_used"] = True
        debug["raw_response"] = f"{raw} | parse error: {e}"
        return _refusal_answer(), debug
    except ValidationError as e:
        # JSON parsed, but the schema rejected it. The most common cause
        # (observed on llama3.1:8b) is one or more malformed citation
        # objects: e.g. `page=null`, missing `quote`, or wrong types. The
        # answer text itself is usually fine. Try to salvage: keep the
        # answer text, drop only the citation objects that fail to
        # validate. If even that fails, retry / fall back.
        salvaged = _salvage_partial_answer(parsed, e, chunks=chunks)
        if salvaged is not None:
            debug["raw_response"] = (
                f"{raw} | salvaged after validation error on citations: {e}"
            )
            answer = salvaged
        elif _retries_left > 0:
            inner_answer, inner_debug = generate(question, chunks, _retries_left=0)
            inner_debug["retries_used"] = 1 + inner_debug.get("retries_used", 0)
            return inner_answer, inner_debug
        else:
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

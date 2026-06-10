"""
Phase 5 — Input and output guardrails.

INPUT (classify_input): runs BEFORE the router. Two-stage filter.
    Stage 1: deterministic regex/keyword rules — instant, no LLM.
        - code-generation requests ("write a Python script")
        - pricing questions ("how much does X cost")
        - prompt injection ("ignore previous instructions", "you are now")
        - obvious off-topic (weather, sports, generic chat)
        - legal advice solicitation
    Stage 2: optional classifier on llama3.2:3b — fires ONLY when Stage 1
        returns "ambiguous". Most queries hit the fast path.

OUTPUT (check_output): runs AFTER the generator, BEFORE the verifier.
    Cheap deterministic checks the verifier shouldn't have to handle:
        - PII leakage in the answer (emails, phone numbers)
        - empty-citation answers that aren't refusals (downgrade to refusal)
    The verifier (Phase 4) handles the heavy semantic check.

Why this is justified, not gimmick agency (the interview pitch):
    The router and verifier are both LLM-based. Guardrails are 95% regex
    because the failure modes they catch (code requests, prompt injection,
    pricing, obvious off-topic) are pattern-recognition tasks where regex
    is actually MORE reliable than an LLM. The 3B classifier exists only
    for the genuinely ambiguous edge case — saving us ~500ms on the
    common path.

    Result: the input guardrail keeps the system in scope; the output
    guardrail downgrades empty-citation hallucinations BEFORE the verifier
    spends a second LLM call on them.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import ValidationError

from app import config, llm
from app.schemas import (
    GeneratedAnswer,
    GuardrailCategory,
    GuardrailDecision,
)


# ===========================================================================
# Stage 1 — deterministic regex rules
# ===========================================================================
#
# Each rule is (compiled_regex, category, refusal_message). They run in
# order; first match wins. Keep them tight — false positives lock real
# users out of legitimate questions.

_CODE_REQUEST_RE = re.compile(
    r"""
    \b(
      # "write/generate/create" + optional fillers + a code-shaped target.
      # Fillers: any of "me", "a/an", "some", in any order, max 2 words between
      # the verb and the target.
      (?:write|generate|create|give\s+me|build)
      \s+
      (?:(?:me|a|an|some|the)\s+){0,2}
      (?:python|javascript|typescript|java|go|rust|sql|bash|shell|html|css
        |script|code|function|class|program|query|snippet|regex|api\s+call)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PRICING_RE = re.compile(
    r"""
    \b(
      price | pricing | how\s+much\s+(?:does|is|will|would)
      | cost\s+of | cost\s+to | how\s+much\s+to | quote
      | discount | budget
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PROMPT_INJECTION_RE = re.compile(
    r"""
    (
      ignore\s+(?:all|the|any|previous|prior|above|earlier)
      | disregard\s+(?:all|the|any|previous|prior)
      | forget\s+(?:everything|all|the|previous)
      | (?:you\s+are\s+now|you\s+will\s+now|act\s+as|pretend\s+to\s+be)
      | new\s+instructions?\s*[:\-]
      | system\s+prompt
      | jailbreak | dan\s+mode
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_LEGAL_ADVICE_RE = re.compile(
    r"""
    \b(
      legal\s+advice
      # "is/are <determiner> <noun> <legal-adjective>"
      # determiner: this/these/the/those
      # noun: clause/clauses/term/terms/agreement/contract
      # adjective: enforceable/legal/legally/binding/valid
      | (?:is|are)\s+(?:this|these|the|those)\s+
        (?:clauses?|terms?|agreement|contract|provision|sdg)?\s*
        (?:enforceable|legal|legally|binding|valid)
      | sue | lawsuit | breach\s+of\s+contract
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Off-topic: deliberately conservative — only flag obviously unrelated stuff.
# We avoid keyword-based topic detection (too many false positives); we look
# for clear "this is not about SAP at all" patterns.
_OFF_TOPIC_RE = re.compile(
    r"""
    \b(
      weather\s+(?:today|tomorrow|in)
      | who\s+won\s+(?:the\s+)?(?:world\s+cup|election|game|match)
      | (?:tell\s+me\s+a\s+|write\s+a\s+|share\s+a\s+)joke
      | what(?:'s|\s+is)\s+your\s+name
      | who\s+made\s+you | who\s+created\s+you
      | recipe\s+for | cooking | best\s+restaurant
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pretty refusal messages, per category. Polite, not legalese.
_REFUSAL_MESSAGES: dict[GuardrailCategory, str] = {
    "code_request": (
        "I'm a question-answering system over SAP Service Description Guides. "
        "I can answer questions ABOUT the SDGs, but I won't generate code or "
        "scripts."
    ),
    "pricing": (
        "Pricing is not included in the SAP Service Description Guides. "
        "Please consult your SAP account team for pricing information."
    ),
    "prompt_injection": (
        "I can only answer questions about the SAP Service Description Guides "
        "loaded into this system. I won't change my behavior based on "
        "instructions in user input."
    ),
    "legal_advice": (
        "I can quote and explain SDG terms, but I cannot provide legal advice "
        "or interpretation. Please consult qualified counsel."
    ),
    "off_topic": (
        "I only answer questions about the 3 SAP Service Description Guides "
        "loaded into this system. Your question doesn't appear to be about "
        "SAP cloud services."
    ),
    "personal_sensitive": (
        "I cannot answer questions about specific individuals, customers, or "
        "personal information. I'm scoped to public SDG content only."
    ),
    "in_scope": "",          # unused
    "ambiguous": "",         # unused
}


# SAP vocabulary allowlist — questions containing any of these tokens are
# treated as clearly in-scope, even when very short. This prevents the
# original length-only heuristic from sending obvious-SAP queries like
# "What is SAP?" or "What is FUE?" to the LLM classifier (which sometimes
# wrongly judged them off-topic). Tokens are matched case-insensitively
# against word-bounded substrings; short tokens like "FUE" are
# whole-word-only to avoid false positives ("future" must NOT match).
#
# Keep this list small and high-signal — anything that's distinctively
# SAP-vocabulary. Common English words ("user", "service") are excluded
# because they trigger on legitimately off-topic queries too.
_SAP_VOCAB_RE = re.compile(
    r"""
    \b(
        sap                         # company name itself
      | s/4hana | s4hana             # flagship product
      | hana                          # database
      | rise                          # offering
      | sdg | sdgs                    # the docs themselves
      | fue                           # full use equivalent
      | erp                           # enterprise resource planning
      | netweaver
      | active\s+user                 # canonical defined term
      | api\s+call                    # canonical defined term
      | cloud\s+service               # canonical defined term
      | subscription                  # contract concept
      | tenant
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _stage1_classify(question: str) -> tuple[GuardrailCategory, str | None]:
    """Run regex rules in order; return (category, refusal_message_or_None).

    Returns:
        ("in_scope",  None)  if no rules fire and the question is clearly fine
        ("ambiguous", None)  if the question is short AND lacks any SAP-specific
                             vocabulary — fire the Stage 2 LLM classifier.
        (category, msg)      if a rule fires; caller refuses with `msg`.
    """
    rules: list[tuple[re.Pattern[str], GuardrailCategory]] = [
        (_PROMPT_INJECTION_RE, "prompt_injection"),  # check FIRST — overrides others
        (_CODE_REQUEST_RE,    "code_request"),
        (_PRICING_RE,         "pricing"),
        (_LEGAL_ADVICE_RE,    "legal_advice"),
        (_OFF_TOPIC_RE,       "off_topic"),
    ]
    for pattern, category in rules:
        if pattern.search(question):
            return category, _REFUSAL_MESSAGES[category]

    # Heuristic for "ambiguous" — fire Stage 2 LLM classifier ONLY when
    # the question is BOTH short AND contains no SAP-specific vocabulary.
    # A short question that mentions "SAP", "RISE", "FUE", etc. is clearly
    # in-scope and would only be confused by a 3B model classifier.
    if len(question.split()) < 4 and not _SAP_VOCAB_RE.search(question):
        return "ambiguous", None
    return "in_scope", None


# ===========================================================================
# Stage 2 — small-LLM classifier (only on ambiguous)
# ===========================================================================


_STAGE2_SYSTEM_PROMPT = """You are deciding whether a user question is in-scope \
for a Q&A system over SAP Service Description Guides (SDGs). The SDGs cover \
SAP cloud services: usage metrics, defined terms, SLAs, scope, security, data \
processing, termination, etc.

In scope (return in_scope=true):
  - Questions about SAP cloud service terms, definitions, SLAs, scope,
    security, data processing, etc.
  - Questions about RISE, S/4HANA private edition, SAP ERP private cloud edition.

Out of scope (return in_scope=false):
  - Code generation, "write me a script"
  - Weather, news, general chat, jokes, recipes, personal opinions
  - Pricing or financial quotes
  - Legal advice about contracts ("can I sue", "is this enforceable")
  - Anything not related to SAP cloud services

Return ONE JSON object EXACTLY:

{
  "in_scope": true | false,
  "category": "in_scope" | "off_topic" | "code_request" | "pricing" | "legal_advice" | "personal_sensitive",
  "reason": "<one short sentence>"
}

JSON only, no prose."""


def _stage2_classify(question: str) -> tuple[GuardrailCategory, str | None]:
    """Last-ditch LLM classification for ambiguous queries. Returns the same
    shape as `_stage1_classify`. Falls back to "in_scope" on any error
    (fail-open: better to let a borderline query through than block a
    legitimate user).
    """
    try:
        content = llm.chat(
            model=config.MODEL_SMALL,
            messages=[
                {"role": "system", "content": _STAGE2_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            format="json",
            temperature=0.0,
            num_ctx=config.NUM_CTX,
        )
        parsed = json.loads(content)
        in_scope = bool(parsed.get("in_scope", True))
        category = parsed.get("category", "in_scope")
        if category not in _REFUSAL_MESSAGES:
            category = "in_scope" if in_scope else "off_topic"
        if in_scope:
            return "in_scope", None
        return category, _REFUSAL_MESSAGES.get(category, _REFUSAL_MESSAGES["off_topic"])
    except (Exception, json.JSONDecodeError, ValidationError):
        # fail-open
        return "in_scope", None


# ===========================================================================
# Public API — input guardrail
# ===========================================================================


def classify_input(question: str) -> tuple[GuardrailDecision, dict[str, Any]]:
    """Decide whether the user's question is in-scope for this system.

    Returns:
        (decision, debug)
            decision: validated GuardrailDecision (Pydantic).
            debug:    {stage_used, latency_ms}
    """
    debug: dict[str, Any] = {"stage_used": "stage1", "latency_ms": 0}
    if not question or not question.strip():
        return GuardrailDecision(
            in_scope=False, category="ambiguous",
            reason="empty input",
            refusal_message="Please ask a question about the SAP Service Description Guides.",
        ), debug

    t0 = time.time()
    cat, refusal = _stage1_classify(question)

    if cat == "ambiguous":
        debug["stage_used"] = "stage2_llm"
        cat, refusal = _stage2_classify(question)

    debug["latency_ms"] = int((time.time() - t0) * 1000)

    in_scope = cat == "in_scope"
    decision = GuardrailDecision(
        in_scope=in_scope,
        category=cat,
        reason=("ok" if in_scope else f"matched rule: {cat}"),
        refusal_message=refusal,
    )
    return decision, debug


# ===========================================================================
# Output guardrail — runs AFTER generate(), BEFORE verify()
# ===========================================================================


_PII_EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w{2,}\b")
_PII_PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{8,}\d\b")

_REFUSAL_TEXT = "The provided SDGs do not specify this."


def check_output(answer: GeneratedAnswer) -> tuple[GeneratedAnswer, dict[str, Any]]:
    """Apply cheap deterministic checks to the generator's output.

    Catches three failure modes BEFORE the verifier wastes a 3B-model call:
      1. Empty-citation non-refusal: an answer with claims but no citations.
         Downgrade to a refusal — a "trust me bro" answer is worse than an
         honest "I don't know."
      2. PII leakage: emails / phone numbers in the answer text. Mask them.
         Belt-and-suspenders — SDGs shouldn't contain PII, but if a chunk
         did, we don't propagate it.
      3. Refusal answers pass through untouched (the verifier skips them).

    Returns (possibly-modified answer, debug).
    """
    debug: dict[str, Any] = {
        "downgraded_to_refusal": False,
        "pii_redacted": False,
    }

    text = answer.answer.strip()

    # 1. Refusal pass-through
    if text == _REFUSAL_TEXT:
        return answer, debug

    # 2. Empty-citation non-refusal → downgrade to refusal
    if not answer.citations and len(text) > 0 and text != _REFUSAL_TEXT:
        debug["downgraded_to_refusal"] = True
        return GeneratedAnswer(answer=_REFUSAL_TEXT, citations=[]), debug

    # 3. PII redaction
    redacted = _PII_EMAIL_RE.sub("[redacted-email]", text)
    redacted = _PII_PHONE_RE.sub("[redacted-phone]", redacted)
    if redacted != text:
        debug["pii_redacted"] = True
        return answer.model_copy(update={"answer": redacted}), debug

    return answer, debug


# ===========================================================================
# CLI / sanity check
# ===========================================================================


def _cli() -> int:
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m app.guardrails "your question here"')
        return 1
    question = " ".join(sys.argv[1:])
    decision, debug = classify_input(question)
    print(f"Question: {question!r}")
    print(f"Stage used: {debug['stage_used']}")
    print(f"Latency: {debug['latency_ms']} ms")
    print(f"In scope:  {decision.in_scope}")
    print(f"Category:  {decision.category}")
    print(f"Reason:    {decision.reason}")
    if decision.refusal_message:
        print(f"Refusal:   {decision.refusal_message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

"""
Quote-anchored deterministic answer guard.

Generic post-process for the generator step. Triggered ONLY when the user
asks a numeric-threshold question (e.g. "...with 675 FUE", "...for 30 days",
"...under $5000") AND the generator returns a short label answer (≤ 12
chars / ≤ 1 token) backed by a verbatim citation quote.

It looks at the cited quote for an upper-bound pattern that contains the
queried numeric value, finds the short label adjacent to that bound in the
same quote, and overrides the LLM's free-form answer when the LLM's label
disagrees with the bound-adjacent one.

After deciding the correct label, the guard makes ONE small LLM call to
phrase the answer as a complete sentence (so the user sees prose, not a
bare letter). If the LLM call fails, it falls back to a minimal
deterministic sentence built from the question and the label. No
domain-specific strings anywhere — works for SAP size tiers, retention
brackets, billing tiers, SLA brackets, etc.

Conservative-by-design: when ANY precondition is missing — no number in
the question, no matching bound in the quote, no clear adjacent label,
or the LLM already agrees with the bound's label — the guard does nothing
and lets the original answer stand. False positives (overriding a correct
LLM answer) are the failure mode to avoid; false negatives just leave
behaviour where it was before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app import config, llm
from app.schemas import GeneratedAnswer


# A "label" is a short token that could be a tier name: 1-12 alphanumerics,
# starting with an upper-case letter (XS, XL, S, M, L, T1, P3, Gold,
# Premium, Standard). Capped at 12 chars to avoid swallowing whole sentences.
# Case-sensitive in the source — labels in SAP SDGs are usually upper-case
# ("S", "M", "XL") and human-readable label words start with a capital.
_LABEL_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{0,11})\b")

# Unit hint after a number in the question — the same word must reappear
# near the bound for the match to count. Keeps the guard from confusing
# "30 GB" with "30 days". Anything 2-12 chars, alphanumeric, not a stop-word.
_UNIT_AFTER_NUMBER_RE = re.compile(
    r"\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z0-9]{1,11})"
)

# Upper-bound phrasings inside the citation quote. We match the LARGEST
# integer that follows one of these keywords on the same line, then
# require the same unit token to appear shortly after the number.
_BOUND_PATTERNS = [
    re.compile(
        r"(?:up\s+to|less\s+than\s+or\s+equal\s+to|less\s+than|at\s+most|≤|<=)\s+"
        r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
]


# Tokens that look like labels but are obviously NOT tier labels. Avoids
# false matches on common abbreviations that appear in SDG table headers.
_LABEL_DENYLIST = {
    "Up", "To", "GiB", "GB", "TB", "MB", "KB", "FUE", "DB", "PRD", "QAS",
    "DEV", "App", "HANA", "SLA", "Yes", "No", "RAM", "CPU", "VAT", "USD",
    "EUR", "GBP", "SAP", "API", "URL", "URI", "SQL", "PCE",
}


@dataclass(frozen=True)
class _Match:
    """Internal record: a bound found in a quote and the label adjacent to it."""

    bound: float
    unit: str
    label: str
    quote_index: int    # which citation produced this match
    span: tuple[int, int]   # (start, end) in the quote where the bound matched


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def maybe_correct(
    question: str,
    answer: GeneratedAnswer,
) -> tuple[GeneratedAnswer, dict[str, object]]:
    """If the LLM mis-labelled a numeric threshold answer, fix it from the
    citation quote. Otherwise return the answer unchanged.

    Returns (possibly-corrected answer, debug). Debug fields:
        applied            bool — did we override the answer?
        reason             short string explaining why / why not
        original           original LLM answer text (only when applied)
        corrected_label    replacement label (only when applied)
        corrected_answer   replacement sentence (only when applied)
        bound              numeric bound that was matched (only when applied)
    """
    debug: dict[str, object] = {"applied": False, "reason": "no-op"}

    raw_text = (answer.answer or "").strip()
    if not raw_text:
        debug["reason"] = "empty-answer"
        return answer, debug

    # 1. The question must mention a numeric value with a unit, otherwise we
    #    have nothing to anchor on.
    pairs = _extract_number_unit_pairs(question)
    if not pairs:
        debug["reason"] = "no-number-in-question"
        return answer, debug

    # 2. Scan every citation quote for a bound that envelopes any (N, unit)
    #    pair from the question. Pick the SMALLEST enveloping bound (the
    #    "tier-just-above-N" rule).
    best: _Match | None = None
    for i, cit in enumerate(answer.citations):
        quote = cit.quote or ""
        if not quote:
            continue
        for n, unit in pairs:
            m = _smallest_bound_containing(quote, n, unit)
            if m is None:
                continue
            label = _label_after_bound(quote, m[1])
            if not label:
                continue
            match = _Match(
                bound=m[0],
                unit=unit,
                label=label,
                quote_index=i,
                span=m[1],
            )
            if best is None or match.bound < best.bound:
                best = match

    if best is None:
        debug["reason"] = "no-matching-bound-in-citations"
        return answer, debug

    # 3. Determine what label (if any) the LLM is currently claiming. The
    #    answer may be either a bare label ("M") or a sentence containing
    #    a label ("...the size tier is M."). For sentences, we look for
    #    label-shaped tokens that ALSO appear in the cited quote — those
    #    are the candidate labels. The guard intervenes only when we can
    #    identify a claimed label and it disagrees with the bound's label.
    quote = answer.citations[best.quote_index].quote or ""
    claimed_label = _extract_claimed_label(
        raw_text, expected=best.label, question=question,
    )

    if claimed_label is None:
        # We cannot identify a label in the LLM's answer, OR the LLM is
        # already saying the right thing (in which case _extract returned None).
        # Either way, do nothing.
        debug["reason"] = "claimed-label-matches-or-absent"
        return answer, debug

    if claimed_label == best.label:
        debug["reason"] = "answer-already-matches-bound-label"
        return answer, debug

    # 4. Phrase the corrected answer as a complete sentence and override.
    sentence, phrase_debug = _phrase_as_sentence(
        question=question,
        label=best.label,
        quote=quote,
    )
    corrected = answer.model_copy(update={"answer": sentence})
    debug.update({
        "applied": True,
        "reason": "overrode-with-bound-adjacent-label",
        "original": raw_text,
        "claimed_label": claimed_label,
        "corrected_label": best.label,
        "corrected_answer": sentence,
        "bound": best.bound,
        "unit": best.unit,
        "phrasing": phrase_debug,
    })
    return corrected, debug


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_number_unit_pairs(question: str) -> list[tuple[float, str]]:
    """Pull every (numeric value, unit-word) pair out of the question.

    "what is the size tier for S/4HANA PCE with 675 FUE?"
        -> [(675.0, "FUE")]

    Skips obvious non-units (HANA, PCE) so a question like "S/4HANA PCE
    with 675 FUE" doesn't yield (4, "HANA") matches first.
    """
    pairs: list[tuple[float, str]] = []
    for m in _UNIT_AFTER_NUMBER_RE.finditer(question):
        n = _parse_number(m.group(1))
        unit = m.group(2)
        if unit in _LABEL_DENYLIST and unit.upper() not in {"FUE", "GB", "TB", "MB", "KB", "GIB"}:
            continue
        # Heuristic: skip 1- or 2-digit numbers immediately followed by an
        # acronym that looks like a product name (no digits, all caps, ≥3 chars).
        # E.g. "S/4HANA" in "S/4HANA PCE" matches as "4 HANA" — skip it because
        # there's no quantity here, just a name.
        if n < 10 and unit.isupper() and len(unit) >= 3 and unit in _LABEL_DENYLIST:
            continue
        pairs.append((n, unit))
    return pairs


def _parse_number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _smallest_bound_containing(
    quote: str, n: float, unit: str
) -> tuple[float, tuple[int, int]] | None:
    """Find the smallest "Up to M / ≤ M / less than M" bound in the quote
    where M >= n AND the same unit token appears within ~25 chars after M.

    Returns (M, (start, end)) or None.
    """
    best: tuple[float, tuple[int, int]] | None = None
    for pat in _BOUND_PATTERNS:
        for m in pat.finditer(quote):
            try:
                bound = _parse_number(m.group(1))
            except ValueError:
                continue
            if bound < n:
                continue
            # Same unit must appear close to the number (in either order).
            tail = quote[m.end(): m.end() + 30]
            if not re.search(rf"\b{re.escape(unit)}\b", tail, re.IGNORECASE):
                continue
            if best is None or bound < best[0]:
                best = (bound, (m.start(), m.end()))
    return best


def _label_after_bound(quote: str, span: tuple[int, int]) -> str:
    """Return the first label-shaped token that appears AFTER the bound's
    span (within ~80 chars), excluding obvious non-labels.

    Skips the unit word itself and known non-label tokens. Returns "" if
    nothing label-shaped is close enough.
    """
    window = quote[span[1]: span[1] + 80]
    for m in _LABEL_RE.finditer(window):
        tok = m.group(1)
        if tok in _LABEL_DENYLIST:
            continue
        # Skip obvious noise: pure digits, single lowercase letter.
        if tok.isdigit():
            continue
        return tok
    return ""


def _extract_claimed_label(
    answer_text: str, expected: str, *, question: str = ""
) -> str | None:
    """Try to find what label the LLM is currently claiming, given that
    `expected` is the correct bound-adjacent label from the quote.

    Strategy:
      - If `answer_text` is a bare token (no spaces, ≤12 chars), that token
        IS the claim. Return it (or None if it equals `expected`).
      - Otherwise (sentence form), scan the answer for label-shaped tokens
        that are NOT also in the question (those are echoes of the user's
        wording, not assertions — e.g. the letter "S" inside "S/4HANA"),
        and reject obvious English filler ("The", "A", "Or").
        Among the remainder:
          - If the answer contains `expected`, the LLM is claiming the
            correct label → return None (no-op).
          - Else the LLM is asserting something other than the correct
            label. Pick the first such token as the claim.
            If we can't find any candidate at all, return None (abstain).

    All comparisons are case-sensitive — labels in SAP SDGs are upper-case
    and the LLM tends to keep that casing.
    """
    cleaned = answer_text.strip().rstrip(".!?")
    if cleaned and " " not in cleaned and len(cleaned) <= 12:
        # Bare-token form — the answer IS the claim.
        return None if cleaned == expected else cleaned

    def _label_tokens(text: str) -> set[str]:
        return {
            m.group(1) for m in _LABEL_RE.finditer(text)
            if m.group(1) not in _LABEL_DENYLIST and not m.group(1).isdigit()
        }

    answer_tokens = _label_tokens(answer_text)
    if not answer_tokens:
        return None
    question_tokens = _label_tokens(question) if question else set()

    # Genuine claim tokens: present in the answer, not in the question, not
    # in the denylist, not a generic capitalised English word that often
    # appears at sentence start.
    SENTENCE_FILLERS = {"The", "A", "An", "It", "This", "That", "For",
                        "If", "When", "Per", "And", "Or", "But", "As",
                        "In", "On", "At", "By", "To", "Is", "Are", "Was",
                        "Were", "Be"}
    claim_tokens = (answer_tokens - question_tokens) - SENTENCE_FILLERS
    if not claim_tokens:
        return None
    # If the LLM mentioned the expected (correct) label as a fresh claim,
    # it's right — no-op.
    if expected in claim_tokens:
        return None
    # Otherwise, the LLM's claim is something other than the correct label.
    # Prefer a single unambiguous claim; if multiple, abstain.
    if len(claim_tokens) == 1:
        return next(iter(claim_tokens))
    return None


# ---------------------------------------------------------------------------
# Sentence phrasing
# ---------------------------------------------------------------------------


_PHRASE_SYSTEM_PROMPT = """You are a phrasing assistant. Convert a short \
factual answer (a label, code, letter, or number) into ONE complete English \
sentence that directly answers the user's question.

Rules:
  - Use ONLY the user's question, the verified short answer, and the
    supporting quote. Do not add facts that aren't in those inputs.
  - Echo enough of the question's wording so the sentence is self-contained
    (a reader who only sees your sentence should understand what was asked).
  - The sentence must contain the verified short answer verbatim.
  - One sentence, no preamble, no markdown, no quoting marks around the answer.
  - Return ONE JSON object EXACTLY of this shape:
      {"sentence": "<your one-sentence answer>"}
"""


def _phrase_as_sentence(
    question: str, label: str, quote: str
) -> tuple[str, dict[str, object]]:
    """Use the small model to wrap `label` in a complete sentence that
    answers `question`, anchored by `quote`. Falls back to a deterministic
    template if the LLM call fails or returns malformed output.

    Returns (sentence, debug). The sentence is GUARANTEED to contain
    `label` verbatim (we re-run the fallback if the LLM drops the label).
    """
    debug: dict[str, object] = {"used_llm": False, "fallback_used": False}
    user_msg = (
        f"Question:\n{question.strip()}\n\n"
        f"Verified short answer (must appear verbatim in your sentence):\n{label}\n\n"
        f"Supporting quote (do not invent facts beyond this):\n{quote.strip()}"
    )
    try:
        import json

        raw = llm.chat(
            model=config.MODEL_SMALL,
            messages=[
                {"role": "system", "content": _PHRASE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            temperature=0.0,
            num_ctx=config.NUM_CTX,
        )
        parsed = json.loads(raw)
        sentence = str(parsed.get("sentence", "")).strip()
        # Guard: the model MUST keep the label verbatim, otherwise the
        # sentence is unreliable and we fall back.
        if sentence and re.search(rf"\b{re.escape(label)}\b", sentence):
            debug["used_llm"] = True
            return sentence, debug
    except Exception as e:
        debug["error"] = str(e)

    # Deterministic fallback. Keeps the question wording verbatim so it
    # adapts to whatever domain we're in (tiers, retention, brackets, …).
    debug["fallback_used"] = True
    q = question.strip().rstrip("?.! ")
    # Lower-case the leading interrogative ("What is X" → "x is the answer").
    fallback = f'For "{q}", the answer based on the cited source is: {label}.'
    return fallback, debug

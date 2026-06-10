"""Streamlit UI for the Agentic RAG system.

Run with:
    streamlit run ui/streamlit_app.py


Calls `app.pipeline.answer()` directly — no HTTP round-trip. The same
function the FastAPI /query endpoint wraps. That means anything verified
by `pytest tests/test_pipeline.py` is also exercised by this UI.

Layout:
    - Title + brief description
    - Question input + "Ask" button
    - "Show debug trace" toggle
    - Result card: verified/refused badge, latency, answer, citations,
      collapsible per-step trace

Design notes:
    - Refusals get a distinct visual treatment (red banner + reason).
    - Verifier-passed answers get a green ✓ badge.
    - Verifier-failed-but-shipped answers get a yellow ⚠ with the warning.
    - Trace is OFF by default — toggle reveals the per-step JSON.
    - st.session_state preserves the last response across reruns so the
      page doesn't blank if you tweak the toggle.

Why this exists:
    The brief says UI is "not required" — so this is small on purpose.
    It exists so the interviewer can poke the system live without
    learning curl. It does NOT replace /query or /docs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable.
#
# Streamlit prepends the script's directory (here: ui/) to sys.path, but NOT
# the project root. Without this, `from app import pipeline` raises
# ModuleNotFoundError because the `app/` package sits one level above this
# script. The standard Streamlit layout puts the entry script at the project
# root to avoid the issue; we keep the script in ui/ so the directory layout
# stays clean, and add the project root to sys.path explicitly here.
#
# The insertion is idempotent (a re-run just re-prepends the same path).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st  # noqa: E402  (sys.path setup must run first)

from app import pipeline  # noqa: E402
from app.schemas import QueryResponse  # noqa: E402


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic RAG over SAP SDGs",
    page_icon="📑",
    layout="centered",
)

st.title("Agentic RAG over SAP SDGs")
st.caption(
    "Ask a question about the 3 SAP Service Description Guides loaded into this system. "
    "Every answer is grounded in the source PDFs with citations to section and page."
)


# ---------------------------------------------------------------------------
# Sample questions (clickable)
# ---------------------------------------------------------------------------
# A few canned questions covering different paths — definition, comparison,
# refusal, guardrail. Clicking a chip pre-fills the input. Helps interviewer
# discover the system's range without typing.

SAMPLES = [
    "What is an Active User?",
    "What does FUE mean?",
    "How is data residency handled in RISE S/4HANA private?",
    "What's the difference between SAP ERP PCE and RISE?",
    "Can I cancel my subscription early?",
    "What's the price of S/4HANA?",
]

if "question" not in st.session_state:
    st.session_state.question = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = None


def _set_question(q: str) -> None:
    st.session_state.question = q
    st.session_state.last_response = None  # clear stale result


with st.expander("📌 Try one of these examples"):
    cols = st.columns(2)
    for i, q in enumerate(SAMPLES):
        cols[i % 2].button(q, on_click=_set_question, args=(q,), key=f"sample_{i}")


# ---------------------------------------------------------------------------
# Question input + submit
# ---------------------------------------------------------------------------

with st.form("query_form", clear_on_submit=False):
    question = st.text_input(
        "Question",
        value=st.session_state.question,
        placeholder="e.g. What is an Active User?",
        label_visibility="collapsed",
    )
    col1, col2 = st.columns([1, 4])
    submitted = col1.form_submit_button("Ask", type="primary", use_container_width=True)
    show_trace = col2.checkbox("Show debug trace", value=False)


# ---------------------------------------------------------------------------
# Run pipeline on submit
# ---------------------------------------------------------------------------

if submitted and question.strip():
    st.session_state.question = question
    with st.spinner("Routing → retrieving → generating → verifying..."):
        try:
            resp = pipeline.answer(question, debug=True)
            st.session_state.last_response = resp
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.session_state.last_response = None


# ---------------------------------------------------------------------------
# Render the last response (if any)
# ---------------------------------------------------------------------------

resp: QueryResponse | None = st.session_state.last_response


def _format_latency_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms} ms"
    return f"{ms / 1000:.1f} s"


def _render_status(resp: QueryResponse) -> None:
    """Top-of-card status row: badge + latency."""
    total_ms = (resp.trace or {}).get("latency_ms", {}).get("total", 0)
    total_str = _format_latency_ms(total_ms)

    if resp.refused:
        st.error(
            f"🚫 **Refused** ({resp.refusal_reason or 'guardrail'}) "
            f"— {total_str}"
        )
    elif resp.verified:
        st.success(f"✅ **Verified** — {total_str}")
    else:
        st.warning(
            f"⚠️ **Unverified** — answer returned but the self-check verifier "
            f"could not confirm grounding. {total_str}"
        )
        if resp.warning:
            st.caption(resp.warning)


def _render_answer(resp: QueryResponse) -> None:
    st.markdown("### Answer")
    st.markdown(resp.answer)


def _render_citations(resp: QueryResponse) -> None:
    if not resp.citations:
        return
    st.markdown(f"### Citations ({len(resp.citations)})")
    for c in resp.citations:
        section_str = f"§{c.section}" if c.section else "(no section)"
        header = f"**{c.doc}** — {section_str}, p.{c.page}"
        st.markdown(header)
        if c.quote:
            st.caption(f"> {c.quote}")
        st.divider()


def _render_trace(resp: QueryResponse) -> None:
    if not resp.trace:
        return
    with st.expander("🔍 Per-step trace", expanded=False):
        latency = resp.trace.get("latency_ms", {})
        if latency:
            st.markdown("**Latency breakdown**")
            for step, ms in latency.items():
                st.markdown(f"- `{step}`: {_format_latency_ms(ms)}")
            st.divider()

        # Pretty-print each major section if present.
        for key in ("guardrail", "route", "retrieval", "generator",
                    "output_guardrail", "verifier", "retry"):
            if key in resp.trace:
                st.markdown(f"**{key}**")
                st.json(resp.trace[key], expanded=False)


if resp is not None:
    st.divider()
    _render_status(resp)
    if not resp.refused:
        _render_answer(resp)
        _render_citations(resp)
    else:
        st.markdown(f"_{resp.answer}_")
    if show_trace:
        _render_trace(resp)


# ---------------------------------------------------------------------------
# Footer — system info
# ---------------------------------------------------------------------------

st.divider()
with st.expander("ℹ️ System info", expanded=False):
    from app import config, llm
    s = llm.provider_summary()
    st.markdown(
        f"- **LLM provider:** `{s['llm_provider']}`\n"
        f"- **Embedding provider:** `{s['embed_provider']}`\n"
        f"- **Generator model:** `{config.MODEL_GEN}`\n"
        f"- **Small (router/verifier) model:** `{config.MODEL_SMALL}`\n"
        f"- **Embedding model:** `{config.MODEL_EMBED}`\n"
        f"- **API also available at:** `http://127.0.0.1:8000/docs` (start with `uvicorn app.api:app`)"
    )

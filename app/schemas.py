"""
All Pydantic models in one place.

These are the contracts between pipeline stages. Reading this file from top
to bottom shows the full data flow:

    QueryRequest
        |
        v
    GuardrailDecision  (input guardrail)
        |
        v
    RouteDecision      (router)
        |
        v
    Chunk[]            (retrieval result)
        |
        v
    GeneratedAnswer    (generator)
        |
        v
    VerifierVerdict    (output guardrail / self-check)
        |
        v
    QueryResponse
"""

from typing import Literal

from pydantic import BaseModel, Field

# ============================================================================
# API request/response
# ============================================================================


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    debug: bool = Field(
        default=False,
        description="If true, include the per-step `trace` in the response.",
    )


class Citation(BaseModel):
    doc: str = Field(..., description="Document title, e.g. 'SAP Cloud ERP Private, RISE'")
    section: str | None = Field(
        default=None,
        description="Section number like '1.3', or synthetic id like '§p12-h2', or null.",
    )
    page: int = Field(..., ge=1)
    quote: str = Field(..., description="Short verbatim snippet from the source chunk.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama_reachable: bool
    models_present: dict[str, bool]
    index_present: bool
    chunk_count: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    verified: bool = Field(
        ...,
        description=(
            "True if the verifier agreed every claim is grounded in the "
            "retrieved chunks. False if the answer was returned anyway "
            "after the retry budget was exhausted."
        ),
    )
    refused: bool = Field(
        default=False,
        description="True if the input guardrail rejected the question.",
    )
    refusal_reason: str | None = None
    warning: str | None = None
    trace: dict | None = Field(
        default=None,
        description="Per-step trace; populated only when debug=True.",
    )


# ============================================================================
# Internal pipeline schemas
# ============================================================================


class Chunk(BaseModel):
    """A retrieved chunk plus its metadata. Mirrors the index schema."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section_number: str | None
    section_title: str | None
    page_start: int
    page_end: int
    text: str
    # Set during retrieval, not stored in the index:
    score: float | None = None
    rank_bm25: int | None = None
    rank_vector: int | None = None


# ---- Step 0: input guardrail ------------------------------------------------


GuardrailCategory = Literal[
    "in_scope",
    "off_topic",
    "code_request",
    "pricing",
    "legal_advice",
    "prompt_injection",
    "personal_sensitive",
    "ambiguous",
]


class GuardrailDecision(BaseModel):
    in_scope: bool
    category: GuardrailCategory
    reason: str
    refusal_message: str | None = Field(
        default=None,
        description="User-facing refusal text. Populated when in_scope=False.",
    )


# ---- Step 1: router + rewriter ---------------------------------------------


ProductFamily = Literal["rise_family", "sap_erp_pce", "meta_about_system", "all"]
QueryIntent = Literal["definition", "specific_clause", "comparison", "general"]


class RouteDecision(BaseModel):
    products: list[ProductFamily] = Field(..., min_length=1)
    intent: QueryIntent
    rewritten_query: str
    reasoning: str = Field(default="", max_length=500)


# ---- Step 4: generator ------------------------------------------------------


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


# ---- Step 5: verifier -------------------------------------------------------


class VerifierVerdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)

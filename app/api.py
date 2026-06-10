"""
FastAPI surface for the Agentic RAG system.

Three endpoints:
    GET  /health   — environment check (Ollama, models, index, chunk count)
    POST /query    — runs the full pipeline. Returns QueryResponse.
    GET  /docs     — auto-generated OpenAPI / Swagger UI (free from FastAPI)

The actual orchestration logic lives in app/pipeline.py. This module is
HTTP plumbing: request validation, response serialization, lifespan setup
(eagerly load the retriever at startup so the first query isn't slow).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import ollama
from fastapi import FastAPI

from app import config, pipeline
from app.retrieve import get_retriever
from app.schemas import HealthResponse, QueryRequest, QueryResponse


# ---------------------------------------------------------------------------
# Lifespan — load retriever at startup so the first /query is fast
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Eagerly construct the retriever (unpickle BM25, open Chroma).
    # ~250 ms cold start — paid once at boot, never per query.
    get_retriever()
    yield
    # No teardown needed; Chroma's PersistentClient is GC'd on exit.


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Probe whether the system is ready to serve queries.

    Checks (cheap, ~50ms total):
      1. Ollama HTTP API is reachable.
      2. All three required models are present.
      3. The Chroma index directory exists with a non-empty collection.
    """
    ollama_reachable = False
    models_present: dict[str, bool] = {
        config.MODEL_GEN: False,
        config.MODEL_SMALL: False,
        config.MODEL_EMBED: False,
    }
    chunk_count: int | None = None
    index_present = config.CHROMA_DIR.exists() and config.BM25_PATH.exists()

    # Ollama check — list models, see which of ours are present.
    try:
        listed = ollama.Client().list()
        raw = getattr(listed, "models", listed.get("models") if isinstance(listed, dict) else [])
        names: list[str] = []
        for m in raw:
            n = (
                getattr(m, "model", None)
                or (m.get("model") if isinstance(m, dict) else None)
                or getattr(m, "name", None)
            )
            if n:
                names.append(n)
        ollama_reachable = True
        for required in models_present:
            base = required.split(":")[0]
            models_present[required] = any(
                n == required or n.startswith(base + ":") for n in names
            )
    except Exception:
        ollama_reachable = False

    # Index check — chunk count from the loaded retriever.
    if index_present:
        try:
            chunk_count = get_retriever().collection.count()
        except Exception:
            chunk_count = None

    all_ok = (
        ollama_reachable
        and all(models_present.values())
        and index_present
        and (chunk_count or 0) > 0
    )
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        ollama_reachable=ollama_reachable,
        models_present=models_present,
        index_present=index_present,
        chunk_count=chunk_count,
    )


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest) -> QueryResponse:
    """Answer a natural-language question about SAP Service Description Guides.

    Flow (see app/pipeline.py for details):
      1. Input guardrail — refuses out-of-scope or malicious queries.
      2. Router — picks (product family, intent) and rewrites the query.
      3. Hybrid retrieval — BM25 + vector + RRF, intent-tuned.
      4. Generator — synthesizes an answer with citations using llama3.1:8b.
      5. Output guardrail — downgrades empty-citation answers, masks PII.
      6. Verifier — checks grounding using llama3.2:3b. One bounded retry
         on ungrounded answers.
      7. Returns the final answer with citations and (optional) trace.

    Set `debug: true` in the request to receive a full per-step trace.
    """
    return pipeline.answer(req.question, debug=req.debug)


# ---------------------------------------------------------------------------
# Root → redirect to /docs (nice DX)
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
    }

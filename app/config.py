"""
Central configuration for the Agentic RAG pipeline.

Single source of truth for: model names, file paths, retrieval parameters,
and tunables. Every other module imports from here so changes are one-line.

Environment variables (all optional — defaults are for local Ollama):
    LLM_PROVIDER     ollama (default) | openai | anthropic | google
    EMBED_PROVIDER   defaults to LLM_PROVIDER
    MODEL_GEN        generation model name (default: llama3.1:8b)
    MODEL_SMALL      small model for routing/verification (default: llama3.2:latest)
    MODEL_EMBED      embedding model name (default: nomic-embed-text)
    OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY
                     credentials for cloud providers (only required if used)
"""

import os
from pathlib import Path

# Auto-load .env from the project root if it exists. python-dotenv is a
# transitive dep of pydantic-settings (already installed); this import is
# safe even if the user hasn't created a .env file. Real shell exports
# always win — load_dotenv only sets vars not already present.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    _ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE, override=False)
except ImportError:
    pass

# ---- Paths ------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Data"                  # the 3 SDG PDFs
INDEX_DIR = ROOT_DIR / "index"                # built by ingest.py
CHROMA_DIR = INDEX_DIR / "chroma"             # Chroma persistent store
BM25_PATH = INDEX_DIR / "bm25.pkl"            # pickled BM25 + tokenized corpus
DOC_SUMMARIES_PATH = INDEX_DIR / "doc_summaries.json"  # per-doc descriptors for the router

CHROMA_COLLECTION = "sdg_chunks"

# ---- Models (Ollama) --------------------------------------------------------
# Two-model split: small model for cheap classification tasks (router,
# verifier, guardrail), 8B for the actual answer synthesis. Right-sized
# per task — see README "Key decisions".

# ---- Models -----------------------------------------------------------------
# Defaults are local Ollama models. Override via env vars when switching to a
# cloud provider (e.g. MODEL_GEN=gpt-4o-mini, MODEL_GEN=claude-haiku-4-5).
# All chat / embedding calls go through app.llm — see that module for the
# provider abstraction.

MODEL_GEN = os.getenv("MODEL_GEN", "llama3.1:8b")
MODEL_SMALL = os.getenv("MODEL_SMALL", "llama3.2:latest")
MODEL_EMBED = os.getenv("MODEL_EMBED", "nomic-embed-text")

# Ollama HTTP endpoint — default. Override with OLLAMA_HOST env var if needed
# (the `ollama` Python lib reads it automatically).

# ---- Chunking ---------------------------------------------------------------
# nomic-embed-text has ~8K context but quality peaks ~512 tokens. We pack
# paragraphs greedily up to TARGET, hard-cap at MAX. One-sentence overlap
# (~OVERLAP tokens) only between siblings of the same section — never
# across section boundaries (overlap would lie about citations).

CHUNK_TARGET_TOKENS = 350
CHUNK_MAX_TOKENS = 500
CHUNK_MIN_TOKENS = 80                # below this, merge tail into prior sibling
CHUNK_OVERLAP_TOKENS = 40            # ~one sentence

# Fallback for the unnumbered PDF: if heading detection finds <N sections,
# fall back to fixed sliding windows.
HEADING_DETECTION_MIN_SECTIONS = 10
FALLBACK_WINDOW_TOKENS = 350
FALLBACK_WINDOW_OVERLAP = 50

# ---- Retrieval --------------------------------------------------------------

BM25_TOP_K = 25                      # before fusion. See note below.
VECTOR_TOP_K = 25                    # before fusion (mirrors BM25 for symmetry)
RRF_K = 60                           # reciprocal rank fusion constant (standard)
FINAL_TOP_K = 5                      # what we send to the generator
COMPARISON_TOP_K = 8                 # bumped for cross-product comparison queries
RETRY_TOP_K = 10                     # if verifier rejects, retry with broader retrieval

# Why BM25_TOP_K=25 not 20: empirical finding from Phase 2. Short canonical
# defined-term chunks (e.g. §1.2 "Active User" — only ~12 tokens) sit at
# BM25 rank 21-25 because longer chunks that mention the same term win on
# absolute term frequency. With top_k=20 those canonical chunks are dropped
# from the merge entirely, even though vector ranks them #1. Bumping to 25
# lets the merge see both signals. Cost is ~5 extra dict lookups per query.

# Safety net: if vector top-1 cosine inside the filtered set is below this
# threshold, we suspect bad routing and drop the doc filter.
VECTOR_SIMILARITY_FALLBACK = 0.35

# Intent-aware RRF weighting.
# Empirical finding from Phase 2 dev: for "What is an X?" definition queries,
# BM25 floods the top-K with high-TF distractors (long chunks that mention X
# many times) and pushes the actual short canonical "§1.X X is..." definition
# below rank 20. Vector embeddings, in contrast, cluster definition-shaped
# chunks tightly and put the canonical definition in their top-5. So when
# intent=definition we boost VECTOR, not BM25 — the opposite of the original
# plan. BM25 still contributes via the RRF merge (it's good at rare,
# distinctive terms like "FUE") but doesn't dominate.
RRF_WEIGHT_BM25_DEFAULT = 1.0
RRF_WEIGHT_VECTOR_DEFAULT = 1.0
RRF_WEIGHT_VECTOR_DEFINITION = 1.5

# ---- Doc identifiers --------------------------------------------------------
# Stable IDs derived from filenames. Used in chunk_id and as Chroma metadata.
# The PDFs in Data/ map to these IDs in ingest.py.

DOC_TITLES = {
    "rise_s4hana_private": "RISE with SAP S/4HANA Cloud, Private Edition",
    "sap_cloud_erp_private": "SAP Cloud ERP Private, RISE",
    "sap_erp_pce": "SAP ERP, private cloud edition",
}

# Product family mapping — the router decides on (product, intent), and this
# dict translates the product label into a doc-id filter for retrieval.
# Critical: the corpus has 2 products in 3 PDFs. The two RISE docs are
# siblings. Routing on filenames would be wrong; routing on product family
PRODUCT_DOCS = {
    "rise_family": ["rise_s4hana_private", "sap_cloud_erp_private"],
    "sap_erp_pce": ["sap_erp_pce"],
    "all": ["rise_s4hana_private", "sap_cloud_erp_private", "sap_erp_pce"],
}

# ---- Generation / verification ---------------------------------------------

GEN_TEMPERATURE = 0.1                # near-deterministic, but slight room for fluent prose
ROUTE_TEMPERATURE = 0.0              # classification — fully deterministic
VERIFY_TEMPERATURE = 0.0             # also classification

MAX_VERIFIER_RETRIES = 1             # hard cap. No agent loops.

# Context window cap for ALL chat calls. Critical on memory-constrained
# machines (16 GB Mac).
#
# Why this matters: recent Ollama versions (>= 0.3) default `num_ctx` to
# the model's full trained context length. For llama3.1:8b that's 131072
# (128K) tokens. The KV cache reserved for 128K context inflates the
# model server's RSS to ~11 GB; on a 16 GB machine this triggers macOS
# swapping and a single chat call takes 80+ seconds.
#
# Our largest real prompt is ~4K tokens (generator with 5 retrieved
# chunks). 8192 gives 2× headroom and keeps the KV cache under 1 GB.
# Empirically: with NUM_CTX=131072 a trivial chat call takes 82s; with
# NUM_CTX=4096 the same call takes 0.7s warm.
NUM_CTX = 8192

# ---- API --------------------------------------------------------------------

API_TITLE = "Agentic RAG over SAP SDGs"
API_VERSION = "0.1.0"
API_DESCRIPTION = (
    "Question-answering over SAP Service Description Guides with a "
    "(product, intent) router, hybrid BM25+vector retrieval, and a "
    "self-check verifier. All LLMs run locally via Ollama."
)

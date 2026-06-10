"""
Hybrid retrieval over the SDG index.

Two retrievers run in parallel for every query:
    1. BM25 (lexical) — exact-match score over tokenized chunk text.
       Catches defined-term queries like "Active User" or "% of Net Recurring Fee"
       where the wording in the question matches the wording in the SDG verbatim.
    2. Vector (semantic) — cosine similarity over `nomic-embed-text` embeddings
       stored in Chroma. Catches paraphrased queries like "how many requests per
       second can I make?" → finds the "API Call" definition without keyword overlap.

Their result lists are merged via Reciprocal Rank Fusion (RRF), which combines
ranks (not raw scores — they live on different scales).

Intent-aware tuning (set by the router upstream):
    - "definition"      → boost BM25 weight in the RRF merge.
    - "comparison"      → enlarge top_k so the generator sees chunks from
                          multiple docs at once.
    - "specific_clause" → standard hybrid.
    - "general"         → standard hybrid.

Safety net:
    If the doc filter from the router starves retrieval (top-1 cosine below
    VECTOR_SIMILARITY_FALLBACK inside the filtered set), we drop the filter
    and search across all docs. Explicit, code-driven fallback — not LLM-driven.

Loading:
    Build ONE Retriever instance per process. It opens Chroma and unpickles
    BM25 once (~250 ms). After that, every search() call is <50 ms total
    (≈25 ms to embed the query + <10 ms to search both indices + a few ms
    for fusion).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from app import config, llm
from app.ingest import _tokenize          # reuse the exact same tokenizer
from app.schemas import Chunk, QueryIntent


# ---------------------------------------------------------------------------
# Retrieval result types
# ---------------------------------------------------------------------------


@dataclass
class RetrievalDebug:
    """Per-query trace for the /query?debug=true response."""

    bm25_ranks: dict[str, int]            # chunk_id → rank (1-indexed)
    vector_ranks: dict[str, int]
    vector_top1_score: float | None       # cosine similarity of best vector hit
    fallback_triggered: bool              # True if we dropped the doc filter
    final_top_k: int
    bm25_weight: float
    vector_weight: float


# ---------------------------------------------------------------------------
# The Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Hybrid BM25 + vector retriever. Stateful: holds the indexes in memory."""

    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or config.INDEX_DIR
        self._load_bm25()
        self._open_chroma()

    # -- loading ----------------------------------------------------------

    def _load_bm25(self) -> None:
        """Unpickle the BM25 payload built by ingest.py.

        The payload contains:
            bm25         BM25Okapi instance
            tokenized    list[list[str]]    (kept for debug; not strictly needed)
            chunks       list[dict]         original Chunk fields (Pydantic dumps)
        """
        bm25_path = self.index_dir / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {bm25_path}. "
                "Run `python -m app.ingest` first."
            )
        with bm25_path.open("rb") as f:
            payload = pickle.load(f)
        self.bm25: BM25Okapi = payload["bm25"]
        # Reconstruct Chunk objects from the dumped dicts.
        self.chunks: list[Chunk] = [Chunk(**d) for d in payload["chunks"]]
        # id → Chunk for O(1) Chroma → Chunk lookups.
        self.chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        # chunk_id → BM25 corpus index, so we can drop filter-excluded chunks.
        self.bm25_index_by_id: dict[str, int] = {
            c.chunk_id: i for i, c in enumerate(self.chunks)
        }

    def _open_chroma(self) -> None:
        """Open the persistent Chroma collection. No vectors are loaded into
        Python memory — Chroma keeps them in its own SQLite + HNSW files.
        """
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = client.get_collection(config.CHROMA_COLLECTION)

    # -- public API -------------------------------------------------------

    def search(
        self,
        query: str,
        docs: list[str] | None = None,
        k: int | None = None,
        intent: QueryIntent | None = None,
    ) -> tuple[list[Chunk], RetrievalDebug]:
        """Return the top-k chunks for `query`, optionally restricted to `docs`.

        Args:
            query:   The (router-rewritten) user question.
            docs:    Optional list of doc_ids to restrict the search to. If
                     None or contains all ids, no filter is applied.
            k:       How many chunks to return. Defaults to FINAL_TOP_K, or
                     COMPARISON_TOP_K when intent == "comparison".
            intent:  Used to tune RRF weights. None → defaults.

        Returns:
            (ranked_chunks, debug) — the chunks are sorted descending by RRF
            score; the debug dataclass exposes per-retriever ranks and any
            fallback that happened.
        """
        # 1. resolve k and weights from intent
        if k is None:
            k = (
                config.COMPARISON_TOP_K
                if intent == "comparison"
                else config.FINAL_TOP_K
            )
        # See config.RRF_WEIGHT_VECTOR_DEFINITION for why definition queries
        # boost vector, NOT BM25. (Empirical finding: BM25 buries short canonical
        # definitions under long high-TF distractors; vector clusters them well.)
        if intent == "definition":
            w_bm25 = config.RRF_WEIGHT_BM25_DEFAULT
            w_vec = config.RRF_WEIGHT_VECTOR_DEFINITION
        else:
            w_bm25 = config.RRF_WEIGHT_BM25_DEFAULT
            w_vec = config.RRF_WEIGHT_VECTOR_DEFAULT

        # Empty / whitespace-only query: short-circuit. Chroma raises on
        # empty embeddings; BM25 returns nothing useful for zero tokens.
        if not query.strip():
            return [], RetrievalDebug(
                bm25_ranks={}, vector_ranks={}, vector_top1_score=None,
                fallback_triggered=False, final_top_k=k,
                bm25_weight=w_bm25, vector_weight=w_vec,
            )

        # Filter: None or "all 3 docs" both mean no filter.
        all_ids = set(config.PRODUCT_DOCS["all"])
        if docs is None or set(docs) >= all_ids:
            filter_set: set[str] | None = None
        else:
            filter_set = set(docs)

        # 2. run both retrievers
        bm25_ranked = self._bm25_search(query, filter_set, top_k=config.BM25_TOP_K)
        vec_ranked, vec_top1_score = self._vector_search(
            query, filter_set, top_k=config.VECTOR_TOP_K
        )
        fallback_triggered = False

        # 3. safety net: starved filtered retrieval → drop the filter
        if (
            filter_set is not None
            and vec_top1_score is not None
            and vec_top1_score < config.VECTOR_SIMILARITY_FALLBACK
        ):
            fallback_triggered = True
            bm25_ranked = self._bm25_search(query, None, top_k=config.BM25_TOP_K)
            vec_ranked, vec_top1_score = self._vector_search(
                query, None, top_k=config.VECTOR_TOP_K
            )

        # 4. RRF merge
        bm25_rank_map = {cid: i + 1 for i, cid in enumerate(bm25_ranked)}
        vec_rank_map = {cid: i + 1 for i, cid in enumerate(vec_ranked)}
        merged_scores: dict[str, float] = {}
        for cid, rank in bm25_rank_map.items():
            merged_scores[cid] = merged_scores.get(cid, 0.0) + w_bm25 * (
                1.0 / (config.RRF_K + rank)
            )
        for cid, rank in vec_rank_map.items():
            merged_scores[cid] = merged_scores.get(cid, 0.0) + w_vec * (
                1.0 / (config.RRF_K + rank)
            )

        # 5. select top-k by merged score
        top_ids = sorted(merged_scores.keys(), key=lambda c: -merged_scores[c])[:k]
        top_chunks: list[Chunk] = []
        for cid in top_ids:
            chunk = self.chunks_by_id.get(cid)
            if chunk is None:
                continue
            # Build a copy with the score+ranks set (don't mutate the cached chunk).
            top_chunks.append(chunk.model_copy(update={
                "score": merged_scores[cid],
                "rank_bm25": bm25_rank_map.get(cid),
                "rank_vector": vec_rank_map.get(cid),
            }))

        debug = RetrievalDebug(
            bm25_ranks=bm25_rank_map,
            vector_ranks=vec_rank_map,
            vector_top1_score=vec_top1_score,
            fallback_triggered=fallback_triggered,
            final_top_k=k,
            bm25_weight=w_bm25,
            vector_weight=w_vec,
        )
        return top_chunks, debug

    # -- BM25 -------------------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        filter_set: set[str] | None,
        top_k: int,
    ) -> list[str]:
        """Score the entire corpus with BM25, then return the top-k chunk_ids.

        BM25 has no native filter, so we score everything (cheap — ~5 ms for
        ~1500 chunks) and post-filter. If a query has zero overlap with the
        corpus vocabulary, BM25 returns all-zeros — we drop those because they
        carry no information.
        """
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # Pair (score, chunk_id), keep only positive scores, sort desc, filter.
        ranked: list[tuple[float, str]] = []
        for i, s in enumerate(scores):
            if s <= 0.0:
                continue
            cid = self.chunks[i].chunk_id
            if filter_set is not None and self.chunks[i].doc_id not in filter_set:
                continue
            ranked.append((float(s), cid))
        ranked.sort(key=lambda p: -p[0])
        return [cid for _score, cid in ranked[:top_k]]

    # -- Vector (Chroma) --------------------------------------------------

    def _vector_search(
        self,
        query: str,
        filter_set: set[str] | None,
        top_k: int,
    ) -> tuple[list[str], float | None]:
        """Embed the query, search Chroma with the doc-id filter applied
        natively, return (chunk_ids_ranked, top1_cosine_similarity).

        Chroma reports DISTANCES (smaller is better) for `cosine` space.
        We convert distance d → similarity s with s = 1.0 - d so callers
        can reason about the safety-net threshold in similarity space.
        """
        emb = llm.embed(model=config.MODEL_EMBED, text=query)
        where: dict[str, Any] | None = None
        if filter_set is not None:
            doc_id_list = sorted(filter_set)
            where = {"doc_id": {"$in": doc_id_list}}
        result = self.collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            where=where,
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        if not ids:
            return [], None
        top1_similarity = 1.0 - float(distances[0]) if distances else None
        return list(ids), top1_similarity


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------
#
# FastAPI's startup hook will call get_retriever() once. Anything else that
# wants a retriever (eval scripts, the CLI sanity tool) can import this and
# get the same instance — avoids re-loading 10 MB of pickle on every call.

_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER

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
import re
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

        # Filter: None means "search every chunk in the index". A non-None
        # `docs` list means "restrict to exactly these doc_ids". The
        # filter is ALWAYS applied when docs is provided — we used to
        # short-circuit "docs == all SDG docs" to None, but that
        # accidentally allowed meta-corpus chunks (README + module
        # docstrings) to leak into normal SDG queries. The pipeline now
        # explicitly opts INTO meta_about_system when it wants meta
        # answers; everywhere else, a meta chunk must never appear.
        if docs is None:
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

        # 5b. heading-anchor boost. RRF buries short canonical clauses (e.g. a
        #     ~50-token "99.9% SLA Eligibility" header chunk) underneath
        #     longer per-product mentions of the same term. For
        #     definition/specific_clause questions, find any chunk whose
        #     heading region contains a distinctive multi-token phrase from
        #     the query verbatim, and promote it to the top of the result.
        #     Generic — no domain-specific tokens. Triggers ONLY when the
        #     question genuinely names something specific; falls back to
        #     RRF-only otherwise.
        anchored_ids: list[str] = []
        if intent in ("definition", "specific_clause"):
            anchored_ids = self._heading_anchor_matches(
                query, filter_set=filter_set, exclude=set(top_ids),
            )

        # Combine: anchored matches first (highest score), then RRF top-k,
        # de-duplicating. Cap at k.
        boost_score = max(merged_scores.values(), default=0.0) + 1.0
        ordered_ids: list[str] = []
        seen_ids: set[str] = set()
        for cid in anchored_ids + top_ids:
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            ordered_ids.append(cid)
            if len(ordered_ids) >= k:
                break

        top_chunks: list[Chunk] = []
        for cid in ordered_ids:
            chunk = self.chunks_by_id.get(cid)
            if chunk is None:
                continue
            score = merged_scores.get(cid, boost_score) if cid not in anchored_ids \
                else boost_score
            # Build a copy with the score+ranks set (don't mutate the cached chunk).
            top_chunks.append(chunk.model_copy(update={
                "score": score,
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

    # -- Heading-anchor boost ---------------------------------------------

    def _heading_anchor_matches(
        self,
        query: str,
        *,
        filter_set: set[str] | None,
        exclude: set[str],
    ) -> list[str]:
        """Find chunks whose `section_title` (the canonical short heading
        each chunk carries from the structural pass of the chunker)
        contains a distinctive multi-token phrase from `query`. Returns
        the matching chunk_ids in priority order: exact title-equals
        first, then title-contains, ties broken by chunk size (shorter
        first — short clauses tend to be canonical definitions).

        Generic — no domain tokens. A phrase qualifies as "distinctive"
        when it is 2+ consecutive non-stopword tokens; single-token
        phrases are NOT used for the heading-anchor boost (too noisy
        — every product mentions "tenant" or "subscription").

        Why `section_title` instead of "first 80 chars of body": every
        per-product entry begins with "Cloud Service Eligible for: …"
        which would make every eligibility row match for a query about
        "99.9% SLA". The chunker assigns a tighter `section_title`
        (e.g. "99.9% SLA Eligibility" for §2.6), which is what we want
        to anchor on.

        Returns at most 5 chunk_ids — heading-anchor matches are
        meant to be a high-signal handful, not a flood.
        """
        # Use only multi-token phrases for heading-anchoring to avoid
        # over-matching on common single tokens.
        phrases = [p for p in _distinctive_phrases(query) if " " in p]
        if not phrases:
            return []

        title_eq: list[tuple[int, str]] = []   # exact phrase == title
        title_in: list[tuple[int, str]] = []   # phrase appears inside title
        for c in self.chunks:
            if filter_set is not None and c.doc_id not in filter_set:
                continue
            if c.chunk_id in exclude:
                continue
            title = (c.section_title or "").strip().lower()
            if not title:
                continue
            size = len(c.text or "")
            for p in phrases:
                if p == title:
                    title_eq.append((size, c.chunk_id))
                    break
                if p in title:
                    title_in.append((size, c.chunk_id))
                    break
        title_eq.sort()
        title_in.sort()
        ordered = [cid for _s, cid in title_eq] + [cid for _s, cid in title_in]
        # Keep the boost narrow — at most 5 hand-picked anchors.
        return ordered[:5]

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
# Distinctive-phrase extraction (used by heading-anchor boost)
# ---------------------------------------------------------------------------


# Stopwords filtered out before phrase building. Conservative list — we
# want to keep most content words (including SAP-specific tokens) and
# drop only true filler. Apostrophe-form contractions covered by the
# tokenizer's lower-cased output.
_PHRASE_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "by", "to", "for", "with", "from", "as",
    "and", "or", "but", "if", "then", "than", "so",
    "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "do", "does", "did", "can", "could", "should", "would", "may", "might",
    "have", "has", "had", "i", "you", "he", "she", "we", "they",
    "me", "us", "them", "my", "your", "our", "their",
    "not", "no", "yes",
    "any", "all", "some", "each", "every", "more", "most", "other",
    "available", "purchase", "purchased",
}

# Token regex matches sequences of letters/digits, plus internal punctuation
# common in SDG vocab (slash, hyphen, period for "99.9", "%", "S/4HANA").
_PHRASE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'/\-.%]*[A-Za-z0-9%]|[A-Za-z]")


def _distinctive_phrases(query: str) -> list[str]:
    """Build a list of distinctive multi-token phrases from `query` for
    heading-anchor matching. A phrase is two or more consecutive
    non-stopword tokens (lower-cased). A single long non-stopword token
    (length ≥ 5) also qualifies on its own — that catches "Tenant",
    "Active" wouldn't but "ActiveUser"-style merged terms would.

    Empty result is fine: the caller treats it as "no boost".
    """
    raw_tokens = _PHRASE_TOKEN_RE.findall(query)
    # Lower-case but PRESERVE the token; stopword check uses lower-cased form.
    tokens = [t.lower() for t in raw_tokens]
    if not tokens:
        return []

    phrases: list[str] = []
    # 1) consecutive runs of 2+ non-stopword tokens
    run: list[str] = []
    for tok in tokens:
        if tok in _PHRASE_STOPWORDS or len(tok) < 2:
            if len(run) >= 2:
                phrases.append(" ".join(run))
            run = []
        else:
            run.append(tok)
    if len(run) >= 2:
        phrases.append(" ".join(run))

    # 2) single distinctive tokens (length ≥ 5, non-stopword) as one-word phrases
    for tok in tokens:
        if len(tok) >= 5 and tok not in _PHRASE_STOPWORDS:
            phrases.append(tok)

    # De-dup while preserving order; cap to keep matching cheap.
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out[:8]


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------
# FastAPI's startup hook will call get_retriever() once. Anything else that
# wants a retriever (eval scripts, the CLI sanity tool) can import this and
# get the same instance — avoids re-loading 10 MB of pickle on every call.

_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER

"""
Offline ingestion pipeline. Run with:

    python -m app.ingest                   # full rebuild
    python -m app.ingest --skip-summaries  # skip the LLM-generated doc summaries
    python -m app.ingest --quick           # tiny sample for fast iteration

Produces under ./index/:
    chroma/                   Chroma persistent vector store
    bm25.pkl                  pickled BM25Okapi + tokenized corpus + chunk metadata
    doc_summaries.json        per-doc descriptors used by the router prompt

Pipeline:
    1. For each PDF in Data/, derive a doc_id and call chunker.chunk_pdf().
    2. Embed every chunk with nomic-embed-text via Ollama (sequential, batch
       progress reporting).
    3. Write all chunks + embeddings into Chroma (metadata fields exposed for
       filtering: doc_id, doc_title, section_number, page_start, page_end).
    4. Tokenize chunk texts and pickle a BM25Okapi index for the lexical
       retriever. Stored alongside the original chunk metadata so retrieve.py
       only needs to load this one file.
    5. Generate per-doc summaries with llama3.1:8b. Each summary describes
       the doc's product family, page count, and 5-15 representative defined
       terms. The router prompt loads this so it can route on (product,
       intent) rather than filenames. Skip with --skip-summaries when you
       don't want to wait ~30s for the LLM call.

Idempotent: rerun any time. The Chroma collection is reset at the start so
reruns produce a clean state.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from app import config, llm
from app.chunker import chunk_pdf
from app.schemas import Chunk

# ---------------------------------------------------------------------------
# Filename → doc_id mapping
# ---------------------------------------------------------------------------
# Stable: changing this requires re-ingesting because chunk_ids embed doc_id.
PDF_TO_DOC_ID: dict[str, str] = {
    "rise with sap s4hana cloud priva.pdf": "rise_s4hana_private",
    "SAP Cloud ERP Private, RISE.pdf": "sap_cloud_erp_private",
    "SAP ERP, private cloud edition.pdf": "sap_erp_pce",
}


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _embed_texts(texts: list[str], model: str) -> list[list[float]]:
    """Embed a batch of texts via Ollama, sequentially (Ollama lib is sync).

    Reports progress every 50 chunks. Total time on M2 for ~1500 chunks is
    ~60-90s with nomic-embed-text.
    """
    out: list[list[float]] = []
    t0 = time.time()
    for i, text in enumerate(texts):
        if i and i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(texts) - i) / rate if rate > 0 else 0
            print(f"    embedded {i}/{len(texts)} ({rate:.1f}/s, ETA {eta:.0f}s)")
        # Goes through app.llm — provider chosen by EMBED_PROVIDER (default: ollama).
        out.append(llm.embed(model=model, text=text))
    print(f"    embedded {len(texts)}/{len(texts)} in {time.time() - t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Chroma writes
# ---------------------------------------------------------------------------


def _write_chroma(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Reset the Chroma collection and (re)populate it with chunks + embeddings.

    We do NOT use Chroma's built-in embedding function — we hand it our own
    vectors so the embedding model is fully under our control (and works
    offline via Ollama).

    Metadata fields are flattened to scalars because Chroma's `where` filter
    operates on scalar metadata keys.
    """
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # Reset collection idempotently.
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass  # didn't exist — fine
    coll = client.create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma requires non-null metadata values. None → "" for section fields.
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    documents: list[str] = []
    for c in chunks:
        ids.append(c.chunk_id)
        documents.append(c.text)
        metadatas.append({
            "doc_id": c.doc_id,
            "doc_title": c.doc_title,
            "section_number": c.section_number or "",
            "section_title": c.section_title or "",
            "page_start": c.page_start,
            "page_end": c.page_end,
        })
    # Chroma supports batched add in a single call. ~1500 vectors × 768 dim
    # is small enough to ship in one shot.
    coll.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"  [chroma] wrote {len(ids)} chunks to '{config.CHROMA_COLLECTION}' at {config.CHROMA_DIR}")


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/-]*")


def _tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenizer for BM25.

    Keeps tokens with internal slashes and hyphens (e.g., "S/4HANA",
    "tailored-option") and apostrophes ("Customer's"). Standard stopword
    removal is intentionally NOT applied: SDG queries often hinge on small
    function words inside defined terms ('% of Net Recurring Fee').
    """
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _write_bm25(chunks: list[Chunk]) -> None:
    """Build BM25Okapi over chunk texts and persist along with the chunks
    themselves. Loading this one file (~10-20 MB) gives retrieve.py both the
    BM25 index and the chunk metadata in a single read.
    """
    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "tokenized": tokenized,
        "chunks": [c.model_dump() for c in chunks],
    }
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with config.BM25_PATH.open("wb") as f:
        pickle.dump(payload, f)
    print(f"  [bm25] wrote {len(chunks)} chunks + index to {config.BM25_PATH}")


# ---------------------------------------------------------------------------
# Doc summaries (for the router prompt)
# ---------------------------------------------------------------------------


_SUMMARY_PROMPT = """You are summarizing a SAP Service Description Guide \
(SDG) so that a downstream router can decide which doc(s) a user's question \
should retrieve from.

Read the sample text below (first ~3000 characters of the document) and \
return ONLY a JSON object of this shape:

  {{
    "product_family": "rise_family" | "sap_erp_pce" | "other",
    "headline": "<one short sentence describing what this SDG covers>",
    "defined_terms": ["term1", "term2", ...]    // 5-15 of the most distinctive
                                                // defined terms in the doc
  }}

Rules:
  - "rise_family" = anything about RISE with SAP S/4HANA Cloud, private edition.
  - "sap_erp_pce" = SAP ERP, private cloud edition (the tailored option).
  - Defined terms are quoted phrases like "Active User", "API Call", "FUE",
    "Cloud Service" that the SDG explicitly defines. Pick the most
    distinctive 5-15. Don't invent terms.

Document title: {title}

Sample text:
---
{sample}
---

JSON only, no prose."""


def _generate_doc_summary(doc_id: str, doc_title: str, chunks: list[Chunk]) -> dict[str, Any]:
    """Use llama3.1:8b to produce one summary dict for a doc. Falls back to
    a minimal summary if the LLM call fails or returns non-JSON.
    """
    # Take the first ~3000 chars of body content (skip empty chunks).
    sample = " \n".join(c.text for c in chunks[:30])[:3000]
    prompt = _SUMMARY_PROMPT.format(title=doc_title, sample=sample)
    fallback = {
        "product_family": "other",
        "headline": doc_title,
        "defined_terms": [],
    }
    try:
        content = llm.chat(
            model=config.MODEL_GEN,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            temperature=0.0,
            num_ctx=config.NUM_CTX,
        )
        parsed = json.loads(content)
        # Light validation: keep only known keys.
        return {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "product_family": parsed.get("product_family", "other"),
            "headline": parsed.get("headline", doc_title),
            "defined_terms": list(parsed.get("defined_terms", []))[:20],
            "chunk_count": len(chunks),
        }
    except Exception as e:
        print(f"  [summary] WARN: LLM summary failed for {doc_id}: {e}; using fallback")
        return {**fallback, "doc_id": doc_id, "doc_title": doc_title, "chunk_count": len(chunks)}


def _write_doc_summaries(chunks_by_doc: dict[str, list[Chunk]], skip: bool) -> None:
    """Write index/doc_summaries.json. The router prompt loads this so it
    knows what each doc covers.
    """
    summaries: list[dict[str, Any]] = []
    for doc_id, chunks in chunks_by_doc.items():
        if skip:
            summaries.append({
                "doc_id": doc_id,
                "doc_title": chunks[0].doc_title if chunks else config.DOC_TITLES.get(doc_id, doc_id),
                "product_family": "other",
                "headline": "(summary skipped)",
                "defined_terms": [],
                "chunk_count": len(chunks),
            })
        else:
            print(f"  [summary] generating for {doc_id} (calls llama3.1:8b once)...")
            summaries.append(_generate_doc_summary(
                doc_id=doc_id,
                doc_title=chunks[0].doc_title,
                chunks=chunks,
            ))
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    config.DOC_SUMMARIES_PATH.write_text(json.dumps(summaries, indent=2))
    print(f"  [summary] wrote {len(summaries)} doc summaries to {config.DOC_SUMMARIES_PATH}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(quick: bool = False, skip_summaries: bool = False) -> int:
    t_start = time.time()
    pdf_paths = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"FATAL: no PDFs in {config.DATA_DIR}", file=sys.stderr)
        return 1

    # Step 1: chunk every PDF
    print(f"\n[1/4] Chunking {len(pdf_paths)} PDFs...")
    chunks_by_doc: dict[str, list[Chunk]] = {}
    for pdf in pdf_paths:
        doc_id = PDF_TO_DOC_ID.get(pdf.name)
        if not doc_id:
            print(f"  [chunker] WARN: skipping unknown PDF {pdf.name}")
            continue
        doc_title = config.DOC_TITLES.get(doc_id, doc_id)
        chunks = chunk_pdf(pdf, doc_id=doc_id, doc_title=doc_title)
        if quick:
            chunks = chunks[:30]
            print(f"  [chunker] --quick: truncated to {len(chunks)} chunks for {doc_id}")
        chunks_by_doc[doc_id] = chunks

    # Step 1b: chunk the meta-corpus (README + module docstrings) so meta
    # questions like "what does this app do?" retrieve from real source
    # text, not a hardcoded paragraph. See app/meta_chunker.py.
    from app.meta_chunker import META_DOC_ID, build_meta_chunks
    meta_chunks = build_meta_chunks()
    if meta_chunks:
        if quick:
            meta_chunks = meta_chunks[:10]
        chunks_by_doc[META_DOC_ID] = meta_chunks
        print(f"  [meta_chunker] {len(meta_chunks)} chunks from README + module docstrings")

    all_chunks: list[Chunk] = []
    for cs in chunks_by_doc.values():
        all_chunks.extend(cs)
    print(f"  total: {len(all_chunks)} chunks across {len(chunks_by_doc)} docs")

    # Step 2: embed every chunk
    print(f"\n[2/4] Embedding {len(all_chunks)} chunks with {config.MODEL_EMBED}...")
    embeddings = _embed_texts([c.text for c in all_chunks], model=config.MODEL_EMBED)
    if len(embeddings) != len(all_chunks):
        print("FATAL: embedding count mismatch", file=sys.stderr)
        return 1

    # Step 3: write Chroma + BM25
    print("\n[3/4] Writing indexes...")
    _write_chroma(all_chunks, embeddings)
    _write_bm25(all_chunks)

    # Step 4: doc summaries
    print("\n[4/4] Doc summaries...")
    _write_doc_summaries(chunks_by_doc, skip=skip_summaries)

    print(f"\nDone in {time.time() - t_start:.1f}s. Index at {config.INDEX_DIR}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the SDG index.")
    parser.add_argument("--quick", action="store_true",
                        help="Truncate to 30 chunks per doc for fast iteration.")
    parser.add_argument("--skip-summaries", action="store_true",
                        help="Skip the LLM-generated doc summaries.")
    args = parser.parse_args()
    return run(quick=args.quick, skip_summaries=args.skip_summaries)


if __name__ == "__main__":
    sys.exit(main())

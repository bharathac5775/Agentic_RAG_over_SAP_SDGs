"""Interactive retrieval sanity check. Run with:

    python -m app.retrieve_demo "What is an Active User?"
    python -m app.retrieve_demo "What is an Active User?" --intent definition
    python -m app.retrieve_demo "PCE vs RISE differences" --intent comparison --docs all
    python -m app.retrieve_demo --canned   # runs 5 representative queries

Prints the top-K chunks plus per-retriever ranks and the RRF debug payload.
This is the "open the box" demo for the interview — it shows exactly which
retriever surfaced each chunk and why.
"""

from __future__ import annotations

import argparse
import sys
import time

from app import config
from app.retrieve import get_retriever
from app.schemas import QueryIntent

CANNED_QUERIES: list[tuple[str, QueryIntent | None, list[str] | None]] = [
    ("What is an Active User?",                                     "definition",      None),
    ('What does "% of Net Recurring Fee" mean?',                    "definition",      None),
    ("How is data residency handled in RISE S/4HANA private?",      "specific_clause", ["rise_s4hana_private", "sap_cloud_erp_private"]),
    ("What's the difference between SAP ERP PCE and RISE?",         "comparison",      None),
    ("Can I cancel my subscription early?",                         "general",         None),
]


def _print_result(query: str, intent: QueryIntent | None, docs: list[str] | None) -> None:
    retr = get_retriever()
    t0 = time.time()
    chunks, dbg = retr.search(query, docs=docs, intent=intent)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"\n{'=' * 78}")
    print(f"Query:    {query!r}")
    print(f"Intent:   {intent}, Docs filter: {docs}")
    print(f"Latency:  {elapsed_ms:.0f} ms")
    print(f"Weights:  bm25={dbg.bm25_weight}, vector={dbg.vector_weight}")
    print(f"Vector top-1 cosine: {dbg.vector_top1_score:.3f}" if dbg.vector_top1_score is not None else "Vector top-1 cosine: n/a")
    print(f"Filter fallback triggered: {dbg.fallback_triggered}")
    print(f"{'=' * 78}")

    for i, c in enumerate(chunks, 1):
        bm = c.rank_bm25 or "—"
        vc = c.rank_vector or "—"
        print(f"\n#{i:>2}  [{c.chunk_id}]")
        print(f"     §{c.section_number} {c.section_title!r}, pp.{c.page_start}-{c.page_end}, doc={c.doc_id}")
        print(f"     RRF score={c.score:.4f}  bm25_rank={bm}  vec_rank={vc}")
        snippet = c.text.replace("\n", " ").strip()
        print(f"     {snippet[:180]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=None,
                        help="The question to retrieve for. Omit if using --canned.")
    parser.add_argument("--intent", choices=["definition", "specific_clause", "comparison", "general"],
                        default=None)
    parser.add_argument("--docs", nargs="+", default=None,
                        help='Doc-ids to filter to. Use "all" to disable filtering.')
    parser.add_argument("--canned", action="store_true",
                        help="Run 5 representative canned queries and exit.")
    args = parser.parse_args()

    if args.canned:
        for q, intent, docs in CANNED_QUERIES:
            _print_result(q, intent, docs)
        return 0

    if not args.query:
        parser.print_help()
        return 1

    docs = args.docs
    if docs == ["all"]:
        docs = config.PRODUCT_DOCS["all"]
    _print_result(args.query, args.intent, docs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

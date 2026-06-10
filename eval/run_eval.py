"""Run the eval harness and print a scorecard.

Usage:
    python -m eval.run_eval                  # all 15 questions
    python -m eval.run_eval --quick          # first 5 only (fast iteration)
    python -m eval.run_eval --json out.json  # also dump full results to JSON

What this measures (automatic, per row):
    - routing_correct   : router's `products` matched expected
    - intent_correct    : router's `intent` matched expected
    - guardrail_correct : refused-when-expected, NOT-refused-when-expected
    - has_citation      : non-refused answers have ≥1 citation
    - substring_present : answer contains the expected key phrase
    - verified          : verifier passed
    - retried           : verifier retry path activated
    - latency_ms        : end-to-end

Aggregate scorecard at the end shows headline numbers. Per-row table shows
each question's pass/fail breakdown so you can read a single row and know
exactly what failed.

Hand-grading note: this harness does NOT auto-grade answer quality
(strict equality on natural-language output is meaningless). The brief
explicitly asked for hand-graded examples — review the dumped per-row
results and assign a manual P/F/Partial column. The README's "What
works / what struggles" section is filled in by reading those rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from app import config, pipeline
from app.schemas import QueryResponse


# ---------------------------------------------------------------------------
# Per-row result type
# ---------------------------------------------------------------------------


@dataclass
class EvalRow:
    id: int
    question: str
    category: str

    # Expected
    expected_products: list[str] | None
    expected_intent: str | None
    expected_refused: bool
    expected_refusal_category: str | None
    expected_substring: str | None
    should_have_citation: bool | None

    # Observed
    actual_products: list[str] | None = None
    actual_intent: str | None = None
    actual_refused: bool = False
    actual_refusal_category: str | None = None
    actual_answer: str = ""
    actual_citation_count: int = 0
    actual_verified: bool = False
    actual_retried: bool = False
    latency_ms: int = 0

    # Auto-graded checks (None = not applicable for this row)
    routing_correct: bool | None = None
    intent_correct: bool | None = None
    guardrail_correct: bool | None = None
    has_citation: bool | None = None
    substring_present: bool | None = None

    notes: str = ""

    # Raw payload preserved for hand-grading
    raw_response: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _grade(row: EvalRow, resp: QueryResponse) -> EvalRow:
    """Mutate `row` with observed values + boolean grades. Returns the row."""
    row.actual_refused = resp.refused
    row.actual_refusal_category = (
        resp.trace.get("guardrail", {}).get("category") if resp.trace else None
    )
    row.actual_answer = resp.answer
    row.actual_citation_count = len(resp.citations)
    row.actual_verified = resp.verified

    if resp.trace:
        row.actual_products = resp.trace.get("route", {}).get("products")
        row.actual_intent = resp.trace.get("route", {}).get("intent")
        row.latency_ms = resp.trace.get("latency_ms", {}).get("total", 0)
        row.actual_retried = "retry" in resp.trace
        row.raw_response = resp.trace

    # ---- Guardrail correctness ----
    # PASS if refused-status matches expectation, AND if expected to refuse,
    # the category also matches.
    if row.expected_refused:
        row.guardrail_correct = (
            row.actual_refused
            and (
                row.expected_refusal_category is None
                or row.actual_refusal_category == row.expected_refusal_category
            )
        )
    else:
        # Was NOT expected to be refused — pass iff actually not refused.
        row.guardrail_correct = not row.actual_refused

    # ---- Routing / intent (only meaningful for non-refused, non-guardrail rows) ----
    if not row.expected_refused and row.expected_products is not None:
        row.routing_correct = (
            row.actual_products is not None
            and sorted(row.actual_products) == sorted(row.expected_products)
        )
    if not row.expected_refused and row.expected_intent is not None:
        row.intent_correct = row.actual_intent == row.expected_intent

    # ---- Citation presence ----
    if row.should_have_citation is True:
        row.has_citation = row.actual_citation_count > 0
    elif row.should_have_citation is False:
        row.has_citation = row.actual_citation_count == 0
    # None → not graded

    # ---- Substring check ----
    if row.expected_substring:
        row.substring_present = (
            row.expected_substring.lower() in row.actual_answer.lower()
        )

    return row


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _mark(b: bool | None) -> str:
    if b is None:
        return f"{_DIM}—{_RESET}"
    return f"{_GREEN}✓{_RESET}" if b else f"{_RED}✗{_RESET}"


def _print_per_row(rows: list[EvalRow]) -> None:
    """Compact table: one line per row, columns for each grade."""
    print()
    print(f"{_BOLD}Per-row results{_RESET}")
    header = (
        f"{'ID':<3} {'Category':<22} {'Guard':<6} {'Route':<6} "
        f"{'Intent':<7} {'Cite':<5} {'Sub':<4} {'Verif':<6} "
        f"{'Retry':<6} {'Lat(ms)':<8}  Question"
    )
    print(_BOLD + header + _RESET)
    print("─" * len(header))
    for r in rows:
        print(
            f"{r.id:<3} {r.category:<22} "
            f"{_mark(r.guardrail_correct):<6} "
            f"{_mark(r.routing_correct):<6} "
            f"{_mark(r.intent_correct):<7} "
            f"{_mark(r.has_citation):<5} "
            f"{_mark(r.substring_present):<4} "
            f"{_mark(r.actual_verified):<6} "
            f"{('R' if r.actual_retried else '-'):<6} "
            f"{r.latency_ms:<8} "
            f"{r.question[:60]!r}"
        )


def _print_aggregates(rows: list[EvalRow]) -> None:
    """Headline scorecard."""

    def _rate(values: list[bool | None]) -> tuple[int, int]:
        graded = [v for v in values if v is not None]
        return sum(1 for v in graded if v), len(graded)

    g_pass, g_total = _rate([r.guardrail_correct for r in rows])
    r_pass, r_total = _rate([r.routing_correct for r in rows])
    i_pass, i_total = _rate([r.intent_correct for r in rows])
    c_pass, c_total = _rate([r.has_citation for r in rows])
    s_pass, s_total = _rate([r.substring_present for r in rows])

    nonrefused = [r for r in rows if not r.actual_refused]
    v_pass = sum(1 for r in nonrefused if r.actual_verified)
    v_total = len(nonrefused)
    retried = sum(1 for r in rows if r.actual_retried)

    latencies = [r.latency_ms for r in rows if r.latency_ms > 0]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0

    print()
    print(f"{_BOLD}Aggregate scorecard{_RESET}")
    print(f"  guardrail correct       : {g_pass}/{g_total}")
    print(f"  routing correct         : {r_pass}/{r_total}")
    print(f"  intent correct          : {i_pass}/{i_total}")
    print(f"  citation present        : {c_pass}/{c_total}")
    print(f"  expected substring      : {s_pass}/{s_total}")
    print(f"  verifier passed         : {v_pass}/{v_total}  (of non-refused)")
    print(f"  retry rate              : {retried}/{len(rows)}")
    print(f"  avg latency (ms)        : {avg_lat:.0f}")
    print(f"  p95 latency (ms)        : {p95_lat}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_eval_set(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def run(quick: bool = False, json_out: Path | None = None) -> int:
    eval_path = Path(__file__).resolve().parent / "eval_set.json"
    raw_rows = _load_eval_set(eval_path)
    if quick:
        raw_rows = raw_rows[:5]

    print(f"\n{_BOLD}Running {len(raw_rows)} eval rows against the live pipeline...{_RESET}")
    print(f"(models: gen={config.MODEL_GEN}, small={config.MODEL_SMALL}, "
          f"embed={config.MODEL_EMBED})")

    rows: list[EvalRow] = []
    t_start = time.time()
    for raw in raw_rows:
        row = EvalRow(
            id=raw["id"],
            question=raw["question"],
            category=raw["category"],
            expected_products=raw.get("expected_products"),
            expected_intent=raw.get("expected_intent"),
            expected_refused=raw.get("expected_refused", False),
            expected_refusal_category=raw.get("expected_refusal_category"),
            expected_substring=raw.get("expected_substring"),
            should_have_citation=raw.get("should_have_citation"),
            notes=raw.get("notes", ""),
        )
        sys.stdout.write(f"  [{row.id:>2}] {row.question[:70]!r}... ")
        sys.stdout.flush()
        try:
            resp = pipeline.answer(row.question, debug=True)
            row = _grade(row, resp)
            verdict = (
                "REFUSED" if row.actual_refused else
                ("✓" if (row.routing_correct is None or row.routing_correct) else "X")
            )
            print(f"{verdict} ({row.latency_ms} ms)")
        except Exception as e:
            print(f"{_RED}ERROR: {e}{_RESET}")
            row.notes = f"runtime error: {e}"
        rows.append(row)

    print(f"\n{_BOLD}Total wall time: {time.time() - t_start:.1f}s{_RESET}")

    _print_per_row(rows)
    _print_aggregates(rows)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps([asdict(r) for r in rows], indent=2, default=str))
        print(f"\nFull results written to {json_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SDG eval harness.")
    parser.add_argument("--quick", action="store_true", help="Run only the first 5 rows.")
    parser.add_argument("--json", type=Path, default=None, help="Path to dump full results.")
    args = parser.parse_args()
    return run(quick=args.quick, json_out=args.json)


if __name__ == "__main__":
    sys.exit(main())

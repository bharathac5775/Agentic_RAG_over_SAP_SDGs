# Eval Scorecard

This is the honest output of `python -m eval.run_eval` on the 15-question hand-graded set in `eval/eval_set.json`. **Do not edit by hand** — re-run the eval to refresh.

## Aggregate scorecard

| Metric | Pass / Total |
|---|---|
| Guardrail decisions | **15 / 15** |
| Routing accuracy (`products`) | **11 / 11** |
| Intent classification | **9 / 11** |
| Citation present (when expected) | **11 / 12** |
| Expected substring in answer | **3 / 4** |
| Verifier passed (of non-refused) | **10 / 11** |
| Retry rate | 1 / 15 |
| Average latency | **14.8 s** |
| p95 latency | 35.8 s |
| Total wall time (15 queries) | 162.8 s |

## What works

- **Defined-term retrieval** (Q1 Active User, Q3 FUE, Q8 % of Net Recurring Fee): all returned canonical definitions with verbatim citations.
- **Guardrails** (Q10 pricing, Q13 code request, Q14 prompt injection, Q15 weather): 4/4 correctly refused at Stage 1 with **0 ms latency** — no LLM calls.
- **Honest refusals** on out-of-corpus questions (Q4 data residency, Q6 termination, Q7 cancellation): system correctly says *"The provided SDGs do not specify this."* instead of fabricating.
- **Routing** is perfect on 11/11 product-routing decisions.

## What struggles

### Q2 — defined-term confusion
Question: *"What is an API Call?"* → answered with the definition of "Entitlements Package" instead. Retrieval surfaced a chunk about Entitlements before the §1.3 API Call chunk; the generator paraphrased the wrong source. **Real correctness bug.** Fix: cross-encoder reranker, or stricter defined-term boost in retrieval.

### Q5 — cross-product comparison
Question: *"What's the difference between SAP ERP PCE and RISE?"* → answer is RISE-skewed because the larger doc dominates retrieval (133 vs 44 pages). Verifier correctly flagged ungrounded; one retry triggered; final answer returned with `verified=false`. **System works as designed — verifier earns its keep.** Fix: balanced retrieval (top-K per product family) for `intent=comparison`.

### Intent boundary cases (Q11, Q12)
The router classified *"security certifications"* as `definition` (expected `specific_clause`) and *"if SAP discontinues a service"* as `general` (expected `specific_clause`). Both are defensible — the questions sit at category boundaries. Not a correctness issue; the answers themselves were reasonable.

## Honest verdict

**13 of 15 questions produce correct, grounded, cited answers.** The 2 that don't:
- Q2 reveals a real retrieval-precision weakness (a cross-encoder reranker would likely fix it).
- Q5 reveals the corpus imbalance for cross-product comparisons (verifier catches it; partial answer ships with `verified: false` warning).

**The verifier's existence is justified.** It caught Q5 cleanly. Without it, the comparison answer would have shipped as if trustworthy.

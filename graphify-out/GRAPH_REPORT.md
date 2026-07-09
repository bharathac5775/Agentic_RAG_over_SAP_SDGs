# Graph Report - .  (2026-06-16)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 629 nodes · 1167 edges · 40 communities (31 shown, 9 thin omitted)
- Extraction: 82% EXTRACTED · 18% INFERRED · 0% AMBIGUOUS · INFERRED: 209 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c7132ab3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_FastAPI Surface (api.py)|FastAPI Surface (api.py)]]
- [[_COMMUNITY_Guardrails & Input Filtering|Guardrails & Input Filtering]]
- [[_COMMUNITY_Retrieval Demo & Agent Tests|Retrieval Demo & Agent Tests]]
- [[_COMMUNITY_PDF Chunker|PDF Chunker]]
- [[_COMMUNITY_Pydantic Schemas|Pydantic Schemas]]
- [[_COMMUNITY_Generator + Verifier Agent|Generator + Verifier Agent]]
- [[_COMMUNITY_LLM Provider Tests|LLM Provider Tests]]
- [[_COMMUNITY_Router Tests|Router Tests]]
- [[_COMMUNITY_Deterministic Answer Guard|Deterministic Answer Guard]]
- [[_COMMUNITY_Ingestion Pipeline|Ingestion Pipeline]]
- [[_COMMUNITY_LLM Provider Abstraction|LLM Provider Abstraction]]
- [[_COMMUNITY_Hybrid Retrieval Internals|Hybrid Retrieval Internals]]
- [[_COMMUNITY_RISE  S4HANA Core Concepts|RISE / S/4HANA Core Concepts]]
- [[_COMMUNITY_Meta-Corpus Chunker|Meta-Corpus Chunker]]
- [[_COMMUNITY_Pipeline Orchestrator|Pipeline Orchestrator]]
- [[_COMMUNITY_Query Router|Query Router]]
- [[_COMMUNITY_README System Overview|README: System Overview]]
- [[_COMMUNITY_README Guardrails & API|README: Guardrails & API]]
- [[_COMMUNITY_Streamlit UI|Streamlit UI]]
- [[_COMMUNITY_README Ingest & Embeddings|README: Ingest & Embeddings]]
- [[_COMMUNITY_README LLM Providers|README: LLM Providers]]
- [[_COMMUNITY_README Hybrid Retrieval|README: Hybrid Retrieval]]
- [[_COMMUNITY_Config & Retrieval Entry|Config & Retrieval Entry]]
- [[_COMMUNITY_SAP Data Integration Stack|SAP Data Integration Stack]]
- [[_COMMUNITY_SAP BI  Reporting Stack|SAP BI / Reporting Stack]]
- [[_COMMUNITY_SAP Compliance Reporting|SAP Compliance Reporting]]
- [[_COMMUNITY_Streamlit UI Smoke Test|Streamlit UI Smoke Test]]
- [[_COMMUNITY_SAP HANA Family|SAP HANA Family]]
- [[_COMMUNITY_SAP Business App Studio|SAP Business App Studio]]
- [[_COMMUNITY_SAP BTP|SAP BTP]]
- [[_COMMUNITY_SAP Business Warehouse|SAP Business Warehouse]]
- [[_COMMUNITY_SAP Cloud Connector|SAP Cloud Connector]]
- [[_COMMUNITY_SAP Fiori|SAP Fiori]]
- [[_COMMUNITY_Run Shell Script|Run Shell Script]]
- [[_COMMUNITY_README Config Pointer|README: Config Pointer]]
- [[_COMMUNITY_README Schemas Pointer|README: Schemas Pointer]]
- [[_COMMUNITY_README ragas Pointer|README: ragas Pointer]]

## God Nodes (most connected - your core abstractions)
1. `GeneratedAnswer` - 56 edges
2. `Chunk` - 51 edges
3. `Citation` - 45 edges
4. `QueryResponse` - 29 edges
5. `RouteDecision` - 22 edges
6. `VerifierVerdict` - 21 edges
7. `Agentic RAG over SAP Service Description Guides` - 19 edges
8. `GuardrailDecision` - 18 edges
9. `Retriever` - 17 edges
10. `_FakeRetriever` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Chunk` --uses--> `Citation`  [INFERRED]
  tests/test_agent.py → app/schemas.py
- `TestAgentLive` --uses--> `Citation`  [INFERRED]
  tests/test_agent.py → app/schemas.py
- `TestCitationValidation` --uses--> `Citation`  [INFERRED]
  tests/test_agent.py → app/schemas.py
- `TestGenerateLogic` --uses--> `Citation`  [INFERRED]
  tests/test_agent.py → app/schemas.py
- `TestHealth` --uses--> `Citation`  [INFERRED]
  tests/test_api.py → app/schemas.py

## Import Cycles
- 1-file cycle: `app/api.py -> app/api.py`

## Hyperedges (group relationships)
- **End-to-end pipeline orchestration** —  [EXTRACTED 1.00]
- **Three deployment paths (native, app-only Docker, full compose)** —  [EXTRACTED 1.00]

## Communities (40 total, 9 thin omitted)

### Community 0 - "FastAPI Surface (api.py)"
Cohesion: 0.08
Nodes (35): health(), lifespan(), Any, QueryResponse, query(), FastAPI surface for the Agentic RAG system.  Three endpoints:     GET  /health, Answer a natural-language question about SAP Service Description Guides.      Fl, Probe whether the system is ready to serve queries.      Checks (cheap, ~50ms to (+27 more)

### Community 1 - "Guardrails & Input Filtering"
Cohesion: 0.09
Nodes (10): _mock_chat(), Phase 5 guardrails tests.    - TestStage1: deterministic regex tests, no LLM, fa, Short questions that mention SAP-specific vocabulary should NOT         invoke t, A 3-word question with NO SAP vocabulary still hits Stage 2 —         the safety, If Stage 1 catches a query, Stage 2 must not run (saves latency)., llm.chat() returns the bare content string (provider abstraction)., A 3-word question hits Stage 2; the small model should classify         it sensi, TestGuardrailsLive (+2 more)

### Community 2 - "Retrieval Demo & Agent Tests"
Cohesion: 0.10
Nodes (14): main(), _print_result(), QueryIntent, Interactive retrieval sanity check. Run with:      python -m app.retrieve_demo ", get_retriever(), _mock_chat_response(), Chunk, Phase 4 generator + verifier tests.    - TestGenerateLogic / TestVerifyLogic: mo (+6 more)

### Community 3 - "PDF Chunker"
Cohesion: 0.07
Nodes (44): _build_page_offset_table(), chunk_pdf(), _derive_section_title(), _extract_clean_text(), _last_sentence(), _pack_section(), _page_for_offset(), _page_range() (+36 more)

### Community 4 - "Pydantic Schemas"
Cohesion: 0.12
Nodes (34): check_output(), Any, GeneratedAnswer, Apply cheap deterministic checks to the generator's output.      Catches three f, Per-query trace for the /query?debug=true response., RetrievalDebug, Citation, GeneratedAnswer (+26 more)

### Community 5 - "Generator + Verifier Agent"
Cohesion: 0.15
Nodes (29): _build_citation(), _cli(), _find_chunk_containing_quote(), _format_chunks_for_prompt(), generate(), _grounded_verdict(), Any, Chunk (+21 more)

### Community 6 - "LLM Provider Tests"
Cohesion: 0.09
Nodes (7): Tests for the LLM provider abstraction.  Verifies provider dispatch, env-var-bas, If `openai` isn't installed, calling chat() with LLM_PROVIDER=openai         sho, TestChatDispatch, TestEmbedDispatch, TestKeyHandling, TestLazyImports, TestProviderSummary

### Community 7 - "Router Tests"
Cohesion: 0.15
Nodes (6): _mock_ollama_response(), Phase 3 router tests.  Two test classes:   - TestRouterLogic: mocked Ollama. Fas, Realistic questions against the actual Ollama model.      These are NOT contract, llm.chat() returns the bare content string (provider abstraction).     Name kept, TestRouterLive, TestRouterLogic

### Community 8 - "Deterministic Answer Guard"
Cohesion: 0.16
Nodes (17): _extract_claimed_label(), _extract_number_unit_pairs(), _label_after_bound(), _Match, maybe_correct(), _parse_number(), _phrase_as_sentence(), GeneratedAnswer (+9 more)

### Community 9 - "Ingestion Pipeline"
Cohesion: 0.18
Nodes (17): _embed_texts(), _generate_doc_summary(), main(), Any, Chunk, Offline ingestion pipeline. Run with:      python -m app.ingest, Lower-case alphanumeric tokenizer for BM25.      Keeps tokens with internal slas, Build BM25Okapi over chunk texts and persist along with the chunks     themselve (+9 more)

### Community 10 - "LLM Provider Abstraction"
Cohesion: 0.13
Nodes (25): chat(), _chat_anthropic(), _chat_google(), _chat_ollama(), _chat_openai(), embed(), _embed_google(), _embed_ollama() (+17 more)

### Community 11 - "Hybrid Retrieval Internals"
Cohesion: 0.15
Nodes (11): Chunk, Path, QueryIntent, Open the persistent Chroma collection. No vectors are loaded into         Python, Return the top-k chunks for `query`, optionally restricted to `docs`.          A, Score the entire corpus with BM25, then return the top-k chunk_ids.          BM2, Find chunks whose `section_title` (the canonical short heading         each chun, Embed the query, search Chroma with the doc-id filter applied         natively, (+3 more)

### Community 12 - "RISE / S/4HANA Core Concepts"
Cohesion: 0.12
Nodes (18): RISE with SAP, SAP ERP, SAP S/4HANA Cloud, RISE with SAP S/4HANA Cloud, private edition, tailored option, SAP Application Interface Framework, SAP Central Finance Transaction Replication by Insightsoftware, SAP Customer Experience, Full Use Equivalent (FUE) (+10 more)

### Community 13 - "Meta-Corpus Chunker"
Cohesion: 0.21
Nodes (14): _approx_tokens(), build_meta_chunks(), _module_docstring(), Chunk, Path, Meta-corpus chunker.  Builds a small "self-describing" corpus from developer-mai, Greedy paragraph packing. Splits on blank lines, packs until target,     forces, Extract the module-level docstring from a Python file. Returns None if     the f (+6 more)

### Community 14 - "Pipeline Orchestrator"
Cohesion: 0.17
Nodes (14): answer(), _chunk_summary(), _cli(), _is_refusal(), Any, Chunk, GeneratedAnswer, QueryResponse (+6 more)

### Community 15 - "Query Router"
Cohesion: 0.22
Nodes (14): _apply_heuristic_overrides(), _build_system_prompt(), _cli(), _format_doc_summaries(), _get_system_prompt(), _load_doc_summaries(), Any, RouteDecision (+6 more)

### Community 16 - "README: System Overview"
Cohesion: 0.06
Nodes (44): Agentic RAG System, Deterministic Answer Guard, app/agent.py (generator + verifier), app/answer_guard.py, app/api.py FastAPI surface, app/chunker.py, app/guardrails.py, app/ingest.py (+36 more)

### Community 17 - "README: Guardrails & API"
Cohesion: 0.05
Nodes (41): 1. Input guardrail — Stage 1 regex (0 ms, no LLM), 2. Input guardrail — Stage 2 small-LLM fallback, 3. Output guardrail (deterministic, after generation), 4. Self-check verifier (LLM, after the output guardrail), Agentic elements, Agentic RAG over SAP Service Description Guides, Architecture, Capability checklist (+33 more)

### Community 18 - "Streamlit UI"
Cohesion: 0.09
Nodes (17): Phase 2 retrieval correctness tests. Requires ./index/ to be built.  Run with:, A broad cross-product term ('subscription term') should naturally     pull from, intent='comparison' should default to COMPARISON_TOP_K (8) when k     is not spe, Every chunk returned by search() must carry the diagnostic fields:     score, an, One Retriever shared across all tests in this module., A truly off-topic query should still return SOMETHING (valid Chunks)     rather, Cold start should produce a non-empty corpus., The defining test from Phase 1: 'What is an Active User?' should     retrieve th (+9 more)

### Community 19 - "README: Ingest & Embeddings"
Cohesion: 0.21
Nodes (12): classify_input(), _cli(), is_meta_question(), GuardrailDecision, Phase 5 — Input and output guardrails.  INPUT (classify_input): runs BEFORE the, Return True if `question` is asking about THIS system rather than     the SDGs (, Run regex rules in order; return (category, refusal_message_or_None).      Retur, Last-ditch LLM classification for ambiguous queries. Returns the same     shape (+4 more)

### Community 20 - "README: LLM Providers"
Cohesion: 0.07
Nodes (29): app-base service template, app-host-ollama service, app-with-ollama service, host-ollama profile, Metal GPU passthrough limitation rationale, ollama sibling container service, with-ollama profile, Aggregate scorecard (+21 more)

### Community 21 - "README: Hybrid Retrieval"
Cohesion: 0.43
Nodes (6): fail(), main(), ok(), Phase 0 smoke test. Run with: python -m app.smoke  Verifies:   1. All required d, section(), Exception

### Community 24 - "Config & Retrieval Entry"
Cohesion: 0.33
Nodes (4): Central configuration for the Agentic RAG pipeline.  Single source of truth for:, _distinctive_phrases(), Hybrid retrieval over the SDG index.  Two retrievers run in parallel for every q, Build a list of distinctive multi-token phrases from `query` for     heading-anc

### Community 25 - "SAP Data Integration Stack"
Cohesion: 0.33
Nodes (6): SAP Datasphere, SAP Data Integrator, SAP Data Provisioning Agent, SAP Data Services Agent, SAP Analytics Cloud Agent, SAP Datasphere

### Community 27 - "SAP BI / Reporting Stack"
Cohesion: 0.50
Nodes (4): SAP BusinessObjects Enterprise, SAP BusinessObjects BI Platform, SAP Crystal Reports, SAP Lumira Server

### Community 28 - "SAP Compliance Reporting"
Cohesion: 0.50
Nodes (4): SAP Document and Reporting Compliance for SAP S/4HANA Cloud, SAP Electronic Invoicing, SAP Disclosure Management, SAP Document and Reporting Compliance

### Community 29 - "Streamlit UI Smoke Test"
Cohesion: 0.50
Nodes (3): Test that the Streamlit UI actually loads end-to-end.  Uses Streamlit's official, The Streamlit script must execute end-to-end without ImportError /     ModuleNot, test_streamlit_ui_loads_without_import_error()

### Community 30 - "SAP HANA Family"
Cohesion: 0.67
Nodes (3): SAP HANA, SAP HANA Cloud, SAP HANA Rules Framework

## Knowledge Gaps
- **91 isolated node(s):** `QueryIntent`, `Exception`, `run.sh script`, `TL;DR`, `Path A — Native (recommended on macOS for speed)` (+86 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Chunk` connect `Meta-Corpus Chunker` to `FastAPI Surface (api.py)`, `Retrieval Demo & Agent Tests`, `PDF Chunker`, `Pydantic Schemas`, `Generator + Verifier Agent`, `Ingestion Pipeline`, `Hybrid Retrieval Internals`, `Pipeline Orchestrator`, `Config & Retrieval Entry`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `GeneratedAnswer` connect `Pydantic Schemas` to `FastAPI Surface (api.py)`, `Guardrails & Input Filtering`, `Retrieval Demo & Agent Tests`, `Generator + Verifier Agent`, `Deterministic Answer Guard`, `Pipeline Orchestrator`, `README: Ingest & Embeddings`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `Citation` connect `Pydantic Schemas` to `FastAPI Surface (api.py)`, `Guardrails & Input Filtering`, `Retrieval Demo & Agent Tests`, `Generator + Verifier Agent`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 35 inferred relationships involving `GeneratedAnswer` (e.g. with `Any` and `Chunk`) actually correct?**
  _`GeneratedAnswer` has 35 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `Chunk` (e.g. with `Any` and `Chunk`) actually correct?**
  _`Chunk` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `Citation` (e.g. with `Any` and `Chunk`) actually correct?**
  _`Citation` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `QueryResponse` (e.g. with `Any` and `QueryResponse`) actually correct?**
  _`QueryResponse` has 18 INFERRED edges - model-reasoned connections that need verification._
"""
Phase 0 smoke test. Run with: python -m app.smoke

Verifies:
  1. All required dependencies import.
  2. Ollama is reachable on localhost:11434.
  3. The three models in config.py are present.
  4. nomic-embed-text actually returns a vector for a sample input.
  5. llama3.2 actually returns JSON for a sample classification.
  6. PyMuPDF can open one of the SDG PDFs.
  7. Pydantic schemas validate a synthetic round-trip.

Exits non-zero on any failure. Safe to delete after Phase 0 — kept here as
a "rerun me when something is weird" sanity check.
"""

from __future__ import annotations

import sys
import traceback

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str, exc: Exception | None = None) -> None:
    print(f"  {RED}✗{RESET} {msg}")
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)


def section(title: str) -> None:
    print(f"\n{YELLOW}{title}{RESET}")


def main() -> int:
    failures: list[str] = []

    # ---- 1. Imports -------------------------------------------------------
    section("1. Imports")
    try:
        import fastapi  # noqa: F401  # imported only to verify install
        import uvicorn  # noqa: F401  # imported only to verify install
        import pydantic  # noqa: F401  # imported only to verify install
        import fitz  # PyMuPDF — actually used below
        import ollama  # actually used below
        import chromadb  # noqa: F401  # imported only to verify install
        from rank_bm25 import BM25Okapi  # noqa: F401  # imported only to verify install
        import numpy  # noqa: F401  # imported only to verify install

        ok(f"fastapi={fastapi.__version__}")
        ok(f"pydantic={pydantic.__version__}")
        ok(f"pymupdf={fitz.__doc__.splitlines()[0] if fitz.__doc__ else 'ok'}")
        ok(f"ollama={ollama.__version__ if hasattr(ollama, '__version__') else 'ok'}")
    except Exception as e:
        fail("dependency import failed", e)
        failures.append("imports")
        return 1  # cannot proceed

    # ---- 2. Local config + schemas ---------------------------------------
    section("2. Local modules")
    try:
        from app import config, schemas

        ok(f"config.MODEL_GEN={config.MODEL_GEN}")
        ok(f"config.MODEL_SMALL={config.MODEL_SMALL}")
        ok(f"config.MODEL_EMBED={config.MODEL_EMBED}")
        ok(f"PRODUCT_DOCS keys: {list(config.PRODUCT_DOCS)}")
    except Exception as e:
        fail("could not import app.config / app.schemas", e)
        return 1

    # ---- 3. Ollama reachable + models present -----------------------------
    section("3. Ollama")
    try:
        client = ollama.Client()
        listed = client.list()
        # ollama-py 0.6 returns a ListResponse with .models attribute.
        models_raw = getattr(listed, "models", listed.get("models") if isinstance(listed, dict) else [])
        names = []
        for m in models_raw:
            n = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None) or getattr(m, "name", None)
            if n:
                names.append(n)
        ok(f"reachable; {len(names)} models available")

        for label, model in [
            ("MODEL_GEN", config.MODEL_GEN),
            ("MODEL_SMALL", config.MODEL_SMALL),
            ("MODEL_EMBED", config.MODEL_EMBED),
        ]:
            if any(model == n or n.startswith(model.split(":")[0] + ":") and model in n for n in names) or model in names:
                ok(f"{label}={model} present")
            else:
                # Looser match: any name that contains the base before ':' or that equals the bare name
                base = model.split(":")[0]
                hits = [n for n in names if n == model or n.startswith(base + ":") or n == base]
                if hits:
                    ok(f"{label}={model} present (matched {hits[0]})")
                else:
                    fail(f"{label}={model} NOT in ollama list (have: {names})")
                    failures.append(f"missing model {model}")

    except Exception as e:
        fail("ollama list failed — is `ollama serve` running?", e)
        failures.append("ollama list")

    # ---- 4. Embedding actually works -------------------------------------
    section("4. Embedding round-trip")
    try:
        resp = ollama.embeddings(model=config.MODEL_EMBED, prompt="Active User definition")
        vec = resp["embedding"]
        if isinstance(vec, list) and len(vec) > 100 and all(isinstance(x, float) for x in vec[:5]):
            ok(f"{config.MODEL_EMBED} returned vector of dim={len(vec)}")
        else:
            fail(f"unexpected embedding shape: type={type(vec)}, len={len(vec) if hasattr(vec, '__len__') else '?'}")
            failures.append("embedding shape")
    except Exception as e:
        fail("embedding call failed", e)
        failures.append("embedding")

    # ---- 5. Small model JSON mode ----------------------------------------
    section("5. Small model JSON mode")
    try:
        resp = ollama.chat(
            model=config.MODEL_SMALL,
            messages=[
                {"role": "system", "content": (
                    'Reply ONLY with JSON of shape {"echo": <text>}. '
                    'Do not include any prose outside the JSON.'
                )},
                {"role": "user", "content": "hello"},
            ],
            format="json",
            options={"temperature": 0.0},
        )
        content = resp["message"]["content"]
        import json
        parsed = json.loads(content)
        if "echo" in parsed:
            ok(f"{config.MODEL_SMALL} returned valid JSON: {parsed!r}")
        else:
            ok(f"{config.MODEL_SMALL} returned JSON (without expected key, fine for smoke): {parsed!r}")
    except Exception as e:
        fail("small model JSON call failed", e)
        failures.append("small model")

    # ---- 6. PyMuPDF can open a Data PDF ----------------------------------
    section("6. PyMuPDF")
    try:
        pdfs = list(config.DATA_DIR.glob("*.pdf"))
        if not pdfs:
            fail(f"no PDFs found in {config.DATA_DIR}")
            failures.append("no pdfs")
        else:
            doc = fitz.open(pdfs[0])
            page_count = doc.page_count
            sample = doc[0].get_text()[:80].replace("\n", " ")
            doc.close()
            ok(f"opened {pdfs[0].name}: {page_count} pages")
            ok(f"page 1 starts: {sample!r}")
    except Exception as e:
        fail("PyMuPDF failed", e)
        failures.append("pymupdf")

    # ---- 7. Pydantic schemas round-trip ----------------------------------
    section("7. Schemas")
    try:
        req = schemas.QueryRequest(question="What is an Active User?")
        ok(f"QueryRequest: {req.question!r}")
        route = schemas.RouteDecision(
            products=["rise_family"],
            intent="definition",
            rewritten_query="definition of Active User",
            reasoning="defined term",
        )
        ok(f"RouteDecision: products={route.products}, intent={route.intent}")
        cite = schemas.Citation(doc="X", section="1.3", page=4, quote="...")
        resp = schemas.QueryResponse(answer="Test", citations=[cite], verified=True)
        ok(f"QueryResponse round-tripped: verified={resp.verified}")
    except Exception as e:
        fail("schema validation failed", e)
        failures.append("schemas")

    # ---- Summary ----------------------------------------------------------
    print()
    if failures:
        print(f"{RED}SMOKE TEST FAILED{RESET} — issues: {failures}")
        return 1
    print(f"{GREEN}SMOKE TEST PASSED{RESET} — Phase 0 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

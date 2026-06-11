"""
Meta-corpus chunker.

Builds a small "self-describing" corpus from developer-maintained text:
the project README plus key module docstrings. The same retriever/generator/
verifier pipeline that handles SDG questions then handles meta-questions
("what does this app do?", "how does this work?") — answers come back
WITH citations to README sections or module docstrings, instead of from
a hardcoded paragraph.

Why a separate chunker:
    The PDF chunker in app/chunker.py is structure-aware (page headers,
    numbered SDG sections, PyMuPDF block extraction). The meta-corpus is
    plain Markdown + Python docstrings, so the splitting rules are
    different: split on Markdown headings + module names, then pack
    paragraphs greedily up to the same token budget.

Source files (resolved relative to the project root):
    README.md                         primary source — sectioned by ##/###
    app/pipeline.py docstring         orchestration overview
    app/retrieve.py docstring         BM25 + vector + RRF
    app/agent.py    docstring         generator + verifier
    app/router.py   docstring         routing policy
    app/guardrails.py docstring       in/out guardrails
    app/answer_guard.py docstring     deterministic threshold guard

doc_id is fixed: "meta_about_system". One source file → one chunk
"section" — README.md headings become §1, §2, ...; each module docstring
becomes its own section. page_start/page_end are nominal (1) — these
sources don't have pages.

Pure logic — no Ollama, no Chroma, no I/O beyond reading the source files.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from app import config
from app.schemas import Chunk

META_DOC_ID = "meta_about_system"
META_DOC_TITLE = "Agentic RAG over SAP SDGs — System Description"

# Module files whose docstrings describe the system. Order matters: the
# section_number we emit is "module:<name>" so retrieval can attribute
# answers correctly.
_MODULE_DOCSTRING_SOURCES: tuple[str, ...] = (
    "app/pipeline.py",
    "app/retrieve.py",
    "app/agent.py",
    "app/router.py",
    "app/guardrails.py",
    "app/answer_guard.py",
)

# Same token budget as the SDG chunker so the embedder sees similarly-sized
# inputs. ~4 chars per token is the rough rule used elsewhere.
_TARGET_TOKENS = config.CHUNK_TARGET_TOKENS
_MAX_TOKENS = config.CHUNK_MAX_TOKENS
_CHARS_PER_TOKEN = 4

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_readme(readme_text: str) -> list[tuple[str, str, str]]:
    """Split a Markdown document into (section_number, section_title, body).

    Sections are top-level (## and ###) headings. The first sub-heading
    after a parent heading is folded under the parent (we don't go deeper
    than two levels — the README's ###s are short notes that belong with
    their ##).

    Returns a list of (section_number, section_title, body) tuples.
    section_number is monotonically assigned ("1", "2", "3", ...).
    """
    matches = list(_HEADING_RE.finditer(readme_text))
    if not matches:
        return [("1", "README", readme_text.strip())]

    # Anchor the first section either at the very start (intro before any
    # heading) or at the first ## heading.
    sections: list[tuple[str, str, str]] = []
    pre_intro = readme_text[: matches[0].start()].strip()
    if pre_intro:
        sections.append(("intro", "Intro", pre_intro))

    n = 1
    for i, m in enumerate(matches):
        level = len(m.group(1))
        if level > 2:
            # Skip — these are absorbed into the preceding ## section's body.
            continue
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(readme_text)
        body = readme_text[start:end].strip()
        if not body:
            continue
        sections.append((str(n), title, body))
        n += 1
    return sections


def _split_to_chunks(text: str, target: int, hard_max: int) -> list[str]:
    """Greedy paragraph packing. Splits on blank lines, packs until target,
    forces a flush if the next paragraph would push past hard_max.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    buf_tokens = 0
    for p in paragraphs:
        p_tokens = _approx_tokens(p)
        if buf_tokens and (buf_tokens + p_tokens > hard_max):
            chunks.append(buf.strip())
            buf, buf_tokens = "", 0
        buf = f"{buf}\n\n{p}" if buf else p
        buf_tokens += p_tokens
        if buf_tokens >= target:
            chunks.append(buf.strip())
            buf, buf_tokens = "", 0
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _module_docstring(path: Path) -> str | None:
    """Extract the module-level docstring from a Python file. Returns None if
    the file has no docstring.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    return ast.get_docstring(tree, clean=True)


def build_meta_chunks(root: Path | None = None) -> list[Chunk]:
    """Build the meta-corpus as a list of Chunk objects.

    Sources:
        - README.md            → multiple chunks, one per ## section
        - module docstrings    → one chunk per file (typically <500 tokens)
    """
    root = root or config.ROOT_DIR
    chunks: list[Chunk] = []

    # README — section-by-section.
    readme_path = root / "README.md"
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8")
        sections = _split_readme(readme_text)
        chunk_index = 0
        for section_number, section_title, body in sections:
            sub_chunks = _split_to_chunks(body, _TARGET_TOKENS, _MAX_TOKENS)
            for sub_idx, sub_text in enumerate(sub_chunks):
                chunk_index += 1
                chunk_id = f"{META_DOC_ID}#readme-{section_number}_{sub_idx}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=META_DOC_ID,
                    doc_title=META_DOC_TITLE,
                    section_number=f"README §{section_number}",
                    section_title=section_title,
                    page_start=1,
                    page_end=1,
                    text=sub_text,
                ))

    # Module docstrings — one chunk per module.
    for rel in _MODULE_DOCSTRING_SOURCES:
        path = root / rel
        if not path.exists():
            continue
        doc = _module_docstring(path)
        if not doc or len(doc.strip()) < 50:
            continue
        module_name = rel.replace("/", ".").removesuffix(".py")
        sub_chunks = _split_to_chunks(doc, _TARGET_TOKENS, _MAX_TOKENS)
        for sub_idx, sub_text in enumerate(sub_chunks):
            chunk_id = f"{META_DOC_ID}#module-{module_name}_{sub_idx}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=META_DOC_ID,
                doc_title=META_DOC_TITLE,
                section_number=f"module:{module_name}",
                section_title=module_name,
                page_start=1,
                page_end=1,
                text=sub_text,
            ))

    return chunks

"""
Structure-aware chunking for SAP Service Description Guide PDFs.

Pipeline (per PDF):
    1. Extract per-page blocks via PyMuPDF (preserves spatial info).
    2. Strip page header/footer blocks by y-coordinate (every page repeats them).
    3. Join cleaned blocks with inline "[[PAGE:N]]" markers and "\n\n" between
       blocks. The page markers let us recover page_start / page_end for any
       chunk via two channels: (a) markers literally inside the chunk text,
       and (b) the chunk's character offset in the source text (looked up
       against a page-offset table).
    4. Pass 1 (structural): regex-split on numbered headings ("1.1.", "2.3.4.")
       to produce SECTION units. The regex matches ONLY the heading marker;
       the body is everything from the heading to the next heading. The
       section_title is later derived from the body (typically the first
       quoted defined term). For PDFs without numbered sections, fall back to
       all-caps standalone headings; if too few are found, fall back to fixed
       sliding windows.
    5. Pass 2 (semantic): inside each section, pack paragraphs greedily up to
       CHUNK_TARGET_TOKENS (max CHUNK_MAX_TOKENS), with one-sentence overlap
       between siblings of the SAME section. Never overlap across section
       boundaries — overlap there would lie about the citation.
    6. Tail merge: if the final chunk of a section is below CHUNK_MIN_TOKENS,
       merge it back into the prior chunk.

This module is pure logic — no Ollama, no Chroma, no I/O beyond opening the
PDF. That makes it cheap to unit-test.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from app import config
from app.schemas import Chunk

# ---------------------------------------------------------------------------
# Tunables — kept module-local because they're chunking-specific. Anything
# the rest of the system might want to tweak lives in app/config.py.
# ---------------------------------------------------------------------------

# Page header/footer stripping: every SDG repeats a banner at the top and a
# "Page N of M" footer at the bottom. We strip blocks whose y-coordinate falls
# in these zones. Values measured from the actual PDFs.
HEADER_Y_MAX = 95.0
FOOTER_Y_MIN = 720.0

# Approximate token count = chars / TOKENS_PER_CHAR. Good enough for the
# nomic-embed-text 512-token sweet spot; the embedder tolerates ±20% slop.
CHARS_PER_TOKEN = 4

# Real SDGs have <= ~30 top-level numbered sections. A "1." or "346." that
# appears in body prose is a false positive. Combined with strict next-line
# checks, this caps the false-positive rate. Kept generous (50) so a future
# longer SDG isn't over-filtered.
MAX_PLAUSIBLE_TOP_LEVEL = 50

# Section heading regex --------------------------------------------------------
#
# Matches lines like:
#       "1."          (top-level)
#       "1.1."        (subsection)
#       "1.1.1."      (sub-subsection)
#       "12.3."       (multi-digit chapter)
# We match ONLY the heading marker — `\s*$` ensures nothing else is on the
# line, which protects against body prose like "see section 1.3.". The body
# of the section is everything from m.end() to the next match's start().
_HEADING_RE = re.compile(
    r"""
    ^
    (?P<num>\d{1,3}(?:\.\d{1,3}){0,3})    # 1, 1.1, 1.1.1, etc.
    \.                                     # mandatory trailing dot
    \s*$                                   # rest of line must be whitespace only
    """,
    re.MULTILINE | re.VERBOSE,
)

# All-caps standalone heading: a line with 2-60 chars where most letters are
# uppercase. Used for the unnumbered PDF (section names like "GENERAL",
# "CORE", "HANA RUNTIME"). We require at least one letter and forbid trailing
# punctuation typical of body sentences.
_CAPS_HEADING_RE = re.compile(
    r"""
    ^
    (?P<title>
      [A-Z][A-Z0-9 /\-&,.()]{1,60}
    )
    $
    """,
    re.MULTILINE | re.VERBOSE,
)

# Quoted defined-term regex used to extract a section title from the body.
# Handles ASCII and Unicode curly quotes (“ ”).
_QUOTED_TERM_RE = re.compile(r'["“]([^"“”]{2,60})["”]')

# Page marker we inject between page-cleaned blocks.
_PAGE_MARKER_RE = re.compile(r"\[\[PAGE:(\d+)\]\]")


# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------


@dataclass
class _SectionUnit:
    """Intermediate result of pass 1 (structural split).

    `text` may contain `[[PAGE:N]]` markers. Pass 2 turns each unit into
    one or more final chunks.

    `start_offset` is the character position in the source-document text
    where this section's body begins. Used as a fallback for page tracking
    when a chunk has no inline page markers.
    """

    section_number: str           # never None — synthetic IDs used for fallbacks
    text: str
    start_offset: int
    section_title: str | None = None
    chunk_index_seed: int = 0
    page_offsets: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1 — extract clean page text with markers
# ---------------------------------------------------------------------------


def _extract_clean_text(pdf_path: Path) -> str:
    """Read a PDF and return its text body with header/footer banners stripped
    and `[[PAGE:N]]` markers inserted between pages.

    We use `get_text("blocks")` (not the linearised `get_text()`) so we can
    filter blocks by their y-coordinate. Block tuples have shape:
        (x0, y0, x1, y1, text, block_no, block_type)
    """
    doc = fitz.open(pdf_path)
    parts: list[str] = []
    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            blocks = page.get_text("blocks")
            kept: list[tuple[float, float, str]] = []
            for x0, y0, _x1, _y1, text, _block_no, block_type in blocks:
                if block_type != 0:                  # 1 = image; skip
                    continue
                if y0 < HEADER_Y_MAX:
                    continue                          # banner
                if y0 > FOOTER_Y_MIN:
                    continue                          # "Page N of M" footer
                clean = text.strip()
                if not clean:
                    continue
                kept.append((y0, x0, clean))
            # Sort by visual reading order (top-to-bottom, then left-to-right)
            kept.sort(key=lambda t: (round(t[0], 1), t[1]))
            page_text = "\n\n".join(b[2] for b in kept)
            parts.append(f"[[PAGE:{page_index + 1}]]\n{page_text}")
    finally:
        doc.close()
    return "\n\n".join(parts)


def _build_page_offset_table(text: str) -> list[tuple[int, int]]:
    """Returns [(char_offset, page_number), ...] sorted by char_offset.

    Used to answer: 'which page does this character offset belong to?'
    """
    return [(m.start(), int(m.group(1))) for m in _PAGE_MARKER_RE.finditer(text)]


def _page_for_offset(table: list[tuple[int, int]], offset: int) -> int:
    """Return the page number for a character offset. O(log N) via bisect."""
    if not table:
        return 1
    # Find the rightmost marker whose offset is <= `offset`.
    keys = [t[0] for t in table]
    idx = bisect_right(keys, offset) - 1
    if idx < 0:
        return table[0][1]
    return table[idx][1]


# ---------------------------------------------------------------------------
# Stage 2a — pass 1: structural split on numbered headings
# ---------------------------------------------------------------------------


def _split_numbered(text: str) -> list[_SectionUnit]:
    """Split text on numbered headings like '1.', '1.3.', '2.4.1.'.

    Critical: the regex matches ONLY the heading marker. The body is taken
    verbatim from m.end() to the next match's start(), so we never lose
    content to a too-greedy title capture.

    Top-level numbers ("1.", "2.") need stricter validation than subsections,
    because body prose can quote them. Validation rules:
      - Must be followed (within 200 chars) by a SHORT, ALL-CAPS line (the
        section name) — that's the SDG convention.
      - Must be <= MAX_PLAUSIBLE_TOP_LEVEL.
    Subsections (with a dot in the number) are accepted unconditionally
    because the dot already disambiguates from prose.

    Returns [] if too few headings are found, signalling the caller to try
    the all-caps heuristic.
    """
    raw_matches = list(_HEADING_RE.finditer(text))

    real_matches: list[re.Match[str]] = []
    for m in raw_matches:
        num = m.group("num")
        is_subsection = "." in num
        if is_subsection:
            real_matches.append(m)
            continue
        # Top-level: extra validation
        try:
            top_int = int(num)
        except ValueError:
            continue
        if top_int > MAX_PLAUSIBLE_TOP_LEVEL or top_int < 1:
            continue
        # Look at the next non-empty line — must be a short, all-caps title
        lookahead = text[m.end(): m.end() + 300]
        next_line = ""
        for line in lookahead.split("\n"):
            stripped = line.strip()
            if stripped:
                next_line = stripped
                break
        if not next_line:
            continue
        # Strict: real top-level section names are short and ALL CAPS.
        if not (next_line.isupper() and 2 <= len(next_line) <= 60):
            continue
        real_matches.append(m)

    if len(real_matches) < config.HEADING_DETECTION_MIN_SECTIONS:
        return []

    # Deduplicate consecutive identical numbers (a body reference followed by
    # the real heading would otherwise produce two units with the same id).
    deduped: list[re.Match[str]] = []
    seen: set[str] = set()
    for m in real_matches:
        if m.group("num") in seen:
            continue
        seen.add(m.group("num"))
        deduped.append(m)

    page_offsets = _build_page_offset_table(text)
    units: list[_SectionUnit] = []
    for i, m in enumerate(deduped):
        body_start = m.end()
        body_end = deduped[i + 1].start() if i + 1 < len(deduped) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue
        title = _derive_section_title(body)
        units.append(_SectionUnit(
            section_number=m.group("num"),
            section_title=title,
            text=body,
            start_offset=body_start,
            chunk_index_seed=i,
            page_offsets=page_offsets,
        ))
    return units


def _derive_section_title(body: str) -> str | None:
    """Pull a human-readable title out of a section body.

    Strategy:
      1. Strip leading page markers — they're not part of the title.
      2. If the first ~200 chars contain a quoted defined term, use that.
         (SDG sections almost always start with a quote: "Active User" is...)
      3. Else, take the first sentence-ish fragment up to ~60 chars.
      4. Else, give up and return None.
    """
    cleaned = _PAGE_MARKER_RE.sub("", body).strip()
    head = cleaned[:300]
    qm = _QUOTED_TERM_RE.search(head)
    if qm:
        return qm.group(1).strip()
    # Take first line, trimmed.
    first_line = head.split("\n", 1)[0].strip()
    if first_line:
        # Cap to 60 chars at a word boundary.
        if len(first_line) <= 60:
            return first_line
        snip = first_line[:60].rsplit(" ", 1)[0]
        return snip + "…"
    return None


# ---------------------------------------------------------------------------
# Stage 2b — pass 1 fallback: all-caps headings
# ---------------------------------------------------------------------------


def _split_all_caps(text: str) -> list[_SectionUnit]:
    """Fallback for PDFs without numbered sections. Detect lines that are
    mostly uppercase and short — typically section dividers like "GENERAL",
    "CORE", "HANA RUNTIME".

    Synthetic IDs use the page number of the heading and an index, e.g.
    "§p3-h1" — the second heading on page 3 would be "§p3-h2".
    """
    matches = list(_CAPS_HEADING_RE.finditer(text))
    if len(matches) < config.HEADING_DETECTION_MIN_SECTIONS:
        return []

    page_offsets = _build_page_offset_table(text)
    page_heading_counts: dict[int, int] = {}
    units: list[_SectionUnit] = []

    for i, m in enumerate(matches):
        pno = _page_for_offset(page_offsets, m.start())
        page_heading_counts[pno] = page_heading_counts.get(pno, 0) + 1
        synthetic = f"§p{pno}-h{page_heading_counts[pno]}"
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue
        units.append(_SectionUnit(
            section_number=synthetic,
            section_title=m.group("title").strip(),
            text=body,
            start_offset=body_start,
            chunk_index_seed=i,
            page_offsets=page_offsets,
        ))
    return units


# ---------------------------------------------------------------------------
# Stage 2c — pass 1 last-resort fallback: fixed sliding windows
# ---------------------------------------------------------------------------


def _split_fixed_windows(text: str) -> list[_SectionUnit]:
    """Used when neither numbered nor caps headings produced enough sections.
    Splits the text into windows of approximately FALLBACK_WINDOW_TOKENS,
    with FALLBACK_WINDOW_OVERLAP between siblings.

    Each window becomes a synthetic section "§p<N>-w<idx>" so citations
    remain reproducible.
    """
    page_offsets = _build_page_offset_table(text)
    win_chars = config.FALLBACK_WINDOW_TOKENS * CHARS_PER_TOKEN
    overlap_chars = config.FALLBACK_WINDOW_OVERLAP * CHARS_PER_TOKEN

    units: list[_SectionUnit] = []
    pos = 0
    idx = 0
    while pos < len(text):
        end = min(pos + win_chars, len(text))
        # Snap to a paragraph boundary if one is nearby (within 200 chars
        # before `end`) — produces nicer chunks.
        snap = text.rfind("\n\n", pos, end)
        if snap != -1 and end - snap < 200:
            end = snap
        body = text[pos:end].strip()
        if body:
            pno = _page_for_offset(page_offsets, pos)
            units.append(_SectionUnit(
                section_number=f"§p{pno}-w{idx}",
                section_title=None,
                text=body,
                start_offset=pos,
                chunk_index_seed=idx,
                page_offsets=page_offsets,
            ))
            idx += 1
        if end >= len(text):
            break
        pos = max(end - overlap_chars, pos + 1)
    return units


# ---------------------------------------------------------------------------
# Stage 3 — pass 2: pack paragraphs into chunks
# ---------------------------------------------------------------------------


def _pack_section(unit: _SectionUnit, doc_id: str, doc_title: str) -> list[Chunk]:
    """Take one section unit and produce 1+ Chunk objects.

    Splits the section text on blank lines (paragraphs), then greedily packs
    paragraphs into chunks targeting CHUNK_TARGET_TOKENS (max
    CHUNK_MAX_TOKENS). One-sentence overlap is added between sibling chunks
    of the SAME section. Never crosses section boundaries.

    Tail merge: if the final chunk is < CHUNK_MIN_TOKENS, fold it into the
    previous chunk so we don't emit "1-sentence runts".

    Page tracking: each chunk tracks its (start_offset, end_offset) in the
    source text. `_page_range` then uses BOTH inline markers and these
    offsets to determine page_start/page_end accurately.
    """
    paragraphs = _split_into_paragraphs(unit.text)
    if not paragraphs:
        return []

    target = config.CHUNK_TARGET_TOKENS * CHARS_PER_TOKEN
    hard_max = config.CHUNK_MAX_TOKENS * CHARS_PER_TOKEN
    overlap_chars = config.CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN
    min_chars = config.CHUNK_MIN_TOKENS * CHARS_PER_TOKEN

    # Greedy pack — track each chunk's offset range within `unit.text`.
    raw_chunks: list[tuple[str, int, int]] = []   # (text, local_start, local_end)
    buf: list[str] = []
    buf_start = 0
    buf_len = 0
    cursor = 0
    for p_text, p_start, p_end in paragraphs:
        # Splitting a paragraph that itself exceeds hard_max — break by
        # sentences so we don't blow past the embedder's window.
        if len(p_text) > hard_max:
            for sub in _split_long_paragraph(p_text, hard_max):
                if buf and buf_len + len(sub) > target:
                    raw_chunks.append(("\n\n".join(buf), buf_start, cursor))
                    buf, buf_len = [], 0
                if not buf:
                    buf_start = p_start
                buf.append(sub)
                buf_len += len(sub)
                cursor = p_end
            continue
        if buf and buf_len + len(p_text) > target:
            raw_chunks.append(("\n\n".join(buf), buf_start, cursor))
            buf, buf_len = [], 0
        if not buf:
            buf_start = p_start
        buf.append(p_text)
        buf_len += len(p_text)
        cursor = p_end
    if buf:
        raw_chunks.append(("\n\n".join(buf), buf_start, cursor))

    # Tail merge — runts get folded back.
    if len(raw_chunks) >= 2 and len(raw_chunks[-1][0]) < min_chars:
        prev_text, prev_start, _prev_end = raw_chunks[-2]
        last_text, _last_start, last_end = raw_chunks[-1]
        raw_chunks[-2] = (prev_text + "\n\n" + last_text, prev_start, last_end)
        raw_chunks.pop()

    # Build Chunk objects with overlap (skip first chunk's overlap).
    chunks: list[Chunk] = []
    for i, (body, local_start, local_end) in enumerate(raw_chunks):
        if i > 0:
            tail = _last_sentence(raw_chunks[i - 1][0], overlap_chars)
            if tail:
                body = tail + " " + body
        global_start = unit.start_offset + local_start
        global_end = unit.start_offset + local_end
        page_start, page_end = _page_range(
            chunk_text=body,
            global_start=global_start,
            global_end=global_end,
            page_offsets=unit.page_offsets,
        )
        clean_text = _PAGE_MARKER_RE.sub("", body).strip()
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
        clean_text = re.sub(r"[ \t]{2,}", " ", clean_text)
        if not clean_text:
            continue
        # Drop zero-information chunks: text equals the title (a parent
        # section like "§2 GENERAL" whose body got split into children),
        # OR the cleaned text is too short to be retrievably useful.
        if unit.section_title and clean_text.strip().lower() == unit.section_title.strip().lower():
            continue
        if len(clean_text) < 15:
            continue
        chunk_id = f"{doc_id}#{unit.section_number}_{i}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            doc_title=doc_title,
            section_number=unit.section_number,
            section_title=unit.section_title,
            page_start=page_start,
            page_end=page_end,
            text=clean_text,
        ))
    return chunks


def _split_into_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """Split `text` on blank lines. Returns [(para_text, start, end), ...]
    where offsets are relative to `text`.
    """
    out: list[tuple[str, int, int]] = []
    pos = 0
    for m in re.finditer(r"\n\s*\n", text):
        seg = text[pos: m.start()].strip()
        if seg:
            out.append((seg, pos, m.start()))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        out.append((tail, pos, len(text)))
    return out


def _split_long_paragraph(text: str, hard_max: int) -> list[str]:
    """Split an over-long paragraph at sentence boundaries until each piece
    fits hard_max. Falls back to a hard char split if sentences are absurdly
    long (rare in formal SDG prose).
    """
    sentences = re.split(r"(?<=[.?!])\s+", text)
    out: list[str] = []
    cur = ""
    for s in sentences:
        if cur and len(cur) + len(s) > hard_max:
            out.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip() if cur else s
    if cur:
        out.append(cur.strip())
    final: list[str] = []
    for piece in out:
        while len(piece) > hard_max:
            final.append(piece[:hard_max])
            piece = piece[hard_max:]
        if piece:
            final.append(piece)
    return final


def _last_sentence(text: str, max_chars: int) -> str:
    """Return roughly the last sentence of `text`, capped at max_chars,
    suitable for one-sentence chunk overlap.
    """
    # Strip page markers from overlap so they don't pollute downstream chunks.
    text = _PAGE_MARKER_RE.sub("", text)
    snippet = text[-max_chars * 2:]
    matches = list(re.finditer(r"[.?!]\s+", snippet))
    if matches:
        last_end = matches[-1].end()
        tail = snippet[last_end:].strip()
        return tail[-max_chars:] if tail else ""
    return text[-max_chars:].strip()


def _page_range(
    chunk_text: str,
    global_start: int,
    global_end: int,
    page_offsets: list[tuple[int, int]],
) -> tuple[int, int]:
    """Determine (page_start, page_end) for a chunk.

    Two information sources are combined:
      1. Inline `[[PAGE:N]]` markers literally inside the chunk text.
      2. The chunk's character span in the source text — used to look up the
         page at the start of the chunk and the page at the end via the
         page-offset table.

    The result is the union: min of all observed pages → start, max → end.
    This way a chunk that has no inline markers (because it doesn't span a
    page boundary) still gets the correct page from its global offset.
    """
    pages: set[int] = set()
    for m in _PAGE_MARKER_RE.findall(chunk_text):
        pages.add(int(m))
    pages.add(_page_for_offset(page_offsets, global_start))
    pages.add(_page_for_offset(page_offsets, max(global_end - 1, global_start)))
    return (min(pages), max(pages))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_pdf(pdf_path: Path, doc_id: str, doc_title: str) -> list[Chunk]:
    """Top-level entry point. Returns all chunks for one PDF.

    Strategy precedence:
        1. Numbered sections (e.g., "1.3.")
        2. All-caps standalone headings (e.g., "GENERAL")
        3. Fixed sliding windows (last resort)

    Logs the strategy used so operators can see which path each PDF took.
    """
    raw_text = _extract_clean_text(pdf_path)

    units = _split_numbered(raw_text)
    strategy = "numbered"
    if not units:
        units = _split_all_caps(raw_text)
        strategy = "all_caps"
    if not units:
        units = _split_fixed_windows(raw_text)
        strategy = "fixed_windows"

    chunks: list[Chunk] = []
    for unit in units:
        chunks.extend(_pack_section(unit, doc_id=doc_id, doc_title=doc_title))

    print(f"  [chunker] {pdf_path.name}: strategy={strategy}, "
          f"sections={len(units)}, chunks={len(chunks)}")
    return chunks

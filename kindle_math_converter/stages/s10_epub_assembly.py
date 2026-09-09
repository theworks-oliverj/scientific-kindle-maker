# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""
Stage 10 — EPUB3 Assembly
Assembles the processed document into a valid EPUB3 file.
SVGs are inlined directly in XHTML — never as <object> tags (KindleGen rejects those).
Runs epubcheck validation after assembly.
Failure mode: Fatal.
"""
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from ..models.document import Document, EquationRegion
from ..models.enums import FormulaClass, ErrorCode, SourceType
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger
# Shared with Stage 8A's per-equation gate — see find_orphaned_use_refs
# docstring for why this whole-document check must never diverge from it.
from .s09_svg_postprocess import find_orphaned_use_refs

log = get_logger("s10_epub_assembly")

BOOK_CSS = """\
/* Kindle-specific equation styles */
@media amzn-kf8 {
  .eq-inline {
    display: inline-block;
    vertical-align: middle;
  }
  .eq-display {
    display: block;
    text-align: center;
    margin: 1em 0;
    /* Reserves the equation number's column. A float is out of flow, so
       text-align:center will not clear it — padding shifts the centring box
       AND shrinks what max-width:100% resolves against, so the number's
       space comes out of the equation's width budget instead of overlapping
       it. Pair with the .eq-number span being emitted BEFORE the SVG. */
    padding-right: 3em;
  }
  .eq-display svg {
    max-width: 100%;
  }
  .eq-number {
    float: right;
  }
  .eq-flagged {
    color: #cc0000;
    font-family: monospace;
    font-size: 0.8em;
    border: 1px solid #cc0000;
    padding: 0 4px;
  }
  .eq-text {
    white-space: nowrap;
  }
}

/* Fallback for non-Kindle EPUB readers */
.eq-inline { display: inline-block; vertical-align: middle; }
.eq-display { display: block; text-align: center; margin: 1em 0; padding-right: 3em; }
.eq-display svg { max-width: 100%; }
.eq-number { float: right; }
.eq-flagged { color: #cc0000; font-family: monospace; border: 1px solid #cc0000; padding: 0 4px; }
.eq-text { white-space: nowrap; }
.eq-img-fallback { max-width: 100%; }
span.eq-inline img.eq-img-fallback { height: 1.2em; width: auto; vertical-align: middle; }
.figure { text-align: center; margin: 1em 0; }
.figure img { max-width: 100%; }

/* Code blocks (<pre> in the source, typically syntax-highlighted). Kindle
   screens are narrow, so a fixed-width pre with no wrap overflows off the
   page — pre-wrap keeps original line breaks but lets long lines fold. */
pre.code {
  font-family: monospace;
  font-size: 0.8em;
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-wrap: break-word;
  background: #f5f5f0;
  border: 1px solid #ccc;
  padding: 0.6em;
  margin: 1em 0;
}

/* Footnotes recovered from the page furniture, collected at the end of the
   section holding their reference. Ordinary paragraphs on purpose — the
   EPUB3 popup markup gets hidden by reading systems (see _footnote_html). */
.footnotes { border-top: 1px solid currentColor; margin-top: 2em; padding-top: 0.5em; }
.footnotes-title { font-size: 0.8em; font-weight: bold; margin: 0 0 0.4em 0; }
.footnote { font-size: 0.85em; margin: 0.4em 0; text-indent: 0; }
.footnote-back { text-decoration: none; }
"""

XHTML_TEMPLATE = """\
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<link rel="stylesheet" type="text/css" href="../styles/book.css"/>
</head>
<body>
{body}
</body>
</html>
"""


def _render_equation(
    region: EquationRegion,
    fallback_images: Optional[dict[str, bytes]] = None,
) -> Optional[str]:
    """
    Returns the HTML snippet for one equation region, or None if the region
    should be omitted from the page entirely (reference/footnote labels
    already present in the surrounding body text, and irrecoverably flagged
    equations with no crop to fall back on — neither should leak a raw
    [EQ:...] placeholder into the reading flow).

    Flagged/unrendered equations with a source crop are embedded as a raster
    <img> (registered in `fallback_images` for the assembler to write into
    the EPUB) — for a reading tool, a raster equation beats a missing one.
    """
    if region.render_as_text:
        span = f'<span class="eq-text">{region.inline_text_repr}</span>'
        if region.footnote_ref_id:
            # A plain link, not epub:type="noteref" — see _footnote_html for
            # why the popup markup had to go. Tapping jumps to the note at the
            # end of the section (always the same file), and the note links
            # back here.
            return (
                f'<a class="noteref" '
                f'id="{_noteref_anchor_id(region.footnote_ref_id)}" '
                f'href="#{region.footnote_ref_id}">{span}</a>'
            )
        return span

    if region.is_reference_label:
        return None

    if region.flagged_for_review or not region.svg_postprocessed:
        if fallback_images is not None and region.source_image_crop:
            fallback_images[region.region_id] = region.source_image_crop
            img = (
                f'<img class="eq-img-fallback" alt="equation {region.region_id}" '
                f'src="../images/{region.region_id}.png"/>'
            )
            if region.formula_class == FormulaClass.INLINE:
                return f'<span class="eq-inline">{img}</span>'
            return f'<div class="eq-display">{_eq_number_html(region)}{img}</div>'
        if region.text_fallback:
            # EPUB/HTML equations have no page raster to crop a fallback
            # image from (source_image_crop is always None for them, except
            # real <img>-sourced equations) — a flagged one with no crop
            # used to be dropped from the page entirely. This is the same
            # last-resort text rendering render_as_text already uses: the
            # MathML's own tokens (s05c_mathml._mathml_text_tokens) or, for
            # MathJax-sourced equations, the raw LaTeX itself (there is no
            # separate source representation to tokenize) — either way it
            # survives even when the derived/given LaTeX is exactly what
            # failed to compile.
            return f'<span class="eq-text">{_escape_text(region.text_fallback)}</span>'
        return None

    svg = region.svg_postprocessed

    if region.formula_class == FormulaClass.INLINE:
        return f'<span class="eq-inline">{svg}</span>'
    else:
        return f'<div class="eq-display">{_eq_number_html(region)}{svg}</div>'


def _noteref_anchor_id(footnote_id: str) -> str:
    """Id of the in-text marker that points at `footnote_id` ("fn_2_1" ->
    "fnref_2_1"), so the note can link back to where the reader was."""
    return "fnref_" + footnote_id[3:] if footnote_id.startswith("fn_") else f"fnref_{footnote_id}"


def _footnote_html(unit: dict, back_link: bool) -> str:
    """One recovered footnote as an ordinary, always-visible paragraph.

    Deliberately NOT <aside epub:type="footnote">. That is the EPUB3 markup
    for popup notes, and reading systems that implement popups take the aside
    out of the normal flow via their own stylesheet — so where the popup does
    not fire, the note is simply invisible. Observed on both a desktop EPUB
    reader and Kindle: all ten recovered notes silently disappeared. A plain
    <p> cannot be hidden that way, and the marker/back-link pair still gives
    two-way navigation everywhere.

    `back_link` must only be True when the marker anchor is in the same
    chapter file — an href="#…" with no target is a dangling fragment.
    """
    back = ""
    if back_link and unit["footnote_id"]:
        back = (
            f' <a class="footnote-back" '
            f'href="#{_noteref_anchor_id(unit["footnote_id"])}">↩</a>'
        )
    attrs = f' id="{unit["footnote_id"]}"' if unit["footnote_id"] else ""
    return f'<p class="footnote"{attrs}>{unit["inner"]}{back}</p>'


def _eq_number_html(region: EquationRegion) -> str:
    """The equation-number span, or "" if the equation is unnumbered.

    Emitted BEFORE the equation body: a right float can only sit beside
    content that follows it, so a number placed after a wide centred SVG got
    pushed against (or below) the equation. See .eq-display's padding-right.
    """
    if not region.equation_number:
        return ""
    return f'<span class="eq-number">{_escape_text(region.equation_number)}</span>'


# Inline-equation placeholder embedded in TextBlock.raw_text by
# s03_mineru_parse (uses only characters that survive _escape_text).
_EQ_PLACEHOLDER_RE = re.compile(r'\[\[EQ:([A-Za-z0-9_]+)\]\]')
_EMPTY_P_RE = re.compile(r'<p>\s*</p>')

# Kindle's publishing guidelines cap a single XHTML file at 300 KB. Measured
# against dvisvgm's uncompressed outlines: the SU3 paper produced a 505 KB
# section and a 229 KB one, and neither rendered reliably on device — while a
# document whose largest file was 91 KB was fine.
#
# Splitting is the ONLY lever here. Sharing repeated glyph outlines between
# equations would cut a maths-heavy file by more than half, but it requires
# <use> to reach into another <svg> element, which Kindle does not resolve —
# see _svgs_referencing_outside_themselves. Cutting per-equation SVG size has
# to happen upstream (one dvisvgm pass per group of equations), not here.
KINDLE_MAX_FILE_BYTES = 300_000
# Headroom below the hard cap: the notes block is appended at flush time,
# after the budget has already been checked, so the ceiling has to leave room
# for a section's worth of footnotes plus the template.
MAX_CHAPTER_BYTES = 220_000

# Amazon's Send to Kindle converter rejects a book whose TOTAL number of inline
# <svg> elements is too high, with a bare E999 and no diagnostic. Established by
# bisection across 15 uploads (2026-09-08/09): 1054 elements converts, 1342 does
# not. The ceiling is on the count of <svg> ELEMENTS per book and nothing else —
# these were each contradicted by a direct pass/fail inversion:
#
#   compressed EPUB size   3.44 MB passed, 3.03 MB failed
#   uncompressed bytes     17.87 MB passed, 1.13 MB failed
#   XHTML file count       180 passed, 61 failed
#   svg-BEARING file count 180 passed, 20 failed
#   PNG count              44 passed, 22 failed
#
# Kindle Previewer does NOT reproduce this — it converts the oversized books
# happily — so the only way to stay safe is to keep the count under budget.
# 900 sits below the largest observed pass (1054) with headroom.
KINDLE_MAX_EQUATIONS_PER_VOLUME = 900

# The footnote id an emitted in-text marker points at (see _render_equation).
_NOTEREF_ID_RE = re.compile(r'<a class="noteref"[^>]*href="#(fn_[^"]+)"')

# A paragraph whose visible text ends in one of these is considered complete;
# anything else at a page boundary is treated as a continuation and merged.
_TERMINAL_TAIL_RE = re.compile(r'[.!?:;…"”\'’)\]]\s*$')

# ── URL / DOI linkification ───────────────────────────────────────────────
# Reference sections carry live URLs ("View online: https://doi.org/…") that
# were reaching the EPUB as dead text. Matched against ALREADY-ESCAPED text:
# a "&" in a query string arrives as "&amp;", which is exactly what an XHTML
# href wants, and the other entities are handled in _linkify.
_URL_RE = re.compile(
    r'(?<![\w@.])('
    r'https?://[^\s<>"]+'
    r'|www\.[^\s<>"]+'
    r'|doi:\s*10\.\d{4,}/[^\s<>"]+'
    r'|10\.\d{4,}/[^\s<>"]+'
    r')',
    re.IGNORECASE,
)
# Sentence punctuation that follows a URL rather than belonging to it.
_URL_TRAILING_RE = re.compile(r'[.,;:!?\'"]+$')
_CLOSERS = {")": "(", "]": "[", "}": "{"}
# Characters legal in the href we emit, checked after "&amp;" is folded back
# to "&". Deliberately excludes "[" and "]" — legal only in a URI host, and
# epubcheck rejects them in a path segment.
_URL_SAFE_RE = re.compile(r"^[A-Za-z0-9\-._~:/?#@!$&'()*+,;=%]+$")


def _trim_url(url: str) -> str:
    """Peels off trailing characters that belong to the surrounding prose
    rather than the address: sentence punctuation and unpaired closing
    brackets. The two interleave — "…1811620]." needs the "." stripped before
    the "]" becomes visible — so peel until stable."""
    while url:
        tm = _URL_TRAILING_RE.search(url)
        if tm:
            url = url[:tm.start()]
            continue
        last = url[-1]
        opener = _CLOSERS.get(last)
        if opener is not None and url.count(last) > url.count(opener):
            url = url[:-1]
            continue
        break
    return url


def _linkify(escaped_text: str) -> str:
    """Wraps bare URLs and DOIs in `escaped_text` in <a href>.

    Must run on escaped text and BEFORE equation placeholders are substituted:
    placeholders hold nothing URL-shaped, but the generated SVG/HTML markup
    would otherwise be scanned (and mangled) by the URL pattern.

    Anything that does not trim down to a clean URI is left as plain text.
    epubcheck is fatal in this stage, so an over-eager match would fail the
    whole build — real sources put URLs inside "[DOI: …]" and "<…>", and
    citation text runs straight on after the closing bracket.

    Known limitations: this is per-text-block, and runs before the cross-page
    continuation merge in `_document_to_chapters`, so a URL split across a
    column or page boundary yields two partial links. And the bare "10.xxxx/…"
    form is a heuristic, unlike the three scheme-bearing forms — anything
    DOI-shaped in running prose becomes a link. Worst case is a dead link, not
    a broken build: _URL_SAFE_RE still guards the href.
    """
    def replace(match: "re.Match[str]") -> str:
        url = match.group(1)
        # The source may wrap a URL in angle brackets or quotes ("<http://…>"),
        # which _escape_text has already turned into entities. Those end the
        # URL — only &amp; may legitimately appear inside one.
        for entity in ("&lt;", "&gt;", "&quot;"):
            url = url.split(entity, 1)[0]

        url = _trim_url(url)
        if not url:
            return match.group(0)

        low = url.lower()
        if low.startswith(("http://", "https://")):
            href = url
        elif low.startswith("www."):
            href = f"http://{url}"
        elif low.startswith("doi:"):
            # "doi: 10.1119/…" — the visible label keeps its prefix and space,
            # so the safety check below has to see the href, not the label.
            href = f"https://doi.org/{url.split(':', 1)[1].strip()}"
        else:
            href = f"https://doi.org/{url}"

        if not _URL_SAFE_RE.match(href.replace("&amp;", "&")):
            return match.group(0)
        # url is only ever truncated from the right, so the rest of the match
        # is a clean suffix — re-emitted verbatim so no source text is dropped.
        return f'<a href="{href}">{url}</a>{match.group(1)[len(url):]}'

    return _URL_RE.sub(replace, escaped_text)


_SVG_ELEMENT_RE = re.compile(r'<svg\b.*?</svg>', re.S)


def _svgs_referencing_outside_themselves(xhtml: str) -> list[str]:
    """Ids that a <use> names but that are not defined inside the same <svg>.

    Every equation SVG must be self-contained. dvisvgm emits each glyph as a
    <path id="…"> in that SVG's own <defs>, drawn by <use href="#…">, and a
    cross-<svg> reference is not resolved by Kindle — the glyph silently
    renders as nothing, leaving a partially drawn equation. epubcheck does not
    catch this (the ids do exist, just in the wrong element), which is how a
    file-wide glyph dedup shipped and broke the first equations of a chapter.

    Delegates to find_orphaned_use_refs (Stage 9) per <svg> block rather than
    keeping its own regex: an earlier version of this function used a
    hand-rolled pair that quietly diverged from Stage 8A's per-equation gate,
    which is how 9 equations in a 2205-equation book passed Stage 8A's check
    with a genuinely undefined glyph and were only caught here, after the
    whole book had already been assembled — see find_orphaned_use_refs.
    """
    orphans: list[str] = []
    for match in _SVG_ELEMENT_RE.finditer(xhtml):
        orphans.extend(sorted(find_orphaned_use_refs(match.group(0))))
    return orphans


# Elements XHTML defines that can legitimately appear inside a table. Anything
# outside this set is unwrapped rather than trusted — see _valid_table_html.
_TABLE_ALLOWED_TAGS = frozenset({
    "table", "thead", "tbody", "tfoot", "tr", "td", "th",
    "caption", "colgroup", "col",
    "a", "b", "br", "code", "em", "i", "p", "span", "strong", "sub", "sup",
})


def _valid_table_html(table_html: str) -> Optional[str]:
    """Returns MinerU's table HTML made safe to inline in XHTML, or None.

    Well-formed XML is NOT the same test as valid XHTML, and the difference is
    fatal. MinerU wraps inline equations inside table cells in a non-standard
    <eq> element:

        <td>Exterior p-Forms and Algebra in <eq>\\mathbb{R}^{n}</eq></td>

    That parses as XML, so an earlier version of this function passed it
    straight through, and epubcheck then rejected the entire book — for one
    such tag in one table of contents, out of a 1400-equation document.

    Unknown elements are unwrapped, keeping their text, rather than sent to the
    raster fallback: a table of contents that renders as searchable text with a
    bare LaTeX fragment in it is worth more than a picture of one.
    """
    try:
        from lxml import etree  # type: ignore
        root = etree.fromstring(table_html.encode())
    except Exception:
        return None
    if root.tag != "table":
        return None

    unknown = {
        el.tag for el in root.iter()
        if isinstance(el.tag, str) and el.tag not in _TABLE_ALLOWED_TAGS
    }
    if unknown:
        # strip_tags drops the elements but keeps their text, children and
        # tails, so cell contents survive intact.
        etree.strip_tags(root, *unknown)
    return etree.tostring(root, encoding="unicode")


def _document_to_chapters(
    document: Document,
    title: str,
    bus: EventBus,
    embedded_images: dict[str, bytes],
) -> list[tuple[Optional[str], str]]:
    """
    Assembles the whole document into section chapters:

    - one global reading-order stream across pages (text, figures, equations)
    - PDF sources: paragraphs that continue across a page boundary are merged
      (the previous page's last text unit lacks terminal punctuation), with
      end-of-line hyphenation removed; a new chapter starts at every heading
      (MinerU title block) — content before the first heading becomes the
      opening chapter. Both exist to reconstruct structure the lossy PDF
      text stream destroyed.
    - EPUB/HTML sources: no continuation merge (real <p> boundaries are
      already authoritative); a new chapter starts at every Page (one Page
      per spine item / HTML file — the EPUB's own chapter structure),
      titled from `Page.chapter_title`. Interior headings render inline
      as <h2> without starting a new chapter file.

    Returns [(chapter_title, xhtml)].
    """
    regions_by_id = {r.region_id: r for r in document.all_equations}
    consumed: set[str] = set()

    def _substitute_placeholder(match: "re.Match[str]") -> str:
        region_id = match.group(1)
        region = regions_by_id.get(region_id)
        consumed.add(region_id)
        if region is None:
            return ""
        html = _render_equation(region, embedded_images)
        if html is None:
            if region.flagged_for_review and not region.is_reference_label:
                bus.emit(
                    "s10_epub_assembly", "equation_skipped",
                    equation_id=region.region_id, reason="flagged_no_render",
                )
            return ""
        if region.formula_class == FormulaClass.DISPLAY:
            # A block element cannot nest inside <p>: split the paragraph.
            return f"</p>{html}<p>"
        return html

    # ── Pass 1: per-page reading-order units ────────────────────────────
    units: list[dict] = []
    # Footnotes are held out of the reading-order stream entirely and placed
    # by Pass 3 at the end of the section holding their reference. Leaving
    # them inline would drop a block between a page's last paragraph and the
    # next page's first, breaking the cross-page continuation merge below.
    footnotes: list[dict] = []
    for page in document.pages:
        page_items: list[tuple[float, float, dict]] = []

        for block in page.text_blocks:
            if not block.raw_text.strip():
                continue
            order = block.reading_order_index if block.reading_order_index is not None else block.bbox.y0
            if block.kind == "code":
                # Not prose: no equation placeholders were ever spliced into
                # code text (s02c_epub._emit_code_block bypasses the MathJax
                # substitution pass), and _linkify must not rewrite a bare
                # URL sitting in a comment or a shell "$VAR" into a <a>/eq
                # span. Escaping still applies — the code text is going
                # inside <pre><code>, so its own < > & need entities.
                inner = _escape_text(block.raw_text)
                tail = block.raw_text.rstrip()
            else:
                inner = _EQ_PLACEHOLDER_RE.sub(
                    _substitute_placeholder, _linkify(_escape_text(block.raw_text))
                )
                # Visible tail (placeholders stripped) decides merge behaviour.
                tail = _EQ_PLACEHOLDER_RE.sub("", block.raw_text).rstrip()
            if block.kind == "footnote":
                footnotes.append({
                    "footnote_id": block.footnote_id, "inner": inner,
                    "page": page.page_number,
                })
                continue
            page_items.append((order, block.bbox.y0, {
                "kind": block.kind, "inner": inner, "tail": tail,
                "page": page.page_number,
            }))

        for figure in page.figures:
            table_html = _valid_table_html(figure.table_html) if figure.table_html else None
            if figure.unsupported_label:
                fig_html = (
                    '<div class="unsupported-content">'
                    f'<p>{_escape_text(figure.unsupported_label)}</p>'
                    '</div>'
                )
            elif table_html is not None:
                # s02c_epub._substitute_table_math leaves [[EQ:region_id]]
                # placeholders as literal text inside cells (table_html is
                # raw markup, never routed through _escape_text/_linkify).
                # .replace() strips _substitute_placeholder's "</p><p>"
                # display-equation split — meaningless inside a <td>, and
                # would otherwise nest a paragraph break inside a table cell.
                table_html = _EQ_PLACEHOLDER_RE.sub(
                    lambda m: _substitute_placeholder(m).replace("</p>", "").replace("<p>", ""),
                    table_html,
                )
                fig_html = f'<div class="figure">{table_html}</div>'
            elif figure.image_bytes:
                embedded_images[figure.figure_id] = figure.image_bytes
                fig_html = (
                    f'<div class="figure"><img alt="{_escape_alt_text(figure.alt_text)}" '
                    f'src="../images/{figure.figure_id}.png"/></div>'
                )
            else:
                continue
            page_items.append((figure.reading_order_index, figure.bbox.y0, {
                "kind": "figure", "html": fig_html, "page": page.page_number,
            }))

        for region in page.equation_regions:
            if region.region_id in consumed:
                continue  # rendered inline within its paragraph
            html = _render_equation(region, embedded_images)
            if html is None:
                if region.flagged_for_review and not region.is_reference_label:
                    bus.emit(
                        "s10_epub_assembly", "equation_skipped",
                        equation_id=region.region_id, reason="flagged_no_render",
                    )
                continue
            order = region.reading_order_index if region.reading_order_index is not None else region.bbox.y0
            page_items.append((order, region.bbox.y0, {
                "kind": "equation", "html": html, "page": page.page_number,
            }))

        page_items.sort(key=lambda x: (x[0], x[1]))
        units.extend(item for _, _, item in page_items)

    # EPUB/HTML already has authoritative paragraph and chapter boundaries
    # (real <p> tags, one Page per spine item) — Pass 2's merge and Pass 3's
    # heading-triggered rechaptering both exist to reconstruct structure a
    # lossy PDF text stream destroyed, and applying them to already-correct
    # EPUB structure risks concatenating two legitimately separate
    # paragraphs or discarding the EPUB's own spine chaptering.
    is_reflow_source = document.metadata.source_type in (SourceType.EPUB, SourceType.HTML)

    # ── Pass 2: merge continuation paragraphs across page boundaries ────
    if is_reflow_source:
        merged = units
    else:
        merged = []
        for unit in units:
            prev = merged[-1] if merged else None
            # A paragraph continues across a column or page break when its
            # previous half does not end in terminal punctuation. The units are
            # already in reading order, so adjacency in the stream is the signal;
            # allow the break within the same page (column) or onto the next
            # page, but not larger jumps.
            if (
                prev is not None
                and unit.get("inner") is not None
                and prev.get("inner") is not None
                and prev["page"] <= unit["page"] <= prev["page"] + 1
                and unit["kind"] == prev["kind"]
                and prev["kind"] == "text"
                and prev["tail"]
                and not _TERMINAL_TAIL_RE.search(prev["tail"])
            ):
                if prev["inner"].rstrip().endswith("-"):
                    prev["inner"] = prev["inner"].rstrip()[:-1] + unit["inner"]
                else:
                    prev["inner"] = prev["inner"].rstrip() + " " + unit["inner"]
                prev["tail"] = unit["tail"]
                prev["page"] = unit["page"]
                continue
            merged.append(unit)

    # ── Pass 3: split into chapters — at page boundaries for EPUB/HTML (the
    # spine/file structure is already the chapter structure), at headings
    # for PDF (see is_reflow_source above) — footnotes placed at section end.
    # `title` is None for a continuation file — same section, split only
    # because it outgrew MAX_CHAPTER_BYTES, and so kept out of the TOC.
    page_titles: dict[int, Optional[str]] = {
        page.page_number: page.chapter_title for page in document.pages
    }
    chapters: list[tuple[Optional[str], list[str]]] = []
    current_title: Optional[str] = (
        page_titles.get(merged[0]["page"], title) if (is_reflow_source and merged) else title
    )
    current_page: Optional[int] = merged[0]["page"] if merged else None
    current_parts: list[str] = []
    current_bytes = 0
    pending_notes = list(footnotes)      # document order, drained as placed
    linked_note_ids = {
        r.footnote_ref_id for r in document.all_equations if r.footnote_ref_id
    }
    refs_in_chapter: set[str] = set()    # note ids whose marker appeared here
    max_page_in_chapter = 0
    # True right after a heading is emitted, until the next unit joins it —
    # see _emit's use of it below.
    last_was_heading = False

    def _place_notes(final: bool = False) -> list[str]:
        """Notes owed by the section just closed.

        A linked note goes with the section holding its reference marker, so
        the popup link and its back-link stay inside one file — it waits for
        that marker however many sections that takes. An unlinked note has no
        anchor to chase, so it waits until a section has moved past its page
        (strictly: contains content from a later one), which is the point its
        page is known to be finished.

        Placing a linked note early would strand it in a different file from
        its marker, which is a dangling fragment and an epubcheck error.
        """
        nonlocal pending_notes
        due, keep = [], []
        for note in pending_notes:
            linked = note["footnote_id"] in linked_note_ids
            owed = final or (
                note["footnote_id"] in refs_in_chapter if linked
                else note["page"] < max_page_in_chapter
            )
            (due if owed else keep).append(note)
        pending_notes = keep
        if not due:
            return []
        return [
            '<div class="footnotes"><p class="footnotes-title">Notes</p>'
            + "".join(
                _footnote_html(note, back_link=note["footnote_id"] in refs_in_chapter)
                for note in due
            )
            + "</div>"
        ]

    def _flush(final: bool = False, split: bool = False) -> None:
        """Closes the current XHTML file. `split` means the section continues
        in the next file, so its notes-so-far are settled here (their markers
        are in THIS file) and the next file carries no TOC entry."""
        nonlocal current_parts, current_bytes, refs_in_chapter
        nonlocal max_page_in_chapter, current_title, last_was_heading
        notes = _place_notes(final) if (current_parts or final) else []
        if current_parts or notes:
            chapters.append((current_title, current_parts + notes))
        if split:
            current_title = None
        current_parts = []
        current_bytes = 0
        refs_in_chapter = set()
        max_page_in_chapter = 0
        last_was_heading = False

    def _emit(part: str, page: int, is_heading: bool = False) -> None:
        nonlocal max_page_in_chapter, current_bytes, last_was_heading
        # Close the file BEFORE the unit that would overflow it, so the budget
        # is a real ceiling rather than one oversized equation past it. Breaks
        # only between units, so no paragraph or equation is ever cut.
        #
        # Exception: never split right after a heading. A heading with
        # nothing under it yet is not a real section boundary — the split
        # would leave a bare "<h2>Nondimensionalization</h2>" as the last
        # line of one file and the paragraph that explains it as the first
        # line of the next, which reading systems that start each XHTML
        # file on a fresh page render as an orphaned page containing only a
        # heading (found via a real Claude-artifact HTML page whose
        # "Nondimensionalization" subsection landed exactly there). Suppress
        # the check for one call so the heading's first content unit is
        # guaranteed to land in the same file, however large that makes it.
        if current_parts and not last_was_heading and current_bytes + len(part) > MAX_CHAPTER_BYTES:
            _flush(split=True)
        current_parts.append(part)
        current_bytes += len(part)
        max_page_in_chapter = max(max_page_in_chapter, page)
        refs_in_chapter.update(_NOTEREF_ID_RE.findall(part))
        last_was_heading = is_heading

    for unit in merged:
        if is_reflow_source and unit["page"] != current_page:
            # New spine item / HTML file — the EPUB's own chapter boundary,
            # not a heading. Byte-size overflow splitting inside one large
            # page (_emit's own _flush(split=True)) is unaffected by this.
            _flush()
            current_title = page_titles.get(unit["page"], current_title)
            current_page = unit["page"]
        if unit["kind"] == "heading":
            if not is_reflow_source:
                _flush()
                current_title = _EQ_PLACEHOLDER_RE.sub(
                    "", re.sub(r"<[^>]+>", "", unit["inner"])
                ).strip() or "Untitled section"
            # Headings hold at most inline math — strip any paragraph-split
            # artifacts a display placeholder would have produced.
            heading_inner = unit["inner"].replace("</p>", "").replace("<p>", "")
            _emit(f"<h2>{heading_inner}</h2>", unit["page"], is_heading=True)
        elif unit["kind"] == "code":
            _emit(f'<pre class="code"><code>{unit["inner"]}</code></pre>', unit["page"])
        elif unit.get("inner") is not None:
            _emit(_EMPTY_P_RE.sub("", f'<p>{unit["inner"]}</p>'), unit["page"])
        else:
            _emit(unit["html"], unit["page"])
    _flush(final=True)

    if not chapters:
        chapters = [(title, ["<p>&#160;</p>"])]

    return [
        (chapter_title, XHTML_TEMPLATE.format(
            title=_escape_text(chapter_title or title), body="\n".join(parts),
        ))
        for chapter_title, parts in chapters
    ]


def _escape_text(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# Amazon's "Enhanced Mobi" converter finds the end of an <img> tag by scanning
# for '>' without honouring quoted attribute values, so any '>' inside alt text
# truncates the tag and fails the entire book with E21018 -- the generic E999
# the Send to Kindle web form reports. Escaping does not help: verified against
# Kindle Previewer 3.107.0, all of '>', '&gt;' and '&#62;' fail, while '&lt;',
# '&amp;' and '&quot;' convert cleanly. So the character itself has to go.
# U+FF1E FULLWIDTH GREATER-THAN keeps the meaning legible to a screen reader.
_ALT_SAFE_GT = "\uff1e"


def _escape_alt_text(text: str) -> str:
    """Escape text for an <img alt="..."> value, dropping Kindle-fatal '>'."""
    return _escape_text(text.replace(">", _ALT_SAFE_GT))


def _build_opf(
    title: str,
    author: str,
    uid: str,
    chapters: list[str],
    image_ids: Optional[list[str]] = None,
) -> str:
    items = "\n    ".join(
        f'<item id="chapter{i+1}" href="content/chapter_{i+1:03d}.xhtml" '
        f'media-type="application/xhtml+xml"'
        # epubcheck OPF-014: items with embedded SVG must declare it
        + (' properties="svg"' if "<svg" in chapter else "")
        + "/>"
        for i, chapter in enumerate(chapters)
    )
    if image_ids:
        items += "\n    " + "\n    ".join(
            f'<item id="img_{img_id}" href="images/{img_id}.png" media-type="image/png"/>'
            for img_id in image_ids
        )
    itemrefs = "\n    ".join(
        f'<itemref idref="chapter{i+1}"/>'
        for i in range(len(chapters))
    )
    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{_escape_text(title)}</dc:title>
    <dc:creator>{_escape_text(author)}</dc:creator>
    <dc:language>en</dc:language>
    <dc:identifier id="uid">{uid}</dc:identifier>
    <meta property="dcterms:modified">{_iso_now()}</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="css" href="styles/book.css" media-type="text/css"/>
    {items}
  </manifest>
  <spine toc="ncx">
    {itemrefs}
  </spine>
</package>
"""


def _build_nav(title: str, chapter_titles: list[tuple[int, str]]) -> str:
    """chapter_titles is (file_index, title) — continuation files produced by
    the MAX_CHAPTER_BYTES split have no entry, so a section that spans several
    files still shows as one line in the TOC."""
    items = "\n    ".join(
        f'<li><a href="content/chapter_{i+1:03d}.xhtml">{_escape_text(ct)}</a></li>'
        for i, ct in chapter_titles
    )
    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en">
<head><meta charset="utf-8"/><title>Table of Contents</title></head>
<body>
<nav epub:type="toc">
  <h1>Table of Contents</h1>
  <ol>
    {items}
  </ol>
</nav>
</body>
</html>
"""


def _build_ncx(title: str, uid: str, chapter_titles: list[tuple[int, str]]) -> str:
    nav_points = "\n  ".join(
        f'<navPoint id="navPoint{n}" playOrder="{n}">'
        f'<navLabel><text>{_escape_text(ct)}</text></navLabel>'
        f'<content src="content/chapter_{i+1:03d}.xhtml"/></navPoint>'
        for n, (i, ct) in enumerate(chapter_titles, start=1)
    )
    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN"
    "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head>
  <meta name="dtb:uid" content="{uid}"/>
</head>
<docTitle><text>{_escape_text(title)}</text></docTitle>
<navMap>
  {nav_points}
</navMap>
</ncx>
"""


def _equation_count(xhtml: str) -> int:
    """Inline <svg> elements in one chapter — what Kindle's ceiling counts."""
    return xhtml.count("<svg")


def _write_epub_zip(
    path: Path,
    title: str,
    author: str,
    uid: str,
    chapters_xhtml: list[str],
    chapter_titles: list[tuple[int, str]],
    images: dict[str, bytes],
) -> None:
    """Write one complete, self-contained EPUB 3 container."""
    import zipfile

    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as zf:
        # mimetype must be first and uncompressed
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", """\
<?xml version="1.0" encoding="utf-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
""")
        zf.writestr("OEBPS/styles/book.css", BOOK_CSS)
        for i, xhtml in enumerate(chapters_xhtml):
            zf.writestr(f"OEBPS/content/chapter_{i+1:03d}.xhtml", xhtml)
        for img_id, png_bytes in images.items():
            zf.writestr(f"OEBPS/images/{img_id}.png", png_bytes)
        zf.writestr("OEBPS/nav.xhtml", _build_nav(title, chapter_titles))
        zf.writestr("OEBPS/toc.ncx", _build_ncx(title, uid, chapter_titles))
        zf.writestr(
            "OEBPS/content.opf",
            _build_opf(title, author, uid, chapters_xhtml, sorted(images)),
        )


def _partition_into_volumes(
    titled_chapters: list[tuple[Optional[str], str]],
    budget: int,
) -> list[list[int]]:
    """
    Group chapter indices into volumes of at most `budget` equations each.

    Cuts land only where a chapter carries a TOC title, so a section split across
    continuation files by MAX_CHAPTER_BYTES is never torn across two volumes.
    A single chapter over budget cannot be split further here and gets a volume
    of its own — the caller warns about it.
    """
    if budget <= 0:
        return [list(range(len(titled_chapters)))]

    volumes: list[list[int]] = []
    current: list[int] = []
    current_eq = 0
    for i, (title, xhtml) in enumerate(titled_chapters):
        eq = _equation_count(xhtml)
        can_cut = title is not None          # never cut before a continuation file
        if current and can_cut and current_eq + eq > budget:
            volumes.append(current)
            current, current_eq = [], 0
        current.append(i)
        current_eq += eq
    if current:
        volumes.append(current)
    return volumes


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# epubcheck's runtime scales with the book. 60 s is comfortable for a paper
# and far too short for a 500-page textbook, so it is derived from the EPUB's
# size with a generous floor.
_EPUBCHECK_BASE_TIMEOUT_S = 120
_EPUBCHECK_SECONDS_PER_MB = 12


def _epubcheck_timeout_s(epub_path: Path) -> int:
    try:
        size_mb = epub_path.stat().st_size / (1024 * 1024)
    except OSError:
        return _EPUBCHECK_BASE_TIMEOUT_S
    return int(_EPUBCHECK_BASE_TIMEOUT_S + size_mb * _EPUBCHECK_SECONDS_PER_MB)


def _run_epubcheck(epub_path: Path, bus: EventBus) -> list[str]:
    """Runs epubcheck. Returns list of error strings. Warnings are ignored.

    A timeout is reported as an error, not as a clean result. epubcheck is the
    pipeline's only automated correctness net — returning [] on timeout means
    "validated OK" to the caller, which is precisely the wrong answer for the
    large books most likely to time out.
    """
    import shutil

    # Prefer an epubcheck wrapper script on PATH (e.g. Homebrew's, which
    # bundles its own JAVA_HOME and works without a system JRE).
    cmd: list[str] | None = None
    wrapper = shutil.which("epubcheck")
    if wrapper:
        cmd = [wrapper, str(epub_path)]
    else:
        java = shutil.which("java")
        if not java:
            log.warning("epubcheck_skipped", reason="java_not_found")
            return []

        # Common epubcheck locations
        epubcheck_paths = [
            Path.home() / ".local" / "lib" / "epubcheck" / "epubcheck.jar",
            Path("/usr/local/lib/epubcheck/epubcheck.jar"),
            Path("/opt/epubcheck/epubcheck.jar"),
        ]
        jar = next((p for p in epubcheck_paths if p.exists()), None)
        if not jar:
            log.warning("epubcheck_skipped", reason="jar_not_found")
            return []
        cmd = [java, "-jar", str(jar), str(epub_path)]

    timeout_s = _epubcheck_timeout_s(epub_path)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
        )
        output = (result.stdout + result.stderr).decode(errors="replace")
        bus.emit("s10_epub_assembly", "epubcheck_output", payload={"output": output[:2000]})

        error_lines = [
            line for line in output.splitlines()
            if "ERROR" in line and "WARNING" not in line
        ]
        return error_lines
    except subprocess.TimeoutExpired:
        log.error("epubcheck_timeout", timeout_s=timeout_s)
        bus.emit("s10_epub_assembly", "epubcheck_timeout", payload={"timeout_s": timeout_s})
        return [
            f"epubcheck did not finish within {timeout_s}s — the EPUB is "
            f"UNVALIDATED. Re-run with --no-epubcheck to skip validation "
            f"deliberately, or validate manually: epubcheck '{epub_path}'"
        ]


def run(
    document: Document,
    output_path: Path,
    bus: EventBus,
    epubcheck_enabled: bool = True,
    max_equations_per_volume: Optional[int] = KINDLE_MAX_EQUATIONS_PER_VOLUME,
) -> tuple[Path, StageResult]:
    """
    Assemble the document into an EPUB at `output_path`.

    When the book carries more than `max_equations_per_volume` inline <svg>
    elements it is split into `_vol01`, `_vol02`, … files alongside
    `output_path`, because Amazon's Send to Kindle converter rejects an
    over-budget book outright with an undiagnosable E999. Pass None to disable
    splitting. The returned path is the first volume; every volume written is
    listed in `StageResult.metrics["volume_paths"]`.
    """
    import os

    t0 = time.perf_counter()
    stage = "s10_epub_assembly"
    warnings: list[str] = []
    errors: list[str] = []

    bus.emit(stage, "stage_start")
    log.info("stage_start", output_path=str(output_path))

    try:
        title = document.metadata.title or Path(document.metadata.source_path).stem
        author = document.metadata.author or "Unknown"
        uid = str(uuid.uuid4())

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Assemble section chapters from the whole document (cross-page
        # paragraph flow, headings). Figure crops and flagged-equation
        # fallbacks are collected as embedded images.
        fallback_images: dict[str, bytes] = {}
        titled_chapters = _document_to_chapters(document, title, bus, fallback_images)
        # Continuation files (title None) stay in the manifest and spine but out
        # of the TOC — see MAX_CHAPTER_BYTES. Per-volume TOC entries are built
        # from titled_chapters below, once the volume grouping is known.
        chapters_xhtml = [x for _, x in titled_chapters]

        # Guard the two silent-corruption classes epubcheck cannot see: a
        # glyph referenced across <svg> elements, and a file over Kindle's
        # 300 KB ceiling. Both render as missing content on device only.
        for i, xhtml in enumerate(chapters_xhtml):
            orphans = _svgs_referencing_outside_themselves(xhtml)
            if orphans:
                msg = (f"chapter_{i+1:03d}: {len(orphans)} glyph(s) referenced "
                       f"across <svg> elements, will not render on Kindle "
                       f"(first: {orphans[:3]})")
                errors.append(msg)
                log.error("svg_cross_reference", chapter=i + 1, count=len(orphans))
            size = len(xhtml.encode("utf-8"))
            if size > KINDLE_MAX_FILE_BYTES:
                warnings.append(
                    f"chapter_{i+1:03d}: {size/1024:.0f} KB exceeds Kindle's "
                    f"{KINDLE_MAX_FILE_BYTES//1024} KB per-file limit"
                )
                log.warning("chapter_oversized", chapter=i + 1, bytes=size)
        if errors:
            return output_path, StageResult(
                stage_name=stage, ok=False,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                warnings=warnings, errors=errors,
            )

        # Kindle rejects a whole book whose inline <svg> count is too high, so
        # split into volumes when over budget. Cuts land on TOC-titled chapters.
        total_equations = sum(_equation_count(x) for x in chapters_xhtml)
        budget = max_equations_per_volume or 0
        groups = (
            _partition_into_volumes(titled_chapters, budget)
            if budget and total_equations > budget
            else [list(range(len(chapters_xhtml)))]
        )

        volume_paths: list[Path] = []
        for vol_no, indices in enumerate(groups, 1):
            single = len(groups) == 1
            vol_title = title if single else f"{title} — Volume {vol_no} of {len(groups)}"
            vol_path = output_path if single else output_path.with_name(
                f"{output_path.stem}_vol{vol_no:02d}{output_path.suffix}"
            )
            vol_chapters = [chapters_xhtml[i] for i in indices]
            # renumber TOC entries against this volume's own chapter ordering
            pos = {orig: new for new, orig in enumerate(indices)}
            vol_titles: list[tuple[int, str]] = [
                (pos[i], t) for i in indices
                if (t := titled_chapters[i][0]) is not None
            ]
            # carry only the images this volume's chapters actually reference
            vol_images = {
                img_id: png for img_id, png in fallback_images.items()
                if any(f"{img_id}.png" in x for x in vol_chapters)
            }
            _write_epub_zip(
                vol_path, vol_title, author,
                uid if single else str(uuid.uuid4()),
                vol_chapters, vol_titles, vol_images,
            )
            volume_paths.append(vol_path)

            vol_eq = sum(_equation_count(x) for x in vol_chapters)
            if budget and vol_eq > budget:
                # one indivisible chapter over budget — cannot split further here
                warnings.append(
                    f"volume {vol_no}: {vol_eq} equations exceeds the "
                    f"{budget}-equation Kindle budget and could not be split further"
                )
                log.warning("volume_over_budget", volume=vol_no, equations=vol_eq)

        if len(groups) > 1:
            warnings.append(
                f"{total_equations} equations exceeds the {budget}-equation Kindle "
                f"limit; split into {len(groups)} volumes"
            )
            log.info("split_into_volumes", volumes=len(groups),
                     equations=total_equations, budget=budget)
            bus.emit(stage, "volumes_split",
                     volumes=len(groups), equations=total_equations)

        # Run epubcheck on every volume — a split that broke one manifest must
        # not slip through because volume 1 happened to be clean.
        if epubcheck_enabled:
            epubcheck_errors = [
                err for vol_path in volume_paths
                for err in _run_epubcheck(vol_path, bus)
            ]
            if epubcheck_errors:
                for err in epubcheck_errors:
                    errors.append(err)
                    bus.emit(stage, "equation_error", equation_id=None, error=err)
                # epubcheck errors are fatal per spec
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                errors.insert(0, ErrorCode.EPUBCHECK_FAILED.value)
                return output_path, StageResult(
                    stage_name=stage,
                    ok=False,
                    duration_ms=duration_ms,
                    warnings=warnings,
                    errors=errors,
                )

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        bus.emit(stage, "stage_end", chapters=len(chapters_xhtml))
        log.info("stage_end", output_path=str(output_path),
                 chapters=len(chapters_xhtml), volumes=len(volume_paths))

        return volume_paths[0], StageResult(
            stage_name=stage,
            ok=True,
            duration_ms=duration_ms,
            warnings=warnings,
            errors=errors,
            metrics={
                "chapters": len(chapters_xhtml),
                "equations": total_equations,
                "volumes": len(volume_paths),
                "volume_paths": [str(p) for p in volume_paths],
            },
        )

    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        errors.append(ErrorCode.EPUB_ASSEMBLY_FAILED.value)
        errors.append(str(exc))
        bus.emit(stage, "stage_end_error", error=str(exc))
        log.error("stage_end", status="error", error=str(exc))
        return output_path, StageResult(
            stage_name=stage,
            ok=False,
            duration_ms=duration_ms,
            errors=errors,
        )

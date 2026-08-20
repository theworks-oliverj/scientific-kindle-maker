"""
Stage 2C — EPUB/HTML Parse

Walks each spine document (EPUB) or the single input file (bare HTML) in DOM
order, producing the same shapes Stage 3 (MinerU) produces for PDF sources:
TextBlocks (prose/headings/lists/footnotes) with `[[EQ:region_id]]`
placeholders spliced in at each inline equation, EquationRegions (from
<math> and equation-like <img>), and FigureBlocks (other <img>, <table>).

Equations are left as raw MathML ("MATHML:...") for Stage 5C to convert —
unchanged contract with s05c_mathml.py.
"""
import posixpath
import re
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

from ..equation_filters import eq_placeholder
from ..models.document import (
    BoundingBox,
    Document,
    DocumentMetadata,
    EquationRegion,
    FigureBlock,
    Page,
    TextBlock,
)
from ..models.enums import ColumnLayout, ErrorCode, FormulaClass, SourceType
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s02c_epub")

STAGE = "s02c_epub"

_EPUB_OPS_NS = "http://www.idpf.org/2007/ops"

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Elements whose content becomes one TextBlock each.
_TEXT_BLOCK_TAGS = {"p", "li", "dd", "dt"} | _HEADING_TAGS
_LIST_TAGS = {"ul", "ol", "dl"}

LATEX_HINT_PATTERN = re.compile(r'[\\^_]|\\[a-zA-Z]+')
MATH_CLASS_PATTERN = re.compile(r'\bmath\b', re.IGNORECASE)

# MathJax/KaTeX-style delimited LaTeX in plain text — a different, very
# common convention from MathML: the <math> element need not exist at all,
# since MathJax scans ordinary text nodes for these delimiters client-side.
# Found by testing against a real HTML artifact with no <math> tags anywhere
# but 36 $$...$$ display blocks and inline $...$ throughout. Display forms
# (\$\$/\\\[) are tried before the single-\$ inline form so a "$$" is never
# misread as two adjacent inline spans.
_MATHJAX_RE = re.compile(
    r'\$\$(?P<disp1>.+?)\$\$'
    r'|\\\[(?P<disp2>.+?)\\\]'
    r'|\\\((?P<inline2>.+?)\\\)'
    r'|\$(?P<inline1>[^$\n]+?)\$',
    re.DOTALL,
)


def _is_equation_img(alt: str, classes: list[str]) -> bool:
    """Heuristic: alt text containing LaTeX characters suggests an equation image."""
    if LATEX_HINT_PATTERN.search(alt):
        return True
    return any(MATH_CLASS_PATTERN.search(c) for c in classes)


def _local(tag) -> str:
    """Strips a namespace off an lxml tag: '{ns}p' -> 'p'."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1] if tag.startswith("{") else tag


def _epub_type(elem) -> str:
    return (elem.get(f"{{{_EPUB_OPS_NS}}}type") or elem.get("epub:type") or "").strip()


def _to_png_bytes(raw: Optional[bytes]) -> Optional[bytes]:
    """Re-encodes arbitrary image bytes as PNG.

    s10_epub_assembly writes every figure/equation-image crop out as
    "{id}.png" with media-type="image/png" unconditionally — a contract the
    PDF path always satisfies (its crops are cut from a PNG page raster).
    EPUB source images are whatever the publisher shipped, commonly JPEG,
    and passing those bytes through verbatim under a .png name/media-type
    is an EPUBCheck OPF-029 error (caught by running epubcheck against a
    real book's output during verification, not anticipated up front).
    """
    if not raw:
        return None
    try:
        import io
        from PIL import Image  # type: ignore
        img = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _table_html_no_ns(elem) -> Optional[str]:
    """Serializes a <table> element without the XHTML namespace declaration.

    `s10_epub_assembly._valid_table_html` requires `root.tag == "table"`
    (bare, no namespace) — a contract written for MinerU's own table_html,
    which never carries one. A table parsed out of a real XHTML document
    carries `xmlns="...1999/xhtml"`, which would otherwise fail that check
    for every EPUB table.

    Round-tripping through lxml.html does NOT strip this on its own — HTML
    has no namespace concept, so `xmlns` just arrives as an ordinary literal
    attribute on the root and is preserved as such (caught by
    test_s02c_epub.test_table_captured_without_xhtml_namespace, which failed
    on the first version of this function that assumed the round-trip alone
    was enough). It has to be removed explicitly.
    """
    try:
        from lxml import etree  # type: ignore
        from lxml import html as lxml_html  # type: ignore
        raw = etree.tostring(elem, encoding="unicode")
        html_root = lxml_html.fromstring(raw)
        html_root.attrib.pop("xmlns", None)
        return lxml_html.tostring(html_root, encoding="unicode")
    except Exception:
        return None


class _ChapterWalker:
    """Walks one document's <body> in DOM order onto `page`.

    `image_resolver(src) -> bytes | None` fetches the real bytes for an
    <img> src — different for EPUB (package lookup) and bare HTML (relative
    file read), so it is injected rather than hardcoded here.
    """

    def __init__(self, page: Page, chapter_num: int, image_resolver: Callable[[str], Optional[bytes]]):
        self.page = page
        self.chapter_num = chapter_num
        self.image_resolver = image_resolver
        self._eq_index = 0
        self._order = 0.0
        self.title_hint: Optional[str] = None

    def _next_order(self) -> float:
        self._order += 1
        return self._order

    def _new_bbox(self, order: float) -> BoundingBox:
        # No real page geometry exists for reflowable HTML. `order` doubles
        # as both reading_order_index and the bbox.y0 tiebreak s10 sorts on,
        # so placement is stable and matches DOM order exactly.
        return BoundingBox(
            x0=0, y0=order, x1=100, y1=20,
            coordinate_system="epub_flow", page_number=self.page.page_number,
        )

    # ── equations ─────────────────────────────────────────────────────

    def _make_equation_region(self, math_el, order: float) -> EquationRegion:
        from lxml import etree  # type: ignore

        self._eq_index += 1
        region_id = f"eq_{self.chapter_num}_{self._eq_index}"
        display = math_el.get("display", "inline")
        formula_class = FormulaClass.DISPLAY if display == "block" else FormulaClass.INLINE

        # Kept as raw MathML (not derived to LaTeX here) even when the
        # element carries a <semantics><annotation> with authoritative TeX —
        # s05c_mathml.py checks for that first, but still routes the result
        # through its simple_text_repr/D1-dedup pass, which annotation-first
        # extraction done here would bypass entirely.
        mathml_str = etree.tostring(math_el, encoding="unicode")

        region = EquationRegion(
            region_id=region_id,
            bbox=self._new_bbox(order),
            formula_class=formula_class,
            source_image_crop=None,
            raw_latex=f"MATHML:{mathml_str}",
            normalized_latex=None,
            cdm_score=None,
            confidence_gate=None,
            svg=None,
            svg_postprocessed=None,
            equation_number=None,
            reading_order_index=order,
        )
        self.page.equation_regions.append(region)
        return region

    def _make_equation_img_region(self, img_el, order: float, src: str) -> EquationRegion:
        self._eq_index += 1
        region_id = f"eq_{self.chapter_num}_{self._eq_index}"
        alt = img_el.get("alt", "")
        region = EquationRegion(
            region_id=region_id,
            bbox=self._new_bbox(order),
            formula_class=FormulaClass.INLINE,
            source_image_crop=self.image_resolver(src),
            raw_latex=alt if alt else None,
            normalized_latex=None,
            cdm_score=None,
            confidence_gate=None,
            svg=None,
            svg_postprocessed=None,
            equation_number=None,
            reading_order_index=order,
        )
        self.page.equation_regions.append(region)
        return region

    def _make_mathjax_region(self, latex: str, display: bool, order: float) -> EquationRegion:
        """A MathJax-delimited span ($...$, $$...$$, \\(...\\), \\[...\\]).

        Unlike <math> (raw_latex="MATHML:..."), this is already real LaTeX —
        exactly what MathJax itself would compile client-side — so it is
        stored directly, without the "MATHML:" prefix. s05c_mathml.py's
        filter only touches regions whose raw_latex starts with that
        prefix, so these skip MathML conversion entirely and go straight to
        s06 validation, same as a PDF-sourced equation. That also means they
        skip s05c's text_fallback assignment (it only runs over MATHML:-
        prefixed regions) — set directly here instead, or a flagged MathJax
        equation would be dropped silently (caught by testing against a
        real Claude-artifact HTML page with no <math> tags at all: the
        first version of this method left text_fallback unset, and 20/126
        flagged equations vanished with zero trace before this line existed).
        """
        self._eq_index += 1
        region_id = f"eq_{self.chapter_num}_{self._eq_index}"
        region = EquationRegion(
            region_id=region_id,
            bbox=self._new_bbox(order),
            formula_class=FormulaClass.DISPLAY if display else FormulaClass.INLINE,
            source_image_crop=None,
            raw_latex=latex.strip(),
            normalized_latex=None,
            cdm_score=None,
            confidence_gate=None,
            svg=None,
            svg_postprocessed=None,
            equation_number=None,
            reading_order_index=order,
            text_fallback=latex.strip(),
        )
        self.page.equation_regions.append(region)
        return region

    def _substitute_mathjax(self, text: str) -> str:
        """Replaces every $...$/$$...$$/\\(...\\)/\\[...\\] span in `text`
        with an equation placeholder. Safe to run on text that already
        contains [[EQ:...]] placeholders from <math>/<img> handling — none
        of those characters are '$' or '\\', so they pass through untouched."""
        def replace(m: "re.Match[str]") -> str:
            if m.group("disp1") is not None:
                region = self._make_mathjax_region(m.group("disp1"), True, self._next_order())
            elif m.group("disp2") is not None:
                region = self._make_mathjax_region(m.group("disp2"), True, self._next_order())
            elif m.group("inline2") is not None:
                region = self._make_mathjax_region(m.group("inline2"), False, self._next_order())
            else:
                region = self._make_mathjax_region(m.group("inline1"), False, self._next_order())
            return eq_placeholder(region.region_id)

        return _MATHJAX_RE.sub(replace, text)

    def _make_noteref_marker(self, order: float, marker_text: str, footnote_id: str) -> EquationRegion:
        """A footnote reference marker ("1", "*") rendered as a linked
        text span — the exact mechanism `s10_epub_assembly._render_equation`
        already implements for MinerU's OCR'd superscript markers
        (render_as_text + footnote_ref_id). Reused as-is: no new s10 markup
        needed for EPUB's real <a epub:type="noteref"> anchors.
        """
        self._eq_index += 1
        region_id = f"eq_{self.chapter_num}_{self._eq_index}"
        region = EquationRegion(
            region_id=region_id,
            bbox=self._new_bbox(order),
            formula_class=FormulaClass.INLINE,
            source_image_crop=None,
            raw_latex=None,
            normalized_latex=None,
            cdm_score=None,
            confidence_gate=None,
            svg=None,
            svg_postprocessed=None,
            equation_number=None,
            reading_order_index=order,
            render_as_text=True,
            inline_text_repr=marker_text or "*",
            footnote_ref_id=footnote_id,
        )
        self.page.equation_regions.append(region)
        return region

    # ── inline content (within one text block) ──────────────────────

    def _inline_text(self, elem) -> str:
        """Concatenates text content of `elem`, splicing in placeholders for
        <math>/equation-<img>/footnote noterefs. Other inline markup (a, b,
        i, em, strong, code, sup, sub, span…) is flattened to its text —
        s10 re-escapes and re-wraps raw_text itself, so embedding real HTML
        here would be double-escaped."""
        parts: list[str] = [elem.text or ""]
        for child in elem:
            tag = _local(child.tag)
            if tag == "math":
                region = self._make_equation_region(child, self._next_order())
                parts.append(eq_placeholder(region.region_id))
            elif tag == "img":
                src = child.get("src", "")
                classes = (child.get("class") or "").split()
                if _is_equation_img(child.get("alt", ""), classes):
                    region = self._make_equation_img_region(child, self._next_order(), src)
                    parts.append(eq_placeholder(region.region_id))
                # A real figure image inline in running text has no clean
                # placement as a TextBlock placeholder — dropped here, same
                # as any other unsupported inline element; standalone
                # figures (the common case) are handled in the block walker.
            elif tag == "a" and _epub_type(child) == "noteref":
                href = (child.get("href") or "").lstrip("#")
                marker_text = "".join(child.itertext()).strip()
                if href:
                    region = self._make_noteref_marker(self._next_order(), marker_text, href)
                    parts.append(eq_placeholder(region.region_id))
                else:
                    parts.append(marker_text)
            else:
                parts.append(self._inline_text(child))
            parts.append(child.tail or "")
        return "".join(parts)

    def _emit_text_block(self, elem, kind: str, footnote_id: Optional[str] = None) -> None:
        order = self._next_order()
        raw_text = self._substitute_mathjax(self._inline_text(elem)).strip()
        if not raw_text:
            return
        self.page.text_blocks.append(
            TextBlock(
                block_id=f"tb_{self.chapter_num}_{len(self.page.text_blocks) + 1}",
                bbox=self._new_bbox(order),
                raw_text=raw_text,
                reading_order_index=order,
                kind=kind,
                footnote_id=footnote_id,
            )
        )
        if kind == "heading" and self.title_hint is None:
            self.title_hint = re.sub(r'\[\[EQ:[A-Za-z0-9_]+\]\]', '', raw_text).strip() or None

    # ── block-level walk ──────────────────────────────────────────────

    def walk_body(self, body) -> None:
        for child in body:
            self._walk_block(child)

    def _walk_block(self, elem) -> None:
        tag = _local(elem.tag)

        if tag in _TEXT_BLOCK_TAGS:
            self._emit_text_block(elem, kind="heading" if tag in _HEADING_TAGS else (
                "list_item" if tag in ("li", "dd") else "text"
            ))
            return

        if tag in _LIST_TAGS:
            for item in elem:
                if _local(item.tag) in ("li", "dt", "dd"):
                    self._emit_text_block(item, kind="list_item")
            return

        if tag == "table":
            table_html = _table_html_no_ns(elem)
            if table_html:
                order = self._next_order()
                self.page.figures.append(FigureBlock(
                    figure_id=f"fig_{self.chapter_num}_{len(self.page.figures) + 1}",
                    bbox=self._new_bbox(order),
                    image_bytes=None,
                    alt_text="table",
                    reading_order_index=order,
                    table_html=table_html,
                ))
            return

        if tag == "img":
            self._walk_standalone_img(elem)
            return

        if tag == "math":
            order = self._next_order()
            region = self._make_equation_region(elem, order)
            # Wrap in a single-placeholder text block so it flows through
            # the same substitution path as every other equation — s10 has
            # no separate "bare display equation" placement branch.
            self.page.text_blocks.append(TextBlock(
                block_id=f"tb_{self.chapter_num}_{len(self.page.text_blocks) + 1}",
                bbox=self._new_bbox(order),
                raw_text=eq_placeholder(region.region_id),
                reading_order_index=order,
                kind="text",
            ))
            return

        if tag == "aside" and _epub_type(elem) in ("footnote", "rearnote", "endnote"):
            fn_id = elem.get("id")
            # A footnote body is usually one or more <p>; harvest each as a
            # kind="footnote" TextBlock. s10's Pass 1 already pulls any
            # kind=="footnote" block out of the normal reading-order stream
            # and places it at section end — no different handling needed
            # here versus a plain paragraph.
            body_paras = [c for c in elem if _local(c.tag) == "p"]
            targets = body_paras or [elem]
            for i, para in enumerate(targets):
                self._emit_text_block(
                    para, kind="footnote",
                    footnote_id=fn_id if i == 0 else None,
                )
            return

        # A container whose subtree has a <canvas> but no <p> is treated as
        # one interactive widget (controls + canvas + legend, typically) and
        # swallowed as a single placeholder rather than walked for individual
        # text fragments — recursing into it would scatter slider labels and
        # legend text as disconnected floating prose. Requiring "no <p>"
        # keeps this from swallowing a <section> that mixes real prose
        # paragraphs with an interactive panel as siblings — found by
        # testing against a real Claude-artifact HTML page with 3 canvas
        # simulations, none of which have a nearby static representation.
        descendants = list(elem.iter())
        if (any(_local(d.tag) == "canvas" for d in descendants)
                and not any(_local(d.tag) == "p" for d in descendants)):
            self._emit_unsupported_content(elem)
            return

        # Generic container (div, section, figure, article, body itself,
        # blockquote…) — recurse to find block content inside it.
        before = (len(self.page.text_blocks), len(self.page.equation_regions), len(self.page.figures))
        for child in elem:
            self._walk_block(child)
        after = (len(self.page.text_blocks), len(self.page.equation_regions), len(self.page.figures))
        # Nothing was extracted from any child element — e.g. a <div> whose
        # entire content is a bare text node, not wrapped in <p>. This is
        # exactly how "$$...$$" display blocks are commonly marked up (found
        # via a real page's `<div class="defn">$$...$$</div>`, no child
        # elements at all). Fall back to the container's own text rather
        # than silently losing it.
        if after == before and (elem.text or "").strip():
            self._emit_text_block(elem, kind="text")

    def _emit_unsupported_content(self, elem) -> None:
        order = self._next_order()
        self.page.figures.append(FigureBlock(
            figure_id=f"fig_{self.chapter_num}_{len(self.page.figures) + 1}",
            bbox=self._new_bbox(order),
            image_bytes=None,
            alt_text="unsupported interactive content",
            reading_order_index=order,
            unsupported_label=self._find_interactive_label(elem),
        ))

    def _find_interactive_label(self, elem) -> str:
        """Best-effort description for an unsupported-content placeholder:
        the nearest heading/caption-like descendant's text, if any."""
        for d in elem.iter():
            tag = _local(d.tag)
            cls = (d.get("class") or "").lower()
            if tag in _HEADING_TAGS or any(k in cls for k in ("heading", "title", "caption", "label")):
                text = " ".join("".join(d.itertext()).split())
                if text:
                    return f'Interactive content not shown — "{text[:80]}"'
        return "Interactive content not shown — this format does not support animations or scripted content."

    def _walk_standalone_img(self, elem) -> None:
        src = elem.get("src", "")
        classes = (elem.get("class") or "").split()
        if _is_equation_img(elem.get("alt", ""), classes):
            order = self._next_order()
            region = self._make_equation_img_region(elem, order, src)
            self.page.text_blocks.append(TextBlock(
                block_id=f"tb_{self.chapter_num}_{len(self.page.text_blocks) + 1}",
                bbox=self._new_bbox(order),
                raw_text=eq_placeholder(region.region_id),
                reading_order_index=order,
                kind="text",
            ))
            return
        order = self._next_order()
        self.page.figures.append(FigureBlock(
            figure_id=f"fig_{self.chapter_num}_{len(self.page.figures) + 1}",
            bbox=self._new_bbox(order),
            image_bytes=self.image_resolver(src),
            alt_text=elem.get("alt") or f"figure on {self.page.page_number}",
            reading_order_index=order,
        ))


def _parse_root(content: bytes) -> tuple[object, bool]:
    """Returns (root_element, used_html_fallback). XML parsing is tried
    first — real publisher EPUBs declare MathML's namespace directly on
    <math> and parse fine as XML. The lxml.html fallback is namespace-blind
    (it cannot see <math> at all), so falling back loses equation detection
    silently unless the caller logs `used_html_fallback`."""
    from lxml import etree  # type: ignore

    try:
        return etree.fromstring(content), False
    except etree.XMLSyntaxError:
        from lxml import html as lxml_html  # type: ignore
        return lxml_html.fromstring(content), True


def _find_body(root):
    for el in root.iter():
        if _local(el.tag) == "body":
            return el
    return root


def _find_title_from_head(root) -> Optional[str]:
    for el in root.iter():
        if _local(el.tag) == "title" and (el.text or "").strip():
            return el.text.strip()
    return None


def _walk_one_chapter(content: bytes, chapter_num: int, image_resolver) -> tuple[Page, Optional[str], bool]:
    root, used_html_fallback = _parse_root(content)
    body = _find_body(root)
    page = Page(
        page_number=chapter_num,
        width_pt=595.0,
        height_pt=842.0,
        column_layout=ColumnLayout.SINGLE,
        image_bytes=None,
    )
    walker = _ChapterWalker(page, chapter_num, image_resolver)
    walker.walk_body(body)
    title = _find_title_from_head(root) or walker.title_hint
    return page, title, used_html_fallback


def _run_epub(metadata: DocumentMetadata, bus: EventBus, warnings: list[str]) -> Document:
    from ebooklib import epub

    if not zipfile.is_zipfile(metadata.source_path):
        raise ValueError("not a valid EPUB (zip) container")

    book = epub.read_epub(metadata.source_path)
    document = Document(metadata=metadata)

    # book.spine (not get_items(ITEM_DOCUMENT)) is the book's own reading
    # order and — confirmed against a real publisher EPUB — correctly
    # excludes nav.xhtml, which get_items() includes as if it were a
    # chapter (it is a document-type item, just not a spine one). Entries
    # marked linear="no" (e.g. a nav page some books do keep in spine,
    # precisely so it's reachable but excluded from the reading order) are
    # skipped for the same reason.
    spine_items = []
    for spine_id, linear in book.spine:
        if linear == "no":
            continue
        item = book.get_item_with_id(spine_id)
        if item is not None:
            spine_items.append(item)

    def make_resolver(chapter_href: str) -> Callable[[str], Optional[bytes]]:
        chapter_dir = posixpath.dirname(chapter_href)

        def resolve(src: str) -> Optional[bytes]:
            if not src:
                return None
            resolved = posixpath.normpath(posixpath.join(chapter_dir, src))
            item = book.get_item_with_href(resolved)
            return _to_png_bytes(item.get_content()) if item else None

        return resolve

    for chapter_num, item in enumerate(spine_items, start=1):
        page, title, used_fallback = _walk_one_chapter(
            item.get_content(), chapter_num, make_resolver(item.file_name),
        )
        if used_fallback:
            warnings.append(f"{item.file_name}: fell back to HTML parsing — MathML in this file will not be detected")
            log.warning("xml_parse_fallback", file=item.file_name)
        page.chapter_title = title
        document.pages.append(page)

    return document


def _run_html(metadata: DocumentMetadata, bus: EventBus, warnings: list[str]) -> Document:
    path = Path(metadata.source_path)
    content = path.read_bytes()

    def resolve(src: str) -> Optional[bytes]:
        if not src or src.startswith(("http://", "https://", "data:")):
            return None
        try:
            return _to_png_bytes((path.parent / src).read_bytes())
        except OSError:
            return None

    document = Document(metadata=metadata)
    page, title, used_fallback = _walk_one_chapter(content, 1, resolve)
    if used_fallback:
        warnings.append(f"{path.name}: fell back to HTML parsing — MathML in this file will not be detected")
        log.warning("xml_parse_fallback", file=path.name)
    page.chapter_title = title or path.stem
    document.pages.append(page)
    return document


def run(
    metadata: DocumentMetadata,
    bus: EventBus,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []

    bus.emit(STAGE, "stage_start")
    log.info("stage_start", path=metadata.source_path)

    try:
        if metadata.source_type == SourceType.EPUB:
            document = _run_epub(metadata, bus, warnings)
        else:
            document = _run_html(metadata, bus, warnings)

        document.metadata.page_count = len(document.pages)

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        total_eq = document.total_equation_count
        bus.emit(STAGE, "stage_end", chapters=len(document.pages), equations_found=total_eq)
        log.info("stage_end", chapters=len(document.pages), equations_found=total_eq)

        return document, StageResult(
            stage_name=STAGE,
            ok=True,
            duration_ms=duration_ms,
            warnings=warnings,
            errors=errors,
            metrics={"chapters": len(document.pages), "equations_found": total_eq},
        )

    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        errors.append(ErrorCode.EPUB_PARSE_FAILED.value)
        errors.append(str(exc))
        bus.emit(STAGE, "stage_end_error", error=str(exc))
        log.error("stage_end", status="error", error=str(exc))
        return Document(metadata=metadata), StageResult(
            stage_name=STAGE,
            ok=False,
            duration_ms=duration_ms,
            errors=errors,
        )

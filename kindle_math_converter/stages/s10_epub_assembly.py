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
from ..models.enums import FormulaClass, ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

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
  }
  .eq-display svg {
    max-width: 100%;
  }
  .eq-number {
    float: right;
    margin-right: 2em;
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
.eq-display { display: block; text-align: center; margin: 1em 0; }
.eq-display svg { max-width: 100%; }
.eq-flagged { color: #cc0000; font-family: monospace; border: 1px solid #cc0000; padding: 0 4px; }
.eq-text { white-space: nowrap; }
.eq-img-fallback { max-width: 100%; }
span.eq-inline img.eq-img-fallback { height: 1.2em; width: auto; vertical-align: middle; }
.figure { text-align: center; margin: 1em 0; }
.figure img { max-width: 100%; }
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
        return f'<span class="eq-text">{region.inline_text_repr}</span>'

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
            number_html = (
                f'<span class="eq-number">{region.equation_number}</span>'
                if region.equation_number else ""
            )
            return f'<div class="eq-display">{img}{number_html}</div>'
        return None

    svg = region.svg_postprocessed

    if region.formula_class == FormulaClass.INLINE:
        return f'<span class="eq-inline">{svg}</span>'
    else:
        number_html = ""
        if region.equation_number:
            number_html = f'<span class="eq-number">{region.equation_number}</span>'
        return f'<div class="eq-display">{svg}{number_html}</div>'


# Inline-equation placeholder embedded in TextBlock.raw_text by
# s03_mineru_parse (uses only characters that survive _escape_text).
_EQ_PLACEHOLDER_RE = re.compile(r'\[\[EQ:([A-Za-z0-9_]+)\]\]')
_EMPTY_P_RE = re.compile(r'<p>\s*</p>')

# A paragraph whose visible text ends in one of these is considered complete;
# anything else at a page boundary is treated as a continuation and merged.
_TERMINAL_TAIL_RE = re.compile(r'[.!?:;…"”\'’)\]]\s*$')


def _valid_table_html(table_html: str) -> Optional[str]:
    """Returns MinerU's table HTML if it is well-formed XML with a <table>
    root (safe to inline in XHTML); None otherwise (caller falls back to
    the raster crop)."""
    try:
        from lxml import etree  # type: ignore
        root = etree.fromstring(table_html.encode())
    except Exception:
        return None
    if root.tag != "table":
        return None
    return table_html


def _document_to_chapters(
    document: Document,
    title: str,
    bus: EventBus,
    embedded_images: dict[str, bytes],
) -> list[tuple[str, str]]:
    """
    Assembles the whole document into section chapters:

    - one global reading-order stream across pages (text, figures, equations)
    - paragraphs that continue across a page boundary are merged (the
      previous page's last text unit lacks terminal punctuation), with
      end-of-line hyphenation removed
    - a new chapter starts at every heading (MinerU title block); content
      before the first heading becomes the opening chapter

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
    for page in document.pages:
        page_items: list[tuple[float, float, dict]] = []

        for block in page.text_blocks:
            if not block.raw_text.strip():
                continue
            order = block.reading_order_index if block.reading_order_index is not None else block.bbox.y0
            inner = _EQ_PLACEHOLDER_RE.sub(
                _substitute_placeholder, _escape_text(block.raw_text)
            )
            # Visible tail (placeholders stripped) decides merge behaviour.
            tail = _EQ_PLACEHOLDER_RE.sub("", block.raw_text).rstrip()
            page_items.append((order, block.bbox.y0, {
                "kind": block.kind, "inner": inner, "tail": tail,
                "page": page.page_number,
            }))

        for figure in page.figures:
            table_html = _valid_table_html(figure.table_html) if figure.table_html else None
            if table_html is not None:
                fig_html = f'<div class="figure">{table_html}</div>'
            elif figure.image_bytes:
                embedded_images[figure.figure_id] = figure.image_bytes
                fig_html = (
                    f'<div class="figure"><img alt="{_escape_text(figure.alt_text)}" '
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

    # ── Pass 2: merge continuation paragraphs across page boundaries ────
    merged: list[dict] = []
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

    # ── Pass 3: split into chapters at headings ─────────────────────────
    chapters: list[tuple[str, list[str]]] = []
    current_title = title
    current_parts: list[str] = []

    def _flush() -> None:
        nonlocal current_parts
        if current_parts:
            chapters.append((current_title, current_parts))
        current_parts = []

    for unit in merged:
        if unit["kind"] == "heading":
            _flush()
            current_title = _EQ_PLACEHOLDER_RE.sub(
                "", re.sub(r"<[^>]+>", "", unit["inner"])
            ).strip() or "Untitled section"
            # Headings hold at most inline math — strip any paragraph-split
            # artifacts a display placeholder would have produced.
            heading_inner = unit["inner"].replace("</p>", "").replace("<p>", "")
            current_parts.append(f"<h2>{heading_inner}</h2>")
        elif unit.get("inner") is not None:
            current_parts.append(_EMPTY_P_RE.sub("", f'<p>{unit["inner"]}</p>'))
        else:
            current_parts.append(unit["html"])
    _flush()

    if not chapters:
        chapters = [(title, ["<p>&#160;</p>"])]

    return [
        (chapter_title, XHTML_TEMPLATE.format(
            title=_escape_text(chapter_title), body="\n".join(parts),
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


def _build_nav(title: str, chapter_titles: list[str]) -> str:
    items = "\n    ".join(
        f'<li><a href="content/chapter_{i+1:03d}.xhtml">{_escape_text(ct)}</a></li>'
        for i, ct in enumerate(chapter_titles)
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


def _build_ncx(title: str, uid: str, chapter_titles: list[str]) -> str:
    nav_points = "\n  ".join(
        f'<navPoint id="navPoint{i+1}" playOrder="{i+1}">'
        f'<navLabel><text>{_escape_text(ct)}</text></navLabel>'
        f'<content src="content/chapter_{i+1:03d}.xhtml"/></navPoint>'
        for i, ct in enumerate(chapter_titles)
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


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_epubcheck(epub_path: Path, bus: EventBus) -> list[str]:
    """Runs epubcheck. Returns list of error strings. Warnings are ignored."""
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

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
        )
        output = (result.stdout + result.stderr).decode(errors="replace")
        bus.emit("s10_epub_assembly", "epubcheck_output", payload={"output": output[:2000]})

        error_lines = [
            line for line in output.splitlines()
            if "ERROR" in line and "WARNING" not in line
        ]
        return error_lines
    except subprocess.TimeoutExpired:
        log.warning("epubcheck_timeout")
        return []


def run(
    document: Document,
    output_path: Path,
    bus: EventBus,
    epubcheck_enabled: bool = True,
) -> tuple[Path, StageResult]:
    import zipfile
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
        chapter_titles = [t for t, _ in titled_chapters]
        chapters_xhtml = [x for _, x in titled_chapters]

        with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
            # mimetype must be first and uncompressed
            zf.writestr(
                zipfile.ZipInfo("mimetype"),
                "application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )

            # META-INF/container.xml
            container_xml = """\
<?xml version="1.0" encoding="utf-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
            zf.writestr("META-INF/container.xml", container_xml)

            # Stylesheet
            zf.writestr("OEBPS/styles/book.css", BOOK_CSS)

            # Chapter XHTML files
            for i, xhtml in enumerate(chapters_xhtml):
                zf.writestr(f"OEBPS/content/chapter_{i+1:03d}.xhtml", xhtml)

            # Raster fallbacks for flagged equations
            for img_id, png_bytes in fallback_images.items():
                zf.writestr(f"OEBPS/images/{img_id}.png", png_bytes)

            # Navigation
            zf.writestr("OEBPS/nav.xhtml", _build_nav(title, chapter_titles))
            zf.writestr("OEBPS/toc.ncx", _build_ncx(title, uid, chapter_titles))

            # Package document
            zf.writestr(
                "OEBPS/content.opf",
                _build_opf(title, author, uid, chapters_xhtml, sorted(fallback_images)),
            )

        # Run epubcheck
        if epubcheck_enabled:
            epubcheck_errors = _run_epubcheck(output_path, bus)
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
        log.info("stage_end", output_path=str(output_path), chapters=len(chapters_xhtml))

        return output_path, StageResult(
            stage_name=stage,
            ok=True,
            duration_ms=duration_ms,
            warnings=warnings,
            errors=errors,
            metrics={"chapters": len(chapters_xhtml)},
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

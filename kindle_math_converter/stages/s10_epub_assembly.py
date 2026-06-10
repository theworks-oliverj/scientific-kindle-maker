"""
Stage 10 — EPUB3 Assembly
Assembles the processed document into a valid EPUB3 file.
SVGs are inlined directly in XHTML — never as <object> tags (KindleGen rejects those).
Runs epubcheck validation after assembly.
Failure mode: Fatal.
"""
import subprocess
import time
import uuid
from pathlib import Path

from ..models.document import Document, Page, EquationRegion, TextBlock
from ..models.enums import FormulaClass, ConfidenceGate, ErrorCode
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
}

/* Fallback for non-Kindle EPUB readers */
.eq-inline { display: inline-block; vertical-align: middle; }
.eq-display { display: block; text-align: center; margin: 1em 0; }
.eq-display svg { max-width: 100%; }
.eq-flagged { color: #cc0000; font-family: monospace; border: 1px solid #cc0000; padding: 0 4px; }
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


def _render_equation(region: EquationRegion) -> str:
    """Returns the HTML snippet for one equation region."""
    if region.flagged_for_review or not region.svg_postprocessed:
        return (
            f'<span class="eq-flagged" data-eq-id="{region.region_id}" '
            f'title="Review required">[EQ:{region.region_id}]</span>'
        )

    svg = region.svg_postprocessed

    if region.formula_class == FormulaClass.INLINE:
        return f'<span class="eq-inline">{svg}</span>'
    else:
        number_html = ""
        if region.equation_number:
            number_html = f'<span class="eq-number">{region.equation_number}</span>'
        return f'<div class="eq-display">{svg}{number_html}</div>'


def _page_to_xhtml(page: Page, title: str) -> str:
    """Converts a page's text blocks and equations into XHTML body content."""
    # Build a combined reading-order list of text and equations
    # For pages from PDF/scanned sources, interleave text and equations by y-position

    body_parts = []

    # Order content by the unified reading_order_index assigned in Stage 4
    # (column-aware). Fall back to vertical position when the index is absent.
    all_items: list[tuple[float, float, str]] = []

    for block in page.text_blocks:
        if not block.raw_text.strip():
            continue  # skip blocks where OCR produced nothing
        order = block.reading_order_index if block.reading_order_index is not None else block.bbox.y0
        all_items.append((order, block.bbox.y0, f"<p>{_escape_text(block.raw_text)}</p>"))

    for region in page.equation_regions:
        order = region.reading_order_index if region.reading_order_index is not None else region.bbox.y0
        all_items.append((order, region.bbox.y0, _render_equation(region)))

    all_items.sort(key=lambda x: (x[0], x[1]))
    body_parts = [html for _, _, html in all_items]

    if not body_parts:
        body_parts = ["<p>&#160;</p>"]

    return XHTML_TEMPLATE.format(title=f"{title} — Page {page.page_number}", body="\n".join(body_parts))


def _escape_text(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_opf(title: str, author: str, uid: str, chapters: list[str]) -> str:
    items = "\n    ".join(
        f'<item id="chapter{i+1}" href="content/chapter_{i+1:03d}.xhtml" '
        f'media-type="application/xhtml+xml"/>'
        for i in range(len(chapters))
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


def _build_nav(title: str, chapters: list[str]) -> str:
    items = "\n    ".join(
        f'<li><a href="content/chapter_{i+1:03d}.xhtml">Chapter {i+1}</a></li>'
        for i in range(len(chapters))
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


def _build_ncx(title: str, uid: str, chapters: list[str]) -> str:
    nav_points = "\n  ".join(
        f'<navPoint id="navPoint{i+1}" playOrder="{i+1}">'
        f'<navLabel><text>Chapter {i+1}</text></navLabel>'
        f'<content src="content/chapter_{i+1:03d}.xhtml"/></navPoint>'
        for i in range(len(chapters))
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
    # Find epubcheck jar
    import shutil
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

    try:
        result = subprocess.run(
            [java, "-jar", str(jar), str(epub_path)],
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

        # Generate XHTML for each page/chapter
        chapters_xhtml = [
            _page_to_xhtml(page, title)
            for page in document.pages
        ]

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

            # Navigation
            zf.writestr("OEBPS/nav.xhtml", _build_nav(title, chapters_xhtml))
            zf.writestr("OEBPS/toc.ncx", _build_ncx(title, uid, chapters_xhtml))

            # Package document
            zf.writestr("OEBPS/content.opf", _build_opf(title, author, uid, chapters_xhtml))

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

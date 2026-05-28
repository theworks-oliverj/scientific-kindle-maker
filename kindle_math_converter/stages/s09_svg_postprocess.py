"""
Stage 9 — SVG Post-Processing for Kindle
Applies all known Kindle/KDP rendering bug mitigations to each SVG.

Fixes applied in order:
  1. Strip font-size from CSS (KDP px→rem corruption, defensive check)
  2. Convert width/height from pt to em (scales with Kindle font size)
  3. Replace hardcoded black fill/stroke with currentColor (dark mode)
  4. Remove XML declaration and namespace clutter (required for inline HTML)
"""
import re
import time

from ..models.document import Document, EquationRegion
from ..models.enums import ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s09_svg_postprocess")


def strip_font_size_css(svg: str) -> str:
    """
    KDP converts all px font-size values to rem, breaking SVG text.
    With --no-fonts this should be unnecessary, but apply defensively.
    """
    svg = re.sub(r'font-size\s*:\s*[\d.]+px\s*;?', '', svg)
    svg = re.sub(r'\bfont-size="[\d.]+(?:px)?"\s*', '', svg)
    return svg


def set_em_dimensions(svg: str, body_font_size_pt: float = 10.0) -> str:
    """
    Converts the root SVG element's width and height from pt to em units.
    Also sets viewBox and vertical-align for inline equations.

    body_font_size_pt: base font size of the document body in points.
    Formula: width_em = width_pt / body_font_size_pt
    """
    from lxml import etree  # type: ignore

    try:
        root = etree.fromstring(svg.encode())
    except etree.XMLSyntaxError:
        return svg

    raw_width = root.get("width", "0")
    raw_height = root.get("height", "0")

    width_pt = float(re.sub(r'[^0-9.]', '', raw_width) or "0")
    height_pt = float(re.sub(r'[^0-9.]', '', raw_height) or "0")

    width_em  = round(width_pt  / body_font_size_pt, 4)
    height_em = round(height_pt / body_font_size_pt, 4)

    # Set viewBox if not present
    if not root.get("viewBox") and width_pt > 0 and height_pt > 0:
        root.set("viewBox", f"0 0 {width_pt} {height_pt}")

    root.set("width",  f"{width_em}em")
    root.set("height", f"{height_em}em")
    root.set("style", "vertical-align: middle;")

    return etree.tostring(root, encoding="unicode")


def apply_current_color(svg: str) -> str:
    """
    Replaces hardcoded black fill/stroke with currentColor for dark mode.
    Only replaces black — colored elements are left alone.
    """
    svg = svg.replace('fill="black"',     'fill="currentColor"')
    svg = svg.replace('fill="#000000"',   'fill="currentColor"')
    svg = svg.replace('fill="#000"',      'fill="currentColor"')
    svg = svg.replace('stroke="black"',   'stroke="currentColor"')
    svg = svg.replace('stroke="#000000"', 'stroke="currentColor"')
    svg = svg.replace('stroke="#000"',    'stroke="currentColor"')
    return svg


def clean_for_inline_embedding(svg: str) -> str:
    """
    Removes <?xml ...?> declaration and redundant namespace declarations
    that are invalid inside HTML5 documents.
    Keeps xmlns="http://www.w3.org/2000/svg" on the root element only.

    dvisvgm emits xlink:href="..." on <use> elements (SVG 1.1 style).
    We rewrite those to plain href="..." (SVG2) BEFORE stripping the
    xmlns:xlink declaration — otherwise we produce invalid XML with
    un-declared namespace prefixes.
    """
    svg = re.sub(r'<\?xml[^?]*\?>', '', svg)
    # SVG2: replace xlink:href with href on any element
    svg = svg.replace('xlink:href=', 'href=')
    # Also rewrite xlink:show and xlink:type if dvisvgm ever emits them
    svg = svg.replace('xlink:show=', 'show=')
    svg = svg.replace('xlink:type=', 'type=')
    # Now safe to drop the namespace declaration (match both quote styles)
    svg = re.sub(r"""\s+xmlns:xlink=["'][^"']*["']""", '', svg)
    return svg.strip()


def postprocess_svg(svg: str, body_font_size_pt: float = 10.0) -> str:
    svg = strip_font_size_css(svg)
    svg = set_em_dimensions(svg, body_font_size_pt)
    svg = apply_current_color(svg)
    svg = clean_for_inline_embedding(svg)
    return svg


def run(
    document: Document,
    bus: EventBus,
    body_font_size_pt: float = 10.0,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    stage = "s09_svg_postprocess"
    warnings: list[str] = []
    errors: list[str] = []
    processed = 0
    failed = 0

    bus.emit(stage, "stage_start")
    log.info("stage_start", body_font_size_pt=body_font_size_pt)

    for region in document.all_equations:
        if not region.svg:
            continue

        try:
            region.svg_postprocessed = postprocess_svg(region.svg, body_font_size_pt)
            processed += 1
            bus.emit(stage, "equation_ok", equation_id=region.region_id)
        except Exception as exc:
            region.error_codes.append(ErrorCode.SVG_POSTPROCESS_FAILED.value)
            warnings.append(f"{region.region_id}: post-process failed — {exc}")
            # Fall back to raw SVG if post-processing fails
            region.svg_postprocessed = region.svg
            failed += 1
            bus.emit(stage, "equation_warning", equation_id=region.region_id, error=str(exc))

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    bus.emit(stage, "stage_end", processed=processed, failed=failed)
    log.info("stage_end", processed=processed, failed=failed)

    return document, StageResult(
        stage_name=stage,
        ok=True,
        duration_ms=duration_ms,
        warnings=warnings,
        errors=errors,
        metrics={"processed": processed, "failed": failed},
    )

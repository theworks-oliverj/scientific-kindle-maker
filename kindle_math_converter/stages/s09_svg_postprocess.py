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
# The pt→em divisor must be the size the wrapper actually rendered at.
from .s06_validation import LATEX_BODY_PT

log = get_logger("s09_svg_postprocess")


def strip_font_size_css(svg: str) -> str:
    """
    KDP converts all px font-size values to rem, breaking SVG text.
    With --no-fonts this should be unnecessary, but apply defensively.
    """
    svg = re.sub(r'font-size\s*:\s*[\d.]+px\s*;?', '', svg)
    svg = re.sub(r'\bfont-size="[\d.]+(?:px)?"\s*', '', svg)
    return svg


def set_em_dimensions(svg: str, body_font_size_pt: float = LATEX_BODY_PT) -> str:
    """
    Converts the root SVG element's width and height from pt to em units.
    Also sets viewBox and vertical-align for inline equations.

    body_font_size_pt: base font size of the document body in points — the
    size the wrapper rendered at, so the result is 1em per body-text line.
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


# Quote- and prefix-agnostic on purpose: dvisvgm's raw output uses single
# quotes and xlink:href (e.g. <use xlink:href='#g1-82'/>), while by the time
# an SVG reaches namespace_svg_ids it has been through set_em_dimensions'
# lxml round-trip (which normalises to double quotes) and
# clean_for_inline_embedding (xlink:href -> href). Matching both forms lets
# find_orphaned_use_refs be called at either point without silently finding
# nothing because it was checking for a form that has not appeared yet.
_ID_RE = re.compile(r'''\bid=["']([^"']+)["']''')
_HREF_RE = re.compile(r'''(?:xlink:)?href=["']#([^"']+)["']''')


def find_orphaned_use_refs(svg: str) -> set[str]:
    """Ids that a href="#..." names but that no id="..." in `svg` declares.

    dvisvgm's contract is that every glyph a <use> draws is defined in that
    same render's own <defs> — but it does not always hold it. Found on a
    271-page book: a handful of equations came back from dvisvgm with a
    `<use xlink:href='#g1-82'/>` and no matching `<path id='g1-82'>`
    anywhere in that equation's own SVG — the glyph was simply never
    emitted. A DIFFERENT, unrelated equation happened to define its own
    glyph under the same locally-numbered id ("gN-MM" is font-index +
    glyph-index local to one dvisvgm invocation, not a real cross-reference),
    which is how this reads as a real id if you only check "does this string
    appear somewhere in the document" rather than "is it defined in THIS
    equation's own render."

    `namespace_svg_ids` cannot catch this: it only rewrites ids it can find
    declared, so an orphaned reference passes through it untouched and
    reaches the assembled book still broken. Call this on dvisvgm's raw
    output, before it is accepted, so the equation can fall back instead.
    """
    defined = set(_ID_RE.findall(svg))
    used = set(_HREF_RE.findall(svg))
    return used - defined


def namespace_svg_ids(svg: str, prefix: str) -> str:
    """
    Makes every glyph id in this SVG unique to one equation by prefixing it.

    dvisvgm numbers glyph definitions (<path id="gN-MM">) locally per render, so
    when many equation SVGs are inlined into one XHTML document the same id="gN-MM"
    repeats — an xs:ID uniqueness violation (epubcheck error; some readers reject the
    file) and a <use href="#gN-MM"> that resolves to the wrong glyph.

    We rewrite both the id declarations and the internal href="#..." references
    consistently. Only internal fragment refs (href="#X") are touched — dvisvgm emits
    nothing else. Must run AFTER clean_for_inline_embedding (which has already converted
    xlink:href -> href), so only the plain href= form remains.

    prefix is the region_id (e.g. "eq_3_36"), already an XML-safe NCName.
    """
    # Collect declared ids
    ids = re.findall(r'\bid="([^"]+)"', svg)
    for old_id in set(ids):
        new_id = f"{prefix}__{old_id}"
        # Rewrite the declaration and any internal reference to it.
        svg = re.sub(rf'\bid="{re.escape(old_id)}"', f'id="{new_id}"', svg)
        svg = re.sub(rf'href="#{re.escape(old_id)}"', f'href="#{new_id}"', svg)
    return svg


def postprocess_svg(svg: str, body_font_size_pt: float = LATEX_BODY_PT, prefix: str | None = None) -> str:
    svg = strip_font_size_css(svg)
    svg = set_em_dimensions(svg, body_font_size_pt)
    svg = apply_current_color(svg)
    svg = clean_for_inline_embedding(svg)
    if prefix:
        svg = namespace_svg_ids(svg, prefix)
    return svg


def run(
    document: Document,
    bus: EventBus,
    body_font_size_pt: float = LATEX_BODY_PT,
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
            region.svg_postprocessed = postprocess_svg(region.svg, body_font_size_pt, region.region_id)
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

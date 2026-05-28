"""
Stage 2C — EPUB/HTML Parse
Uses ebooklib to unpack the EPUB. Parses each XHTML document in spine order.
Finds MathML, equation images, and publisher math conventions.
"""
import io
import time
import re
from pathlib import Path

from ..models.document import Document, DocumentMetadata, Page, TextBlock, BoundingBox, EquationRegion
from ..models.enums import ColumnLayout, FormulaClass, ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s02c_epub")

LATEX_HINT_PATTERN = re.compile(r'[\\^_]|\\[a-zA-Z]+')
MATH_CLASS_PATTERN = re.compile(r'\bmath\b', re.IGNORECASE)


def _is_equation_img(alt: str, classes: list[str]) -> bool:
    """Heuristic: alt text containing LaTeX characters suggests an equation image."""
    if LATEX_HINT_PATTERN.search(alt):
        return True
    return any(MATH_CLASS_PATTERN.search(c) for c in classes)


def run(
    metadata: DocumentMetadata,
    bus: EventBus,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    stage = "s02c_epub"
    warnings: list[str] = []
    errors: list[str] = []

    bus.emit(stage, "stage_start")
    log.info("stage_start", path=metadata.source_path)

    try:
        import ebooklib  # type: ignore
        from ebooklib import epub
        from lxml import etree  # type: ignore

        book = epub.read_epub(metadata.source_path)
        document = Document(metadata=metadata)

        spine_items = [
            item for item in book.get_items()
            if item.get_type() == ebooklib.ITEM_DOCUMENT
        ]

        NS = {
            "xhtml": "http://www.w3.org/1999/xhtml",
            "math": "http://www.w3.org/1998/Math/MathML",
        }

        for chapter_num, item in enumerate(spine_items):
            content = item.get_content()
            try:
                root = etree.fromstring(content)
            except etree.XMLSyntaxError:
                # Try as HTML
                from lxml import html as lxml_html  # type: ignore
                root = lxml_html.fromstring(content)

            page = Page(
                page_number=chapter_num + 1,
                width_pt=595.0,   # A4 width in pt — placeholder for EPUB
                height_pt=842.0,
                column_layout=ColumnLayout.SINGLE,
                image_bytes=None,
            )

            eq_index = 0

            # Find MathML elements — skip detection, go directly to Stage 5B
            for math_el in root.iter("{http://www.w3.org/1998/Math/MathML}math"):
                mathml_str = etree.tostring(math_el, encoding="unicode")
                display = math_el.get("display", "inline")
                formula_class = FormulaClass.DISPLAY if display == "block" else FormulaClass.INLINE

                region = EquationRegion(
                    region_id=f"eq_{chapter_num + 1}_{eq_index}",
                    bbox=BoundingBox(
                        x0=0, y0=0, x1=100, y1=20,
                        coordinate_system="epub_placeholder",
                        page_number=chapter_num + 1,
                    ),
                    formula_class=formula_class,
                    source_image_crop=None,
                    raw_latex=None,
                    normalized_latex=None,
                    cdm_score=None,
                    confidence_gate=None,
                    svg=None,
                    svg_postprocessed=None,
                    equation_number=None,
                )
                # Store MathML as raw_latex placeholder for Stage 5B conversion
                region.raw_latex = f"MATHML:{mathml_str}"
                page.equation_regions.append(region)
                eq_index += 1

            # Find equation images by heuristic
            for img_el in root.iter():
                tag = img_el.tag
                if isinstance(tag, str) and tag.lower().endswith("}img") or tag == "img":
                    alt = img_el.get("alt", "")
                    classes = img_el.get("class", "").split()
                    if _is_equation_img(alt, classes):
                        region = EquationRegion(
                            region_id=f"eq_{chapter_num + 1}_{eq_index}",
                            bbox=BoundingBox(
                                x0=0, y0=0, x1=100, y1=20,
                                coordinate_system="epub_placeholder",
                                page_number=chapter_num + 1,
                            ),
                            formula_class=FormulaClass.INLINE,
                            source_image_crop=None,
                            raw_latex=alt if alt else None,
                            normalized_latex=None,
                            cdm_score=None,
                            confidence_gate=None,
                            svg=None,
                            svg_postprocessed=None,
                            equation_number=None,
                        )
                        page.equation_regions.append(region)
                        eq_index += 1

            document.pages.append(page)

        # Update page count in metadata
        document.metadata.page_count = len(document.pages)

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        total_eq = document.total_equation_count
        bus.emit(stage, "stage_end", chapters=len(document.pages), equations_found=total_eq)
        log.info("stage_end", chapters=len(document.pages), equations_found=total_eq)

        return document, StageResult(
            stage_name=stage,
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
        bus.emit(stage, "stage_end_error", error=str(exc))
        log.error("stage_end", status="error", error=str(exc))
        return Document(metadata=metadata), StageResult(
            stage_name=stage,
            ok=False,
            duration_ms=duration_ms,
            errors=errors,
        )

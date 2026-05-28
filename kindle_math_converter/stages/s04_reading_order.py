"""
Stage 4 — Reading Order Reconstruction
Assigns reading_order_index to all content regions.
Handles single-column and two-column layouts.
Also extracts equation numbering if present.
"""
import re
import time

from ..models.document import Document, TextBlock, EquationRegion
from ..models.enums import ColumnLayout, ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s04_reading_order")

EQ_NUMBER_PATTERN = re.compile(r'\((\d+(?:\.\d+)*)\)|\[([A-Z]\.\d+)\]')


def _extract_equation_number(text: str) -> str | None:
    """Extracts equation number like (3.14) or [A.1] from nearby text."""
    m = EQ_NUMBER_PATTERN.search(text)
    if m:
        return m.group(0)
    return None


def run(
    document: Document,
    bus: EventBus,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    stage = "s04_reading_order"
    warnings: list[str] = []
    errors: list[str] = []
    ambiguous_count = 0

    bus.emit(stage, "stage_start")
    log.info("stage_start", pages=len(document.pages))

    try:
        for page in document.pages:
            layout = page.column_layout
            all_regions: list[TextBlock | EquationRegion] = list(page.text_blocks) + list(page.equation_regions)

            if not all_regions:
                continue

            if layout == ColumnLayout.SINGLE or layout == ColumnLayout.MIXED:
                sorted_regions = sorted(all_regions, key=lambda r: (r.bbox.y0, r.bbox.x0))
                for idx, region in enumerate(sorted_regions):
                    if isinstance(region, TextBlock):
                        region.reading_order_index = idx
                    # EquationRegion does not have reading_order_index field;
                    # position is implicit from page order + equation_regions list order

            elif layout == ColumnLayout.DOUBLE:
                page_mid = page.width_pt / 2
                left_col = []
                right_col = []
                spanning = []

                for region in all_regions:
                    center_x = (region.bbox.x0 + region.bbox.x1) / 2
                    crosses_boundary = region.bbox.x0 < page_mid and region.bbox.x1 > page_mid
                    if crosses_boundary:
                        spanning.append(region)
                        warnings.append(
                            f"Page {page.page_number}: region spans column boundary"
                        )
                        ambiguous_count += 1
                    elif center_x < page_mid:
                        left_col.append(region)
                    else:
                        right_col.append(region)

                left_col.sort(key=lambda r: r.bbox.y0)
                right_col.sort(key=lambda r: r.bbox.y0)

                ordered = left_col + right_col + spanning
                for idx, region in enumerate(ordered):
                    if isinstance(region, TextBlock):
                        region.reading_order_index = idx

            # Assign reading_order_index to equation_regions in page order
            sorted_eq = sorted(page.equation_regions, key=lambda r: (r.bbox.y0, r.bbox.x0))
            page.equation_regions = sorted_eq

            # Extract equation numbers from nearby text blocks
            _assign_equation_numbers(page.equation_regions, page.text_blocks, page.width_pt)

        if ambiguous_count > 0:
            bus.emit(stage, "equation_warning", ambiguous_regions=ambiguous_count)

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        bus.emit(stage, "stage_end", ambiguous_regions=ambiguous_count)
        log.info("stage_end", ambiguous_count=ambiguous_count)

        return document, StageResult(
            stage_name=stage,
            ok=True,
            duration_ms=duration_ms,
            warnings=warnings,
            errors=errors,
            metrics={"ambiguous_regions": ambiguous_count},
        )

    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        errors.append(str(exc))
        bus.emit(stage, "stage_end_error", error=str(exc))
        log.error("stage_end", status="error", error=str(exc))
        return document, StageResult(
            stage_name=stage,
            ok=False,
            duration_ms=duration_ms,
            errors=errors,
        )


def _assign_equation_numbers(
    equation_regions: list[EquationRegion],
    text_blocks: list[TextBlock],
    page_width_pt: float,
    search_distance_pt: float = 80.0,
) -> None:
    """
    Looks for equation number text to the right of each display equation.
    If a text block like (3.14) is within search_distance_pt to the right,
    stores it in region.equation_number.
    """
    for region in equation_regions:
        for block in text_blocks:
            # Must be to the right of the equation and at similar vertical position
            horiz_ok = (
                block.bbox.x0 > region.bbox.x1
                and block.bbox.x0 - region.bbox.x1 < search_distance_pt
            )
            vert_ok = (
                abs((block.bbox.y0 + block.bbox.y1) / 2 - (region.bbox.y0 + region.bbox.y1) / 2)
                < region.bbox.height
            )
            if horiz_ok and vert_ok:
                num = _extract_equation_number(block.raw_text)
                if num:
                    region.equation_number = num
                    break

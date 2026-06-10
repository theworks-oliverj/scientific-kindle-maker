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

            if layout == ColumnLayout.DOUBLE:
                ordered, n_spanning = _order_double_column(all_regions, page.width_pt)
                ambiguous_count += n_spanning
                if n_spanning:
                    warnings.append(
                        f"Page {page.page_number}: {n_spanning} region(s) span the column boundary"
                    )
            else:  # SINGLE or MIXED
                ordered = sorted(all_regions, key=lambda r: (r.bbox.y0, r.bbox.x0))

            # Assign a unified reading order to BOTH text blocks and equations so
            # Stage 10 can interleave them correctly (esp. double-column pages).
            for idx, region in enumerate(ordered):
                region.reading_order_index = idx

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


def _order_double_column(
    regions: list[TextBlock | EquationRegion],
    page_width_pt: float,
) -> tuple[list[TextBlock | EquationRegion], int]:
    """
    Orders a double-column page using band segmentation.

    A full-width region (one that crosses the column midline, e.g. a display
    equation spanning both columns) acts as a horizontal band break: everything
    above it is read left-column-then-right-column, then the full-width region is
    emitted at its vertical position, then the next band begins. This keeps a
    centered display equation in its correct reading position instead of being
    dumped at the end of the page (the previous behaviour).

    Returns (ordered_regions, spanning_count).
    """
    mid = page_width_pt / 2
    ordered: list[TextBlock | EquationRegion] = []
    left: list[TextBlock | EquationRegion] = []
    right: list[TextBlock | EquationRegion] = []
    spanning_count = 0

    def flush_band() -> None:
        left.sort(key=lambda r: r.bbox.y0)
        right.sort(key=lambda r: r.bbox.y0)
        ordered.extend(left)
        ordered.extend(right)
        left.clear()
        right.clear()

    for region in sorted(regions, key=lambda r: r.bbox.y0):
        crosses = region.bbox.x0 < mid and region.bbox.x1 > mid
        if crosses:
            flush_band()
            ordered.append(region)
            spanning_count += 1
        elif (region.bbox.x0 + region.bbox.x1) / 2 < mid:
            left.append(region)
        else:
            right.append(region)

    flush_band()
    return ordered, spanning_count


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

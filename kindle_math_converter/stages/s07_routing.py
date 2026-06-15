"""
Stage 7 — Confidence Routing
Purely a partition: splits equations into pass_list and fallback_list
based on confidence_gate set in Stage 6.
"""
import time

from ..models.document import Document, EquationRegion
from ..models.enums import ConfidenceGate
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s07_routing")


def run(
    document: Document,
    bus: EventBus,
) -> tuple[Document, list[EquationRegion], list[EquationRegion], StageResult]:
    """
    Returns (document, pass_list, fallback_list, stage_result).
    pass_list:     gate == PASS — go to Stage 8A
    fallback_list: gate == FALLBACK | FLAGGED — go to Stage 8B
    """
    t0 = time.perf_counter()
    stage = "s07_routing"

    bus.emit(stage, "stage_start")
    log.info("stage_start", total_equations=document.total_equation_count)

    pass_list: list[EquationRegion] = []
    fallback_list: list[EquationRegion] = []

    for region in document.all_equations:
        # Handled outside the SVG pipeline (Stage 5B) — never reach 8A/8B
        if region.render_as_text or region.is_reference_label or region.dedup_canonical_id is not None:
            continue

        gate = region.confidence_gate
        if gate == ConfidenceGate.PASS or gate == ConfidenceGate.REPAIR:
            pass_list.append(region)
        else:
            fallback_list.append(region)

    bus.emit(
        stage,
        "stage_end",
        pass_count=len(pass_list),
        fallback_count=len(fallback_list),
    )
    log.info("stage_end", pass_count=len(pass_list), fallback_count=len(fallback_list))

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    return document, pass_list, fallback_list, StageResult(
        stage_name=stage,
        ok=True,
        duration_ms=duration_ms,
        metrics={
            "pass_count": len(pass_list),
            "fallback_count": len(fallback_list),
        },
    )

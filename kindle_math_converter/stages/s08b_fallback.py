# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""
Stage 8B — Fallback Handling
For equations that failed Stage 8A or Stage 6.
Path 1: Mathpix API (if configured)
Path 2: Flag for human review
"""
import time
from typing import Optional

from ..models.document import EquationRegion
from ..models.enums import ConfidenceGate, ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s08b_fallback")


def _mathpix_svg(
    image_bytes: bytes,
    app_id: str,
    app_key: str,
) -> str | None:
    """
    Sends the equation image to Mathpix Convert API and requests SVG output.
    Returns SVG string or None on failure.
    """
    import base64
    import requests  # type: ignore

    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "src": f"data:image/png;base64,{b64}",
        "formats": ["svg"],
        "ocr": ["math"],
    }
    headers = {
        "app_id": app_id,
        "app_key": app_key,
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            "https://api.mathpix.com/v3/text",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("svg")
    except Exception:
        return None


def run(
    fallback_list: list[EquationRegion],
    bus: EventBus,
    mathpix_app_id: Optional[str] = None,
    mathpix_app_key: Optional[str] = None,
) -> tuple[list[EquationRegion], StageResult]:
    t0 = time.perf_counter()
    stage = "s08b_fallback"
    warnings: list[str] = []
    errors: list[str] = []
    mathpix_ok = 0
    flagged = 0

    bus.emit(stage, "stage_start", equations=len(fallback_list))
    log.info("stage_start", equations=len(fallback_list))

    use_mathpix = bool(mathpix_app_id and mathpix_app_key)

    for region in fallback_list:
        svg = None

        if use_mathpix and region.source_image_crop:
            try:
                svg = _mathpix_svg(region.source_image_crop, mathpix_app_id, mathpix_app_key)
                if svg:
                    region.svg = svg
                    region.fallback_used = True
                    region.confidence_gate = ConfidenceGate.FALLBACK
                    mathpix_ok += 1
                    bus.emit(stage, "fallback_triggered", equation_id=region.region_id, method="mathpix")
            except Exception as exc:
                region.error_codes.append(ErrorCode.MATHPIX_API_FAILED.value)
                warnings.append(f"{region.region_id}: Mathpix failed — {exc}")

        if not svg:
            # Flag for human review
            region.flagged_for_review = True
            region.confidence_gate = ConfidenceGate.FLAGGED
            flagged += 1
            bus.emit(stage, "fallback_triggered", equation_id=region.region_id, method="flagged")
            log.warning("equation_flagged", equation_id=region.region_id)

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    bus.emit(stage, "stage_end", mathpix_ok=mathpix_ok, flagged=flagged)
    log.info("stage_end", mathpix_ok=mathpix_ok, flagged=flagged)

    return fallback_list, StageResult(
        stage_name=stage,
        ok=True,
        duration_ms=duration_ms,
        warnings=warnings,
        errors=errors,
        metrics={"mathpix_ok": mathpix_ok, "flagged": flagged},
    )

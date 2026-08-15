# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

import time
from contextlib import contextmanager

from .logger import get_logger


@contextmanager
def stage_span(stage_name: str, **context):
    """
    Context manager for timing an entire stage.
    Logs start and end events with duration.

    Usage:
        with stage_span("s06_validation", document="feynman_lectures.pdf") as span:
            # ... do work ...
            span["equations_processed"] = 47

    The span dict is available inside the block for accumulating metrics.
    On exit, all span contents are logged with the duration.
    """
    log = get_logger(stage_name)
    span = {"stage": stage_name, **context}
    t0 = time.perf_counter()
    log.info("stage_start", **span)
    try:
        yield span
        span["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        log.info("stage_end", status="ok", **span)
    except Exception as exc:
        span["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        log.error("stage_end", status="error", error=str(exc), **span)
        raise


@contextmanager
def equation_span(equation_id: str, stage: str, **context):
    """
    Context manager for timing one equation's processing within a stage.
    Binds equation_id to all log calls within the block.

    Usage:
        with equation_span("eq_3_12", "s06_validation") as eq_span:
            eq_span["raw_latex"] = raw
            eq_span["cdm_score"] = score
    """
    log = get_logger(stage).bind(equation_id=equation_id, **context)
    t0 = time.perf_counter()
    eq_span = {}
    try:
        yield eq_span
        log.debug(
            "equation_processed",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            **eq_span,
        )
    except Exception as exc:
        log.error(
            "equation_failed",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            error=str(exc),
            **eq_span,
        )
        raise

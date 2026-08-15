# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(log_file: Path, verbose: bool = False) -> None:
    """
    Call once at startup (in main.py / CLI entry point).

    Produces two outputs:
      1. log_file: newline-delimited JSON, one object per event.
         Machine-readable. Used by report_builder.py.
      2. stderr: human-readable rich output (via RichHandler).
         Suppressed if verbose=False except for warnings and errors.

    Log levels:
      DEBUG   — per-equation detail (recognition output, CDM scores, cache lookups)
      INFO    — stage start/end, document-level summary, equation counts
      WARNING — recoverable issues (repair triggered, fallback used, low CDM score)
      ERROR   — stage failure, equation flagged, compile error
      CRITICAL — pipeline abort
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    json_file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    json_file_handler.setLevel(logging.DEBUG)

    from rich.logging import RichHandler

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        level=logging.DEBUG if verbose else logging.WARNING,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(json_file_handler)
    root_logger.addHandler(rich_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        logger_factory=structlog.PrintLoggerFactory(file=json_file_handler.stream),
    )


def get_logger(stage: str):
    """
    Returns a bound logger with the stage name pre-bound.
    Usage: log = get_logger("s06_validation")
    """
    return structlog.get_logger().bind(stage=stage)

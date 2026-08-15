# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class StageResult:
    """
    Returned by every stage. Pipeline orchestrator checks ok before proceeding.
    Failure at any stage records the error and, depending on severity, either
    aborts the pipeline or continues with degraded output.
    """
    stage_name: str
    ok: bool
    duration_ms: float
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    document_path: str
    started_at: datetime
    finished_at: Optional[datetime]
    ok: bool
    output_epub_path: Optional[str]
    stage_results: list[StageResult] = field(default_factory=list)
    total_equations: int = 0
    equations_passed: int = 0
    equations_repaired: int = 0
    equations_fallback: int = 0
    equations_flagged: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fatal_error: Optional[str] = None

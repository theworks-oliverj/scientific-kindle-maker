# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""
Stage 11 — Output and Delivery
Copies the EPUB to the final output location.
Writes the HTML report, JSON result, and structured log reference.
"""
import json
import shutil
import time
from pathlib import Path

from ..models.document import Document
from ..models.enums import ErrorCode
from ..models.results import PipelineResult, StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s11_output")


def _result_to_dict(result: PipelineResult) -> dict:
    return {
        "document_path": result.document_path,
        "started_at": result.started_at.isoformat() if result.started_at else None,
        "finished_at": result.finished_at.isoformat() if result.finished_at else None,
        "ok": result.ok,
        "output_epub_path": result.output_epub_path,
        "total_equations": result.total_equations,
        "equations_passed": result.equations_passed,
        "equations_repaired": result.equations_repaired,
        "equations_fallback": result.equations_fallback,
        "equations_flagged": result.equations_flagged,
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "fatal_error": result.fatal_error,
        "stage_results": [
            {
                "stage_name": sr.stage_name,
                "ok": sr.ok,
                "duration_ms": sr.duration_ms,
                "warnings": sr.warnings,
                "errors": sr.errors,
                "metrics": sr.metrics,
            }
            for sr in result.stage_results
        ],
    }


def write_result_json(result: PipelineResult, output_dir: Path, stem: str) -> None:
    """
    Writes the result JSON. Called from pipeline.py's finally block AFTER
    result.ok, result.finished_at, and result.total_equations are set.
    Must NOT be called from inside _run_stages() — those fields are not yet populated.
    """
    json_path = output_dir / f"{stem}_result.json"
    json_path.write_text(
        json.dumps(_result_to_dict(result), indent=2, default=str),
        encoding="utf-8",
    )


def run(
    epub_path: Path,
    result: PipelineResult,
    document: Document | None,
    event_bus: EventBus,
    output_dir: Path,
    cache_dump: list[dict] | None = None,
) -> StageResult:
    t0 = time.perf_counter()
    stage = "s11_output"
    warnings: list[str] = []
    errors: list[str] = []

    bus_emit = event_bus.emit
    bus_emit(stage, "stage_start")
    log.info("stage_start", output_dir=str(output_dir))

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(result.document_path).stem

        # Copy EPUB to final location
        final_epub = output_dir / f"{stem}.epub"
        if epub_path.exists():
            shutil.copy2(str(epub_path), str(final_epub))
            result.output_epub_path = str(final_epub)

        # Cache dump (optional)
        if cache_dump is not None:
            cache_path = output_dir / f"{stem}_cache.json"
            cache_path.write_text(
                json.dumps(cache_dump, indent=2),
                encoding="utf-8",
            )

        # NOTE: HTML report and result JSON are written in pipeline.py's finally
        # block, AFTER result.ok / finished_at / total_equations are populated.
        # Writing them here would produce incomplete data (ok=False, counts=0).

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        bus_emit(stage, "stage_end", epub=str(final_epub))
        log.info("stage_end", epub=str(final_epub))

        return StageResult(
            stage_name=stage,
            ok=True,
            duration_ms=duration_ms,
            warnings=warnings,
            errors=errors,
            metrics={"output_epub": str(final_epub)},
        )

    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        errors.append(ErrorCode.OUTPUT_WRITE_FAILED.value)
        errors.append(str(exc))
        bus_emit(stage, "stage_end_error", error=str(exc))
        log.error("stage_end", status="error", error=str(exc))
        return StageResult(
            stage_name=stage,
            ok=False,
            duration_ms=duration_ms,
            errors=errors,
        )

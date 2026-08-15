# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""
Pipeline Orchestrator
Runs all stages in sequence for a single document.
Models are loaded once and passed to the stages that need them.
"""
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from .cache.equation_cache import SessionEquationCache
from .models.document import Document
from .models.enums import SourceType
from .models.results import PipelineResult, StageResult
from .observability.event_bus import EventBus
from .observability.logger import get_logger

from .stages import (
    s01_classifier,
    s02b_visual,
    s02c_epub,
    s03_mineru_parse,
    s05c_mathml,
    s06_validation,
    s07_routing,
    s08a_svg_render,
    s08b_fallback,
    s09_svg_postprocess,
    s10_epub_assembly,
    s11_output,
)

log = get_logger("pipeline")


class FatalPipelineError(Exception):
    pass


@dataclass
class PipelineConfig:
    # CDM thresholds
    cdm_pass_threshold: float = 0.88
    cdm_repair_threshold: float = 0.70
    # Rendering — must match the LaTeX wrapper's \documentclass size, or
    # equations stop matching the reader's body text size (see LATEX_BODY_PT).
    body_font_size_pt: float = s06_validation.LATEX_BODY_PT
    tectonic_timeout_s: int = 45
    dvisvgm_timeout_s: int = 10
    max_repair_attempts: int = 2
    # MinerU (replaces old s03–s05b for PDF sources)
    mineru_backend: str = "vlm-engine"
    # Pages per MinerU invocation. Batching is what makes a long book
    # resumable — a failure costs one batch, not the whole parse. Documents
    # at or under one batch behave exactly as they did before.
    mineru_batch_size: int = 50
    # Per-batch timeout is derived from its page count unless mineru_timeout_s
    # is set explicitly. A single fixed cap cannot fit both a 6-page paper and
    # a 500-page textbook.
    mineru_timeout_per_page_s: int = 120
    mineru_timeout_s: Optional[int] = None
    mineru_reuse_existing: bool = True
    # MinerU's per-figure description pass. Its cost scales with figure count —
    # no measurable gain on a figure-sparse paper, potentially real on an
    # image-heavy book. Disabling it drops FigureBlock.alt_text.
    mineru_image_analysis: bool = True
    # Fallback
    mathpix_app_id: Optional[str] = None
    mathpix_app_key: Optional[str] = None
    # Output
    output_dir: str = "./output"
    verbose_logging: bool = False
    # CDM method
    cdm_method: str = "ssim"   # "ssim" | "cdm" (future)
    # Validation
    epubcheck_enabled: bool = True
    # Cache dump
    dump_cache: bool = False
    # Parallelism (Stage 6/8A per-equation work — None = concurrency.py default)
    max_parallel_workers: Optional[int] = None


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.bus = EventBus()
        self.cache = SessionEquationCache()

    def run(self, input_path: str, output_dir: str) -> PipelineResult:
        result = PipelineResult(
            document_path=input_path,
            started_at=datetime.utcnow(),
            finished_at=None,
            ok=False,
            output_epub_path=None,
        )
        document: Optional[Document] = None

        try:
            document = self._run_stages(input_path, output_dir, result)
            result.ok = True
        except FatalPipelineError as exc:
            result.fatal_error = str(exc)
            log.error("pipeline_fatal", error=str(exc))
        except Exception as exc:
            result.fatal_error = str(exc)
            log.error("pipeline_unexpected_error", error=str(exc))
        finally:
            result.finished_at = datetime.utcnow()

            # Populate equation counts from document
            if document:
                all_eq = document.all_equations
                result.total_equations = len(all_eq)
                result.equations_passed   = sum(1 for e in all_eq if e.confidence_gate and e.confidence_gate.value == "pass")
                result.equations_repaired = sum(1 for e in all_eq if e.repair_attempts > 0 and not e.flagged_for_review)
                result.equations_fallback = sum(1 for e in all_eq if e.fallback_used)
                result.equations_flagged  = sum(1 for e in all_eq if e.flagged_for_review)

            # Cache stats
            stats = self.cache.stats
            result.cache_hits   = stats["hits"]
            result.cache_misses = stats["misses"]

            # Write report and result JSON after all result fields are populated.
            # This is the single authoritative write — s11_output.run() no longer
            # writes these files so that ok/finished_at/total_equations are correct.
            try:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(input_path).stem
                report_path = out_dir / f"{stem}_report.html"
                from .observability.report_builder import build_report
                build_report(result, document, self.bus, report_path)
                from .stages.s11_output import write_result_json
                write_result_json(result, out_dir, stem)
                # Recognition snapshot — the only artefact that can prove a
                # parser change did not silently alter equation *contents*.
                # Written on failed runs too: a partial parse is still worth
                # diffing against.
                from .qa.latex_snapshot import write_snapshot
                write_snapshot(document, out_dir, stem)
            except Exception as exc:
                log.error("report_generation_failed", error=str(exc))

            self.cache.clear()

        return result

    def _run_stages(self, input_path: str, output_dir: str, result: PipelineResult) -> Document:
        cfg = self.config
        out_dir = Path(output_dir)

        # Stage 1: Input classification (fatal on failure)
        metadata, sr1 = s01_classifier.run(input_path, self.bus)
        result.stage_results.append(sr1)
        if not sr1.ok or metadata is None:
            raise FatalPipelineError(f"Stage 1 failed: {sr1.errors}")

        # Stage 2: Source-specific extraction (fatal on failure).
        # ALL PDFs — scanned and digital — are rasterized via s02b: MinerU
        # owns text/formula extraction either way, and the 300dpi page
        # rasters supply equation/figure crops for the quality gate and
        # raster fallbacks.
        source_type = metadata.source_type
        if source_type == SourceType.EPUB or source_type == SourceType.HTML:
            document, sr2 = s02c_epub.run(metadata, self.bus)
        else:
            document, sr2 = s02b_visual.run(metadata, self.bus)

        result.stage_results.append(sr2)
        if not sr2.ok:
            raise FatalPipelineError(f"Stage 2 failed: {sr2.errors}")

        # Stage 3 (MinerU): layout + reading order + text OCR + formula
        # recognition in one subprocess pass (fatal on failure). EPUB/HTML
        # sources skip MinerU — s02c already extracted text and MathML.
        if source_type in (SourceType.EPUB, SourceType.HTML):
            document, sr5c = s05c_mathml.run(document, self.bus)
            result.stage_results.append(sr5c)
        else:
            document, sr3 = s03_mineru_parse.run(
                document,
                self.bus,
                source_pdf=input_path,
                work_dir=out_dir / "mineru",
                backend=cfg.mineru_backend,
                timeout_s=cfg.mineru_timeout_s,
                reuse_existing=cfg.mineru_reuse_existing,
                batch_size=cfg.mineru_batch_size,
                timeout_per_page_s=cfg.mineru_timeout_per_page_s,
                image_analysis=cfg.mineru_image_analysis,
            )
            result.stage_results.append(sr3)
            if not sr3.ok:
                raise FatalPipelineError(f"Stage 3 (MinerU) failed: {sr3.errors}")

        # Stage 6: LaTeX validation and repair (non-fatal per equation)
        document, sr6 = s06_validation.run(
            document,
            self.bus,
            cdm_pass_threshold=cfg.cdm_pass_threshold,
            cdm_repair_threshold=cfg.cdm_repair_threshold,
            max_repair_attempts=cfg.max_repair_attempts,
            cdm_method=cfg.cdm_method,
            max_parallel_workers=cfg.max_parallel_workers,
            # Lives in the output dir so it is scoped to this book and gets
            # cleaned up with it. Re-running a long book then costs no
            # tectonic compiles for equations that already succeeded.
            compile_cache_dir=out_dir / ".compile_cache",
        )
        result.stage_results.append(sr6)

        # Stage 7: Confidence routing (non-fatal)
        document, pass_list, fallback_list, sr7 = s07_routing.run(document, self.bus)
        result.stage_results.append(sr7)

        # Stage 8A: SVG rendering of passed equations (non-fatal per equation)
        _, failed_from_render, sr8a = s08a_svg_render.run(
            pass_list,
            self.cache,
            self.bus,
            document=document,
            tectonic_timeout=cfg.tectonic_timeout_s,
            dvisvgm_timeout=cfg.dvisvgm_timeout_s,
            max_parallel_workers=cfg.max_parallel_workers,
        )
        result.stage_results.append(sr8a)

        # Route render failures to fallback
        all_fallback = fallback_list + failed_from_render

        # Stage 8B: Fallback handling (Mathpix or flag)
        _, sr8b = s08b_fallback.run(
            all_fallback,
            self.bus,
            mathpix_app_id=cfg.mathpix_app_id,
            mathpix_app_key=cfg.mathpix_app_key,
        )
        result.stage_results.append(sr8b)

        # Stage 9: SVG post-processing (non-fatal per equation)
        document, sr9 = s09_svg_postprocess.run(document, self.bus, cfg.body_font_size_pt)
        result.stage_results.append(sr9)

        # Stage 10: EPUB assembly (fatal on failure)
        epub_tmp = out_dir / f"{Path(input_path).stem}_assembled.epub"
        epub_path, sr10 = s10_epub_assembly.run(document, epub_tmp, self.bus, cfg.epubcheck_enabled)
        result.stage_results.append(sr10)
        if not sr10.ok:
            raise FatalPipelineError(f"Stage 10 (EPUB assembly) failed: {sr10.errors}")

        # Stage 11: Output and delivery
        cache_dump = self.cache.dump() if cfg.dump_cache else None
        sr11 = s11_output.run(epub_path, result, document, self.bus, out_dir, cache_dump)
        result.stage_results.append(sr11)

        return document

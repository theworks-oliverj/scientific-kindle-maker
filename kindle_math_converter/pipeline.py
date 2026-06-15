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
from .model_manager import load_all_models, get_models_dir
from .models.document import Document
from .models.enums import SourceType
from .models.results import PipelineResult, StageResult
from .observability.event_bus import EventBus
from .observability.logger import get_logger

from .stages import (
    s01_classifier,
    s02a_digital_pdf,
    s02b_visual,
    s02c_epub,
    s03_detection,
    s04_reading_order,
    s05a_text_ocr,
    s05b_formula_recognition,
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
    # Rendering
    body_font_size_pt: float = 10.0
    tectonic_timeout_s: int = 15
    dvisvgm_timeout_s: int = 10
    max_repair_attempts: int = 2
    # Detection
    formula_detection_confidence: float = 0.35
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
    # Debug dump (Stage 3/5B detection & recognition audit trail)
    s03_debug_dump_enabled: bool = False
    s03_debug_dump_dir: Optional[str] = None


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.bus = EventBus()
        self.cache = SessionEquationCache()
        # load_all_models handles downloading on first run via model_manager.py
        self.models = load_all_models(get_models_dir())

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
            except Exception as exc:
                log.error("report_generation_failed", error=str(exc))

            self.cache.clear()

        return result

    def _run_stages(self, input_path: str, output_dir: str, result: PipelineResult) -> Document:
        cfg = self.config
        out_dir = Path(output_dir)

        debug_dump_dir: Optional[str] = None
        if cfg.s03_debug_dump_enabled:
            debug_dump_dir = cfg.s03_debug_dump_dir or str(out_dir / "debug_crops")

        # Stage 1: Input classification (fatal on failure)
        metadata, sr1 = s01_classifier.run(input_path, self.bus)
        result.stage_results.append(sr1)
        if not sr1.ok or metadata is None:
            raise FatalPipelineError(f"Stage 1 failed: {sr1.errors}")

        # Stage 2: Source-specific extraction (fatal on failure)
        source_type = metadata.source_type
        if source_type == SourceType.EPUB or source_type == SourceType.HTML:
            document, sr2 = s02c_epub.run(metadata, self.bus)
        elif source_type == SourceType.VISUAL_PDF or metadata.is_scanned:
            document, sr2 = s02b_visual.run(metadata, self.bus)
        else:
            document, sr2 = s02a_digital_pdf.run(metadata, self.bus)

        result.stage_results.append(sr2)
        if not sr2.ok:
            raise FatalPipelineError(f"Stage 2 failed: {sr2.errors}")

        # Stage 3: Layout and formula detection (fatal on failure)
        if source_type not in (SourceType.EPUB, SourceType.HTML):
            model_bundle = s03_detection.ModelBundle(
                layout_model=self.models.get("layout"),
                formula_model=self.models.get("formula"),
            )
            document, sr3 = s03_detection.run(
                document,
                model_bundle,
                self.bus,
                formula_confidence_threshold=cfg.formula_detection_confidence,
                debug_dump_dir=debug_dump_dir,
            )
            result.stage_results.append(sr3)
            if not sr3.ok:
                raise FatalPipelineError(f"Stage 3 failed: {sr3.errors}")

        # Stage 4: Reading order (non-fatal — continue on warning)
        document, sr4 = s04_reading_order.run(document, self.bus)
        result.stage_results.append(sr4)

        # Stage 5A: Text OCR (non-fatal)
        document, sr5a = s05a_text_ocr.run(document, self.models.get("paddleocr"), self.bus)
        result.stage_results.append(sr5a)

        # Stage 5B: Formula recognition (non-fatal per equation)
        document, sr5b = s05b_formula_recognition.run(
            document, self.models.get("pix2tex"), self.bus, debug_dump_dir=debug_dump_dir,
        )
        result.stage_results.append(sr5b)

        # Stage 6: LaTeX validation and repair (non-fatal per equation)
        document, sr6 = s06_validation.run(
            document,
            self.bus,
            cdm_pass_threshold=cfg.cdm_pass_threshold,
            cdm_repair_threshold=cfg.cdm_repair_threshold,
            max_repair_attempts=cfg.max_repair_attempts,
            cdm_method=cfg.cdm_method,
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

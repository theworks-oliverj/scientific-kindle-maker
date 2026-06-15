"""
Stage 5A — Text OCR
For digital PDFs: no-op (text already extracted in Stage 2A).
For scanned PDFs: runs PaddleOCR v4 on text block regions identified by DocLayout-YOLO.
"""
import time

from ..models.document import Document
from ..models.enums import SourceType, ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s05a_text_ocr")

OCR_CONFIDENCE_THRESHOLD = 0.70


def run(
    document: Document,
    ocr_model,
    bus: EventBus,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    stage = "s05a_text_ocr"
    warnings: list[str] = []
    errors: list[str] = []

    bus.emit(stage, "stage_start")

    # Digital PDFs already have text — this stage is a no-op for them
    if document.metadata.source_type in (SourceType.LATEX_PDF,) and not document.metadata.is_scanned:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        bus.emit(stage, "stage_end", skipped=True)
        log.info("stage_end", skipped=True, reason="digital_pdf_text_already_extracted")
        return document, StageResult(
            stage_name=stage,
            ok=True,
            duration_ms=duration_ms,
            metrics={"skipped": True},
        )

    low_confidence_count = 0
    blocks_processed = 0

    try:
        import io
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        total_blocks = sum(len(page.text_blocks) for page in document.pages)

        for page in document.pages:
            for block in page.text_blocks:
                # OCR the masked crop produced in Stage 3 (equation pixels already
                # whitewashed). We do NOT re-crop from the page image: block.bbox is in
                # PDF points, not pixels, so cropping with it would be wrong.
                if not block.source_image_crop:
                    continue

                crop = Image.open(io.BytesIO(block.source_image_crop)).convert("RGB")
                crop_array = np.array(crop)

                # PaddleOCR 3.x: cls= removed; orientation set via use_textline_orientation in constructor
                result = ocr_model.ocr(crop_array)
                if not result or not result[0]:
                    continue

                text_parts = []
                min_confidence = 1.0

                # PaddleOCR 3.x .ocr() returns a list of dict-like OCRResult objects
                # with "rec_texts" / "rec_scores" keys (not per-line [bbox, (text, conf)] tuples).
                page_result = result[0]
                rec_texts = page_result.get("rec_texts", [])
                rec_scores = page_result.get("rec_scores", [])

                for text, conf in zip(rec_texts, rec_scores):
                    text_parts.append(str(text))
                    if conf < min_confidence:
                        min_confidence = conf
                    if conf < OCR_CONFIDENCE_THRESHOLD:
                        low_confidence_count += 1
                        warnings.append(
                            f"Low OCR confidence ({conf:.2f}) on page {page.page_number} block {block.block_id}"
                        )
                        bus.emit(
                            stage,
                            "equation_warning",
                            equation_id=None,
                            block_id=block.block_id,
                            confidence=conf,
                        )

                block.raw_text = " ".join(text_parts)
                blocks_processed += 1

                if blocks_processed % 25 == 0:
                    log.info("ocr_progress", blocks_done=blocks_processed, blocks_total=total_blocks)

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        bus.emit(stage, "stage_end", blocks_processed=blocks_processed, low_confidence=low_confidence_count)
        log.info("stage_end", blocks_processed=blocks_processed)

        return document, StageResult(
            stage_name=stage,
            ok=True,
            duration_ms=duration_ms,
            warnings=warnings,
            errors=errors,
            metrics={
                "blocks_processed": blocks_processed,
                "low_confidence_blocks": low_confidence_count,
            },
        )

    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        errors.append(ErrorCode.OCR_LOW_CONFIDENCE.value)
        errors.append(str(exc))
        bus.emit(stage, "stage_end_error", error=str(exc))
        log.error("stage_end", status="error", error=str(exc))
        return document, StageResult(
            stage_name=stage,
            ok=False,
            duration_ms=duration_ms,
            errors=errors,
        )

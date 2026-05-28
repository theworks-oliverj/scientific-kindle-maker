"""
Stage 2A — Digital PDF Extraction
Extracts text blocks and bounding boxes from LaTeX-typeset PDFs via PyMuPDF.
Text spans using math fonts are tagged as potential inline equation hints.
"""
import time

from ..models.document import Document, DocumentMetadata, Page, TextBlock, BoundingBox
from ..models.enums import ColumnLayout
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger
from .s01_classifier import MATH_FONT_PREFIXES

log = get_logger("s02a_digital_pdf")


def run(
    metadata: DocumentMetadata,
    bus: EventBus,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    stage = "s02a_digital_pdf"
    warnings: list[str] = []
    errors: list[str] = []

    bus.emit(stage, "stage_start")
    log.info("stage_start", path=metadata.source_path)

    try:
        import pymupdf  # type: ignore

        doc_obj = pymupdf.open(metadata.source_path)
        document = Document(metadata=metadata)

        for page_num in range(len(doc_obj)):
            mupdf_page = doc_obj[page_num]
            rect = mupdf_page.rect

            page = Page(
                page_number=page_num + 1,
                width_pt=rect.width,
                height_pt=rect.height,
                column_layout=metadata.column_layout,
                image_bytes=None,
            )

            blocks_data = mupdf_page.get_text("dict")
            block_index = 0

            for block in blocks_data.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue

                full_text_parts = []
                has_math_span = False

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_name = span.get("font", "").lower()
                        text = span.get("text", "")
                        full_text_parts.append(text)
                        if any(font_name.startswith(p) for p in MATH_FONT_PREFIXES):
                            has_math_span = True

                raw_text = " ".join(full_text_parts).strip()
                if not raw_text:
                    continue

                bbox_data = block.get("bbox", (0, 0, 0, 0))
                bbox = BoundingBox(
                    x0=bbox_data[0],
                    y0=bbox_data[1],
                    x1=bbox_data[2],
                    y1=bbox_data[3],
                    coordinate_system="pdf_points",
                    page_number=page_num + 1,
                )

                text_block = TextBlock(
                    block_id=f"tb_{page_num + 1}_{block_index}",
                    bbox=bbox,
                    raw_text=raw_text,
                    reading_order_index=block_index,
                )
                page.text_blocks.append(text_block)

                # Tag blocks containing math font spans — Stage 3 confirms via detection
                if has_math_span:
                    log.debug(
                        "math_font_hint",
                        page=page_num + 1,
                        block_id=text_block.block_id,
                    )

                block_index += 1

            document.pages.append(page)

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        bus.emit(stage, "stage_end", pages=len(document.pages))
        log.info("stage_end", pages=len(document.pages))

        return document, StageResult(
            stage_name=stage,
            ok=True,
            duration_ms=duration_ms,
            warnings=warnings,
            errors=errors,
            metrics={"pages_extracted": len(document.pages)},
        )

    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        errors.append(str(exc))
        bus.emit(stage, "stage_end_error", error=str(exc))
        log.error("stage_end", status="error", error=str(exc))
        # Return empty document so orchestrator can handle gracefully
        return Document(metadata=metadata), StageResult(
            stage_name=stage,
            ok=False,
            duration_ms=duration_ms,
            errors=errors,
        )

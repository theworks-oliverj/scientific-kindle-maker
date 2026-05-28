"""
Stage 3 — Layout and Formula Detection
Runs DocLayout-YOLO and YOLOv8-MFD on page images to detect equation regions.
Failure mode: Fatal.
"""
import io
import time
from dataclasses import dataclass
from typing import Any

from ..models.document import Document, Page, EquationRegion, BoundingBox
from ..models.enums import FormulaClass, ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s03_detection")

FORMULA_DETECTION_CONFIDENCE_DEFAULT = 0.35

# Minimum crop dimensions in pixels to be considered a real equation.
# At 300 DPI, 1pt ≈ 4.17 px. A lone punctuation mark (©, ?, hyphen) typically
# occupies < 20×20 px. Real inline equations are at least ~25px wide and ~12px tall.
MIN_EQUATION_WIDTH_PX  = 20
MIN_EQUATION_HEIGHT_PX = 12


@dataclass
class ModelBundle:
    layout_model: Any    # DocLayout-YOLO
    formula_model: Any   # YOLOv8-MFD


def _page_to_image(page: Page, dpi: int = 150):
    """
    Returns a PIL Image for a page.
    For visual PDFs: decode stored image bytes.
    For digital PDFs: rasterize at dpi (for detection only — not for OCR).
    """
    from PIL import Image

    if page.image_bytes:
        return Image.open(io.BytesIO(page.image_bytes)).convert("RGB")

    # Digital PDF — no stored image; caller must provide rasterized image
    # This function is called with the rasterized image already in image_bytes
    # for detection pass. If image_bytes is None, return a placeholder.
    return Image.new("RGB", (int(page.width_pt), int(page.height_pt)), color=255)


def _crop_image(pil_img, x0: float, y0: float, x1: float, y1: float, padding: int = 4) -> bytes:
    """Crops a PIL Image at the given pixel coordinates with uniform padding. Returns PNG bytes."""
    width, height = pil_img.size
    cx0 = max(0, int(x0) - padding)
    cy0 = max(0, int(y0) - padding)
    cx1 = min(width, int(x1) + padding)
    cy1 = min(height, int(y1) + padding)

    crop = pil_img.crop((cx0, cy0, cx1, cy1))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def _rasterize_pdf_page_for_detection(pdf_path: str, page_num: int, dpi: int = 150) -> bytes:
    """Rasterizes a single PDF page at low DPI just for detection (not OCR)."""
    import pymupdf  # type: ignore

    doc = pymupdf.open(pdf_path)
    page = doc[page_num]
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csGRAY)
    return pix.tobytes("png")


def run(
    document: Document,
    models: ModelBundle,
    bus: EventBus,
    formula_confidence_threshold: float = FORMULA_DETECTION_CONFIDENCE_DEFAULT,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    stage = "s03_detection"
    warnings: list[str] = []
    errors: list[str] = []
    total_detected = 0

    bus.emit(stage, "stage_start")
    log.info("stage_start", pages=len(document.pages))

    try:
        from PIL import Image  # type: ignore

        for page in document.pages:
            # Get or rasterize page image
            if page.image_bytes:
                pil_img = Image.open(io.BytesIO(page.image_bytes)).convert("RGB")
                img_w, img_h = pil_img.size
                detection_dpi = 300  # already at high DPI from Stage 2B
            else:
                # Digital PDF — rasterize at 150 DPI for detection
                detection_dpi = 150
                raw_bytes = _rasterize_pdf_page_for_detection(
                    document.metadata.source_path,
                    page.page_number - 1,
                    dpi=detection_dpi,
                )
                pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                img_w, img_h = pil_img.size
                # Store detection image for later use in this stage
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                if not page.image_bytes:
                    page.image_bytes = buf.getvalue()

            # Scale factors: image pixels → PDF points
            px_to_pt_x = page.width_pt / img_w
            px_to_pt_y = page.height_pt / img_h

            # Run formula detection model
            try:
                detections = _run_formula_detection(models.formula_model, pil_img)
            except Exception as exc:
                warnings.append(f"Page {page.page_number} detection failed: {exc}")
                log.warning("formula_detection_failed", page=page.page_number, error=str(exc))
                continue

            eq_index = len(page.equation_regions)

            for det in detections:
                if det["confidence"] < formula_confidence_threshold:
                    continue

                px_x0, px_y0, px_x1, px_y1 = det["bbox"]

                # Size filter: reject detections too small to be a real equation
                # (false positives: ©, ?, lone dash, apostrophe).
                px_w = px_x1 - px_x0
                px_h = px_y1 - px_y0
                if px_w < MIN_EQUATION_WIDTH_PX or px_h < MIN_EQUATION_HEIGHT_PX:
                    bus.emit(
                        stage,
                        "equation_skipped",
                        page=page.page_number,
                        reason="bbox_too_small",
                        width=round(px_w, 1),
                        height=round(px_h, 1),
                    )
                    continue

                formula_class = (
                    FormulaClass.DISPLAY
                    if det.get("class_name", "").startswith("display")
                    else FormulaClass.INLINE
                )

                crop_bytes = _crop_image(pil_img, px_x0, px_y0, px_x1, px_y1)

                # Convert to PDF point coordinate system
                bbox = BoundingBox(
                    x0=px_x0 * px_to_pt_x,
                    y0=px_y0 * px_to_pt_y,
                    x1=px_x1 * px_to_pt_x,
                    y1=px_y1 * px_to_pt_y,
                    coordinate_system="pdf_points",
                    page_number=page.page_number,
                )

                region = EquationRegion(
                    region_id=f"eq_{page.page_number}_{eq_index}",
                    bbox=bbox,
                    formula_class=formula_class,
                    source_image_crop=crop_bytes,
                    raw_latex=None,
                    normalized_latex=None,
                    cdm_score=None,
                    confidence_gate=None,
                    svg=None,
                    svg_postprocessed=None,
                    equation_number=None,
                )
                page.equation_regions.append(region)
                eq_index += 1
                total_detected += 1

                bus.emit(
                    stage,
                    "equation_ok",
                    equation_id=region.region_id,
                    formula_class=formula_class.value,
                    confidence=det["confidence"],
                )

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        bus.emit(stage, "stage_end", equations_detected=total_detected)
        log.info("stage_end", equations_detected=total_detected)

        return document, StageResult(
            stage_name=stage,
            ok=True,
            duration_ms=duration_ms,
            warnings=warnings,
            errors=errors,
            metrics={"equations_detected": total_detected},
        )

    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        errors.append(ErrorCode.FORMULA_DETECTION_FAILED.value)
        errors.append(str(exc))
        bus.emit(stage, "stage_end_error", error=str(exc))
        log.error("stage_end", status="error", error=str(exc))
        return document, StageResult(
            stage_name=stage,
            ok=False,
            duration_ms=duration_ms,
            errors=errors,
        )


def _run_formula_detection(formula_model, pil_img) -> list[dict]:
    """
    Runs YOLOv8-MFD formula detection on a PIL image.
    Returns list of dicts with keys: bbox (x0,y0,x1,y1), confidence, class_name.

    PDF-Extract-Kit class mapping (opendatalab/PDF-Extract-Kit-1.0):
      0 = "embedding"  → inline formula
      1 = "isolated"   → display (block) formula
    """
    # PDF-Extract-Kit class-ID → our convention
    _CLASS_MAP = {0: "inline_formula", 1: "display_formula"}

    results = formula_model(pil_img)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = _CLASS_MAP.get(cls_id, "inline_formula")
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class_name": class_name,
            })
    return detections

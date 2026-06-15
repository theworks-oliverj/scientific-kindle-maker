"""
Stage 3 — Layout and Formula Detection
Runs DocLayout-YOLO and YOLOv8-MFD on page images to detect equation regions.
Failure mode: Fatal.
"""
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..models.document import Document, Page, EquationRegion, TextBlock, BoundingBox
from ..models.enums import FormulaClass, ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s03_detection")

FORMULA_DETECTION_CONFIDENCE_DEFAULT = 0.35
TEXT_DETECTION_CONFIDENCE_DEFAULT = 0.35

# Minimum crop dimensions in pixels to be considered a real equation.
# At 300 DPI, 1pt ≈ 4.17 px. A lone punctuation mark (©, ?, hyphen) typically
# occupies < 20×20 px. Real inline equations are at least ~25px wide and ~12px tall.
MIN_EQUATION_WIDTH_PX  = 20
MIN_EQUATION_HEIGHT_PX = 12

# DocLayout-YOLO (DocStructBench) class names that carry readable body text.
# Excludes: "abandon" (headers/footers/page numbers), "figure", "table",
# and "isolate_formula" (equations are YOLOv8-MFD's job).
TEXT_LAYOUT_CLASSES = frozenset({
    "title", "plain text", "figure_caption", "table_caption",
    "table_footnote", "formula_caption",
})


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


def _crop_and_mask_text(
    pil_img,
    tx0: float, ty0: float, tx1: float, ty1: float,
    equation_px_boxes: list[tuple[float, float, float, float]],
    padding: int = 4,
    mask_margin: int = 2,
) -> bytes:
    """
    Crops a text region and whitewashes any overlapping equation pixels so the
    OCR engine never sees garbled inline-math glyphs (the equation is rendered
    separately as SVG). Returns PNG bytes.
    """
    from PIL import Image, ImageDraw

    width, height = pil_img.size
    cx0 = max(0, int(tx0) - padding)
    cy0 = max(0, int(ty0) - padding)
    cx1 = min(width, int(tx1) + padding)
    cy1 = min(height, int(ty1) + padding)

    crop = pil_img.crop((cx0, cy0, cx1, cy1)).convert("L")
    draw = ImageDraw.Draw(crop)

    for ex0, ey0, ex1, ey1 in equation_px_boxes:
        # Skip equations that don't overlap this text region at all.
        if ex1 <= cx0 or ex0 >= cx1 or ey1 <= cy0 or ey0 >= cy1:
            continue
        # Translate to crop-local coords and paint white (255), with a small margin.
        rx0 = max(0, int(ex0) - cx0 - mask_margin)
        ry0 = max(0, int(ey0) - cy0 - mask_margin)
        rx1 = min(crop.width, int(ex1) - cx0 + mask_margin)
        ry1 = min(crop.height, int(ey1) - cy0 + mask_margin)
        if rx1 > rx0 and ry1 > ry0:
            draw.rectangle([rx0, ry0, rx1, ry1], fill=255)

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def _dump_debug_crop(
    pil_img,
    debug_dir: Path,
    page_number: int,
    name: str,
    x0: float, y0: float, x1: float, y1: float,
) -> None:
    """Writes a labeled crop PNG to `debug_dir/page_<NN>/<name>.png` for Stage 1.1's audit dump."""
    page_dir = debug_dir / f"page_{page_number:02d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / f"{name}.png").write_bytes(_crop_image(pil_img, x0, y0, x1, y1))


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
    text_confidence_threshold: float = TEXT_DETECTION_CONFIDENCE_DEFAULT,
    debug_dump_dir: Optional[str] = None,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    stage = "s03_detection"
    warnings: list[str] = []
    errors: list[str] = []
    total_detected = 0
    total_text_detected = 0

    debug_dir = Path(debug_dump_dir) if debug_dump_dir else None
    detection_records: list[dict] = []

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
            # Accepted equation pixel bboxes — used to mask math out of text crops below.
            equation_px_boxes: list[tuple[float, float, float, float]] = []

            for det_index, det in enumerate(detections):
                if det["confidence"] < formula_confidence_threshold:
                    continue

                px_x0, px_y0, px_x1, px_y1 = det["bbox"]

                record: Optional[dict] = None
                if debug_dir is not None:
                    record = {
                        "page": page.page_number,
                        "det_index": det_index,
                        "class_name": det.get("class_name"),
                        "confidence": round(det["confidence"], 4),
                        "bbox_px": [round(px_x0, 1), round(px_y0, 1), round(px_x1, 1), round(px_y1, 1)],
                        "width_px": round(px_x1 - px_x0, 1),
                        "height_px": round(px_y1 - px_y0, 1),
                    }

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
                    if record is not None and debug_dir is not None:
                        record["status"] = "too_small"
                        record["region_id"] = None
                        detection_records.append(record)
                        _dump_debug_crop(
                            pil_img, debug_dir, page.page_number,
                            f"det_{det_index}_too_small", px_x0, px_y0, px_x1, px_y1,
                        )
                    continue

                formula_class = (
                    FormulaClass.DISPLAY
                    if det.get("class_name", "").startswith("display")
                    else FormulaClass.INLINE
                )

                crop_bytes = _crop_image(pil_img, px_x0, px_y0, px_x1, px_y1)
                equation_px_boxes.append((px_x0, px_y0, px_x1, px_y1))

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

                if record is not None and debug_dir is not None:
                    record["status"] = "accepted"
                    record["region_id"] = region.region_id
                    record["formula_class"] = formula_class.value
                    detection_records.append(record)
                    _dump_debug_crop(
                        pil_img, debug_dir, page.page_number,
                        region.region_id, px_x0, px_y0, px_x1, px_y1,
                    )

                bus.emit(
                    stage,
                    "equation_ok",
                    equation_id=region.region_id,
                    formula_class=formula_class.value,
                    confidence=det["confidence"],
                )

            # ── Body-text region detection (DocLayout-YOLO) ──────────────────
            # Populates page.text_blocks with equation pixels masked out, so the
            # OCR stage (s05a) never sees garbled inline math. Skipped silently if
            # the layout model is unavailable.
            if models.layout_model is not None:
                try:
                    text_dets = _run_layout_detection(models.layout_model, pil_img)
                except Exception as exc:
                    warnings.append(f"Page {page.page_number} layout detection failed: {exc}")
                    log.warning("layout_detection_failed", page=page.page_number, error=str(exc))
                    text_dets = []

                tb_index = len(page.text_blocks)
                for det in text_dets:
                    if det["confidence"] < text_confidence_threshold:
                        continue

                    tx0, ty0, tx1, ty1 = det["bbox"]
                    if (tx1 - tx0) < MIN_EQUATION_WIDTH_PX or (ty1 - ty0) < MIN_EQUATION_HEIGHT_PX:
                        continue

                    masked_crop = _crop_and_mask_text(
                        pil_img, tx0, ty0, tx1, ty1, equation_px_boxes
                    )

                    text_bbox = BoundingBox(
                        x0=tx0 * px_to_pt_x,
                        y0=ty0 * px_to_pt_y,
                        x1=tx1 * px_to_pt_x,
                        y1=ty1 * px_to_pt_y,
                        coordinate_system="pdf_points",
                        page_number=page.page_number,
                    )

                    page.text_blocks.append(TextBlock(
                        block_id=f"tb_{page.page_number}_{tb_index}",
                        bbox=text_bbox,
                        raw_text="",                # filled by s05a OCR
                        reading_order_index=tb_index,
                        source_image_crop=masked_crop,
                    ))
                    tb_index += 1
                    total_text_detected += 1

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            with open(debug_dir / "detections.jsonl", "w") as f:
                for rec in detection_records:
                    f.write(json.dumps(rec) + "\n")

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        bus.emit(stage, "stage_end", equations_detected=total_detected,
                 text_blocks_detected=total_text_detected)
        log.info("stage_end", equations_detected=total_detected,
                 text_blocks_detected=total_text_detected)

        return document, StageResult(
            stage_name=stage,
            ok=True,
            duration_ms=duration_ms,
            warnings=warnings,
            errors=errors,
            metrics={
                "equations_detected": total_detected,
                "text_blocks_detected": total_text_detected,
            },
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


def _run_layout_detection(layout_model, pil_img) -> list[dict]:
    """
    Runs DocLayout-YOLO on a PIL image and returns only text-bearing regions.
    Returns list of dicts with keys: bbox (x0,y0,x1,y1), confidence, class_name.

    Class names are read from layout_model.names (e.g. {0:'title', 1:'plain text',
    8:'isolate_formula', ...}) and filtered by TEXT_LAYOUT_CLASSES — robust to any
    class-id reordering between model versions.
    """
    names = getattr(layout_model, "names", {}) or {}
    results = layout_model(pil_img)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, str(cls_id))
            if class_name not in TEXT_LAYOUT_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf[0]),
                "class_name": class_name,
            })
    return detections

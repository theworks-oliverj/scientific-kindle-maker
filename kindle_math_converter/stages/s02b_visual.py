"""
Stage 2B — Visual/Scanned PDF Extraction
Rasterizes each page to 300 DPI grayscale PNG. All downstream stages
for this source type operate on images, not PDF internal structure.
"""
import io
import time

from ..models.document import Document, DocumentMetadata, Page
from ..models.enums import ColumnLayout, ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s02b_visual")


def _deskew_image(img):
    """
    Deskews image if rotation > 0.5° using OpenCV Hough line detection.
    Returns a PIL Image.
    """
    import cv2  # type: ignore
    import numpy as np
    from PIL import Image

    img_array = np.array(img.convert("L"))
    edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return img

    return img.rotate(-median_angle, expand=True, fillcolor=255)


def _normalize_contrast(img):
    """
    Applies CLAHE (adaptive histogram equalization) on grayscale image.
    Keeps grayscale — UniMERNet performs better on grayscale than binarized.
    """
    import cv2  # type: ignore
    import numpy as np
    from PIL import Image

    gray = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return Image.fromarray(equalized)


def run(
    metadata: DocumentMetadata,
    bus: EventBus,
    dpi: int = 300,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    stage = "s02b_visual"
    warnings: list[str] = []
    errors: list[str] = []

    bus.emit(stage, "stage_start")
    log.info("stage_start", path=metadata.source_path, dpi=dpi)

    try:
        from pdf2image import convert_from_path  # type: ignore

        pil_pages = convert_from_path(
            metadata.source_path,
            dpi=dpi,
            grayscale=True,
            thread_count=2,
        )

        document = Document(metadata=metadata)

        for page_num, pil_img in enumerate(pil_pages):
            # Deskew and normalize contrast
            pil_img = _deskew_image(pil_img)
            pil_img = _normalize_contrast(pil_img)

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            # Convert pixel dimensions to approximate pt (at 72pt/inch)
            width_pt = pil_img.width * 72.0 / dpi
            height_pt = pil_img.height * 72.0 / dpi

            page = Page(
                page_number=page_num + 1,
                width_pt=width_pt,
                height_pt=height_pt,
                column_layout=metadata.column_layout,
                image_bytes=image_bytes,
            )
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
            metrics={"pages_rasterized": len(document.pages), "dpi": dpi},
        )

    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        errors.append(ErrorCode.PAGE_RENDER_FAILED.value)
        errors.append(str(exc))
        bus.emit(stage, "stage_end_error", error=str(exc))
        log.error("stage_end", status="error", error=str(exc))
        return Document(metadata=metadata), StageResult(
            stage_name=stage,
            ok=False,
            duration_ms=duration_ms,
            errors=errors,
        )

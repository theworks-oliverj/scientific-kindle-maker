# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""
Stage 2B — Visual/Scanned PDF Extraction
Rasterizes each page to 300 DPI grayscale PNG. All downstream stages
for this source type operate on images, not PDF internal structure.

Memory: pages are rasterized to a temporary directory and encoded one at a
time, never held together. Passing all pages through memory instead peaks at
6.6 GB on a 500-page book (measured) — poppler's whole stdout is buffered as
one bytes object and then parsed into a list of PIL images that all stay
alive. `first_page`/`last_page` additionally let the caller work a page range
at a time.
"""
import io
import os
import tempfile
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
        # HoughLinesP's per-line shape varies across OpenCV builds/versions
        # — usually (1, 4), sometimes flattened to (4,) — so normalize
        # before unpacking instead of assuming line[0] is the 4-tuple.
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
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
    first_page: int | None = None,
    last_page: int | None = None,
) -> tuple[Document, StageResult]:
    """Rasterizes pages `first_page`..`last_page` (1-indexed, inclusive; None
    means the whole document). `Page.page_number` stays global, so a Document
    built from a page range still numbers its pages as they appear in the PDF."""
    t0 = time.perf_counter()
    stage = "s02b_visual"
    warnings: list[str] = []
    errors: list[str] = []

    bus.emit(stage, "stage_start")
    log.info("stage_start", path=metadata.source_path, dpi=dpi,
             first_page=first_page, last_page=last_page)

    try:
        from PIL import Image
        from pdf2image import convert_from_path  # type: ignore

        document = Document(metadata=metadata)
        page_offset = first_page if first_page else 1

        with tempfile.TemporaryDirectory(prefix="kmc_raster_") as raster_dir:
            # paths_only: poppler writes the page rasters to raster_dir and we
            # open them one at a time, so peak memory is one page, not the book.
            raster_paths = convert_from_path(
                metadata.source_path,
                dpi=dpi,
                grayscale=True,
                thread_count=2,
                first_page=first_page,
                last_page=last_page,
                output_folder=raster_dir,
                paths_only=True,
            )

            for page_num, raster_path in enumerate(raster_paths):
                src = Image.open(raster_path)
                src.load()
                # Deskew and normalize contrast. _normalize_contrast always
                # returns a fresh image, so src can be released immediately.
                pil_img = _normalize_contrast(_deskew_image(src))
                src.close()

                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                image_bytes = buf.getvalue()

                # Convert pixel dimensions to approximate pt (at 72pt/inch)
                width_pt = pil_img.width * 72.0 / dpi
                height_pt = pil_img.height * 72.0 / dpi
                pil_img.close()

                # Drop the intermediate raster as we go so disk use is bounded
                # by the batch too, not just memory.
                try:
                    os.unlink(raster_path)
                except OSError:
                    pass

                page = Page(
                    page_number=page_offset + page_num,
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

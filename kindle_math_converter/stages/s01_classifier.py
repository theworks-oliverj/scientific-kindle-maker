# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""
Stage 1 — Input Classifier
Determines source type, detects math fonts, scanned pages, column layout.
Failure mode: Fatal (abort pipeline).
"""
import time
from pathlib import Path

from ..models.document import DocumentMetadata
from ..models.enums import SourceType, ColumnLayout, ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s01_classifier")

MATH_FONT_PREFIXES = (
    "cmmi",   # Computer Modern Math Italic
    "cmsy",   # Computer Modern Math Symbols
    "cmex",   # Computer Modern Math Extensions
    "msam",   # AMS Symbol font A
    "msbm",   # AMS Symbol font B
    "stmary", # St Mary Road symbols
    "esint",  # Extended integrals
    "mtmi",   # MathTime Professional II
    "mtsy",   # MathTime Professional II symbols
)


def detect_math_fonts(pdf_path: str) -> bool:
    """
    Opens the PDF and scans the font table of the first 10 pages.
    Returns True if any math font prefix is found.
    """
    import pymupdf  # type: ignore

    doc = pymupdf.open(pdf_path)
    for page_num in range(min(10, len(doc))):
        page = doc[page_num]
        for font in page.get_fonts(full=True):
            font_name = font[3].lower()  # index 3 is the base font name
            if any(font_name.startswith(p) for p in MATH_FONT_PREFIXES):
                return True
    return False


def detect_scanned(pdf_path: str) -> bool:
    """
    A page is considered scanned if PyMuPDF returns fewer than 50 characters
    of text for it. Check first 5 pages and vote by majority.
    """
    import pymupdf  # type: ignore

    doc = pymupdf.open(pdf_path)
    scanned_count = 0
    sample = min(5, len(doc))
    for i in range(sample):
        text = doc[i].get_text("text")
        if len(text.strip()) < 50:
            scanned_count += 1
    return scanned_count > sample // 2


def detect_column_layout(pdf_path: str) -> ColumnLayout:
    """
    Heuristic column layout detection based on text block x-positions.
    If a significant number of text blocks cluster in both left and right
    halves of pages, classify as double-column.
    """
    import pymupdf  # type: ignore

    doc = pymupdf.open(pdf_path)
    sample = min(10, len(doc))
    left_blocks = 0
    right_blocks = 0

    for page_num in range(sample):
        page = doc[page_num]
        page_width = page.rect.width
        mid = page_width / 2
        blocks = page.get_text("blocks")
        for b in blocks:
            bx0, bx1 = b[0], b[2]
            center = (bx0 + bx1) / 2
            if center < mid * 0.9:
                left_blocks += 1
            elif center > mid * 1.1:
                right_blocks += 1

    total = left_blocks + right_blocks
    if total == 0:
        return ColumnLayout.SINGLE
    right_ratio = right_blocks / total
    if right_ratio > 0.3:
        return ColumnLayout.DOUBLE
    return ColumnLayout.SINGLE


def _extract_pdf_metadata(pdf_path: str) -> tuple[str | None, str | None, int]:
    import pymupdf  # type: ignore

    doc = pymupdf.open(pdf_path)
    meta = doc.metadata or {}
    title = meta.get("title") or None
    author = meta.get("author") or None
    page_count = len(doc)
    return title, author, page_count


def run(
    input_path: str,
    bus: EventBus,
) -> tuple[DocumentMetadata | None, StageResult]:
    t0 = time.perf_counter()
    stage = "s01_classifier"
    warnings: list[str] = []
    errors: list[str] = []

    bus.emit(stage, "stage_start", path=input_path)
    log.info("stage_start", path=input_path)

    path = Path(input_path)
    suffix = path.suffix.lower()

    try:
        # Determine source type from extension
        if suffix == ".epub":
            import zipfile
            if not zipfile.is_zipfile(input_path):
                errors.append(ErrorCode.EPUB_PARSE_FAILED.value)
                bus.emit(stage, "stage_end", error="not a valid EPUB (zip) container")
                return None, StageResult(
                    stage_name=stage,
                    ok=False,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    errors=errors,
                )
            source_type = SourceType.EPUB
        elif suffix in (".html", ".htm", ".xhtml"):
            source_type = SourceType.HTML
        elif suffix == ".pdf":
            source_type = _classify_pdf(input_path, warnings)
        else:
            errors.append(ErrorCode.UNSUPPORTED_FORMAT.value)
            bus.emit(stage, "stage_end", error=f"Unsupported format: {suffix}")
            return None, StageResult(
                stage_name=stage,
                ok=False,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                errors=errors,
            )

        if suffix == ".pdf":
            # Check for encryption
            import pymupdf  # type: ignore
            doc = pymupdf.open(input_path)
            if doc.is_encrypted:
                errors.append(ErrorCode.ENCRYPTED_PDF.value)
                return None, StageResult(
                    stage_name=stage,
                    ok=False,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    errors=errors,
                )

            title, author, page_count = _extract_pdf_metadata(input_path)
            has_math_fonts = detect_math_fonts(input_path)
            is_scanned = detect_scanned(input_path)
            column_layout = detect_column_layout(input_path)

            if page_count == 0:
                errors.append(ErrorCode.EMPTY_DOCUMENT.value)
                return None, StageResult(
                    stage_name=stage,
                    ok=False,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    errors=errors,
                )
        else:
            # EPUB / HTML — basic metadata
            title = path.stem
            author = None
            page_count = 0  # chapters counted later
            has_math_fonts = False
            is_scanned = False
            column_layout = ColumnLayout.SINGLE

        metadata = DocumentMetadata(
            source_path=input_path,
            source_type=source_type,
            title=title,
            author=author,
            page_count=page_count,
            has_math_fonts=has_math_fonts,
            is_scanned=is_scanned,
            column_layout=column_layout,
        )

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        bus.emit(
            stage,
            "stage_end",
            source_type=source_type.value,
            has_math_fonts=has_math_fonts,
            is_scanned=is_scanned,
            column_layout=column_layout.value,
        )
        log.info(
            "stage_end",
            source_type=source_type.value,
            has_math_fonts=has_math_fonts,
            is_scanned=is_scanned,
        )
        return metadata, StageResult(
            stage_name=stage,
            ok=True,
            duration_ms=duration_ms,
            warnings=warnings,
            errors=errors,
            metrics={
                "source_type": source_type.value,
                "page_count": page_count,
                "has_math_fonts": has_math_fonts,
                "is_scanned": is_scanned,
                "column_layout": column_layout.value,
            },
        )

    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        errors.append(str(exc))
        bus.emit(stage, "stage_end_error", error=str(exc))
        log.error("stage_end", status="error", error=str(exc))
        return None, StageResult(
            stage_name=stage,
            ok=False,
            duration_ms=duration_ms,
            errors=errors,
        )


def _classify_pdf(pdf_path: str, warnings: list[str]) -> SourceType:
    try:
        has_math = detect_math_fonts(pdf_path)
        is_scanned = detect_scanned(pdf_path)
        if is_scanned:
            return SourceType.VISUAL_PDF
        if has_math:
            return SourceType.LATEX_PDF
        return SourceType.VISUAL_PDF
    except Exception as exc:
        warnings.append(f"Font detection failed: {exc}")
        return SourceType.VISUAL_PDF

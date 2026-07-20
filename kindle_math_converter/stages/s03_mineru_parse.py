"""
Stage 3 (MinerU) — Layout, reading order, text OCR and formula recognition
in one pass.

Replaces the old s03 (YOLO detection) + s04 (reading order) + s05a
(PaddleOCR) + s05b (pix2tex): invokes the MinerU CLI (`-b vlm-engine`,
MLX-accelerated on Apple Silicon) as a subprocess and maps its middle.json
onto the in-memory Document model consumed by s06–s11.

Subprocess isolation is deliberate: MinerU's runtime (PyTorch/MLX) never
shares a process with ours, so the old dual-OpenMP SIGILL class of failure
is structurally impossible.

middle.json schema this is written against (observed from MinerU 3.4.4):
  pdf_info[] per page:
    page_idx, page_size [w_pt, h_pt]
    para_blocks[]: type ∈ {title, text, list, ref_text, interline_equation,
                           image, table}, bbox (PDF points), index (reading
                           order), lines[].spans[]:
      type ∈ {text, inline_equation, interline_equation}, content (LaTeX for
      equations), bbox
    discarded_blocks[]: headers/footers — intentionally dropped.

Inline equations are embedded in their paragraph's raw_text as
"[[EQ:region_id]]" placeholders; s10 substitutes the rendered form (HTML
text span or inline SVG) in place, keeping sentences intact.
"""
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image  # type: ignore

from ..equation_filters import (
    extract_equation_tag,
    normalize_for_dedup,
    simple_text_repr,
)
from ..models.document import BoundingBox, Document, EquationRegion, Page, TextBlock
from ..models.enums import FormulaClass
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s03_mineru_parse")

STAGE = "s03_mineru_parse"

# Block types whose lines become TextBlock paragraphs. Unknown types also
# fall through to text handling (defensive: better a paragraph than a hole).
_EQUATION_BLOCK_TYPE = "interline_equation"
_MEDIA_BLOCK_TYPES = {"image", "table"}

_CROP_PAD_PX = 4


def eq_placeholder(region_id: str) -> str:
    """Placeholder embedded in TextBlock.raw_text where an inline equation
    sits; substituted by s10. Uses only characters that survive XML escaping."""
    return f"[[EQ:{region_id}]]"


def _find_middle_json(work_dir: Path, pdf_stem: str) -> Optional[Path]:
    matches = list(work_dir.glob(f"*/*/{pdf_stem}_middle.json"))
    if matches:
        return matches[0]
    # Fallback: any middle.json under work_dir (stem may differ in unicode form)
    matches = list(work_dir.glob("*/*/*_middle.json"))
    return matches[0] if matches else None


def _invoke_mineru(
    source_pdf: str,
    work_dir: Path,
    backend: str,
    timeout_s: int,
    bus: EventBus,
) -> None:
    mineru_bin = Path(sys.executable).parent / "mineru"
    if not mineru_bin.exists():
        raise RuntimeError(
            f"mineru CLI not found at {mineru_bin} — pip install 'mineru[vlm,mlx]'"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(mineru_bin), "-p", source_pdf, "-o", str(work_dir), "-b", backend]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    log_path = work_dir / "mineru.log"
    bus.emit(STAGE, "mineru_invoke", backend=backend, log=str(log_path))
    log.info("mineru_invoke", cmd=" ".join(cmd))
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=timeout_s, env=env,
        )
    if proc.returncode != 0:
        tail = ""
        try:
            tail = log_path.read_text(errors="replace")[-2000:]
        except OSError:
            pass
        raise RuntimeError(
            f"mineru exited with code {proc.returncode}; log tail:\n{tail}"
        )


def _page_image_and_scale(page: Page, page_size: list) -> tuple[Optional[Image.Image], float, float]:
    """Opens the page raster (if any) and returns (image, sx, sy) where
    sx/sy convert MinerU PDF-point coords to raster pixels."""
    if not page.image_bytes:
        return None, 1.0, 1.0
    img = Image.open(io.BytesIO(page.image_bytes))
    img.load()
    w_pt, h_pt = float(page_size[0]) or 1.0, float(page_size[1]) or 1.0
    return img, img.width / w_pt, img.height / h_pt


def _crop_png(img: Image.Image, bbox_pts: list, sx: float, sy: float) -> Optional[bytes]:
    """Cuts bbox (PDF points) from the 300dpi page raster as PNG bytes,
    with a small pad. Returns None for degenerate boxes."""
    x0 = max(0, int(bbox_pts[0] * sx) - _CROP_PAD_PX)
    y0 = max(0, int(bbox_pts[1] * sy) - _CROP_PAD_PX)
    x1 = min(img.width, int(bbox_pts[2] * sx) + _CROP_PAD_PX)
    y1 = min(img.height, int(bbox_pts[3] * sy) + _CROP_PAD_PX)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    buf = io.BytesIO()
    img.crop((x0, y0, x1, y1)).save(buf, format="PNG")
    return buf.getvalue()


def _pixel_bbox(bbox_pts: list, sx: float, sy: float, page_number: int) -> BoundingBox:
    return BoundingBox(
        x0=bbox_pts[0] * sx, y0=bbox_pts[1] * sy,
        x1=bbox_pts[2] * sx, y1=bbox_pts[3] * sy,
        coordinate_system="image_pixels",
        page_number=page_number,
    )


def _iter_lines(block: dict):
    """Yields line dicts from a block, recursing into nested sub-blocks
    (list items, image/table captions)."""
    for line in block.get("lines", []):
        yield line
    for sub in block.get("blocks", []):
        yield from _iter_lines(sub)


def run(
    document: Document,
    bus: EventBus,
    source_pdf: str,
    work_dir: Path,
    backend: str = "vlm-engine",
    timeout_s: int = 3600,
    reuse_existing: bool = True,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []

    bus.emit(STAGE, "stage_start")
    log.info("stage_start", source=source_pdf, backend=backend)

    pdf_stem = Path(source_pdf).stem
    middle_path = _find_middle_json(work_dir, pdf_stem) if reuse_existing else None
    if middle_path is not None:
        bus.emit(STAGE, "mineru_reused", middle_json=str(middle_path))
        log.info("mineru_reused", middle_json=str(middle_path))
    else:
        try:
            _invoke_mineru(source_pdf, work_dir, backend, timeout_s, bus)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
            log.error("mineru_failed", error=str(exc))
            return document, StageResult(
                stage_name=STAGE, ok=False,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                warnings=warnings, errors=errors,
            )
        middle_path = _find_middle_json(work_dir, pdf_stem)
        if middle_path is None:
            errors.append(f"mineru produced no middle.json under {work_dir}")
            return document, StageResult(
                stage_name=STAGE, ok=False,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                warnings=warnings, errors=errors,
            )

    with open(middle_path) as f:
        middle = json.load(f)
    pdf_info = middle.get("pdf_info", [])

    if len(pdf_info) != len(document.pages):
        warnings.append(
            f"page count mismatch: mineru={len(pdf_info)} document={len(document.pages)}"
        )

    n_text_blocks = 0
    n_display = 0
    n_inline = 0
    n_rendered_as_text = 0
    n_media_skipped = 0

    for page_info in pdf_info:
        page_idx = page_info.get("page_idx", 0)
        if page_idx >= len(document.pages):
            continue
        page = document.pages[page_idx]
        page_number = page.page_number
        page_img, sx, sy = _page_image_and_scale(page, page_info.get("page_size", [612, 792]))
        eq_counter = 0

        for block in page_info.get("para_blocks", []):
            btype = block.get("type", "text")
            block_order = block.get("index", block.get("bbox", [0, 0, 0, 0])[1])
            bbox_pts = block.get("bbox", [0, 0, 0, 0])

            if btype == _EQUATION_BLOCK_TYPE:
                spans = [
                    s
                    for line in _iter_lines(block)
                    for s in line.get("spans", [])
                    if s.get("type") == _EQUATION_BLOCK_TYPE and s.get("content")
                ]
                if not spans:
                    warnings.append(f"empty interline_equation block on page {page_number}")
                    continue
                for span in spans:
                    eq_counter += 1
                    latex, number = extract_equation_tag(span["content"])
                    region = EquationRegion(
                        region_id=f"eq_{page_number}_{eq_counter}",
                        bbox=_pixel_bbox(span.get("bbox", bbox_pts), sx, sy, page_number),
                        formula_class=FormulaClass.DISPLAY,
                        source_image_crop=(
                            _crop_png(page_img, span.get("bbox", bbox_pts), sx, sy)
                            if page_img else None
                        ),
                        raw_latex=latex,
                        normalized_latex=None, cdm_score=None, confidence_gate=None,
                        svg=None, svg_postprocessed=None,
                        equation_number=number,
                        reading_order_index=block_order,
                    )
                    page.equation_regions.append(region)
                    n_display += 1
                    bus.emit(STAGE, "equation_ok", equation_id=region.region_id, raw_latex=latex)
                continue

            if btype in _MEDIA_BLOCK_TYPES:
                # Figures/tables are not embedded yet; captions still carry
                # useful text, so fall through and harvest text spans only.
                n_media_skipped += 1
                bus.emit(STAGE, "media_block_skipped", block_type=btype, page=page_number)

            # Text-like block (text, title, list, ref_text, captions, unknown)
            parts: list[str] = []
            for line in _iter_lines(block):
                for span in line.get("spans", []):
                    stype = span.get("type", "text")
                    content = (span.get("content") or "").strip()
                    if not content:
                        continue
                    if stype == "inline_equation":
                        eq_counter += 1
                        region_id = f"eq_{page_number}_{eq_counter}"
                        text_repr = simple_text_repr(content)
                        region = EquationRegion(
                            region_id=region_id,
                            bbox=_pixel_bbox(span.get("bbox", bbox_pts), sx, sy, page_number),
                            formula_class=FormulaClass.INLINE,
                            source_image_crop=(
                                _crop_png(page_img, span.get("bbox", bbox_pts), sx, sy)
                                if (page_img and text_repr is None) else None
                            ),
                            raw_latex=content,
                            normalized_latex=None, cdm_score=None, confidence_gate=None,
                            svg=None, svg_postprocessed=None,
                            equation_number=None,
                            reading_order_index=block_order,
                            render_as_text=text_repr is not None,
                            inline_text_repr=text_repr,
                        )
                        page.equation_regions.append(region)
                        parts.append(eq_placeholder(region_id))
                        n_inline += 1
                        if text_repr is not None:
                            n_rendered_as_text += 1
                    elif stype == _EQUATION_BLOCK_TYPE:
                        # Display equation embedded in a text block: keep as
                        # its own region; s10 renders it from the placeholder.
                        eq_counter += 1
                        region_id = f"eq_{page_number}_{eq_counter}"
                        latex, number = extract_equation_tag(content)
                        region = EquationRegion(
                            region_id=region_id,
                            bbox=_pixel_bbox(span.get("bbox", bbox_pts), sx, sy, page_number),
                            formula_class=FormulaClass.DISPLAY,
                            source_image_crop=(
                                _crop_png(page_img, span.get("bbox", bbox_pts), sx, sy)
                                if page_img else None
                            ),
                            raw_latex=latex,
                            normalized_latex=None, cdm_score=None, confidence_gate=None,
                            svg=None, svg_postprocessed=None,
                            equation_number=number,
                            reading_order_index=block_order,
                        )
                        page.equation_regions.append(region)
                        parts.append(eq_placeholder(region_id))
                        n_display += 1
                    else:
                        parts.append(content)

            raw_text = " ".join(parts).strip()
            if not raw_text:
                continue
            page.text_blocks.append(
                TextBlock(
                    block_id=f"tb_{page_number}_{len(page.text_blocks) + 1}",
                    bbox=_pixel_bbox(bbox_pts, sx, sy, page_number),
                    raw_text=raw_text,
                    reading_order_index=block_order,
                )
            )
            n_text_blocks += 1

        if page_img is not None:
            page_img.close()

    # D1 dedup by recognized LaTeX: canonical region gets the SVG in s08a,
    # duplicates copy it and are re-namespaced in s09.
    n_deduped = 0
    seen_by_key: dict[str, EquationRegion] = {}
    for region in document.all_equations:
        if region.render_as_text or not region.raw_latex:
            continue
        key = normalize_for_dedup(region.raw_latex)
        canonical = seen_by_key.get(key)
        if canonical is None:
            seen_by_key[key] = region
        else:
            region.dedup_canonical_id = canonical.region_id
            n_deduped += 1
            bus.emit(
                STAGE, "equation_skipped",
                equation_id=region.region_id, reason="dedup",
                canonical_id=canonical.region_id,
            )

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    metrics = {
        "text_blocks": n_text_blocks,
        "display_equations": n_display,
        "inline_equations": n_inline,
        "rendered_as_text": n_rendered_as_text,
        "deduped": n_deduped,
        "media_blocks_skipped": n_media_skipped,
    }
    bus.emit(STAGE, "stage_end", equation_id=None, **metrics)
    log.info("stage_end", **metrics)
    return document, StageResult(
        stage_name=STAGE, ok=True, duration_ms=duration_ms,
        warnings=warnings, errors=errors, metrics=metrics,
    )

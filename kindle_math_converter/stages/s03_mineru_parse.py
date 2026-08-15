# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

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
    discarded_blocks[]: type ∈ {header, footer, page_number, page_footnote}.
      The first three are page furniture and dropped; `page_footnote` is real
      content (footnote bodies, which carry citations and inline maths) and is
      harvested — see the footnote section of the page loop.

Inline equations are embedded in their paragraph's raw_text as
"[[EQ:region_id]]" placeholders; s10 substitutes the rendered form (HTML
text span or inline SVG) in place, keeping sentences intact.
"""
import io
import json
import os
import re
import shutil
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
from ..models.document import BoundingBox, Document, EquationRegion, FigureBlock, Page, TextBlock
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
# The one discarded_blocks type that is content rather than page furniture
# (the others seen in this corpus: header, footer, page_number).
_FOOTNOTE_BLOCK_TYPE = "page_footnote"

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


def _page_batches(total_pages: int, batch_size: int) -> list[tuple[int, int]]:
    """Splits the document into 0-based inclusive [start, end] page ranges.

    A batch_size of 0 (or one that covers the document) yields a single range,
    which is exactly the pre-batching behaviour.
    """
    if batch_size <= 0 or total_pages <= batch_size:
        return [(0, max(0, total_pages - 1))]
    return [
        (start, min(start + batch_size - 1, total_pages - 1))
        for start in range(0, total_pages, batch_size)
    ]


def _batch_timeout(pages: int, timeout_s: Optional[int], per_page_s: int) -> int:
    """Timeout for one batch. Derived from page count unless overridden.

    A fixed cap is the wrong shape here: 50 pages at the observed 55 s/page is
    already 46 min, and region density (not page count) drives the content
    pass, so a dense page can overrun any constant. The floor covers MinerU's
    ~10 s fixed start-up on tiny batches.
    """
    if timeout_s:
        return timeout_s
    return max(600, pages * per_page_s)


def _load_pdf_info(middle_path: Path, page_offset: int) -> list[dict]:
    """Reads one middle.json and re-bases its page indices onto the document.

    MinerU numbers pages *within the range it was given*: `-s 50 -e 99` still
    reports page_idx 0,1,2… Without this offset every batch after the first
    would overwrite the first batch's pages — silently, and with plausible
    looking output.
    """
    with open(middle_path) as f:
        middle = json.load(f)
    pages = middle.get("pdf_info", [])
    if page_offset:
        for page_info in pages:
            page_info["page_idx"] = page_info.get("page_idx", 0) + page_offset
    return pages


def _invoke_mineru(
    source_pdf: str,
    work_dir: Path,
    backend: str,
    timeout_s: int,
    bus: EventBus,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    image_analysis: bool = True,
) -> None:
    mineru_bin = Path(sys.executable).parent / "mineru"
    if not mineru_bin.exists():
        raise RuntimeError(
            f"mineru CLI not found at {mineru_bin} — pip install 'mineru[vlm,mlx]'"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(mineru_bin), "-p", source_pdf, "-o", str(work_dir), "-b", backend]
    if start_page is not None and end_page is not None:
        # -s/-e are 0-based and inclusive.
        cmd += ["-s", str(start_page), "-e", str(end_page)]
    if not image_analysis:
        # Skips the VLM's per-figure description pass. Measured as no gain on a
        # figure-sparse paper, but the work scales with figure count, so it is
        # worth having for image-heavy books. Costs FigureBlock.alt_text.
        cmd += ["--image-analysis", "false"]
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


# A standalone "^{N}" inline equation renders as "<sup>N</sup>" (see
# equation_filters.simple_text_repr) — the shape of a footnote reference in
# the body text, and of the marker that opens the footnote itself.
_SUP_MARKER_RE = re.compile(r'^<sup>(\d+)</sup>$')
_RAW_MARKER_RE = re.compile(r'^\^\s*\{\s*(\d+)\s*\}$')

# Footnote text blocks sort after every para_block on their page: they are
# pulled out of the main stream by s10 anyway, so this only orders them
# among themselves (and keeps them last if anything ever leaks through).
_FOOTNOTE_ORDER_BASE = 1_000_000


def _marker_number(region: EquationRegion) -> Optional[str]:
    """The footnote number this region marks, or None if it isn't a marker."""
    if not region.render_as_text or not region.inline_text_repr:
        return None
    m = _SUP_MARKER_RE.match(region.inline_text_repr)
    return m.group(1) if m else None


def _link_footnotes_to_markers(
    footnote_blocks: list[TextBlock],
    footnote_numbers: list[Optional[str]],
    body_markers: list[EquationRegion],
) -> int:
    """Points each body reference at the footnote it opens.

    A page footnote's marker is by definition printed on the same page, so
    matching is scoped to one page — which is what makes it safe. A pairing
    is only made when the number is unambiguous on BOTH sides: exactly one
    footnote and exactly one body marker carry it. Anything else stays
    unlinked rather than risk sending the reader to the wrong note. OCR does
    misread these — in the SU3 paper footnote 3's marker came through as "8",
    colliding with the real footnote 8 — and a body "^{2}" may simply be an
    exponent rather than a reference.

    Every footnote keeps its anchor id either way; only the link is withheld.
    Returns the number of pairs linked.
    """
    fn_counts: dict[str, int] = {}
    for number in footnote_numbers:
        if number:
            fn_counts[number] = fn_counts.get(number, 0) + 1

    markers_by_number: dict[str, list[EquationRegion]] = {}
    for region in body_markers:
        number = _marker_number(region)
        if number:
            markers_by_number.setdefault(number, []).append(region)

    linked = 0
    for block, number in zip(footnote_blocks, footnote_numbers):
        if not number or fn_counts.get(number, 0) != 1:
            continue
        candidates = markers_by_number.get(number, [])
        if len(candidates) != 1:
            continue
        candidates[0].footnote_ref_id = block.footnote_id
        linked += 1
    return linked


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
    timeout_s: Optional[int] = None,
    reuse_existing: bool = True,
    batch_size: int = 50,
    timeout_per_page_s: int = 120,
    image_analysis: bool = True,
) -> tuple[Document, StageResult]:
    """Parses `source_pdf` with MinerU and maps the result onto `document`.

    The parse runs in page-range batches, each into its own subdirectory with
    its own cached middle.json. That is what makes a long book resumable: a
    crash or timeout in batch 7 leaves batches 1–6 on disk, and a re-run picks
    up from there instead of starting a six-hour parse again.
    """
    t0 = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []

    bus.emit(STAGE, "stage_start")
    log.info("stage_start", source=source_pdf, backend=backend)

    def _failed() -> tuple[Document, StageResult]:
        return document, StageResult(
            stage_name=STAGE, ok=False,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            warnings=warnings, errors=errors,
        )

    pdf_stem = Path(source_pdf).stem
    total_pages = len(document.pages)

    # A whole-document middle.json sitting directly under work_dir is honoured
    # as-is. That covers parses made before batching existed, and — the reason
    # it matters — parses produced somewhere else entirely (a GPU box, the
    # hosted API) and dropped in. This is the external-compute seam.
    whole_doc_parse = _find_middle_json(work_dir, pdf_stem) if reuse_existing else None

    pdf_info: list[dict] = []
    if whole_doc_parse is not None:
        bus.emit(STAGE, "mineru_reused", middle_json=str(whole_doc_parse))
        log.info("mineru_reused", middle_json=str(whole_doc_parse))
        pdf_info = _load_pdf_info(whole_doc_parse, page_offset=0)
    else:
        batches = _page_batches(total_pages, batch_size)
        for batch_no, (start, end) in enumerate(batches, 1):
            n_pages = end - start + 1
            # Each batch owns a subdirectory. MinerU always writes to
            # <out>/<stem>/vlm/<stem>_middle.json, so batches sharing one
            # directory would overwrite each other and _find_middle_json would
            # return whichever survived.
            batch_dir = work_dir / f"batch_{start:05d}_{end:05d}"
            middle_path = _find_middle_json(batch_dir, pdf_stem) if reuse_existing else None

            if middle_path is not None:
                bus.emit(STAGE, "mineru_batch_reused", batch=batch_no,
                         first_page=start + 1, last_page=end + 1)
                log.info("mineru_batch_reused", batch=f"{batch_no}/{len(batches)}",
                         pages=f"{start + 1}-{end + 1}")
            else:
                bus.emit(STAGE, "mineru_batch_start", batch=batch_no,
                         batches=len(batches), first_page=start + 1, last_page=end + 1)
                log.info("mineru_batch_start", batch=f"{batch_no}/{len(batches)}",
                         pages=f"{start + 1}-{end + 1}")
                # Start from an empty directory. --fresh-parse is used exactly
                # when the existing parse is suspect, so re-invoking MinerU on
                # top of its own previous output — which it may or may not
                # overwrite wholesale — is the one thing that must not happen.
                if batch_dir.exists():
                    shutil.rmtree(batch_dir, ignore_errors=True)
                try:
                    _invoke_mineru(
                        source_pdf, batch_dir, backend,
                        _batch_timeout(n_pages, timeout_s, timeout_per_page_s),
                        bus, start_page=start, end_page=end,
                        image_analysis=image_analysis,
                    )
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    errors.append(
                        f"mineru failed on batch {batch_no}/{len(batches)} "
                        f"(pages {start + 1}-{end + 1}): {exc}"
                    )
                    log.error("mineru_failed", batch=batch_no, error=str(exc))
                    # Completed batches stay on disk — re-running resumes here.
                    return _failed()
                middle_path = _find_middle_json(batch_dir, pdf_stem)
                if middle_path is None:
                    errors.append(f"mineru produced no middle.json under {batch_dir}")
                    return _failed()

            pdf_info.extend(_load_pdf_info(middle_path, page_offset=start))

    if len(pdf_info) != len(document.pages):
        warnings.append(
            f"page count mismatch: mineru={len(pdf_info)} document={len(document.pages)}"
        )

    n_text_blocks = 0
    n_display = 0
    n_inline = 0
    n_rendered_as_text = 0
    n_figures = 0
    n_footnotes = 0
    n_footnote_links = 0

    for page_info in pdf_info:
        page_idx = page_info.get("page_idx", 0)
        if page_idx >= len(document.pages):
            continue
        page = document.pages[page_idx]
        page_number = page.page_number
        page_img, sx, sy = _page_image_and_scale(page, page_info.get("page_size", [612, 792]))
        eq_counter = 0

        def harvest_text_block(
            block: dict, block_order, bbox_pts: list, kind: str = "text"
        ) -> None:
            """Walks a text-like block's spans into a TextBlock, creating
            inline/display EquationRegions with placeholders as it goes."""
            nonlocal eq_counter, n_text_blocks, n_display, n_inline, n_rendered_as_text
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
                return
            page.text_blocks.append(
                TextBlock(
                    block_id=f"tb_{page_number}_{len(page.text_blocks) + 1}",
                    bbox=_pixel_bbox(bbox_pts, sx, sy, page_number),
                    raw_text=raw_text,
                    reading_order_index=block_order,
                    kind=kind,
                )
            )
            n_text_blocks += 1

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
                # Figure/table: embed the *_body sub-block as a raster crop
                # (its span "content" is MinerU's VLM description — used as
                # alt text, never as page text, since it may also contain
                # stray labels OCR'd from inside the drawing). Caption and
                # footnote sub-blocks are harvested as normal text.
                body_bbox: Optional[list] = None
                alt_text = ""
                table_html: Optional[str] = None
                for sub in block.get("blocks", []):
                    sub_type = sub.get("type", "")
                    if sub_type.endswith("_body"):
                        body_bbox = sub.get("bbox", bbox_pts)
                        alt_text = " ".join(
                            " ".join((s.get("content") or "").split())
                            for line in sub.get("lines", [])
                            for s in line.get("spans", [])
                        ).strip()
                        for line in sub.get("lines", []):
                            for s in line.get("spans", []):
                                if s.get("html"):
                                    table_html = s["html"]
                    else:
                        harvest_text_block(sub, block_order, sub.get("bbox", bbox_pts))

                fig_bytes = (
                    _crop_png(page_img, body_bbox or bbox_pts, sx, sy)
                    if page_img else None
                )
                figure = FigureBlock(
                    figure_id=f"fig_{page_number}_{len(page.figures) + 1}",
                    bbox=_pixel_bbox(body_bbox or bbox_pts, sx, sy, page_number),
                    image_bytes=fig_bytes,
                    alt_text=alt_text or f"{btype} on page {page_number}",
                    reading_order_index=block_order,
                    table_html=table_html,
                )
                page.figures.append(figure)
                n_figures += 1
                bus.emit(
                    STAGE, "figure_embedded",
                    figure_id=figure.figure_id, block_type=btype,
                    has_image=fig_bytes is not None,
                )
                continue

            # List-like container (list of bullets, references section):
            # nested sub-blocks are separate logical items — one paragraph
            # each, not one flattened run. Ordered within the block by y0.
            if block.get("blocks") and not block.get("lines"):
                # Fractional order preserves the sub-block array sequence —
                # a y0 tiebreak would interleave columns when the list spans
                # two columns (MinerU's array is already in reading order).
                for i, sub in enumerate(block.get("blocks", [])):
                    harvest_text_block(
                        sub, block_order + i / 1024, sub.get("bbox", bbox_pts),
                        kind="list_item",
                    )
                continue

            # Text-like block (text, title, ref_text, unknown)
            harvest_text_block(
                block, block_order, bbox_pts,
                kind="heading" if btype == "title" else "text",
            )

        # ── Page footnotes (from discarded_blocks) ───────────────────────
        # MinerU types its discard pile, and `page_footnote` entries are real
        # content — footnote bodies carrying citations and, in this corpus,
        # inline maths. Only headers/footers/page numbers are furniture.
        # Harvested through the same path as body text so their inline
        # equations become placeholders and render properly.
        #
        # Snapshot the body markers FIRST: a footnote opens with its own
        # "^{N}" span, which becomes a marker-shaped region too and would
        # otherwise pollute the pool it is matched against.
        body_markers = [r for r in page.equation_regions if _marker_number(r)]
        footnote_blocks: list[TextBlock] = []
        footnote_numbers: list[Optional[str]] = []
        for i, block in enumerate(page_info.get("discarded_blocks", [])):
            if block.get("type") != _FOOTNOTE_BLOCK_TYPE:
                continue
            before = len(page.text_blocks)
            harvest_text_block(
                block, _FOOTNOTE_ORDER_BASE + i, block.get("bbox", [0, 0, 0, 0]),
                kind="footnote",
            )
            if len(page.text_blocks) == before:
                continue  # empty after harvesting
            note = page.text_blocks[-1]
            note.footnote_id = f"fn_{page_number}_{len(footnote_blocks) + 1}"
            footnote_blocks.append(note)
            first_span = next(
                (
                    (s.get("content") or "").strip()
                    for line in _iter_lines(block)
                    for s in line.get("spans", [])
                    if (s.get("content") or "").strip()
                ),
                "",
            )
            m = _RAW_MARKER_RE.match(first_span)
            footnote_numbers.append(m.group(1) if m else None)
            n_footnotes += 1

        n_footnote_links += _link_footnotes_to_markers(
            footnote_blocks, footnote_numbers, body_markers,
        )

        if page_img is not None:
            page_img.close()

        # Release the full-page raster now that every crop for this page has
        # been cut. s03 is the only reader of Page.image_bytes — downstream
        # stages work from the much smaller per-region crops — so holding it
        # would make resident memory scale with book length for nothing
        # (~0.5–1.25 MB/page, i.e. 0.25–0.6 GB on a 500-page book).
        page.image_bytes = None

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
        "figures_embedded": n_figures,
        "footnotes": n_footnotes,
        "footnote_links": n_footnote_links,
    }
    bus.emit(STAGE, "stage_end", equation_id=None, **metrics)
    log.info("stage_end", **metrics)
    return document, StageResult(
        stage_name=STAGE, ok=True, duration_ms=duration_ms,
        warnings=warnings, errors=errors, metrics=metrics,
    )

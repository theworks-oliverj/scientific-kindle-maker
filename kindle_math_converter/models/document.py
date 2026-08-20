# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .enums import SourceType, FormulaClass, ColumnLayout, ConfidenceGate


@dataclass
class BoundingBox:
    """
    Coordinate system: origin top-left, units are points (PDF) or pixels (image).
    The stage that creates the bbox records which coordinate system it used.
    All downstream stages must respect that coordinate system or convert explicitly.
    """
    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_system: str  # "pdf_points" | "image_pixels"
    page_number: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class FailureReason:
    """
    Structured per-equation diagnostic set at the first failure point.
    Replaces the free-form error_codes list for actionable reporting.
    """
    code: str           # ErrorCode value, e.g. "E501"
    sub_code: str       # Fine-grained sub-code, e.g. "E501.undef_cmd"
    stage: str          # Stage name, e.g. "s06_validation"
    detail: str         # Human-readable: "Undefined command: \\phis"
    recoverable: bool   # True = repair loop could fix it


@dataclass
class EquationRegion:
    """
    Represents one detected equation — the central object passed between
    Stages 3 through 9.
    """
    region_id: str                            # Unique within document: "eq_{page}_{index}"
    bbox: BoundingBox
    formula_class: FormulaClass
    source_image_crop: Optional[bytes]        # PNG bytes of the cropped region, set in Stage 3
    raw_latex: Optional[str]                  # Set in Stage 5B
    normalized_latex: Optional[str]           # Set in Stage 6
    cdm_score: Optional[float]               # Set in Stage 6
    confidence_gate: Optional[ConfidenceGate] # Set in Stage 7
    svg: Optional[str]                        # Set in Stage 8A or 8B
    svg_postprocessed: Optional[str]          # Set in Stage 9
    equation_number: Optional[str]            # e.g. "(3.14)" if numbered in source
    repair_attempts: int = 0
    fallback_used: bool = False
    flagged_for_review: bool = False
    error_codes: list[str] = field(default_factory=list)
    failure_reason: Optional[FailureReason] = None
    reading_order_index: Optional[float] = None  # Unified reading order across text/equations/figures
    dedup_canonical_id: Optional[str] = None   # Set in Stage 5B: region_id of canonical duplicate
    is_reference_label: bool = False           # Set in Stage 5B: looks like a footnote/reference marker
    render_as_text: bool = False               # Set in Stage 5B: simple enough to render as HTML text
    inline_text_repr: Optional[str] = None     # Set in Stage 5B: HTML text repr when render_as_text
    footnote_ref_id: Optional[str] = None      # Set in Stage 3: this "^{N}" marks footnote <id>, linked in s10
    # Set in Stage 2C/5C (EPUB/HTML only): plain text to show if this
    # equation is flagged and has no source_image_crop to fall back to as
    # an image (true for every EPUB/HTML equation except real <img>-sourced
    # ones). For MathML, built from its own mi/mn/mo/mtext tokens (not from
    # raw_latex, which may be exactly what failed to compile). For MathJax
    # ($...$/$$...$$), the raw LaTeX itself, since there is no separate
    # source representation to tokenize. Last-resort fallback so a flagged
    # EPUB/HTML equation is never silently dropped — see
    # s10_epub_assembly._render_equation.
    text_fallback: Optional[str] = None
    # Set in Stage 6: the XDV from the compile the gate accepted. Stage 8A
    # renders SVG from this instead of compiling the identical .tex a second
    # time (both stages build the wrapper with s06's _build_latex_wrapper).
    # A few KB per equation; ~20 MB across a 2000-equation book.
    compiled_xdv: Optional[bytes] = None
    # The exact LaTeX compiled_xdv was produced from. s08A renders
    # normalized_latex, which is not always the string s06 compiled (a
    # rejected repair, or normalization that stripped a comment), so reuse is
    # gated on this matching rather than assumed — a mismatch just means s08A
    # compiles as it always did, never that it renders the wrong equation.
    compiled_xdv_latex: Optional[str] = None


@dataclass
class TextBlock:
    block_id: str
    bbox: BoundingBox
    raw_text: str
    reading_order_index: float
    source_image_crop: Optional[bytes] = None  # Masked PNG crop (scanned src), set in Stage 3; OCR'd in Stage 5A
    kind: str = "text"                         # "text" | "heading" | "list_item" | "footnote"
    footnote_id: Optional[str] = None          # kind == "footnote": target id for its noteref anchor


@dataclass
class FigureBlock:
    """A figure/table region embedded as a raster image (crop from the page
    raster). Captions are separate TextBlocks — this is just the graphic."""
    figure_id: str                 # "fig_{page}_{index}"
    bbox: BoundingBox
    image_bytes: Optional[bytes]   # PNG crop of the figure body
    alt_text: str                  # MinerU's VLM description of the image
    reading_order_index: float
    table_html: Optional[str] = None  # MinerU table HTML — preferred over the image when XML-valid
    # Set in Stage 2C (EPUB/HTML only): a short description of interactive
    # content (canvas/JS animation) this figure stands in for. When set,
    # s10 renders a plain user-visible placeholder box instead of an image —
    # there is no static image to show, and no headless browser in this
    # pipeline to render one.
    unsupported_label: Optional[str] = None


@dataclass
class Page:
    page_number: int               # 1-indexed
    width_pt: float
    height_pt: float
    column_layout: ColumnLayout
    image_bytes: Optional[bytes]   # Set in Stage 2B for visual PDFs
    text_blocks: list[TextBlock] = field(default_factory=list)
    equation_regions: list[EquationRegion] = field(default_factory=list)
    figures: list[FigureBlock] = field(default_factory=list)
    # Set in Stage 2C (EPUB/HTML only): the spine item's real title (<title>
    # or its first heading). s10 uses this as the chapter title directly
    # instead of re-deriving one from heading detection — the EPUB's own
    # spine/file boundaries are already its chapter structure.
    chapter_title: Optional[str] = None


@dataclass
class DocumentMetadata:
    source_path: str
    source_type: SourceType
    title: Optional[str]
    author: Optional[str]
    page_count: int
    has_math_fonts: bool    # True if cmmi/cmsy/cmex detected in font table
    is_scanned: bool
    column_layout: ColumnLayout  # Dominant layout of the document


@dataclass
class Document:
    metadata: DocumentMetadata
    pages: list[Page] = field(default_factory=list)

    @property
    def all_equations(self) -> list[EquationRegion]:
        return [eq for page in self.pages for eq in page.equation_regions]

    @property
    def total_equation_count(self) -> int:
        return len(self.all_equations)

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
    reading_order_index: Optional[int] = None  # Set in Stage 4 (unified with text blocks)


@dataclass
class TextBlock:
    block_id: str
    bbox: BoundingBox
    raw_text: str
    reading_order_index: int
    source_image_crop: Optional[bytes] = None  # Masked PNG crop (scanned src), set in Stage 3; OCR'd in Stage 5A


@dataclass
class Page:
    page_number: int               # 1-indexed
    width_pt: float
    height_pt: float
    column_layout: ColumnLayout
    image_bytes: Optional[bytes]   # Set in Stage 2B for visual PDFs
    text_blocks: list[TextBlock] = field(default_factory=list)
    equation_regions: list[EquationRegion] = field(default_factory=list)


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

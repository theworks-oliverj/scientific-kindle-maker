from enum import Enum


class SourceType(str, Enum):
    LATEX_PDF  = "latex_pdf"
    VISUAL_PDF = "visual_pdf"
    EPUB       = "epub"
    HTML       = "html"


class FormulaClass(str, Enum):
    INLINE  = "inline"
    DISPLAY = "display"


class ColumnLayout(str, Enum):
    SINGLE = "single"
    DOUBLE = "double"
    MIXED  = "mixed"


class ConfidenceGate(str, Enum):
    PASS     = "pass"
    REPAIR   = "repair"
    FALLBACK = "fallback"
    FLAGGED  = "flagged"


class ErrorCode(str, Enum):
    # Stage 1
    UNSUPPORTED_FORMAT       = "E001"
    ENCRYPTED_PDF            = "E002"
    EMPTY_DOCUMENT           = "E003"
    # Stage 2
    FONT_EXTRACTION_FAILED   = "E101"
    PAGE_RENDER_FAILED       = "E102"
    EPUB_PARSE_FAILED        = "E103"
    # Stage 3
    LAYOUT_MODEL_FAILED      = "E201"
    FORMULA_DETECTION_FAILED = "E202"
    # Stage 4
    READING_ORDER_AMBIGUOUS  = "E301"
    # Stage 5
    OCR_LOW_CONFIDENCE       = "E401"
    MER_FAILED               = "E402"
    # Stage 5B sub-codes (fine-grained recognition failures)
    MER_EMPTY_OUTPUT         = "E402.empty"          # pix2tex returned empty string
    MER_TOO_SHORT            = "E402.short"          # output < 3 chars — recognition likely failed
    MER_UNICODE_CHARS        = "E402.unicode"        # output has literal Unicode math chars
    CROP_TRUNCATED           = "E402.crop_truncated" # delimiter imbalance signals cut crop
    RECOGNITION_SPARSE       = "E402.recognition_sparse"  # large crop but very short LaTeX
    # Stage 6
    LATEX_COMPILE_FAILED     = "E501"
    LATEX_REPAIR_EXHAUSTED   = "E502"
    CDM_RENDER_FAILED        = "E503"
    # Stage 6 sub-codes (fine-grained validation failures)
    CDM_COMPILE_ERROR        = "E501.compile"      # tectonic returned non-zero
    CDM_UNDEFINED_CMD        = "E501.undef_cmd"    # \command not found by tectonic
    CDM_BRACE_MISMATCH       = "E501.brace"        # unmatched { } or $ signs
    CDM_SCANNED_SSIM         = "E503.scanned"      # SSIM low because source is scanned
    CDM_PLAUSIBILITY         = "E503.plausibility" # Track B plausibility below threshold
    CROP_TOO_SMALL           = "E503.crop_size"    # crop < 20×10 px — detection artifact
    # Stage 8
    TECTONIC_FAILED          = "E601"
    DVISVGM_FAILED           = "E602"
    MATHPIX_API_FAILED       = "E603"
    # Stage 9
    SVG_POSTPROCESS_FAILED   = "E701"
    # Stage 10
    EPUB_ASSEMBLY_FAILED     = "E801"
    EPUBCHECK_FAILED         = "E802"
    # Stage 11
    OUTPUT_WRITE_FAILED      = "E901"

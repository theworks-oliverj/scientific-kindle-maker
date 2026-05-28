"""
Stage 5B — Formula Recognition
Runs pix2tex (LaTeX-OCR) on all equation image crops to produce raw LaTeX.
For EPUB sources with MathML: converts MathML to LaTeX deterministically
(no model required).

This stage ALWAYS runs — it is never skipped, even for cache hits.
The cache operates only on the render step (Stage 8A).

Model: pix2tex (pip install pix2tex>=0.1.2)
API:   model(pil_image: PIL.Image) -> str   (LaTeX string)
pix2tex has no native batch API; images are processed sequentially.
This is acceptable on CPU — the bottleneck is tectonic, not pix2tex.
"""
import io
import time
from itertools import islice

from PIL import Image  # type: ignore

from ..image.crop_preprocessor import preprocess_crop_for_ocr
from ..image.crop_quality import delimiter_imbalance, recognition_sparsity
from ..models.document import Document, EquationRegion, FailureReason
from ..models.enums import ErrorCode, SourceType
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s05b_formula_recognition")


def _batched(iterable, size: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def _mathml_to_latex(mathml_str: str) -> str:
    """
    Rule-based MathML to LaTeX conversion using lxml.
    Handles common MathML elements. Returns best-effort LaTeX.
    """
    from lxml import etree  # type: ignore

    NS = "http://www.w3.org/1998/Math/MathML"

    try:
        root = etree.fromstring(mathml_str.encode())
    except etree.XMLSyntaxError:
        return mathml_str

    def convert_node(node) -> str:
        tag = node.tag.replace(f"{{{NS}}}", "") if isinstance(node.tag, str) and node.tag.startswith(f"{{{NS}}}") else (node.tag if isinstance(node.tag, str) else "")
        children = [convert_node(c) for c in node]
        text = (node.text or "").strip()

        if tag == "math":
            return " ".join(children)
        elif tag in ("mi", "mn"):
            return text
        elif tag == "mo":
            op_map = {
                "∑": r"\sum", "∏": r"\prod", "∫": r"\int",
                "±": r"\pm", "∞": r"\infty", "≤": r"\leq",
                "≥": r"\geq", "≠": r"\neq", "→": r"\to",
                "∈": r"\in", "⊂": r"\subset", "∩": r"\cap",
                "∪": r"\cup", "·": r"\cdot", "×": r"\times",
            }
            return op_map.get(text, text)
        elif tag == "msup":
            return f"{children[0]}^{{{children[1]}}}" if len(children) >= 2 else text
        elif tag == "msub":
            return f"{children[0]}_{{{children[1]}}}" if len(children) >= 2 else text
        elif tag == "msubsup":
            if len(children) >= 3:
                return f"{children[0]}_{{{children[1]}}}^{{{children[2]}}}"
            return text
        elif tag == "mfrac":
            if len(children) >= 2:
                return r"\frac{" + children[0] + "}{" + children[1] + "}"
            return text
        elif tag == "msqrt":
            return r"\sqrt{" + " ".join(children) + "}"
        elif tag == "mroot":
            if len(children) >= 2:
                return r"\sqrt[" + children[1] + "]{" + children[0] + "}"
            return text
        elif tag == "mrow":
            return " ".join(children)
        elif tag in ("mtext", "ms"):
            return r"\text{" + text + "}"
        elif tag == "mover":
            over_map = {"→": r"\vec", "^": r"\hat", "~": r"\tilde", "¯": r"\bar"}
            if len(children) >= 2:
                accent = over_map.get(children[1], r"\overset{" + children[1] + "}")
                return f"{accent}{{{children[0]}}}"
            return text
        else:
            return text + " ".join(children)

    return convert_node(root)


def _bytes_to_pil(img_bytes: bytes) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


def _run_pix2tex(model, img_bytes: bytes) -> str | None:
    """
    Runs pix2tex on a single equation image.
    Returns a LaTeX string or None on failure.

    pix2tex API: model(pil_image) -> str
    The model accepts PIL Images and returns a LaTeX string directly.
    Internally it resizes the image to its expected input resolution.
    """
    pil_img = _bytes_to_pil(img_bytes)
    if pil_img is None:
        return None
    try:
        latex = model(pil_img)
        return latex if isinstance(latex, str) else str(latex)
    except Exception:
        return None


_SPARSITY_THRESHOLD = 5_000   # px²/char — above this, crop is suspiciously sparse
_IMBALANCE_THRESHOLD = 1      # abs(open - close) above this is flagged


def _check_recognition_quality(
    region: EquationRegion,
    latex: str,
    crop_bytes: bytes | None,
    stage: str,
    bus: EventBus,
) -> None:
    """
    Runs two fast quality checks on a successful pix2tex result and annotates
    the region with diagnostic error codes.  Does NOT discard the LaTeX —
    the validation stage (s06) will use plausibility to route accordingly.

    Check 1 — delimiter balance:
      Counts (, [, \\left vs ), ], \\right.  Any imbalance means a matching
      bracket was in the part of the formula that got cut off.

    Check 2 — recognition sparsity:
      Large crop + very short LaTeX → pix2tex likely only saw fragments.
    """
    imbalance = delimiter_imbalance(latex)
    if abs(imbalance) >= _IMBALANCE_THRESHOLD:
        direction = "closing" if imbalance > 0 else "opening"
        region.error_codes.append(ErrorCode.CROP_TRUNCATED.value)
        # Only set failure_reason if none has been set yet (don't overwrite a worse one)
        if region.failure_reason is None:
            region.failure_reason = FailureReason(
                code=ErrorCode.CROP_TRUNCATED.value,
                sub_code=ErrorCode.CROP_TRUNCATED.value,
                stage=stage,
                detail=(
                    f"Delimiter imbalance {imbalance:+d} — {direction} bracket "
                    f"likely outside crop boundary"
                ),
                recoverable=False,
            )
        bus.emit(
            stage, "equation_warning",
            equation_id=region.region_id,
            error="delimiter_imbalance",
            imbalance=imbalance,
        )

    if crop_bytes is not None:
        sparsity = recognition_sparsity(crop_bytes, latex)
        if sparsity > _SPARSITY_THRESHOLD:
            region.error_codes.append(ErrorCode.RECOGNITION_SPARSE.value)
            if region.failure_reason is None:
                region.failure_reason = FailureReason(
                    code=ErrorCode.RECOGNITION_SPARSE.value,
                    sub_code=ErrorCode.RECOGNITION_SPARSE.value,
                    stage=stage,
                    detail=(
                        f"Recognition sparsity {sparsity:,.0f} px²/char — "
                        f"large crop but short LaTeX suggests truncated or blank crop"
                    ),
                    recoverable=False,
                )
            bus.emit(
                stage, "equation_warning",
                equation_id=region.region_id,
                error="recognition_sparse",
                sparsity=round(sparsity),
            )


def run(
    document: Document,
    pix2tex_model,
    bus: EventBus,
) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    stage = "s05b_formula_recognition"
    warnings: list[str] = []
    errors: list[str] = []
    recognized = 0
    failed = 0

    bus.emit(stage, "stage_start")
    log.info("stage_start", total_equations=document.total_equation_count)

    is_scanned = document.metadata.source_type == SourceType.VISUAL_PDF
    all_regions = document.all_equations

    # MathML regions (EPUB source): convert deterministically, no model needed
    mathml_regions = [r for r in all_regions if r.raw_latex and r.raw_latex.startswith("MATHML:")]
    image_regions  = [r for r in all_regions if r.source_image_crop is not None]

    for region in mathml_regions:
        try:
            mathml_str = region.raw_latex[len("MATHML:"):]
            region.raw_latex = _mathml_to_latex(mathml_str)
            bus.emit(stage, "equation_ok", equation_id=region.region_id, source="mathml_converted")
            recognized += 1
        except Exception as exc:
            region.error_codes.append(ErrorCode.MER_FAILED.value)
            warnings.append(f"MathML conversion failed for {region.region_id}: {exc}")
            failed += 1

    # Image-based regions: run pix2tex per-equation
    if pix2tex_model is None:
        # No model available — all image equations go to fallback
        for region in image_regions:
            region.error_codes.append(ErrorCode.MER_FAILED.value)
            region.failure_reason = FailureReason(
                code=ErrorCode.MER_FAILED.value,
                sub_code=ErrorCode.MER_FAILED.value,
                stage=stage,
                detail="pix2tex model not loaded — install pix2tex and re-run download-models",
                recoverable=False,
            )
            bus.emit(stage, "equation_error", equation_id=region.region_id, error="pix2tex_unavailable")
        failed += len(image_regions)
        if image_regions:
            errors.append("pix2tex model not loaded — install pix2tex and re-run download-models")
    else:
        for region in image_regions:
            crop = (
                preprocess_crop_for_ocr(region.source_image_crop)
                if is_scanned and region.source_image_crop
                else region.source_image_crop
            )
            latex = _run_pix2tex(pix2tex_model, crop)
            if latex is None:
                region.error_codes.append(ErrorCode.MER_FAILED.value)
                region.failure_reason = FailureReason(
                    code=ErrorCode.MER_FAILED.value,
                    sub_code=ErrorCode.MER_FAILED.value,
                    stage=stage,
                    detail="pix2tex raised an exception — see pipeline log",
                    recoverable=False,
                )
                bus.emit(stage, "equation_error", equation_id=region.region_id, error="pix2tex_failed")
                failed += 1
            elif not latex.strip():
                region.error_codes.append(ErrorCode.MER_EMPTY_OUTPUT.value)
                region.failure_reason = FailureReason(
                    code=ErrorCode.MER_FAILED.value,
                    sub_code=ErrorCode.MER_EMPTY_OUTPUT.value,
                    stage=stage,
                    detail="pix2tex returned empty string — crop may be too small or blank",
                    recoverable=False,
                )
                bus.emit(stage, "equation_error", equation_id=region.region_id, error="pix2tex_empty")
                failed += 1
            elif len(latex.strip()) < 3:
                region.error_codes.append(ErrorCode.MER_TOO_SHORT.value)
                region.failure_reason = FailureReason(
                    code=ErrorCode.MER_FAILED.value,
                    sub_code=ErrorCode.MER_TOO_SHORT.value,
                    stage=stage,
                    detail=f"pix2tex output too short ({len(latex.strip())} chars): {latex!r}",
                    recoverable=False,
                )
                bus.emit(stage, "equation_error", equation_id=region.region_id, error="pix2tex_too_short")
                failed += 1
            else:
                region.raw_latex = latex
                # Quality checks: detect likely crop truncation before validation
                _check_recognition_quality(region, latex, crop, stage, bus)
                bus.emit(
                    stage,
                    "equation_ok",
                    equation_id=region.region_id,
                    raw_latex=latex,
                )
                recognized += 1

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    bus.emit(stage, "stage_end", recognized=recognized, failed=failed)
    log.info("stage_end", recognized=recognized, failed=failed)

    return document, StageResult(
        stage_name=stage,
        ok=True,
        duration_ms=duration_ms,
        warnings=warnings,
        errors=errors,
        metrics={"recognized": recognized, "failed": failed},
    )

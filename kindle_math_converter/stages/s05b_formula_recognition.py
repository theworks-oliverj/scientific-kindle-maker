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
import json
import re
import time
from itertools import islice
from pathlib import Path
from typing import Optional

from PIL import Image  # type: ignore

from ..image.crop_preprocessor import preprocess_crop_for_ocr
from ..image.crop_quality import delimiter_imbalance, recognition_sparsity
from ..models.document import Document, EquationRegion, FailureReason
from ..models.enums import ErrorCode, FormulaClass, SourceType
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


# ── Stage 1.2: simple inline expressions rendered as plain text ───────────

_GREEK_MAP = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "varpi": "π", "rho": "ρ", "varrho": "ρ", "sigma": "σ",
    "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

# \mathbb{} letters that exist as single codepoints in the BMP "Letterlike
# Symbols" block (U+2100-214F). Any other letter falls through to SVG —
# never emit codepoints from the Mathematical Alphanumeric Symbols block
# (U+1D400+), which is commonly missing from e-ink fonts.
_BLACKBOARD_MAP = {
    "N": "ℕ", "R": "ℝ", "Z": "ℤ", "C": "ℂ", "Q": "ℚ", "P": "ℙ",
}


def _resolve_symbol(token: str) -> Optional[str]:
    """
    Resolves `token` to a single displayable character if it is a single
    Latin letter/digit or a recognized Greek macro (e.g. "\\Psi"). Returns
    None if `token` isn't a single base symbol.
    """
    token = token.strip()
    if len(token) == 1 and (token.isalpha() or token.isdigit()):
        return token
    m = re.fullmatch(r'\\([A-Za-z]+)', token)
    if m:
        return _GREEK_MAP.get(m.group(1))
    return None


def _take_group_or_token(s: str) -> Optional[tuple[str, str]]:
    """
    Consumes one "unit" from the start of `s`: either a brace-delimited
    group `{...}` (returning its inner content) or a single token (a LaTeX
    command `\\xyz` or one character). Returns (unit, remainder), or None if
    `s` is empty or has unbalanced braces.
    """
    if not s:
        return None
    if s[0] == "{":
        depth = 0
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[1:i], s[i + 1:]
        return None
    m = re.match(r'\\[A-Za-z]+|.', s)
    if not m:
        return None
    return m.group(0), s[m.end():]


_SIMPLE_SUFFIX_RE = re.compile(r'^[.,;:)\]]*(=\s*\d+)?[.,;:)\]]*$')
_OUTER_PUNCT_RE = re.compile(r'^([(\[]*)(.*?)([)\].,;:]*)$')


def _simple_text_repr(latex: str) -> Optional[str]:
    """
    If `latex` is simple enough to render as plain HTML text — one base
    symbol (a Latin letter, or one wrapped in \\mathbf{}/\\mathrm{}/\\mathbb{}),
    optionally one _{...}/^{...}, optionally simple trailing punctuation or
    "=N" — returns the HTML snippet. Returns None for anything else, which
    keeps using the SVG pipeline. See plan 1.2's "Design discussion".
    """
    s = latex.strip()
    if not s:
        return None

    wrapper: Optional[str] = None
    remainder = s

    m = re.match(r'\\(mathbf|mathrm|mathbb)\{', remainder)
    if m:
        depth = 0
        for i in range(m.end() - 1, len(remainder)):
            if remainder[i] == "{":
                depth += 1
            elif remainder[i] == "}":
                depth -= 1
                if depth == 0:
                    wrapper = m.group(1)
                    base_str = remainder[m.end():i]
                    remainder = remainder[i + 1:]
                    break
        else:
            return None
    else:
        gm = re.match(r'\\[A-Za-z]+', remainder)
        if gm:
            base_str = gm.group(0)
            remainder = remainder[gm.end():]
        elif remainder[0].isalpha() or remainder[0].isdigit():
            base_str = remainder[0]
            remainder = remainder[1:]
        else:
            return None

    pm = _OUTER_PUNCT_RE.match(base_str)
    assert pm is not None  # pattern matches any string (all groups allow empty)
    pre_punct, base_str, post_punct = pm.group(1), pm.group(2), pm.group(3)

    base_char = _resolve_symbol(base_str)
    if base_char is None:
        return None

    if wrapper == "mathbb":
        base_char = _BLACKBOARD_MAP.get(base_char)
        if base_char is None:
            return None
    elif wrapper == "mathbf":
        base_char = f"<b>{base_char}</b>"

    out = f"{pre_punct}{base_char}{post_punct}"

    if remainder[:1] in ("_", "^"):
        marker = remainder[0]
        token = _take_group_or_token(remainder[1:])
        if token is None:
            return None
        content, remainder = token
        content = content.strip()
        if re.fullmatch(r'\d+', content):
            sub_content = content
        else:
            sub_char = _resolve_symbol(content)
            if sub_char is None:
                m2 = re.fullmatch(r'\\(mathrm|mathbf)\{(.+)\}', content)
                sub_char = _resolve_symbol(m2.group(2)) if m2 else None
            if sub_char is None:
                return None
            sub_content = sub_char
        tag = "sub" if marker == "_" else "sup"
        out = f"{out}<{tag}>{sub_content}</{tag}>"

    suffix = remainder.strip()
    if suffix:
        if not _SIMPLE_SUFFIX_RE.match(suffix):
            return None
        out += suffix.replace(" ", "")

    return out


# ── Stage 1.4: reference/footnote-marker filter ───────────────────────────

_GREEK_STRIP_RES = [(re.compile(rf'\\{name}\b'), ch) for name, ch in _GREEK_MAP.items()]
_LATEX_COMMAND_RE = re.compile(r'\\[a-zA-Z]+|\\[!,;:]')
_SUBSUP_MARKER_RE = re.compile(r'[\^_]')
_BRACE_RE = re.compile(r'[{}]')

_EQUATION_NUMBER_LABEL_RE = re.compile(r'^\(?\s*\d+(?:\.\d+)*\s*\)?$')
_FOOTNOTE_MARKER_RE = re.compile(
    r'^(\d{1,3}[A-Za-zΑ-ω]{0,4}[.,]?|[A-Za-z]{1,2}\d{1,3}[.,]?)$'
)


def _strip_latex_formatting(latex: str) -> str:
    """
    Reduces `latex` to its literal "content" characters: resolves Greek
    macros to their Unicode letters, then drops all remaining LaTeX commands,
    braces and sub/superscript markers. Used by `_is_reference_label` to spot
    footnote/citation markers and equation-number labels mis-detected as
    formulas.
    """
    s = latex
    for pattern, ch in _GREEK_STRIP_RES:
        s = pattern.sub(ch, s)
    s = _LATEX_COMMAND_RE.sub('', s)
    s = s.replace('\\', '')
    s = _SUBSUP_MARKER_RE.sub('', s)
    s = _BRACE_RE.sub('', s)
    return s.strip()


def _is_reference_label(latex: str) -> bool:
    """
    Returns True if `latex` looks like a footnote/citation marker (e.g.
    "34A.", "S20", "60See.") or an equation-number label (e.g. "(9)") rather
    than a real formula. See plan 1.4 for the evidence.
    """
    core = _strip_latex_formatting(latex)
    if not core:
        return False
    return bool(_EQUATION_NUMBER_LABEL_RE.match(core) or _FOOTNOTE_MARKER_RE.match(core))


def _normalize_for_dedup(latex: str) -> str:
    return re.sub(r'\s+', ' ', latex.strip())


def _assign_recovered_equation_numbers(regions: list[EquationRegion]) -> None:
    """
    For `is_reference_label` regions whose stripped LaTeX is a genuine
    equation-number label (e.g. "(9)"), finds the nearest non-label display
    equation to its left on the same page row and sets its `equation_number`.
    Mirrors s04_reading_order._assign_equation_numbers, but runs after
    recognition so it also works for scanned sources (where s04 has no LaTeX
    to go on yet).
    """
    for region in regions:
        if not region.is_reference_label or not region.raw_latex:
            continue
        core = _strip_latex_formatting(region.raw_latex)
        if not _EQUATION_NUMBER_LABEL_RE.match(core):
            continue
        number = core if core.startswith("(") else f"({core})"

        best: Optional[EquationRegion] = None
        for other in regions:
            if (
                other is region
                or other.is_reference_label
                or other.render_as_text
                or other.dedup_canonical_id is not None
                or other.formula_class != FormulaClass.DISPLAY
                or other.bbox.page_number != region.bbox.page_number
            ):
                continue
            same_row = abs(
                (other.bbox.y0 + other.bbox.y1) - (region.bbox.y0 + region.bbox.y1)
            ) < 2 * region.bbox.height
            to_the_left = other.bbox.x1 <= region.bbox.x0
            if same_row and to_the_left and (best is None or other.bbox.x1 > best.bbox.x1):
                best = other

        if best is not None and not best.equation_number:
            best.equation_number = number


def run(
    document: Document,
    pix2tex_model,
    bus: EventBus,
    debug_dump_dir: Optional[str] = None,
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

            if (recognized + failed) % 25 == 0:
                log.info(
                    "pix2tex_progress",
                    done=recognized + failed,
                    total=len(image_regions),
                    recognized=recognized,
                    failed=failed,
                )

    # ── Stage 1.2: render simple inline expressions as text ───────────────
    rendered_as_text = 0
    for region in mathml_regions + image_regions:
        if not region.raw_latex:
            continue
        text_repr = _simple_text_repr(region.raw_latex)
        if text_repr is not None:
            region.render_as_text = True
            region.inline_text_repr = text_repr
            rendered_as_text += 1
            bus.emit(stage, "equation_skipped", equation_id=region.region_id, reason="render_as_text")

    # ── Stage 1.4: reference/footnote-marker filter ────────────────────────
    skipped_as_reference_label = 0
    for region in all_regions:
        if region.render_as_text or not region.raw_latex:
            continue
        if _is_reference_label(region.raw_latex):
            region.is_reference_label = True
            skipped_as_reference_label += 1
            bus.emit(stage, "equation_skipped", equation_id=region.region_id, reason="reference_label")

    _assign_recovered_equation_numbers(all_regions)

    # ── Stage 1.3: dedup by recognized LaTeX (D1) ───────────────────────────
    deduped = 0
    seen_by_key: dict[str, EquationRegion] = {}
    for region in all_regions:
        if region.render_as_text or region.is_reference_label or not region.raw_latex:
            continue
        key = _normalize_for_dedup(region.raw_latex)
        canonical = seen_by_key.get(key)
        if canonical is None:
            seen_by_key[key] = region
        else:
            region.dedup_canonical_id = canonical.region_id
            deduped += 1
            bus.emit(
                stage, "equation_skipped",
                equation_id=region.region_id, reason="dedup",
                canonical_id=canonical.region_id,
            )

    if debug_dump_dir is not None:
        debug_dir = Path(debug_dump_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_dir / "recognition.jsonl", "w") as f:
            for region in all_regions:
                f.write(json.dumps({
                    "region_id": region.region_id,
                    "raw_latex": region.raw_latex,
                    "render_as_text": region.render_as_text,
                    "inline_text_repr": region.inline_text_repr,
                    "is_reference_label": region.is_reference_label,
                    "dedup_canonical_id": region.dedup_canonical_id,
                    "equation_number": region.equation_number,
                }) + "\n")

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    bus.emit(
        stage, "stage_end",
        recognized=recognized, failed=failed,
        rendered_as_text=rendered_as_text,
        skipped_as_reference_label=skipped_as_reference_label,
        deduped=deduped,
    )
    log.info(
        "stage_end",
        recognized=recognized, failed=failed,
        rendered_as_text=rendered_as_text,
        skipped_as_reference_label=skipped_as_reference_label,
        deduped=deduped,
    )

    return document, StageResult(
        stage_name=stage,
        ok=True,
        duration_ms=duration_ms,
        warnings=warnings,
        errors=errors,
        metrics={
            "recognized": recognized,
            "failed": failed,
            "rendered_as_text": rendered_as_text,
            "skipped_as_reference_label": skipped_as_reference_label,
            "deduped": deduped,
        },
    )

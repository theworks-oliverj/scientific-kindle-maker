# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""
Stage 5C — MathML conversion (EPUB/HTML sources only).

EPUB sources bypass MinerU (s03_mineru_parse); their equations arrive from
s02c as raw MathML tagged "MATHML:...". This stage converts them to LaTeX
deterministically (no model), then applies the recognizer-agnostic filters
(simple-text rendering, D1 dedup) that s03_mineru_parse applies for PDF
sources. Formerly part of s05b_formula_recognition.
"""
import time

from ..equation_filters import normalize_for_dedup, simple_text_repr
from ..models.document import Document, EquationRegion
from ..models.enums import ErrorCode
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s05c_mathml")

STAGE = "s05c_mathml"

# TeX-encoding spellings seen in the wild for <semantics><annotation>.
_TEX_ANNOTATION_ENCODINGS = {"application/x-tex", "tex", "latex", "text/x-tex"}

# Operators MinerU's own \\tag-less recognizer never produces raw, but real
# publisher MathML uses constantly — mapped the same way the existing table
# below maps \\sum/\\pm/etc. Extended (not replaced) 2026-08-19 against a
# real ~2200-equation fixture (see memory/pipeline state): mfenced (1168
# occurrences) and mtable (340) were entirely unhandled; munder/munderover
# (493/47) fell through to the generic accent-map guess mover uses, which is
# wrong for the dominant real case (limits on an operator, not an accent).
_OP_MAP = {
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int",
    "±": r"\pm", "∞": r"\infty", "≤": r"\leq",
    "≥": r"\geq", "≠": r"\neq", "→": r"\to",
    "∈": r"\in", "⊂": r"\subset", "∩": r"\cap",
    "∪": r"\cup", "·": r"\cdot", "×": r"\times",
}


def _mathml_annotation_tex(mathml_str: str) -> "str | None":
    """Looks for <semantics><annotation encoding="...tex..."> inside the raw
    MathML and returns its text verbatim if present — authoritative
    publisher LaTeX, strictly better than anything the recursive converter
    below can derive from the presentation tree. Not assumed to usually be
    present (0/2200 in the real fixture this was built against) — the
    derivation path still has to carry the load on its own."""
    from lxml import etree  # type: ignore

    try:
        root = etree.fromstring(mathml_str.encode())
    except etree.XMLSyntaxError:
        return None
    for ann in root.iter():
        tag = ann.tag.rsplit("}", 1)[-1] if isinstance(ann.tag, str) else ""
        if tag != "annotation":
            continue
        encoding = (ann.get("encoding") or "").strip().lower()
        if encoding in _TEX_ANNOTATION_ENCODINGS and (ann.text or "").strip():
            return ann.text.strip()
    return None


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
            return _OP_MAP.get(text, text)
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
        elif tag == "munder":
            # Dominant real case: a limit under an operator (\sum, \lim,
            # \prod) — a plain subscript is the correct LaTeX for that,
            # unlike mover's accent-map approach which doesn't apply here.
            if len(children) >= 2:
                return f"{children[0]}_{{{children[1]}}}"
            return text
        elif tag == "munderover":
            if len(children) >= 3:
                return f"{children[0]}_{{{children[1]}}}^{{{children[2]}}}"
            return text
        elif tag == "mfenced":
            open_ch = node.get("open", "(")
            close_ch = node.get("close", ")")
            seps = node.get("separators", ",")
            # mfenced's separator string is positional (one char per gap
            # between children); repeat the last one if there are more gaps
            # than characters, matching the MathML spec's own fallback rule.
            parts: list[str] = []
            for i, child_latex in enumerate(children):
                if i > 0:
                    sep = seps[min(i - 1, len(seps) - 1)] if seps else ""
                    parts.append(sep)
                parts.append(child_latex)
            inner = " ".join(parts)
            left = f"\\left{open_ch}" if open_ch else r"\left."
            right = f"\\right{close_ch}" if close_ch else r"\right."
            return f"{left} {inner} {right}"
        elif tag == "mtable":
            rows = [c for c in children if c is not None]
            return r"\begin{matrix}" + r" \\ ".join(rows) + r"\end{matrix}"
        elif tag == "mtr":
            return " & ".join(children)
        elif tag == "mtd":
            return " ".join(children)
        elif tag in ("mstyle", "mpadded"):
            # Transparent wrapper — passes its children through unchanged.
            # Explicit rather than left to the catch-all below: mstyle is
            # the single most frequent tag in real publisher MathML (14636
            # occurrences in the fixture this was calibrated against).
            return " ".join(children)
        elif tag == "mspace":
            return ""
        else:
            return text + " ".join(children)

    return convert_node(root)


def run(document: Document, bus: EventBus) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    warnings: list[str] = []
    converted = 0
    failed = 0
    rendered_as_text = 0
    deduped = 0
    from_annotation = 0

    bus.emit(STAGE, "stage_start")

    mathml_regions = [
        r for r in document.all_equations
        if r.raw_latex and r.raw_latex.startswith("MATHML:")
    ]

    for region in mathml_regions:
        try:
            mathml_src = (region.raw_latex or "")[len("MATHML:"):]
            region.text_fallback = _mathml_text_tokens(mathml_src)
            tex = _mathml_annotation_tex(mathml_src)
            if tex is not None:
                region.raw_latex = tex
                from_annotation += 1
            else:
                region.raw_latex = _mathml_to_latex(mathml_src)
            bus.emit(STAGE, "equation_ok", equation_id=region.region_id, source="mathml_converted")
            converted += 1
        except Exception as exc:
            region.error_codes.append(ErrorCode.MER_FAILED.value)
            warnings.append(f"MathML conversion failed for {region.region_id}: {exc}")
            failed += 1

    for region in mathml_regions:
        if not region.raw_latex:
            continue
        text_repr = simple_text_repr(region.raw_latex)
        if text_repr is not None:
            region.render_as_text = True
            region.inline_text_repr = text_repr
            rendered_as_text += 1
            bus.emit(STAGE, "equation_skipped", equation_id=region.region_id, reason="render_as_text")

    seen_by_key: dict[str, EquationRegion] = {}
    for region in mathml_regions:
        if region.render_as_text or not region.raw_latex:
            continue
        key = normalize_for_dedup(region.raw_latex)
        canonical = seen_by_key.get(key)
        if canonical is None:
            seen_by_key[key] = region
        else:
            region.dedup_canonical_id = canonical.region_id
            deduped += 1
            bus.emit(
                STAGE, "equation_skipped",
                equation_id=region.region_id, reason="dedup",
                canonical_id=canonical.region_id,
            )

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    metrics = {
        "converted": converted,
        "failed": failed,
        "rendered_as_text": rendered_as_text,
        "deduped": deduped,
        "from_annotation": from_annotation,
    }
    bus.emit(STAGE, "stage_end", equation_id=None, **metrics)
    log.info("stage_end", **metrics)
    return document, StageResult(
        stage_name=STAGE, ok=True, duration_ms=duration_ms,
        warnings=warnings, errors=[], metrics=metrics,
    )


def _mathml_text_tokens(mathml_str: str) -> "str | None":
    """Plain space-joined text from MathML's own mi/mn/mo/mtext tokens, in
    document order — the last-resort fallback s10_epub_assembly renders when
    an equation is flagged and has no source_image_crop to fall back to as
    an image. Derived from the MathML DOM directly (not from raw_latex,
    which may be exactly the string that failed to compile) so it survives
    even a total derivation/compile failure."""
    from lxml import etree  # type: ignore

    try:
        root = etree.fromstring(mathml_str.encode())
    except etree.XMLSyntaxError:
        return None

    tokens: list[str] = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1] if isinstance(el.tag, str) else ""
        if tag in ("mi", "mn", "mo", "mtext"):
            text = (el.text or "").strip()
            if text:
                tokens.append(text)
    joined = " ".join(tokens).strip()
    return joined or None

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


def run(document: Document, bus: EventBus) -> tuple[Document, StageResult]:
    t0 = time.perf_counter()
    warnings: list[str] = []
    converted = 0
    failed = 0
    rendered_as_text = 0
    deduped = 0

    bus.emit(STAGE, "stage_start")

    mathml_regions = [
        r for r in document.all_equations
        if r.raw_latex and r.raw_latex.startswith("MATHML:")
    ]

    for region in mathml_regions:
        try:
            mathml_src = (region.raw_latex or "")[len("MATHML:"):]
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
    }
    bus.emit(STAGE, "stage_end", equation_id=None, **metrics)
    log.info("stage_end", **metrics)
    return document, StageResult(
        stage_name=STAGE, ok=True, duration_ms=duration_ms,
        warnings=warnings, errors=[], metrics=metrics,
    )

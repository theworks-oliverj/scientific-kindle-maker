# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""
Stage 6 — LaTeX Validation and Repair Loop

One validation gate, applied to every source (PDF, EPUB, HTML alike):
  1. Applies Unicode → LaTeX substitution (Tier 1).
  2. Applies structural repair rules (Tier 2).
  3. Compiles with tectonic (binary compile check).
  4. Scores plausibility of the LaTeX string (heuristic 0–1).
Thresholds: pass >= 0.65, repair >= 0.40, fallback < 0.40.

Repair rules are applied before the first compile attempt.

A second track (SSIM against the source page crop, for digital-PDF sources)
existed historically but was removed: it mass-flagged correct recognitions
over mere font differences between the source and tectonic's render, so
`is_scanned` was hardcoded True for every source — see run() — which made
that branch permanently unreachable.
"""
import io
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from ..cache.equation_cache import PersistentCompileCache
from ..concurrency import parallel_for_each
from ..models.document import Document, EquationRegion, FailureReason
from ..models.enums import ConfidenceGate, ErrorCode, FormulaClass
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger

log = get_logger("s06_validation")

# ---------------------------------------------------------------------------
# Tier 1 — Unicode character → LaTeX command map
# pix2tex frequently outputs literal Unicode characters (especially Greek)
# instead of LaTeX commands. These break tectonic immediately.
# ---------------------------------------------------------------------------
UNICODE_TO_LATEX: dict[str, str] = {
    # Lowercase Greek
    "α": r"\alpha",   "β": r"\beta",    "γ": r"\gamma",   "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta",    "η": r"\eta",     "θ": r"\theta",
    "ι": r"\iota",    "κ": r"\kappa",   "λ": r"\lambda",  "μ": r"\mu",
    "ν": r"\nu",      "ξ": r"\xi",      "π": r"\pi",      "ρ": r"\rho",
    "σ": r"\sigma",   "τ": r"\tau",     "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi",     "ψ": r"\psi",     "ω": r"\omega",
    # Uppercase Greek
    "Γ": r"\Gamma",   "Δ": r"\Delta",   "Θ": r"\Theta",   "Λ": r"\Lambda",
    "Ξ": r"\Xi",      "Π": r"\Pi",      "Σ": r"\Sigma",   "Υ": r"\Upsilon",
    "Φ": r"\Phi",     "Ψ": r"\Psi",     "Ω": r"\Omega",
    # Common math symbols that tectonic won't accept as Unicode
    "∇": r"\nabla",   "∂": r"\partial", "ℏ": r"\hbar",    "∞": r"\infty",
    "≈": r"\approx",  "≡": r"\equiv",   "∝": r"\propto",  "≠": r"\neq",
    "≤": r"\leq",     "≥": r"\geq",     "⟨": r"\langle",  "⟩": r"\rangle",
    "†": r"\dagger",  "‡": r"\ddagger", "·": r"\cdot",    "∓": r"\mp",
    "⊗": r"\otimes",  "⊕": r"\oplus",   "ℓ": r"\ell",     "ℜ": r"\Re",
    "ℑ": r"\Im",      "∈": r"\in",      "∉": r"\notin",   "⊂": r"\subset",
    "⊃": r"\supset",  "∩": r"\cap",     "∪": r"\cup",     "∀": r"\forall",
    "∃": r"\exists",  "→": r"\to",      "←": r"\leftarrow","↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow","⇐": r"\Leftarrow","↦": r"\mapsto", "×": r"\times",
    "÷": r"\div",     "±": r"\pm",      "′": "'",          "−": "-",
    "·": r"\cdot",    "∑": r"\sum",     "∏": r"\prod",    "∫": r"\int",
    # Dash variants — all map to ASCII hyphen-minus so tectonic can compile them
    "—": "-",   # em-dash U+2014
    "–": "-",   # en-dash U+2013
    "‐": "-",   # hyphen U+2010
    "‑": "-",   # non-breaking hyphen U+2011
    "‒": "-",   # figure dash U+2012
    "―": "-",   # horizontal bar U+2015
}


def _apply_unicode_substitutions(latex: str) -> str:
    for char, cmd in UNICODE_TO_LATEX.items():
        latex = latex.replace(char, cmd)
    return latex


# A row break immediately followed by "[" is read by LaTeX as the optional
# vertical-space form \\[<dimen>], so "\\ [ A, B ] = 0" fails with "Missing
# number, treated as zero". MinerU emits exactly this shape for multi-line
# commutator relations (\begin{array}{l} ... \\ [ H_1, H_2 ] = 0 ...).
# Inserting an empty group makes the "[" ordinary again.
#
# The negative lookahead spares a REAL \\[<dimen>] (e.g. "\\[2pt]"), which
# EPUB/MathML sources do emit — bracing that would typeset a literal "[2pt]".
_ROW_BREAK_BRACKET_RE = re.compile(
    r"(\\\\)(\s*)\[(?!\s*-?[\d.]+\s*(?:pt|em|ex|mm|cm|in|bp|pc|dd|cc|sp|mu)\s*\])"
)


def _fix_row_break_bracket(latex: str) -> str:
    return _ROW_BREAK_BRACKET_RE.sub(r"\1\2{}[", latex)


# ---------------------------------------------------------------------------
# Tier 2 — Structural regex repair rules
# Applied after Tier 1. Order matters: subscript/superscript rules last.
# ---------------------------------------------------------------------------
REPAIR_RULES: list[tuple[str | re.Pattern, str | Callable]] = [
    # Command name typos from pix2tex
    (r"\\In\b",               r"\\int"),
    (r"\\Sum\b",              r"\\sum"),
    (r"\\Prod\b",             r"\\prod"),
    (r"\\lim\s+it\b",         r"\\lim"),
    # Greek case confusion — pix2tex on scanned docs confuses similar-looking
    # upper/lower case pairs. In physics equations \Lambda in subscript/exponent
    # position is extremely rare; prefer \lambda. Same for \Eta→\eta, \Nu→\nu.
    # These run before the subscript-bracing rules so the context is still raw.
    (r"\\Lambda(?=\s*[_^{}\s]|$)", r"\\lambda"),  # Λ→λ when in operator position
    (r"\{\\Lambda\}",              r"{\\lambda}"),  # Λ inside braces → λ
    (r"\\Eta\b",                   r"\\eta"),
    (r"\\Nu\b",                    r"\\nu"),
    # Row break followed by "[": see _fix_row_break_bracket. Kept here too so
    # the repair loop still heals it if a later rule ever re-introduces it.
    (_ROW_BREAK_BRACKET_RE,   r"\1\2{}["),
    # Decorated letters — normalize spacing
    (r"\\vec\s*\{([^}]+)\}",  r"\\vec{\1}"),
    (r"\\hat\s*\{([^}]+)\}",  r"\\hat{\1}"),
    (r"\\dot\s*\{([^}]+)\}",  r"\\dot{\1}"),
    (r"\\ddot\s*\{([^}]+)\}", r"\\ddot{\1}"),
    (r"\\bar\s*\{([^}]+)\}",  r"\\bar{\1}"),
    (r"\\tilde\s*\{([^}]+)\}",r"\\tilde{\1}"),
    # Subscript/superscript: bare multi-char arguments need braces
    # e.g. x_ij → x_{ij}, e^-x → e^{-x}
    (r"_([^\{])([^\s\^\_{}\)]+)", r"_{\1\2}"),
    (r"\^([^\{])([^\s\^\_{}\)]+)", r"^{\1\2}"),
]

COMPILED_RULES = [
    (re.compile(pat) if isinstance(pat, str) else pat, repl)
    for pat, repl in REPAIR_RULES
]


def _apply_repair_rules(latex: str) -> str:
    for pattern, repl in COMPILED_RULES:
        latex = pattern.sub(repl, latex)
    return latex


# ---------------------------------------------------------------------------
# Package detection — auto-extend \usepackage based on commands present
# ---------------------------------------------------------------------------
_CMD_TO_PKG: dict[str, str] = {
    r"\hbar":        "physics",
    r"\bra":         "braket",
    r"\ket":         "braket",
    r"\braket":      "braket",
    r"\mathbb":      "amssymb",
    r"\boldsymbol":  "amssymb",
    r"\mathcal":     "amsfonts",
    r"\dagger":      "amssymb",
    r"\ddagger":     "amssymb",
    r"\otimes":      "amssymb",
    r"\oplus":       "amssymb",
    r"\langle":      "amsmath",
    r"\rangle":      "amsmath",
}
_BASE_PACKAGES = "amsmath,amssymb,amsfonts,physics"

# Body font size the wrapper renders at. Stage 9 divides the SVG's pt
# dimensions by this to get em, so 1em == one body-text line — keep the two
# in lockstep or equations stop matching the reader's font size.
# article only accepts 10, 11 or 12 as the \documentclass size.
LATEX_BODY_PT = 12.0

# Force Computer Modern for text-mode glyphs.
#
# tectonic's default bundle maps the roman family to Latin Modern, which ships
# only as .otf. dvisvgm cannot embed those (no psfonts.map, no lm*.pfb in the
# bundle cache), so ANY equation containing \text{...} died with
# "ERROR: failed to release font" and fell back to a raster crop. cmr12.pfb IS
# in the bundle, and _dvisvgm_env() already puts that directory on TEXFONTS.
#
# Both halves are needed: [OT1]{fontenc} alone still asks for lmr12.pfb (glyphs
# silently dropped); \rmdefault alone leaves the OT1->LM mapping in place.
# The bundle has no ec*.pfb, so T1 is not an option — meaning non-ASCII inside
# \text{} may still fail, which degrades to the existing raster fallback.
_FONT_PREAMBLE = "\\usepackage[OT1]{fontenc}\\renewcommand{\\rmdefault}{cmr}"


def _build_latex_wrapper(latex: str, formula_class: FormulaClass) -> str:
    extra: set[str] = set()
    for cmd, pkg in _CMD_TO_PKG.items():
        if cmd in latex and pkg not in _BASE_PACKAGES:
            extra.add(pkg)
    pkg_line = _BASE_PACKAGES + ("," + ",".join(sorted(extra)) if extra else "")
    body = f"\\[\n{latex}\n\\]" if formula_class == FormulaClass.DISPLAY else f"${latex}$"
    return (
        f"\\documentclass[{LATEX_BODY_PT:g}pt]{{article}}\n"
        f"\\usepackage{{{pkg_line}}}\n"
        f"{_FONT_PREAMBLE}\n"
        f"\\pagestyle{{empty}}\n"
        f"\\begin{{document}}\n"
        f"{body}\n"
        f"\\end{{document}}\n"
    )


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

# Set once by run() before the thread pool starts, read by _compile_to_xdv.
# A module-level handle keeps the cache out of the signature of the three
# functions between run() and the compile, none of which otherwise care.
_compile_cache: PersistentCompileCache = PersistentCompileCache(None)


def _compile_to_xdv(
    latex: str,
    formula_class: FormulaClass,
    timeout: int = 45,
) -> tuple[bytes | None, str]:
    """
    Compiles LaTeX to XDV via tectonic — the compile check for Track B.
    Returns (xdv_bytes, stderr_description); stderr is empty on success.

    Track B only needs to know whether the LaTeX compiles, so this replaces
    the PDF build plus pymupdf rasterize that `_compile_to_png` does: the PNG
    it produced was discarded. XDV is also exactly what s08a needs to make the
    SVG, so keeping the bytes lets that stage skip recompiling the identical
    .tex — the two stages share `_build_latex_wrapper`, so it really is the
    same input.
    """
    cached = _compile_cache.get(latex, formula_class.value)
    if cached is not None:
        return cached, ""

    with tempfile.TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        tex_file = work / "equation.tex"
        tex_file.write_text(_build_latex_wrapper(latex, formula_class), encoding="utf-8")

        try:
            import shutil as _shutil
            _tectonic = (
                _shutil.which("tectonic")
                or next((p for p in ("/opt/homebrew/bin/tectonic", "/usr/local/bin/tectonic") if Path(p).exists()), "tectonic")
            )
            proc = subprocess.run(
                [_tectonic, "--outfmt=xdv", str(tex_file)],
                capture_output=True,
                timeout=timeout,
                cwd=work,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace").strip()
                return None, stderr or "tectonic exited non-zero"

            xdv_path = work / "equation.xdv"
            if not xdv_path.exists():
                return None, "tectonic produced no XDV"
            xdv = xdv_path.read_bytes()
            _compile_cache.put(latex, formula_class.value, xdv)
            return xdv, ""

        except subprocess.TimeoutExpired:
            return None, "tectonic timeout"
        except FileNotFoundError:
            return None, "tectonic not found — install via: brew install tectonic"
        except Exception as exc:
            return None, str(exc)


def _compile_to_png(
    latex: str,
    formula_class: FormulaClass,
    timeout: int = 45,
) -> tuple[bytes | None, str]:
    """
    Compiles LaTeX to PNG via tectonic for CDM scoring.
    Returns (png_bytes, stderr_description).
    stderr_description is empty on success, populated on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        tex_file = work / "equation.tex"
        tex_file.write_text(_build_latex_wrapper(latex, formula_class), encoding="utf-8")

        try:
            import shutil as _shutil
            _tectonic = (
                _shutil.which("tectonic")
                or next((p for p in ("/opt/homebrew/bin/tectonic", "/usr/local/bin/tectonic") if Path(p).exists()), "tectonic")
            )
            proc = subprocess.run(
                [_tectonic, "--outfmt=pdf", str(tex_file)],
                capture_output=True,
                timeout=timeout,
                cwd=work,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace").strip()
                return None, stderr or "tectonic exited non-zero"

            pdf_path = work / "equation.pdf"
            if not pdf_path.exists():
                return None, "tectonic produced no PDF"

            import pymupdf  # type: ignore
            doc = pymupdf.open(str(pdf_path))
            if not doc:
                return None, "pymupdf could not open PDF"
            page = doc[0]
            mat = pymupdf.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csGRAY)
            return pix.tobytes("png"), ""

        except subprocess.TimeoutExpired:
            return None, "tectonic timeout"
        except FileNotFoundError:
            return None, "tectonic not found — install via: brew install tectonic"
        except Exception as exc:
            return None, str(exc)


def warm_up_tectonic() -> None:
    """
    Compiles representative inline and display equations once before
    entering a parallel section, so tectonic's font-cache bundle
    (~/Library/Caches/Tectonic/...) is fully populated — including the
    extra fonts pulled in by \\frac/\\sum/\\int — before multiple worker
    threads compile concurrently for the first time. Errors are ignored —
    this is best-effort warm-up, not a correctness requirement.
    """
    try:
        _compile_to_png("x", FormulaClass.INLINE)
        _compile_to_png(r"\sum_{i=1}^{n} \frac{x^2}{i}", FormulaClass.DISPLAY)
    except Exception:
        pass


def _classify_tectonic_error(stderr: str, latex: str) -> tuple[str, str]:
    """Maps tectonic stderr to (ErrorCode sub_code, human-readable detail)."""
    if not stderr:
        return ErrorCode.CDM_COMPILE_ERROR.value, "Compilation failed (no stderr)"
    if "Undefined control sequence" in stderr:
        m = re.search(r"\\([a-zA-Z]+)", stderr.split("Undefined control sequence")[-1][:60])
        cmd = f"\\{m.group(1)}" if m else "unknown"
        return ErrorCode.CDM_UNDEFINED_CMD.value, f"Undefined command: {cmd}"
    if "Missing $" in stderr or "missing $" in stderr.lower():
        return ErrorCode.CDM_BRACE_MISMATCH.value, "Missing $ delimiter"
    if "Extra }" in stderr or "Missing {" in stderr or "Extra {" in stderr:
        return ErrorCode.CDM_BRACE_MISMATCH.value, f"Unmatched braces: {stderr[:80]}"
    return ErrorCode.CDM_COMPILE_ERROR.value, stderr[:120]


# ---------------------------------------------------------------------------
# Plausibility scoring
# ---------------------------------------------------------------------------

_KNOWN_LATEX_COMMANDS: frozenset[str] = frozenset({
    r"\frac", r"\int", r"\sum", r"\prod", r"\sqrt", r"\left", r"\right",
    r"\text", r"\begin", r"\end", r"\cdot", r"\times", r"\div", r"\pm", r"\mp",
    r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv", r"\propto",
    r"\infty", r"\partial", r"\nabla",
    r"\vec", r"\hat", r"\bar", r"\dot", r"\ddot", r"\tilde",
    r"\overline", r"\underline", r"\overbrace", r"\underbrace",
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon",
    r"\zeta", r"\eta", r"\theta", r"\vartheta", r"\iota", r"\kappa",
    r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\varpi", r"\rho",
    r"\sigma", r"\varsigma", r"\tau", r"\upsilon", r"\phi", r"\varphi",
    r"\chi", r"\psi", r"\omega",
    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi", r"\Pi",
    r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi", r"\Omega",
    r"\hbar", r"\ell", r"\Re", r"\Im", r"\aleph",
    r"\mathbb", r"\mathcal", r"\mathbf", r"\mathrm", r"\mathit",
    r"\boldsymbol", r"\operatorname",
    r"\langle", r"\rangle", r"\lvert", r"\rvert",
    r"\bra", r"\ket", r"\braket",
    r"\dagger", r"\ddagger", r"\otimes", r"\oplus",
    r"\cap", r"\cup", r"\subset", r"\supset", r"\subseteq", r"\supseteq",
    r"\in", r"\notin", r"\forall", r"\exists", r"\neg", r"\wedge", r"\vee",
    r"\to", r"\rightarrow", r"\leftarrow", r"\Rightarrow", r"\Leftarrow",
    r"\leftrightarrow", r"\Leftrightarrow", r"\mapsto",
    r"\lim", r"\max", r"\min", r"\sup", r"\inf", r"\exp", r"\ln", r"\log",
    r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
    r"\arcsin", r"\arccos", r"\arctan", r"\sinh", r"\cosh", r"\tanh",
    r"\det", r"\dim", r"\ker",
    r"\quad", r"\qquad",
    r"\label", r"\ref", r"\tag",
    r"\pmatrix", r"\bmatrix", r"\vmatrix", r"\matrix",
    r"\underset", r"\overset", r"\stackrel",
})

_PHYSICS_SIGNALS: frozenset[str] = frozenset({
    r"\frac", r"\int", r"\sum", r"\prod", r"\partial", r"\nabla",
    r"\vec", r"\hat", r"\bar", r"\hbar", r"\infty",
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\lambda",
    r"\mu", r"\nu", r"\pi", r"\sigma", r"\omega", r"\Omega",
    r"\phi", r"\psi", r"\theta", r"\Sigma", r"\Delta",
})


def _plausibility_score(latex: str) -> float:
    """
    Heuristic plausibility score for a LaTeX string, 0–1.
    Does not require compilation. Used for Track B (scanned sources).
    """
    stripped = latex.strip()
    if not stripped:
        return 0.0
    if len(stripped) < 2:
        return 0.05

    score = 1.0

    # Length penalties
    if len(stripped) > 600:
        score -= 0.15
    if len(stripped) < 4:
        score -= 0.3

    # Structural: unmatched braces
    open_b = stripped.count("{")
    close_b = stripped.count("}")
    if abs(open_b - close_b) > 2:
        score -= 0.3
    elif abs(open_b - close_b) > 0:
        score -= 0.1

    # Unmatched dollar signs (only relevant if present at all)
    if "$" in stripped and stripped.count("$") % 2 != 0:
        score -= 0.2

    # Remaining Unicode math chars after substitution (shouldn't be any)
    unicode_math = sum(1 for c in stripped if ord(c) > 127)
    if unicode_math:
        score -= 0.08 * min(unicode_math, 4)

    # Unknown \commands penalise; known ones are neutral or positive
    all_cmds = re.findall(r"\\[a-zA-Z]+", stripped)
    if all_cmds:
        unknown = [c for c in all_cmds if c not in _KNOWN_LATEX_COMMANDS]
        unknown_ratio = len(unknown) / len(all_cmds)
        if unknown_ratio > 0.5:
            score -= 0.25
        elif unknown_ratio > 0.25:
            score -= 0.1

    # Positive signal: recognizable physics/math commands
    physics_hits = sum(1 for cmd in _PHYSICS_SIGNALS if cmd in stripped)
    score = min(1.0, score + 0.04 * physics_hits)

    return max(0.0, min(1.0, score))


def _validate_scanned(
    latex: str,
    region: EquationRegion,
) -> tuple[ConfidenceGate, FailureReason | None, bytes | None]:
    """
    Track B validation: compile check + plausibility score.
    Sets region.cdm_score to the plausibility value (used by report).

    Returns the compiled XDV alongside the verdict rather than writing it to
    the region: the repair loop calls this with candidate LaTeX that may be
    rejected, and only the caller knows which candidate won.
    """
    # Guard: skip if crop is implausibly small (detection artifact)
    if region.source_image_crop:
        try:
            from PIL import Image as _PIL
            img = _PIL.open(io.BytesIO(region.source_image_crop))
            w, h = img.size
            if w < 20 or h < 10:
                region.cdm_score = 0.0
                return ConfidenceGate.FALLBACK, FailureReason(
                    code=ErrorCode.CDM_RENDER_FAILED.value,
                    sub_code=ErrorCode.CROP_TOO_SMALL.value,
                    stage="s06_validation",
                    detail=f"Crop too small ({w}×{h} px) — likely a detection artifact",
                    recoverable=False,
                ), None
        except Exception:
            pass

    # Compile check (binary pass/fail). The XDV comes back to the caller for
    # s08a to reuse — see _compile_to_xdv.
    xdv, stderr = _compile_to_xdv(latex, region.formula_class)
    if stderr:
        sub_code, detail = _classify_tectonic_error(stderr, latex)
        region.cdm_score = 0.0
        return ConfidenceGate.FALLBACK, FailureReason(
            code=ErrorCode.LATEX_COMPILE_FAILED.value,
            sub_code=sub_code,
            stage="s06_validation",
            detail=detail,
            recoverable=sub_code in (ErrorCode.CDM_UNDEFINED_CMD.value, ErrorCode.CDM_BRACE_MISMATCH.value),
        ), None

    # Plausibility scoring
    score = _plausibility_score(latex)
    region.cdm_score = score

    if score >= 0.65:
        return ConfidenceGate.PASS, None, xdv
    elif score >= 0.40:
        return ConfidenceGate.REPAIR, FailureReason(
            code=ErrorCode.CDM_RENDER_FAILED.value,
            sub_code=ErrorCode.CDM_PLAUSIBILITY.value,
            stage="s06_validation",
            detail=f"Plausibility {score:.2f} — compiled OK but recognition uncertain",
            recoverable=True,
        ), xdv
    else:
        return ConfidenceGate.FALLBACK, FailureReason(
            code=ErrorCode.CDM_RENDER_FAILED.value,
            sub_code=ErrorCode.CDM_PLAUSIBILITY.value,
            stage="s06_validation",
            detail=f"Plausibility {score:.2f} — recognition likely failed",
            recoverable=False,
        ), xdv


# ---------------------------------------------------------------------------
# Normalisation (cosmetic only — must not change semantic content)
# ---------------------------------------------------------------------------

def normalize_latex(latex: str) -> str:
    # A negative lookbehind keeps escaped \% (literal percent) intact —
    # without it "50\%" normalizes to "50\" and fails to compile.
    latex = re.sub(r"(?<!\\)%[^\n]*", "", latex)
    latex = _fix_row_break_bracket(latex)
    latex = re.sub(r"\s+", " ", latex).strip()
    latex = re.sub(r"\s*\^\s*", "^", latex)
    latex = re.sub(r"\s*_\s*", "_", latex)
    return latex


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

_GATE_RANK: dict[ConfidenceGate, int] = {
    ConfidenceGate.PASS: 0,
    ConfidenceGate.REPAIR: 1,
    ConfidenceGate.FALLBACK: 2,
    ConfidenceGate.FLAGGED: 3,
}


def _validate_region(
    region: EquationRegion,
    max_repair_attempts: int,
    stage: str,
) -> dict:
    """
    Validates/repairs a single equation region, mutating it in place.

    Returns a result record for sequential aggregation by the caller —
    `counter` (passed/repaired/fallback), an optional `warning` string, and
    an optional `event` (event_type, payload) for `bus.emit`. Keeping result
    aggregation out of this function makes it safe to call from a thread pool
    (each region only reads/writes its own attributes).
    """
    # ── No LaTeX at all (recognition failed upstream) ───────────────
    if not region.raw_latex:
        region.confidence_gate = ConfidenceGate.FALLBACK
        if region.failure_reason is None:
            region.failure_reason = FailureReason(
                code=ErrorCode.LATEX_COMPILE_FAILED.value,
                sub_code=ErrorCode.MER_EMPTY_OUTPUT.value,
                stage=stage,
                detail="No LaTeX produced by recognition stage",
                recoverable=False,
            )
        region.error_codes.append(ErrorCode.LATEX_COMPILE_FAILED.value)
        return {"counter": "fallback", "warning": None, "event": None}

    # ── Tier 1: Unicode → LaTeX (always, before any compile attempt) ─
    # The row-break fix runs here too, not just in the repair loop: without it
    # the first compile fails outright and the equation burns repair attempts
    # on a problem that has one deterministic answer.
    #
    # Normalisation is applied up front rather than only at the end. It is
    # whitespace and comment handling that TeX itself discards, so the compile
    # is unaffected — but it means the gate compiles the *same string* s08a
    # later renders, which is what lets s08a reuse the XDV instead of running
    # tectonic a second time. normalize_latex is idempotent, so the final
    # `region.normalized_latex = normalize_latex(latex)` still holds.
    latex = normalize_latex(
        _fix_row_break_bracket(_apply_unicode_substitutions(region.raw_latex))
    )

    # ── Compile + plausibility gate (the only track — see run()) ─────
    gate, fr, xdv = _validate_scanned(latex, region)

    # Repair loop: try structural repairs if not already PASS
    attempts = 0
    while gate != ConfidenceGate.PASS and attempts < max_repair_attempts:
        repaired_latex = _apply_repair_rules(latex)
        if repaired_latex == latex:
            break  # rules produced no change
        new_gate, new_fr, new_xdv = _validate_scanned(repaired_latex, region)
        if _GATE_RANK[new_gate] < _GATE_RANK[gate]:
            latex = repaired_latex
            gate = new_gate
            fr = new_fr
            xdv = new_xdv
            region.repair_attempts += 1
        attempts += 1

    region.normalized_latex = normalize_latex(latex)
    region.confidence_gate = gate
    region.failure_reason = fr
    # Hand the accepted compile to s08a, tagged with the exact string it
    # came from. A rejected repair candidate never lands here, and s08a
    # only reuses the XDV when that string matches what it is rendering.
    region.compiled_xdv = xdv
    region.compiled_xdv_latex = latex if xdv else None

    if gate == ConfidenceGate.PASS:
        return {
            "counter": "passed",
            "warning": None,
            "event": ("equation_ok", {"equation_id": region.region_id, "plausibility": region.cdm_score}),
        }
    elif gate == ConfidenceGate.REPAIR:
        return {
            "counter": "repaired",
            "warning": f"{region.region_id}: plausibility {region.cdm_score:.2f} — review recommended",
            "event": ("equation_warning", {"equation_id": region.region_id, "plausibility": region.cdm_score}),
        }
    else:
        region.error_codes.append(fr.code if fr else ErrorCode.LATEX_REPAIR_EXHAUSTED.value)
        return {
            "counter": "fallback",
            "warning": None,
            "event": ("equation_error", {
                "equation_id": region.region_id,
                "sub_code": fr.sub_code if fr else "unknown",
                "detail": fr.detail if fr else "",
            }),
        }


def run(
    document: Document,
    bus: EventBus,
    cdm_pass_threshold: float = 0.88,
    cdm_repair_threshold: float = 0.70,
    max_repair_attempts: int = 2,
    cdm_method: str = "ssim",
    max_parallel_workers: int | None = None,
    compile_cache_dir: Path | None = None,
) -> tuple[Document, StageResult]:
    global _compile_cache
    _compile_cache = PersistentCompileCache(compile_cache_dir)
    t0 = time.perf_counter()
    stage = "s06_validation"
    warnings: list[str] = []
    errors: list[str] = []
    passed = repaired = fallback_count = 0

    bus.emit(stage, "stage_start")
    log.info("stage_start", equations=document.total_equation_count)

    # All sources now reach s06 via recognition (MinerU for PDFs, MathML
    # conversion for EPUBs), so the compile+plausibility gate is the single
    # quality gate for every source — see the module docstring for why the
    # SSIM-against-source-crop track was removed rather than kept dormant.

    # ── Skip regions handled outside the SVG pipeline (Stage 5B) ────
    eligible = [
        region for region in document.all_equations
        if not (region.render_as_text or region.is_reference_label or region.dedup_canonical_id is not None)
    ]

    if eligible:
        warm_up_tectonic()

    results = parallel_for_each(
        eligible,
        lambda region: _validate_region(region, max_repair_attempts, stage),
        max_workers=max_parallel_workers,
    )

    # ── Sequential retry: fallbacks from the parallel pass get one calm,
    # uncontended re-validation. Transient tectonic timeouts under the thread
    # pool (bundle-cache and CPU contention) otherwise flag equations whose
    # LaTeX is perfectly compilable; a genuine failure just fails again.
    for i, (region, result) in enumerate(zip(eligible, results)):
        if result["counter"] != "fallback":
            continue
        retry = _validate_region(region, max_repair_attempts, stage)
        if retry["counter"] != "fallback":
            region.error_codes = [
                c for c in region.error_codes
                if c not in (ErrorCode.LATEX_COMPILE_FAILED.value,
                             ErrorCode.LATEX_REPAIR_EXHAUSTED.value)
            ]
            bus.emit(stage, "validate_retry_ok", equation_id=region.region_id)
            results[i] = retry

    for result in results:
        if result["counter"] == "passed":
            passed += 1
        elif result["counter"] == "repaired":
            repaired += 1
        else:
            fallback_count += 1
        if result["warning"]:
            warnings.append(result["warning"])
        if result["event"] is not None:
            event_type, payload = result["event"]
            bus.emit(stage, event_type, **payload)

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    bus.emit(stage, "stage_end", passed=passed, repaired=repaired, fallback=fallback_count)
    log.info("stage_end", passed=passed, repaired=repaired, fallback=fallback_count)

    return document, StageResult(
        stage_name=stage,
        ok=True,
        duration_ms=duration_ms,
        warnings=warnings,
        errors=errors,
        metrics={"passed": passed, "repaired": repaired, "fallback": fallback_count},
    )

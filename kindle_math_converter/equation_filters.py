"""
Recognizer-agnostic LaTeX post-processing filters.

Extracted from s05b_formula_recognition (Phase 1) so they survive the MinerU
migration: these operate on recognized LaTeX strings and are independent of
which model produced them.

- simple_text_repr():   simple inline expressions → Unicode/HTML text spans
                        (bypasses tectonic/SVG entirely)
- is_reference_label(): footnote/citation markers and bare equation-number
                        labels mis-detected as formulas
- normalize_for_dedup(): canonical key for D1 dedup by recognized LaTeX
- extract_equation_tag(): split a trailing \\tag{N} into (latex, "(N)")
"""
import re
from typing import Optional

# ── symbol maps ───────────────────────────────────────────────────────────

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


def _resolve_script_content(content: str) -> Optional[str]:
    """
    Resolves the content of a _{...}/^{...} script to displayable text:
    digits pass through, a single symbol resolves via _resolve_symbol
    (unwrapping one \\mathrm{}/\\mathbf{}), simple trailing punctuation like
    ")" is kept (footnote markers such as "a)"). Returns None if too complex.
    """
    content = content.strip()
    if re.fullmatch(r'\d+', content):
        return content
    pm = _OUTER_PUNCT_RE.match(content)
    assert pm is not None
    pre, core, post = pm.group(1), pm.group(2), pm.group(3)
    ch = _resolve_symbol(core)
    if ch is None:
        m2 = re.fullmatch(r'\\(mathrm|mathbf)\{(.+)\}', core)
        ch = _resolve_symbol(m2.group(2)) if m2 else None
    if ch is None:
        return None
    return f"{pre}{ch}{post}"


def simple_text_repr(latex: str) -> Optional[str]:
    """
    If `latex` is simple enough to render as plain HTML text — one base
    symbol (a Latin letter, or one wrapped in \\mathbf{}/\\mathrm{}/\\mathbb{}),
    optionally one _{...}/^{...}, optionally simple trailing punctuation or
    "=N" — returns the HTML snippet. Also handles a bare leading script with
    no base symbol (e.g. "^{8}" footnote markers, which MinerU emits as
    standalone inline-equation spans). Returns None for anything else, which
    keeps using the SVG pipeline.
    """
    s = latex.strip()
    if not s:
        return None

    # Bare footnote/subscript marker: "^{8}", "_{2}", "^{a)}"
    if s[0] in ("_", "^"):
        marker = s[0]
        token = _take_group_or_token(s[1:])
        if token is None:
            return None
        content, remainder = token
        script = _resolve_script_content(content)
        if script is None or remainder.strip():
            return None
        tag = "sub" if marker == "_" else "sup"
        return f"<{tag}>{script}</{tag}>"

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
        sub_content = _resolve_script_content(content)
        if sub_content is None:
            return None
        tag = "sub" if marker == "_" else "sup"
        out = f"{out}<{tag}>{sub_content}</{tag}>"

    suffix = remainder.strip()
    if suffix:
        if not _SIMPLE_SUFFIX_RE.match(suffix):
            return None
        out += suffix.replace(" ", "")

    return out


# ── reference/footnote-marker filter ──────────────────────────────────────

# `\b` does not work as a terminator here: `\psi_{2}` has no word boundary
# between "psi" and "_" (both are \w), so `\\psi\b` fails to match and the
# whole `\psi_{2}` falls through to _LATEX_COMMAND_RE, which strips `\psi`
# as a bare command and loses the symbol entirely. A negative lookahead for
# another ASCII letter has the same "don't match a longer macro name" effect
# as `\b` without that failure mode.
_GREEK_STRIP_RES = [(re.compile(rf'\\{name}(?![A-Za-z])'), ch) for name, ch in _GREEK_MAP.items()]
_LATEX_COMMAND_RE = re.compile(r'\\[a-zA-Z]+|\\[!,;:]')
_SUBSUP_MARKER_RE = re.compile(r'[\^_]')
_BRACE_RE = re.compile(r'[{}]')

# Equation-number labels require parens — a bare digit string (e.g. a
# stripped "\psi_{2}" -> "2") is not a label.
_EQUATION_NUMBER_LABEL_RE = re.compile(r'^\(\s*\d+(?:\.\d+)*\s*\)$')
# Letter(s)-then-digits (e.g. "S20") is deliberately NOT matched here: it also
# matches common physics variable names like "B1", "B2", "S2" (subscripted
# symbols such as B_1, |B_2\rangle, \mathfrak{S}_2), which would be silently
# dropped from the EPUB if flagged as reference labels. Digit-led markers
# (footnote numbers like "34A.", "60See.") are unambiguous and kept.
_FOOTNOTE_MARKER_RE = re.compile(r'^\d{1,3}[A-Za-zΑ-ω]{0,4}[.,]?$')


def strip_latex_formatting(latex: str) -> str:
    """
    Reduces `latex` to its literal "content" characters: resolves Greek
    macros to their Unicode letters, then drops all remaining LaTeX commands,
    braces and sub/superscript markers. Used by `is_reference_label` to spot
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


def is_reference_label(latex: str) -> bool:
    """
    Returns True if `latex` looks like a footnote/citation marker (e.g.
    "34A.", "60See.") or an equation-number label (e.g. "(9)") rather than a
    real formula.
    """
    core = strip_latex_formatting(latex)
    if not core:
        return False
    return bool(_EQUATION_NUMBER_LABEL_RE.match(core) or _FOOTNOTE_MARKER_RE.match(core))


def normalize_for_dedup(latex: str) -> str:
    return re.sub(r'\s+', ' ', latex.strip())


def eq_placeholder(region_id: str) -> str:
    """Placeholder embedded in TextBlock.raw_text where an equation sits;
    substituted by s10. Uses only characters that survive XML escaping.
    Shared by s03_mineru_parse.py and s02c_epub.py — both source paths must
    emit the exact same format for s10's substitution regex to find it."""
    return f"[[EQ:{region_id}]]"


# ── equation-number tags ──────────────────────────────────────────────────

_TAG_RE = re.compile(r'\\tag\s*\{([^{}]*)\}')


def extract_equation_tag(latex: str) -> tuple[str, Optional[str]]:
    """
    Splits every \\tag{N} out of `latex` (MinerU emits display equations with
    their printed number as "\\tag {N}"). Returns (latex_without_tags,
    "(N)" or None). The number is re-attached typographically by s10 via
    the .eq-number span, so no tag may reach tectonic.

    ALL tags must go, not just the first: a multi-row array can carry one per
    row, and a surviving \\tag does not error inside \\[...\\] — it silently
    typesets the number at the *page* right margin, which balloons the
    tight-bbox SVG (measured: a 28.7pt equation became a 207.7pt box) and
    duplicates the number that s10 already renders.

    Multiple numbers are joined ("(5), (6)") since one span carries them all.
    """
    numbers = [
        n if n.startswith("(") else f"({n})"
        for n in (m.group(1).strip() for m in _TAG_RE.finditer(latex))
        if n
    ]
    stripped = _TAG_RE.sub("", latex).strip()
    return stripped, ", ".join(numbers) if numbers else None

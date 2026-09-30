# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""
Recovering tables that arrive as something other than real table markup, and
the maths inside them.

Two problems, one module because the second is the fix for the first:

1. A table that arrives as a **Markdown pipe table in a paragraph**. MinerU's
   VLM describes a table-shaped figure by writing GFM ("| t | w |\\n| --- |
   --- |"), and that lands in a <p>, so the reader sees pipes and dashes
   instead of a table. Peliti shipped 27 of these, Hartmann 76.

2. A table whose cells carry **bare, undelimited LaTeX** — see below.

Both are source-agnostic: `pipe_table_to_html` and `substitute_table_math` are
called from the MinerU path (s03) and the EPUB/HTML path (s02c) alike, so a
pipe table pasted into an EPUB is recovered the same way as one MinerU invents.

──────────────────────────────────────────────────────────────────────────────

Finding bare LaTeX inside MinerU table cells.

MinerU emits a table's HTML with the maths left as raw LaTeX in the cells and
**no delimiters** — `\\asymp`, `x_{0}^{-1} \\mathrm{e}^{-x/x_{0}}`. The EPUB/HTML
path already handles table maths (`s02c_epub._substitute_table_math`), but it
keys on MathJax delimiters (`$...$`, `\\(...\\)`), so neither condition is met for
a PDF and the LaTeX is serialized verbatim into the book. Peliti shipped with
141 such runs across 16 files, the Notation table reading as `\\langle \\ldots
\\rangle` on the device.

Two cases, because they carry different risk:

  * a cell with no prose is ONE equation — 126 of Peliti's 141 cells
  * a cell mixing prose and maths ("Affinity of cycle \\alpha.") keeps its prose
    and substitutes only the maths, so a description column is never italicised

Runs are found by anchoring on a LaTeX control sequence (or a bare sub/
superscript) and growing outward over whatever binds to it — identifiers,
scripts, balanced groups — because that is how the maths nests. An earlier
left-to-right tokenizer cut identifiers off their own subscripts, turning
`A_\\alpha` into `A` + `_\\alpha`.
"""
import re
from typing import Optional

_CMD = re.compile(r"\\[a-zA-Z]+\*?|\\[^a-zA-Z]")
# An identifier character. Unicode-aware on purpose: OCR'd maths routinely
# arrives with real Greek letters and combining accents rather than control
# sequences ("σ²_p", "χ̂(n)"), and an ASCII-only class cut the base symbol off
# its own script, emitting the fragment "^{2}_p" as an equation of its own.
# `_` is excluded because it is the subscript operator, not part of a name.
_IDENT_CHAR = re.compile(r"[^\W_]|['\u0300-\u036f\u00b2\u00b3\u00b9]", re.UNICODE)
# A prose word: 3+ letters. Two anywhere in the cell make it prose — they need
# not be adjacent, because short connectives ("of", "in") break up a run:
# "Affinity of cycle \alpha." is prose plus maths, not one big equation.
# One such word alone is kept as maths, so "(expression)_{xx'}" survives whole.
_PROSE_WORD = re.compile(r"\b[A-Za-z][A-Za-z'-]{2,}\b")
# Anything that can sit BETWEEN two maths runs without breaking the expression.
# A 3+ letter word ("and", "given") is real prose and keeps the runs apart.
_CONNECTIVE = re.compile(r"^[\s=+\-*/^_()\[\].,:;|<>0-9A-Za-z]*$")
_HAS_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
# Cheap pre-filter: no backslash command and no script means no maths.
_ANY_MATH = re.compile(r"\\[a-zA-Z]+|[_^]")
_TEXT_ARG = re.compile(r"\\(?:text|mathrm|mathbf|mbox|textrm)\s*\{[^{}]*\}")


def _close_group(s: str, i: int, open_c: str = "{", close_c: str = "}") -> int:
    """Index just past the balanced group starting at s[i]."""
    depth = 0
    while i < len(s):
        if s[i] == open_c:
            depth += 1
        elif s[i] == close_c:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(s)


def _close_group_back(s: str, i: int) -> int:
    """Index of the '{' matching the '}' that ends at i."""
    depth = 0
    j = i - 1
    while j >= 0:
        if s[j] == "}":
            depth += 1
        elif s[j] == "{":
            depth -= 1
            if depth == 0:
                return j
        j -= 1
    return 0


def _grow_right(s: str, j: int) -> int:
    """Extend past everything binding to the run that ends at j."""
    while j < len(s):
        if s[j] == "{":
            j = _close_group(s, j)
        elif s[j] in "_^":
            j += 1
            if j < len(s) and s[j] == "{":
                j = _close_group(s, j)
            elif j < len(s) and s[j] == "\\":
                m = _CMD.match(s, j)
                j = m.end() if m else j + 1
            elif j < len(s):
                j += 1
        elif s[j] == "\\":
            m = _CMD.match(s, j)
            if not m:
                break
            j = m.end()
        elif s[j] == "(":
            j = _close_group(s, j, "(", ")")
        elif _IDENT_CHAR.match(s[j]) and j > 0 and s[j - 1] in "_^{}\\)":
            j += 1
            # a combining accent belongs to the character it sits on
            while j < len(s) and "̀" <= s[j] <= "ͯ":
                j += 1
        else:
            break
    return j


def _grow_left(s: str, i: int) -> int:
    """Extend back over the identifier a script belongs to: the `\\alpha` in
    `A_\\alpha` is part of `A`, not a run of its own."""
    while i > 0:
        if s[i - 1] in "_^":
            i -= 1
            while i > 0 and (_IDENT_CHAR.match(s[i - 1]) or s[i - 1] == "}"):
                i = _close_group_back(s, i) if s[i - 1] == "}" else i - 1
        else:
            break
    return i


def _math_runs(cell: str) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []

    def push(a: int, b: int) -> None:
        if runs and a <= runs[-1][1]:
            runs[-1] = (runs[-1][0], max(runs[-1][1], b))
        else:
            runs.append((a, b))

    i = 0
    while i < len(cell):
        if cell[i] == "\\":
            m = _CMD.match(cell, i)
            if m:
                b = _grow_right(cell, m.end())
                push(_grow_left(cell, i), b)
                i = b
                continue
        if cell[i] in "_^" and i + 1 < len(cell):
            b = _grow_right(cell, i)
            push(_grow_left(cell, i + 1), b)
            i = b
            continue
        i += 1
    return runs


def _merge_connected(cell: str, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Join runs separated only by operators/digits, so `\\eta_C = 1` stays one
    equation rather than rendered maths followed by literal "= 1"."""
    if not runs:
        return runs
    merged = [runs[0]]
    for a, b in runs[1:]:
        gap = cell[merged[-1][1]:a]
        if _CONNECTIVE.match(gap) and not _HAS_WORD.search(gap):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def is_pure_math(cell: str) -> bool:
    """True when the cell has no prose outside `\\text{}`-style arguments."""
    stripped = _TEXT_ARG.sub(" ", cell)
    stripped = re.sub(r"\\[a-zA-Z]+\*?", " ", stripped)
    return len(_PROSE_WORD.findall(stripped)) < 2


def segment_cell(cell: str) -> list[tuple[str, str]]:
    """
    Split a table cell into `("math"|"text", chunk)` pairs.

    Concatenating the chunks always reproduces `cell` exactly — the split is
    lossless, so nothing can be dropped by a bad boundary.
    """
    if not _ANY_MATH.search(cell):
        return [("text", cell)]
    if is_pure_math(cell):
        return [("math", cell)]
    out: list[tuple[str, str]] = []
    prev = 0
    for a, b in _merge_connected(cell, _math_runs(cell)):
        if a > prev:
            out.append(("text", cell[prev:a]))
        out.append(("math", cell[a:b]))
        prev = b
    if prev < len(cell):
        out.append(("text", cell[prev:]))
    return out


# Cell text lives between tags; capture the tag boundaries so only the text is
# rewritten and the table's own markup survives verbatim.
_CELL_TEXT = re.compile(r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]>)", re.S | re.I)

# Delimited maths, in case a cell carries it: MinerU's VLM writes \(...\) when
# it transcribes a table as Markdown, and pasted EPUB/HTML content can carry
# any of these. Longest delimiters first so $$ wins over $.
_DELIMITED = re.compile(
    r"\\\[(?P<d1>.+?)\\\]|\\\((?P<i1>.+?)\\\)|\$\$(?P<d2>.+?)\$\$|\$(?P<i2>.+?)\$",
    re.S,
)
# An already-registered placeholder. Must be stepped over rather than re-parsed:
# "[[EQ:eq_1]]" contains an underscore, which the bare-LaTeX scanner would
# otherwise treat as a subscript.
_PLACEHOLDER = re.compile(r"\[\[EQ:[A-Za-z0-9_]+\]\]")


def substitute_table_math(table_html: str, make_equation) -> str:
    """
    Replace the maths in every cell of `table_html` with equation placeholders.

    Handles both delimited maths (``\\(...\\)``, ``$...$``) and bare LaTeX, so a
    cell is covered whether it came from MinerU, from a Markdown pipe table, or
    from pasted EPUB/HTML.

    `make_equation(latex) -> placeholder` is called once per maths run and is
    responsible for registering the region, returning the placeholder to splice
    in. Cells containing no maths are left byte-identical.
    """
    def rewrite_plain(text: str) -> str:
        """Bare-LaTeX pass over a stretch known to hold no placeholders."""
        if not _ANY_MATH.search(text):
            return text
        return "".join(
            make_equation(chunk) if kind == "math" else chunk
            for kind, chunk in segment_cell(text)
        )

    def cell(m: "re.Match[str]") -> str:
        open_tag, body, close_tag = m.group(1), m.group(2), m.group(3)
        # a cell carrying its own markup is left alone: rewriting it risks
        # corrupting the table for no gain
        if "<" in body:
            return m.group(0)

        # delimited maths first, so "\(x\)" is not mistaken for the control
        # sequence "\(" by the bare-LaTeX scanner
        body = _DELIMITED.sub(
            lambda d: make_equation(next(g for g in d.groups() if g is not None).strip()),
            body,
        )
        # then bare LaTeX, stepping over any placeholder already present
        out, prev = [], 0
        for p in _PLACEHOLDER.finditer(body):
            out.append(rewrite_plain(body[prev:p.start()]))
            out.append(p.group(0))
            prev = p.end()
        out.append(rewrite_plain(body[prev:]))
        return f"{open_tag}{''.join(out)}{close_tag}"

    return _CELL_TEXT.sub(cell, table_html)


# ── Markdown pipe tables ──────────────────────────────────────────────────

# A row: starts and ends with '|'. The separator row is all dashes and colons.
_PIPE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_PIPE_SEP = re.compile(r"^[\s|:-]+$")


def _split_row(line: str) -> list[str]:
    inner = _PIPE_ROW.match(line)
    return [c.strip() for c in inner.group(1).split("|")] if inner else []


def looks_like_pipe_table(text: str) -> bool:
    """A Markdown table needs a header row, a `---` separator, and a body row."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    if not all(_PIPE_ROW.match(ln) for ln in lines):
        return False
    return bool(_PIPE_SEP.match(lines[1])) and "-" in lines[1]


def pipe_table_to_html(text: str) -> Optional[str]:
    """
    Convert a Markdown pipe table to table markup, or None if `text` is not one.

    Emits the same bare `<table>` shape MinerU produces, so it flows through
    `s10_epub_assembly._valid_table_html` and the placeholder substitution
    unchanged. Ragged rows are padded rather than rejected — the VLM's
    transcription is not always square, and dropping the table would be worse
    than an empty trailing cell.
    """
    if not looks_like_pipe_table(text):
        return None
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    header = _split_row(lines[0])
    body = [_split_row(ln) for ln in lines[2:]]
    width = max([len(header)] + [len(r) for r in body])

    def row(cells: list[str], tag: str) -> str:
        cells = cells + [""] * (width - len(cells))
        return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

    return "<table>" + row(header, "th") + "".join(row(r, "td") for r in body) + "</table>"

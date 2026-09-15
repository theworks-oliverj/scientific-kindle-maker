"""
Tests for bare-LaTeX detection in MinerU table cells (table_math.py).

MinerU emits table HTML with undelimited LaTeX in the cells. The EPUB path's
_substitute_table_math keys on MathJax delimiters, which MinerU never produces,
so PDFs shipped with raw LaTeX visible on the device — Peliti had 141 such runs
across 16 files, its Notation table reading as "\\langle \\ldots \\rangle".

The sample cells below are real, taken from the shipped
`07_peliti_FIXED.epub`, so a future tweak to the regexes cannot silently
re-break them — in particular by re-italicising the description column.
"""
from kindle_math_converter.table_math import (
    is_pure_math,
    segment_cell,
    substitute_table_math,
)

# Verbatim from Peliti's Notation and constants tables.
PURE_MATH_CELLS = [
    r"\asymp",
    r"\langle \ldots \rangle",
    r"\left.\frac{\partial X}{\partial Y}\right)_Z",
    r"A_\alpha",
    r"(\mathcal{C}_\alpha)",
    r"D_{\text{KL}}(p\|q)",
    r"F^{\text{neq}}",
    r"x_{0}^{-1} \mathrm{e}^{-x/x_{0}}",
    r"C_{\alpha\beta}(t)",
]

MIXED_CELLS = [
    r"Affinity of cycle \alpha.",
    r"Joint Shannon entropy of \mathcal{S}_1 and \mathcal{S}_2.",
    r"Conditional Shannon entropy of \mathcal{S}_1 given \mathcal{S}_2.",
    r"Thermal efficiency (\eta^{\text{th}} = -W/E_{\text{hot}}).",
]

PROSE_CELLS = [
    "Leading exponential order.",
    "Average over probability distributions.",
    "Distribution",
    "",
]


def test_segmentation_is_lossless_for_every_sample():
    """A bad boundary must never drop characters — concatenating the chunks
    always reproduces the input."""
    for cell in PURE_MATH_CELLS + MIXED_CELLS + PROSE_CELLS:
        assert "".join(c for _, c in segment_cell(cell)) == cell, cell


def test_pure_maths_cells_become_a_single_equation():
    for cell in PURE_MATH_CELLS:
        assert is_pure_math(cell), cell
        assert segment_cell(cell) == [("math", cell)], cell


def test_prose_cells_are_untouched():
    for cell in PROSE_CELLS:
        assert segment_cell(cell) == [("text", cell)], cell


def test_prose_in_a_mixed_cell_is_not_turned_into_maths():
    """The regression that matters: italicising a whole description column."""
    segs = segment_cell(r"Affinity of cycle \alpha.")
    text = "".join(c for k, c in segs if k == "text")
    assert "Affinity of cycle" in text
    maths = [c for k, c in segs if k == "math"]
    assert maths and all("Affinity" not in m for m in maths)


def test_prose_words_keep_separate_equations_apart():
    # "and" must not be swallowed into the maths on either side
    segs = segment_cell(r"Joint Shannon entropy of \mathcal{S}_1 and \mathcal{S}_2.")
    maths = [c for k, c in segs if k == "math"]
    assert len(maths) == 2, maths
    assert all("and" not in m for m in maths)


def test_operators_between_maths_runs_do_not_split_the_equation():
    # `\eta^{...} = -W/E_{...}` is one expression, not maths + literal "= -W/"
    segs = segment_cell(r"Thermal efficiency (\eta^{\text{th}} = -W/E_{\text{hot}}).")
    maths = [c for k, c in segs if k == "math"]
    assert len(maths) == 1, maths
    assert "=" in maths[0] and "E_{\\text{hot}}" in maths[0]


def test_an_identifier_stays_with_its_own_subscript():
    # the tokenizer this replaced split "A_\alpha" into "A" + "_\alpha"
    assert segment_cell(r"A_\alpha") == [("math", r"A_\alpha")]


def test_substitution_registers_each_run_and_splices_placeholders():
    seen = []

    def make(latex):
        seen.append(latex)
        return f"[[EQ:eq_{len(seen)}]]"

    html = (r"<table><tr><td>\asymp</td><td>Leading exponential order.</td></tr>"
            r"<tr><td>A_\alpha</td><td>Affinity of cycle \alpha.</td></tr></table>")
    out = substitute_table_math(html, make)

    assert seen == [r"\asymp", r"A_\alpha", r"\alpha"]
    assert "[[EQ:eq_1]]" in out and "[[EQ:eq_3]]" in out
    assert "\\asymp" not in out, "raw LaTeX left in the cell"
    # prose survives verbatim
    assert "Leading exponential order." in out
    assert "Affinity of cycle " in out
    # the table's own markup is untouched
    assert out.startswith("<table><tr><td>") and out.endswith("</table>")


def test_a_table_with_no_maths_is_returned_unchanged():
    html = "<table><tr><td>Distribution</td><td>Mean</td></tr></table>"
    assert substitute_table_math(html, lambda x: "SHOULD NOT HAPPEN") == html


def test_cells_containing_nested_markup_are_left_alone():
    """Rewriting a cell that carries its own tags risks corrupting the table;
    those keep the current behaviour rather than being half-processed."""
    html = r"<table><tr><td><b>\alpha</b></td></tr></table>"
    assert substitute_table_math(html, lambda x: "[[EQ:x]]") == html


# ── Markdown pipe tables ──────────────────────────────────────────────────
# MinerU's VLM transcribes a table-shaped figure as GFM, which lands in a <p>.
# Peliti shipped 27 of these, Hartmann 76 — verbatim shapes below.

PIPE_TABLE = (
    "| Time Interval | State Value |\n"
    "| --- | --- |\n"
    "| t0 to t1 | x0 |\n"
    "| t1 to t2 | x1 |"
)


def test_a_markdown_pipe_table_is_recognised():
    from kindle_math_converter.table_math import looks_like_pipe_table
    assert looks_like_pipe_table(PIPE_TABLE)


def test_ordinary_prose_is_not_mistaken_for_a_table():
    from kindle_math_converter.table_math import looks_like_pipe_table
    for text in [
        "This sentence | has a pipe but is not a table.",
        "| only | a | header |",
        "| head |\n| --- |",
        "Normal paragraph text.",
        "",
    ]:
        assert not looks_like_pipe_table(text), text


def test_pipe_table_becomes_table_markup():
    from kindle_math_converter.table_math import pipe_table_to_html
    html = pipe_table_to_html(PIPE_TABLE)
    assert html is not None
    assert html.startswith("<table>") and html.endswith("</table>")
    assert html.count("<tr>") == 3
    assert "<th>Time Interval</th>" in html
    assert "<td>t0 to t1</td>" in html
    assert "---" not in html, "separator row leaked into the output"


def test_ragged_rows_are_padded_rather_than_dropped():
    from kindle_math_converter.table_math import pipe_table_to_html
    html = pipe_table_to_html("| a | b | c |\n| --- | --- | --- |\n| 1 |")
    assert html is not None
    assert html.count("<td>") == 3, "short row should be padded to full width"


def test_non_table_text_converts_to_none():
    from kindle_math_converter.table_math import pipe_table_to_html
    assert pipe_table_to_html("just a paragraph") is None


def test_maths_inside_a_recovered_pipe_table_is_registered():
    from kindle_math_converter.table_math import pipe_table_to_html
    seen = []

    def make(latex):
        seen.append(latex)
        return f"[[EQ:e{len(seen)}]]"

    html = pipe_table_to_html("| t | w (k_B T) |\n| --- | --- |\n| 0 | 25 |")
    assert html is not None
    out = substitute_table_math(html, make)
    assert seen, "bare LaTeX in a recovered table was not registered"
    assert "[[EQ:e1]]" in out


def test_delimited_maths_is_registered_with_delimiters_stripped():
    """MinerU writes \\(...\\) when transcribing Markdown; pasted HTML may use
    $...$. Both must reach the equation pipeline as bare LaTeX."""
    seen = []

    def make(latex):
        seen.append(latex)
        return "[[EQ:d]]"

    out = substitute_table_math(
        r"<table><tr><td>\(\Delta x\)</td><td>$y^2$</td></tr></table>", make)
    assert seen == [r"\Delta x", "y^2"], seen
    assert "\\(" not in out and "$" not in out


def test_an_existing_placeholder_is_not_reparsed():
    """s02c substitutes MathJax before the table is built, so cells can already
    hold placeholders. The underscore in "[[EQ:eq_1]]" must not be read as a
    subscript and re-registered."""
    calls = []
    out = substitute_table_math(
        "<table><tr><td>[[EQ:eq_1]]</td></tr></table>",
        lambda tex: calls.append(tex) or "[[EQ:new]]")
    assert calls == [], f"placeholder was re-parsed as maths: {calls}"
    assert "[[EQ:eq_1]]" in out

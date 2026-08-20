"""
Table-driven tests for the MathML -> LaTeX conversion in s05c_mathml.py.

Each case is a real MathML shape from the tag-frequency audit of a real
~2200-equation publisher EPUB (see the "Lessons carried over" section of
the build plan this module was written against) — mfenced, mtable and
munder/munderover were the actual gaps found, not hypothetical ones.

These call the private conversion helpers directly rather than running the
full Pipeline: fast (milliseconds, no subprocess), and a failure here points
straight at the conversion rule that's wrong.
"""
from kindle_math_converter.stages.s05c_mathml import (
    _mathml_annotation_tex,
    _mathml_text_tokens,
    _mathml_to_latex,
)


def test_basic_superscript():
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup>'
        '</math>'
    )
    assert _mathml_to_latex(mathml) == "E = m c^{2}"


def test_mfenced_default_parens():
    # <mfenced> with no attributes defaults to open="(" close=")" separators=",".
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<mfenced><mi>a</mi><mi>b</mi></mfenced>'
        '</math>'
    )
    latex = _mathml_to_latex(mathml)
    assert r"\left(" in latex and r"\right)" in latex
    assert "a" in latex and "b" in latex


def test_mfenced_custom_brackets():
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<mfenced open="[" close="]"><mi>x</mi></mfenced>'
        '</math>'
    )
    latex = _mathml_to_latex(mathml)
    assert r"\left[" in latex and r"\right]" in latex


def test_mtable_becomes_matrix():
    # A 2x2 system — the real fixture's dominant mtable shape.
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<mtable>'
        '<mtr><mtd><mi>a</mi></mtd><mtd><mi>b</mi></mtd></mtr>'
        '<mtr><mtd><mi>c</mi></mtd><mtd><mi>d</mi></mtd></mtr>'
        '</mtable>'
        '</math>'
    )
    latex = _mathml_to_latex(mathml)
    assert r"\begin{matrix}" in latex and r"\end{matrix}" in latex
    assert "&" in latex  # cell separator within a row
    assert r"\\" in latex  # row separator


def test_munder_is_a_subscript_not_an_accent():
    # The dominant real case: a limit under an operator (\sum, \lim), which
    # is a plain subscript in LaTeX — not the accent-map mover uses.
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<munder><mo>&#8721;</mo><mi>n</mi></munder>'
        '</math>'
    )
    latex = _mathml_to_latex(mathml)
    assert r"\sum_{n}" == latex


def test_munderover_sub_and_superscript():
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<munderover><mo>&#8721;</mo><mi>n</mi><mi>N</mi></munderover>'
        '</math>'
    )
    latex = _mathml_to_latex(mathml)
    assert r"\sum_{n}^{N}" == latex


def test_mstyle_is_a_transparent_wrapper():
    # mstyle should not alter the wrapped content — it was the single most
    # frequent tag (14636x) in the real fixture, almost always used this way.
    plain = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<msub><mi>k</mi><mn>1</mn></msub>'
        '</math>'
    )
    styled = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<msub><mi>k</mi><mstyle><mrow><mn>1</mn></mrow></mstyle></msub>'
        '</math>'
    )
    assert _mathml_to_latex(plain) == _mathml_to_latex(styled)


def test_annotation_tex_extracted_verbatim():
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<semantics>'
        '<mrow><mi>x</mi></mrow>'
        '<annotation encoding="application/x-tex">x^{2} + 1</annotation>'
        '</semantics>'
        '</math>'
    )
    assert _mathml_annotation_tex(mathml) == "x^{2} + 1"


def test_annotation_tex_absent_returns_none():
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<mi>x</mi>'
        '</math>'
    )
    assert _mathml_annotation_tex(mathml) is None


def test_text_tokens_fallback_survives_bad_mathml():
    # The fallback is built from mi/mn/mo/mtext tokens directly, not from
    # derived LaTeX — so it must still work even for content the converter
    # above renders badly (or that later fails to compile).
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup>'
        '</math>'
    )
    tokens = _mathml_text_tokens(mathml)
    assert tokens == "E = m c 2"

"""
Regression tests for <img alt="..."> escaping in s10_epub_assembly.

Amazon's "Enhanced Mobi" converter locates the end of an <img> tag by scanning
for '>' without honouring quoted attribute values. Any '>' inside alt text
therefore truncates the tag and fails the whole book with E21018 — surfaced by
the Send to Kindle web form as a bare E999.

Verified empirically against Kindle Previewer 3.107.0 with single-figure probe
EPUBs: '>', '&gt;' and '&#62;' in alt each fail conversion, while '&lt;',
'&amp;' and '&quot;' convert cleanly. This is why alt text needs an escaper of
its own rather than the general-purpose _escape_text.
"""
from kindle_math_converter.stages.s10_epub_assembly import (
    _ALT_SAFE_GT,
    _escape_alt_text,
    _escape_text,
)


def test_alt_text_contains_no_greater_than_in_any_form():
    out = _escape_alt_text('A["x"] --> B["y"]')
    assert ">" not in out
    assert "&gt;" not in out
    assert "&#62;" not in out


def test_alt_text_still_escapes_the_other_xml_specials():
    out = _escape_alt_text('salt & pepper <tag> "quoted"')
    assert "&amp;" in out
    assert "&lt;" in out
    assert "&quot;" in out


def test_ampersand_is_escaped_before_the_gt_substitution():
    # the substituted character must not arrive as a literal '&...;' sequence
    # that a later escape pass would double-encode
    out = _escape_alt_text("a > b & c")
    assert out == f"a {_ALT_SAFE_GT} b &amp; c"


def test_mermaid_alt_text_from_a_real_failing_book():
    # shape taken from Peliti chapter_017, the block that reproduced E21018
    alt = '```mermaid graph TD A["MICROSCOPIC"] -->|"coarse graining"| B["MESO"] ```'
    out = _escape_alt_text(alt)
    assert ">" not in out and "&gt;" not in out
    assert "mermaid" in out  # content preserved, only the fatal char replaced


def test_plain_alt_text_is_left_readable():
    assert _escape_alt_text("A lab photograph") == "A lab photograph"


def test_general_escape_text_still_emits_gt_for_element_content():
    # element text is not affected by the <img> parser bug; only attributes are
    assert _escape_text("a > b") == "a &gt; b"

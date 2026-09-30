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
import re

from kindle_math_converter.models.document import (
    BoundingBox,
    Document,
    DocumentMetadata,
    FigureBlock,
    Page,
)
from kindle_math_converter.models.enums import ColumnLayout, SourceType
from kindle_math_converter.observability.event_bus import EventBus
from kindle_math_converter.stages.s10_epub_assembly import (
    _ALT_SAFE_GT,
    _document_to_chapters,
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


def test_gt_substitution_does_not_disturb_ampersand_escaping():
    # '>' is replaced first, then _escape_text handles '&'; the substituted
    # character must not arrive as a literal '&...;' that gets double-encoded
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


def _one_figure_document(alt_text: str) -> Document:
    box = BoundingBox(
        x0=0.0, y0=0.0, x1=100.0, y1=100.0,
        coordinate_system="pdf_points", page_number=1,
    )
    figure = FigureBlock(
        figure_id="fig_1_1",
        bbox=box,
        image_bytes=b"\x89PNG\r\n\x1a\n",  # only needs to be truthy to take the image branch
        alt_text=alt_text,
        reading_order_index=0.0,
    )
    page = Page(
        page_number=1,
        width_pt=612.0,
        height_pt=792.0,
        column_layout=ColumnLayout.SINGLE,
        image_bytes=None,
        figures=[figure],
    )
    meta = DocumentMetadata(
        source_path="probe.pdf",
        source_type=SourceType.LATEX_PDF,
        title="Probe",
        author=None,
        page_count=1,
        has_math_fonts=False,
        is_scanned=False,
        column_layout=ColumnLayout.SINGLE,
    )
    return Document(metadata=meta, pages=[page])


def test_assembled_chapter_never_emits_gt_inside_an_img_alt():
    """The real wiring check: _document_to_chapters must route figure alt text
    through _escape_alt_text, not _escape_text. Covers s10_epub_assembly.py's
    figure branch, which the helper-level tests above do not execute."""
    # both real-world shapes: a mermaid diagram, and plain math inequalities
    # (Hartmann chapter_109, the alt that actually tripped E21018 first)
    alt = 'A["x"] --> B["y"] and u*>0 v*>0 T*<0'
    chapters = _document_to_chapters(
        _one_figure_document(alt), title="Probe", bus=EventBus(), embedded_images={},
    )
    body = "\n".join(html for _, html in chapters)

    assert "<img" in body, "figure branch did not run — fixture never reached line 524"
    for match in re.finditer(r'alt="(.*?)"', body, re.S):
        assert ">" not in match.group(1)
        assert "&gt;" not in match.group(1)
    assert _ALT_SAFE_GT in body, "the safe substitute should appear in the output"

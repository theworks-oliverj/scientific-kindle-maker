"""
Unit tests for the s02c_epub.py DOM-order walker — the part that used to be
missing entirely (it only ever extracted equations, never prose; see the
build plan's Context section). These feed small in-memory XHTML fragments
straight to _walk_one_chapter, without needing a full .epub file on disk.
"""
import io

from kindle_math_converter.stages.s02c_epub import _to_png_bytes, _walk_one_chapter


def _walk(xhtml: str):
    page, title, used_html_fallback = _walk_one_chapter(
        xhtml.encode(), chapter_num=1, image_resolver=lambda src: None,
    )
    return page, title, used_html_fallback


def test_prose_paragraph_becomes_a_text_block():
    page, _, fallback = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<p>Hello world.</p>'
        '</body></html>'
    )
    assert not fallback
    assert len(page.text_blocks) == 1
    assert page.text_blocks[0].kind == "text"
    assert page.text_blocks[0].raw_text == "Hello world."


def test_heading_becomes_heading_block_and_sets_title_hint():
    page, title, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<h1>Chapter Title</h1><p>Body text.</p>'
        '</body></html>'
    )
    assert page.text_blocks[0].kind == "heading"
    assert title == "Chapter Title"


def test_list_items_become_separate_list_item_blocks():
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<ul><li>First</li><li>Second</li></ul>'
        '</body></html>'
    )
    kinds = [b.kind for b in page.text_blocks]
    texts = [b.raw_text for b in page.text_blocks]
    assert kinds == ["list_item", "list_item"]
    assert texts == ["First", "Second"]


def test_inline_math_becomes_placeholder_in_surrounding_prose():
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<p>Before '
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math>'
        ' after.</p>'
        '</body></html>'
    )
    assert len(page.equation_regions) == 1
    region = page.equation_regions[0]
    assert page.text_blocks[0].raw_text == f"Before [[EQ:{region.region_id}]] after."
    assert region.raw_latex is not None and region.raw_latex.startswith("MATHML:")


def test_equation_like_img_gets_a_placeholder_and_no_ordinary_figure():
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<p>See <img src="eq1.png" alt="\\alpha + \\beta"/> here.</p>'
        '</body></html>'
    )
    assert len(page.equation_regions) == 1
    assert len(page.figures) == 0
    assert "[[EQ:" in page.text_blocks[0].raw_text


def test_standalone_figure_img_is_not_treated_as_equation():
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<div><img src="photo.jpg" alt="A lab photograph"/></div>'
        '</body></html>'
    )
    assert len(page.equation_regions) == 0
    assert len(page.figures) == 1
    assert page.figures[0].alt_text == "A lab photograph"


def test_table_captured_without_xhtml_namespace():
    # s10's _valid_table_html requires root.tag == "table" (bare, no
    # namespace) — a table parsed straight out of an XHTML document would
    # otherwise carry xmlns="...1999/xhtml" and fail that check silently.
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<table><tr><td>1</td></tr></table>'
        '</body></html>'
    )
    assert len(page.figures) == 1
    table_html = page.figures[0].table_html
    assert table_html is not None
    assert "xmlns" not in table_html
    assert table_html.strip().startswith("<table")


def test_footnote_aside_marked_as_footnote_block():
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><body>'
        '<aside epub:type="footnote" id="fn1"><p>A note.</p></aside>'
        '</body></html>'
    )
    footnote_blocks = [b for b in page.text_blocks if b.kind == "footnote"]
    assert len(footnote_blocks) == 1
    assert footnote_blocks[0].footnote_id == "fn1"
    assert footnote_blocks[0].raw_text == "A note."


def test_noteref_marker_links_to_footnote_via_placeholder():
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><body>'
        '<p>A claim<a epub:type="noteref" href="#fn1">1</a>.</p>'
        '<aside epub:type="footnote" id="fn1"><p>The note body.</p></aside>'
        '</body></html>'
    )
    marker_regions = [r for r in page.equation_regions if r.footnote_ref_id == "fn1"]
    assert len(marker_regions) == 1
    assert marker_regions[0].render_as_text is True
    assert marker_regions[0].inline_text_repr == "1"
    assert f"[[EQ:{marker_regions[0].region_id}]]" in page.text_blocks[0].raw_text


def test_mathjax_display_dollar_delimiters_become_equation_region():
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<p>Before $$x^2 + y^2 = r^2$$ after.</p>'
        '</body></html>'
    )
    assert len(page.equation_regions) == 1
    region = page.equation_regions[0]
    assert region.raw_latex == "x^2 + y^2 = r^2"
    assert region.formula_class.value == "display"
    assert not region.raw_latex.startswith("MATHML:")  # bypasses s05c conversion
    assert page.text_blocks[0].raw_text == f"Before [[EQ:{region.region_id}]] after."


def test_mathjax_inline_dollar_delimiter():
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<p>The rate $\\Omega \\approx 7.27$ is tiny.</p>'
        '</body></html>'
    )
    assert len(page.equation_regions) == 1
    region = page.equation_regions[0]
    assert region.raw_latex == "\\Omega \\approx 7.27"
    assert region.formula_class.value == "inline"


def test_mathjax_equation_has_text_fallback_set():
    # MathJax equations bypass s05c_mathml.py entirely (their raw_latex
    # never gets the "MATHML:" prefix, so s05c's filter skips them) — which
    # means they also skip s05c's text_fallback assignment. A flagged
    # MathJax equation with no text_fallback silently vanishes exactly like
    # the original all-prose-dropped bug this session started by fixing.
    # Caught by testing against a real HTML page: 20/126 equations vanished
    # with zero trace before this field was set directly in s02c_epub.py.
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<p>$$x^2$$</p>'
        '</body></html>'
    )
    region = page.equation_regions[0]
    assert region.text_fallback == "x^2"


def test_mathjax_bracket_paren_delimiters():
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<p>Inline \\(a+b\\) and display \\[c = d\\] done.</p>'
        '</body></html>'
    )
    assert len(page.equation_regions) == 2
    display = [r for r in page.equation_regions if r.formula_class.value == "display"][0]
    inline = [r for r in page.equation_regions if r.formula_class.value == "inline"][0]
    assert display.raw_latex == "c = d"
    assert inline.raw_latex == "a+b"


def test_bare_div_dollar_display_block_not_wrapped_in_p():
    # Real-world pattern: <div class="defn">$$...$$</div> with no <p> at
    # all — found in a real HTML artifact. Without the "container with no
    # extracted children falls back to its own text" rule, this vanishes
    # entirely (the div has no <p>/<h*>/<li> child for _walk_block to find).
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<div class="defn">$$E = mc^2$$</div>'
        '</body></html>'
    )
    assert len(page.equation_regions) == 1
    assert page.equation_regions[0].raw_latex == "E = mc^2"


def test_canvas_panel_swallowed_as_unsupported_placeholder():
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<div class="interactive">'
        '<div class="heading">Lab 1 — Dispersion Surface</div>'
        '<div class="controls"><label>k</label><input type="range"/></div>'
        '<canvas id="b1-canvas" width="780" height="380"></canvas>'
        '</div>'
        '</body></html>'
    )
    assert len(page.figures) == 1
    fig = page.figures[0]
    assert fig.unsupported_label is not None
    assert "Lab 1" in fig.unsupported_label
    # Control labels ("k") must not leak through as disconnected prose.
    assert len(page.text_blocks) == 0


def test_canvas_alongside_real_prose_paragraph_is_not_swallowed():
    # A <section> mixing a real <p> with a sibling interactive panel must
    # keep the prose — only the canvas-only sub-container is swallowed.
    page, _, _ = _walk(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<section>'
        '<p>Real prose that must survive.</p>'
        '<div class="interactive"><canvas id="c1"></canvas></div>'
        '</section>'
        '</body></html>'
    )
    prose = [b for b in page.text_blocks if "Real prose" in b.raw_text]
    assert len(prose) == 1
    assert len(page.figures) == 1
    assert page.figures[0].unsupported_label is not None


def test_jpeg_bytes_get_reencoded_as_png():
    # s10_epub_assembly writes every figure out as "{id}.png" with
    # media-type="image/png" unconditionally (a contract the PDF path
    # always satisfies — its crops are cut from a PNG raster). EPUB source
    # images are whatever the publisher shipped; passing real-book JPEG
    # bytes through under a .png name failed epubcheck (OPF-029) on the
    # real fixture this was caught against, before this conversion existed.
    from PIL import Image

    jpeg_buf = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(jpeg_buf, format="JPEG")

    png_bytes = _to_png_bytes(jpeg_buf.getvalue())

    assert png_bytes is not None
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG file signature
    assert Image.open(io.BytesIO(png_bytes)).format == "PNG"


def test_to_png_bytes_returns_none_for_empty_input():
    assert _to_png_bytes(None) is None
    assert _to_png_bytes(b"") is None


def test_xml_parse_fallback_flag_set_on_malformed_input():
    # lxml.html is namespace-blind — falling back to it silently loses
    # MathML detection, so callers must be told this happened.
    _, _, used_fallback = _walk("<html><body><p>Unclosed tag<p></body></html>")
    assert used_fallback is True

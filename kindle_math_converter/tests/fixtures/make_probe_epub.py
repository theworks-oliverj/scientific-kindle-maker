"""
Builds a small, self-contained EPUB with one paragraph of prose before and
after one MathML equation — used by test_epub_convert.py as a full-pipeline
smoke test.

This is deliberately the same shape as the EPUB used, in an earlier
investigation session, to first prove the EPUB path was dropping all prose:
running it through Pipeline.run() and grepping the output XHTML for the
prose text is exactly the check that would have caught that bug before it
shipped. That is what test_epub_convert.py does with the file this builds.
"""
from pathlib import Path

PROSE_BEFORE = "This is prose before the equation that must survive the pipeline intact."
PROSE_AFTER = "This is prose after the equation that must also survive intact."


def build(path: Path) -> Path:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("probe-epub-001")
    book.set_title("Probe Test Book")
    book.set_language("en")
    book.add_author("Test Fixture")

    chapter = epub.EpubHtml(title="Chapter One", file_name="chap_01.xhtml", lang="en")
    # The MathML namespace is declared on <math> itself (not as a root-level
    # prefix) — matches how real publisher EPUBs do it. A root-level prefix
    # (xmlns:m="...") gets silently stripped by ebooklib's serializer, which
    # is what caused the original investigation's synthetic fixture to look
    # namespace-broken in a way no real EPUB actually is.
    chapter.content = (
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        '<head><title>Chapter One</title></head><body>'
        '<h1>Chapter One</h1>'
        f'<p>{PROSE_BEFORE}</p>'
        '<p>Einstein\'s famous equation is '
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup>'
        '</math>.</p>'
        f'<p>{PROSE_AFTER}</p>'
        '</body></html>'
    )
    book.add_item(chapter)
    book.toc = (epub.Link("chap_01.xhtml", "Chapter One", "chap1"),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]

    epub.write_epub(str(path), book)
    return path

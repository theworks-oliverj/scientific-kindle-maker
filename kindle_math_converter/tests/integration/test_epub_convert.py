"""
Full-pipeline smoke test for the EPUB path.

This is, almost verbatim, the check that was run by hand during the session
that found the EPUB path was silently dropping all prose (see the build
plan's Context section): build a tiny EPUB, run it through Pipeline.run(),
grep the output XHTML for the prose text. If this test had existed before,
it would have failed loudly instead of the pipeline reporting "Complete"
over a book with no content.

Not marked slow: no MinerU, no GPU, no external file — tectonic compiles a
couple of tiny equations, which takes well under a second.
"""
import zipfile

from kindle_math_converter.pipeline import Pipeline, PipelineConfig
from kindle_math_converter.tests.fixtures.make_probe_epub import (
    PROSE_AFTER,
    PROSE_BEFORE,
    build,
)


def _chapter_xhtml_files(epub_path) -> list[str]:
    with zipfile.ZipFile(epub_path) as zf:
        names = [
            n for n in zf.namelist()
            if n.endswith(".xhtml") and "nav" not in n.lower() and "toc" not in n.lower()
        ]
        return [zf.read(n).decode("utf-8") for n in names]


def test_epub_prose_and_equation_survive_the_pipeline(tmp_path):
    epub_path = build(tmp_path / "probe.epub")
    out_dir = tmp_path / "out"

    config = PipelineConfig(output_dir=str(out_dir), epubcheck_enabled=False)
    result = Pipeline(config).run(str(epub_path), str(out_dir))

    assert result.ok, result.fatal_error
    assert result.output_epub_path is not None

    chapters = _chapter_xhtml_files(result.output_epub_path)
    combined = "\n".join(chapters)

    # The actual regression: prose must appear verbatim, not just an empty
    # <p>&#160;</p> body (what the pipeline produced before this was fixed).
    assert PROSE_BEFORE in combined
    assert PROSE_AFTER in combined

    # The equation must render as something — SVG, rendered text, or the
    # MathML-token fallback — never a raw [[EQ:...]] placeholder leaking
    # into the reading flow, and never silently absent.
    assert "[[EQ:" not in combined
    assert ("<svg" in combined) or ('class="eq-text"' in combined)

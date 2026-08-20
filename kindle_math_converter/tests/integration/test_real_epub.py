"""
Full-pipeline test against a real, math-heavy publisher EPUB — the fixture
this whole EPUB/HTML rebuild was calibrated against (see the build plan's
Context section: mfenced 1168x, mtable 340x, munder 493x/47x, no <table>,
no footnotes, 35 images). Not committed to the repo (it's the user's
personal library) and not the same book on every machine, so this is a
sanity check against a specific real book on this machine, not a portable
CI fixture — that's what test_epub_convert.py's synthetic probe is for.

Marked slow: skipped by default because the source file lives outside the
repo. Run explicitly with `pytest -m slow` or plain `pytest` (no -m filter).
"""
from pathlib import Path

import pytest

from kindle_math_converter.pipeline import Pipeline, PipelineConfig

REAL_EPUB = Path(
    "/Users/oliverjandette/Documents/Library/Books/Non-equilibrium Thermodynamics/"
    "Quantum Statistical Mechanics.epub"
)


@pytest.mark.slow
def test_real_epub_converts_without_dropping_equations(tmp_path):
    if not REAL_EPUB.exists():
        pytest.skip(f"real fixture not present at {REAL_EPUB} — this is a local-library fixture")

    out_dir = tmp_path / "out"
    config = PipelineConfig(output_dir=str(out_dir), epubcheck_enabled=False)
    result = Pipeline(config).run(str(REAL_EPUB), str(out_dir))

    assert result.ok, result.fatal_error
    assert result.total_equations > 2000  # ~2205 at last count

    # The whole point of this rebuild: an equation is allowed to be flagged
    # (compile/derivation can fail — see README "EPUB/HTML MathML
    # derivation"), but never silently dropped with no reader-visible trace.
    # flagged_for_review equations without a rendered fallback used to
    # vanish entirely (s10_epub_assembly._render_equation returning None) —
    # that gap is what EquationRegion.text_fallback closes.
    assert result.equations_flagged / result.total_equations < 0.15

    assert result.output_epub_path is not None
    epub_path = Path(result.output_epub_path)
    assert epub_path.exists()

    import zipfile
    with zipfile.ZipFile(epub_path) as zf:
        chapter_files = [
            n for n in zf.namelist()
            if n.endswith(".xhtml") and "nav" not in n.lower()
        ]
        assert len(chapter_files) >= 6  # one per real chapter, at minimum
        combined = "\n".join(zf.read(n).decode("utf-8") for n in chapter_files)

    assert "[[EQ:" not in combined  # no leaked placeholders
    assert "&#160;</p>" not in combined or len(combined) > 100_000  # not an empty-book regression

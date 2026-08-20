"""
PDF-path regression test using a real, equation-dense document — the
portable sibling of test_pdf_regression.py.

That test needs a source PDF from the user's personal library and is
marked slow (skipped by default) because of it. This one uses an
openly-licensed arXiv paper (CC BY 4.0 — see fixtures/tdho/ATTRIBUTION.md)
committed straight into the repo, so it's not slow in any sense and runs
as part of the normal `pytest -m "not slow"` CI gate.

Like test_pdf_regression.py, this reuses a committed MinerU parse
(fixtures/tdho/mineru/) rather than re-running MinerU — CI has no MinerU
install (see requirements-ci.txt), and local parses are deterministic
(see memory/pipeline_state_vector.md) so full snapshot identity is the
right assertion, not "close enough".
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

from kindle_math_converter.models.results import PipelineResult
from kindle_math_converter.pipeline import Pipeline, PipelineConfig
from kindle_math_converter.qa.latex_snapshot import build_snapshot, compare, format_report

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "tdho"
TDHO_PDF = FIXTURE_DIR / "tdho_excerpt.pdf"
CACHED_MINERU_PARSE = FIXTURE_DIR / "mineru"
BASELINE_SNAPSHOT = FIXTURE_DIR / "tdho_baseline_latex_snapshot.json"


def test_tdho_paper_recognition_matches_baseline(tmp_path):
    out_dir = tmp_path / "out"
    # Reuses the committed parse (mineru_reuse_existing=True is the
    # default) — the same "copy the mineru dir into a fresh output dir"
    # mechanism the README documents for humans.
    shutil.copytree(CACHED_MINERU_PARSE, out_dir / "mineru")

    config = PipelineConfig(output_dir=str(out_dir), epubcheck_enabled=False)
    pipeline = Pipeline(config)
    result = PipelineResult(
        document_path=str(TDHO_PDF), started_at=datetime.utcnow(),
        finished_at=None, ok=False, output_epub_path=None,
    )
    document = pipeline._run_stages(str(TDHO_PDF), str(out_dir), result)

    candidate = build_snapshot(document)
    baseline = json.loads(BASELINE_SNAPSHOT.read_text(encoding="utf-8"))

    report = compare(baseline, candidate)
    assert report["identical"], format_report(report)

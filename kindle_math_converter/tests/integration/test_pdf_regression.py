"""
PDF-path regression test — gives the *other* branch (untouched by this
session's EPUB/HTML work) the same kind of pytest coverage, using
infrastructure that already existed: a cached MinerU parse and the
snapshot-diff oracle (qa/latex_snapshot.py, normally driven by hand via
`main.py compare-snapshots`).

Marked slow (skipped by default, `pytest` with no -m filter runs it) because
it depends on a source PDF from the user's personal library, outside the
repo — same reasoning as test_real_epub.py. It is NOT slow in wall-clock
terms: reusing the cached parse under output/TEST/VDIGITAL/mineru (which the
README documents as reusable — copy it into a fresh output dir) means no
MinerU/GPU work happens, just tectonic compiles for 152 equations, well
under a minute.

Local parses are deterministic (see memory/pipeline_state_vector.md) — only
remote/GPU parses vary run to run — so this asserts full snapshot identity,
not just "close enough".
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from kindle_math_converter.models.results import PipelineResult
from kindle_math_converter.pipeline import Pipeline, PipelineConfig
from kindle_math_converter.qa.latex_snapshot import build_snapshot, compare, format_report

SU3_PDF = Path(
    "/Users/oliverjandette/Documents/Projects/Geometric Traffic Analysis/"
    "Starting Bibliography/Three-Dimensional Isotropic Harmonic Oscillator and SU3.pdf"
)
CACHED_MINERU_PARSE = (
    Path(__file__).resolve().parents[3] / "output" / "TEST" / "VDIGITAL" / "mineru"
)
BASELINE_SNAPSHOT = Path(__file__).parent.parent / "fixtures" / "su3_baseline_latex_snapshot.json"


@pytest.mark.slow
def test_su3_paper_recognition_matches_baseline(tmp_path):
    if not SU3_PDF.exists():
        pytest.skip(f"source PDF not present at {SU3_PDF} — this is a local-library fixture")
    if not CACHED_MINERU_PARSE.exists():
        pytest.skip(f"cached MinerU parse not present at {CACHED_MINERU_PARSE}")

    out_dir = tmp_path / "out"
    # Reuses the cached parse (mineru_reuse_existing=True is the default) —
    # this is the same "copy the mineru dir into a fresh output dir" reuse
    # mechanism the README documents for humans, exercised here by a test.
    shutil.copytree(CACHED_MINERU_PARSE, out_dir / "mineru")

    config = PipelineConfig(output_dir=str(out_dir), epubcheck_enabled=False)
    pipeline = Pipeline(config)
    result = PipelineResult(
        document_path=str(SU3_PDF), started_at=datetime.utcnow(),
        finished_at=None, ok=False, output_epub_path=None,
    )
    document = pipeline._run_stages(str(SU3_PDF), str(out_dir), result)

    candidate = build_snapshot(document)
    baseline = json.loads(BASELINE_SNAPSHOT.read_text(encoding="utf-8"))

    report = compare(baseline, candidate)
    assert report["identical"], format_report(report)

"""
Regression coverage for the pipeline's JSON log file.

Every stage module does `log = get_logger("stagename")` at *import* time —
before `configure_logging()` ever runs (Python resolves `pipeline.py`'s
stage imports before `main()`'s body executes). structlog's own
`BoundLoggerLazyProxy.bind()` resolves against whatever global config exists
at the exact moment it's called, not lazily at each log call — so every
stage logger used to bind against structlog's unconfigured defaults, and the
JSON log file was empty of pipeline events on every real run this project
has ever made. `get_logger()` now returns a `_LazyStageLogger` that defers
that resolution to each log call instead (see observability/logger.py).
This test exercises the real code path (`configure_logging()` +
`Pipeline.run()`, same as `main.py`) rather than trusting a smaller repro.
"""
import json

from kindle_math_converter.observability.logger import configure_logging
from kindle_math_converter.pipeline import Pipeline, PipelineConfig
from kindle_math_converter.tests.fixtures.make_probe_epub import build


def _read_json_lines(log_file):
    records = []
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_log_file_captures_every_stage(tmp_path):
    epub_path = build(tmp_path / "probe.epub")
    out_dir = tmp_path / "out"
    log_file = out_dir / "probe_pipeline.log"

    configure_logging(log_file, verbose=False)
    config = PipelineConfig(output_dir=str(out_dir), epubcheck_enabled=False)
    result = Pipeline(config).run(str(epub_path), str(out_dir))

    assert result.ok, result.fatal_error
    records = _read_json_lines(log_file)
    assert records, "log file is empty — the eager-bind regression this test guards against"

    stages_seen = {r.get("stage") for r in records}
    # Every stage the EPUB path actually runs must have logged something —
    # not just the "_meta" run_start marker or the pipeline's own logger.
    for expected_stage in (
        "s01_classifier", "s02c_epub", "s05c_mathml",
        "s06_validation", "s10_epub_assembly",
    ):
        assert expected_stage in stages_seen, (
            f"{expected_stage} never wrote a log record — its module-level "
            f"`log = get_logger(...)` bound before configure_logging() ran"
        )


def test_log_file_appends_and_delimits_runs_instead_of_overwriting(tmp_path):
    epub_path = build(tmp_path / "probe.epub")
    out_dir = tmp_path / "out"
    log_file = out_dir / "probe_pipeline.log"
    config = PipelineConfig(output_dir=str(out_dir), epubcheck_enabled=False)

    configure_logging(log_file, verbose=False)
    Pipeline(config).run(str(epub_path), str(out_dir))
    records_after_first = _read_json_lines(log_file)

    # A later re-run of the same book (e.g. after a fix) must not destroy the
    # evidence of the earlier run — that's the one thing worth keeping for
    # someone diagnosing an issue found weeks after the fact.
    configure_logging(log_file, verbose=False)
    Pipeline(config).run(str(epub_path), str(out_dir))
    records_after_second = _read_json_lines(log_file)

    assert len(records_after_second) > len(records_after_first)
    run_starts = [r for r in records_after_second if r.get("event") == "run_start"]
    assert len(run_starts) == 2
    assert run_starts[0]["run_id"] != run_starts[1]["run_id"]


def test_unexpected_pipeline_failure_is_logged_with_a_traceback(tmp_path):
    epub_path = build(tmp_path / "probe.epub")
    out_dir = tmp_path / "out"
    log_file = out_dir / "probe_pipeline.log"

    configure_logging(log_file, verbose=False)
    config = PipelineConfig(output_dir=str(out_dir), epubcheck_enabled=False)
    pipeline = Pipeline(config)

    def boom(*_args, **_kwargs):
        raise TypeError("simulated unexpected bug")

    pipeline._run_stages = boom
    result = pipeline.run(str(epub_path), str(out_dir))

    assert not result.ok
    assert result.fatal_error == "simulated unexpected bug"

    records = _read_json_lines(log_file)
    crash_records = [r for r in records if r.get("event") == "pipeline_unexpected_error"]
    assert len(crash_records) == 1
    assert "Traceback (most recent call last)" in crash_records[0].get("exception", "")

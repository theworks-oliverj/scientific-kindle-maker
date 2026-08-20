import logging
import sys
import uuid
from pathlib import Path

import structlog


class _TeeStream:
    """
    Writes to the JSON log file and, when `verbose`, also echoes the same
    raw JSON lines to stdout.

    structlog's `PrintLoggerFactory` writes directly to a single file
    object, bypassing stdlib logging (and RichHandler) entirely — so
    without this, `--verbose` would have no effect on structlog's own
    stage/equation events (only on third-party libraries' stdlib-logging
    output via RichHandler), a dead-looking flag. This keeps `--verbose`'s
    documented meaning ("detailed per-equation logging in terminal") true;
    the file always receives everything regardless of `verbose`.
    """

    def __init__(self, primary, echo_to=None):
        self._primary = primary
        self._echo_to = echo_to

    def write(self, s):
        self._primary.write(s)
        if self._echo_to is not None:
            self._echo_to.write(s)

    def flush(self):
        self._primary.flush()
        if self._echo_to is not None:
            self._echo_to.flush()


class _LazyStageLogger:
    """
    Defers structlog's `.bind()` to each actual log call instead of doing it
    at construction time.

    Every stage module does `log = get_logger("stagename")` at import time —
    which happens before `configure_logging()` runs, since Python resolves
    `pipeline.py`'s stage imports before `main()`'s body executes. structlog's
    `BoundLoggerLazyProxy.bind()` resolves against whatever global config
    exists at the *exact moment it's called*, not lazily at log time — so
    every stage logger used to bind against structlog's unconfigured
    defaults (a bare stdout `ConsoleRenderer`) and never routed through
    `configure_logging()`'s file handler at all. The JSON log file was
    consequently empty of pipeline events on every run. This wrapper defers
    resolution to each `.info()`/`.warning()`/etc. call, by which point
    `configure_logging()` has always already run.
    """

    def __init__(self, stage: str):
        self._stage = stage

    def _bound(self):
        return structlog.get_logger().bind(stage=self._stage)

    def __getattr__(self, name):
        return getattr(self._bound(), name)


def configure_logging(log_file: Path, verbose: bool = False) -> None:
    """
    Call once at startup (in main.py / CLI entry point), before any stage
    does real work.

    Produces:
      1. log_file: newline-delimited JSON, one object per event. Appended
         to (not overwritten) across runs against the same output
         directory, so a run's record survives a later re-run of the same
         book — the case this exists for is a book found to have issues
         weeks after it was processed. Each run is delimited by its own
         "run_start" event carrying a `run_id`. This is the primary
         post-hoc diagnostic record; report_builder.py's HTML report is
         built from a separate in-memory EventBus, not this file.
      2. stderr, via stdlib logging + RichHandler: third-party library
         output, gated by `verbose` as before. When `verbose=True`,
         structlog's own stage/equation JSON lines are also echoed to
         stdout as they're written (raw JSON, not pretty-printed) — see
         `_TeeStream`. With `verbose=False` the console stays quiet for
         routine events; an unexpected (not explicitly classified) pipeline
         failure still prints a full traceback to the console directly,
         unconditionally, from pipeline.py's top-level handler, regardless
         of `verbose`.

    Log levels:
      DEBUG   — per-equation detail (recognition output, CDM scores, cache lookups)
      INFO    — stage start/end, document-level summary, equation counts
      WARNING — recoverable issues (repair triggered, fallback used, low CDM score)
      ERROR   — stage failure, equation flagged, compile error
      CRITICAL — pipeline abort

    Note on console output: structlog's `PrintLoggerFactory` writes directly
    to a file object, bypassing stdlib logging (and therefore RichHandler)
    entirely — so `verbose` previously had no real effect on structlog's own
    console visibility despite appearing to (the pre-fix eager-bind bug meant
    every stage logger printed unconditionally via structlog's unconfigured
    defaults instead, regardless of `verbose`). `_TeeStream` restores
    `verbose`'s intended meaning directly rather than leaving it silently
    dead.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    from rich.logging import RichHandler

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        level=logging.DEBUG if verbose else logging.WARNING,
    )

    # Deliberately NOT attached to json_file_handler: third-party libraries
    # (PIL, MinerU) log through stdlib logging too, and root-logging them
    # into the same file structlog writes to would interleave plain-text
    # lines into what's documented as pure newline-delimited JSON. Stdlib
    # logging's job here is console-only, gated by `verbose`.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()  # repeated configure_logging() calls (e.g. in tests) must not stack duplicate handlers
    root_logger.addHandler(rich_handler)

    # structlog gets its own file handle so the JSONL contract above holds
    # regardless of what third-party libraries log through stdlib.
    log_fh = open(log_file, "a", encoding="utf-8")
    stream = _TeeStream(log_fh, sys.stdout if verbose else None)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,  # renders exc_info=True into a string "exception" field, ahead of JSONRenderer
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        logger_factory=structlog.PrintLoggerFactory(file=stream),
    )

    structlog.get_logger().bind(stage="_meta").info("run_start", run_id=uuid.uuid4().hex[:12])


def get_logger(stage: str) -> _LazyStageLogger:
    """
    Returns a logger-like object bound to `stage`, resolved against
    structlog's actual configuration at each log call rather than at this
    call. Usage unchanged: log = get_logger("s06_validation")
    """
    return _LazyStageLogger(stage)

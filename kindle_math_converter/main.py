"""
CLI entry point for the Kindle Math Converter.

Usage: python -m kindle_math_converter.main [OPTIONS] INPUT_PATH
"""
import os
import sys
import webbrowser
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .model_manager import load_all_models, get_models_dir
from .observability.logger import configure_logging
from .pipeline import Pipeline, PipelineConfig

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Kindle Math Converter — convert PDF/EPUB to Kindle EPUB3 with SVG equations."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("download-models")
@click.option("--models-dir", default=None,
              help="Override model cache directory (default: ~/.kindle_converter/models).")
def download_models_cmd(models_dir: str | None) -> None:
    """
    Pre-download all ML model weights from HuggingFace.

    \b
    Models downloaded:
      - DocLayout-YOLO   (juliozhao/DocLayout-YOLO-DocStructBench)
      - YOLOv8-MFD       (opendatalab/PDF-Extract-Kit-1.0 → models/MFD/yolov8_mix.pt)
      - pix2tex          (auto-managed by pix2tex package, ~1.4 GB)
      - PaddleOCR        (auto-managed by paddleocr package)

    Weights are cached in ~/.kindle_converter/models/ and reused on subsequent runs.
    Run this once before converting documents for the first time.
    """
    from pathlib import Path as _Path
    mdir = _Path(models_dir) if models_dir else get_models_dir()

    configure_logging(mdir / "download.log", verbose=True)

    console.print(f"\n[bold]Downloading model weights[/bold]")
    console.print(f"  Cache dir: {mdir}\n")

    with console.status("[bold green]Downloading…", spinner="dots"):
        models = load_all_models(mdir)

    loaded  = [k for k, v in models.items() if v is not None]
    missing = [k for k, v in models.items() if v is None]

    console.print(f"  Loaded:  {', '.join(loaded) or 'none'}")
    if missing:
        console.print(f"  [yellow]Missing: {', '.join(missing)}[/yellow]")
        console.print("  Install missing packages from requirements.txt and re-run.")
    else:
        console.print("\n[bold green]All models ready.[/bold green]")


@cli.command("convert")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default="./output", show_default=True,
              help="Directory for output files.")
@click.option("--verbose", is_flag=True, default=False,
              help="Show detailed per-equation logging in terminal.")
@click.option("--cdm-threshold", default=0.88, show_default=True, type=float,
              help="CDM pass threshold (0–1).")
@click.option("--font-size", default=10.0, show_default=True, type=float,
              help="Body font size in pt for SVG em sizing.")
@click.option("--mathpix-id", default=None, envvar="MATHPIX_APP_ID",
              help="Mathpix App ID for fallback recognition.")
@click.option("--mathpix-key", default=None, envvar="MATHPIX_APP_KEY",
              help="Mathpix App Key for fallback recognition.")
@click.option("--open-report", is_flag=True, default=False,
              help="Open the HTML report in browser after completion.")
@click.option("--dump-cache", is_flag=True, default=False,
              help="Write cache dump JSON to output dir after completion.")
@click.option("--no-epubcheck", is_flag=True, default=False,
              help="Skip epubcheck validation (useful if Java is not available).")
@click.option("--debug-dump", is_flag=True, default=False,
              help="Write per-detection crops and detections/recognition JSONL "
                   "to <output-dir>/debug_crops for auditing Stage 3/5B.")
def main(
    input_path: str,
    output_dir: str,
    verbose: bool,
    cdm_threshold: float,
    font_size: float,
    mathpix_id: str | None,
    mathpix_key: str | None,
    open_report: bool,
    dump_cache: bool,
    no_epubcheck: bool,
    debug_dump: bool,
) -> None:
    """
    Convert a PDF or EPUB to Kindle-compatible EPUB3 with properly rendered
    math equations as scalable SVG.

    \b
    Examples:
      python -m kindle_math_converter.main convert feynman_lectures_v2.pdf
      python -m kindle_math_converter.main convert griffiths_em.pdf --output-dir ./books --verbose
      python -m kindle_math_converter.main convert quantum_mechanics.epub --cdm-threshold 0.85
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem
    log_file = out_dir / f"{stem}_pipeline.log"

    configure_logging(log_file, verbose=verbose)

    config = PipelineConfig(
        cdm_pass_threshold=cdm_threshold,
        cdm_repair_threshold=max(0.0, cdm_threshold - 0.18),
        body_font_size_pt=font_size,
        mathpix_app_id=mathpix_id,
        mathpix_app_key=mathpix_key,
        output_dir=output_dir,
        verbose_logging=verbose,
        dump_cache=dump_cache,
        epubcheck_enabled=not no_epubcheck,
        s03_debug_dump_enabled=debug_dump,
    )

    console.print(f"\n[bold]Kindle Math Converter[/bold]")
    console.print(f"  Source:  {input_path}")
    console.print(f"  Output:  {output_dir}\n")

    pipeline = Pipeline(config)

    with console.status("[bold green]Processing…", spinner="dots"):
        result = pipeline.run(input_path, output_dir)

    _print_summary(result, out_dir, stem, open_report)

    sys.exit(0 if result.ok else 1)


def _print_summary(result, out_dir: Path, stem: str, open_report: bool) -> None:
    from datetime import timedelta

    duration = ""
    if result.finished_at and result.started_at:
        secs = int((result.finished_at - result.started_at).total_seconds())
        duration = f"{secs // 60}m {secs % 60}s"

    total = result.total_equations
    passed = result.equations_passed
    repaired = result.equations_repaired
    fallback = result.equations_fallback
    flagged = result.equations_flagged

    cache_total = result.cache_hits + result.cache_misses
    hit_rate = f"{100 * result.cache_hits / cache_total:.1f}%" if cache_total else "n/a"

    report_path = out_dir / f"{stem}_report.html"
    epub_path = result.output_epub_path or str(out_dir / f"{stem}.epub")

    status_color = "green" if result.ok else "red"
    status_text = "Complete" if result.ok else "FAILED"

    lines = [
        f"[{status_color}]{status_text}[/{status_color}]",
        f"  Source:     {result.document_path}",
        f"  Output:     {epub_path}",
        f"  Equations:  {total} total",
        f"              {passed} passed  ({100*passed/total:.1f}%)" if total else "              0 equations",
        f"               {repaired} repaired ({100*repaired/total:.1f}%)" if total else "",
        f"               {fallback} fallback  ({100*fallback/total:.1f}%)" if total else "",
        f"               {flagged} flagged   ({100*flagged/total:.1f}%)" if total else "",
        f"  Cache:       {result.cache_hits} hits / {result.cache_misses} misses ({hit_rate} hit rate)",
        f"  Duration:    {duration}",
        f"  Report:     {report_path}",
    ]

    console.print("\n" + "\n".join(l for l in lines if l))

    if flagged > 0:
        console.print(
            f"\n[bold yellow]  ⚠  {flagged} equation(s) could not be rendered.[/bold yellow]"
        )
        console.print(f"     Open the report and review: {report_path}")

    if result.fatal_error:
        console.print(f"\n[bold red]  Fatal error: {result.fatal_error}[/bold red]")

    if open_report and report_path.exists():
        webbrowser.open(str(report_path))


if __name__ == "__main__":
    cli()

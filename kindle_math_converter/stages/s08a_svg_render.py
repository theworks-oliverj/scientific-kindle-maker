"""
Stage 8A — SVG Rendering
Renders validated equations to SVG via tectonic + dvisvgm.
Checks session cache before invoking tectonic.

Key design invariants:
  - dvisvgm is ALWAYS called with --no-fonts (eliminates KDP px→rem bug)
  - Cache is checked before rendering, populated after successful render
  - Tectonic timeout: 15s per equation; dvisvgm timeout: 10s per equation
"""
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ..cache.equation_cache import SessionEquationCache
from ..concurrency import parallel_for_each
from ..models.document import Document, EquationRegion, FailureReason
from ..models.enums import ConfidenceGate, ErrorCode, FormulaClass
from ..models.results import StageResult
from ..observability.event_bus import EventBus
from ..observability.logger import get_logger
# Reuse the dynamic wrapper builder so package sets stay in sync with Stage 6
from .s06_validation import _build_latex_wrapper, warm_up_tectonic

log = get_logger("s08a_svg_render")


def _find_binary(name: str) -> str:
    """
    Returns the full path for a binary, checking common Homebrew locations
    in addition to the current PATH. Raises FileNotFoundError if not found.
    """
    import shutil
    found = shutil.which(name)
    if found:
        return found
    # Homebrew on Apple Silicon / Intel
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
        p = Path(prefix) / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"'{name}' not found — install via: brew install {name}")


def _dvisvgm_env() -> dict[str, str]:
    """
    Builds the environment dict for dvisvgm so it can trace Type1 glyphs to paths.

    dvisvgm needs two things that are not on PATH by default with a Homebrew
    tectonic install:
      LIBGS    — path to libgs.dylib so dvisvgm can call Ghostscript for
                 Type1/OTF tracing.  Without this, --no-fonts produces <use>
                 references to undefined symbols (blank equations in EPUB).
      TEXFONTS — colon-separated directories containing .pfb / .tfm files.
                 Tectonic caches these under ~/Library/Caches/Tectonic on macOS
                 and ~/.cache/Tectonic on Linux.
    """
    env = os.environ.copy()

    # ── Ghostscript dylib ────────────────────────────────────────────────
    if "LIBGS" not in env:
        candidates = [
            Path("/opt/homebrew/lib/libgs.dylib"),          # Apple Silicon brew
            Path("/usr/local/lib/libgs.dylib"),              # Intel brew
            Path("/usr/lib/x86_64-linux-gnu/libgs.so.9"),   # Debian/Ubuntu
            Path("/usr/lib/libgs.so"),
        ]
        for p in candidates:
            if p.exists():
                env["LIBGS"] = str(p)
                break

    # ── Tectonic font cache ──────────────────────────────────────────────
    if "TEXFONTS" not in env:
        cache_roots = [
            Path.home() / "Library" / "Caches" / "Tectonic" / "bundles" / "data",  # macOS
            Path.home() / ".cache" / "Tectonic" / "bundles" / "data",               # Linux
        ]
        font_dirs: list[str] = []
        for root in cache_roots:
            if root.is_dir():
                for subdir in root.iterdir():
                    if subdir.is_dir() and any(subdir.glob("*.pfb")):
                        font_dirs.append(str(subdir))
        if font_dirs:
            existing = env.get("TEXFONTS", "")
            env["TEXFONTS"] = ":".join(font_dirs) + (":" + existing if existing else ":")

    return env


class LatexCompileError(Exception):
    pass


class SVGRenderError(Exception):
    pass


def latex_to_xdv(latex: str, formula_class: FormulaClass, work_dir: Path, timeout: int = 15) -> Path:
    """
    Wraps the equation in a minimal document and compiles to XDV via tectonic.

    tectonic's intermediate format is XDV (extended DVI), NOT dvi.
    --outfmt=dvi is rejected; --outfmt=xdv is the correct flag.
    dvisvgm accepts XDV files with the same flags as DVI.
    """
    tex_file = work_dir / "equation.tex"
    tex_file.write_text(_build_latex_wrapper(latex, formula_class), encoding="utf-8")

    result = subprocess.run(
        [_find_binary("tectonic"), "--outfmt=xdv", str(tex_file)],
        capture_output=True,
        timeout=timeout,
        cwd=work_dir,
    )
    if result.returncode != 0:
        raise LatexCompileError(result.stderr.decode(errors="replace"))

    xdv_path = work_dir / "equation.xdv"
    if not xdv_path.exists():
        raise LatexCompileError("tectonic produced no XDV output")

    return xdv_path


def xdv_to_svg(xdv_path: Path, work_dir: Path, timeout: int = 10) -> str:
    """
    Converts XDV to SVG using dvisvgm.

    Critical flags:
      --no-fonts    Convert all glyphs to <path> elements.
                    MANDATORY — eliminates KDP px→rem font-size corruption bug.
      --exact-bbox  Precise bounding box from glyph outlines, not TFM metrics.
      --precision=4 4 decimal places in path coordinates.
      --bbox=min    Tight bounding box around actual content.

    dvisvgm handles XDV files the same as DVI — no extra flags needed.
    """
    svg_path = work_dir / "equation.svg"
    result = subprocess.run(
        [
            _find_binary("dvisvgm"),
            "--no-fonts",
            "--exact-bbox",
            "--precision=4",
            "--bbox=min",
            f"--output={svg_path}",
            str(xdv_path),
        ],
        capture_output=True,
        timeout=timeout,
        cwd=work_dir,
        env=_dvisvgm_env(),
    )
    if result.returncode != 0:
        raise SVGRenderError(result.stderr.decode(errors="replace"))

    if not svg_path.exists():
        raise SVGRenderError("dvisvgm produced no SVG output")

    return svg_path.read_text(encoding="utf-8")


def _render_region(
    region: EquationRegion,
    latex: str,
    cache: SessionEquationCache,
    tectonic_timeout: int,
    dvisvgm_timeout: int,
    stage: str,
) -> dict:
    """
    Renders a single equation region via tectonic + dvisvgm, mutating it in
    place (svg, confidence_gate, error_codes, failure_reason on failure).

    Returns a result record for sequential aggregation by the caller —
    `rendered` (bool), an optional `warning` string, and an `event`
    (event_type, payload) for `bus.emit`. Each region/latex pair is
    independent (own temp dir, own subprocess calls), so this is safe to
    call from a thread pool; `cache.put` is internally locked.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            # s06's gate already compiled this .tex (both stages build it with
            # s06's _build_latex_wrapper) and kept the XDV. Reuse it rather
            # than paying for a second identical tectonic run — but only when
            # it was compiled from the very string being rendered here.
            # `latex` is normalized_latex, which can differ from what s06
            # compiled, so this equality is the safety property: a mismatch
            # costs a compile, never a wrong equation.
            if region.compiled_xdv and region.compiled_xdv_latex == latex:
                xdv_path = work / "equation.xdv"
                xdv_path.write_bytes(region.compiled_xdv)
            else:
                xdv_path = latex_to_xdv(latex, region.formula_class, work, tectonic_timeout)
            svg = xdv_to_svg(xdv_path, work, dvisvgm_timeout)

        region.svg = svg
        cache.put(latex, svg, region.cdm_score or 0.0)
        return {
            "rendered": True,
            "warning": None,
            "event": ("equation_ok", {"equation_id": region.region_id}),
        }

    except subprocess.TimeoutExpired:
        region.confidence_gate = ConfidenceGate.FALLBACK
        region.error_codes.append(ErrorCode.TECTONIC_FAILED.value)
        region.failure_reason = FailureReason(
            code=ErrorCode.TECTONIC_FAILED.value,
            sub_code=ErrorCode.TECTONIC_FAILED.value,
            stage=stage,
            detail="tectonic timed out during SVG render",
            recoverable=False,
        )
        return {
            "rendered": False,
            "warning": f"{region.region_id}: tectonic timeout",
            "event": ("equation_error", {"equation_id": region.region_id, "error": "timeout"}),
        }

    except LatexCompileError as exc:
        region.confidence_gate = ConfidenceGate.FALLBACK
        region.error_codes.append(ErrorCode.TECTONIC_FAILED.value)
        region.failure_reason = FailureReason(
            code=ErrorCode.TECTONIC_FAILED.value,
            sub_code=ErrorCode.TECTONIC_FAILED.value,
            stage=stage,
            detail=f"tectonic compile error: {str(exc)[:120]}",
            recoverable=False,
        )
        return {
            "rendered": False,
            "warning": f"{region.region_id}: compile failed — {exc}",
            "event": ("equation_error", {"equation_id": region.region_id, "error": str(exc)}),
        }

    except (SVGRenderError, FileNotFoundError) as exc:
        region.confidence_gate = ConfidenceGate.FALLBACK
        region.error_codes.append(ErrorCode.DVISVGM_FAILED.value)
        region.failure_reason = FailureReason(
            code=ErrorCode.DVISVGM_FAILED.value,
            sub_code=ErrorCode.DVISVGM_FAILED.value,
            stage=stage,
            detail=f"dvisvgm error: {str(exc)[:120]}",
            recoverable=False,
        )
        return {
            "rendered": False,
            "warning": f"{region.region_id}: dvisvgm failed — {exc}",
            "event": ("equation_error", {"equation_id": region.region_id, "error": str(exc)}),
        }


def run(
    pass_list: list[EquationRegion],
    cache: SessionEquationCache,
    bus: EventBus,
    document: Document | None = None,
    tectonic_timeout: int = 15,
    dvisvgm_timeout: int = 10,
    max_parallel_workers: int | None = None,
) -> tuple[list[EquationRegion], list[EquationRegion], StageResult]:
    """
    Returns (rendered_list, failed_list, stage_result).
    failed_list equations are routed to Stage 8B.
    """
    t0 = time.perf_counter()
    stage = "s08a_svg_render"
    warnings: list[str] = []
    errors: list[str] = []
    rendered = 0
    cache_hits = 0
    failed_list: list[EquationRegion] = []

    bus.emit(stage, "stage_start", equations=len(pass_list))
    log.info("stage_start", equations=len(pass_list))

    # ── Pass 1 (sequential, cheap): resolve no-latex/cache-hit regions ──────
    # Only regions that actually need a tectonic/dvisvgm subprocess call are
    # handed to the thread pool.
    to_render: list[tuple[EquationRegion, str]] = []
    for region in pass_list:
        latex = region.normalized_latex or region.raw_latex or ""
        if not latex:
            region.confidence_gate = ConfidenceGate.FALLBACK
            region.error_codes.append(ErrorCode.TECTONIC_FAILED.value)
            failed_list.append(region)
            continue

        # Check cache first (never skip recognition — only render is cached)
        cached = cache.get(latex)
        if cached is not None:
            region.svg = cached.svg
            cache_hits += 1
            bus.emit(stage, "cache_hit", equation_id=region.region_id)
            rendered += 1
            continue

        to_render.append((region, latex))

    # ── Pass 2 (parallel): tectonic + dvisvgm for cache misses ──────────────
    if to_render:
        warm_up_tectonic()

    results = parallel_for_each(
        to_render,
        lambda item: _render_region(item[0], item[1], cache, tectonic_timeout, dvisvgm_timeout, stage),
        max_workers=max_parallel_workers,
    )

    # ── Pass 3 (sequential retry): parallel-pass failures get one calm,
    # uncontended attempt. Transient tectonic timeouts under an 8-way thread
    # pool (bundle-cache and CPU contention) are common enough that retrying
    # alone recovers most of them; a genuine compile error fails again fast.
    retry_failed = 0
    for (region, latex), result in zip(to_render, results):
        if not result["rendered"]:
            retry = _render_region(region, latex, cache, tectonic_timeout, dvisvgm_timeout, stage)
            if retry["rendered"]:
                region.confidence_gate = ConfidenceGate.PASS
                region.failure_reason = None
                region.error_codes = [
                    c for c in region.error_codes
                    if c not in (ErrorCode.TECTONIC_FAILED.value, ErrorCode.DVISVGM_FAILED.value)
                ]
                bus.emit(stage, "render_retry_ok", equation_id=region.region_id)
                result.update(retry)
            else:
                retry_failed += 1
                result.update(retry)

    for (region, _latex), result in zip(to_render, results):
        if result["rendered"]:
            rendered += 1
        else:
            failed_list.append(region)
        if result["warning"]:
            warnings.append(result["warning"])
        event_type, payload = result["event"]
        bus.emit(stage, event_type, **payload)

    # ── Stage 1.3: copy the canonical's raw SVG to its dedup duplicates ────
    # Duplicates never appear in pass_list (filtered out in Stage 7), so they
    # need to be reached via `document`. Only the RAW svg is copied — each
    # duplicate has its own region_id, so Stage 9's per-region
    # namespace_svg_ids() call produces distinct glyph ids for it.
    dedup_copies = 0
    if document is not None:
        canonical_by_id = {r.region_id: r for r in pass_list if r.svg is not None}
        for region in document.all_equations:
            if region.dedup_canonical_id is None:
                continue
            canonical = canonical_by_id.get(region.dedup_canonical_id)
            if canonical is None:
                continue
            region.svg = canonical.svg
            region.cdm_score = canonical.cdm_score
            region.confidence_gate = canonical.confidence_gate
            dedup_copies += 1
            bus.emit(stage, "equation_ok", equation_id=region.region_id, source="dedup_copy")

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    # How many renders skipped a redundant tectonic run by reusing s06's
    # compile. A low number means normalization is changing the LaTeX between
    # the two stages and the saving is not being realised.
    xdv_reused = sum(
        1 for region, latex in to_render
        if region.compiled_xdv and region.compiled_xdv_latex == latex
    )
    bus.emit(stage, "stage_end", rendered=rendered, cache_hits=cache_hits, failed=len(failed_list), dedup_copies=dedup_copies, xdv_reused=xdv_reused)
    log.info("stage_end", rendered=rendered, cache_hits=cache_hits, failed=len(failed_list), dedup_copies=dedup_copies, xdv_reused=xdv_reused)

    rendered_list = [r for r in pass_list if r.svg is not None]
    return rendered_list, failed_list, StageResult(
        stage_name=stage,
        ok=True,
        duration_ms=duration_ms,
        warnings=warnings,
        errors=errors,
        metrics={
            "rendered": rendered,
            "cache_hits": cache_hits,
            "xdv_reused": xdv_reused,
            "failed": len(failed_list),
            "dedup_copies": dedup_copies,
        },
    )

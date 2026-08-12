"""
Modal spike — run MinerU's parse on a remote GPU and bring the result home.

WHY THIS EXISTS
    MinerU is 86–95% of this pipeline's wall clock (45 s/page on the local M4),
    and every local speedup measured so far corrupts equations. A remote GPU is
    the remaining lever. But on Linux MinerU runs `vllm-engine`, not the Mac's
    `mlx-engine` — same weights, different inference engine — so it carries the
    same recognition risk that got `hybrid-engine` and 8-bit quantization
    rejected. This script exists to measure that risk before anything is built
    on top of it.

    It answers one question: does a remote parse recognise the same LaTeX?

HOW IT PLUGS IN
    It does not touch the pipeline. s03 already reuses any middle.json found
    under `<output-dir>/mineru/`, so this writes the remote parse into exactly
    that layout and the normal local run picks it up and skips MinerU. That
    seam is why the integration is one standalone file.

ROLLBACK
    Nothing imports this and it imports nothing from kindle_math_converter.
    Delete the file (and `pip uninstall modal`) and the project is unchanged.

USAGE
    pip install modal
    python3 -m modal setup            # browser auth, one time

    # 1. parse remotely (start with the 6-page reference paper — it is cheapest)
    modal run modal_app.py --pdf "/path/to/paper.pdf" --output-dir ./output/REMOTE

    # 2. run the pipeline locally against it — s03 reuses the remote parse
    python main.py convert "/path/to/paper.pdf" --output-dir ./output/REMOTE

    # 3. the actual verdict
    python main.py compare-snapshots \\
        ./output/LOCAL_BASELINE/paper_latex_snapshot.json \\
        ./output/REMOTE/paper_latex_snapshot.json

COST
    Billed per second, no minimum. The parse itself is seconds for a short
    paper; the real cost is the one-time 2.2 GB model download, which the
    Volume below caches so later runs skip it. Use --pages to cap spend while
    testing.
"""
import modal

# Pinned to the locally installed version. The whole point is comparing engines,
# so everything else must be held identical — an unpinned version would confound
# the result, and for a shared repo it is what stops two users getting different
# LaTeX from the same PDF.
MINERU_VERSION = "3.4.4"
MODEL_REPO = "opendatalab/MinerU2.5-Pro-2605-1.2B"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # opencv (pulled in by mineru) needs these at import time
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(f"mineru[core,vlm,vllm]=={MINERU_VERSION}")
    .env(
        {
            # Same workaround the local runs use — Xet transfers stall on some
            # networks and the failure looks like a hang, not an error.
            "HF_HUB_DISABLE_XET": "1",
            "MINERU_MODEL_SOURCE": "huggingface",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

# The 2.2 GB model downloads once and is reused by every later run. Without
# this each invocation re-downloads it, which would dominate the bill.
hf_cache = modal.Volume.from_name("kmc-mineru-hf-cache", create_if_missing=True)

app = modal.App("kindle-math-converter-parse")


@app.function(
    image=image,
    gpu="A10G",
    # Generous but bounded. A short paper is a couple of minutes including the
    # first model download; a 500-page book would be well under an hour.
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def parse(
    pdf_bytes: bytes,
    pdf_name: str,
    start_page: int | None = None,
    end_page: int | None = None,
) -> tuple[bytes, str]:
    """Runs the MinerU CLI exactly as the local pipeline does, and returns the
    output tree as a gzipped tar plus the captured log.

    Invoking the CLI rather than importing MinerU is deliberate: it is the same
    call `_invoke_mineru` makes locally, so a difference in the result is a
    difference in the engine, not in how we drove it.
    """
    import io
    import subprocess
    import tarfile
    import time
    from pathlib import Path

    work = Path("/tmp/kmc")
    src_dir, out_dir = work / "src", work / "out"
    src_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = src_dir / pdf_name
    pdf_path.write_bytes(pdf_bytes)

    cmd = ["mineru", "-p", str(pdf_path), "-o", str(out_dir), "-b", "vlm-engine"]
    if start_page is not None and end_page is not None:
        cmd += ["-s", str(start_page), "-e", str(end_page)]

    print(f"[kmc] {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    log = (proc.stdout or "") + (proc.stderr or "")
    print(f"[kmc] mineru exited {proc.returncode} after {elapsed:.0f}s", flush=True)

    if proc.returncode != 0:
        raise RuntimeError(f"mineru failed (exit {proc.returncode}):\n{log[-4000:]}")

    # Persist the freshly downloaded model so later runs skip the download.
    hf_cache.commit()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(out_dir)))
    return buf.getvalue(), log


@app.local_entrypoint()
def main(
    pdf: str,
    output_dir: str = "./output/REMOTE",
    pages: str = "",
    save_log: bool = True,
):
    """Uploads `pdf`, parses it remotely, and unpacks the result into
    `<output_dir>/mineru/` — the location s03 already looks in.

    `--pages` accepts "START-END" (0-based, inclusive) to cap spend while
    testing; omit it to parse the whole document.
    """
    import io
    import tarfile
    import time
    from pathlib import Path

    src = Path(pdf)
    if not src.is_file():
        raise SystemExit(f"not a file: {pdf}")

    start = end = None
    if pages:
        try:
            start_s, end_s = pages.split("-", 1)
            start, end = int(start_s), int(end_s)
        except ValueError:
            raise SystemExit(f'--pages must look like "0-5", got {pages!r}')

    mineru_dir = Path(output_dir) / "mineru"
    mineru_dir.mkdir(parents=True, exist_ok=True)

    print(f"uploading {src.name} ({src.stat().st_size / 1e6:.1f} MB)…")
    t0 = time.perf_counter()
    tar_bytes, log = parse.remote(src.read_bytes(), src.name, start, end)
    elapsed = time.perf_counter() - t0

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        # filter="data" refuses absolute paths, "..", and odd metadata. The
        # archive is ours, but extraction is the wrong place to assume that.
        tar.extractall(mineru_dir, filter="data")

    middles = sorted(mineru_dir.rglob("*_middle.json"))
    print(f"\nround trip: {elapsed:.0f}s   unpacked into {mineru_dir}")
    for m in middles:
        print(f"  {m.relative_to(mineru_dir)}  ({m.stat().st_size / 1024:.0f} KB)")

    if save_log:
        log_path = Path(output_dir) / "mineru_remote.log"
        log_path.write_text(log, encoding="utf-8")
        print(f"  remote log: {log_path}")

    if not middles:
        raise SystemExit(
            "no middle.json came back — the parse produced nothing usable; "
            "check the remote log before spending more credit."
        )

    print(
        "\nnext:\n"
        f'  python main.py convert "{pdf}" --output-dir {output_dir}\n'
        "  python main.py compare-snapshots "
        f"<local-baseline>_latex_snapshot.json {output_dir}/{src.stem}_latex_snapshot.json"
    )

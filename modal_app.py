"""
Modal remote parse — run MinerU's GPU work off-box and bring middle.json home.

WHY THIS EXISTS
    MinerU is 86–95% of this pipeline's wall clock (45 s/page on the local M4),
    and every local speedup measured so far corrupts equations. A remote GPU is
    the remaining lever.

WHAT IS ALREADY KNOWN (measured, see the parser-experiments note)
    * Recognition is a peer, not an upgrade. vllm-engine and mlx-engine each sit
      near 98.7% on the reference corpus with different errors.
    * vllm-engine is NOT reproducible: 2 of 153 equations differed between two
      identical runs. MinerU already requests greedy decoding
      (temperature=0.0, top_k=1) on every backend, so this is vLLM's batched
      execution reordering floating-point reductions — a property to design
      around, not a setting to fix.
    * Throughput varied 7× between two runs (3.8 vs 28 s/page) and init varied
      4× (177 vs 688 s). Crucially both moved TOGETHER, which points at CPU
      starvation rather than the GPU: vLLM init is CPU-heavy, and the parse does
      rasterization and image preprocessing between GPU calls. The first runs
      requested no CPU at all, so they got Modal's minimum reservation and
      whatever the host had spare.

HOW IT PLUGS IN
    It does not touch the pipeline. s03 reuses any middle.json found under
    `<output-dir>/mineru/`, so results are unpacked into exactly that layout and
    a normal local run picks them up and skips MinerU. `--output-root` here IS
    the pipeline's `--output-dir`; point both at the same directory and there is
    nothing to move by hand. See the layout note further down.

ROLLBACK
    Nothing imports this; it imports nothing from kindle_math_converter.
    Delete the file (and `pip uninstall modal`) and the project is unchanged.

USAGE
    # One book, ready for the pipeline to pick up (the normal case).
    modal run modal_app.py --pdf "/path/book.pdf" \
        --output-root ~/Desktop/KindleReads/Book --batch-pages 250
    python main.py convert "/path/book.pdf" \
        --output-dir ~/Desktop/KindleReads/Book --parse-batch-size 250

    # --parse-batch-size must equal --batch-pages: it is what the batch
    # directory names encode. The run verifies this before it prints the
    # command, and re-running fills in any batch that failed.

    # Parser QA — needs the labelled layout, results moved into place by hand.
    modal run modal_app.py --pdf "/path/a.pdf" --repeat 2 --layout labelled
    modal run modal_app.py --pdf "/path/a.pdf" --pages 0-5 --layout labelled
"""
import re

import modal

# Pinned to the local install. The comparison is between inference engines, so
# everything else is held identical; unpinned, two users could get different
# LaTeX from the same PDF. See modal-requirements.txt.
MINERU_VERSION = "3.4.4"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # opencv needs these at import
    # uv, from a lock file. pip cannot resolve this tree — it backtracks through
    # cffi sdists indefinitely (observed >10 min, no end). uv does it in ~5 s.
    .uv_pip_install(requirements=["modal-requirements.txt"])
    .env(
        {
            "HF_HUB_DISABLE_XET": "1",  # Xet transfers stall on some networks
            "MINERU_MODEL_SOURCE": "huggingface",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

# The 2.2 GB model downloads once and is reused. Without it every invocation
# re-downloads and the bill is dominated by transfer.
hf_cache = modal.Volume.from_name("kmc-mineru-hf-cache", create_if_missing=True)
# Finished batches land here as they complete, so a failure late in a long book
# does not discard the work already done.
results = modal.Volume.from_name("kmc-parse-results", create_if_missing=True)

app = modal.App("kindle-math-converter-parse")

# Explicit CPU and memory. The earlier runs requested neither, which is the
# leading explanation for their 4–7× timing swings — see the module docstring.
# GPU: L4 is cheaper than A10G and, like it, has compute capability >= 8.0, so
# MinerU takes the same custom-logits-processor code path. A T4 (7.5) would
# silently take a different one and invalidate any comparison.
GPU = "L4"
# 8 cores / 16 GiB. Measured against 4 / 8, which came out ~7% cheaper per page
# on the mean but with a 1.7x spread between two identical parses, against 1.05x
# here. On a 1000-page book that trade is roughly $0.10 saved in exchange for
# throughput you cannot predict — and unpredictable throughput is what started
# this whole investigation, since it is also what makes a long run's cost and
# duration unquotable. Reliability is worth more than the dime.
#
# Note the comparison was flawed (CPU and memory moved together, n=2, different
# host classes), so this is the conservative choice rather than a proven
# optimum. The README records how to settle it properly if it ever matters.
CPU = 8.0
MEMORY_MB = 16384

# A wall-clock cap is the wrong instrument for detecting a stuck parse: set it
# short and a legitimately long book is killed, set it long and a hang burns the
# whole budget before anyone notices. So the timeout here is only a last-resort
# backstop, sized for the longest book we would ever submit, and the real
# control is the stall watchdog below.
TIMEOUT_S = 6 * 60 * 60

# MinerU emits tqdm progress continuously while it works. Silence therefore
# means stuck, not slow — a slow page still prints. If nothing is written for
# this long, the batch is killed and the run moves on rather than paying for a
# wedged container. Generous enough to cover vLLM engine init (52–128 s
# measured) plus a hard page.
STALL_TIMEOUT_S = 420

# Modal keeps a container alive after its last input and bills for that idle
# time; the default is 60 s. `modal run` tears the ephemeral app down at exit so
# it does not apply there, but this makes the intent explicit and matters if the
# function is ever deployed rather than run.
SCALEDOWN_WINDOW_S = 2

# Deliberately NOT pinning region or cloud. Modal charges **1.5–1.75x base
# prices** for region selection, which would take this config from $1.30/h to
# $1.95–2.28/h.
#
# It is also not needed. `cpu=` and `memory=` are reservations, so our slice is
# the same whatever host we land on, and `gpu="L4"` pins the accelerator. What
# the host still affects is second-order: which CPU generation those 8 cores
# belong to, and contention with neighbours for memory bandwidth, the PCIe path
# to the GPU, and the network path to the Volume. Paying a 50–75% surcharge to
# narrow that is a bad trade — and the spread it would address was never even
# isolated (see the README's note on that flawed experiment).

# tqdm writes "<done>/<total>" — e.g. "Predict:  48%|####  | 94/197 [01:13<...]".
# That counter is the only trustworthy evidence that work is actually moving;
# the surrounding text redraws whether or not it is.
_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

# ---------------------------------------------------------------------------
# Output layout
#
# s03 reuses a parse it finds on disk instead of running MinerU, which is the
# whole reason remote parsing is useful. But it only finds one laid out its way:
#
#     whole document   <out>/mineru/<stem>/vlm/<stem>_middle.json
#     one batch        <out>/mineru/batch_{start:05d}_{end:05d}/<stem>/vlm/…
#
# where <out> is the pipeline's --output-dir. Writing anything else means
# renaming directories by hand between the download and the local run, and a
# slip there does not fail loudly — it attaches one batch's equations to
# another batch's pages, because s03 re-bases page_idx by the offset the
# directory name encodes. So this module writes that layout directly.
#
# DUPLICATED ON PURPOSE. modal_app.py imports nothing from
# kindle_math_converter so that deleting this file is a complete rollback, and
# that isolation is worth more than sharing two small pieces of logic. What is
# duplicated, and what must stay in sync with
# kindle_math_converter/stages/s03_mineru_parse.py:
#
#   1. the directory name format `batch_{start:05d}_{end:05d}`
#   2. the range algebra in `_page_batches`, INCLUDING its short-circuit when
#      total_pages <= batch_size (one range covering the document, not a
#      truncated first batch)
#
# Item 2 is the easy one to miss: the two implementations agree today for every
# case, and a plausible-looking edit to either could silently break that.
LAYOUT_S03 = "s03"
LAYOUT_LABELLED = "labelled"


def _page_batches(total_pages: int, batch_size: int) -> list[tuple[int, int]]:
    """Mirror of s03's `_page_batches`. See the note above before editing."""
    if batch_size <= 0 or total_pages <= batch_size:
        return [(0, max(0, total_pages - 1))]
    return [
        (start, min(start + batch_size - 1, total_pages - 1))
        for start in range(0, total_pages, batch_size)
    ]


def _batch_dir_name(start: int, end: int) -> str:
    """Mirror of s03's batch directory name. See the note above before editing."""
    return f"batch_{start:05d}_{end:05d}"


def _telemetry() -> dict:
    """What we actually got, not what the host happens to have.

    The first version of this logged `os.cpu_count()` and `/proc/meminfo`, which
    report the *physical host* — 24 cores and 381 GB on one run, 20 and 190 on
    the next — while saying nothing about this container's slice. That made the
    timing differences between runs unattributable, which is the exact failure
    it existed to prevent. It now records the allocation itself, and the host CPU
    model, since a core is not a fixed unit of speed across CPU generations.
    """
    import os
    import subprocess

    out: dict[str, object] = {}

    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        out["gpu"] = gpu.stdout.strip()
    except Exception as exc:
        out["gpu"] = f"unavailable: {exc}"

    # OUR allocation. sched_getaffinity is usually the honest answer inside a
    # container; cgroup quota is the other, and they disagree often enough that
    # both are worth having. Sandboxed runtimes may expose neither, so this
    # records which method answered rather than silently reporting "unknown".
    try:
        out["affinity_cpus"] = len(os.sched_getaffinity(0))
    except Exception:
        out["affinity_cpus"] = None
    quota = None
    for path, parse in (
        ("/sys/fs/cgroup/cpu.max", lambda t: None if t.split()[0] == "max"
         else float(t.split()[0]) / float(t.split()[1])),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", None),  # v1, needs the period too
    ):
        try:
            text = open(path).read().strip()
            if parse:
                quota = parse(text)
            else:
                period = float(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
                q = float(text)
                quota = None if q < 0 else q / period
            if quota is not None:
                break
        except Exception:
            continue
    out["cgroup_cpu_limit"] = quota
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = open(path).read().strip()
            out["cgroup_mem_limit_gib"] = (
                None if text == "max" else round(int(text) / 1024**3, 1)
            )
            break
        except Exception:
            continue

    # Host context. Not our slice, but a core's speed depends on which CPU it is,
    # so this is what makes "same core count, different throughput" explicable.
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                out["host_cpu_model"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    out["host_cpu_count"] = os.cpu_count()
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                out["host_mem"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return out


@app.function(
    image=image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MB,
    timeout=TIMEOUT_S,
    scaledown_window=SCALEDOWN_WINDOW_S,
    volumes={"/root/.cache/huggingface": hf_cache, "/results": results},
)
def parse_jobs(jobs: list[dict], stall_timeout_s: int = STALL_TIMEOUT_S) -> dict:
    """Runs every job in ONE container, sequentially.

    One container is the point: vLLM engine init costs 177–688 s and is paid per
    container, not per document. Three separate invocations would pay it three
    times. It also makes the comparison sharper — jobs in one container share
    hardware, so a timing difference between them cannot be blamed on scheduling.

    Each job is {name, pdf: bytes, start, end, label}. Results are committed to
    the results Volume as each finishes.
    """
    import io
    import shutil
    import subprocess
    import tarfile
    import threading
    import time
    from pathlib import Path

    telemetry = _telemetry()
    print(f"[kmc] hardware: {telemetry}", flush=True)

    work = Path("/tmp/kmc")
    out: list[dict] = []

    for i, job in enumerate(jobs, 1):
        label = job.get("label") or f"job{i}"
        src_dir = work / label / "src"
        out_dir = work / label / "out"
        shutil.rmtree(work / label, ignore_errors=True)
        src_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = src_dir / job["name"]
        pdf_path.write_bytes(job["pdf"])

        cmd = ["mineru", "-p", str(pdf_path), "-o", str(out_dir), "-b", "vlm-engine"]
        if job.get("start") is not None and job.get("end") is not None:
            cmd += ["-s", str(job["start"]), "-e", str(job["end"])]

        print(f"\n[kmc] === {label} ({i}/{len(jobs)}) ===", flush=True)
        print(f"[kmc] {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()

        # Stream rather than capture: capture_output=True buffers until exit,
        # which here means minutes of silence indistinguishable from a hang.
        # A watchdog thread reads the clock while the main thread reads output —
        # if output stops, the parse is stuck and the container is burning money
        # for nothing, so kill it instead of waiting out the function timeout.
        lines: list[str] = []
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None

        # Watching for *output* is not enough. A process can keep printing while
        # the work behind it is wedged — a redrawing progress bar or a heartbeat
        # log looks identical to progress from the outside. So the watchdog
        # tracks the progress COUNTER (tqdm's "94/197") and requires it to
        # advance. Chatty-but-frozen is the failure this catches; silence is
        # only the easy case.
        #
        # Before any counter exists (vLLM init prints plenty and counts
        # nothing), it falls back to time-since-output, which is the right
        # measure for that phase.
        lines: list[str] = []
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None

        now = time.monotonic
        last_output = [now()]
        last_advance = [now()]
        progress: list[tuple[int, int] | None] = [None]  # (done, total), None until seen
        stalled = [""]

        def watchdog() -> None:
            beat = now()
            while proc.poll() is None:
                t = now()
                tracking = progress[0] is not None
                # Once a counter exists, frozen progress is the signal. Until
                # then, silence is.
                quiet = t - (last_advance[0] if tracking else last_output[0])

                if t - beat >= 60:  # visibility while it is healthy
                    beat = t
                    where = f"progress {progress[0][0]}/{progress[0][1]}" if tracking else "starting up"
                    print(
                        f"[kmc] {label} alive — {where}, "
                        f"{quiet:.0f}s since last advance (limit {stall_timeout_s}s)",
                        flush=True,
                    )

                if quiet > stall_timeout_s:
                    stalled[0] = (
                        f"progress frozen at {progress[0][0]}/{progress[0][1]}"
                        if tracking else "no output"
                    )
                    print(
                        f"[kmc] {label} STALLED — {stalled[0]} for {quiet:.0f}s "
                        f"(limit {stall_timeout_s}s); killing",
                        flush=True,
                    )
                    proc.kill()
                    return
                time.sleep(5)

        watch = threading.Thread(target=watchdog, daemon=True)
        watch.start()

        for line in proc.stdout:
            last_output[0] = now()
            lines.append(line)
            # tqdm renders "<done>/<total>"; a change in either element counts
            # as advancing, so moving between phases (layout -> content) is not
            # mistaken for a stall.
            m = _PROGRESS_RE.search(line)
            if m:
                state = (int(m.group(1)), int(m.group(2)))
                if state != progress[0]:
                    progress[0] = state
                    last_advance[0] = now()
            print(f"[mineru:{label}] {line.rstrip()[:150]}", flush=True)
        rc = proc.wait()
        watch.join(timeout=10)
        elapsed = time.perf_counter() - t0
        log = "".join(lines)
        if stalled[0]:
            rc = rc or -1
            log += f"\n[kmc] killed: {stalled[0]} for >{stall_timeout_s}s\n"
        print(f"[kmc] {label} exited {rc} after {elapsed:.0f}s", flush=True)

        record: dict = {
            "label": label,
            "name": job["name"],
            "start": job.get("start"),
            "end": job.get("end"),
            "returncode": rc,
            "stalled": stalled[0],
            "elapsed_s": round(elapsed, 1),
            "log_tail": log[-3000:],
            "tar": None,
        }

        if rc == 0:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for p in sorted(out_dir.rglob("*")):
                    if p.is_file():
                        tar.add(p, arcname=str(p.relative_to(out_dir)))
            record["tar"] = buf.getvalue()
            # Persist before moving on. This is what makes a long book
            # survivable: a failure on batch 9 keeps batches 1–8.
            keep = Path("/results") / label
            shutil.rmtree(keep, ignore_errors=True)
            keep.mkdir(parents=True, exist_ok=True)
            (keep / "output.tar.gz").write_bytes(record["tar"])
            results.commit()
            print(f"[kmc] {label} committed to results volume", flush=True)
        else:
            print(f"[kmc] {label} FAILED — continuing with remaining jobs", flush=True)

        out.append(record)

    hf_cache.commit()
    return {"telemetry": telemetry, "jobs": out}


@app.local_entrypoint()
def main(
    pdf: str,
    output_root: str = "./output/REMOTE",
    repeat: int = 1,
    batch_pages: int = 0,
    pages: str = "",
    layout: str = LAYOUT_S03,
):
    """`--pdf` takes one path or several comma-separated.

    `--repeat N` parses each document N times (determinism check).
    `--batch-pages N` splits each document into N-page ranges (exercises the
    resumable path that protects a long book).
    `--layout` picks the on-disk shape of the result:

      s03 (default)  ready for the pipeline to reuse — `--output-root` IS the
                     book's `--output-dir`, and a local run picks the parse up
                     with no hand-moving. One book per invocation.
      labelled       one directory per job, named after the job. Needed for
                     `--repeat` and `--pages`, which have no place in the s03
                     layout because their outputs collide or are never looked
                     for. Results must be moved into place by hand.
    """
    import io
    import json
    import tarfile
    import time
    from pathlib import Path

    if layout not in (LAYOUT_S03, LAYOUT_LABELLED):
        raise SystemExit(f"--layout must be {LAYOUT_S03} or {LAYOUT_LABELLED}")

    paths = [Path(p.strip()) for p in pdf.split(",") if p.strip()]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"not a file: {p}")

    # The s03 layout keys directories on page range alone, so anything that
    # produces two jobs with the same range writes them to the same place and
    # the second silently overwrites the first. The label layout exists for
    # exactly these cases; refuse rather than quietly lose a parse.
    #
    # --repeat is the dangerous one. Its only purpose is feeding
    # `compare-snapshots` a second parse of the same book to check that the GPU
    # gave consistent answers. Overwritten, that check compares a file with
    # itself and reports "identical" — a test built to catch parser
    # non-determinism would pass every time while verifying nothing.
    if layout == LAYOUT_S03:
        why = None
        if len(paths) > 1:
            why = (f"{len(paths)} PDFs share one --output-root; s03 layout holds "
                   f"one book (its work_dir is <output-dir>/mineru)")
        elif repeat > 1:
            why = ("--repeat writes every repetition to the same page-range "
                   "directory, so only the last survives and a determinism "
                   "check would compare a file with itself")
        elif pages:
            why = ("--pages writes an ad-hoc range s03 never looks for; it "
                   "derives batch names from the whole document")
        if why:
            raise SystemExit(
                f"refusing to run: {why}.\n"
                f"Use --layout {LAYOUT_LABELLED} for this, or run one book per "
                f"invocation with its own --output-root."
            )

    def page_count(path: Path) -> int:
        try:
            import pymupdf
            return pymupdf.open(str(path)).page_count
        except Exception:
            return 0

    jobs: list[dict] = []
    for path in paths:
        data = path.read_bytes()
        ranges: list[tuple[int | None, int | None]] = [(None, None)]
        if pages:
            s, e = pages.split("-", 1)
            ranges = [(int(s), int(e))]
        elif batch_pages > 0:
            n = page_count(path)
            if n:
                # Built from the shared mirror of s03's own batching, so the
                # directory names agree by construction rather than by
                # coincidence.
                ranges = list(_page_batches(n, batch_pages))
        for rep in range(1, repeat + 1):
            for bi, (s, e) in enumerate(ranges, 1):
                # Index-prefixed so the same PDF can be listed twice (the
                # determinism check does exactly that) without labels colliding
                # and one job's output overwriting another's.
                parts = [f"{len(jobs) + 1:02d}", path.stem[:22].replace(" ", "_")]
                if repeat > 1:
                    parts.append(f"rep{rep}")
                if len(ranges) > 1:
                    parts.append(f"b{bi}")
                jobs.append({
                    "label": "__".join(parts),
                    "name": path.name,
                    "pdf": data,
                    "start": s,
                    "end": e,
                })

    print(f"submitting {len(jobs)} job(s) in ONE container "
          f"(gpu={GPU}, cpu={CPU}, mem={MEMORY_MB}MB, timeout={TIMEOUT_S}s)")
    for j in jobs:
        rng = "whole" if j["start"] is None else f"pages {j['start']}-{j['end']}"
        print(f"   {j['label']:38s} {rng}")

    t0 = time.perf_counter()
    res = parse_jobs.remote(jobs)
    wall = time.perf_counter() - t0

    print(f"\nhardware: {json.dumps(res['telemetry'], indent=None)}")
    print(f"round trip: {wall:.0f}s\n")
    print(f"{'job':40s} {'rc':>3s} {'secs':>7s}  output")

    root = Path(output_root)
    work_dir = root / "mineru"
    written: list[Path] = []
    for rec in res["jobs"]:
        if layout == LAYOUT_LABELLED:
            dest = root / rec["label"] / "mineru"
        elif rec["start"] is None:
            # Whole-document parse. The tar already carries <stem>/vlm/…, which
            # is what s03 globs for directly under work_dir.
            dest = work_dir
        else:
            dest = work_dir / _batch_dir_name(rec["start"], rec["end"])

        note = "-"
        if rec["tar"]:
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(rec["tar"]), mode="r:gz") as tar:
                tar.extractall(dest, filter="data")
            found = sorted(dest.rglob("*_middle.json"))
            note = f"{len(found)} middle.json -> {dest}"
            written.append(dest)
        else:
            note = "FAILED (see log below)"
        print(f"{rec['label']:40s} {rec['returncode']:>3} {rec['elapsed_s']:>7.0f}  {note}")

    work_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (work_dir if layout == LAYOUT_S03 else root) / "remote_run_summary.json"
    summary = {
        "layout": layout,
        "batch_pages": batch_pages,
        "telemetry": res["telemetry"],
        "wall_s": round(wall, 1),
        "jobs": [{k: v for k, v in r.items() if k != "tar"} for r in res["jobs"]],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary + logs: {summary_path}")

    if layout == LAYOUT_S03:
        _verify_s03_layout(written, work_dir, batch_pages, paths[0], output_root)

    failed = [r for r in res["jobs"] if r["returncode"] != 0]
    if failed:
        print(f"\n{len(failed)} job(s) failed:")
        for r in failed:
            print(f"--- {r['label']} ---\n{r['log_tail'][-1200:]}")


def _verify_s03_layout(written, work_dir, batch_pages, pdf_path, output_root) -> None:
    """Checks the directories we just wrote are ones s03 will actually find.

    The failure this exists for is silent and expensive. If the local run is
    given a different --parse-batch-size than the remote run used, s03 looks
    for batch directories that do not exist, finds nothing to reuse, and parses
    the book again locally at ~45 s/page — six hours on a 480-page book, with
    no error, presenting only as slowness. Printing the right command helps
    only if it is copied correctly every time, so check it here instead: this
    process already knows the page count and the batch size.
    """
    from pathlib import Path

    ok = True
    missing: list[str] = []
    expected: set[str] = set()

    # s03 finds a parse with glob("*/*/*_middle.json") relative to the batch
    # directory (or work_dir, whole-document). Anything shallower or deeper is
    # invisible to it, so verify at that exact depth rather than recursively —
    # a recursive search would pass on a layout s03 cannot read.
    for dest in written:
        if not list(Path(dest).glob("*/*/*_middle.json")):
            print(f"\n  PROBLEM: {dest} has no <stem>/vlm/*_middle.json — "
                  f"s03 will not see this parse.")
            ok = False

    try:
        import pymupdf
        n = pymupdf.open(str(pdf_path)).page_count
    except Exception:
        n = 0

    if n and batch_pages > 0:
        expected = {_batch_dir_name(s, e) for s, e in _page_batches(n, batch_pages)}
        actual = {p.name for p in work_dir.iterdir()
                  if p.is_dir() and p.name.startswith("batch_")}
        # Unexpected names mean the two batching implementations have drifted;
        # that is a bug and every page offset downstream is suspect.
        for name in sorted(actual - expected):
            print(f"\n  PROBLEM: {name} is not a batch s03 would ask for at "
                  f"--parse-batch-size {batch_pages}. The batching in this file "
                  f"and in s03_mineru_parse.py have diverged.")
            ok = False
        # Missing ones are not an error — that is resumability working. s03
        # parses just those locally, which is slow but correct, so say so
        # plainly rather than implying the run is broken.
        missing = sorted(expected - actual)

    if ok and not missing:
        print(f"\n  layout verified — s03 will reuse this parse.")
    elif ok:
        # Deliberately not "verified": what was written is findable, but the
        # book is only partly parsed, and saying otherwise invites a local run
        # that quietly spends hours on the gap.
        print(f"\n  layout OK, but {len(missing)} of {len(expected)} batches are "
              f"missing ({', '.join(missing[:3])}{'…' if len(missing) > 3 else ''}).")
        print(f"  The local run will parse those pages itself (~45 s/page). "
              f"Re-run this command first to fill them in remotely instead.")

    flag = f" --parse-batch-size {batch_pages}" if batch_pages > 0 else ""
    print(f"\nnext, locally:\n"
          f"  python main.py convert \"{pdf_path}\" "
          f"--output-dir \"{output_root}\"{flag}")
    if batch_pages > 0:
        print(f"\n  --parse-batch-size MUST be {batch_pages} — it is what the batch "
              f"directory names above encode.")

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
    a normal local run picks them up and skips MinerU.

ROLLBACK
    Nothing imports this; it imports nothing from kindle_math_converter.
    Delete the file (and `pip uninstall modal`) and the project is unchanged.

USAGE
    modal run modal_app.py --pdf "/path/a.pdf"
    modal run modal_app.py --pdf "/path/a.pdf,/path/b.pdf" --output-root ./output/REMOTE
    modal run modal_app.py --pdf "/path/a.pdf" --repeat 2      # determinism check
    modal run modal_app.py --pdf "/path/a.pdf" --batch-pages 5 # resumable batches
"""
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


def _telemetry() -> dict:
    """What hardware we actually landed on. The first two runs differed 4–7× in
    speed with no record of what they ran on, which made the cause unfalsifiable.
    Never again."""
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
    out["os_cpu_count"] = os.cpu_count()
    try:
        # cgroup v2 quota is what the container may actually use, which is not
        # the same as the host core count os.cpu_count() reports.
        quota = open("/sys/fs/cgroup/cpu.max").read().split()
        out["cgroup_cpu_max"] = f"{quota[0]}/{quota[1]}"
    except Exception:
        out["cgroup_cpu_max"] = "unknown"
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                out["mem_total"] = line.split(":", 1)[1].strip()
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

        last_output = [time.monotonic()]
        stalled = [False]

        def watchdog() -> None:
            while proc.poll() is None:
                quiet = time.monotonic() - last_output[0]
                if quiet > stall_timeout_s:
                    stalled[0] = True
                    print(
                        f"[kmc] {label} STALLED — no output for {quiet:.0f}s "
                        f"(limit {stall_timeout_s}s); killing",
                        flush=True,
                    )
                    proc.kill()
                    return
                time.sleep(5)

        watch = threading.Thread(target=watchdog, daemon=True)
        watch.start()

        for line in proc.stdout:
            last_output[0] = time.monotonic()
            lines.append(line)
            print(f"[mineru:{label}] {line.rstrip()[:150]}", flush=True)
        rc = proc.wait()
        watch.join(timeout=10)
        elapsed = time.perf_counter() - t0
        log = "".join(lines)
        if stalled[0]:
            rc = rc or -1
            log += f"\n[kmc] killed after {stall_timeout_s}s without output\n"
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
):
    """`--pdf` takes one path or several comma-separated.

    `--repeat N` parses each document N times (determinism check).
    `--batch-pages N` splits each document into N-page ranges (exercises the
    resumable path that protects a long book).
    """
    import io
    import json
    import tarfile
    import time
    from pathlib import Path

    paths = [Path(p.strip()) for p in pdf.split(",") if p.strip()]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"not a file: {p}")

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
                ranges = [(s, min(s + batch_pages - 1, n - 1))
                          for s in range(0, n, batch_pages)]
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
    for rec in res["jobs"]:
        dest = root / rec["label"] / "mineru"
        note = "-"
        if rec["tar"]:
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(rec["tar"]), mode="r:gz") as tar:
                tar.extractall(dest, filter="data")
            found = sorted(dest.rglob("*_middle.json"))
            note = f"{len(found)} middle.json -> {dest}"
        else:
            note = "FAILED (see log below)"
        print(f"{rec['label']:40s} {rec['returncode']:>3} {rec['elapsed_s']:>7.0f}  {note}")

    (root).mkdir(parents=True, exist_ok=True)
    summary = {
        "telemetry": res["telemetry"],
        "wall_s": round(wall, 1),
        "jobs": [{k: v for k, v in r.items() if k != "tar"} for r in res["jobs"]],
    }
    (root / "remote_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary + logs: {root / 'remote_run_summary.json'}")

    failed = [r for r in res["jobs"] if r["returncode"] != 0]
    if failed:
        print(f"\n{len(failed)} job(s) failed:")
        for r in failed:
            print(f"--- {r['label']} ---\n{r['log_tail'][-1200:]}")

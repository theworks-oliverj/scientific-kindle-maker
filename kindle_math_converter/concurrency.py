"""
Shared concurrency helper for per-equation pipeline work (Phase 2).

s06_validation and s08a_svg_render each do independent, self-contained work
per equation region (own temp dir, own tectonic/dvisvgm subprocess calls,
which release the GIL while running) — embarrassingly parallel across CPU
cores via a thread pool.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 4)


def parallel_for_each(items: list[T], fn: Callable[[T], R], max_workers: Optional[int] = None) -> list[R]:
    """
    Applies `fn` to each item, in parallel, returning results in input order.
    """
    if not items:
        return []
    workers = max_workers or DEFAULT_MAX_WORKERS
    workers = max(1, min(workers, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fn, items))

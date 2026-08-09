import hashlib
import re
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CachedRender:
    svg: str
    cdm_score: float
    latex: str   # Original (pre-normalization) — kept for debugging


def _normalize_latex(latex: str) -> str:
    """
    Cosmetic normalization only.

    INVARIANT: normalize(a) == normalize(b) IFF render(a) and render(b)
    produce pixel-identical output.

    NEVER normalize:
      - Index letters (mu, nu, rho, sigma and all others)
      - Sub/superscript position (T_ab != T^ab)
      - Font commands (\\mathbf, \\mathit, \\boldsymbol)
      - Spacing commands that affect alignment (\\quad, \\qquad, \\,, \\!)

    SAFE to normalize (TeX ignores these):
      - Whitespace around ^ and _
      - Multiple consecutive spaces
      - Trailing/leading whitespace
      - TeX comments (% to end of line)
    """
    latex = re.sub(r'%[^\n]*', '', latex)
    latex = re.sub(r'\s+', ' ', latex).strip()
    latex = re.sub(r'\s*\^\s*', '^', latex)
    latex = re.sub(r'\s*_\s*', '_', latex)
    return latex


def _cache_key(latex: str) -> str:
    """
    SHA-256 of normalized LaTeX.
    Different indices produce different keys by construction.
    T^{\\mu\\nu} and T^{\\mu\\rho} are different keys.
    """
    return hashlib.sha256(_normalize_latex(latex).encode('utf-8')).hexdigest()


class PersistentCompileCache:
    """
    On-disk cache of tectonic compiles, keyed by normalized LaTeX.

    Why this and not just the session render cache: once s08A reuses s06's
    XDV, the surviving per-equation cost is s06's tectonic invocation. That
    cost is paid again on every re-run, and a 500-page book will not come out
    right on the first attempt — the second run of ~2000 equations should not
    recompile all of them.

    One file per key rather than a single index: lookups are a stat(), writes
    are independent, and the thread pool in s06 needs no locking because two
    workers writing the same key write identical bytes.

    Only successful compiles are stored. A failure is cheap to rediscover and
    caching one risks pinning a failure caused by a transient condition (a
    timeout under load, a missing font on first use).
    """

    def __init__(self, cache_dir: "Path | None" = None):
        self.dir = cache_dir
        self._hits = 0
        self._misses = 0
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, latex: str, formula_class: str) -> "Path | None":
        if self.dir is None:
            return None
        key = hashlib.sha256(
            f"{formula_class}\x00{_normalize_latex(latex)}".encode("utf-8")
        ).hexdigest()
        return self.dir / f"{key}.xdv"

    def get(self, latex: str, formula_class: str) -> bytes | None:
        path = self._path(latex, formula_class)
        if path is None:
            return None
        try:
            data = path.read_bytes()
        except OSError:
            self._misses += 1
            return None
        self._hits += 1
        return data

    def put(self, latex: str, formula_class: str, xdv: bytes) -> None:
        path = self._path(latex, formula_class)
        if path is None or not xdv:
            return
        try:
            # Write-then-rename so a crash mid-write cannot leave a truncated
            # XDV that a later run would happily render.
            tmp = path.with_suffix(".xdv.tmp")
            tmp.write_bytes(xdv)
            tmp.replace(path)
        except OSError:
            pass

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "compile_cache_hits": self._hits,
            "compile_cache_misses": self._misses,
            "compile_cache_hit_rate": round(self._hits / total, 4) if total else 0.0,
        }


class SessionEquationCache:
    """
    Per-document in-memory render cache.
    Caches only the tectonic + dvisvgm render step.
    UniMERNet recognition and CDM validation always run — never cached.

    Scaling note (Option B — batch/service mode):
      Replace self._store with a Redis hash scoped to job_id.
      Key format: "job:{job_id}:eq:{sha256}"
      TTL: job duration + 10 min buffer (auto-expire).
      Add SET NX to prevent thundering herd when multiple workers
      process the same equation concurrently.
    """

    def __init__(self):
        self._store: dict[str, CachedRender] = {}
        self._hits: int = 0
        self._misses: int = 0
        # Phase 2: s08a_svg_render processes equations in a thread pool, and
        # multiple workers may hit get()/put() concurrently for distinct
        # equations that normalize to the same LaTeX.
        self._lock = threading.Lock()

    def get(self, latex: str) -> CachedRender | None:
        key = _cache_key(latex)
        with self._lock:
            result = self._store.get(key)
            if result is not None:
                self._hits += 1
            else:
                self._misses += 1
            return result

    def put(self, latex: str, svg: str, cdm_score: float) -> None:
        """
        Only call after successful CDM validation.
        Do not cache renders that failed validation —
        a bad render cached is worse than a cache miss.
        """
        key = _cache_key(latex)
        with self._lock:
            self._store[key] = CachedRender(svg=svg, cdm_score=cdm_score, latex=latex)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def dump(self) -> list[dict]:
        """
        Debug utility. Returns all cached entries as dicts for inspection.
        Call at end of document processing to audit cache contents.
        """
        return [
            {
                "key": _cache_key(v.latex),
                "latex": v.latex,
                "cdm_score": v.cdm_score,
                "svg_length": len(v.svg),
            }
            for v in self._store.values()
        ]

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "entries": len(self._store),
        }

"""
Recognition snapshots — the regression oracle for parser experiments.

The funnel counts (277 regions, 29 to tectonic, 0 flagged) are deterministic
and easy to check, but funnel-stable does NOT imply LaTeX-identical: a backend
change can recognise the same number of equations and get their *contents*
wrong. Nothing downstream would notice — the equation gate is compile plus
plausibility (accepted gap #4), so a wrong-but-compilable equation passes
silently.

So the real check is an exact string comparison of every recognised equation
against a known-good run. This module writes one snapshot per pipeline run and
diffs two of them.

Snapshots are keyed by page and reading order rather than by region_id, so a
change that shifts numbering (an extra region on page 3 renuming everything
after it) shows up as the local insertion it is instead of a total mismatch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models.document import Document

SNAPSHOT_SUFFIX = "_latex_snapshot.json"


def build_snapshot(document: Document | None) -> dict[str, Any]:
    """Every recognised equation, in document order, with the fields a
    recognition change would move."""
    equations: list[dict[str, Any]] = []
    if document is not None:
        for page in document.pages:
            for eq in sorted(
                page.equation_regions,
                key=lambda e: (e.reading_order_index or 0.0, e.region_id),
            ):
                equations.append({
                    "page": eq.bbox.page_number,
                    "order": eq.reading_order_index,
                    "region_id": eq.region_id,
                    "class": eq.formula_class.value,
                    "raw_latex": eq.raw_latex,
                    "render_as_text": eq.render_as_text,
                    "inline_text_repr": eq.inline_text_repr,
                    "equation_number": eq.equation_number,
                    "deduped": bool(eq.dedup_canonical_id),
                })
    return {
        "version": 1,
        "counts": {
            "equations": len(equations),
            "display": sum(1 for e in equations if e["class"] == "display"),
            "inline": sum(1 for e in equations if e["class"] == "inline"),
            "render_as_text": sum(1 for e in equations if e["render_as_text"]),
            "deduped": sum(1 for e in equations if e["deduped"]),
            "pages": len(document.pages) if document else 0,
        },
        "equations": equations,
    }


def write_snapshot(document: Document | None, out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}{SNAPSHOT_SUFFIX}"
    path.write_text(
        json.dumps(build_snapshot(document), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return path


def _key(eq: dict) -> tuple:
    return (eq.get("page"), eq.get("order"), eq.get("class"))


def compare(baseline: dict, candidate: dict) -> dict[str, Any]:
    """Diffs two snapshots. `changed` is the one that matters — same slot,
    different LaTeX — because that is the failure a funnel check misses."""
    base_eqs = {_key(e): e for e in baseline.get("equations", [])}
    cand_eqs = {_key(e): e for e in candidate.get("equations", [])}

    changed, added, removed = [], [], []
    for key, b in base_eqs.items():
        c = cand_eqs.get(key)
        if c is None:
            removed.append(b)
        elif (b.get("raw_latex") != c.get("raw_latex")
              or b.get("render_as_text") != c.get("render_as_text")
              or b.get("inline_text_repr") != c.get("inline_text_repr")):
            changed.append({
                "page": b.get("page"),
                "region_id": b.get("region_id"),
                "baseline": b.get("raw_latex"),
                "candidate": c.get("raw_latex"),
                "baseline_text_repr": b.get("inline_text_repr"),
                "candidate_text_repr": c.get("inline_text_repr"),
            })
    for key, c in cand_eqs.items():
        if key not in base_eqs:
            added.append(c)

    total = max(1, len(base_eqs))
    return {
        "identical": not (changed or added or removed),
        "baseline_counts": baseline.get("counts", {}),
        "candidate_counts": candidate.get("counts", {}),
        "n_changed": len(changed),
        "n_added": len(added),
        "n_removed": len(removed),
        "pct_changed": round(100.0 * len(changed) / total, 2),
        "changed": changed,
        "added": added,
        "removed": removed,
    }


def format_report(report: dict[str, Any], max_examples: int = 15) -> str:
    lines: list[str] = []
    bc, cc = report["baseline_counts"], report["candidate_counts"]
    lines.append("counts      baseline -> candidate")
    for k in sorted(set(bc) | set(cc)):
        b, c = bc.get(k), cc.get(k)
        flag = "" if b == c else "   <-- CHANGED"
        lines.append(f"  {k:16s} {str(b):>6s} -> {str(c):>6s}{flag}")

    if report["identical"]:
        lines.append("\nLaTeX: IDENTICAL across every equation — safe for this corpus.")
        return "\n".join(lines)

    lines.append(
        f"\nLaTeX: {report['n_changed']} changed ({report['pct_changed']}%), "
        f"{report['n_added']} added, {report['n_removed']} removed"
    )
    for item in report["changed"][:max_examples]:
        lines.append(f"\n  p{item['page']} {item['region_id']}")
        lines.append(f"    baseline : {item['baseline']!r}")
        lines.append(f"    candidate: {item['candidate']!r}")
    remaining = report["n_changed"] - min(report["n_changed"], max_examples)
    if remaining > 0:
        lines.append(f"\n  … and {remaining} more changed equations")
    return "\n".join(lines)

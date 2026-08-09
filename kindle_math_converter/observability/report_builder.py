"""
Generates a self-contained HTML report after pipeline completion.
"""
import base64
from pathlib import Path

from jinja2 import Environment, BaseLoader

from .event_bus import EventBus
from ..models.results import PipelineResult
from ..models.document import Document


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Conversion Report — {{ result.document_path }}</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 2rem auto;
         padding: 0 1rem; color: #222; }
  h1 { font-size: 1.4rem; font-weight: 500; }
  h2 { font-size: 1.1rem; font-weight: 500; margin-top: 2rem; border-bottom: 1px solid #ddd; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { background: #f5f5f5; text-align: left; padding: 6px 8px; border: 1px solid #ddd; }
  td { padding: 5px 8px; border: 1px solid #eee; vertical-align: top; }
  tr.flagged td { background: #fff3f3; }
  tr.fallback td { background: #fff8ee; }
  tr.repaired td { background: #fffde7; }
  .badge { display: inline-block; padding: 2px 6px; border-radius: 3px;
           font-size: 0.75rem; font-weight: 500; }
  .badge-pass     { background: #e6f4ea; color: #1e6b3c; }
  .badge-repair   { background: #fff8e1; color: #7a5000; }
  .badge-fallback { background: #fff0e0; color: #8a3a00; }
  .badge-flagged  { background: #fce8e8; color: #9b1c1c; }
  .eq-detail { display: flex; gap: 1rem; align-items: flex-start; }
  .eq-crop img { max-width: 300px; border: 1px solid #ddd; }
  .eq-latex { font-family: monospace; font-size: 0.8rem; background: #f8f8f8;
              padding: 8px; border-radius: 4px; white-space: pre-wrap; }
  .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
  .summary-card { background: #f9f9f9; border: 1px solid #e0e0e0;
                  border-radius: 6px; padding: 1rem; }
  .summary-card .number { font-size: 1.8rem; font-weight: 500; }
  .summary-card .label  { font-size: 0.8rem; color: #666; }
  details summary { cursor: pointer; font-weight: 500; }
  .error-code { font-family: monospace; font-size: 0.8rem; color: #c00; }
  input[type=text] { width: 100%; padding: 6px; margin: 6px 0;
                     border: 1px solid #ccc; border-radius: 4px; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>Conversion Report</h1>
<p style="color: #666; font-size:0.85rem">
  Source: <code>{{ result.document_path }}</code> &nbsp;|&nbsp;
  Started: {{ result.started_at }} &nbsp;|&nbsp;
  Duration: {{ total_duration_s }}s &nbsp;|&nbsp;
  Status: <strong>{{ "OK" if result.ok else "FAILED" }}</strong>
</p>
<div class="summary-grid">
  <div class="summary-card">
    <div class="number">{{ result.total_equations }}</div>
    <div class="label">Total equations</div>
  </div>
  <div class="summary-card">
    <div class="number" style="color:#1e6b3c">{{ result.equations_passed }}</div>
    <div class="label">Passed (CDM &ge; threshold)</div>
  </div>
  <div class="summary-card">
    <div class="number" style="color:#7a5000">{{ result.equations_repaired }}</div>
    <div class="label">Repaired</div>
  </div>
  <div class="summary-card">
    <div class="number" style="color:#9b1c1c">{{ result.equations_flagged }}</div>
    <div class="label">Flagged for review</div>
  </div>
</div>
<h2>Stage Summary</h2>
<table>
  <tr><th>Stage</th><th>Status</th><th>Duration</th><th>Metrics</th><th>Warnings/Errors</th></tr>
  {% for sr in result.stage_results %}
  <tr>
    <td>{{ sr.stage_name }}</td>
    <td>{{ "✓" if sr.ok else "✗" }}</td>
    <td>{{ sr.duration_ms }}ms</td>
    <td>{% for k, v in sr.metrics.items() %}{{ k }}: {{ v }}<br>{% endfor %}</td>
    <td>
      {% for w in sr.warnings %}<span style="color:#7a5000">⚠ {{ w }}</span><br>{% endfor %}
      {% for e in sr.errors %}<span class="error-code">✗ {{ e }}</span><br>{% endfor %}
    </td>
  </tr>
  {% endfor %}
</table>
<h2>Cache Statistics</h2>
<p>
  Hits: {{ result.cache_hits }} &nbsp;|&nbsp;
  Misses: {{ result.cache_misses }} &nbsp;|&nbsp;
  Hit rate: {{ "%.1f"|format(cache_hit_rate * 100) }}%
</p>
<h2>Equation Detail</h2>
<p style="font-size:0.8rem;color:#666">
  Sort by clicking column headers. Filter by typing below.
  Rows highlighted in red are flagged for review.
  {% if crops_omitted %}
  <br><strong>{{ crops_omitted }}</strong> further source crops were omitted to keep
  this report openable — inspect them in the output directory instead.
  {% endif %}
</p>
<input type="text" id="eq-filter" placeholder="Filter by equation ID, LaTeX, page...">
<table id="eq-table">
  <tr>
    <th onclick="sortTable(0)">ID</th>
    <th onclick="sortTable(1)">Page</th>
    <th onclick="sortTable(2)">Class</th>
    <th onclick="sortTable(3)">Gate</th>
    <th onclick="sortTable(4)">CDM</th>
    <th onclick="sortTable(5)">Repairs</th>
    <th>Detail</th>
  </tr>
  {% for eq in equations %}
  <tr class="{{ eq.confidence_gate.value if eq.confidence_gate else '' }}">
    <td><code>{{ eq.region_id }}</code></td>
    <td>{{ eq.bbox.page_number }}</td>
    <td>{{ eq.formula_class.value }}</td>
    <td>
      <span class="badge badge-{{ eq.confidence_gate.value if eq.confidence_gate else 'pass' }}">
        {{ eq.confidence_gate.value if eq.confidence_gate else "—" }}
      </span>
    </td>
    <td>{{ "%.3f"|format(eq.cdm_score) if eq.cdm_score else "—" }}</td>
    <td>{{ eq.repair_attempts }}</td>
    <td>
      <details>
        <summary>View</summary>
        <div class="eq-detail">
          {% if eq.source_image_crop and eq.region_id in crop_ids %}
          <div class="eq-crop">
            <strong>Source crop:</strong><br>
            <img src="data:image/png;base64,{{ eq.source_image_crop | b64 }}" alt="equation crop">
          </div>
          {% elif eq.source_image_crop %}
          <div class="eq-crop" style="font-size:0.8rem;color:#666">
            Source crop not embedded (equation passed the gate).
          </div>
          {% endif %}
          <div>
            <strong>Raw LaTeX:</strong>
            <div class="eq-latex">{{ eq.raw_latex or "—" }}</div>
            {% if eq.normalized_latex and eq.normalized_latex != eq.raw_latex %}
            <strong>Normalized:</strong>
            <div class="eq-latex">{{ eq.normalized_latex }}</div>
            {% endif %}
            {% if eq.error_codes %}
            <strong>Errors:</strong>
            {% for ec in eq.error_codes %}
            <span class="error-code">{{ ec }}</span>
            {% endfor %}
            {% endif %}
            {% if eq.failure_reason %}
            <strong>Failure detail:</strong>
            <div class="eq-latex" style="color:#9b1c1c">
              <span class="error-code">{{ eq.failure_reason.sub_code }}</span>
              &nbsp;{{ eq.failure_reason.detail }}
              <br><small style="color:#666">stage: {{ eq.failure_reason.stage }}
              {% if eq.failure_reason.recoverable %} · recoverable{% endif %}</small>
            </div>
            {% endif %}
            {% if eq.svg_postprocessed %}
            <strong>Rendered SVG preview:</strong><br>
            {{ eq.svg_postprocessed | safe }}
            {% endif %}
          </div>
        </div>
      </details>
    </td>
  </tr>
  {% endfor %}
</table>
<h2>Full Error Log</h2>
<details>
  <summary>{{ error_events | length }} error events</summary>
  <table>
    <tr><th>Time</th><th>Stage</th><th>Equation</th><th>Message</th></tr>
    {% for ev in error_events %}
    <tr>
      <td style="font-size:0.75rem;white-space:nowrap">{{ ev.timestamp }}</td>
      <td>{{ ev.stage }}</td>
      <td><code>{{ ev.equation_id or "—" }}</code></td>
      <td>{{ ev.payload }}</td>
    </tr>
    {% endfor %}
  </table>
</details>
<script>
document.getElementById('eq-filter').addEventListener('input', function() {
  const q = this.value.toLowerCase();
  document.querySelectorAll('#eq-table tr:not(:first-child)').forEach(row => {
    row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
});
function sortTable(col) {
  const table = document.getElementById('eq-table');
  const rows = Array.from(table.rows).slice(1);
  const asc = table.dataset.sortAsc !== String(col);
  table.dataset.sortAsc = asc ? col : '';
  rows.sort((a, b) => {
    const av = a.cells[col].textContent.trim();
    const bv = b.cells[col].textContent.trim();
    return asc ? av.localeCompare(bv, undefined, {numeric: true})
               : bv.localeCompare(av, undefined, {numeric: true});
  });
  rows.forEach(r => table.appendChild(r));
}
</script>
</body>
</html>
"""


# Source crops run 20–100 KB each and base64 inflates them by a third, so
# embedding one per equation makes the report scale with the book: ~2000
# equations would produce a 50–250 MB single HTML file that no browser will
# open. The report is also written from Pipeline.run()'s `finally` block, so
# that cost lands on failed runs too. Only equations someone would actually
# open the crop for are embedded.
_MAX_EMBEDDED_CROPS = 200


def _crops_to_embed(equations: list) -> tuple[set[str], int]:
    """Region ids whose source crop is worth embedding, and how many were
    dropped past the cap. A clean pass needs no visual check; anything that
    was flagged, fell back, or did not pass the gate does."""
    interesting = [
        eq for eq in equations
        if eq.source_image_crop and (
            eq.flagged_for_review
            or eq.fallback_used
            or (eq.confidence_gate is not None and eq.confidence_gate.value != "pass")
        )
    ]
    kept = interesting[:_MAX_EMBEDDED_CROPS]
    return {eq.region_id for eq in kept}, len(interesting) - len(kept)


def build_report(
    result: PipelineResult,
    document: Document | None,
    event_bus: EventBus,
    output_path: Path,
) -> None:
    def b64_filter(data: bytes | None) -> str:
        if data is None:
            return ""
        return base64.b64encode(data).decode()

    total_duration_s = (
        (result.finished_at - result.started_at).total_seconds()
        if result.finished_at else 0
    )
    total = result.cache_hits + result.cache_misses
    cache_hit_rate = result.cache_hits / total if total > 0 else 0.0

    error_events = (
        event_bus.events_by_type("equation_error")
        + event_bus.events_by_type("stage_end_error")
    )

    equations = document.all_equations if document else []
    crop_ids, crops_omitted = _crops_to_embed(equations)

    env = Environment(loader=BaseLoader())
    env.filters["b64"] = b64_filter
    template = env.from_string(REPORT_TEMPLATE)
    html = template.render(
        result=result,
        document=document,
        equations=equations,
        crop_ids=crop_ids,
        crops_omitted=crops_omitted,
        event_bus=event_bus,
        error_events=error_events,
        total_duration_s=round(total_duration_s, 1),
        cache_hit_rate=cache_hit_rate,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

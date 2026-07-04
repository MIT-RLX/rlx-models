// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Self-contained HTML benchmark report (Chart.js + Plotly heatmaps / 3D).

use crate::bench::sweep::{SweepReport, SweepRow};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::Path;

const IMPL_COLORS: &[(&str, &str)] = &[
    ("rustfft", "#4e79a7"),
    ("rlx_op_fft", "#f28e2b"),
    ("rlx_op_ifft", "#ff9d4d"),
    ("butterfly_eager", "#59a14f"),
    ("butterfly_compiled", "#e15759"),
];

pub fn read_sweep_json(path: &Path) -> Result<SweepReport> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn write_sweep_html(path: &Path, report: &SweepReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, render_sweep_html(report))
        .with_context(|| format!("write {}", path.display()))
}

pub fn render_sweep_html(report: &SweepReport) -> String {
    let json = serde_json::to_string(report).unwrap_or_else(|_| "{}".to_string());
    let generated = chrono_lite_now();
    let weights = report
        .weights
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "exact twiddles".to_string());
    let devices = unique_strings(&report.rows, |r| &r.device);
    let directions = unique_strings(&report.rows, |r| &r.direction);
    let impls = impl_labels(report.with_butterfly_compiled);
    let heatmap_sections = build_plotly_sections(report);
    let chart_sections = build_chart_sections(report);
    let table_sections = build_table_sections(report);

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RLX-FFT Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.27.0/plotly.min.js"></script>
<style>
  :root {{
    --bg: #0f1419;
    --panel: #1a2332;
    --text: #e6edf3;
    --muted: #8b949e;
    --border: #30363d;
    --accent: #58a6ff;
    --fast: #238636;
    --slow: #da3633;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}
  header {{
    padding: 2rem 2.5rem 1rem;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #161b22 0%, var(--bg) 100%);
  }}
  h1 {{ margin: 0 0 0.5rem; font-size: 1.75rem; font-weight: 600; }}
  .meta {{ color: var(--muted); font-size: 0.95rem; }}
  .meta span {{ margin-right: 1.25rem; }}
  main {{ padding: 1.5rem 2.5rem 3rem; max-width: 1400px; margin: 0 auto; }}
  section {{ margin-bottom: 2.5rem; }}
  h2 {{
    font-size: 1.25rem;
    margin: 0 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}
  h3 {{ font-size: 1rem; color: var(--accent); margin: 1.5rem 0 0.75rem; }}
  .note {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 2rem;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 1.25rem;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1rem 0.5rem;
  }}
  .card h4 {{
    margin: 0 0 0.75rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--muted);
  }}
  canvas {{ max-height: 280px; }}
  .plotly {{ width: 100%; min-height: 360px; }}
  .plotly-3d {{ min-height: 480px; }}
  .tabs {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }}
  .tab {{
    background: #21262d; border: 1px solid var(--border); color: var(--muted);
    padding: 0.35rem 0.75rem; border-radius: 6px; cursor: pointer; font-size: 0.85rem;
  }}
  .tab.active {{ background: var(--accent); color: #0d1117; border-color: var(--accent); font-weight: 600; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
  }}
  th, td {{
    border: 1px solid var(--border);
    padding: 0.45rem 0.6rem;
    text-align: right;
  }}
  th {{ background: #21262d; color: var(--muted); font-weight: 500; }}
  th.left, td.left {{ text-align: left; }}
  td.best {{ outline: 2px solid var(--fast); outline-offset: -2px; font-weight: 600; }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin: 1rem 0 2rem;
    font-size: 0.85rem;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 0.4rem; }}
  .swatch {{ width: 12px; height: 12px; border-radius: 2px; }}
  footer {{
    padding: 1rem 2.5rem 2rem;
    color: var(--muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>
<header>
  <h1>RLX-FFT Benchmark Report</h1>
  <div class="meta">
    <span>Generated {generated}</span>
    <span>iters={iters}</span>
    <span>compiled={compiled}</span>
    <span>weights={weights}</span>
    <span>sweep {sweep_ms:.0} ms</span>
  </div>
</header>
<main>
  <div class="note">
    <strong>How to read this report.</strong>
    Timings are milliseconds per iteration (lower is faster).
    <code>rustfft</code> and <code>butterfly_eager</code> always run on CPU.
    <code>rlx_op_fft</code> runs on the backend named in each section (CPU or Metal/CUDA/…).
    Green cell outline = fastest implementation for that n_fft × batch cell.
    Max-error columns show precision vs rustfft (when available).
  </div>
  <div class="legend">{legend_html}</div>

  <section>
    <h2>Performance heatmaps &amp; 3D surfaces</h2>
    <p style="color:var(--muted);font-size:0.9rem;margin-top:0">
      Each panel shows ms/iter over <strong>n_fft × batch</strong>. Use tabs to switch implementation.
      3D surfaces use log-Z. Ratio heatmaps: green &lt; 1 = faster than rustfft.
    </p>
    {heatmap_sections}
  </section>

  <section>
    <h2>Line charts (fixed batch)</h2>
    {chart_sections}
  </section>

  <section>
    <h2>Full tables</h2>
    {table_sections}
  </section>

  <section>
    <h2>Precision summary</h2>
    <div id="precision-table"></div>
  </section>
</main>
<footer>rlx-fft bench-sweep · backends: {devices} · directions: {directions}</footer>
<script>
const REPORT = {json};
const IMPLS = {impls_json};
const COLORS = {colors_json};

function rowKey(r) {{
  return [r.direction, r.device, r.batch, r.n_fft, r.implementation].join('|');
}}

function lookup(rows, dir, device, batch, n, impl) {{
  const hit = rows.find(r =>
    r.direction === dir && r.device === device &&
    r.batch === batch && r.n_fft === n && r.implementation === impl
  );
  return hit ? hit.ms : null;
}}

function fmtMs(v) {{
  if (v == null || Number.isNaN(v)) return 'n/a';
  if (v < 0.01) return v.toExponential(2);
  return v.toFixed(4);
}}

function fmtErr(v) {{
  if (v == null || Number.isNaN(v)) return '—';
  return v.toExponential(2);
}}

function bestImpl(rows, dir, device, batch, n, impls) {{
  let best = null, bestMs = Infinity;
  for (const imp of impls) {{
    const ms = lookup(rows, dir, device, batch, n, imp);
    if (ms != null && ms < bestMs) {{ bestMs = ms; best = imp; }}
  }}
  return best;
}}

function heatColor(ms, minMs, maxMs) {{
  if (ms == null || Number.isNaN(ms)) return 'transparent';
  const t = maxMs > minMs ? (ms - minMs) / (maxMs - minMs) : 0;
  const r = Math.round(35 + t * 120);
  const g = Math.round(134 - t * 90);
  const b = Math.round(54 - t * 20);
  return `rgba(${{r}},${{g}},${{b}},0.25)`;
}}

function buildPrecisionTable() {{
  const rows = REPORT.rows.filter(r => r.implementation !== 'rustfft' && r.max_err != null && !Number.isNaN(r.max_err));
  const byImpl = {{}};
  for (const r of rows) {{
    if (!byImpl[r.implementation]) byImpl[r.implementation] = [];
    byImpl[r.implementation].push(r.max_err);
  }}
  let html = '<table><thead><tr><th class="left">Implementation</th><th>count</th><th>max err</th><th>median err</th></tr></thead><tbody>';
  for (const imp of IMPLS) {{
    const errs = byImpl[imp] || [];
    if (!errs.length) continue;
    errs.sort((a,b) => a-b);
    const max = errs[errs.length-1];
    const med = errs[Math.floor(errs.length/2)];
    html += `<tr><td class="left">${{imp}}</td><td>${{errs.length}}</td><td>${{fmtErr(max)}}</td><td>${{fmtErr(med)}}</td></tr>`;
  }}
  html += '</tbody></table>';
  document.getElementById('precision-table').innerHTML = html;
}}

function buildMatrix(dir, device, impl, n_ffts, batches) {{
  const z = n_ffts.map(n =>
    batches.map(b => lookup(REPORT.rows, dir, device, b, n, impl))
  );
  return {{ z, n_ffts, batches }};
}}

function winnerMatrix(dir, device, n_ffts, batches) {{
  const z = n_ffts.map(n =>
    batches.map(b => {{
      let best = null, bestMs = Infinity;
      for (let i = 0; i < IMPLS.length; i++) {{
        const ms = lookup(REPORT.rows, dir, device, b, n, IMPLS[i]);
        if (ms != null && ms < bestMs) {{ bestMs = ms; best = i; }}
      }}
      return best;
    }})
  );
  return {{ z, n_ffts, batches }};
}}

function ratioMatrix(dir, device, numer, denom, n_ffts, batches) {{
  const z = n_ffts.map(n =>
    batches.map(b => {{
      const a = lookup(REPORT.rows, dir, device, b, n, numer);
      const d = lookup(REPORT.rows, dir, device, b, n, denom);
      if (a == null || d == null || d === 0) return null;
      return a / d;
    }})
  );
  return {{ z, n_ffts, batches }};
}}

const PLOTLY_LAYOUT = {{
  paper_bgcolor: '#1a2332',
  plot_bgcolor: '#1a2332',
  font: {{ color: '#e6edf3', size: 11 }},
  margin: {{ l: 60, r: 20, t: 40, b: 50 }},
}};

function plotHeatmap(el, title, z, x, y, colorscale, zmid) {{
  const zLog = z.map(row => row.map(v => (v == null || v <= 0) ? null : Math.log10(v)));
  const data = [{{
    z: zLog, x, y, type: 'heatmap',
    customdata: z,
    colorscale: colorscale || 'Viridis',
    zmid: zmid != null ? Math.log10(zmid) : undefined,
    hovertemplate: 'n=%{{y}} batch=%{{x}}<br>%{{customdata:.4f}} ms<extra></extra>',
    colorbar: {{ tickfont: {{ color: '#8b949e' }}, title: 'log10(ms)' }},
  }}];
  const layout = {{
    ...PLOTLY_LAYOUT,
    title: {{ text: title, font: {{ size: 13, color: '#8b949e' }} }},
    xaxis: {{ title: 'batch', type: 'log', tickvals: x }},
    yaxis: {{ title: 'n_fft', tickvals: y }},
  }};
  Plotly.newPlot(el, data, layout, {{ responsive: true, displayModeBar: true }});
}}

function plotSurface3d(el, title, z, x, y) {{
  const data = [{{
    z, x, y, type: 'surface',
    colorscale: 'Viridis',
    hovertemplate: 'n=%{{y}} batch=%{{x}}<br>%{{z:.4f}} ms<extra></extra>',
    colorbar: {{ tickfont: {{ color: '#8b949e' }} }},
  }}];
  const layout = {{
    ...PLOTLY_LAYOUT,
    title: {{ text: title, font: {{ size: 13, color: '#8b949e' }} }},
    scene: {{
      xaxis: {{ title: 'batch', type: 'log', backgroundcolor: '#1a2332', gridcolor: '#30363d' }},
      yaxis: {{ title: 'n_fft', backgroundcolor: '#1a2332', gridcolor: '#30363d' }},
      zaxis: {{ title: 'ms (log)', type: 'log', backgroundcolor: '#1a2332', gridcolor: '#30363d' }},
      bgcolor: '#1a2332',
    }},
    margin: {{ l: 0, r: 0, t: 40, b: 0 }},
  }};
  Plotly.newPlot(el, data, layout, {{ responsive: true, displayModeBar: true }});
}}

function plotWinnerHeatmap(el, title, z, x, y) {{
  const labels = IMPLS;
  const data = [{{
    z, x, y, type: 'heatmap',
    colorscale: [
      [0, '#4e79a7'], [0.33, '#f28e2b'], [0.66, '#59a14f'], [1, '#e15759']
    ],
    zmin: 0, zmax: Math.max(0, labels.length - 1),
    hovertemplate: 'n=%{{y}} batch=%{{x}}<br>winner: %{{text}}<extra></extra>',
    text: z.map(row => row.map(i => i == null ? '' : labels[i])),
    showscale: false,
  }}];
  const layout = {{
    ...PLOTLY_LAYOUT,
    title: {{ text: title, font: {{ size: 13, color: '#8b949e' }} }},
    xaxis: {{ title: 'batch', type: 'log', tickvals: x }},
    yaxis: {{ title: 'n_fft', tickvals: y }},
  }};
  Plotly.newPlot(el, data, layout, {{ responsive: true, displayModeBar: true }});
}}

document.querySelectorAll('[data-plotly]').forEach(el => {{
  const spec = JSON.parse(el.dataset.plotly);
  const {{ z, n_ffts, batches }} = spec.kind === 'winner'
    ? winnerMatrix(spec.dir, spec.device, spec.n_ffts, spec.batches)
    : spec.kind === 'ratio'
    ? ratioMatrix(spec.dir, spec.device, spec.numer, spec.denom, spec.n_ffts, spec.batches)
    : buildMatrix(spec.dir, spec.device, spec.impl, spec.n_ffts, spec.batches);
  const title = spec.title;
  if (spec.kind === 'surface') plotSurface3d(el, title, z, batches, n_ffts);
  else if (spec.kind === 'winner') plotWinnerHeatmap(el, title, z, batches, n_ffts);
  else if (spec.kind === 'ratio') plotHeatmap(el, title, z, batches, n_ffts, 'RdYlGn_r', 1.0);
  else plotHeatmap(el, title, z, batches, n_ffts);
}});

document.querySelectorAll('.tab-group').forEach(group => {{
  const tabs = group.querySelectorAll('.tab');
  const panels = group.querySelectorAll('.panel');
  tabs.forEach((tab, i) => {{
    tab.addEventListener('click', () => {{
      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      panels[i].classList.add('active');
      panels[i].querySelectorAll('[data-plotly]').forEach(el => Plotly.Plots.resize(el));
    }});
  }});
}});

const charts = [];
document.querySelectorAll('canvas[data-chart]').forEach(canvas => {{
  const spec = JSON.parse(canvas.dataset.chart);
  const labels = spec.n_ffts.map(n => 'n=' + n);
  const datasets = spec.impls.map(imp => ({{
    label: imp,
    data: spec.n_ffts.map(n => lookup(REPORT.rows, spec.dir, spec.device, spec.batch, n, imp)),
    borderColor: COLORS[imp] || '#999',
    backgroundColor: (COLORS[imp] || '#999') + '99',
    tension: 0.15,
    spanGaps: true,
  }}));
  charts.push(new Chart(canvas, {{
    type: spec.log ? 'line' : 'bar',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: '#8b949e' }} }},
        title: {{ display: false }},
      }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
        y: {{
          type: spec.log ? 'logarithmic' : 'linear',
          title: {{ display: true, text: 'ms / iter', color: '#8b949e' }},
          ticks: {{ color: '#8b949e' }},
          grid: {{ color: '#30363d' }},
        }},
      }},
    }},
  }}));
}});

buildPrecisionTable();
</script>
</body>
</html>"##,
        generated = generated,
        iters = report.iters,
        compiled = report.with_butterfly_compiled,
        weights = html_escape(&weights),
        sweep_ms = report.elapsed_ms,
        legend_html = legend_html(report.with_butterfly_compiled),
        heatmap_sections = heatmap_sections,
        chart_sections = chart_sections,
        table_sections = table_sections,
        devices = html_escape(&devices.join(", ")),
        directions = html_escape(&directions.join(", ")),
        json = json,
        impls_json = serde_json::to_string(&impls).unwrap_or_else(|_| "[]".to_string()),
        colors_json = colors_json(),
    )
}

fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("UTC {secs}")
}

fn colors_json() -> String {
    let mut obj = serde_json::Map::new();
    for (k, v) in IMPL_COLORS {
        obj.insert(k.to_string(), serde_json::Value::String(v.to_string()));
    }
    serde_json::Value::Object(obj).to_string()
}

fn legend_html(with_compiled: bool) -> String {
    impl_labels(with_compiled)
        .iter()
        .map(|imp| {
            let color = IMPL_COLORS
                .iter()
                .find(|(k, _)| *k == imp.as_str())
                .map(|(_, c)| *c)
                .unwrap_or("#999");
            format!(
                r#"<div class="legend-item"><span class="swatch" style="background:{color}"></span>{imp}</div>"#
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn impl_labels(with_compiled: bool) -> Vec<String> {
    let mut v = vec![
        "rustfft".to_string(),
        "rlx_op_fft".to_string(),
        "rlx_op_ifft".to_string(),
        "butterfly_eager".to_string(),
    ];
    if with_compiled {
        v.push("butterfly_compiled".to_string());
    }
    v
}

fn unique_strings(rows: &[SweepRow], f: impl Fn(&SweepRow) -> &str) -> Vec<String> {
    rows.iter()
        .map(f)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn axis_values(report: &SweepReport, dir: &str, device: &str) -> (Vec<usize>, Vec<usize>) {
    let batches: Vec<usize> = report
        .rows
        .iter()
        .filter(|r| r.direction == dir && r.device == device)
        .map(|r| r.batch)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let n_ffts: Vec<usize> = report
        .rows
        .iter()
        .filter(|r| r.direction == dir && r.device == device)
        .map(|r| r.n_fft)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    (n_ffts, batches)
}

fn plotly_div(spec: &serde_json::Value, class: &str) -> String {
    format!(
        r#"<div class="plotly {class}" data-plotly='{spec}'></div>"#,
        class = class,
        spec = html_escape(&spec.to_string())
    )
}

fn build_plotly_sections(report: &SweepReport) -> String {
    let impls = impl_labels(report.with_butterfly_compiled);
    let directions = unique_strings(&report.rows, |r| &r.direction);
    let devices = unique_strings(&report.rows, |r| &r.device);
    let mut out = String::new();

    for dir in &directions {
        for device in &devices {
            let (n_ffts, batches) = axis_values(report, dir, device);
            if n_ffts.is_empty() || batches.is_empty() {
                continue;
            }
            let group_id = slug(&format!("hm-{dir}-{device}"));
            out.push_str(&format!("<h3>{dir} — {device}</h3>"));
            out.push_str(&format!(r#"<div class="tab-group" id="{group_id}">"#));

            // Tab: per-implementation heatmaps
            out.push_str(r#"<div class="tabs">"#);
            for (i, imp) in impls.iter().enumerate() {
                let active = if i == 0 { " active" } else { "" };
                out.push_str(&format!(
                    r#"<button class="tab{active}" type="button">{imp} heatmap</button>"#
                ));
            }
            out.push_str(r#"<button class="tab" type="button">3D surface (rlx)</button>"#);
            out.push_str(r#"<button class="tab" type="button">winner</button>"#);
            out.push_str(r#"<button class="tab" type="button">rlx / rustfft</button>"#);
            out.push_str("</div>");

            for (i, imp) in impls.iter().enumerate() {
                let active = if i == 0 { " active" } else { "" };
                let spec = serde_json::json!({
                    "kind": "heatmap",
                    "title": format!("{imp} · {dir} · {device}"),
                    "dir": dir,
                    "device": device,
                    "impl": imp,
                    "n_ffts": n_ffts,
                    "batches": batches,
                });
                out.push_str(&format!(
                    r#"<div class="panel{active}"><div class="card">{plot}</div></div>"#,
                    plot = plotly_div(&spec, "")
                ));
            }

            let surface_spec = serde_json::json!({
                "kind": "surface",
                "title": format!("rlx_op_fft 3D · {dir} · {device}"),
                "dir": dir,
                "device": device,
                "impl": "rlx_op_fft",
                "n_ffts": n_ffts,
                "batches": batches,
            });
            out.push_str(&format!(
                r#"<div class="panel"><div class="card">{plot}</div></div>"#,
                plot = plotly_div(&surface_spec, "plotly-3d")
            ));

            let winner_spec = serde_json::json!({
                "kind": "winner",
                "title": format!("Fastest impl · {dir} · {device}"),
                "dir": dir,
                "device": device,
                "n_ffts": n_ffts,
                "batches": batches,
            });
            out.push_str(&format!(
                r#"<div class="panel"><div class="card">{plot}</div></div>"#,
                plot = plotly_div(&winner_spec, "")
            ));

            let ratio_spec = serde_json::json!({
                "kind": "ratio",
                "title": format!("rlx_op_fft / rustfft · {dir} · {device}"),
                "dir": dir,
                "device": device,
                "numer": "rlx_op_fft",
                "denom": "rustfft",
                "n_ffts": n_ffts,
                "batches": batches,
            });
            out.push_str(&format!(
                r#"<div class="panel"><div class="card">{plot}</div></div>"#,
                plot = plotly_div(&ratio_spec, "")
            ));

            out.push_str("</div>");
        }
    }
    out
}

fn build_chart_sections(report: &SweepReport) -> String {
    let impls = impl_labels(report.with_butterfly_compiled);
    let directions = unique_strings(&report.rows, |r| &r.direction);
    let devices = unique_strings(&report.rows, |r| &r.device);
    let mut out = String::new();

    for dir in &directions {
        out.push_str(&format!("<h3>{dir}</h3><div class=\"grid\">"));
        for device in &devices {
            let batches: Vec<usize> = report
                .rows
                .iter()
                .filter(|r| r.direction == *dir && r.device == *device)
                .map(|r| r.batch)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let n_ffts: Vec<usize> = report
                .rows
                .iter()
                .filter(|r| r.direction == *dir && r.device == *device)
                .map(|r| r.n_fft)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();

            for batch in batches {
                let id = slug(&format!("{dir}-{device}-b{batch}"));
                let spec = serde_json::json!({
                    "dir": dir,
                    "device": device,
                    "batch": batch,
                    "n_ffts": n_ffts,
                    "impls": impls,
                    "log": true,
                });
                out.push_str(&format!(
                    r#"<div class="card"><h4>{dir} · {device} · batch={batch}</h4><canvas id="{id}" data-chart='{spec}'></canvas></div>"#,
                    spec = html_escape(&spec.to_string())
                ));
            }
        }
        out.push_str("</div>");
    }
    out
}

fn build_table_sections(report: &SweepReport) -> String {
    let impls = impl_labels(report.with_butterfly_compiled);
    let directions = unique_strings(&report.rows, |r| &r.direction);
    let devices = unique_strings(&report.rows, |r| &r.device);
    let mut out = String::new();

    for dir in &directions {
        for device in &devices {
            out.push_str(&format!("<h3>{dir} — {device}</h3>"));
            let batches: Vec<usize> = report
                .rows
                .iter()
                .filter(|r| r.direction == *dir && r.device == *device)
                .map(|r| r.batch)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let n_ffts: Vec<usize> = report
                .rows
                .iter()
                .filter(|r| r.direction == *dir && r.device == *device)
                .map(|r| r.n_fft)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();

            out.push_str("<table><thead><tr><th class=\"left\">impl \\ n_fft</th>");
            for b in &batches {
                out.push_str(&format!("<th colspan=\"{}\">batch={b}</th>", n_ffts.len()));
            }
            out.push_str("</tr><tr><th class=\"left\"></th>");
            for _ in &batches {
                for n in &n_ffts {
                    out.push_str(&format!("<th>n={n}</th>"));
                }
            }
            out.push_str("</tr></thead><tbody>");

            for imp in &impls {
                out.push_str(&format!("<tr><td class=\"left\">{imp}</td>"));
                for batch in &batches {
                    let mut ms_vals = Vec::new();
                    for n in &n_ffts {
                        if let Some(r) = report.rows.iter().find(|r| {
                            r.direction == *dir
                                && r.device == *device
                                && r.batch == *batch
                                && r.n_fft == *n
                                && r.implementation == *imp
                        }) {
                            ms_vals.push(r.ms);
                        }
                    }
                    let min = ms_vals.iter().copied().fold(f64::INFINITY, f64::min);
                    let max = ms_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                    for n in &n_ffts {
                        let cell = report.rows.iter().find(|r| {
                            r.direction == *dir
                                && r.device == *device
                                && r.batch == *batch
                                && r.n_fft == *n
                                && r.implementation == *imp
                        });
                        if let Some(r) = cell {
                            let best = best_impl_ms(report, dir, device, *batch, *n, &impls);
                            let is_best = (r.ms - best).abs() < 1e-9;
                            let bg = heat_color(r.ms, min, max);
                            let cls = if is_best { " class=\"best\"" } else { "" };
                            out.push_str(&format!(
                                "<td{cls} style=\"background:{bg}\">{ms}<br><span style=\"color:#8b949e;font-size:0.75rem\">err {err}</span></td>",
                                ms = format_ms(r.ms),
                                err = format_err(r.max_err),
                                bg = bg,
                                cls = cls,
                            ));
                        } else {
                            out.push_str("<td>n/a</td>");
                        }
                    }
                }
                out.push_str("</tr>");
            }
            out.push_str("</tbody></table>");
        }
    }
    out
}

fn best_impl_ms(
    report: &SweepReport,
    dir: &str,
    device: &str,
    batch: usize,
    n: usize,
    impls: &[String],
) -> f64 {
    impls
        .iter()
        .filter_map(|imp| {
            report.rows.iter().find(|r| {
                r.direction == dir
                    && r.device == device
                    && r.batch == batch
                    && r.n_fft == n
                    && r.implementation == *imp
            })
        })
        .map(|r| r.ms)
        .fold(f64::INFINITY, f64::min)
}

fn heat_color(ms: f64, min: f64, max: f64) -> String {
    if ms.is_nan() {
        return "transparent".to_string();
    }
    let t = if max > min {
        (ms - min) / (max - min)
    } else {
        0.0
    };
    let r = (35.0 + t * 120.0) as u8;
    let g = (134.0 - t * 90.0) as u8;
    let b = (54.0 - t * 20.0) as u8;
    format!("rgba({r},{g},{b},0.25)")
}

fn format_ms(ms: f64) -> String {
    if ms.is_nan() {
        "n/a".to_string()
    } else if ms < 0.01 {
        format!("{ms:.2e}")
    } else {
        format!("{ms:.4}")
    }
}

fn format_err(err: Option<f32>) -> String {
    match err {
        Some(v) if !v.is_nan() => format!("{v:.2e}"),
        _ => "—".to_string(),
    }
}

fn slug(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

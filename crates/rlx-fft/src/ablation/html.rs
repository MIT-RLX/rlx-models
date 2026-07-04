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

//! Self-contained HTML ablation report (Chart.js + Plotly heatmaps).

use crate::ablation::{AblationReport, AblationRow, ablation_winners, tier_summary};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::Path;

const VARIANT_COLORS: &[(&str, &str)] = &[
    ("rustfft", "#4e79a7"),
    ("rlx_op_fft", "#f28e2b"),
    ("rlx_op_ifft", "#ff9d4d"),
    ("butterfly_eager", "#59a14f"),
    ("butterfly_compiled", "#e15759"),
    ("stockham_eager", "#76b7b2"),
    ("stockham_compiled", "#edc948"),
    ("fused_spectral_eager", "#b07aa1"),
    ("fused_spectral_compiled", "#ff9da7"),
    ("butterfly_q8", "#9c755f"),
    ("butterfly_unitary", "#bab0ac"),
    ("domain_twiddle", "#86bcb6"),
    ("welch_rustfft", "#4c78a8"),
    ("welch_rlx_op_fft", "#f58518"),
    ("welch_butterfly_eager", "#54a24b"),
    ("welch_butterfly_compiled", "#e45756"),
];

pub fn read_ablation_json(path: &Path) -> Result<AblationReport> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn write_ablation_html(path: &Path, report: &AblationReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, render_ablation_html(report))
        .with_context(|| format!("write {}", path.display()))
}

pub fn render_ablation_html(report: &AblationReport) -> String {
    let json = serde_json::to_string(report).unwrap_or_else(|_| "{}".to_string());
    let generated = chrono_lite_now();
    let variants = unique_variants(&report.rows);
    let devices = unique_strings(&report.rows, |r| &r.device);
    let tier_wins = tier_summary(report);
    let tier_summary_html = build_tier_summary_html(&tier_wins);
    let legend_html = build_legend_html(&variants);
    let heatmap_sections = build_plotly_sections(report, &variants, &devices);
    let chart_sections = build_chart_sections(report, &variants, &devices);
    let table_sections = build_table_sections(report, &variants, &devices);
    let winners_html = build_winners_table(report);

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RLX-FFT Ablation Report</title>
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
  canvas {{ max-height: 320px; }}
  .plotly {{ width: 100%; min-height: 360px; }}
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
  .tier-pill {{
    display: inline-block;
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    font-size: 0.75rem;
    background: #21262d;
    color: var(--muted);
  }}
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
  <h1>RLX-FFT Variant Ablation</h1>
  <div class="meta">
    <span>Generated {generated}</span>
    <span>iters={iters}</span>
    <span>train_steps={train_steps}</span>
    <span>elapsed {elapsed_ms:.0} ms</span>
    <span>{variant_count} variants</span>
  </div>
</header>
<main>
  <div class="note">
    <strong>How to read this report.</strong>
    Timings are milliseconds per forward FFT iteration (lower is faster).
    <code>max_err</code> is vs <code>rustfft</code> on the same fixed signal (lower is better).
    Tier A = Stockham / fused spectral; B = compiled / Q8; C = unitary / domain-adaptive.
    Green cell outline = fastest variant for that n_fft × batch cell.
  </div>

  <section>
    <h2>Tier win counts</h2>
    <p style="color:var(--muted);font-size:0.9rem;margin-top:0">
      Number of (n_fft, batch, device) cells where a tier’s variant was fastest.
    </p>
    {tier_summary_html}
    {winners_html}
  </section>

  <div class="legend">{legend_html}</div>

  <section>
    <h2>Performance heatmaps</h2>
    <p style="color:var(--muted);font-size:0.9rem;margin-top:0">
      ms/iter over <strong>n_fft × batch</strong>. Use tabs to switch variant or view winner / ratio panels.
    </p>
    {heatmap_sections}
  </section>

  <section>
    <h2>Bar charts (fixed batch)</h2>
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
<footer>rlx-fft ablation · devices: {devices_footer}</footer>
<script>
const REPORT = {json};
const VARIANTS = {variants_json};
const COLORS = {colors_json};

function lookup(rows, device, batch, n, variant) {{
  const hit = rows.find(r =>
    r.device === device && r.batch === batch &&
    r.n_fft === n && r.variant === variant
  );
  return hit ? hit : null;
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

function buildMatrix(device, variant, n_ffts, batches) {{
  const z = n_ffts.map(n =>
    batches.map(b => {{
      const r = lookup(REPORT.rows, device, b, n, variant);
      return r ? r.ms : null;
    }})
  );
  return {{ z, n_ffts, batches }};
}}

function buildErrMatrix(device, variant, n_ffts, batches) {{
  const z = n_ffts.map(n =>
    batches.map(b => {{
      const r = lookup(REPORT.rows, device, b, n, variant);
      return r ? r.max_err : null;
    }})
  );
  return {{ z, n_ffts, batches }};
}}

function winnerMatrix(device, n_ffts, batches) {{
  const z = n_ffts.map(n =>
    batches.map(b => {{
      let best = null, bestMs = Infinity;
      for (let i = 0; i < VARIANTS.length; i++) {{
        const r = lookup(REPORT.rows, device, b, n, VARIANTS[i]);
        if (r && r.ms != null && !Number.isNaN(r.ms) && r.ms < bestMs) {{
          bestMs = r.ms; best = i;
        }}
      }}
      return best;
    }})
  );
  return {{ z, n_ffts, batches }};
}}

function ratioMatrix(device, numer, denom, n_ffts, batches) {{
  const z = n_ffts.map(n =>
    batches.map(b => {{
      const a = lookup(REPORT.rows, device, b, n, numer);
      const d = lookup(REPORT.rows, device, b, n, denom);
      if (!a || !d || d.ms == null || d.ms === 0) return null;
      return a.ms / d.ms;
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

function plotHeatmap(el, title, z, x, y, colorscale, zmid, errMode) {{
  const zPlot = errMode
    ? z
    : z.map(row => row.map(v => (v == null || v <= 0) ? null : Math.log10(v)));
  const data = [{{
    z: zPlot, x, y, type: 'heatmap',
    customdata: z,
    colorscale: colorscale || 'Viridis',
    zmid: (!errMode && zmid != null) ? Math.log10(zmid) : zmid,
    hovertemplate: errMode
      ? 'n=%{{y}} batch=%{{x}}<br>err %{{customdata:.2e}}<extra></extra>'
      : 'n=%{{y}} batch=%{{x}}<br>%{{customdata:.4f}} ms<extra></extra>',
    colorbar: {{ tickfont: {{ color: '#8b949e' }}, title: errMode ? 'max_err' : 'log10(ms)' }},
  }}];
  const layout = {{
    ...PLOTLY_LAYOUT,
    title: {{ text: title, font: {{ size: 13, color: '#8b949e' }} }},
    xaxis: {{ title: 'batch', type: 'log', tickvals: x }},
    yaxis: {{ title: 'n_fft', tickvals: y }},
  }};
  Plotly.newPlot(el, data, layout, {{ responsive: true, displayModeBar: true }});
}}

function plotWinnerHeatmap(el, title, z, x, y) {{
  const labels = VARIANTS;
  const data = [{{
    z, x, y, type: 'heatmap',
    colorscale: [
      [0, '#4e79a7'], [0.2, '#f28e2b'], [0.4, '#59a14f'], [0.6, '#76b7b2'],
      [0.8, '#b07aa1'], [1, '#e15759']
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
  let z, n_ffts = spec.n_ffts, batches = spec.batches;
  if (spec.kind === 'winner') {{
    ({{ z, n_ffts, batches }}) = winnerMatrix(spec.device, spec.n_ffts, spec.batches);
    plotWinnerHeatmap(el, spec.title, z, batches, n_ffts);
  }} else if (spec.kind === 'ratio') {{
    ({{ z, n_ffts, batches }}) = ratioMatrix(spec.device, spec.numer, spec.denom, spec.n_ffts, spec.batches);
    plotHeatmap(el, spec.title, z, batches, n_ffts, 'RdYlGn_r', 1.0, false);
  }} else if (spec.kind === 'err') {{
    ({{ z, n_ffts, batches }}) = buildErrMatrix(spec.device, spec.variant, spec.n_ffts, spec.batches);
    plotHeatmap(el, spec.title, z, batches, n_ffts, 'YlOrRd', null, true);
  }} else {{
    ({{ z, n_ffts, batches }}) = buildMatrix(spec.device, spec.variant, spec.n_ffts, spec.batches);
    plotHeatmap(el, spec.title, z, batches, n_ffts, 'Viridis', null, false);
  }}
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

function buildPrecisionTable() {{
  const rows = REPORT.rows.filter(r => r.variant !== 'rustfft' && r.max_err != null && !Number.isNaN(r.max_err));
  const byVar = {{}};
  for (const r of rows) {{
    if (!byVar[r.variant]) byVar[r.variant] = [];
    byVar[r.variant].push(r.max_err);
  }}
  let html = '<table><thead><tr><th class="left">Variant</th><th>tier</th><th>count</th><th>max err</th><th>median err</th></tr></thead><tbody>';
  for (const v of VARIANTS) {{
    const errs = byVar[v] || [];
    if (!errs.length) continue;
    errs.sort((a,b) => a-b);
    const tier = rows.find(r => r.variant === v)?.tier || '—';
    const max = errs[errs.length-1];
    const med = errs[Math.floor(errs.length/2)];
    html += `<tr><td class="left">${{v}}</td><td><span class="tier-pill">${{tier}}</span></td><td>${{errs.length}}</td><td>${{fmtErr(max)}}</td><td>${{fmtErr(med)}}</td></tr>`;
  }}
  html += '</tbody></table>';
  document.getElementById('precision-table').innerHTML = html;
}}

const charts = [];
document.querySelectorAll('canvas[data-chart]').forEach(canvas => {{
  const spec = JSON.parse(canvas.dataset.chart);
  const datasets = spec.variants.map(v => {{
    const pts = spec.n_ffts.map(n => {{
      const r = lookup(REPORT.rows, spec.device, spec.batch, n, v);
      return r ? r.ms : null;
    }});
    return {{
      label: v,
      data: pts,
      borderColor: COLORS[v] || '#999',
      backgroundColor: (COLORS[v] || '#999') + '99',
      tension: 0.2,
      spanGaps: false,
    }};
  }});
  charts.push(new Chart(canvas, {{
    type: 'bar',
    data: {{ labels: spec.n_ffts.map(String), datasets }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: '#8b949e', boxWidth: 12 }} }},
        title: {{ display: false }},
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'n_fft', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
        y: {{ type: spec.log ? 'logarithmic' : 'linear', title: {{ display: true, text: 'ms', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
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
        train_steps = report.train_steps,
        elapsed_ms = report.elapsed_ms,
        variant_count = variants.len(),
        tier_summary_html = tier_summary_html,
        winners_html = winners_html,
        legend_html = legend_html,
        heatmap_sections = heatmap_sections,
        chart_sections = chart_sections,
        table_sections = table_sections,
        devices_footer = devices.join(", "),
        json = json,
        variants_json = serde_json::to_string(&variants).unwrap_or_else(|_| "[]".to_string()),
        colors_json = colors_json(),
    )
}

fn build_tier_summary_html(wins: &std::collections::HashMap<String, usize>) -> String {
    if wins.is_empty() {
        return "<p>No timing data.</p>".to_string();
    }
    let mut tiers: Vec<_> = wins.iter().collect();
    tiers.sort_by(|a, b| b.1.cmp(a.1));
    let max = *tiers[0].1;
    let mut out = String::from("<div class=\"grid\">");
    for (tier, count) in tiers {
        let pct = if max > 0 {
            (*count as f64 / max as f64 * 100.0) as u32
        } else {
            0
        };
        out.push_str(&format!(
            r#"<div class="card"><h4>Tier {tier}</h4><div style="font-size:2rem;font-weight:600">{count}</div><div style="color:var(--muted);font-size:0.85rem">wins · bar {pct}%</div></div>"#
        ));
    }
    out.push_str("</div>");
    out
}

fn build_winners_table(report: &AblationReport) -> String {
    let winners = ablation_winners(report);
    if winners.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "<h3>Fastest variant per cell</h3><table><thead><tr>\
         <th class=\"left\">direction</th><th class=\"left\">n_fft</th><th>batch</th><th class=\"left\">device</th>\
         <th class=\"left\">variant</th><th>ms</th></tr></thead><tbody>",
    );
    for (n, b, d, dir, variant, ms) in winners {
        out.push_str(&format!(
            "<tr><td class=\"left\">{dir}</td><td class=\"left\">{n}</td><td>{b}</td><td class=\"left\">{d}</td>\
             <td class=\"left\">{variant}</td><td>{ms:.4}</td></tr>"
        ));
    }
    out.push_str("</tbody></table>");
    out
}

fn build_legend_html(variants: &[String]) -> String {
    variants
        .iter()
        .map(|v| {
            let color = variant_color(v);
            format!(
                r#"<div class="legend-item"><span class="swatch" style="background:{color}"></span>{v}</div>"#
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_plotly_sections(
    report: &AblationReport,
    variants: &[String],
    devices: &[String],
) -> String {
    let heatmap_variants: Vec<&String> = variants
        .iter()
        .filter(|v| {
            matches!(
                v.as_str(),
                "rustfft"
                    | "rlx_op_fft"
                    | "butterfly_eager"
                    | "stockham_eager"
                    | "butterfly_q8"
                    | "domain_twiddle"
            )
        })
        .collect();

    let mut out = String::new();
    for device in devices {
        let n_ffts = n_ffts_for_device(report, device);
        let batches = batches_for_device(report, device);
        if n_ffts.is_empty() || batches.is_empty() {
            continue;
        }
        let group_id = slug(&format!("heat-{device}"));
        out.push_str(&format!(
            "<h3>{device}</h3><div class=\"tab-group\" id=\"{group_id}\">"
        ));
        out.push_str("<div class=\"tabs\">");
        for (i, v) in heatmap_variants.iter().enumerate() {
            let active = if i == 0 { " active" } else { "" };
            out.push_str(&format!(
                r#"<button class="tab{active}" type="button">{v}</button>"#
            ));
        }
        out.push_str(r#"<button class="tab" type="button">winner</button>"#);
        out.push_str(r#"<button class="tab" type="button">rlx / rustfft</button>"#);
        out.push_str(r#"<button class="tab" type="button">butterfly err</button>"#);
        out.push_str("</div>");

        for (i, v) in heatmap_variants.iter().enumerate() {
            let active = if i == 0 { " active" } else { "" };
            let spec = serde_json::json!({
                "kind": "ms",
                "title": format!("{v} · {device}"),
                "device": device,
                "variant": v,
                "n_ffts": n_ffts,
                "batches": batches,
            });
            out.push_str(&format!(
                r#"<div class="panel{active}"><div class="card">{plot}</div></div>"#,
                plot = plotly_div(&spec)
            ));
        }

        let winner_spec = serde_json::json!({
            "kind": "winner",
            "title": format!("Fastest variant · {device}"),
            "device": device,
            "n_ffts": n_ffts,
            "batches": batches,
        });
        out.push_str(&format!(
            r#"<div class="panel"><div class="card">{plot}</div></div>"#,
            plot = plotly_div(&winner_spec)
        ));

        let ratio_spec = serde_json::json!({
            "kind": "ratio",
            "title": format!("rlx_op_fft / rustfft · {device}"),
            "device": device,
            "numer": "rlx_op_fft",
            "denom": "rustfft",
            "n_ffts": n_ffts,
            "batches": batches,
        });
        out.push_str(&format!(
            r#"<div class="panel"><div class="card">{plot}</div></div>"#,
            plot = plotly_div(&ratio_spec)
        ));

        let err_spec = serde_json::json!({
            "kind": "err",
            "title": format!("butterfly_eager max_err · {device}"),
            "device": device,
            "variant": "butterfly_eager",
            "n_ffts": n_ffts,
            "batches": batches,
        });
        out.push_str(&format!(
            r#"<div class="panel"><div class="card">{plot}</div></div>"#,
            plot = plotly_div(&err_spec)
        ));

        out.push_str("</div>");
    }
    out
}

fn build_chart_sections(
    report: &AblationReport,
    variants: &[String],
    devices: &[String],
) -> String {
    let chart_variants: Vec<String> = variants
        .iter()
        .filter(|v| {
            !matches!(
                v.as_str(),
                "fused_spectral_eager" | "fused_spectral_compiled" | "butterfly_unitary"
            )
        })
        .cloned()
        .collect();

    let mut out = String::new();
    for device in devices {
        let batches = batches_for_device(report, device);
        let n_ffts = n_ffts_for_device(report, device);
        out.push_str(&format!("<h3>{device}</h3><div class=\"grid\">"));
        for batch in batches {
            let id = slug(&format!("chart-{device}-b{batch}"));
            let spec = serde_json::json!({
                "device": device,
                "batch": batch,
                "n_ffts": n_ffts,
                "variants": chart_variants,
                "log": true,
            });
            out.push_str(&format!(
                r#"<div class="card"><h4>{device} · batch={batch}</h4><canvas id="{id}" data-chart='{spec}'></canvas></div>"#,
                spec = html_escape(&spec.to_string())
            ));
        }
        out.push_str("</div>");
    }
    out
}

fn build_table_sections(
    report: &AblationReport,
    variants: &[String],
    devices: &[String],
) -> String {
    let mut out = String::new();
    for device in devices {
        let batches = batches_for_device(report, device);
        let n_ffts = n_ffts_for_device(report, device);
        out.push_str(&format!("<h3>{device}</h3>"));
        out.push_str("<table><thead><tr><th class=\"left\">variant</th><th>tier</th>");
        for b in &batches {
            out.push_str(&format!("<th colspan=\"{}\">batch={b}</th>", n_ffts.len()));
        }
        out.push_str("</tr><tr><th class=\"left\"></th><th></th>");
        for _ in &batches {
            for n in &n_ffts {
                out.push_str(&format!("<th>n={n}</th>"));
            }
        }
        out.push_str("</tr></thead><tbody>");

        for variant in variants {
            let tier = report
                .rows
                .iter()
                .find(|r| r.variant == *variant)
                .map(|r| r.tier.as_str())
                .unwrap_or("?");
            out.push_str(&format!(
                "<tr><td class=\"left\">{variant}</td><td><span class=\"tier-pill\">{tier}</span></td>"
            ));
            for batch in &batches {
                let mut ms_vals = Vec::new();
                for n in &n_ffts {
                    if let Some(r) = find_row(report, device, *batch, *n, variant) {
                        ms_vals.push(r.ms);
                    }
                }
                let min = ms_vals.iter().copied().fold(f64::INFINITY, f64::min);
                let max = ms_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                for n in &n_ffts {
                    if let Some(r) = find_row(report, device, *batch, *n, variant) {
                        let best = best_variant_ms(report, device, *batch, *n, variants);
                        let is_best = !r.ms.is_nan() && (r.ms - best).abs() < 1e-9;
                        let bg = heat_color(r.ms, min, max);
                        let cls = if is_best { " class=\"best\"" } else { "" };
                        out.push_str(&format!(
                            "<td{cls} style=\"background:{bg}\">{ms}<br><span style=\"color:#8b949e;font-size:0.75rem\">err {err}</span></td>",
                            cls = cls,
                            bg = bg,
                            ms = format_ms(r.ms),
                            err = format_err(r.max_err),
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
    out
}

fn find_row<'a>(
    report: &'a AblationReport,
    device: &str,
    batch: usize,
    n_fft: usize,
    variant: &str,
) -> Option<&'a AblationRow> {
    report.rows.iter().find(|r| {
        r.device == device && r.batch == batch && r.n_fft == n_fft && r.variant == variant
    })
}

fn best_variant_ms(
    report: &AblationReport,
    device: &str,
    batch: usize,
    n_fft: usize,
    variants: &[String],
) -> f64 {
    variants
        .iter()
        .filter_map(|v| find_row(report, device, batch, n_fft, v))
        .map(|r| r.ms)
        .filter(|ms| !ms.is_nan())
        .fold(f64::INFINITY, f64::min)
}

fn n_ffts_for_device(report: &AblationReport, device: &str) -> Vec<usize> {
    report
        .rows
        .iter()
        .filter(|r| r.device == device)
        .map(|r| r.n_fft)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn batches_for_device(report: &AblationReport, device: &str) -> Vec<usize> {
    report
        .rows
        .iter()
        .filter(|r| r.device == device)
        .map(|r| r.batch)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn unique_variants(rows: &[AblationRow]) -> Vec<String> {
    let mut v: Vec<String> = rows
        .iter()
        .map(|r| r.variant.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    v.sort_by(|a, b| {
        let tier_a = rows
            .iter()
            .find(|r| r.variant == *a)
            .map(|r| r.tier.as_str())
            .unwrap_or("");
        let tier_b = rows
            .iter()
            .find(|r| r.variant == *b)
            .map(|r| r.tier.as_str())
            .unwrap_or("");
        tier_a.cmp(tier_b).then_with(|| a.cmp(b))
    });
    v
}

fn unique_strings(rows: &[AblationRow], f: impl Fn(&AblationRow) -> &str) -> Vec<String> {
    rows.iter()
        .map(f)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn plotly_div(spec: &serde_json::Value) -> String {
    format!(
        r#"<div class="plotly" data-plotly="{}"></div>"#,
        html_escape(&spec.to_string())
    )
}

fn variant_color(name: &str) -> &str {
    VARIANT_COLORS
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, c)| *c)
        .unwrap_or("#999")
}

fn colors_json() -> String {
    let mut obj = serde_json::Map::new();
    for (k, v) in VARIANT_COLORS {
        obj.insert(k.to_string(), serde_json::Value::String(v.to_string()));
    }
    serde_json::Value::Object(obj).to_string()
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

fn format_err(err: f32) -> String {
    if err.is_nan() {
        "—".to_string()
    } else {
        format!("{err:.2e}")
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

fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("UTC {secs}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ablation::AblationRow;

    fn sample_report() -> AblationReport {
        AblationReport {
            iters: 10,
            train_steps: 20,
            both_dirs: true,
            with_welch: true,
            limit_sweep: false,
            n_ffts: vec![256],
            elapsed_ms: 1234.0,
            rows: vec![
                AblationRow {
                    tier: "baseline".into(),
                    variant: "rustfft".into(),
                    direction: "Forward".into(),
                    n_fft: 256,
                    batch: 8,
                    device: "cpu".into(),
                    iters: 10,
                    ms: 0.5,
                    max_err: 0.0,
                    train_steps: 0,
                    param_count: 0,
                    memory_bytes: 0,
                    status: "ok".into(),
                    note: None,
                },
                AblationRow {
                    tier: "baseline".into(),
                    variant: "rlx_op_fft".into(),
                    direction: "Forward".into(),
                    n_fft: 256,
                    batch: 8,
                    device: "cpu".into(),
                    iters: 10,
                    ms: 0.3,
                    max_err: 1e-5,
                    train_steps: 0,
                    param_count: 0,
                    memory_bytes: 0,
                    status: "ok".into(),
                    note: None,
                },
                AblationRow {
                    tier: "A".into(),
                    variant: "stockham_eager".into(),
                    direction: "Forward".into(),
                    n_fft: 256,
                    batch: 8,
                    device: "cpu".into(),
                    iters: 10,
                    ms: 0.8,
                    max_err: 1e-5,
                    train_steps: 0,
                    param_count: 896,
                    memory_bytes: 896 * 4,
                    status: "ok".into(),
                    note: None,
                },
            ],
        }
    }

    #[test]
    fn render_contains_key_sections() {
        let html = render_ablation_html(&sample_report());
        assert!(html.contains("RLX-FFT Variant Ablation"));
        assert!(html.contains("Tier win counts"));
        assert!(html.contains("Performance heatmaps"));
        assert!(html.contains("stockham_eager"));
        assert!(html.contains("buildPrecisionTable"));
    }

    #[test]
    fn roundtrip_json_html() {
        let report = sample_report();
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("ablation.json");
        let html_path = dir.path().join("ablation.html");
        crate::ablation::write_ablation_json(&json_path, &report).unwrap();
        let loaded = read_ablation_json(&json_path).unwrap();
        write_ablation_html(&html_path, &loaded).unwrap();
        let html = std::fs::read_to_string(&html_path).unwrap();
        assert!(html.contains("rlx_op_fft"));
    }
}

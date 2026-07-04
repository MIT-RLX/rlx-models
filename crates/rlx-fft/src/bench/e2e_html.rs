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

//! Self-contained HTML report for end-to-end learned FFT validation benches.

use crate::bench::e2e::{E2eBenchReport, E2eBenchRow};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::Path;

const BACKEND_COLORS: &[(&str, &str)] = &[
    ("rustfft_ref", "#4e79a7"),
    ("rlx_op_fft", "#f28e2b"),
    ("butterfly_eager", "#59a14f"),
    ("learned_model", "#76b7b2"),
    ("learned_q8", "#b07aa1"),
    ("learned_hard", "#edc948"),
    ("learned_compiled", "#e15759"),
    ("learned_distilled", "#ff7f0e"),
    ("learned_distilled_ternary", "#bcbd22"),
];

const CHART_BACKENDS: &[&str] = &[
    "rlx_op_fft",
    "learned_compiled",
    "learned_distilled",
    "learned_distilled_ternary",
    "learned_hard",
    "learned_model",
];

pub fn read_e2e_json(path: &Path) -> Result<E2eBenchReport> {
    crate::bench::e2e::read_e2e_json(path)
}

pub fn write_e2e_html(path: &Path, report: &E2eBenchReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, render_e2e_html(report))
        .with_context(|| format!("write {}", path.display()))
}

pub fn render_e2e_html(report: &E2eBenchReport) -> String {
    let json = serde_json::to_string(report).unwrap_or_else(|_| "{}".to_string());
    let generated = chrono_lite_now();
    let meta = &report.meta;
    let n_fft = meta
        .n_fft
        .max(report.rows.first().map(|r| r.n_fft).unwrap_or(0));
    let devices = if meta.devices.is_empty() {
        unique_strings(&report.rows, |r| &r.device)
    } else {
        meta.devices.clone()
    };
    let batches = if meta.batches.is_empty() {
        unique_usize(&report.rows, |r| r.batch)
    } else {
        meta.batches.clone()
    };
    let pipelines = unique_strings(&report.rows, |r| &r.pipeline);
    let backends = unique_strings(&report.rows, |r| &r.backend);
    let colors_json =
        serde_json::to_string(&backend_color_map(&backends)).unwrap_or_else(|_| "{}".into());
    let chart_backends: Vec<String> = CHART_BACKENDS
        .iter()
        .filter(|b| backends.iter().any(|x| x == *b))
        .map(|s| (*s).to_string())
        .collect();
    let chart_backends_json =
        serde_json::to_string(&chart_backends).unwrap_or_else(|_| "[]".into());
    let summary_cards = build_summary_cards(report, n_fft, &devices, &batches);
    let gate_table = build_gate_table(report);
    let distillation_section = build_distillation_section(report);
    let per_batch_section = build_per_batch_train_section(report);
    let mel_chart_sections =
        build_pipeline_chart_sections(report, "mel", &devices, &batches, &chart_backends);
    let other_pipeline_sections = pipelines
        .iter()
        .filter(|p| *p != "mel")
        .map(|p| build_pipeline_chart_sections(report, p, &devices, &batches, &chart_backends))
        .collect::<Vec<_>>()
        .join("\n");
    let heatmap_sections = build_heatmap_sections(report, &devices, &batches);
    let scatter_section = build_scatter_section(&devices);
    let line_sections = build_line_sections(&devices, &batches);
    let table_html = build_full_table(report);
    let legend_html = build_legend(&backends);

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RLX-FFT E2E Validation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.27.0/plotly.min.js"></script>
<style>
  :root {{
    --bg: #0f1419; --panel: #1a2332; --text: #e6edf3; --muted: #8b949e;
    --border: #30363d; --accent: #58a6ff; --pass: #238636; --fail: #da3633;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }}
  header {{
    padding: 2rem 2.5rem 1rem; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #161b22 0%, var(--bg) 100%);
  }}
  h1 {{ margin: 0 0 0.5rem; font-size: 1.75rem; font-weight: 600; }}
  .meta {{ color: var(--muted); font-size: 0.95rem; }}
  .meta span {{ margin-right: 1.25rem; }}
  main {{ padding: 1.5rem 2.5rem 3rem; max-width: 1400px; margin: 0 auto; }}
  section {{ margin-bottom: 2.5rem; }}
  h2 {{
    font-size: 1.25rem; margin: 0 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}
  h3 {{ font-size: 1rem; color: var(--accent); margin: 1.5rem 0 0.75rem; }}
  .note {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem 1.25rem; color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem;
  }}
  .cards {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .card-stat {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 1rem 1.1rem;
  }}
  .card-stat .label {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  .card-stat .value {{ font-size: 1.35rem; font-weight: 600; margin-top: 0.25rem; }}
  .card-stat .sub {{ color: var(--muted); font-size: 0.85rem; margin-top: 0.15rem; }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.25rem;
  }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 1rem 1rem 0.5rem;
  }}
  .card h4 {{ margin: 0 0 0.75rem; font-size: 0.9rem; font-weight: 500; color: var(--muted); }}
  canvas {{ max-height: 300px; }}
  .plotly {{ width: 100%; min-height: 340px; }}
  .tabs {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }}
  .tab {{
    background: #21262d; border: 1px solid var(--border); color: var(--muted);
    padding: 0.35rem 0.85rem; border-radius: 6px; cursor: pointer; font-size: 0.85rem;
  }}
  .tab.active {{ background: #30363d; color: var(--text); border-color: var(--accent); }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ padding: 0.45rem 0.65rem; border-bottom: 1px solid var(--border); text-align: right; }}
  th.left, td.left {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 500; }}
  tr:hover td {{ background: rgba(88,166,255,0.06); }}
  .pass {{ color: var(--pass); font-weight: 600; }}
  .fail {{ color: var(--fail); font-weight: 600; }}
  .warn {{ color: #d29922; font-weight: 600; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; margin: 1rem 0 2rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; color: var(--muted); }}
  .swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
  footer {{ padding: 1rem 2.5rem 2rem; color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<header>
  <h1>RLX-FFT E2E Validation Report</h1>
  <div class="meta">
    <span>generated {generated}</span>
    <span>n_fft={n_fft}</span>
    <span>n_mels={n_mels}</span>
    <span>bench iters={iters}</span>
    <span>devices: {devices_footer}</span>
    <span>batches: {batches_footer}</span>
  </div>
</header>
<main>
  <div class="note">
    Compares reference (<code>rustfft_ref</code>), native RLX FFT (<code>rlx_op_fft</code>),
    compiled learned teacher (<code>learned_compiled</code>), and distilled deploy
    (<code>learned_distilled</code>) across mel, welch, q8, and denoise pipelines.
    Speed gate: distilled mel latency ≤ 1.05× <code>rlx_op_fft</code> and error ≤ 1.5× on GPU.
  </div>

  <section>
    <h2>Summary</h2>
    {summary_cards}
  </section>

  {distillation_section}

  {per_batch_section}

  <section>
    <h2>Speed gate (mel pipeline)</h2>
    {gate_table}
  </section>

  <div class="legend">{legend_html}</div>

  <section>
    <h2>Mel pipeline — latency &amp; precision</h2>
    {mel_chart_sections}
  </section>

  <section>
    <h2>Latency vs batch (mel)</h2>
    {line_sections}
  </section>

  <section>
    <h2>Precision vs speed (mel scatter)</h2>
    {scatter_section}
  </section>

  <section>
    <h2>Heatmaps — distilled vs RLX FFT (mel)</h2>
    <p style="color:var(--muted);font-size:0.9rem;margin-top:0">
      Ratio &lt; 1 means distilled is faster than <code>rlx_op_fft</code>; error ratio &lt; 1 means lower max error.
    </p>
    {heatmap_sections}
  </section>

  {other_pipeline_sections}

  <section>
    <h2>Full results table</h2>
    {table_html}
  </section>
</main>
<footer>rlx-fft bench-e2e · total bench time {elapsed:.1} ms</footer>

<script>
const REPORT = {json};
const CHART_BACKENDS = {chart_backends_json};
const COLORS = {colors_json};

function lookup(pipeline, device, batch, backend) {{
  return REPORT.rows.find(r =>
    r.pipeline === pipeline && r.device === device &&
    r.batch === batch && r.backend === backend
  ) || null;
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

const PLOTLY_LAYOUT = {{
  paper_bgcolor: '#1a2332', plot_bgcolor: '#1a2332',
  font: {{ color: '#e6edf3', size: 11 }},
  margin: {{ l: 60, r: 20, t: 40, b: 50 }},
}};

function plotRatioHeatmap(el, title, z, x, y, zmid) {{
  const data = [{{
    z, x, y, type: 'heatmap', colorscale: 'RdYlGn_r', zmid,
    hovertemplate: 'device=%{{y}} batch=%{{x}}<br>ratio %{{z:.3f}}<extra></extra>',
    colorbar: {{ tickfont: {{ color: '#8b949e' }}, title: 'ratio' }},
  }}];
  Plotly.newPlot(el, data, {{
    ...PLOTLY_LAYOUT,
    title: {{ text: title, font: {{ size: 13, color: '#8b949e' }} }},
    xaxis: {{ title: 'batch' }},
    yaxis: {{ title: 'device' }},
  }}, {{ responsive: true, displayModeBar: true }});
}}

document.querySelectorAll('[data-plotly]').forEach(el => {{
  const spec = JSON.parse(el.dataset.plotly);
  plotRatioHeatmap(el, spec.title, spec.z, spec.batches, spec.devices, spec.zmid);
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
      if (window.chartRegistry) {{
        panels[i].querySelectorAll('canvas').forEach(c => {{
          const ch = window.chartRegistry[c.id];
          if (ch) ch.resize();
        }});
      }}
      panels[i].querySelectorAll('[data-plotly]').forEach(el => Plotly.Plots.resize(el));
    }});
  }});
}});

window.chartRegistry = {{}};

function makeGroupedBar(canvasId, labels, datasets, yLabel, logY) {{
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (window.chartRegistry[canvasId]) window.chartRegistry[canvasId].destroy();
  window.chartRegistry[canvasId] = new Chart(ctx, {{
    type: 'bar',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
        y: {{
          type: logY ? 'logarithmic' : 'linear',
          title: {{ display: true, text: yLabel, color: '#8b949e' }},
          ticks: {{ color: '#8b949e' }},
          grid: {{ color: '#30363d' }},
        }},
      }},
    }},
  }});
}}

function makeLineChart(canvasId, labels, datasets, yLabel) {{
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (window.chartRegistry[canvasId]) window.chartRegistry[canvasId].destroy();
  window.chartRegistry[canvasId] = new Chart(ctx, {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'batch', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
        y: {{
          type: 'logarithmic',
          title: {{ display: true, text: yLabel, color: '#8b949e' }},
          ticks: {{ color: '#8b949e' }},
          grid: {{ color: '#30363d' }},
        }},
      }},
    }},
  }});
}}

function makeScatter(canvasId, datasets) {{
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (window.chartRegistry[canvasId]) window.chartRegistry[canvasId].destroy();
  window.chartRegistry[canvasId] = new Chart(ctx, {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: '#8b949e' }} }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => `${{ctx.dataset.label}}: ${{fmtMs(ctx.parsed.x)}} ms, err ${{fmtErr(ctx.parsed.y)}}`,
          }},
        }},
      }},
      scales: {{
        x: {{
          type: 'logarithmic',
          title: {{ display: true, text: 'latency (ms)', color: '#8b949e' }},
          ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }},
        }},
        y: {{
          type: 'logarithmic',
          title: {{ display: true, text: 'max_err', color: '#8b949e' }},
          ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }},
        }},
      }},
    }},
  }});
}}

{chart_init_js}
</script>
</body>
</html>"##,
        generated = generated,
        n_fft = n_fft,
        n_mels = report.n_mels,
        iters = report.iters,
        devices_footer = devices.join(", "),
        batches_footer = batches
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        summary_cards = summary_cards,
        distillation_section = distillation_section,
        per_batch_section = per_batch_section,
        gate_table = gate_table,
        mel_chart_sections = mel_chart_sections,
        line_sections = line_sections,
        scatter_section = scatter_section,
        heatmap_sections = heatmap_sections,
        other_pipeline_sections = if other_pipeline_sections.is_empty() {
            String::new()
        } else {
            format!("<section><h2>Other pipelines</h2>{other_pipeline_sections}</section>")
        },
        table_html = table_html,
        legend_html = legend_html,
        elapsed = report.elapsed_ms,
        json = json,
        chart_backends_json = chart_backends_json,
        colors_json = colors_json,
        chart_init_js = build_chart_init_js(report, &devices, &batches, &chart_backends),
    )
}

fn build_summary_cards(
    report: &E2eBenchReport,
    n_fft: usize,
    devices: &[String],
    batches: &[usize],
) -> String {
    let meta = &report.meta;
    let train_steps = meta
        .train_steps
        .or_else(|| meta.teacher.as_ref().map(|t| t.steps))
        .map(|s| s.to_string())
        .unwrap_or_else(|| "—".into());
    let distill_steps = meta
        .distill_steps
        .or_else(|| meta.distill.as_ref().map(|d| d.steps))
        .map(|s| s.to_string())
        .unwrap_or_else(|| "—".into());
    let teacher_mel = meta
        .teacher
        .as_ref()
        .map(|t| format!("{:.2e}", t.final_mel_max_err))
        .unwrap_or_else(|| "—".into());
    let distill_mel_teacher = meta
        .distill
        .as_ref()
        .map(|d| format!("{:.2e}", d.final_mel_err_vs_teacher))
        .unwrap_or_else(|| "—".into());
    let distill_mel_ref = meta
        .distill
        .as_ref()
        .map(|d| format!("{:.2e}", d.final_mel_err_vs_ref))
        .unwrap_or_else(|| "—".into());

    let mut best_distilled: Option<&E2eBenchRow> = None;
    for row in &report.rows {
        if row.pipeline == "mel"
            && row.backend == "learned_distilled"
            && best_distilled.map(|b| row.ms < b.ms).unwrap_or(true)
        {
            best_distilled = Some(row);
        }
    }
    let best_speed = best_distilled
        .map(|r| format!("{:.4} ms ({}/{})", r.ms, r.device, r.batch))
        .unwrap_or_else(|| "—".into());

    format!(
        r#"<div class="cards">
  <div class="card-stat"><div class="label">n_fft</div><div class="value">{n_fft}</div><div class="sub">transform size</div></div>
  <div class="card-stat"><div class="label">Train iterations</div><div class="value">{train_steps}</div><div class="sub">teacher butterfly</div></div>
  <div class="card-stat"><div class="label">Distill iterations</div><div class="value">{distill_steps}</div><div class="sub">student deploy</div></div>
  <div class="card-stat"><div class="label">Teacher mel err</div><div class="value">{teacher_mel}</div><div class="sub">vs rustfft ref</div></div>
  <div class="card-stat"><div class="label">Distill vs teacher</div><div class="value">{distill_mel_teacher}</div><div class="sub">mel max err</div></div>
  <div class="card-stat"><div class="label">Distill vs ref</div><div class="value">{distill_mel_ref}</div><div class="sub">mel max err</div></div>
  <div class="card-stat"><div class="label">Fastest distilled</div><div class="value">{best_speed}</div><div class="sub">{device_count} devices · {batch_count} batches</div></div>
  <div class="card-stat"><div class="label">Bench iters</div><div class="value">{iters}</div><div class="sub">per backend timing</div></div>
</div>"#,
        n_fft = n_fft,
        train_steps = train_steps,
        distill_steps = distill_steps,
        teacher_mel = teacher_mel,
        distill_mel_teacher = distill_mel_teacher,
        distill_mel_ref = distill_mel_ref,
        best_speed = best_speed,
        device_count = devices.len(),
        batch_count = batches.len(),
        iters = report.iters,
    )
}

fn build_per_batch_train_section(report: &E2eBenchReport) -> String {
    if report.meta.per_batch.is_empty() {
        return String::new();
    }
    let batches: Vec<usize> = report.meta.per_batch.iter().map(|p| p.batch).collect();
    let teacher_mel: Vec<Option<f64>> = report
        .meta
        .per_batch
        .iter()
        .map(|p| p.teacher.as_ref().map(|t| t.final_mel_max_err as f64))
        .collect();
    let teacher_welch: Vec<Option<f64>> = report
        .meta
        .per_batch
        .iter()
        .map(|p| p.teacher.as_ref().map(|t| t.final_welch_max_err as f64))
        .collect();
    let distill_teacher: Vec<Option<f64>> = report
        .meta
        .per_batch
        .iter()
        .map(|p| {
            p.distill
                .as_ref()
                .map(|d| d.final_mel_err_vs_teacher as f64)
        })
        .collect();
    let distill_ref: Vec<Option<f64>> = report
        .meta
        .per_batch
        .iter()
        .map(|p| p.distill.as_ref().map(|d| d.final_mel_err_vs_ref as f64))
        .collect();
    let train_ms: Vec<Option<f64>> = report
        .meta
        .per_batch
        .iter()
        .map(|p| p.teacher.as_ref().map(|t| t.elapsed_ms))
        .collect();
    let distill_ms: Vec<Option<f64>> = report
        .meta
        .per_batch
        .iter()
        .map(|p| p.distill.as_ref().map(|d| d.elapsed_ms))
        .collect();

    let batch_labels = serde_json::to_string(&batches).unwrap_or_else(|_| "[]".into());
    let table_rows: String = report
        .meta
        .per_batch
        .iter()
        .map(|p| {
            let t_mel = p
                .teacher
                .as_ref()
                .map(|t| format!("{:.3e}", t.final_mel_max_err))
                .unwrap_or_else(|| "—".into());
            let d_teacher = p
                .distill
                .as_ref()
                .map(|d| format!("{:.3e}", d.final_mel_err_vs_teacher))
                .unwrap_or_else(|| "—".into());
            let d_ref = p
                .distill
                .as_ref()
                .map(|d| format!("{:.3e}", d.final_mel_err_vs_ref))
                .unwrap_or_else(|| "—".into());
            let t_ms = p
                .teacher
                .as_ref()
                .map(|t| format!("{:.0}", t.elapsed_ms))
                .unwrap_or_else(|| "—".into());
            let d_ms = p
                .distill
                .as_ref()
                .map(|d| format!("{:.0}", d.elapsed_ms))
                .unwrap_or_else(|| "—".into());
            format!(
                "<tr><td class=\"left\">{}</td><td>{t_mel}</td><td>{d_teacher}</td><td>{d_ref}</td><td>{t_ms}</td><td>{d_ms}</td></tr>",
                p.batch
            )
        })
        .collect();

    format!(
        r#"<section>
  <h2>Training &amp; distillation vs batch</h2>
  <div class="grid">
    <div class="card"><h4>Max mel error vs batch</h4><canvas id="train-err-batch"></canvas></div>
    <div class="card"><h4>Train / distill time vs batch</h4><canvas id="train-ms-batch"></canvas></div>
  </div>
  <h3>Per-batch summary</h3>
  <table>
    <thead><tr>
      <th class="left">batch</th><th>teacher mel err</th><th>distill vs teacher</th>
      <th>distill vs ref</th><th>train ms</th><th>distill ms</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</section>
<script>
(function() {{
  const labels = {batch_labels};
  const errDatasets = [
    {{ label: 'teacher vs ref (mel)', data: {teacher_mel}, borderColor: '#e15759', backgroundColor: '#e15759', fill: false, tension: 0.15 }},
    {{ label: 'distill vs teacher (mel)', data: {distill_teacher}, borderColor: '#ff7f0e', backgroundColor: '#ff7f0e', fill: false, tension: 0.15 }},
    {{ label: 'distill vs ref (mel)', data: {distill_ref}, borderColor: '#59a14f', backgroundColor: '#59a14f', fill: false, tension: 0.15 }},
    {{ label: 'teacher vs ref (welch)', data: {teacher_welch}, borderColor: '#76b7b2', backgroundColor: '#76b7b2', fill: false, tension: 0.15, borderDash: [4,4] }},
  ];
  const msDatasets = [
    {{ label: 'train ms', data: {train_ms}, borderColor: '#f28e2b', backgroundColor: '#f28e2b', fill: false, tension: 0.15 }},
    {{ label: 'distill ms', data: {distill_ms}, borderColor: '#4e79a7', backgroundColor: '#4e79a7', fill: false, tension: 0.15 }},
  ];
  window.chartRegistry = window.chartRegistry || {{}};
  function line(id, datasets, yLabel, logY) {{
    const ctx = document.getElementById(id);
    if (!ctx) return;
    if (window.chartRegistry[id]) window.chartRegistry[id].destroy();
    window.chartRegistry[id] = new Chart(ctx, {{
      type: 'line',
      data: {{ labels, datasets }},
      options: {{
        responsive: true,
        scales: {{
          x: {{ type: 'logarithmic', title: {{ display: true, text: 'batch', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
          y: {{ type: logY ? 'logarithmic' : 'linear', title: {{ display: true, text: yLabel, color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
        }},
      }},
    }});
  }}
  line('train-err-batch', errDatasets, 'max_err', true);
  line('train-ms-batch', msDatasets, 'ms', true);
}})();
</script>"#,
        batch_labels = batch_labels,
        teacher_mel = serde_json::to_string(&teacher_mel).unwrap_or_else(|_| "[]".into()),
        teacher_welch = serde_json::to_string(&teacher_welch).unwrap_or_else(|_| "[]".into()),
        distill_teacher = serde_json::to_string(&distill_teacher).unwrap_or_else(|_| "[]".into()),
        distill_ref = serde_json::to_string(&distill_ref).unwrap_or_else(|_| "[]".into()),
        train_ms = serde_json::to_string(&train_ms).unwrap_or_else(|_| "[]".into()),
        distill_ms = serde_json::to_string(&distill_ms).unwrap_or_else(|_| "[]".into()),
        table_rows = table_rows,
    )
}

fn build_distillation_section(report: &E2eBenchReport) -> String {
    let Some(teacher) = report.meta.teacher.as_ref() else {
        return String::new();
    };
    let distill = report.meta.distill.as_ref();
    let distill_mel_teacher = distill.map(|d| d.final_mel_err_vs_teacher);
    let distill_welch_teacher = distill.map(|d| d.final_welch_err_vs_teacher);
    let distill_mel_ref = distill.map(|d| d.final_mel_err_vs_ref);

    let labels = "[\"Teacher vs ref (mel)\", \"Teacher vs ref (welch)\", \"Distill vs teacher (mel)\", \"Distill vs teacher (welch)\", \"Distill vs ref (mel)\"]";
    let data = format!(
        "[{:.4e}, {:.4e}, {}, {}, {}]",
        teacher.final_mel_max_err,
        teacher.final_welch_max_err,
        distill_mel_teacher
            .map(|v| format!("{v:.4e}"))
            .unwrap_or_else(|| "null".into()),
        distill_welch_teacher
            .map(|v| format!("{v:.4e}"))
            .unwrap_or_else(|| "null".into()),
        distill_mel_ref
            .map(|v| format!("{v:.4e}"))
            .unwrap_or_else(|| "null".into()),
    );

    format!(
        r#"<section>
  <h2>Training &amp; distillation precision</h2>
  <div class="grid">
    <div class="card">
      <h4>Max error after training / distillation</h4>
      <canvas id="distill-bar"></canvas>
    </div>
    <div class="card">
      <h4>Teacher training stats</h4>
      <table>
        <tr><td class="left">Spectrum err</td><td>{spec:.3e}</td></tr>
        <tr><td class="left">Mel err</td><td>{mel:.3e}</td></tr>
        <tr><td class="left">Welch err</td><td>{welch:.3e}</td></tr>
        <tr><td class="left">Mean gate</td><td>{gate:.3}</td></tr>
        <tr><td class="left">Active gates</td><td>{active} / {active_hard} hard</td></tr>
        <tr><td class="left">Q8</td><td>{q8}</td></tr>
        <tr><td class="left">Train time</td><td>{train_ms:.0} ms</td></tr>
      </table>
    </div>
  </div>
</section>
<script>
(function() {{
  const labels = {labels};
  const vals = {data};
  const ctx = document.getElementById('distill-bar');
  if (!ctx) return;
  window.chartRegistry = window.chartRegistry || {{}};
  window.chartRegistry['distill-bar'] = new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'max_err',
        data: vals,
        backgroundColor: ['#e15759','#76b7b2','#ff7f0e','#edc948','#59a14f'],
      }}],
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ type: 'logarithmic', title: {{ display: true, text: 'max_err', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
        y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
      }},
    }},
  }});
}})();
</script>"#,
        spec = teacher.final_spectrum_max_err,
        mel = teacher.final_mel_max_err,
        welch = teacher.final_welch_max_err,
        gate = teacher.mean_gate,
        active = teacher.active_gates,
        active_hard = teacher.active_gates_hard,
        q8 = if teacher.q8_enabled { "yes" } else { "no" },
        train_ms = teacher.elapsed_ms,
        labels = labels,
        data = data,
    )
}

fn build_gate_table(report: &E2eBenchReport) -> String {
    let mut groups: BTreeSet<(String, usize)> = BTreeSet::new();
    for row in &report.rows {
        if row.pipeline == "mel" {
            groups.insert((row.device.clone(), row.batch));
        }
    }
    let mut rows_html = String::new();
    for (dev, batch) in groups {
        let subset: Vec<_> = report
            .rows
            .iter()
            .filter(|r| r.pipeline == "mel" && r.device == dev && r.batch == batch)
            .collect();
        let rlx = subset.iter().find(|r| r.backend == "rlx_op_fft");
        let dist = subset.iter().find(|r| r.backend == "learned_distilled");
        let compiled = subset.iter().find(|r| r.backend == "learned_compiled");
        if let (Some(r), Some(d)) = (rlx, dist) {
            let speed_ok = d.ms <= r.ms * 1.05;
            let acc_ok = d.max_err <= r.max_err * 1.5 + 1e-3;
            let status = if speed_ok && acc_ok {
                ("PASS", "pass")
            } else if speed_ok {
                ("PASS_SPEED", "warn")
            } else if acc_ok {
                ("PASS_ACC", "warn")
            } else {
                ("FAIL", "fail")
            };
            let speed_ratio = d.ms / r.ms;
            let err_ratio = if r.max_err > 0.0 {
                d.max_err / r.max_err
            } else {
                0.0
            };
            let compiled_ms = compiled
                .map(|c| format!("{:.4}", c.ms))
                .unwrap_or_else(|| "—".into());
            rows_html.push_str(&format!(
                "<tr><td class=\"left\">{dev}</td><td>{batch}</td><td>{rlx_ms:.4}</td><td>{dist_ms:.4}</td><td>{compiled_ms}</td><td>{speed_ratio:.3}</td><td>{rlx_err:.3e}</td><td>{dist_err:.3e}</td><td>{err_ratio:.3}</td><td class=\"{cls}\">{status}</td></tr>",
                dev = dev,
                batch = batch,
                rlx_ms = r.ms,
                dist_ms = d.ms,
                compiled_ms = compiled_ms,
                speed_ratio = speed_ratio,
                rlx_err = r.max_err,
                dist_err = d.max_err,
                err_ratio = err_ratio,
                cls = status.1,
                status = status.0,
            ));
        }
    }
    format!(
        r#"<table>
<thead><tr>
  <th class="left">device</th><th>batch</th><th>rlx ms</th><th>distilled ms</th><th>compiled ms</th>
  <th>speed ratio</th><th>rlx err</th><th>dist err</th><th>err ratio</th><th>gate</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"#
    )
}

fn build_pipeline_chart_sections(
    _report: &E2eBenchReport,
    pipeline: &str,
    devices: &[String],
    batches: &[usize],
    chart_backends: &[String],
) -> String {
    if devices.is_empty() || batches.is_empty() || chart_backends.is_empty() {
        return String::new();
    }
    let slug = pipeline.replace('-', "_");
    let tabs: String = devices
        .iter()
        .enumerate()
        .map(|(i, d)| {
            format!(
                r#"<button class="tab{active}" data-tab="{slug}-{i}">{d}</button>"#,
                active = if i == 0 { " active" } else { "" },
                slug = slug,
                d = d
            )
        })
        .collect();
    let panels: String = devices
        .iter()
        .enumerate()
        .map(|(i, d)| {
            format!(
                r#"<div class="panel{active}" id="panel-{slug}-{i}">
  <div class="grid">
    <div class="card"><h4>Latency (ms) — {pipeline} / {d}</h4><canvas id="lat-{slug}-{i}"></canvas></div>
    <div class="card"><h4>Max error — {pipeline} / {d}</h4><canvas id="err-{slug}-{i}"></canvas></div>
  </div>
</div>"#,
                active = if i == 0 { " active" } else { "" },
                slug = slug,
                i = i,
                pipeline = pipeline,
                d = d
            )
        })
        .collect();

    format!(
        r#"<h3>{pipeline}</h3>
<div class="tab-group" id="tg-{slug}">
  <div class="tabs">{tabs}</div>
  {panels}
</div>"#,
        pipeline = pipeline,
        slug = slug,
        tabs = tabs,
        panels = panels,
    )
}

fn build_heatmap_sections(
    report: &E2eBenchReport,
    devices: &[String],
    batches: &[usize],
) -> String {
    if devices.is_empty() || batches.is_empty() {
        return String::new();
    }
    let speed_z: Vec<Vec<f64>> = devices
        .iter()
        .map(|dev| {
            batches
                .iter()
                .map(|batch| {
                    let rlx = lookup_row(report, "mel", dev, *batch, "rlx_op_fft");
                    let dist = lookup_row(report, "mel", dev, *batch, "learned_distilled");
                    match (rlx, dist) {
                        (Some(r), Some(d)) if r.ms > 0.0 => Some(d.ms / r.ms),
                        _ => None,
                    }
                })
                .map(|v| v.unwrap_or(f64::NAN))
                .collect()
        })
        .collect();
    let err_z: Vec<Vec<f64>> = devices
        .iter()
        .map(|dev| {
            batches
                .iter()
                .map(|batch| {
                    let rlx = lookup_row(report, "mel", dev, *batch, "rlx_op_fft");
                    let dist = lookup_row(report, "mel", dev, *batch, "learned_distilled");
                    match (rlx, dist) {
                        (Some(r), Some(d)) if r.max_err > 0.0 => {
                            Some(d.max_err as f64 / r.max_err as f64)
                        }
                        (Some(_), Some(d)) => Some(d.max_err as f64),
                        _ => None,
                    }
                })
                .map(|v| v.unwrap_or(f64::NAN))
                .collect()
        })
        .collect();

    let speed_spec = serde_json::json!({
        "title": "Speed ratio (distilled / rlx_op_fft)",
        "devices": devices,
        "batches": batches,
        "z": speed_z,
        "zmid": 1.0,
    });
    let err_spec = serde_json::json!({
        "title": "Error ratio (distilled / rlx_op_fft)",
        "devices": devices,
        "batches": batches,
        "z": err_z,
        "zmid": 1.0,
    });

    format!(
        r#"<div class="grid">
  <div class="card"><div class="plotly" data-plotly='{speed_spec}'></div></div>
  <div class="card"><div class="plotly" data-plotly='{err_spec}'></div></div>
</div>"#,
        speed_spec = speed_spec.to_string().replace('\'', "&#39;"),
        err_spec = err_spec.to_string().replace('\'', "&#39;"),
    )
}

fn build_scatter_section(devices: &[String]) -> String {
    let tabs: String = devices
        .iter()
        .enumerate()
        .map(|(i, d)| {
            format!(
                r#"<button class="tab{active}" data-tab="scatter-{i}">{d}</button>"#,
                active = if i == 0 { " active" } else { "" },
                d = d
            )
        })
        .collect();
    let panels: String = devices
        .iter()
        .enumerate()
        .map(|(i, d)| {
            format!(
                r#"<div class="panel{active}" id="panel-scatter-{i}">
  <div class="card"><h4>Latency vs max_err — mel / {d}</h4><canvas id="scatter-{i}"></canvas></div>
</div>"#,
                active = if i == 0 { " active" } else { "" },
                d = d,
                i = i
            )
        })
        .collect();
    format!(
        r#"<div class="tab-group" id="tg-scatter">
  <div class="tabs">{tabs}</div>
  {panels}
</div>"#
    )
}

fn build_line_sections(devices: &[String], batches: &[usize]) -> String {
    if devices.is_empty() || batches.is_empty() {
        return String::new();
    }
    let tabs: String = devices
        .iter()
        .enumerate()
        .map(|(i, d)| {
            format!(
                r#"<button class="tab{active}" data-tab="line-{i}">{d}</button>"#,
                active = if i == 0 { " active" } else { "" },
                d = d
            )
        })
        .collect();
    let panels: String = devices
        .iter()
        .enumerate()
        .map(|(i, d)| {
            format!(
                r#"<div class="panel{active}" id="panel-line-{i}">
  <div class="card"><h4>Latency vs batch — mel / {d}</h4><canvas id="line-{i}"></canvas></div>
</div>"#,
                active = if i == 0 { " active" } else { "" },
                d = d,
                i = i
            )
        })
        .collect();
    format!(
        r#"<div class="tab-group" id="tg-line">
  <div class="tabs">{tabs}</div>
  {panels}
</div>"#
    )
}

fn build_full_table(report: &E2eBenchReport) -> String {
    let mut rows: Vec<_> = report.rows.iter().collect();
    rows.sort_by(|a, b| {
        (&a.pipeline, &a.device, a.batch, &a.backend).cmp(&(
            &b.pipeline,
            &b.device,
            b.batch,
            &b.backend,
        ))
    });
    let body: String = rows
        .iter()
        .map(|r| {
            format!(
                "<tr><td class=\"left\">{}</td><td class=\"left\">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.4}</td><td>{:.3e}</td><td>{}</td></tr>",
                r.pipeline,
                r.backend,
                r.device,
                r.batch,
                r.n_fft,
                r.ms,
                r.max_err,
                r.mean_gate
                    .map(|g| format!("{g:.3}"))
                    .unwrap_or_else(|| "—".into()),
            )
        })
        .collect();
    format!(
        r#"<table>
<thead><tr>
  <th class="left">pipeline</th><th class="left">backend</th><th class="left">device</th>
  <th>batch</th><th>n_fft</th><th>ms</th><th>max_err</th><th>mean_gate</th>
</tr></thead>
<tbody>{body}</tbody>
</table>"#
    )
}

fn build_legend(backends: &[String]) -> String {
    backends
        .iter()
        .map(|b| {
            let color = backend_color(b);
            format!(
                r#"<span class="legend-item"><span class="swatch" style="background:{color}"></span>{b}</span>"#
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_chart_init_js(
    report: &E2eBenchReport,
    devices: &[String],
    batches: &[usize],
    chart_backends: &[String],
) -> String {
    let batch_labels = serde_json::to_string(batches).unwrap_or_else(|_| "[]".into());
    let mut js = String::new();

    for (di, dev) in devices.iter().enumerate() {
        let lat_datasets =
            chart_datasets_json(report, "mel", dev, batches, chart_backends, |r| r.ms);
        let err_datasets = chart_datasets_json(report, "mel", dev, batches, chart_backends, |r| {
            r.max_err as f64
        });
        js.push_str(&format!(
            "makeGroupedBar('lat-mel-{di}', {batch_labels}, {lat_datasets}, 'ms', true);\n",
        ));
        js.push_str(&format!(
            "makeGroupedBar('err-mel-{di}', {batch_labels}, {err_datasets}, 'max_err', true);\n",
        ));

        let line_datasets = line_datasets_json(report, "mel", dev, batches, chart_backends);
        js.push_str(&format!(
            "makeLineChart('line-{di}', {batch_labels}, {line_datasets}, 'ms');\n",
        ));

        let scatter_datasets = scatter_datasets_json(report, "mel", dev, chart_backends);
        js.push_str(&format!(
            "makeScatter('scatter-{di}', {scatter_datasets});\n",
        ));
    }

    let pipelines: BTreeSet<String> = report.rows.iter().map(|r| r.pipeline.clone()).collect();
    for pipeline in pipelines {
        if pipeline == "mel" {
            continue;
        }
        let slug = pipeline.replace('-', "_");
        for (di, dev) in devices.iter().enumerate() {
            let lat_datasets =
                chart_datasets_json(report, &pipeline, dev, batches, chart_backends, |r| r.ms);
            let err_datasets =
                chart_datasets_json(report, &pipeline, dev, batches, chart_backends, |r| {
                    r.max_err as f64
                });
            js.push_str(&format!(
                "makeGroupedBar('lat-{slug}-{di}', {batch_labels}, {lat_datasets}, 'ms', true);\n",
            ));
            js.push_str(&format!(
                "makeGroupedBar('err-{slug}-{di}', {batch_labels}, {err_datasets}, 'max_err', true);\n",
            ));
        }
    }

    js
}

fn chart_datasets_json<F>(
    report: &E2eBenchReport,
    pipeline: &str,
    device: &str,
    batches: &[usize],
    backends: &[String],
    value: F,
) -> String
where
    F: Fn(&E2eBenchRow) -> f64,
{
    let datasets: Vec<serde_json::Value> = backends
        .iter()
        .map(|backend| {
            let data: Vec<Option<f64>> = batches
                .iter()
                .map(|batch| lookup_row(report, pipeline, device, *batch, backend).map(&value))
                .collect();
            serde_json::json!({
                "label": backend,
                "data": data,
                "backgroundColor": backend_color(backend),
                "borderColor": backend_color(backend),
            })
        })
        .collect();
    serde_json::to_string(&datasets).unwrap_or_else(|_| "[]".into())
}

fn line_datasets_json(
    report: &E2eBenchReport,
    pipeline: &str,
    device: &str,
    batches: &[usize],
    backends: &[String],
) -> String {
    let datasets: Vec<serde_json::Value> = backends
        .iter()
        .map(|backend| {
            let data: Vec<Option<f64>> = batches
                .iter()
                .map(|batch| lookup_row(report, pipeline, device, *batch, backend).map(|r| r.ms))
                .collect();
            serde_json::json!({
                "label": backend,
                "data": data,
                "borderColor": backend_color(backend),
                "backgroundColor": backend_color(backend),
                "fill": false,
                "tension": 0.15,
            })
        })
        .collect();
    serde_json::to_string(&datasets).unwrap_or_else(|_| "[]".into())
}

fn scatter_datasets_json(
    report: &E2eBenchReport,
    pipeline: &str,
    device: &str,
    backends: &[String],
) -> String {
    let datasets: Vec<serde_json::Value> = backends
        .iter()
        .map(|backend| {
            let points: Vec<[f64; 2]> = report
                .rows
                .iter()
                .filter(|r| r.pipeline == pipeline && r.device == device && r.backend == *backend)
                .map(|r| [r.ms, r.max_err as f64])
                .collect();
            serde_json::json!({
                "label": backend,
                "data": points,
                "backgroundColor": backend_color(backend),
                "pointRadius": 6,
            })
        })
        .collect();
    serde_json::to_string(&datasets).unwrap_or_else(|_| "[]".into())
}

fn lookup_row<'a>(
    report: &'a E2eBenchReport,
    pipeline: &str,
    device: &str,
    batch: usize,
    backend: &str,
) -> Option<&'a E2eBenchRow> {
    report.rows.iter().find(|r| {
        r.pipeline == pipeline && r.device == device && r.batch == batch && r.backend == backend
    })
}

fn unique_strings(rows: &[E2eBenchRow], f: impl Fn(&E2eBenchRow) -> &String) -> Vec<String> {
    rows.iter()
        .map(f)
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn unique_usize(rows: &[E2eBenchRow], f: impl Fn(&E2eBenchRow) -> usize) -> Vec<usize> {
    rows.iter()
        .map(f)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn backend_color_map(backends: &[String]) -> std::collections::BTreeMap<String, String> {
    backends
        .iter()
        .map(|b| (b.clone(), backend_color(b).to_string()))
        .collect()
}

fn backend_color(name: &str) -> &'static str {
    BACKEND_COLORS
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, c)| *c)
        .unwrap_or("#8b949e")
}

fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix {secs}")
}

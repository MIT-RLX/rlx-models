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

//! HTML report for multi-n_fft training study (train regime × eval n_fft matrix).

use crate::train_multi::{MultiTrainReport, best_regime_per_eval};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::Path;

pub fn read_multi_train_json(path: &Path) -> Result<MultiTrainReport> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn write_multi_train_html(path: &Path, report: &MultiTrainReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, render_multi_train_html(report))
        .with_context(|| format!("write {}", path.display()))
}

pub fn render_multi_train_html(report: &MultiTrainReport) -> String {
    let json = serde_json::to_string(report).unwrap_or_else(|_| "{}".to_string());
    let regimes = unique_regimes(report);
    let table_html = build_matrix_table(report, &regimes);
    let winners_html = build_winners_html(report);
    let chart_specs = build_chart_specs(report, &regimes);

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RLX-FFT Multi-n_fft Training Study</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.27.0/plotly.min.js"></script>
<style>
  :root {{
    --bg: #0f1419; --panel: #1a2332; --text: #e6edf3; --muted: #8b949e;
    --border: #30363d; --accent: #58a6ff; --fast: #238636;
  }}
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5; }}
  header {{ padding: 2rem 2.5rem 1rem; border-bottom: 1px solid var(--border); }}
  h1 {{ margin: 0 0 0.5rem; font-size: 1.75rem; }}
  .meta {{ color: var(--muted); font-size: 0.95rem; }}
  .meta span {{ margin-right: 1.25rem; }}
  main {{ padding: 1.5rem 2.5rem 3rem; max-width: 1200px; margin: 0 auto; }}
  section {{ margin-bottom: 2rem; }}
  h2 {{ font-size: 1.2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
  .note {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem; color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 1rem 0; }}
  th, td {{ border: 1px solid var(--border); padding: 0.45rem 0.6rem; text-align: right; }}
  th {{ background: #21262d; color: var(--muted); }}
  th.left, td.left {{ text-align: left; }}
  td.best {{ outline: 2px solid var(--fast); font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1rem; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }}
  .plotly {{ width: 100%; min-height: 380px; }}
  canvas {{ max-height: 300px; }}
</style>
</head>
<body>
<header>
  <h1>Multi-n_fft Training Study</h1>
  <div class="meta">
    <span>batch={batch}</span>
    <span>max_steps={max_steps}</span>
    <span>min_steps={min_steps}</span>
    <span>until_converged={until_converged}</span>
    <span>n_fft={n_ffts:?}</span>
    <span>eval_batches={eval_batches}</span>
    <span>elapsed {elapsed:.0} ms</span>
  </div>
</header>
<main>
  <div class="note">
    Compares training <strong>schedules</strong> (single size vs mixed round-robin / random / balanced)
    across FFT sizes. Each row is a training regime; columns are eval n_fft.
    Lower roundtrip max_err is better. Weights are size-specific — cross-size eval uses the
    twiddles trained for that eval n_fft within each regime.
  </div>
  {winners_html}
  <section>
    <h2>Roundtrip max error (train regime × eval n_fft)</h2>
    <div class="card"><div class="plotly" id="heatmap-rt"></div></div>
    {table_html}
  </section>
  <section>
    <h2>Encoder spectrum max error</h2>
    <div class="card"><div class="plotly" id="heatmap-enc"></div></div>
  </section>
  <section>
    <h2>Per eval size — regime comparison</h2>
    <div class="grid" id="charts"></div>
  </section>
</main>
<script>
const REPORT = {json};
const REGIMES = {regimes_json};
const N_FFTS = {n_ffts_json};
const CHART_SPECS = {chart_specs};

function cell(regime, evalN, field) {{
  const r = REPORT.rows.find(x => x.regime === regime && x.eval_n_fft === evalN);
  return r ? r[field] : null;
}}

function matrix(field) {{
  return REGIMES.map(regime => N_FFTS.map(n => cell(regime, n, field)));
}}

const layout = {{
  paper_bgcolor: '#1a2332', plot_bgcolor: '#1a2332',
  font: {{ color: '#e6edf3', size: 11 }}, margin: {{ l: 140, r: 20, t: 40, b: 50 }},
}};

function plotHeat(id, title, z, colorscale) {{
  Plotly.newPlot(id, [{{
    z, x: N_FFTS, y: REGIMES, type: 'heatmap',
    colorscale: colorscale || 'YlOrRd',
    hovertemplate: 'eval n=%{{x}}<br>%{{y}}<br>%{{z:.2e}}<extra></extra>',
  }}], {{ ...layout, title: {{ text: title, font: {{ size: 13, color: '#8b949e' }} }},
    xaxis: {{ title: 'eval n_fft' }}, yaxis: {{ title: 'train regime' }} }},
    {{ responsive: true }});
}}

plotHeat('heatmap-rt', 'roundtrip_max_err', matrix('roundtrip_max_err'), 'YlOrRd');
plotHeat('heatmap-enc', 'encoder_spectrum_max_err', matrix('encoder_spectrum_max_err'), 'Viridis');

const chartsEl = document.getElementById('charts');
CHART_SPECS.forEach(spec => {{
  const wrap = document.createElement('div');
  wrap.className = 'card';
  wrap.innerHTML = '<h4>eval n_fft=' + spec.eval_n + '</h4><canvas></canvas>';
  chartsEl.appendChild(wrap);
  const canvas = wrap.querySelector('canvas');
  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: spec.regimes,
      datasets: [
        {{ label: 'rt max err', data: spec.rt_max, backgroundColor: '#59a14f99' }},
        {{ label: 'enc max err', data: spec.enc_max, backgroundColor: '#4e79a799' }},
      ],
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', maxRotation: 45 }}, grid: {{ color: '#30363d' }} }},
        y: {{ type: 'logarithmic', ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
      }},
    }},
  }});
}});
</script>
</body>
</html>"##,
        batch = report.batch,
        max_steps = report.max_steps,
        min_steps = report.min_steps,
        until_converged = report.until_converged,
        n_ffts = report.n_ffts,
        eval_batches = report.eval_batches,
        elapsed = report.elapsed_ms,
        winners_html = winners_html,
        table_html = table_html,
        json = json,
        regimes_json = serde_json::to_string(&regimes).unwrap_or_else(|_| "[]".into()),
        n_ffts_json = serde_json::to_string(&report.n_ffts).unwrap_or_else(|_| "[]".into()),
        chart_specs = chart_specs,
    )
}

fn unique_regimes(report: &MultiTrainReport) -> Vec<String> {
    report
        .rows
        .iter()
        .map(|r| r.regime.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn build_winners_html(report: &MultiTrainReport) -> String {
    let winners = best_regime_per_eval(report);
    if winners.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "<section><h2>Best regime per eval n_fft</h2><table><thead><tr>\
         <th class=\"left\">eval n_fft</th><th class=\"left\">best regime</th><th>rt_max</th></tr></thead><tbody>",
    );
    for (n, regime, err) in winners {
        out.push_str(&format!(
            "<tr><td class=\"left\">{n}</td><td class=\"left\">{regime}</td><td>{err:.3e}</td></tr>"
        ));
    }
    out.push_str("</tbody></table></section>");
    out
}

fn build_matrix_table(report: &MultiTrainReport, regimes: &[String]) -> String {
    let mut out = String::from(
        "<table><thead><tr><th class=\"left\">regime</th><th class=\"left\">schedule</th>",
    );
    for &n in &report.n_ffts {
        out.push_str(&format!("<th>n={n} rt_max</th>"));
    }
    out.push_str("</tr></thead><tbody>");

    for regime in regimes {
        let schedule = report
            .rows
            .iter()
            .find(|r| r.regime == *regime)
            .map(|r| r.schedule.as_str())
            .unwrap_or("—");
        out.push_str(&format!(
            "<tr><td class=\"left\">{regime}</td><td class=\"left\">{schedule}</td>"
        ));
        let mut vals: Vec<f32> = Vec::new();
        for &n in &report.n_ffts {
            if let Some(r) = report
                .rows
                .iter()
                .find(|r| r.regime == *regime && r.eval_n_fft == n)
            {
                vals.push(r.roundtrip_max_err);
            }
        }
        let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
        for &n in &report.n_ffts {
            if let Some(r) = report
                .rows
                .iter()
                .find(|r| r.regime == *regime && r.eval_n_fft == n)
            {
                let cls = if (r.roundtrip_max_err - min).abs() < 1e-9 && vals.len() > 1 {
                    " class=\"best\""
                } else {
                    ""
                };
                out.push_str(&format!(
                    "<td{cls}>{err:.3e}</td>",
                    cls = cls,
                    err = r.roundtrip_max_err
                ));
            } else {
                out.push_str("<td>—</td>");
            }
        }
        out.push_str("</tr>");
    }
    out.push_str("</tbody></table>");
    out
}

fn build_chart_specs(report: &MultiTrainReport, regimes: &[String]) -> String {
    let mut specs = Vec::new();
    for &eval_n in &report.n_ffts {
        let mut rs = Vec::new();
        let mut rt = Vec::new();
        let mut enc = Vec::new();
        for regime in regimes {
            if let Some(r) = report
                .rows
                .iter()
                .find(|r| r.regime == *regime && r.eval_n_fft == eval_n)
            {
                rs.push(regime.clone());
                rt.push(r.roundtrip_max_err);
                enc.push(r.encoder_spectrum_max_err);
            }
        }
        specs.push(serde_json::json!({
            "eval_n": eval_n,
            "regimes": rs,
            "rt_max": rt,
            "enc_max": enc,
        }));
    }
    serde_json::to_string(&specs).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train_multi::MultiTrainEvalRow;

    #[test]
    fn render_multi_train_html_smoke() {
        let report = MultiTrainReport {
            batch: 8,
            n_ffts: vec![64, 128],
            eval_batches: 4,
            seed: 0,
            grad_clip: 1.0,
            project_twiddles: true,
            use_fused_train: true,
            optimizer: "adam".into(),
            elapsed_ms: 50.0,
            max_steps: 100,
            min_steps: 300,
            until_converged: false,
            rows: vec![MultiTrainEvalRow {
                regime: "single_64".into(),
                schedule: "single".into(),
                train_sizes: vec![64],
                eval_n_fft: 64,
                train_steps_total: 100,
                train_elapsed_ms: 10.0,
                encoder_spectrum_mse: 0.0,
                encoder_spectrum_max_err: 1e-5,
                decoder_time_mse: 0.0,
                decoder_time_max_err: 1e-5,
                roundtrip_mse: 0.0,
                roundtrip_max_err: 0.01,
                converged: true,
                final_holdout_mse: 0.01,
                checkpoint: None,
            }],
        };
        let html = render_multi_train_html(&report);
        assert!(html.contains("Multi-n_fft Training Study"));
        assert!(html.contains("single_64"));
    }
}

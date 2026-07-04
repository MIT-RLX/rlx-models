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

//! Full study HTML: ranked ablation configs, loss/error/memory charts, 3D loss maps, activation heatmaps.

use crate::ablation::{AblationReport, ablation_row_ok, ablation_winners, top5_variants_per_n_fft};
use crate::model::config::is_gpu_device_label;
use crate::study::telemetry::StudyTelemetryBundle;
use crate::train::multi::MultiTrainReport;
use anyhow::{Context, Result};
use std::path::Path;

fn json_for_script<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string(v)
        .unwrap_or_else(|_| "null".into())
        .replace("</", "<\\/")
}

#[derive(Debug, Clone, Default)]
pub struct FullStudyInputs {
    pub ablation: Option<AblationReport>,
    pub multi_train: Option<MultiTrainReport>,
    pub telemetry: Option<StudyTelemetryBundle>,
}

pub fn write_full_study_html(path: &Path, inputs: &FullStudyInputs) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, render_full_study_html(inputs))
        .with_context(|| format!("write {}", path.display()))
}

pub fn render_full_study_html(inputs: &FullStudyInputs) -> String {
    let ablation_json = inputs
        .ablation
        .as_ref()
        .map(json_for_script)
        .unwrap_or_else(|| "null".into());
    let train_json = inputs
        .multi_train
        .as_ref()
        .map(json_for_script)
        .unwrap_or_else(|| "null".into());
    let telemetry_json = inputs
        .telemetry
        .as_ref()
        .map(|t| {
            let mut sorted = t.clone();
            sorted.models.sort_by(|a, b| {
                a.final_mel_err
                    .partial_cmp(&b.final_mel_err)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a.final_spec_err
                            .partial_cmp(&b.final_spec_err)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
            json_for_script(&sorted)
        })
        .unwrap_or_else(|| "null".into());

    let ranked_table = build_ranked_table(inputs);
    let top5_section = build_top5_per_nfft_section(inputs);
    let limits_section = build_limits_section(inputs);
    let overview_cards = build_overview_cards(inputs);
    let model_panels = build_model_panels(inputs);
    let n_fft_meta = inputs
        .ablation
        .as_ref()
        .map(|a| {
            if a.n_ffts.is_empty() {
                let mut ns: Vec<usize> = a.rows.iter().map(|r| r.n_fft).collect();
                ns.sort_unstable();
                ns.dedup();
                ns
            } else {
                a.n_ffts.clone()
            }
        })
        .unwrap_or_default();
    let n_fft_meta_json = json_for_script(&n_fft_meta);

    format!(
        r##"<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RLX-FFT Full Study Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.27.0/plotly.min.js"></script>
<style>
:root {{ --bg:#0a0e14; --panel:#141b24; --text:#e6edf3; --muted:#8b949e; --border:#2d3640;
  --accent:#58a6ff; --fast:#3fb950; --gold:#d4a72c; --err:#f85149; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
header {{ padding:2rem 2rem 1rem; border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,#111820,var(--bg)); }}
h1 {{ margin:0 0 .5rem; font-size:1.8rem; }}
.meta {{ color:var(--muted); font-size:.92rem; }}
nav {{ display:flex; flex-wrap:wrap; gap:.5rem; padding:1rem 2rem 0; }}
nav button {{ background:var(--panel); border:1px solid var(--border); color:var(--text);
  padding:.45rem .9rem; border-radius:6px; cursor:pointer; font-size:.88rem; }}
nav button.active {{ border-color:var(--accent); color:var(--accent); }}
main {{ padding:1.5rem 2rem 3rem; max-width:1500px; margin:0 auto; }}
.panel {{ display:none; }} .panel.active {{ display:block; }}
section {{ margin-bottom:2rem; }}
h2 {{ font-size:1.2rem; border-bottom:1px solid var(--border); padding-bottom:.4rem; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:.85rem; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:1rem; }}
.card.gold {{ border-color:var(--gold); }}
.card h4 {{ margin:0 0 .3rem; font-size:.75rem; color:var(--muted); text-transform:uppercase; }}
.card .val {{ font-size:1.4rem; font-weight:600; }}
.card .sub {{ font-size:.82rem; color:var(--muted); }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(400px,1fr)); gap:1rem; }}
.plotly {{ width:100%; min-height:340px; }}
table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
th,td {{ border:1px solid var(--border); padding:.4rem .55rem; text-align:right; }}
th {{ background:#1c2430; color:var(--muted); position:sticky; top:0; }}
th.left,td.left {{ text-align:left; }}
tr.rank1 {{ background:rgba(63,185,80,.12); }}
tr.rank2 {{ background:rgba(88,166,255,.08); }}
tr.rank3 {{ background:rgba(212,167,44,.08); }}
tr.rank4 {{ background:rgba(177,143,255,.06); }}
tr.rank5 {{ background:rgba(255,166,87,.06); }}
.top5-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:1rem; }}
.top5-scroll {{ overflow-x:auto; padding-bottom:.5rem; }}
.top5-block h3 {{ margin:0 0 .5rem; font-size:1rem; color:var(--accent); }}
tr.fail {{ background:rgba(248,81,73,.08); }}
tr.limit {{ background:rgba(212,167,44,.1); }}
.model-block {{ border:1px solid var(--border); border-radius:10px; padding:1rem; margin-bottom:1.5rem; background:var(--panel); }}
.model-block h3 {{ margin:0 0 .75rem; color:var(--accent); }}
.note {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:1rem; color:var(--muted); font-size:.9rem; }}
</style></head><body>
<header>
  <h1>RLX-FFT Full Study Report</h1>
  <div class="meta">Ablation CSV → HTML · ranked by speed · capacity limits · loss/error/memory · training telemetry</div>
</header>
<nav id="nav">
  <button type="button" class="active" data-tab="ranked">Ranked configs</button>
  <button type="button" data-tab="limits">Capacity limits</button>
  <button type="button" data-tab="metrics">Loss / error / memory</button>
  <button type="button" data-tab="models">Trained models</button>
  <button type="button" data-tab="charts">All charts</button>
</nav>
<main>
  <div class="panel active" id="panel-ranked">
    <div class="note">Top 5 = fastest <strong>distinct variant</strong> per n_fft (best direction/device/batch for that variant). Full table below ranks every configuration by ms/iter.</div>
    {overview_cards}
    <section><h2>Top 5 variants per n_fft</h2><div class="top5-scroll">{top5_section}</div></section>
    <section><h2>All ranked ablation configurations</h2>{ranked_table}</section>
  </div>
  <div class="panel" id="panel-limits">
    <div class="note">Capacity study across n_fft = 64 … 131072 on CPU + GPU. Shows max working batch, pass/fail counts, and where compiled paths stop.</div>
    {limits_section}
    <section><h2>n_fft scaling (rustfft forward)</h2><div class="card"><div class="plotly" id="limit-scaling"></div></div></section>
    <section class="grid2">
      <div><h2>Coverage: ok ratio by device × n_fft</h2><div class="plotly" id="limit-coverage"></div></div>
      <div><h2>Max batch (rustfft forward)</h2><div class="plotly" id="limit-max-batch"></div></div>
    </section>
  </div>
  <div class="panel" id="panel-metrics">
    <section><h2>Speed vs accuracy (all configs)</h2><div class="card"><div class="plotly" id="scatter-ms-err"></div></div></section>
    <section class="grid2">
      <div><h2>Parameter memory by variant</h2><div class="card"><canvas id="chart-memory"></canvas></div></div>
      <div><h2>Parameter count by variant</h2><div class="card"><canvas id="chart-params"></canvas></div></div>
    </section>
    <section><h2>Max error distribution</h2><div class="card"><canvas id="chart-error"></canvas></div></section>
    <section><h2>Top 5 variants per n_fft (speed)</h2><div class="grid2" id="top5-charts"></div></section>
    <section><h2>All trained models — loss curves (best accuracy first)</h2><div class="card"><canvas id="chart-all-loss"></canvas></div></section>
    <section class="grid2" id="metric-heatmaps"></section>
  </div>
  <div class="panel" id="panel-models">
    <div class="note">Each trained model: gradient-descent loss curve, 3D loss landscape (twiddle slice), gate/layer activation heatmap.</div>
    {model_panels}
  </div>
  <div class="panel" id="panel-charts">
    <section class="grid2" id="speed-grid"></section>
    <section><h2>Training study (multi-n_fft)</h2><div class="plotly" id="train-heat"></div></section>
  </div>
</main>
<script>
const ABLATION = {ablation_json};
const TRAIN = {train_json};
const TELEMETRY = {telemetry_json};
const COLORS = {colors_json};
const N_FFT_SWEEP = {n_fft_meta_json};

function okRow(r) {{ return r.status === 'ok' && validMs(r); }}
function nFftList() {{
  if (N_FFT_SWEEP && N_FFT_SWEEP.length) return N_FFT_SWEEP.slice();
  return [...new Set(rows().map(r=>r.n_fft))].sort((a,b)=>a-b);
}}

function resizeCharts() {{
  if (window.Plotly) {{
    document.querySelectorAll('.plotly').forEach(el => {{
      if (el.id && el.offsetParent !== null) Plotly.Plots.resize(el);
    }});
  }}
  if (window.Chart && Chart.instances) {{
    Object.values(Chart.instances).forEach(c => c.resize());
  }}
}}
document.getElementById('nav').onclick = e => {{
  const b = e.target.closest('button[data-tab]'); if (!b) return;
  document.querySelectorAll('#nav button').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  document.getElementById('panel-' + b.dataset.tab).classList.add('active');
  setTimeout(resizeCharts, 60);
}};
window.addEventListener('resize', () => setTimeout(resizeCharts, 60));

function rows() {{ return (ABLATION && ABLATION.rows) ? ABLATION.rows.slice() : []; }}
function validMs(r) {{ return r.ms != null && !Number.isNaN(r.ms) && r.ms > 0; }}

// --- scatter ms vs max_err ---
(function() {{
  const pts = rows().filter(r => validMs(r) && r.max_err != null && isFinite(r.max_err));
  Plotly.newPlot('scatter-ms-err', [{{
    x: pts.map(r => r.ms), y: pts.map(r => r.max_err), text: pts.map(r => r.variant+' '+r.device+' n='+r.n_fft),
    mode:'markers', type:'scatter', marker:{{ size:8, color: pts.map(r => COLORS[r.variant]||'#888'), opacity:.85 }},
  }}], {{ paper_bgcolor:'#141b24', plot_bgcolor:'#141b24', font:{{color:'#e6edf3'}},
    xaxis:{{ title:'ms/iter', type:'log' }}, yaxis:{{ title:'max_err', type:'log' }},
    title:{{ text:'Speed vs precision', font:{{color:'#8b949e'}} }} }}, {{responsive:true}});
}})();

// --- memory + param count charts ---
(function() {{
  const byVarMem = {{}}, byVarParams = {{}};
  rows().forEach(r => {{
    if (!byVarMem[r.variant]) byVarMem[r.variant] = r.memory_bytes||0;
    if (!byVarParams[r.variant]) byVarParams[r.variant] = r.param_count||0;
  }});
  const labelsMem = Object.keys(byVarMem).sort((a,b) => byVarMem[b]-byVarMem[a]);
  new Chart(document.getElementById('chart-memory'), {{
    type:'bar',
    data:{{ labels: labelsMem, datasets:[{{ label:'bytes (params×4)', data: labelsMem.map(l=>byVarMem[l]),
      backgroundColor: labelsMem.map(l=>(COLORS[l]||'#666')+'bb') }}] }},
    options:{{ indexAxis:'y', plugins:{{ legend:{{display:false}} }},
      scales:{{ x:{{ ticks:{{color:'#8b949e'}}, grid:{{color:'#2d3640'}} }},
        y:{{ ticks:{{color:'#8b949e', font:{{size:9}}}}, grid:{{display:false}} }} }}
  }});
  const labelsParams = Object.keys(byVarParams).sort((a,b) => byVarParams[b]-byVarParams[a]);
  new Chart(document.getElementById('chart-params'), {{
    type:'bar',
    data:{{ labels: labelsParams, datasets:[{{ label:'param count', data: labelsParams.map(l=>byVarParams[l]),
      backgroundColor: labelsParams.map(l=>(COLORS[l]||'#666')+'bb') }}] }},
    options:{{ indexAxis:'y', plugins:{{ legend:{{display:false}} }},
      scales:{{ x:{{ ticks:{{color:'#8b949e'}}, grid:{{color:'#2d3640'}} }},
        y:{{ ticks:{{color:'#8b949e', font:{{size:9}}}}, grid:{{display:false}} }} }}
  }});
}})();

// --- error by variant ---
(function() {{
  const byVar = {{}};
  rows().filter(r => r.max_err != null && isFinite(r.max_err)).forEach(r => {{
    if (!byVar[r.variant]) byVar[r.variant] = [];
    byVar[r.variant].push(r.max_err);
  }});
  const labels = Object.keys(byVar).sort();
  const med = labels.map(v => {{
    const a = byVar[v].sort((x,y)=>x-y); return a[Math.floor(a.length/2)];
  }});
  new Chart(document.getElementById('chart-error'), {{
    type:'bar',
    data:{{ labels, datasets:[{{ label:'median max_err', data: med,
      backgroundColor: labels.map(l=>(COLORS[l]||'#666')+'aa') }}] }},
    options:{{ plugins:{{ legend:{{display:false}} }},
      scales:{{ y:{{ type:'log', ticks:{{color:'#8b949e'}}, grid:{{color:'#2d3640'}} }},
        x:{{ ticks:{{color:'#8b949e', font:{{size:8}}}}, grid:{{display:false}} }} }}
  }});
}})();

// --- ms heatmaps per device ---
(function() {{
  const el = document.getElementById('metric-heatmaps');
  const devices = [...new Set(rows().map(r=>r.device))].sort();
  devices.forEach(dev => {{
    const sub = rows().filter(r => r.device===dev && r.direction==='Forward' && validMs(r));
    const nFfts = nFftList();
    const batches = [...new Set(sub.map(r=>r.batch))].sort((a,b)=>a-b);
    const variants = [...new Set(sub.map(r=>r.variant))].sort();
    variants.forEach(v => {{
      const div = document.createElement('div'); div.className='card';
      const id = 'hm-'+dev+'-'+v; div.innerHTML = '<h4>'+dev+' · '+v+' · ms</h4><div class="plotly" id="'+id+'"></div>';
      el.appendChild(div);
      const z = nFfts.map(n => batches.map(b => {{
        const r = sub.find(x => x.n_fft===n && x.batch===b && x.variant===v);
        return (r && okRow(r)) ? r.ms : null;
      }}));
      Plotly.newPlot(id, [{{ z, x:batches, y:nFfts, type:'heatmap', colorscale:'Viridis',
        hovertemplate:'n=%{{y}} batch=%{{x}} ms=%{{z:.4f}}<extra></extra>' }}],
        {{ paper_bgcolor:'#141b24', plot_bgcolor:'#141b24', font:{{color:'#e6edf3',size:10}},
          yaxis:{{ type:'log', title:'n_fft', tickvals:nFfts, ticktext:nFfts.map(String) }},
          margin:{{l:55,r:10,t:30,b:40}} }}, {{responsive:true}});
    }});
  }});
}})();

// --- capacity / limit charts ---
(function() {{
  const nFfts = nFftList();
  const devices = [...new Set(rows().map(r=>r.device))].sort();
  const rust = rows().filter(r => r.variant==='rustfft' && r.direction==='Forward' && okRow(r));
  if (document.getElementById('limit-scaling') && rust.length) {{
    const traces = devices.map(dev => {{
      const pts = nFfts.map(n => {{
        const c = rust.filter(r => r.device===dev && r.n_fft===n).sort((a,b)=>a.batch-b.batch);
        return c.length ? c[0] : null;
      }}).filter(Boolean);
      return {{ x: pts.map(r=>r.n_fft), y: pts.map(r=>r.ms), name: dev, mode:'lines+markers', type:'scatter' }};
    }});
    Plotly.newPlot('limit-scaling', traces,
      {{ paper_bgcolor:'#141b24', plot_bgcolor:'#141b24', font:{{color:'#e6edf3'}},
        xaxis:{{ type:'log', title:'n_fft' }}, yaxis:{{ type:'log', title:'ms/iter (min batch)' }},
        title:{{ text:'RustFFT forward scaling', font:{{color:'#8b949e'}} }} }}, {{responsive:true}});
  }}
  if (document.getElementById('limit-coverage')) {{
    const z = devices.map(dev => nFfts.map(n => {{
      const sub = rows().filter(r => r.device===dev && r.n_fft===n);
      if (!sub.length) return null;
      return sub.filter(okRow).length / sub.length;
    }}));
    Plotly.newPlot('limit-coverage', [{{ z, x:nFfts, y:devices, type:'heatmap', colorscale:'Greens', zmin:0, zmax:1 }}],
      {{ paper_bgcolor:'#141b24', plot_bgcolor:'#141b24', font:{{color:'#e6edf3'}},
        xaxis:{{ type:'log', title:'n_fft' }}, title:{{ text:'Fraction configs OK', font:{{color:'#8b949e'}} }} }}, {{responsive:true}});
  }}
  if (document.getElementById('limit-max-batch')) {{
    const z = devices.map(dev => nFfts.map(n => {{
      const ok = rust.filter(r => r.device===dev && r.n_fft===n);
      return ok.length ? Math.max(...ok.map(r=>r.batch)) : null;
    }}));
    Plotly.newPlot('limit-max-batch', [{{ z, x:nFfts, y:devices, type:'heatmap', colorscale:'Blues' }}],
      {{ paper_bgcolor:'#141b24', plot_bgcolor:'#141b24', font:{{color:'#e6edf3'}},
        xaxis:{{ type:'log', title:'n_fft' }}, title:{{ text:'Max batch (rustfft fwd)', font:{{color:'#8b949e'}} }} }}, {{responsive:true}});
  }}
}})();

// --- speed bar charts (largest batch per n_fft) ---
(function() {{
  const el = document.getElementById('speed-grid');
  const devices = [...new Set(rows().map(r=>r.device))].sort();
  devices.forEach(dev => {{
    const nFfts = nFftList().filter(n => rows().some(r => r.device===dev && r.n_fft===n));
    nFfts.forEach(n => {{
      const batches = [...new Set(rows().filter(r=>r.device===dev&&r.n_fft===n&&r.direction==='Forward'&&okRow(r)).map(r=>r.batch))].sort((a,b)=>b-a);
      const batch = batches[0] || 8;
      const sub = rows().filter(r => r.device===dev && r.batch===batch && r.n_fft===n &&
        r.direction==='Forward' && okRow(r)).sort((a,b)=>a.ms-b.ms);
      const div = document.createElement('div'); div.className='card';
      const cid = 'c-'+dev+'-'+n;
      div.innerHTML = '<h4>'+dev+' forward n='+n+' batch='+batch+' (max batch, best first)</h4><canvas id="'+cid+'"></canvas>';
      el.appendChild(div);
      new Chart(document.getElementById(cid), {{
        type:'bar',
        data:{{ labels: sub.map(r=>r.variant), datasets:[{{ data: sub.map(r=>r.ms),
          backgroundColor: sub.map(r=>(COLORS[r.variant]||'#666')+'cc') }}] }},
        options:{{ indexAxis:'y', plugins:{{ legend:{{display:false}} }},
          scales:{{ x:{{ type:'log', ticks:{{color:'#8b949e'}}, grid:{{color:'#2d3640'}} }},
            y:{{ ticks:{{color:'#8b949e', font:{{size:8}}}}, grid:{{display:false}} }} }}
      }});
    }});
  }});
}})();

// --- combined loss curves (all models, best mel err first) ---
(function() {{
  const el = document.getElementById('chart-all-loss');
  if (!el || !TELEMETRY || !TELEMETRY.models || !TELEMETRY.models.length) return;
  const sorted = TELEMETRY.models.slice().sort((a,b) => a.final_mel_err - b.final_mel_err);
  const palette = ['#58a6ff','#3fb950','#f85149','#d4a72c','#b07aa1','#86bcb6'];
  new Chart(el, {{
    type:'line',
    data:{{
      datasets: sorted.map((m,i) => ({{
        label: m.model_id + ' (mel=' + m.final_mel_err.toExponential(2) + ')',
        data: (m.loss_curve||[]).map(p => ({{x:p.step, y:p.mel_err}})),
        borderColor: palette[i % palette.length],
        tension: .2,
        pointRadius: 0,
        parsing: false,
      }})),
    }},
    options:{{ responsive:true, plugins:{{ legend:{{ labels:{{color:'#8b949e', font:{{size:10}}}} }} }},
      scales:{{ x:{{ type:'linear', title:{{display:true,text:'step',color:'#8b949e'}} }},
        y:{{ type:'log', title:{{display:true,text:'mel max err',color:'#8b949e'}},
          ticks:{{color:'#8b949e'}}, grid:{{color:'#2d3640'}} }} }}
  }});
}})();

// --- top-5 bar charts per n_fft ---
(function() {{
  const el = document.getElementById('top5-charts');
  if (!el) return;
    const nFfts = nFftList();
  nFfts.forEach(n => {{
    const best = {{}};
    rows().filter(r => r.n_fft===n && okRow(r)).forEach(r => {{
      const prev = best[r.variant];
      if (!prev || r.ms < prev.ms || (r.ms===prev.ms && r.max_err < prev.max_err)) best[r.variant]=r;
    }});
    const top = Object.values(best).sort((a,b)=>a.ms-b.ms || a.max_err-b.max_err).slice(0,5);
    const div = document.createElement('div'); div.className='card';
    const cid = 'top5-'+n;
    div.innerHTML = '<h4>n_fft='+n+' · top 5 variants (ms)</h4><canvas id="'+cid+'"></canvas>';
    el.appendChild(div);
    new Chart(document.getElementById(cid), {{
      type:'bar',
      data:{{ labels: top.map(r=>r.variant), datasets:[{{ data: top.map(r=>r.ms),
        backgroundColor: top.map(r=>(COLORS[r.variant]||'#666')+'cc') }}] }},
      options:{{ indexAxis:'y', plugins:{{ legend:{{display:false}},
        tooltip:{{ callbacks:{{ afterLabel: ctx => {{
          const r = top[ctx.dataIndex];
          return r ? r.direction+' · '+r.device+' · batch '+r.batch+'\\nmax_err '+r.max_err.toExponential(2) : '';
        }} }} }} }},
        scales:{{ x:{{ type:'log', ticks:{{color:'#8b949e'}}, grid:{{color:'#2d3640'}} }},
          y:{{ ticks:{{color:'#8b949e', font:{{size:8}}}}, grid:{{display:false}} }} }}
    }});
  }});
}})();

// --- model telemetry: loss curves + 3D + heatmaps ---
(function() {{
  if (!TELEMETRY || !TELEMETRY.models) return;
  TELEMETRY.models.forEach((m, idx) => {{
    const lc = document.getElementById('loss-'+idx);
    if (lc && m.loss_curve && m.loss_curve.length) {{
      new Chart(lc, {{
        type:'line',
        data:{{
          labels: m.loss_curve.map(p=>p.step),
          datasets:[
            {{ label:'total_loss', data: m.loss_curve.map(p=>p.total_loss), borderColor:'#58a6ff', tension:.2 }},
            {{ label:'mel_err', data: m.loss_curve.map(p=>p.mel_err), borderColor:'#3fb950', tension:.2 }},
            {{ label:'spec_err', data: m.loss_curve.map(p=>p.spec_err), borderColor:'#f85149', tension:.2 }},
          ]
        }},
        options:{{ responsive:true, plugins:{{ legend:{{ labels:{{color:'#8b949e'}} }} }},
          scales:{{ x:{{ title:{{display:true,text:'step',color:'#8b949e'}} }},
            y:{{ type:'log', ticks:{{color:'#8b949e'}}, grid:{{color:'#2d3640'}} }} }}
      }});
    }}
    const ls = document.getElementById('landscape-'+idx);
    if (ls && m.landscape) {{
      Plotly.newPlot(ls, [{{ x:m.landscape.x, y:m.landscape.y, z:m.landscape.z, type:'surface',
        colorscale:'Portland', opacity:.92 }}],
        {{ paper_bgcolor:'#141b24', plot_bgcolor:'#141b24', font:{{color:'#e6edf3'}},
          title:{{ text:'Loss landscape (GD slice)', font:{{size:12,color:'#8b949e'}} }},
          scene:{{ xaxis:{{title:m.landscape.x_label}}, yaxis:{{title:m.landscape.y_label}},
            zaxis:{{title:'mel MSE', type:'log'}} }} }}, {{responsive:true}});
    }}
    const gh = document.getElementById('gates-'+idx);
    if (gh && m.heatmap) {{
      const st = m.heatmap.stages, bf = m.heatmap.butterflies;
      const z = [];
      for (let s=0; s<st; s++) {{
        const row = [];
        for (let b=0; b<bf; b++) row.push(m.heatmap.gates[s*bf+b] ?? 0);
        z.push(row);
      }}
      Plotly.newPlot(gh, [{{ z, type:'heatmap', colorscale:'YlGnBu', zmin:0, zmax:1,
        hovertemplate:'stage=%{{y}} butterfly=%{{x}} gate=%{{z:.3f}}<extra></extra>' }}],
        {{ paper_bgcolor:'#141b24', plot_bgcolor:'#141b24', font:{{color:'#e6edf3'}},
          title:{{ text:'Gate activation by stage × butterfly', font:{{size:11,color:'#8b949e'}} }},
          xaxis:{{ title:'butterfly' }}, yaxis:{{ title:'stage' }} }}, {{responsive:true}});
    }}
    const th = document.getElementById('tw-'+idx);
    if (th && m.heatmap && m.heatmap.twiddle_mag && m.heatmap.twiddle_mag.length) {{
      const st = m.heatmap.stages, bf = m.heatmap.butterflies;
      const z = [];
      for (let s=0; s<st; s++) {{
        const row = [];
        for (let b=0; b<bf; b++) row.push(m.heatmap.twiddle_mag[s*bf+b] ?? 0);
        z.push(row);
      }}
      Plotly.newPlot(th, [{{ z, type:'heatmap', colorscale:'Hot',
        hovertemplate:'stage=%{{y}} butterfly=%{{x}} |w|=%{{z:.3f}}<extra></extra>' }}],
        {{ paper_bgcolor:'#141b24', plot_bgcolor:'#141b24', font:{{color:'#e6edf3'}},
          title:{{ text:'Twiddle magnitude |w|', font:{{size:11,color:'#8b949e'}} }} }}, {{responsive:true}});
    }}
    const fh = document.getElementById('freq-'+idx);
    if (fh && m.heatmap && m.heatmap.freq_mask && m.heatmap.freq_mask.length) {{
      const n = m.heatmap.freq_mask.length;
      Plotly.newPlot(fh, [{{ z:[m.heatmap.freq_mask], x:Array.from({{length:n}},(_,i)=>i),
        type:'heatmap', colorscale:'Blues', zmin:0, zmax:1,
        hovertemplate:'bin=%{{x}} mask=%{{z:.3f}}<extra></extra>' }}],
        {{ paper_bgcolor:'#141b24', plot_bgcolor:'#141b24', font:{{color:'#e6edf3'}},
          title:{{ text:'Frequency mask activation', font:{{size:11,color:'#8b949e'}} }},
          yaxis:{{ showticklabels:false }} }}, {{responsive:true}});
    }}
  }});
}})();

// --- multi-train heatmap ---
(function() {{
  const el = document.getElementById('train-heat');
  if (!el) return;
  if (!TRAIN || !TRAIN.rows) {{
    el.innerHTML = '<p style="color:#8b949e;padding:1rem">No multi-train data — pass <code>--train-json</code>.</p>';
    return;
  }}
  const regimes = [...new Set(TRAIN.rows.map(r=>r.regime))].sort();
  const nFfts = TRAIN.n_ffts || [...new Set(TRAIN.rows.map(r=>r.eval_n_fft))].sort((a,b)=>a-b);
  const z = regimes.map(reg => nFfts.map(n => {{
    const r = TRAIN.rows.find(x => x.regime===reg && x.eval_n_fft===n);
    return r ? r.roundtrip_max_err : null;
  }}));
  Plotly.newPlot('train-heat', [{{ z, x:nFfts, y:regimes, type:'heatmap', colorscale:'YlOrRd' }}],
    {{ paper_bgcolor:'#141b24', plot_bgcolor:'#141b24', font:{{color:'#e6edf3'}},
      title:{{ text:'Multi-train roundtrip max error', font:{{color:'#8b949e'}} }} }}, {{responsive:true}});
}})();
</script></body></html>"##,
        overview_cards = overview_cards,
        top5_section = top5_section,
        limits_section = limits_section,
        ranked_table = ranked_table,
        n_fft_meta_json = n_fft_meta_json,
        model_panels = model_panels,
        ablation_json = ablation_json,
        train_json = train_json,
        telemetry_json = telemetry_json,
        colors_json = variant_colors_json(),
    )
}

fn build_limits_section(inputs: &FullStudyInputs) -> String {
    let Some(ab) = &inputs.ablation else {
        return "<p>No ablation data.</p>".into();
    };
    let n_ffts: Vec<usize> = if ab.n_ffts.is_empty() {
        let mut ns: Vec<usize> = ab.rows.iter().map(|r| r.n_fft).collect();
        ns.sort_unstable();
        ns.dedup();
        ns
    } else {
        ab.n_ffts.clone()
    };
    let devices: Vec<String> = {
        let mut d: Vec<_> = ab.rows.iter().map(|r| r.device.clone()).collect();
        d.sort();
        d.dedup();
        d
    };
    let mut cards = String::from(r#"<section><div class="cards">"#);
    for dev in &devices {
        let max_n = n_ffts
            .iter()
            .rev()
            .find(|&&n| {
                ab.rows.iter().any(|r| {
                    r.device == *dev && r.n_fft == n && r.variant == "rustfft" && ablation_row_ok(r)
                })
            })
            .copied()
            .unwrap_or(0);
        let kind = if is_gpu_device_label(dev) {
            "GPU"
        } else {
            "CPU"
        };
        cards.push_str(&format!(
            r#"<div class="card gold"><h4>Max n_fft · {dev}</h4><div class="val">{max_n}</div>
            <div class="sub">{kind} · rustfft forward OK</div></div>"#
        ));
    }
    let max_compiled = n_ffts
        .iter()
        .rev()
        .find(|&&n| {
            ab.rows.iter().any(|r| {
                r.n_fft == n
                    && ablation_row_ok(r)
                    && (r.variant.contains("compiled") || r.variant == "rlx_op_fft")
            })
        })
        .copied()
        .unwrap_or(0);
    cards.push_str(&format!(
        r#"<div class="card"><h4>Max compiled n_fft</h4><div class="val">{max_compiled}</div>
        <div class="sub">any compiled / rlx_op path OK</div></div>"#
    ));
    let fail_count = ab.rows.iter().filter(|r| r.status != "ok").count();
    cards.push_str(&format!(
        r#"<div class="card"><h4>Failures recorded</h4><div class="val">{fail_count}</div>
        <div class="sub">prepare_fail + bench_fail</div></div>"#
    ));
    cards.push_str("</div></section>");

    let mut table = String::from(
        "<section><h2>Per n_fft capacity</h2><table><thead><tr>\
         <th>n_fft</th><th class=\"left\">device</th><th>ok</th><th>fail</th><th>max_batch</th>\
         <th class=\"left\">fastest variant</th><th>ms</th><th>max_err</th></tr></thead><tbody>",
    );
    for &n in &n_ffts {
        for dev in &devices {
            let sub: Vec<_> = ab
                .rows
                .iter()
                .filter(|r| r.n_fft == n && r.device == *dev)
                .collect();
            if sub.is_empty() {
                continue;
            }
            let ok_n = sub.iter().filter(|r| ablation_row_ok(r)).count();
            let fail_n = sub.len() - ok_n;
            let max_batch = sub
                .iter()
                .filter(|r| ablation_row_ok(r))
                .map(|r| r.batch)
                .max()
                .unwrap_or(0);
            let fastest = sub
                .iter()
                .filter(|r| ablation_row_ok(r))
                .min_by(|a, b| a.ms.partial_cmp(&b.ms).unwrap());
            let row_class = if ok_n == 0 {
                "fail"
            } else if fail_n > 0 {
                "limit"
            } else {
                ""
            };
            if let Some(f) = fastest {
                table.push_str(&format!(
                    "<tr class=\"{row_class}\"><td>{n}</td><td class=\"left\">{dev}</td><td>{ok_n}</td>\
                     <td>{fail_n}</td><td>{max_batch}</td><td class=\"left\">{}</td><td>{:.4}</td><td>{:.3e}</td></tr>",
                    f.variant, f.ms, f.max_err
                ));
            } else {
                table.push_str(&format!(
                    "<tr class=\"fail\"><td>{n}</td><td class=\"left\">{dev}</td><td>0</td>\
                     <td>{fail_n}</td><td>0</td><td class=\"left\">—</td><td>—</td><td>—</td></tr>"
                ));
            }
        }
    }
    table.push_str("</tbody></table></section>");
    format!("{cards}{table}")
}

fn build_top5_per_nfft_section(inputs: &FullStudyInputs) -> String {
    let Some(ab) = &inputs.ablation else {
        return "<p>No ablation data.</p>".into();
    };
    let groups = top5_variants_per_n_fft(ab);
    if groups.is_empty() {
        return "<p>No valid timing rows.</p>".into();
    }
    let mut out = String::from(r#"<div class="top5-grid">"#);
    for (n, rows) in groups {
        out.push_str(&format!(r#"<div class="top5-block card"><h3>n_fft = {n}</h3><table><thead><tr>
          <th>#</th><th class="left">variant</th><th class="left">dir</th><th class="left">device</th>
          <th>batch</th><th>ms</th><th>max_err</th><th>params</th></tr></thead><tbody>"#));
        for (rank, r) in rows.iter().enumerate() {
            let rank_class = match rank {
                0 => "rank1",
                1 => "rank2",
                2 => "rank3",
                3 => "rank4",
                4 => "rank5",
                _ => "",
            };
            out.push_str(&format!(
                "<tr class=\"{rank_class}\"><td>{}</td><td class=\"left\">{}</td><td class=\"left\">{}</td>\
                 <td class=\"left\">{}</td><td>{}</td><td>{:.4}</td><td>{:.3e}</td><td>{}</td></tr>",
                rank + 1,
                r.variant,
                r.direction,
                r.device,
                r.batch,
                r.ms,
                r.max_err,
                r.param_count,
            ));
        }
        out.push_str("</tbody></table></div>");
    }
    out.push_str("</div>");
    out
}

fn build_ranked_table(inputs: &FullStudyInputs) -> String {
    let Some(ab) = &inputs.ablation else {
        return "<p>No ablation data.</p>".into();
    };
    let mut sorted: Vec<_> = ab.rows.iter().collect();
    sorted.sort_by(|a, b| {
        let a_ok = ablation_row_ok(a);
        let b_ok = ablation_row_ok(b);
        match (a_ok, b_ok) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => {
                a.ms.partial_cmp(&b.ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a.max_err
                            .partial_cmp(&b.max_err)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            }
        }
    });
    let mut out = String::from(
        "<div style=\"max-height:520px;overflow:auto\"><table><thead><tr>\
         <th>#</th><th class=\"left\">tier</th><th class=\"left\">variant</th><th class=\"left\">dir</th>\
         <th class=\"left\">device</th><th>n_fft</th><th>batch</th><th>ms</th><th>max_err</th>\
         <th class=\"left\">status</th><th>params</th><th>memory</th></tr></thead><tbody>",
    );
    for (rank, r) in sorted.iter().enumerate() {
        let rank_class = match rank {
            0 => "rank1",
            1 => "rank2",
            2 => "rank3",
            _ => "",
        };
        let mem_kb = r.memory_bytes as f64 / 1024.0;
        let ms_str = if r.ms.is_nan() {
            "—".to_string()
        } else {
            format!("{:.4}", r.ms)
        };
        let err_str = if r.max_err.is_nan() {
            "—".to_string()
        } else {
            format!("{:.3e}", r.max_err)
        };
        out.push_str(&format!(
            "<tr class=\"{rank_class}\"><td>{}</td><td class=\"left\">{}</td><td class=\"left\">{}</td>\
             <td class=\"left\">{}</td><td class=\"left\">{}</td><td>{}</td><td>{}</td>\
             <td>{ms_str}</td><td>{err_str}</td><td class=\"left\">{}</td><td>{}</td><td>{mem_kb:.1} KiB</td></tr>",
            rank + 1,
            r.tier,
            r.variant,
            r.direction,
            r.device,
            r.n_fft,
            r.batch,
            &r.status,
            r.param_count,
        ));
    }
    out.push_str("</tbody></table></div>");
    out
}

fn build_overview_cards(inputs: &FullStudyInputs) -> String {
    let mut s = String::from(r#"<section><div class="cards">"#);
    if let Some(ab) = &inputs.ablation {
        let wins = ablation_winners(ab);
        if let Some((n, b, d, dir, var, ms)) =
            wins.iter().min_by(|a, b| a.5.partial_cmp(&b.5).unwrap())
        {
            s.push_str(&format!(
                r#"<div class="card gold"><h4>Fastest overall</h4><div class="val">{ms:.4} ms</div>
                <div class="sub">{var} · {dir} · n={n} b={b} {d}</div></div>"#
            ));
        }
        if let Some(r) = ab
            .rows
            .iter()
            .filter(|r| r.max_err.is_finite())
            .min_by(|a, b| a.max_err.partial_cmp(&b.max_err).unwrap())
        {
            s.push_str(&format!(
                r#"<div class="card"><h4>Best precision</h4><div class="val">{:.2e}</div>
                <div class="sub">{} · n={} {}</div></div>"#,
                r.max_err, r.variant, r.n_fft, r.device
            ));
        }
        let total_mem: usize = ab.rows.iter().map(|r| r.memory_bytes).max().unwrap_or(0);
        s.push_str(&format!(
            r#"<div class="card"><h4>Largest model</h4><div class="val">{:.1} KiB</div>
            <div class="sub">param storage (f32)</div></div>"#,
            total_mem as f64 / 1024.0
        ));
    }
    if let Some(t) = &inputs.telemetry {
        s.push_str(&format!(
            r#"<div class="card"><h4>Trained models</h4><div class="val">{}</div>
            <div class="sub">with loss curves + heatmaps</div></div>"#,
            t.models.len()
        ));
    }
    s.push_str("</div></section>");
    s
}

fn build_model_panels(inputs: &FullStudyInputs) -> String {
    let Some(t) = &inputs.telemetry else {
        return "<p>No model telemetry — run with <code>--run-model-studies</code>.</p>".into();
    };
    if t.models.is_empty() {
        return "<p>No models collected.</p>".into();
    }
    let mut models: Vec<_> = t.models.iter().collect();
    models.sort_by(|a, b| {
        a.final_mel_err
            .partial_cmp(&b.final_mel_err)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.final_spec_err
                    .partial_cmp(&b.final_spec_err)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let mut out = String::from(
        "<div class=\"note\">Models sorted by <strong>final mel error</strong> (best first).</div>",
    );
    for (i, m) in models.iter().enumerate() {
        let p = &m.params;
        out.push_str(&format!(
            r#"<div class="model-block" id="model-{i}">
            <h3>{id} · {variant} · n={n_fft} batch={batch}</h3>
            <div class="cards">
              <div class="card"><h4>Params</h4><div class="val">{total}</div>
                <div class="sub">tw={tw} gates={gates} mask={mask} denoiser={den} mel={mel}</div></div>
              <div class="card"><h4>Memory</h4><div class="val">{mem_kib:.1} KiB</div></div>
              <div class="card"><h4>Final mel err</h4><div class="val">{mel_err:.2e}</div></div>
              <div class="card"><h4>Final spec err</h4><div class="val">{spec_err:.2e}</div></div>
            </div>
            <div class="grid2">
              <div class="card"><h4>Gradient descent loss</h4><canvas id="loss-{i}"></canvas></div>
              <div class="card"><div class="plotly" id="landscape-{i}"></div></div>
              <div class="card"><div class="plotly" id="gates-{i}"></div></div>
              <div class="card"><div class="plotly" id="tw-{i}"></div></div>
              <div class="card"><div class="plotly" id="freq-{i}"></div></div>
            </div></div>"#,
            i = i,
            id = m.model_id,
            variant = m.variant,
            n_fft = m.n_fft,
            batch = m.batch,
            total = p.total_params,
            tw = p.twiddles,
            gates = p.gates,
            mask = p.freq_mask,
            den = p.denoiser,
            mel = p.mel_filters,
            mem_kib = p.memory_bytes as f64 / 1024.0,
            mel_err = m.final_mel_err,
            spec_err = m.final_spec_err,
        ));
    }
    out
}

fn variant_colors_json() -> String {
    let map: std::collections::BTreeMap<&str, &str> = [
        ("rustfft", "#4e79a7"),
        ("rlx_op_fft", "#f28e2b"),
        ("rlx_op_ifft", "#ff9d4d"),
        ("butterfly_eager", "#59a14f"),
        ("butterfly_compiled", "#e15759"),
        ("butterfly_q8", "#8cd17d"),
        ("butterfly_unitary", "#bab0ac"),
        ("stockham_eager", "#76b7b2"),
        ("stockham_compiled", "#edc948"),
        ("fused_spectral_eager", "#af7aa1"),
        ("fused_spectral_compiled", "#ff9da7"),
        ("domain_twiddle", "#86bcb6"),
        ("learned_e2e", "#b07aa1"),
        ("welch_rustfft", "#4c78a8"),
        ("welch_rlx_op_fft", "#72b7b2"),
        ("welch_butterfly_eager", "#54a24b"),
        ("welch_butterfly_compiled", "#eeca3b"),
    ]
    .into_iter()
    .collect();
    serde_json::to_string(&map).unwrap_or_else(|_| "{}".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ablation::AblationRow;

    fn row(variant: &str, n: usize, ms: f64, err: f32) -> AblationRow {
        AblationRow {
            tier: "baseline".into(),
            variant: variant.into(),
            direction: "Forward".into(),
            n_fft: n,
            batch: 8,
            device: "cpu".into(),
            iters: 5,
            ms,
            max_err: err,
            train_steps: 0,
            param_count: 100,
            memory_bytes: 400,
            status: "ok".into(),
            note: None,
        }
    }

    #[test]
    fn render_includes_top5_section() {
        let html = render_full_study_html(&FullStudyInputs {
            ablation: Some(AblationReport {
                iters: 5,
                train_steps: 0,
                both_dirs: true,
                with_welch: true,
                limit_sweep: false,
                n_ffts: vec![64],
                elapsed_ms: 1.0,
                rows: vec![row("rustfft", 64, 0.01, 0.0)],
            }),
            multi_train: None,
            telemetry: None,
        });
        assert!(html.contains("Top 5 variants per n_fft"));
        assert!(html.contains("top5-charts"));
        assert!(html.contains("Capacity limits"));
        assert!(html.contains("limit-scaling"));
    }
}

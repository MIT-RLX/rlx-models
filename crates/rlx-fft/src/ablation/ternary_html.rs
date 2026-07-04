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

//! HTML report for ternary architecture ablation.

use crate::ablation::ternary::{
    TernaryAblationReport, ternary_aggregate_variants, ternary_pareto_frontier,
    ternary_recommendation,
};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::Path;

pub fn read_ternary_ablation_json(path: &Path) -> Result<TernaryAblationReport> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn write_ternary_ablation_html(path: &Path, report: &TernaryAblationReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, render_ternary_ablation_html(report))
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub fn render_ternary_ablation_html(report: &TernaryAblationReport) -> String {
    let json = serde_json::to_string(report).unwrap_or_else(|_| "{}".to_string());
    let mut devices = BTreeSet::new();
    let mut configs: BTreeSet<(usize, usize, String)> = BTreeSet::new();
    for r in &report.rows {
        devices.insert(r.device.clone());
        configs.insert((r.n_fft, r.batch, r.device.clone()));
    }

    let mut sections = String::new();
    for (n_fft, batch, device) in &configs {
        let rec =
            ternary_recommendation(report, device, *n_fft, *batch).unwrap_or_else(|| "n/a".into());
        let points = ternary_aggregate_variants(report, device, *n_fft, *batch);
        let front = ternary_pareto_frontier(&points);
        sections.push_str(&format!(
            r#"<section class="card"><h2>n_fft={n_fft} batch={batch} device={device}</h2>
<p><strong>Pareto pick:</strong> {rec}</p>
<table><thead><tr><th>variant</th><th>exec</th><th>prune</th><th>mean ms</th><th>mel</th><th>denoise</th><th>q8</th><th>welch</th><th>compute</th></tr></thead><tbody>"#
        ));
        for p in front {
            sections.push_str(&format!(
                r#"<tr><td>{}</td><td>{}</td><td>{:?}</td><td>{:.4}</td><td>{:.2e}</td><td>{:.2e}</td><td>{:.2e}</td><td>{:.2e}</td><td>{:?}</td></tr>"#,
                p.variant,
                p.exec,
                p.prune_target,
                p.mean_ms,
                p.mel_norm,
                p.denoise_norm,
                p.q8_norm,
                p.welch_norm,
                p.compute_fraction
            ));
        }
        sections.push_str("</tbody></table></section>");
    }

    let scatter_id = "pareto-scatter";
    let mut scatter_traces = String::new();
    for (n_fft, batch, device) in &configs {
        let points = ternary_aggregate_variants(report, device, *n_fft, *batch);
        for p in &points {
            scatter_traces.push_str(&format!(
                r#"{{x:[{:.4}],y:[{:.4e}],mode:'markers',name:'{} {} {:?}',text:['{}@{}']}},"#,
                p.mean_ms, p.max_norm_err, p.variant, p.exec, p.prune_target, p.variant, p.exec
            ));
        }
    }

    format!(
        r##"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Ternary Architecture Ablation</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.27.0/plotly.min.js"></script>
<style>
body{{font-family:system-ui,sans-serif;margin:1rem;background:#0f1419;color:#e6edf3}}
.card{{background:#1a2332;padding:1rem;margin:1rem 0;border-radius:8px}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #30363d;padding:.4rem;text-align:left}}
h1{{color:#58a6ff}}
</style></head><body>
<h1>Ternary Architecture Ablation</h1>
<p>quick={} elapsed_ms={:.1} devices={:?}</p>
<div id="{scatter_id}" style="height:420px"></div>
{sections}
<script>
const data=[{scatter_traces}];
Plotly.newPlot('{scatter_id}',data,{{title:'Pareto: mean latency vs max normalized error',xaxis:{{title:'mean ms'}},yaxis:{{title:'max norm err',type:'log'}}}},{{responsive:true}});
const report={json};
</script>
</body></html>"##,
        report.quick,
        report.elapsed_ms,
        devices,
        scatter_id = scatter_id,
        sections = sections,
        scatter_traces = scatter_traces,
        json = json
    )
}

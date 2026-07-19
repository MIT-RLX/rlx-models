//! Self-contained HTML report (no external CDN).

use std::fs;
use std::path::Path;

use anyhow::Result;

use super::json::{BenchRow, Summary};

pub fn write_html(path: &Path, rows: &[BenchRow], summary: &Summary) -> Result<()> {
    let mut sorted = rows.to_vec();
    sorted.sort_by(|a, b| {
        (&a.model, &a.phrase, &a.device, &a.scenario).cmp(&(
            &b.model,
            &b.phrase,
            &b.device,
            &b.scenario,
        ))
    });

    let mut body = String::new();
    body.push_str("<h1>RLX TTS Bench</h1>\n");
    body.push_str(&format!(
        "<p class=\"sum\">ok {} · skipped {} · failed {} · total {}</p>\n",
        summary.n_ok, summary.n_skipped, summary.n_failed, summary.n_rows
    ));

    body.push_str("<h2>By model</h2>\n<table><tr><th>model</th><th>ok</th><th>median RTF</th><th>median Whisper cov</th></tr>\n");
    for (m, s) in &summary.by_model {
        body.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
            esc(m),
            s.n_ok,
            fmt_opt(s.median_rtf, 3),
            fmt_opt(s.median_whisper_cov, 2),
        ));
    }
    body.push_str("</table>\n");

    body.push_str(&heatmap_svg(rows));

    body.push_str("<h2>Rows</h2>\n<table>\n<tr>");
    for h in [
        "model", "device", "phrase", "scenario", "status", "RTF", "ms", "cos_cpu", "whisper",
        "stft_cos", "peak", "wav", "note",
    ] {
        body.push_str(&format!("<th>{h}</th>"));
    }
    body.push_str("</tr>\n");

    for r in &sorted {
        let whisper = r
            .whisper
            .as_ref()
            .map(|w| {
                format!(
                    "{:.0}% ({}/{})",
                    w.coverage * 100.0,
                    w.content_hits,
                    w.content_total
                )
            })
            .unwrap_or_else(|| "—".into());
        let whisper_cls = r
            .whisper
            .as_ref()
            .map(|w| if w.coverage >= 0.7 { "good" } else { "bad" })
            .unwrap_or("");
        let cos = r.cosine_vs_cpu;
        let cos_cls = cos
            .map(|c| {
                if c >= 0.9 {
                    "good"
                } else if c >= 0.5 {
                    "mid"
                } else {
                    "bad"
                }
            })
            .unwrap_or("");
        let stft = r
            .spectral
            .as_ref()
            .map(|s| format!("{:.3}", s.stft_cosine))
            .unwrap_or_else(|| "—".into());
        let peak = r
            .noise
            .as_ref()
            .map(|n| format!("{:.3}", n.peak))
            .unwrap_or_else(|| "—".into());
        let wav = r
            .wav_rel
            .as_ref()
            .map(|p| format!("<a href=\"{}\">wav</a>", esc(p)))
            .unwrap_or_else(|| "—".into());
        let note = r
            .error
            .as_ref()
            .or(r.skip_reason.as_ref())
            .map(|s| esc(&truncate(s, 80)))
            .unwrap_or_default();
        body.push_str(&format!(
            "<tr class=\"{}\"><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\
             <td>{}</td><td>{}</td><td class=\"{}\">{}</td><td class=\"{}\">{}</td>\
             <td>{}</td><td>{}</td><td>{}</td><td class=\"note\">{}</td></tr>\n",
            esc(&r.status),
            esc(&r.model),
            esc(&r.device),
            esc(&r.phrase),
            esc(&r.scenario),
            esc(&r.status),
            fmt_opt(r.rtf, 3),
            fmt_opt(r.wall_ms, 1),
            cos_cls,
            fmt_opt(cos, 4),
            whisper_cls,
            esc(&whisper),
            esc(&stft),
            esc(&peak),
            wav,
            note,
        ));
    }
    body.push_str("</table>\n");

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>RLX TTS Bench</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#0f1419;color:#e7ecf1}}
h1,h2{{font-weight:600}}
.sum{{opacity:.8}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 28px}}
th,td{{border:1px solid #2a3440;padding:6px 8px;text-align:left}}
th{{background:#1a2330;position:sticky;top:0}}
tr.ok td{{background:#122018}}
tr.skipped td{{background:#1a1a14;opacity:.85}}
tr.failed td{{background:#241416}}
.good{{color:#6dce8b}}
.mid{{color:#e0c35a}}
.bad{{color:#e07a7a}}
.note{{max-width:280px;font-size:11px;opacity:.75}}
a{{color:#7eb6ff}}
svg.heat{{display:block;margin:16px 0}}
</style></head><body>
{body}
</body></html>"#
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)?;
    Ok(())
}

fn heatmap_svg(rows: &[BenchRow]) -> String {
    let ok: Vec<_> = rows
        .iter()
        .filter(|r| r.status == "ok" && r.rtf.is_some())
        .collect();
    if ok.is_empty() {
        return String::new();
    }
    let mut models: Vec<String> = ok.iter().map(|r| r.model.clone()).collect();
    models.sort();
    models.dedup();
    let mut devices: Vec<String> = ok.iter().map(|r| r.device.clone()).collect();
    devices.sort();
    devices.dedup();
    let cell = 28;
    let left = 120;
    let top = 40;
    let w = left + devices.len() * cell + 20;
    let h = top + models.len() * cell + 20;
    let mut s = format!(
        "<h2>RTF heatmap</h2>\n<svg class=\"heat\" width=\"{w}\" height=\"{h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    );
    for (di, d) in devices.iter().enumerate() {
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" fill=\"#9aa\" font-size=\"11\" transform=\"rotate(-40 {} {})\">{}</text>\n",
            left + di * cell + 4,
            top - 4,
            left + di * cell + 4,
            top - 4,
            esc(d)
        ));
    }
    for (mi, m) in models.iter().enumerate() {
        s.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" fill=\"#ccc\" font-size=\"11\" text-anchor=\"end\">{}</text>\n",
            left - 6,
            top + mi * cell + 18,
            esc(m)
        ));
        for (di, d) in devices.iter().enumerate() {
            let rtf = ok
                .iter()
                .filter(|r| &r.model == m && &r.device == d)
                .filter_map(|r| r.rtf)
                .fold(None, |acc: Option<f64>, v| {
                    Some(acc.map(|a| a.min(v)).unwrap_or(v))
                });
            let (fill, label) = match rtf {
                Some(v) => {
                    let t = (1.0 / (1.0 + v)).clamp(0.0, 1.0);
                    let g = (80.0 + 140.0 * t) as u8;
                    let r = (200.0 - 120.0 * t) as u8;
                    (format!("rgb({r},{g},90)"), format!("{v:.2}"))
                }
                None => ("#222".into(), "—".into()),
            };
            let x = left + di * cell;
            let y = top + mi * cell;
            s.push_str(&format!(
                "<rect x=\"{x}\" y=\"{y}\" width=\"{}\" height=\"{}\" fill=\"{fill}\" stroke=\"#333\"/>\n",
                cell - 2,
                cell - 2
            ));
            s.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" fill=\"#111\" font-size=\"9\" text-anchor=\"middle\">{}</text>\n",
                x + cell / 2,
                y + 18,
                label
            ));
        }
    }
    s.push_str("</svg>\n");
    s
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn fmt_opt(v: Option<f64>, digits: usize) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.digits$}"),
        _ => "—".into(),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

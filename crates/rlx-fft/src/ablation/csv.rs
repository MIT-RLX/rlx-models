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

//! Ablation results as CSV — source of truth for study HTML reports.

use crate::ablation::{AblationReport, AblationRow};
use crate::ablation::{ablation_row_ok, top5_variants_per_n_fft};
use crate::model::config::is_gpu_device_label;
use anyhow::{Context, Result, ensure};
use std::path::{Path, PathBuf};

pub const ROWS_CSV: &str = "ablation_rows.csv";
pub const META_CSV: &str = "ablation_meta.csv";
pub const TOP5_CSV: &str = "ablation_top5.csv";
pub const LIMITS_CSV: &str = "ablation_limits.csv";

fn csv_escape(s: &str) -> String {
    if s.contains(['"', ',', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn write_f64(w: &mut String, v: f64) {
    if v.is_nan() {
        w.push_str("nan");
    } else {
        w.push_str(&v.to_string());
    }
}

fn write_f32(w: &mut String, v: f32) {
    if v.is_nan() {
        w.push_str("nan");
    } else {
        w.push_str(&v.to_string());
    }
}

fn parse_f64(s: &str) -> f64 {
    if s.eq_ignore_ascii_case("nan") {
        f64::NAN
    } else {
        s.parse().unwrap_or(f64::NAN)
    }
}

fn parse_f32(s: &str) -> f32 {
    if s.eq_ignore_ascii_case("nan") {
        f32::NAN
    } else {
        s.parse().unwrap_or(f32::NAN)
    }
}

fn parse_bool(s: &str) -> bool {
    matches!(s.trim(), "1" | "true" | "True" | "TRUE")
}

pub fn write_ablation_rows_csv(path: &Path, rows: &[AblationRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = String::from(
        "tier,variant,direction,n_fft,batch,device,iters,ms,max_err,train_steps,param_count,memory_bytes,status,note\n",
    );
    for r in rows {
        out.push_str(&csv_escape(&r.tier));
        out.push(',');
        out.push_str(&csv_escape(&r.variant));
        out.push(',');
        out.push_str(&csv_escape(&r.direction));
        out.push(',');
        out.push_str(&r.n_fft.to_string());
        out.push(',');
        out.push_str(&r.batch.to_string());
        out.push(',');
        out.push_str(&csv_escape(&r.device));
        out.push(',');
        out.push_str(&r.iters.to_string());
        out.push(',');
        write_f64(&mut out, r.ms);
        out.push(',');
        write_f32(&mut out, r.max_err);
        out.push(',');
        out.push_str(&r.train_steps.to_string());
        out.push(',');
        out.push_str(&r.param_count.to_string());
        out.push(',');
        out.push_str(&r.memory_bytes.to_string());
        out.push(',');
        out.push_str(&csv_escape(&r.status));
        out.push(',');
        out.push_str(&csv_escape(r.note.as_deref().unwrap_or("")));
        out.push('\n');
    }
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))
}

pub fn write_ablation_meta_csv(path: &Path, report: &AblationReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let n_ffts = if report.n_ffts.is_empty() {
        let mut ns: Vec<usize> = report.rows.iter().map(|r| r.n_fft).collect();
        ns.sort_unstable();
        ns.dedup();
        ns
    } else {
        report.n_ffts.clone()
    };
    let mut out = String::from("key,value\n");
    out.push_str(&format!("iters,{}\n", report.iters));
    out.push_str(&format!("train_steps,{}\n", report.train_steps));
    out.push_str(&format!("both_dirs,{}\n", report.both_dirs));
    out.push_str(&format!("with_welch,{}\n", report.with_welch));
    out.push_str(&format!("limit_sweep,{}\n", report.limit_sweep));
    out.push_str(&format!("elapsed_ms,{}\n", report.elapsed_ms));
    out.push_str(&format!(
        "n_ffts,\"{}\"\n",
        n_ffts
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",")
    ));
    out.push_str(&format!("row_count,{}\n", report.rows.len()));
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))
}

pub fn write_ablation_top5_csv(path: &Path, report: &AblationReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = String::from(
        "n_fft,rank,variant,direction,device,batch,ms,max_err,param_count,memory_bytes\n",
    );
    for (n, rows) in top5_variants_per_n_fft(report) {
        for (rank, r) in rows.iter().enumerate() {
            out.push_str(&format!("{n},{}", rank + 1));
            out.push(',');
            out.push_str(&csv_escape(&r.variant));
            out.push(',');
            out.push_str(&csv_escape(&r.direction));
            out.push(',');
            out.push_str(&csv_escape(&r.device));
            out.push(',');
            out.push_str(&r.batch.to_string());
            out.push(',');
            write_f64(&mut out, r.ms);
            out.push(',');
            write_f32(&mut out, r.max_err);
            out.push(',');
            out.push_str(&r.param_count.to_string());
            out.push(',');
            out.push_str(&r.memory_bytes.to_string());
            out.push('\n');
        }
    }
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))
}

pub fn write_ablation_limits_csv(path: &Path, report: &AblationReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let n_ffts: Vec<usize> = if report.n_ffts.is_empty() {
        let mut ns: Vec<usize> = report.rows.iter().map(|r| r.n_fft).collect();
        ns.sort_unstable();
        ns.dedup();
        ns
    } else {
        report.n_ffts.clone()
    };
    let mut devices: Vec<String> = report.rows.iter().map(|r| r.device.clone()).collect();
    devices.sort();
    devices.dedup();

    let mut out = String::from(
        "n_fft,device,device_kind,ok_count,fail_count,max_batch,fastest_variant,fastest_ms,fastest_max_err,max_n_fft_rustfft\n",
    );
    for dev in &devices {
        let max_n = n_ffts
            .iter()
            .rev()
            .find(|&&n| {
                report.rows.iter().any(|r| {
                    r.device == *dev && r.n_fft == n && r.variant == "rustfft" && ablation_row_ok(r)
                })
            })
            .copied()
            .unwrap_or(0);
        let kind = if is_gpu_device_label(dev) {
            "gpu"
        } else {
            "cpu"
        };
        for &n in &n_ffts {
            let sub: Vec<_> = report
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
            if let Some(f) = fastest {
                out.push_str(&format!("{n},"));
                out.push_str(&csv_escape(dev));
                out.push(',');
                out.push_str(kind);
                out.push(',');
                out.push_str(&ok_n.to_string());
                out.push(',');
                out.push_str(&fail_n.to_string());
                out.push(',');
                out.push_str(&max_batch.to_string());
                out.push(',');
                out.push_str(&csv_escape(&f.variant));
                out.push(',');
                write_f64(&mut out, f.ms);
                out.push(',');
                write_f32(&mut out, f.max_err);
                out.push(',');
                out.push_str(&max_n.to_string());
                out.push('\n');
            } else {
                out.push_str(&format!(
                    "{n},{},{kind},0,{fail_n},0,,nan,nan,{max_n}\n",
                    csv_escape(dev)
                ));
            }
        }
    }
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))
}

/// Write all ablation CSV artifacts into a directory.
pub fn write_ablation_csv_dir(dir: &Path, report: &AblationReport) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    write_ablation_rows_csv(&dir.join(ROWS_CSV), &report.rows)?;
    write_ablation_meta_csv(&dir.join(META_CSV), report)?;
    write_ablation_top5_csv(&dir.join(TOP5_CSV), report)?;
    write_ablation_limits_csv(&dir.join(LIMITS_CSV), report)?;
    eprintln!(
        "wrote CSV bundle: {}/{} (+ {}, {})",
        dir.display(),
        ROWS_CSV,
        TOP5_CSV,
        LIMITS_CSV
    );
    Ok(())
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if !in_quotes => in_quotes = true,
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cur.push('"');
                } else {
                    in_quotes = false;
                }
            }
            ',' if !in_quotes => {
                fields.push(cur.clone());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    fields.push(cur);
    fields
}

pub fn read_ablation_rows_csv(path: &Path) -> Result<Vec<AblationRow>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut lines = text.lines();
    let header = lines.next().context("empty csv")?;
    ensure!(
        header.starts_with("tier,variant,direction"),
        "unexpected header in {}",
        path.display()
    );
    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f = split_csv_line(line);
        ensure!(f.len() >= 14, "bad csv row (expected 14 fields): {line}");
        rows.push(AblationRow {
            tier: f[0].clone(),
            variant: f[1].clone(),
            direction: f[2].clone(),
            n_fft: f[3].parse().context("n_fft")?,
            batch: f[4].parse().context("batch")?,
            device: f[5].clone(),
            iters: f[6].parse().context("iters")?,
            ms: parse_f64(&f[7]),
            max_err: parse_f32(&f[8]),
            train_steps: f[9].parse().context("train_steps")?,
            param_count: f[10].parse().unwrap_or(0),
            memory_bytes: f[11].parse().unwrap_or(0),
            status: if f[12].is_empty() {
                "ok".into()
            } else {
                f[12].clone()
            },
            note: if f[13].is_empty() {
                None
            } else {
                Some(f[13].clone())
            },
        });
    }
    Ok(rows)
}

pub fn read_ablation_meta_csv(path: &Path) -> Result<AblationReport> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut iters = 0usize;
    let mut train_steps = 0usize;
    let mut both_dirs = true;
    let mut with_welch = true;
    let mut limit_sweep = false;
    let mut elapsed_ms = 0.0;
    let mut n_ffts = Vec::new();
    for line in text.lines().skip(1) {
        let Some((k, v)) = line.split_once(',') else {
            continue;
        };
        match k {
            "iters" => iters = v.parse().unwrap_or(0),
            "train_steps" => train_steps = v.parse().unwrap_or(0),
            "both_dirs" => both_dirs = parse_bool(v),
            "with_welch" => with_welch = parse_bool(v),
            "limit_sweep" => limit_sweep = parse_bool(v),
            "elapsed_ms" => elapsed_ms = v.parse().unwrap_or(0.0),
            "n_ffts" => {
                let stripped = v.trim_matches('"');
                n_ffts = stripped
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            _ => {}
        }
    }
    Ok(AblationReport {
        iters,
        train_steps,
        both_dirs,
        with_welch,
        limit_sweep,
        n_ffts,
        elapsed_ms,
        rows: Vec::new(),
    })
}

pub fn read_ablation_csv_dir(dir: &Path) -> Result<AblationReport> {
    let rows_path = dir.join(ROWS_CSV);
    let meta_path = dir.join(META_CSV);
    ensure!(rows_path.is_file(), "missing {}", rows_path.display());
    let mut report = if meta_path.is_file() {
        read_ablation_meta_csv(&meta_path)?
    } else {
        AblationReport {
            iters: 0,
            train_steps: 0,
            both_dirs: true,
            with_welch: true,
            limit_sweep: false,
            n_ffts: Vec::new(),
            elapsed_ms: 0.0,
            rows: Vec::new(),
        }
    };
    report.rows = read_ablation_rows_csv(&rows_path)?;
    if report.n_ffts.is_empty() {
        let mut ns: Vec<usize> = report.rows.iter().map(|r| r.n_fft).collect();
        ns.sort_unstable();
        ns.dedup();
        report.n_ffts = ns;
    }
    Ok(report)
}

pub fn ablation_csv_bundle_paths(dir: &Path) -> [PathBuf; 4] {
    [
        dir.join(ROWS_CSV),
        dir.join(META_CSV),
        dir.join(TOP5_CSV),
        dir.join(LIMITS_CSV),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row() -> AblationRow {
        AblationRow {
            tier: "baseline".into(),
            variant: "rustfft".into(),
            direction: "Forward".into(),
            n_fft: 128,
            batch: 8,
            device: "metal".into(),
            iters: 5,
            ms: 0.01,
            max_err: 0.0,
            train_steps: 0,
            param_count: 896,
            memory_bytes: 3584,
            status: "ok".into(),
            note: None,
        }
    }

    #[test]
    fn csv_roundtrip_rows() {
        let dir = std::env::temp_dir().join("rlx_fft_ablation_csv_test");
        let _ = std::fs::remove_dir_all(&dir);
        let report = AblationReport {
            iters: 3,
            train_steps: 8,
            both_dirs: true,
            with_welch: true,
            limit_sweep: true,
            n_ffts: vec![64, 128],
            elapsed_ms: 99.0,
            rows: vec![sample_row()],
        };
        write_ablation_csv_dir(&dir, &report).unwrap();
        let loaded = read_ablation_csv_dir(&dir).unwrap();
        assert_eq!(loaded.rows.len(), 1);
        assert_eq!(loaded.rows[0].variant, "rustfft");
        assert_eq!(loaded.iters, 3);
        assert!(dir.join(TOP5_CSV).is_file());
        assert!(dir.join(LIMITS_CSV).is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

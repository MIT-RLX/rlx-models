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

//! Multi-dimensional benchmark sweep → JSON + ASCII chart.

use crate::bench::bench_all_dir;
use crate::config::{SUPPORTED_N_FFT, TransformDir};
use crate::device::resolve_train_device;
use anyhow::{Context, Result, ensure};
use rlx_runtime::{Device, is_available};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepRow {
    pub direction: String,
    pub n_fft: usize,
    pub batch: usize,
    pub device: String,
    pub iters: usize,
    pub implementation: String,
    pub ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_err: Option<f32>,
    pub butterfly_weights: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepReport {
    pub iters: usize,
    pub with_butterfly_compiled: bool,
    pub weights: Option<PathBuf>,
    pub elapsed_ms: f64,
    pub rows: Vec<SweepRow>,
}

pub fn parse_csv_usize(s: &str, label: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let v: usize = part
            .parse()
            .with_context(|| format!("{label} value {part}"))?;
        out.push(v);
    }
    ensure!(!out.is_empty(), "{label} must list at least one value");
    Ok(out)
}

/// Parse batch list: comma-separated values, or a range `1-1024` / `1..1024` (powers of two).
pub fn parse_batch_spec(s: &str, label: &str) -> Result<Vec<usize>> {
    parse_usize_spec(s, label)
}

/// Same grammar as [`parse_batch_spec`] — used for `--k` / top-K sweeps.
pub fn parse_k_spec(s: &str, label: &str) -> Result<Vec<usize>> {
    parse_usize_spec(s, label)
}

fn parse_usize_spec(s: &str, label: &str) -> Result<Vec<usize>> {
    let s = s.trim();
    if let Some((lo, hi)) = parse_usize_range(s) {
        let batches = power_of_two_inclusive(lo, hi);
        ensure!(
            !batches.is_empty(),
            "{label} range {lo}-{hi} contains no power-of-two batch sizes"
        );
        return Ok(batches);
    }
    parse_csv_usize(s, label)
}

fn parse_usize_range(s: &str) -> Option<(usize, usize)> {
    for sep in ["..=", "..", "-"] {
        if let Some((a, b)) = s.split_once(sep) {
            let lo = a.trim().parse().ok()?;
            let hi = b.trim().parse().ok()?;
            return Some((lo, hi));
        }
    }
    None
}

fn power_of_two_inclusive(lo: usize, hi: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut b = 1usize;
    while b < lo {
        b = b.saturating_mul(2);
        if b == 0 {
            break;
        }
    }
    while b <= hi {
        out.push(b);
        b = b.saturating_mul(2);
        if b == 0 {
            break;
        }
    }
    out
}

pub fn available_devices() -> Vec<&'static str> {
    let mut devices = vec!["cpu"];
    for (name, dev) in [
        ("metal", Device::Metal),
        ("mlx", Device::Mlx),
        ("ane", Device::Ane),
        ("cuda", Device::Cuda),
        ("rocm", Device::Rocm),
        ("wgpu", Device::Gpu),
        ("vulkan", Device::Vulkan),
    ] {
        if is_available(dev) {
            devices.push(name);
        }
    }
    devices
}

pub fn run_sweep(
    n_ffts: &[usize],
    batches: &[usize],
    devices: &[&str],
    directions: &[TransformDir],
    iters: usize,
    with_butterfly_compiled: bool,
    weights: Option<&Path>,
) -> Result<SweepReport> {
    ensure!(iters >= 1);
    let started = Instant::now();
    let mut rows = Vec::new();

    for dir in directions {
        for &n_fft in n_ffts {
            if !SUPPORTED_N_FFT.contains(&n_fft) {
                anyhow::bail!("unsupported n_fft={n_fft}; supported: {SUPPORTED_N_FFT:?}");
            }
            for &batch in batches {
                for device_name in devices {
                    eprintln!(
                        "[sweep] {:?} n_fft={n_fft} batch={batch} device={device_name}…",
                        dir
                    );
                    let device = resolve_train_device(Some(device_name))?;
                    // Skip batches beyond the device's hard ceiling (e.g. wgpu's
                    // 65535 dispatch cap) instead of letting the backend panic.
                    let cap = crate::max_batch::auto_max_fft_batch(
                        device,
                        crate::max_batch::FftProblem::f32(n_fft),
                    );
                    if batch > cap.max_batch {
                        eprintln!(
                            "[sweep] skip batch={batch} on {device_name}: exceeds max {} ({:?}-limited)",
                            cap.max_batch, cap.limited_by
                        );
                        continue;
                    }
                    let report = bench_all_dir(
                        n_fft,
                        batch,
                        iters,
                        *dir,
                        device,
                        with_butterfly_compiled,
                        weights,
                    )?;

                    rows.push(SweepRow {
                        direction: format!("{dir:?}"),
                        n_fft,
                        batch,
                        device: device_name.to_string(),
                        iters,
                        implementation: "rustfft".to_string(),
                        ms: report.rustfft_ms,
                        max_err: None,
                        butterfly_weights: report.butterfly_weights.clone(),
                    });
                    rows.push(SweepRow {
                        direction: format!("{dir:?}"),
                        n_fft,
                        batch,
                        device: device_name.to_string(),
                        iters,
                        implementation: if dir.is_forward() {
                            "rlx_op_fft".to_string()
                        } else {
                            "rlx_op_ifft".to_string()
                        },
                        ms: report.rlx_fft_ms,
                        max_err: Some(report.rlx_fft_err),
                        butterfly_weights: report.butterfly_weights.clone(),
                    });
                    rows.push(SweepRow {
                        direction: format!("{dir:?}"),
                        n_fft,
                        batch,
                        device: device_name.to_string(),
                        iters,
                        implementation: "butterfly_eager".to_string(),
                        ms: report.butterfly_eager_ms,
                        max_err: Some(report.butterfly_eager_err),
                        butterfly_weights: report.butterfly_weights.clone(),
                    });
                    if with_butterfly_compiled {
                        rows.push(SweepRow {
                            direction: format!("{dir:?}"),
                            n_fft,
                            batch,
                            device: device_name.to_string(),
                            iters,
                            implementation: "butterfly_compiled".to_string(),
                            ms: report.butterfly_compiled_ms,
                            max_err: Some(report.butterfly_compiled_err),
                            butterfly_weights: report.butterfly_weights.clone(),
                        });
                    }
                }
            }
        }
    }

    Ok(SweepReport {
        iters,
        with_butterfly_compiled,
        weights: weights.map(Path::to_path_buf),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        rows,
    })
}

pub fn write_sweep_json(path: &Path, report: &SweepReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("write {}", path.display()))
}

pub fn print_sweep_chart(report: &SweepReport) {
    let implementations = impl_order(report.with_butterfly_compiled);
    let devices: Vec<String> = report
        .rows
        .iter()
        .map(|r| r.device.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let directions: Vec<String> = report
        .rows
        .iter()
        .map(|r| r.direction.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    eprintln!(
        "\n=== FFT benchmark sweep (iters={}, compiled={}) ===\n",
        report.iters, report.with_butterfly_compiled
    );

    for dir in &directions {
        for device in &devices {
            eprintln!("--- {dir} @ {device} ---");
            let batches: Vec<usize> = report
                .rows
                .iter()
                .filter(|r| r.direction == *dir && r.device == *device)
                .map(|r| r.batch)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            let n_ffts: Vec<usize> = report
                .rows
                .iter()
                .filter(|r| r.direction == *dir && r.device == *device)
                .map(|r| r.n_fft)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();

            for &batch in &batches {
                eprintln!("\n  batch={batch}");
                print_header(&implementations);
                for &n_fft in &n_ffts {
                    let mut cells = Vec::new();
                    for imp in &implementations {
                        let ms = report
                            .rows
                            .iter()
                            .find(|r| {
                                r.direction == *dir
                                    && r.device == *device
                                    && r.batch == batch
                                    && r.n_fft == n_fft
                                    && r.implementation == *imp
                            })
                            .map(|r| r.ms)
                            .unwrap_or(f64::NAN);
                        cells.push(format_ms(ms));
                    }
                    eprintln!("  n={:<5} {}", n_fft, cells.join(" | "));
                }
            }
            eprintln!();
        }
    }

    eprintln!("Legend: ms/iter (lower is faster). rustfft is always CPU reference.");
    eprintln!("Total sweep time: {:.1} ms\n", report.elapsed_ms);
}

pub fn sweep_markdown_chart(report: &SweepReport) -> String {
    let implementations = impl_order(report.with_butterfly_compiled);
    let mut md = String::new();
    md.push_str("# RLX-FFT benchmark sweep\n\n");
    md.push_str(&format!(
        "iters={}, compiled={}, sweep_ms={:.0}\n\n",
        report.iters, report.with_butterfly_compiled, report.elapsed_ms
    ));

    let devices: Vec<String> = report
        .rows
        .iter()
        .map(|r| r.device.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for device in &devices {
        for dir in ["Forward", "Inverse"] {
            let subset: Vec<&SweepRow> = report
                .rows
                .iter()
                .filter(|r| r.device == *device && r.direction == dir)
                .collect();
            if subset.is_empty() {
                continue;
            }
            md.push_str(&format!("## {dir} — `{device}`\n\n"));
            md.push_str("| n_fft \\ batch |");
            let batches: Vec<usize> = subset
                .iter()
                .map(|r| r.batch)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            for b in &batches {
                md.push_str(&format!(" **b={b}** |"));
            }
            md.push('\n');
            md.push_str("| --- |");
            for _ in &batches {
                md.push_str(" --- |");
            }
            md.push('\n');

            for imp in &implementations {
                md.push_str(&format!("| **{imp}** |"));
                for batch in &batches {
                    let n_ffts: Vec<usize> = subset
                        .iter()
                        .filter(|r| r.batch == *batch && r.implementation == *imp)
                        .map(|r| r.n_fft)
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    let mut parts = Vec::new();
                    for n in n_ffts {
                        if let Some(r) = subset
                            .iter()
                            .find(|r| r.batch == *batch && r.n_fft == n && r.implementation == *imp)
                        {
                            parts.push(format!("n{n}: {:.4}ms", r.ms));
                        }
                    }
                    md.push_str(&format!(" {} |", parts.join("<br>")));
                }
                md.push('\n');
            }
            md.push('\n');
        }
    }
    md
}

fn impl_order(with_compiled: bool) -> Vec<&'static str> {
    let mut v = vec!["rustfft", "rlx_op_fft", "rlx_op_ifft", "butterfly_eager"];
    if with_compiled {
        v.push("butterfly_compiled");
    }
    v
}

fn print_header(implementations: &[&str]) {
    eprintln!(
        "  {:<8} {}",
        "n_fft",
        implementations
            .iter()
            .map(|s| format!("{s:>16}"))
            .collect::<Vec<_>>()
            .join(" | ")
    );
}

fn format_ms(ms: f64) -> String {
    if ms.is_nan() {
        "           n/a".to_string()
    } else {
        format!("{:>12.4}", ms)
    }
}

#[cfg(test)]
mod batch_spec_tests {
    use super::*;

    #[test]
    fn batch_range_pow2() {
        assert_eq!(
            parse_batch_spec("1-1024", "batch").unwrap(),
            vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        );
    }

    #[test]
    fn k_range_pow2() {
        assert_eq!(parse_k_spec("4-64", "--k").unwrap(), vec![4, 8, 16, 32, 64]);
    }
}

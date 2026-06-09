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

//! Ablation study across FFT variants (Tiers A/B/C).

use crate::bench_sweep::available_devices;
use crate::config::{
    FftLearnConfig, SUPPORTED_N_FFT, compiled_ok_for_limit_sweep, limit_sweep_batches,
    train_steps_for_n_fft, welch_ok_for_config, welch_ok_for_limit_sweep,
};
use crate::device::resolve_train_device;
use crate::rlx_fft::interleaved_to_block;
use crate::study_telemetry::variant_param_breakdown;
use crate::variants::{
    VariantState, bench_variant_ms, bench_variant_ms_inverse, bench_variant_ms_welch,
    fixed_ablation_signal, fixed_ablation_spectrum, fixed_ablation_welch_signal,
    variant_inverse_error, variant_spectrum_error, variant_welch_error, variants_for_direction,
    variants_for_welch,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationRow {
    pub tier: String,
    pub variant: String,
    pub direction: String,
    pub n_fft: usize,
    pub batch: usize,
    pub device: String,
    pub iters: usize,
    pub ms: f64,
    pub max_err: f32,
    pub train_steps: usize,
    #[serde(default)]
    pub param_count: usize,
    #[serde(default)]
    pub memory_bytes: usize,
    /// `ok`, `prepare_fail`, or `bench_fail`.
    #[serde(default = "default_status_ok")]
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

fn default_status_ok() -> String {
    "ok".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationReport {
    pub iters: usize,
    pub train_steps: usize,
    #[serde(default)]
    pub both_dirs: bool,
    #[serde(default = "default_with_welch")]
    pub with_welch: bool,
    #[serde(default)]
    pub limit_sweep: bool,
    #[serde(default)]
    pub n_ffts: Vec<usize>,
    pub elapsed_ms: f64,
    pub rows: Vec<AblationRow>,
}

fn default_with_welch() -> bool {
    true
}

pub fn run_ablation(
    n_ffts: &[usize],
    batches: &[usize],
    devices: &[&str],
    iters: usize,
    train_steps: usize,
    seed: u64,
    with_compiled: bool,
    both_dirs: bool,
    with_welch: bool,
) -> Result<AblationReport> {
    run_ablation_opts(
        n_ffts,
        batches,
        devices,
        iters,
        train_steps,
        seed,
        with_compiled,
        both_dirs,
        with_welch,
        false,
    )
}

/// Devices for limit sweeps: CPU + every GPU backend built into the binary.
pub fn limit_sweep_devices() -> Vec<String> {
    available_devices()
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// Full n_fft capacity sweep: adaptive batch per size, per-device compiled ceiling, records failures.
pub fn run_limit_sweep(
    devices: &[&str],
    iters: usize,
    train_steps: usize,
    seed: u64,
) -> Result<AblationReport> {
    let started = Instant::now();
    let n_ffts: Vec<usize> = SUPPORTED_N_FFT.to_vec();
    let mut rows = Vec::new();
    for &n_fft in &n_ffts {
        let batches = limit_sweep_batches(n_fft);
        let steps = train_steps_for_n_fft(train_steps, n_fft);
        let with_welch = welch_ok_for_limit_sweep(n_fft);
        for device_name in devices {
            let with_compiled = compiled_ok_for_limit_sweep(n_fft, device_name);
            eprintln!(
                "[limit-sweep] n_fft={n_fft} device={device_name} batches={batches:?} \
                 compiled={with_compiled} welch={with_welch} train_steps={steps}"
            );
            let sub = run_ablation_opts(
                &[n_fft],
                &batches,
                &[device_name],
                iters,
                steps,
                seed.wrapping_add(n_fft as u64)
                    .wrapping_add(device_name.len() as u64),
                with_compiled,
                true,
                with_welch,
                true,
            )?;
            rows.extend(sub.rows);
        }
    }
    Ok(AblationReport {
        iters,
        train_steps,
        both_dirs: true,
        with_welch: true,
        limit_sweep: true,
        n_ffts: n_ffts.clone(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        rows,
    })
}

fn run_ablation_opts(
    n_ffts: &[usize],
    batches: &[usize],
    devices: &[&str],
    iters: usize,
    train_steps: usize,
    seed: u64,
    with_compiled: bool,
    both_dirs: bool,
    with_welch: bool,
    record_failures: bool,
) -> Result<AblationReport> {
    ensure!(iters >= 1);
    let started = Instant::now();
    let mut rows = Vec::new();
    let mut directions: Vec<(&str, u8)> = vec![("Forward", 0)];
    if both_dirs {
        directions.push(("Inverse", 1));
    }
    if with_welch {
        directions.push(("Welch", 2));
    }

    for &n_fft in n_ffts {
        if !SUPPORTED_N_FFT.contains(&n_fft) {
            anyhow::bail!("unsupported n_fft={n_fft}");
        }
        for &batch in batches {
            for device_name in devices {
                let device = resolve_train_device(Some(device_name))?;
                let cfg = FftLearnConfig::new(n_fft, batch)?;
                let signal = fixed_ablation_signal(seed, batch, n_fft);
                let spectrum = fixed_ablation_spectrum(seed.wrapping_add(1), batch, n_fft);
                let spectrum_block = interleaved_to_block(&spectrum, batch, n_fft);
                let welch_signal = fixed_ablation_welch_signal(seed.wrapping_add(2), batch, n_fft);

                for &(dir_label, dir_kind) in &directions {
                    if dir_kind == 2 && !welch_ok_for_config(n_fft, batch) {
                        continue;
                    }
                    let variants = match dir_kind {
                        0 => variants_for_direction(with_compiled, true),
                        1 => variants_for_direction(with_compiled, false),
                        _ => variants_for_welch(with_compiled),
                    };
                    for &variant in &variants {
                        eprintln!(
                            "[ablation] {dir_label} {} n={n_fft} batch={batch} device={device_name}…",
                            variant.label()
                        );
                        let mut state = VariantState::new(&cfg);
                        state.set_inverse_spectrum(spectrum.clone());
                        state.set_inverse_input_block(spectrum_block.clone());
                        if let Err(e) = state.prepare(variant, &cfg, device, train_steps, seed) {
                            eprintln!("  skip {}: {e:#}", variant.label());
                            if record_failures {
                                let params = variant_param_breakdown(variant, &cfg);
                                rows.push(AblationRow {
                                    tier: variant.tier().to_string(),
                                    variant: variant.label().to_string(),
                                    direction: dir_label.to_string(),
                                    n_fft,
                                    batch,
                                    device: device_name.to_string(),
                                    iters,
                                    ms: f64::NAN,
                                    max_err: f32::NAN,
                                    train_steps: if variant.needs_training() {
                                        train_steps
                                    } else {
                                        0
                                    },
                                    param_count: params.total_params,
                                    memory_bytes: params.memory_bytes,
                                    status: "prepare_fail".into(),
                                    note: Some(format!("{e:#}")),
                                });
                            }
                            continue;
                        }
                        let (ms, max_err) = match dir_kind {
                            0 => {
                                let ms = match bench_variant_ms(
                                    &mut state, variant, &cfg, &signal, iters,
                                ) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        eprintln!("  bench fail {}: {e:#}", variant.label());
                                        f64::NAN
                                    }
                                };
                                let err =
                                    variant_spectrum_error(&mut state, variant, &cfg, &signal)
                                        .unwrap_or(f32::NAN);
                                (ms, err)
                            }
                            1 => {
                                let ms = match bench_variant_ms_inverse(
                                    &mut state, variant, &cfg, iters,
                                ) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        eprintln!("  bench fail {}: {e:#}", variant.label());
                                        f64::NAN
                                    }
                                };
                                let err = variant_inverse_error(&mut state, variant, &cfg)
                                    .unwrap_or(f32::NAN);
                                (ms, err)
                            }
                            _ => {
                                let ms = match bench_variant_ms_welch(
                                    &mut state,
                                    variant,
                                    &cfg,
                                    &welch_signal,
                                    iters,
                                ) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        eprintln!("  bench fail {}: {e:#}", variant.label());
                                        f64::NAN
                                    }
                                };
                                let err =
                                    variant_welch_error(&mut state, variant, &cfg, &welch_signal)
                                        .unwrap_or(f32::NAN);
                                (ms, err)
                            }
                        };
                        let params = variant_param_breakdown(variant, &cfg);
                        let status = if ms.is_nan() {
                            "bench_fail".to_string()
                        } else {
                            "ok".to_string()
                        };
                        rows.push(AblationRow {
                            tier: variant.tier().to_string(),
                            variant: variant.label().to_string(),
                            direction: dir_label.to_string(),
                            n_fft,
                            batch,
                            device: device_name.to_string(),
                            iters,
                            ms,
                            max_err,
                            train_steps: if variant.needs_training() {
                                train_steps
                            } else {
                                0
                            },
                            param_count: params.total_params,
                            memory_bytes: params.memory_bytes,
                            status,
                            note: None,
                        });
                    }
                }
            }
        }
    }

    Ok(AblationReport {
        iters,
        train_steps,
        both_dirs,
        with_welch,
        limit_sweep: false,
        n_ffts: n_ffts.to_vec(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        rows,
    })
}

pub fn write_ablation_json(path: &Path, report: &AblationReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("write {}", path.display()))
}

pub fn print_ablation_table(report: &AblationReport) {
    eprintln!(
        "\n=== FFT variant ablation (iters={}, train_steps={}, both_dirs={}, with_welch={}) ===\n",
        report.iters, report.train_steps, report.both_dirs, report.with_welch
    );
    let mut keys: Vec<(usize, usize, String, String)> = report
        .rows
        .iter()
        .map(|r| (r.n_fft, r.batch, r.device.clone(), r.direction.clone()))
        .collect();
    keys.sort();
    keys.dedup();

    for (n_fft, batch, device, direction) in keys {
        eprintln!("--- {direction} n_fft={n_fft} batch={batch} device={device} ---");
        eprintln!(
            "{:<24} {:>6} {:>10} {:>10}",
            "variant", "tier", "ms", "max_err"
        );
        let mut tier_rows: Vec<&AblationRow> = report
            .rows
            .iter()
            .filter(|r| {
                r.n_fft == n_fft
                    && r.batch == batch
                    && r.device == device
                    && r.direction == direction
            })
            .collect();
        tier_rows.sort_by(|a, b| a.ms.partial_cmp(&b.ms).unwrap_or(std::cmp::Ordering::Equal));
        for r in &tier_rows {
            eprintln!(
                "{:<24} {:>6} {:>10.4} {:>10.3e}",
                r.variant, r.tier, r.ms, r.max_err
            );
        }
        if let Some(best) = tier_rows.first() {
            eprintln!("  → fastest: {} ({:.4} ms)\n", best.variant, best.ms);
        }
    }
    eprintln!("Total ablation time: {:.1} ms\n", report.elapsed_ms);
}

pub fn ablation_winners(
    report: &AblationReport,
) -> Vec<(usize, usize, String, String, String, f64)> {
    let mut out = Vec::new();
    let mut keys: Vec<(usize, usize, String, String)> = report
        .rows
        .iter()
        .filter(|r| !r.ms.is_nan())
        .map(|r| (r.n_fft, r.batch, r.device.clone(), r.direction.clone()))
        .collect();
    keys.sort();
    keys.dedup();
    for (n, b, d, dir) in keys {
        let best = report
            .rows
            .iter()
            .filter(|r| {
                r.n_fft == n
                    && r.batch == b
                    && r.device == d
                    && r.direction == dir
                    && !r.ms.is_nan()
            })
            .min_by(|a, b| a.ms.partial_cmp(&b.ms).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(r) = best {
            out.push((n, b, d.clone(), dir, r.variant.clone(), r.ms));
        }
    }
    out
}

pub fn ablation_row_ok(r: &AblationRow) -> bool {
    r.status == "ok" && !r.ms.is_nan() && r.ms > 0.0
}

/// Best row per variant at each n_fft, then top 5 by ms (tie-break max_err).
pub fn top5_variants_per_n_fft(ab: &AblationReport) -> Vec<(usize, Vec<&AblationRow>)> {
    let mut n_ffts: Vec<usize> = ab.rows.iter().map(|r| r.n_fft).collect();
    n_ffts.sort_unstable();
    n_ffts.dedup();
    let mut out = Vec::new();
    for n in n_ffts {
        let mut best: HashMap<&str, &AblationRow> = HashMap::new();
        for r in &ab.rows {
            if r.n_fft != n || !ablation_row_ok(r) {
                continue;
            }
            let replace = match best.get(r.variant.as_str()) {
                None => true,
                Some(prev) => {
                    r.ms < prev.ms
                        || (r.ms == prev.ms
                            && r.max_err.partial_cmp(&prev.max_err)
                                == Some(std::cmp::Ordering::Less))
                }
            };
            if replace {
                best.insert(r.variant.as_str(), r);
            }
        }
        let mut rows: Vec<_> = best.into_values().collect();
        rows.sort_by(|a, b| {
            a.ms.partial_cmp(&b.ms)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.max_err
                        .partial_cmp(&b.max_err)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        rows.truncate(5);
        out.push((n, rows));
    }
    out
}

pub fn merge_ablation_reports(reports: &[AblationReport]) -> AblationReport {
    let mut rows = Vec::new();
    let mut n_fft_set = Vec::new();
    let mut elapsed_ms = 0.0;
    let first = reports.first();
    for rep in reports {
        rows.extend(rep.rows.iter().cloned());
        elapsed_ms += rep.elapsed_ms;
        for &n in &rep.n_ffts {
            if !n_fft_set.contains(&n) {
                n_fft_set.push(n);
            }
        }
    }
    n_fft_set.sort_unstable();
    AblationReport {
        iters: first.map(|r| r.iters).unwrap_or(0),
        train_steps: first.map(|r| r.train_steps).unwrap_or(0),
        both_dirs: first.map(|r| r.both_dirs).unwrap_or(true),
        with_welch: first.map(|r| r.with_welch).unwrap_or(true),
        limit_sweep: reports.iter().any(|r| r.limit_sweep),
        n_ffts: n_fft_set,
        elapsed_ms,
        rows,
    }
}

pub fn tier_summary(report: &AblationReport) -> std::collections::HashMap<String, usize> {
    let mut wins = std::collections::HashMap::new();
    for (_, _, _, _, variant, _) in ablation_winners(report) {
        let tier = report
            .rows
            .iter()
            .find(|r| r.variant == variant)
            .map(|r| r.tier.clone())
            .unwrap_or_else(|| "?".to_string());
        *wins.entry(tier).or_insert(0) += 1;
    }
    wins
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(variant: &str, n: usize, ms: f64) -> AblationRow {
        AblationRow {
            tier: "baseline".into(),
            variant: variant.into(),
            direction: "Forward".into(),
            n_fft: n,
            batch: 8,
            device: "cpu".into(),
            iters: 5,
            ms,
            max_err: 0.0,
            train_steps: 0,
            param_count: 100,
            memory_bytes: 400,
            status: "ok".into(),
            note: None,
        }
    }

    #[test]
    fn top5_picks_distinct_variants_per_n_fft() {
        let ab = AblationReport {
            iters: 5,
            train_steps: 0,
            both_dirs: true,
            with_welch: true,
            limit_sweep: false,
            n_ffts: vec![64, 128],
            elapsed_ms: 1.0,
            rows: vec![
                sample_row("rustfft", 64, 0.01),
                sample_row("rustfft", 64, 0.02),
                sample_row("rlx_op_fft", 64, 0.015),
                sample_row("butterfly_eager", 64, 0.012),
                sample_row("rustfft", 128, 0.02),
                sample_row("rlx_op_fft", 128, 0.018),
            ],
        };
        let top = top5_variants_per_n_fft(&ab);
        assert_eq!(top.len(), 2);
        let n64 = top.iter().find(|(n, _)| *n == 64).unwrap();
        assert_eq!(n64.1.len(), 3);
        assert_eq!(n64.1[0].variant, "rustfft");
        assert_eq!(n64.1[0].ms, 0.01);
    }

    #[test]
    fn ablation_includes_rlx_op_fft_forward() {
        let report = run_ablation(&[64], &[4], &["cpu"], 2, 0, 1, false, false, true).unwrap();
        assert!(
            report
                .rows
                .iter()
                .any(|r| r.variant == "rlx_op_fft" && r.direction == "Forward")
        );
        let rlx = report
            .rows
            .iter()
            .find(|r| r.variant == "rlx_op_fft" && r.n_fft == 64)
            .expect("rlx row");
        assert!(rlx.max_err < 1e-3, "rlx max_err={}", rlx.max_err);
    }

    #[test]
    fn ablation_rlx_op_ifft_inverse() {
        let report = run_ablation(&[64], &[4], &["cpu"], 2, 0, 1, false, true, true).unwrap();
        assert!(
            report
                .rows
                .iter()
                .any(|r| r.variant == "rlx_op_ifft" && r.direction == "Inverse")
        );
        let rlx = report
            .rows
            .iter()
            .find(|r| r.variant == "rlx_op_ifft")
            .expect("rlx ifft");
        assert!(rlx.max_err < 1e-3, "rlx ifft max_err={}", rlx.max_err);
    }

    #[test]
    fn ablation_includes_welch_variants() {
        let report = run_ablation(&[64], &[4], &["cpu"], 2, 0, 1, false, false, true).unwrap();
        assert!(
            report
                .rows
                .iter()
                .any(|r| r.variant == "welch_rustfft" && r.direction == "Welch")
        );
        let rlx = report
            .rows
            .iter()
            .find(|r| r.variant == "welch_rlx_op_fft")
            .expect("welch rlx");
        assert!(rlx.max_err < 1e-2, "welch rlx max_err={}", rlx.max_err);
    }
}

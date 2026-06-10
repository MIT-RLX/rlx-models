//! Benchmark Welch vs fast top-K peaks (rustfft + compiled + learned).

use crate::bench_sweep::{parse_batch_spec, parse_k_spec};
use crate::device::{ensure_backend_ready, resolve_train_device};
use crate::peak::{
    WelchPeakParams, WelchPeaksScratch, peak_max_err, welch_peaks_rustfft,
    welch_peaks_rustfft_with_scratch,
};
use crate::train::random_batch;
use crate::train_e2e::{E2eTrainConfig, train_fast_learned_model};
use crate::welch::{WelchParams, welch_rustfft};
use crate::welch_peaks_compile::{
    CompiledRlxWelchPeaksFused, compile_learned_welch_peaks, default_welch_peaks_hard_threshold,
};
use crate::welch_peaks_cost::{algorithm_bandwidth_gbps, useful_bytes_touched};
use crate::welch_peaks_picker::{AutoWelchPeaks, WelchPeaksPickMode, WelchPeaksStrategy};
use anyhow::{Context, Result, ensure};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WelchPeaksBenchRow {
    pub path: String,
    pub n_fft: usize,
    pub batch: usize,
    pub k: usize,
    pub device: String,
    pub iters: usize,
    pub ms: f64,
    pub output_len: usize,
    pub peak_err: Option<f32>,
    /// Achieved algorithm bandwidth (GB/s) from useful bytes / wall time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algo_bw_gbps: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WelchPeaksBenchReport {
    pub n_fft: usize,
    pub batch: usize,
    pub k: usize,
    pub elapsed_ms: f64,
    pub rows: Vec<WelchPeaksBenchRow>,
}

#[derive(Debug, Clone)]
pub struct WelchPeaksBenchOpts {
    pub n_fft: usize,
    pub batch: usize,
    pub k: usize,
    pub device_name: String,
    pub iters: usize,
    pub train_steps: usize,
    pub seed: u64,
    pub with_compiled: bool,
    pub with_ultra_fast: bool,
    /// `Auto` picks by batch/device; `Force(_)` overrides.
    pub pick_mode: WelchPeaksPickMode,
}

fn time_iters<F>(iters: usize, mut f: F) -> Result<f64>
where
    F: FnMut() -> Result<()>,
{
    for _ in 0..iters.saturating_sub(1) {
        f()?;
    }
    let t0 = Instant::now();
    f()?;
    Ok(t0.elapsed().as_secs_f64() * 1000.0)
}

pub fn run_welch_peaks_batch_sweep(
    opts: &WelchPeaksBenchOpts,
    batch_csv: &str,
) -> Result<WelchPeaksBenchReport> {
    run_welch_peaks_sweep(opts, batch_csv, &opts.k.to_string())
}

pub fn run_welch_peaks_k_sweep(
    opts: &WelchPeaksBenchOpts,
    k_csv: &str,
) -> Result<WelchPeaksBenchReport> {
    run_welch_peaks_sweep(opts, &opts.batch.to_string(), k_csv)
}

/// Sweep `--batch` and/or `--k` (Cartesian product when both are multi-valued).
pub fn run_welch_peaks_sweep(
    opts: &WelchPeaksBenchOpts,
    batch_csv: &str,
    k_csv: &str,
) -> Result<WelchPeaksBenchReport> {
    let batches = parse_batch_spec(batch_csv, "--batch")?;
    let ks = parse_k_spec(k_csv, "--k")?;
    ensure!(!batches.is_empty() && !ks.is_empty());
    let started = Instant::now();
    let mut all_rows = Vec::new();
    let mut last_n_fft = opts.n_fft;
    let mut last_batch = opts.batch;
    let mut last_k = opts.k;

    for &batch in &batches {
        for &k in &ks {
            let mut run_opts = opts.clone();
            run_opts.batch = batch;
            run_opts.k = k;
            if batch >= 4096 {
                run_opts.train_steps = 0;
                run_opts.iters = run_opts.iters.min(10);
            }
            let report = run_welch_peaks_bench_opts(&run_opts)?;
            last_n_fft = report.n_fft;
            last_batch = report.batch;
            last_k = report.k;
            all_rows.extend(report.rows);
        }
    }

    Ok(WelchPeaksBenchReport {
        n_fft: last_n_fft,
        batch: last_batch,
        k: last_k,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        rows: all_rows,
    })
}

pub fn run_welch_peaks_bench(
    n_fft: usize,
    batch: usize,
    k: usize,
    device_name: &str,
    iters: usize,
    train_steps: usize,
    seed: u64,
) -> Result<WelchPeaksBenchReport> {
    run_welch_peaks_bench_opts(&WelchPeaksBenchOpts {
        n_fft,
        batch,
        k,
        device_name: device_name.into(),
        iters,
        train_steps,
        seed,
        with_compiled: true,
        with_ultra_fast: true,
        pick_mode: WelchPeaksPickMode::Auto,
    })
}

pub fn run_welch_peaks_bench_opts(opts: &WelchPeaksBenchOpts) -> Result<WelchPeaksBenchReport> {
    ensure!(opts.iters >= 1 && opts.k >= 1);
    let started = Instant::now();
    let device = resolve_train_device(Some(&opts.device_name))?;
    ensure_backend_ready(device)?;

    let fast = WelchPeakParams::fast_for_n_fft(opts.n_fft, opts.k);
    let ultra = WelchPeakParams::ultra_fast_for_n_fft(opts.n_fft, opts.k);
    let full_welch = WelchParams::for_n_fft(opts.n_fft);
    let frame = full_welch.frame_len();

    let mut rng = StdRng::seed_from_u64(opts.seed);
    let signal = random_batch(&mut rng, opts.batch, frame);
    let fast_signal = fast.welch.truncate_batch(&signal, opts.batch, frame)?;
    let ultra_signal = ultra.welch.truncate_batch(&signal, opts.batch, frame)?;

    let ref_peaks = welch_peaks_rustfft(
        &signal,
        opts.batch,
        WelchPeakParams::reference_for_n_fft(opts.n_fft, opts.k),
    )?;

    let mut rows = Vec::new();
    let mut scratch = WelchPeaksScratch::new(opts.batch.max(1), fast.n_bins());

    let needs_model = opts.train_steps > 0
        || matches!(
            opts.pick_mode,
            WelchPeaksPickMode::Force(WelchPeaksStrategy::LearnedCompiled)
        );
    let model = if needs_model {
        let steps = opts
            .train_steps
            .max(if opts.train_steps == 0 { 100 } else { 0 });
        let (m, _) = train_fast_learned_model(&E2eTrainConfig {
            n_fft: opts.n_fft,
            batch: opts.batch,
            steps,
            seed: opts.seed,
            peak_k: opts.k,
            ..E2eTrainConfig::default()
        })?;
        Some(m)
    } else {
        None
    };

    // Full Welch PSD (dense, 8 segments).
    let ms = time_iters(opts.iters, || {
        let _ = welch_rustfft(&signal, opts.batch, full_welch)?;
        Ok(())
    })?;
    rows.push(WelchPeaksBenchRow {
        path: "welch_full_psd".into(),
        n_fft: opts.n_fft,
        batch: opts.batch,
        k: opts.k,
        device: opts.device_name.clone(),
        iters: opts.iters,
        ms,
        output_len: opts.batch * full_welch.n_bins(),
        peak_err: None,
        algo_bw_gbps: None,
    });

    // Fast Welch peaks — rustfft, 2 segments + top-K.
    let pred_fast = welch_peaks_rustfft(&fast_signal, opts.batch, fast)?;
    let err_fast = peak_max_err(&pred_fast, &ref_peaks);
    let ms = time_iters(opts.iters, || {
        let _ = welch_peaks_rustfft(&fast_signal, opts.batch, fast)?;
        Ok(())
    })?;
    rows.push(WelchPeaksBenchRow {
        path: "welch_fast_peaks_rustfft".into(),
        n_fft: opts.n_fft,
        batch: opts.batch,
        k: opts.k,
        device: opts.device_name.clone(),
        iters: opts.iters,
        ms,
        output_len: fast.output_len(opts.batch),
        peak_err: Some(err_fast),
        algo_bw_gbps: Some(algorithm_bandwidth_gbps(
            useful_bytes_touched(opts.batch, fast),
            ms,
        )),
    });

    // Streaming + scratch reuse (same math, less alloc).
    let pred_stream =
        welch_peaks_rustfft_with_scratch(&fast_signal, opts.batch, fast, Some(&mut scratch))?;
    ensure!(pred_stream == pred_fast);
    let ms = time_iters(opts.iters, || {
        let _ =
            welch_peaks_rustfft_with_scratch(&fast_signal, opts.batch, fast, Some(&mut scratch))?;
        Ok(())
    })?;
    rows.push(WelchPeaksBenchRow {
        path: "welch_fast_peaks_streaming".into(),
        n_fft: opts.n_fft,
        batch: opts.batch,
        k: opts.k,
        device: opts.device_name.clone(),
        iters: opts.iters,
        ms,
        output_len: fast.output_len(opts.batch),
        peak_err: Some(err_fast),
        algo_bw_gbps: Some(algorithm_bandwidth_gbps(
            useful_bytes_touched(opts.batch, fast),
            ms,
        )),
    });

    if opts.with_ultra_fast {
        let pred_ultra = welch_peaks_rustfft(&ultra_signal, opts.batch, ultra)?;
        let err_ultra = peak_max_err(&pred_ultra, &ref_peaks);
        let ms = time_iters(opts.iters, || {
            let _ = welch_peaks_rustfft(&ultra_signal, opts.batch, ultra)?;
            Ok(())
        })?;
        rows.push(WelchPeaksBenchRow {
            path: "welch_ultra_fast_peaks".into(),
            n_fft: opts.n_fft,
            batch: opts.batch,
            k: opts.k,
            device: opts.device_name.clone(),
            iters: opts.iters,
            ms,
            output_len: ultra.output_len(opts.batch),
            peak_err: Some(err_ultra),
            algo_bw_gbps: Some(algorithm_bandwidth_gbps(
                useful_bytes_touched(opts.batch, ultra),
                ms,
            )),
        });
    }

    // Auto or forced picker.
    {
        let model_ref = model.as_ref();
        let mut auto = AutoWelchPeaks::with_options(
            opts.batch,
            opts.n_fft,
            opts.k,
            Some(&opts.device_name),
            model_ref,
            opts.pick_mode,
        )?;
        let mode_label = if opts.pick_mode.is_auto() {
            "auto"
        } else {
            "forced"
        };
        eprintln!(
            "[welch-peaks] picker ({mode_label}): batch={} device={:?} -> {}",
            opts.batch,
            auto.device,
            auto.strategy_label()
        );
        let pred_auto = auto.welch_peaks_batch(&signal)?;
        let err_auto = peak_max_err(&pred_auto, &ref_peaks);
        let ms = time_iters(opts.iters, || {
            let _ = auto.welch_peaks_batch(&signal)?;
            Ok(())
        })?;
        rows.push(WelchPeaksBenchRow {
            path: format!("welch_peaks_picker_{}", auto.strategy_label()),
            n_fft: opts.n_fft,
            batch: opts.batch,
            k: opts.k,
            device: opts.device_name.clone(),
            iters: opts.iters,
            ms,
            output_len: auto.peak_params().output_len(opts.batch),
            peak_err: Some(err_auto),
            algo_bw_gbps: Some(algorithm_bandwidth_gbps(
                useful_bytes_touched(opts.batch, auto.peak_params()),
                ms,
            )),
        });
    }

    if opts.with_compiled {
        if let Ok(mut compiled) = CompiledRlxWelchPeaksFused::compile(opts.batch, fast, device) {
            let pred = compiled.welch_peaks_batch(&fast_signal)?;
            let err = peak_max_err(&pred, &ref_peaks);
            let ms = time_iters(opts.iters, || {
                let _ = compiled.welch_peaks_batch(&fast_signal)?;
                Ok(())
            })?;
            rows.push(WelchPeaksBenchRow {
                path: format!("welch_fast_peaks_rlx_{:?}", device).to_lowercase(),
                n_fft: opts.n_fft,
                batch: opts.batch,
                k: opts.k,
                device: opts.device_name.clone(),
                iters: opts.iters,
                ms,
                output_len: fast.output_len(opts.batch),
                peak_err: Some(err),
                algo_bw_gbps: Some(algorithm_bandwidth_gbps(
                    useful_bytes_touched(opts.batch, fast),
                    ms,
                )),
            });
        }
    }

    if let Some(model) = &model {
        let pred = model.welch_peaks_batch(&fast_signal, opts.batch, fast)?;
        let err = peak_max_err(&pred, &ref_peaks);
        let ms = time_iters(opts.iters, || {
            let _ = model.welch_peaks_batch(&fast_signal, opts.batch, fast)?;
            Ok(())
        })?;
        rows.push(WelchPeaksBenchRow {
            path: "learned_fast_peaks".into(),
            n_fft: opts.n_fft,
            batch: opts.batch,
            k: opts.k,
            device: opts.device_name.clone(),
            iters: opts.iters,
            ms,
            output_len: fast.output_len(opts.batch),
            peak_err: Some(err),
            algo_bw_gbps: Some(algorithm_bandwidth_gbps(
                useful_bytes_touched(opts.batch, fast),
                ms,
            )),
        });

        let hard = model.clone().with_hard_gates(0.5);
        let pred_h = hard.welch_peaks_batch(&fast_signal, opts.batch, fast)?;
        let err_h = peak_max_err(&pred_h, &ref_peaks);
        let ms = time_iters(opts.iters, || {
            let _ = hard.welch_peaks_batch(&fast_signal, opts.batch, fast)?;
            Ok(())
        })?;
        rows.push(WelchPeaksBenchRow {
            path: "learned_fast_peaks_hard_gates".into(),
            n_fft: opts.n_fft,
            batch: opts.batch,
            k: opts.k,
            device: opts.device_name.clone(),
            iters: opts.iters,
            ms,
            output_len: fast.output_len(opts.batch),
            peak_err: Some(err_h),
            algo_bw_gbps: Some(algorithm_bandwidth_gbps(
                useful_bytes_touched(opts.batch, fast),
                ms,
            )),
        });

        if opts.with_compiled {
            if let Ok(mut compiled) = compile_learned_welch_peaks(
                &hard,
                opts.batch,
                fast,
                device,
                default_welch_peaks_hard_threshold(),
            ) {
                let pred_c = compiled.welch_peaks_batch(&fast_signal, &mut scratch)?;
                let err_c = peak_max_err(&pred_c, &ref_peaks);
                let ms = time_iters(opts.iters, || {
                    let _ = compiled.welch_peaks_batch(&fast_signal, &mut scratch)?;
                    Ok(())
                })?;
                rows.push(WelchPeaksBenchRow {
                    path: format!("learned_fast_peaks_compiled_{:?}", compiled.run_device())
                        .to_lowercase(),
                    n_fft: opts.n_fft,
                    batch: opts.batch,
                    k: opts.k,
                    device: opts.device_name.clone(),
                    iters: opts.iters,
                    ms,
                    output_len: fast.output_len(opts.batch),
                    peak_err: Some(err_c),
                    algo_bw_gbps: Some(algorithm_bandwidth_gbps(
                        useful_bytes_touched(opts.batch, fast),
                        ms,
                    )),
                });
            }
        }
    }

    let _ = device;
    Ok(WelchPeaksBenchReport {
        n_fft: opts.n_fft,
        batch: opts.batch,
        k: opts.k,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        rows,
    })
}

pub fn print_welch_peaks_table(report: &WelchPeaksBenchReport) {
    let batches: Vec<usize> = report
        .rows
        .iter()
        .map(|r| r.batch)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let ks: Vec<usize> = report
        .rows
        .iter()
        .map(|r| r.k)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let multi = batches.len() > 1 || ks.len() > 1;

    eprintln!(
        "\n=== Welch peaks bench n_fft={} batch={} k={} ===\n",
        report.n_fft, report.batch, report.k
    );
    for r in &report.rows {
        let prefix = if multi {
            format!("[batch={} k={}] ", r.batch, r.k)
        } else {
            String::new()
        };
        eprintln!(
            "  {prefix}{:40} ms={:8.4} out={:6} peak_err={}",
            r.path,
            r.ms,
            r.output_len,
            r.peak_err
                .map(|e| format!("{e:.3e}"))
                .unwrap_or_else(|| "n/a".into())
        );
    }
    if !multi {
        let full = report
            .rows
            .iter()
            .find(|r| r.path == "welch_full_psd")
            .map(|r| r.ms);
        let fast = report
            .rows
            .iter()
            .find(|r| r.path == "welch_fast_peaks_rustfft")
            .map(|r| r.ms);
        if let (Some(f), Some(s)) = (full, fast) {
            eprintln!("\n  fast_peaks vs full_welch: {:.2}x", f / s);
        }
        if let Some(base) = fast {
            for r in &report.rows {
                if r.path.contains("compiled")
                    || r.path.contains("streaming")
                    || r.path.contains("ultra")
                {
                    eprintln!("  {} vs rustfft_fast: {:.2}x", r.path, base / r.ms);
                }
            }
        }
    }

    if batches.len() > 1 && ks.len() == 1 {
        eprintln!("\n  --- batch crossover (rustfft_fast vs auto picker) ---");
        for batch in &batches {
            let rust = report
                .rows
                .iter()
                .find(|r| r.batch == *batch && r.path == "welch_fast_peaks_rustfft");
            let picker = report
                .rows
                .iter()
                .find(|r| r.batch == *batch && r.path.starts_with("welch_peaks_picker_"));
            let rlx = report
                .rows
                .iter()
                .find(|r| r.batch == *batch && r.path.starts_with("welch_fast_peaks_rlx_"));
            if let (Some(r), Some(p)) = (rust, picker) {
                let ratio = r.ms / p.ms;
                let pick = p
                    .path
                    .strip_prefix("welch_peaks_picker_")
                    .unwrap_or(&p.path);
                eprintln!(
                    "  batch={batch:6} k={} rustfft={:.4}ms picker={:.4}ms ({pick}) ratio={ratio:.2}x {}",
                    r.k,
                    r.ms,
                    p.ms,
                    if ratio >= 1.0 {
                        "picker wins"
                    } else {
                        "rustfft wins"
                    }
                );
            }
            if let (Some(r), Some(x)) = (rust, rlx) {
                let ratio = r.ms / x.ms;
                eprintln!(
                    "  batch={batch:6} k={} rustfft={:.4}ms rlx_fused={:.4}ms ratio={ratio:.2}x {}",
                    r.k,
                    r.ms,
                    x.ms,
                    if ratio >= 1.0 {
                        "rlx wins"
                    } else {
                        "rustfft wins"
                    }
                );
            }
        }
    }

    if ks.len() > 1 {
        eprintln!("\n  --- k crossover (ms by path) ---");
        for batch in &batches {
            eprintln!("  batch={batch}:");
            for k in &ks {
                let rust = report.rows.iter().find(|r| {
                    r.batch == *batch && r.k == *k && r.path == "welch_fast_peaks_rustfft"
                });
                let stream = report.rows.iter().find(|r| {
                    r.batch == *batch && r.k == *k && r.path == "welch_fast_peaks_streaming"
                });
                let rlx = report.rows.iter().find(|r| {
                    r.batch == *batch && r.k == *k && r.path.starts_with("welch_fast_peaks_rlx_")
                });
                let picker = report.rows.iter().find(|r| {
                    r.batch == *batch && r.k == *k && r.path.starts_with("welch_peaks_picker_")
                });
                eprint!("    k={k:3}");
                if let Some(r) = rust {
                    eprint!(" rustfft={:.4}ms", r.ms);
                }
                if let Some(s) = stream {
                    eprint!(" stream={:.4}ms", s.ms);
                }
                if let Some(x) = rlx {
                    eprint!(" rlx={:.4}ms", x.ms);
                }
                if let Some(p) = picker {
                    eprint!(" picker={:.4}ms", p.ms);
                }
                if let (Some(r), Some(x)) = (rust, rlx) {
                    eprint!(" (rlx {:.2}x vs rustfft)", r.ms / x.ms);
                }
                eprintln!();
            }
        }
    }
    eprintln!("\nTotal: {:.1} ms\n", report.elapsed_ms);
}

pub fn write_welch_peaks_json(path: &Path, report: &WelchPeaksBenchReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("write {}", path.display()))
}

//! Phase-by-phase Welch peaks fusion benchmark (rlx + rlx-fft).

use crate::exec::device::{ensure_backend_ready, resolve_train_device};
use crate::spectral::peak::WelchPeakParams;
use crate::spectral::welch_peaks_compile::{
    CompiledRlxWelchPeaksFused, build_welch_peaks_fused_graph, compile_rlx_welch_peaks,
};
use crate::spectral::welch_peaks_cost::welch_peaks_fusion_target;
use crate::train::random_batch;
use anyhow::{Context, Result, ensure};
use rand::prelude::*;
use rlx_compile::fusion_benefit::fusion_benefit;
use rlx_compile::{FusionOptions, run_fusion_pipeline, supported_for_target, supports_op};
use rlx_ir::OpKind;
use rlx_runtime::graph_io::{GraphIoProfile, profile_graph_io};
use rlx_runtime::{Device, Session};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionPhaseRow {
    pub phase: String,
    pub batch: usize,
    pub k: usize,
    pub device: String,
    pub ms: f64,
    pub io: GraphIoProfile,
    pub predicted_cost_ns: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionPhaseReport {
    pub n_fft: usize,
    pub batch: usize,
    pub k: usize,
    pub device: String,
    pub fft_only_graph: GraphIoProfile,
    pub fused_graph: GraphIoProfile,
    pub fusion_launches_saved: isize,
    pub fusion_sync_saved: isize,
    pub fusion_readback_saved: i64,
    pub fusion_gate_fuse: bool,
    pub fusion_gate_score_ns: f64,
    pub fusion_gate_min_gain_ns: f64,
    pub fused_auto_viable: bool,
    pub rows: Vec<FusionPhaseRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionPhaseSweepReport {
    pub reports: Vec<FusionPhaseReport>,
}

pub fn run_fusion_phase_sweep(
    n_fft: usize,
    batch_csv: &str,
    k: usize,
    device_name: &str,
    iters: usize,
    seed: u64,
) -> Result<FusionPhaseSweepReport> {
    use crate::bench::sweep::parse_batch_spec;
    let batches = parse_batch_spec(batch_csv, "--batch")?;
    let mut reports = Vec::with_capacity(batches.len());
    for &batch in &batches {
        let mut run_iters = iters;
        if batch >= 4096 {
            run_iters = run_iters.min(15);
        }
        reports.push(run_fusion_phase_bench(
            n_fft,
            batch,
            k,
            device_name,
            run_iters,
            seed,
        )?);
    }
    Ok(FusionPhaseSweepReport { reports })
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

pub fn run_fusion_phase_bench(
    n_fft: usize,
    batch: usize,
    k: usize,
    device_name: &str,
    iters: usize,
    seed: u64,
) -> Result<FusionPhaseReport> {
    ensure!(iters >= 1 && k >= 1);
    let device = resolve_train_device(Some(device_name))?;
    ensure_backend_ready(device)?;

    let peak_params = WelchPeakParams::fast_for_n_fft(n_fft, k);
    let frame = peak_params.frame_len();

    let mut rng = StdRng::seed_from_u64(seed);
    let signal = random_batch(&mut rng, batch, frame);
    let mut scratch = crate::spectral::peak::WelchPeaksScratch::new(batch, peak_params.n_bins());

    let fft_graph = {
        let mut g = rlx_ir::Graph::new("fft_only");
        use rlx_ir::infer::GraphExt;
        let segs = g.input(
            "segs",
            rlx_ir::Shape::new(
                &[batch * peak_params.welch.n_segments, n_fft],
                rlx_ir::DType::F32,
            ),
        );
        let zeros = g.sub(segs, segs);
        let block = g.concat_(vec![segs, zeros], 1);
        let y = g.fft(block, false);
        g.set_outputs(vec![y]);
        g
    };
    let fused_graph = build_welch_peaks_fused_graph(batch, peak_params);
    let io_fft_raw = profile_graph_io(&fft_graph);
    let io_fused_raw = profile_graph_io(&fused_graph);
    let gate_bd = crate::spectral::welch_peaks_cost::welch_peaks_fusion_gate_breakdown(
        device, batch, n_fft, k,
    );
    let fusion_gate_fuse = gate_bd.should_fuse;
    let (io_fft, io_fused) =
        crate::spectral::welch_peaks_cost::welch_peaks_fusion_io_profiles(batch, n_fft, k, device);
    let benefit = fusion_benefit(&io_fft, &io_fused);

    let predicted_ns =
        crate::spectral::welch_peaks_cost::estimate_fused_graph_ns(batch, peak_params, device, 1.0);

    let mut rows = Vec::new();

    let mut legacy = compile_rlx_welch_peaks(batch, peak_params, device)?;
    let ms = time_iters(iters, || {
        let _ = legacy.welch_peaks_batch(&signal, &mut scratch)?;
        Ok(())
    })?;
    rows.push(FusionPhaseRow {
        phase: "baseline_interleaved_readback".into(),
        batch,
        k,
        device: device_name.into(),
        ms,
        io: io_fft_raw,
        predicted_cost_ns: predicted_ns,
    });

    let mut block_path = compile_rlx_welch_peaks(batch, peak_params, device)?;
    let ms = time_iters(iters, || {
        let _ = block_path.welch_peaks_batch_block(&signal, &mut scratch)?;
        Ok(())
    })?;
    rows.push(FusionPhaseRow {
        phase: "phase1_block_layout".into(),
        batch,
        k,
        device: device_name.into(),
        ms,
        io: io_fft_raw,
        predicted_cost_ns: predicted_ns,
    });

    let mut fused = CompiledRlxWelchPeaksFused::compile(batch, peak_params, device)?;
    let ms = time_iters(iters, || {
        let _ = fused.welch_peaks_batch(&signal)?;
        Ok(())
    })?;
    rows.push(FusionPhaseRow {
        phase: "phase2_fused_welch_peaks_op".into(),
        batch,
        k,
        device: device_name.into(),
        ms,
        io: io_fused_raw,
        predicted_cost_ns: predicted_ns,
    });

    let mut dual = build_welch_peaks_fused_graph(batch, peak_params);
    let peaks_id = dual.outputs[0];
    let spec_id = dual.node(peaks_id).inputs[0];
    dual.set_outputs(vec![spec_id, peaks_id]);
    let io_dual = profile_graph_io(&dual);
    let target = welch_peaks_fusion_target(device);
    let mut supported: Vec<OpKind> = supported_for_target(target).to_vec();
    if !supports_op(&supported, OpKind::Fft) {
        supported.push(OpKind::Fft);
    }
    if !supports_op(&supported, OpKind::WelchPeaks) {
        supported.push(OpKind::WelchPeaks);
    }
    let optimized = run_fusion_pipeline(dual, target, &supported, FusionOptions::default());
    let io_compile = profile_graph_io(&optimized);
    rows.push(FusionPhaseRow {
        phase: "phase3_compile_peaks_output_gate".into(),
        batch,
        k,
        device: device_name.into(),
        ms: 0.0,
        io: io_compile,
        predicted_cost_ns: 0.0,
    });
    if fusion_gate_fuse && io_dual.host_output_bytes > io_compile.host_output_bytes {
        let mut exec = Session::new(device).compile(optimized);
        let window = crate::spectral::welch::hann_window(peak_params.welch.n_fft);
        let ms = time_iters(iters, || {
            let segs = crate::spectral::welch::welch_windowed_segments(
                &signal,
                batch,
                peak_params.welch,
                &window,
            )?;
            let _ = exec.run(&[("segs", &segs)]);
            Ok(())
        })?;
        if let Some(row) = rows
            .iter_mut()
            .find(|r| r.phase == "phase3_compile_peaks_output_gate")
        {
            row.ms = ms;
        }
    }

    let _ = Session::new(device);

    Ok(FusionPhaseReport {
        n_fft,
        batch,
        k,
        device: device_name.into(),
        fft_only_graph: io_fft_raw,
        fused_graph: io_fused_raw,
        fusion_launches_saved: benefit.launches_saved,
        fusion_sync_saved: benefit.sync_points_saved,
        fusion_readback_saved: benefit.host_readback_bytes_saved,
        fusion_gate_fuse,
        fusion_gate_score_ns: gate_bd.score_ns,
        fusion_gate_min_gain_ns: gate_bd.min_gain_ns,
        fused_auto_viable: crate::spectral::welch_peaks_cost::fused_welch_peaks_auto_viable(device),
        rows,
    })
}

pub fn print_fusion_phase_report(report: &FusionPhaseReport) {
    eprintln!(
        "\n=== Fusion phase bench n_fft={} batch={} k={} device={} ===",
        report.n_fft, report.batch, report.k, report.device
    );
    eprintln!(
        "  IO fft-only: launches={} sync={} host_out={} B device_traffic={} B",
        report.fft_only_graph.kernel_launches,
        report.fft_only_graph.sync_points,
        report.fft_only_graph.host_output_bytes,
        report.fft_only_graph.device_traffic_bytes,
    );
    eprintln!(
        "  IO fused:    launches={} sync={} host_out={} B device_traffic={} B",
        report.fused_graph.kernel_launches,
        report.fused_graph.sync_points,
        report.fused_graph.host_output_bytes,
        report.fused_graph.device_traffic_bytes,
    );
    eprintln!(
        "  fusion benefit: launches_saved={} sync_saved={} readback_saved={} B \
         gate_score={:.3}ms min_gain={:.3}ms gate_fuse={} auto_viable={}",
        report.fusion_launches_saved,
        report.fusion_sync_saved,
        report.fusion_readback_saved,
        report.fusion_gate_score_ns / 1e6,
        report.fusion_gate_min_gain_ns / 1e6,
        report.fusion_gate_fuse,
        report.fused_auto_viable,
    );
    let base = report.rows.first().map(|r| r.ms).unwrap_or(1.0).max(1e-9);
    for r in &report.rows {
        eprintln!(
            "  {:32} ms={:8.4} speedup={:.2}x host_out={} B",
            r.phase,
            r.ms,
            base / r.ms.max(1e-9),
            r.io.host_output_bytes,
        );
    }
    if let Some(p2) = report
        .rows
        .iter()
        .find(|r| r.phase == "phase2_fused_welch_peaks_op")
    {
        if p2.predicted_cost_ns > 0.0 {
            let pred_ms = p2.predicted_cost_ns / 1e6;
            let ratio = p2.ms / pred_ms.max(1e-9);
            eprintln!(
                "  phase2 IO-model: predicted={pred_ms:.3}ms measured={:.3}ms ratio={ratio:.2}x",
                p2.ms,
            );
            if let Ok(device) = resolve_train_device(Some(&report.device)) {
                let cur = crate::spectral::welch_peaks_cost::fused_io_compute_scale_for_calibration(
                    device,
                );
                eprintln!(
                    "  suggested fused_io_compute_scale: {:.2} (current {:.2})",
                    cur * ratio,
                    cur,
                );
            }
        }
        let picker_pred_ms = crate::spectral::welch_peaks_cost::estimate_welch_peaks_costs(
            resolve_train_device(Some(&report.device)).unwrap_or(Device::Cpu),
            report.batch,
            report.n_fft,
            report.k,
            false,
            None,
            0,
        )
        .rlx_ns
            / 1e6;
        eprintln!("  picker rlx estimate: {picker_pred_ms:.3}ms (calibrated fused scale)",);
    }
    eprintln!();
}

pub fn print_fusion_phase_sweep_summary(sweep: &FusionPhaseSweepReport) {
    if sweep.reports.len() <= 1 {
        return;
    }
    eprintln!("=== Fusion phase crossover (speedup vs baseline) ===");
    eprintln!(
        "{:>8} {:>8} {:>8} {:>10} {:>6} {:>6}",
        "batch", "phase1", "phase2", "gate_ms", "fuse", "auto"
    );
    for r in &sweep.reports {
        let base = r.rows.first().map(|x| x.ms).unwrap_or(1.0).max(1e-9);
        let p1 = r
            .rows
            .iter()
            .find(|x| x.phase == "phase1_block_layout")
            .map(|x| base / x.ms.max(1e-9))
            .unwrap_or(0.0);
        let p2 = r
            .rows
            .iter()
            .find(|x| x.phase == "phase2_fused_welch_peaks_op")
            .map(|x| base / x.ms.max(1e-9))
            .unwrap_or(0.0);
        eprintln!(
            "{:>8} {:>7.2}x {:>7.2}x {:>9.2} {:>6} {:>6}",
            r.batch,
            p1,
            p2,
            r.fusion_gate_score_ns / 1e6,
            r.fusion_gate_fuse,
            r.fused_auto_viable,
        );
    }
    eprintln!();
}

pub fn write_fusion_phase_json(path: &std::path::Path, report: &FusionPhaseReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("write {}", path.display()))
}

pub fn write_fusion_phase_sweep_json(
    path: &std::path::Path,
    sweep: &FusionPhaseSweepReport,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(sweep)?)
        .with_context(|| format!("write {}", path.display()))
}

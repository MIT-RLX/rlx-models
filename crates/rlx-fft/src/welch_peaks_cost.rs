//! Ayala-style latency–bandwidth cost model for Welch peaks strategy selection.
//!
//! Uses `rlx_runtime::graph_io` IO profiles and per-backend [`BackendCostModel`]s:
//! `T ≈ L·M + S/W` (sync/launch × latency + bytes / effective bandwidth).

use crate::peak::WelchPeakParams;
use crate::welch_peaks_compile::build_welch_peaks_fused_graph;
use rlx_runtime::Device;
use rlx_runtime::cost::{BackendCostModel, estimate_graph_cost_with_io};
use rlx_runtime::graph_io::{GraphIoProfile, profile_graph_io};

/// Per-strategy predicted cost (nanoseconds) for debugging / benches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelchPeaksCostEstimates {
    pub ultra_ns: f64,
    pub fast_ns: f64,
    pub rlx_ns: f64,
    pub learned_ns: f64,
}

/// Useful bytes moved for algorithm-bandwidth reporting (segments in + peaks out).
pub fn useful_bytes_touched(batch: usize, params: WelchPeakParams) -> u64 {
    let n = params.welch.n_fft;
    let segs = params.welch.n_segments;
    let k = params.k;
    let seg_in = (batch * segs * n * 4) as u64;
    let peaks_out = (batch * k * 2 * 4) as u64;
    seg_in + peaks_out
}

/// Achieved algorithm bandwidth in GB/s from useful bytes and measured time (ms).
pub fn algorithm_bandwidth_gbps(useful_bytes: u64, time_ms: f64) -> f64 {
    if time_ms <= 0.0 {
        return 0.0;
    }
    let secs = time_ms / 1000.0;
    (useful_bytes as f64 / secs) / 1e9
}

/// Ayala transfer term: `M·L_dispatch + sync·L_roundtrip + S_device/W + S_host/W_rb`.
pub fn ayala_io_cost_ns(io: &GraphIoProfile, model: &dyn BackendCostModel) -> f64 {
    let mut t = model.roundtrip_overhead_ns();
    t += io.kernel_launches as f64 * model.dispatch_overhead_ns();
    t += io.sync_points as f64 * model.roundtrip_overhead_ns();
    t += io.device_traffic_bytes as f64 / model.memory_bw().max(1.0);
    t += io.host_readback_bytes(model.unified_memory()) as f64 / model.host_readback_bw().max(1.0);
    t
}

/// Static IO profile for CPU rustfft + streaming top-K (no RLX graph / no GPU sync).
pub fn rustfft_peaks_io_profile(batch: usize, params: WelchPeakParams) -> GraphIoProfile {
    let n = params.welch.n_fft;
    let segs = params.welch.n_segments;
    let k = params.k;
    let seg_rows = batch * segs;
    let spectrum_bytes = (seg_rows * n * 2 * 4) as u64;
    let segment_bytes = (seg_rows * n * 4) as u64;
    let peaks_bytes = (batch * k * 2 * 4) as u64;
    GraphIoProfile {
        kernel_launches: seg_rows + batch,
        sync_points: 0,
        host_output_bytes: peaks_bytes,
        device_traffic_bytes: segment_bytes.saturating_add(spectrum_bytes) + peaks_bytes,
    }
}

fn estimate_rustfft_peaks_ns(batch: usize, params: WelchPeakParams) -> f64 {
    let io = rustfft_peaks_io_profile(batch, params);
    #[cfg(feature = "cpu")]
    {
        let model = rlx_runtime::cost::CpuCostModel::new();
        ayala_io_cost_ns(&io, &model)
    }
    #[cfg(not(feature = "cpu"))]
    {
        let _ = (batch, params);
        // Fallback constants (GB/s as bytes/ns units match runtime cost models).
        let dispatch = 50.0;
        let roundtrip = 0.0;
        let bw = 50.0;
        let mut t = roundtrip;
        t += io.kernel_launches as f64 * dispatch;
        t += io.device_traffic_bytes as f64 / bw.max(1.0);
        t += io.host_output_bytes as f64 / bw.max(1.0);
        t
    }
}

fn legacy_ultra_fast_max_batch(device: Device) -> usize {
    if is_gpu_device(device) { 128 } else { 256 }
}

/// Current `fused_io_compute_scale` for a device (for `bench-fusion-phases` calibration hints).
pub fn fused_io_compute_scale_for_calibration(device: Device) -> f64 {
    fused_io_compute_scale(device)
}

/// IO-only graph cost under-estimates FFT + host `WelchPeaks` compute; scale from phase-2 bench.
fn fused_io_compute_scale(device: Device) -> f64 {
    match device {
        Device::Metal | Device::Mlx | Device::Ane => 7.5,
        // Rig phase-2: io-only ~69ms, measured fused ~28ms at batch 8192.
        Device::Cuda | Device::Rocm => 0.43,
        // Phase 5 native in-arena WelchPeaks (preliminary; tune via `rig.sh bench-rlx-fft-welch-peaks`).
        Device::Gpu | Device::Vulkan | Device::WebGpu | Device::DirectX | Device::OpenGl => 2.2,
        _ => 1.0,
    }
}

/// `(mid_batch ln coef, large_batch log2 coef)` for CPU rustfft vs GPU path on this device class.
fn rustfft_gpu_adjustment_coeffs(device: Device) -> (f64, f64) {
    match device {
        Device::Metal | Device::Mlx | Device::Ane => (0.22, 1.15),
        Device::Cuda | Device::Rocm => (0.15, 0.85),
        Device::Gpu | Device::Vulkan | Device::WebGpu | Device::DirectX | Device::OpenGl => {
            (0.12, 0.0)
        }
        _ => (0.0, 0.0),
    }
}

/// CPU rustfft vs GPU: unified-memory hosts see more cache contention at large batch.
fn rustfft_gpu_compare_adjustment(batch: usize, base_ns: f64, device: Device) -> f64 {
    if !is_gpu_device(device) {
        return base_ns;
    }
    let (mid_ln, large_log2) = rustfft_gpu_adjustment_coeffs(device);
    let mut ns = base_ns;
    if batch >= 512 && mid_ln > 0.0 {
        let log_b = ((batch as f64) / 512.0).ln().max(0.0);
        ns *= 1.0 + log_b * mid_ln;
    }
    if batch > 2048 && large_log2 > 0.0 {
        let log_b = ((batch as f64) / 2048.0).log2().max(0.0);
        ns *= 1.0 + log_b * large_log2;
    }
    ns
}

pub(crate) fn estimate_fused_graph_ns(
    batch: usize,
    params: WelchPeakParams,
    device: Device,
    compute_scale: f64,
) -> f64 {
    let graph = build_welch_peaks_fused_graph(batch, params);
    let io = profile_graph_io(&graph);
    let io_only = estimate_with_device(&graph, &io, device);
    let scale = fused_io_compute_scale(device) * compute_scale.clamp(0.25, 1.0);
    let scaled = io_only * scale;
    let small_batch_floor = if is_gpu_device(device) && batch < 512 {
        400_000.0
    } else {
        0.0
    };
    if scale >= 1.0 {
        scaled.max(io_only + small_batch_floor)
    } else {
        // Rig-calibrated scale < 1 (CUDA io-only over-estimates); do not clamp back to io_only.
        scaled.max(small_batch_floor)
    }
}

fn estimate_with_device(graph: &rlx_ir::Graph, io: &GraphIoProfile, device: Device) -> f64 {
    #[cfg(feature = "cpu")]
    if device == Device::Cpu {
        let model = rlx_runtime::cost::CpuCostModel::new();
        return estimate_graph_cost_with_io(graph, &model, io);
    }
    #[cfg(feature = "metal")]
    if device == Device::Metal {
        let model = rlx_runtime::cost::MetalCostModel::new();
        return estimate_graph_cost_with_io(graph, &model, io);
    }
    #[cfg(all(feature = "mlx", rlx_mlx_host))]
    if matches!(device, Device::Mlx | Device::Ane) {
        let model = rlx_runtime::cost::MlxCostModel::new();
        return estimate_graph_cost_with_io(graph, &model, io);
    }
    #[cfg(feature = "cuda")]
    if device == Device::Cuda {
        let model = rlx_runtime::cost::CudaCostModel::new();
        return estimate_graph_cost_with_io(graph, &model, io);
    }
    #[cfg(feature = "rocm")]
    if device == Device::Rocm {
        let model = rlx_runtime::cost::RocmCostModel::new();
        return estimate_graph_cost_with_io(graph, &model, io);
    }
    #[cfg(feature = "gpu")]
    if matches!(
        device,
        Device::Gpu | Device::Vulkan | Device::WebGpu | Device::DirectX | Device::OpenGl
    ) {
        let model = rlx_runtime::cost::WgpuCostModel::new();
        return estimate_graph_cost_with_io(graph, &model, io);
    }

    // No calibrated model for this device — IO-only Ayala estimate with discrete-GPU defaults.
    let io_only = GraphIoProfile {
        kernel_launches: io.kernel_launches,
        sync_points: io.sync_points,
        host_output_bytes: io.host_output_bytes,
        device_traffic_bytes: io.device_traffic_bytes,
    };
    struct DiscreteGpuModel;
    impl BackendCostModel for DiscreteGpuModel {
        fn device(&self) -> Device {
            Device::Cuda
        }
        fn sgemm_gflops(&self, _: usize, _: usize, _: usize) -> f64 {
            800.0
        }
        fn dispatch_overhead_ns(&self) -> f64 {
            2_000.0
        }
        fn roundtrip_overhead_ns(&self) -> f64 {
            20_000.0
        }
        fn memory_bw(&self) -> f64 {
            800.0
        }
        fn host_readback_bw(&self) -> f64 {
            50.0
        }
        fn unified_memory(&self) -> bool {
            false
        }
        fn num_threads(&self) -> usize {
            1
        }
    }
    let fallback = DiscreteGpuModel;
    ayala_io_cost_ns(&io_only, &fallback)
        + graph
            .nodes()
            .iter()
            .filter(|n| !matches!(n.op, rlx_ir::Op::Input { .. } | rlx_ir::Op::Param { .. }))
            .count() as f64
            * fallback.dispatch_overhead_ns()
}

fn learned_compute_scale(active: Option<usize>, total: usize) -> f64 {
    let Some(active) = active else {
        return 1.0;
    };
    if total == 0 {
        return 1.0;
    }
    let ratio = active as f64 / total as f64;
    0.30 + 0.70 * ratio
}

pub(crate) fn is_gpu_device(device: Device) -> bool {
    matches!(
        device,
        Device::Metal
            | Device::Mlx
            | Device::Cuda
            | Device::Rocm
            | Device::Gpu
            | Device::Vulkan
            | Device::DirectX
            | Device::WebGpu
            | Device::OpenGl
            | Device::Ane
            | Device::Tpu
    )
}

/// Compile fusion target for IO gate / bench reporting.
pub fn welch_peaks_fusion_target(device: Device) -> rlx_compile::FusionTarget {
    use rlx_compile::FusionTarget;
    match device {
        Device::Metal => FusionTarget::Metal,
        Device::Mlx | Device::Ane => FusionTarget::Mlx,
        Device::Cuda => FusionTarget::Cuda,
        Device::Rocm => FusionTarget::Rocm,
        Device::Gpu | Device::Vulkan | Device::WebGpu | Device::DirectX | Device::OpenGl => {
            FusionTarget::Wgpu
        }
        Device::Tpu => FusionTarget::Tpu,
        Device::Cpu => FusionTarget::Cpu,
    }
}

fn map_fusion_io(p: GraphIoProfile) -> rlx_compile::fusion_benefit::GraphIoProfile {
    rlx_compile::fusion_benefit::GraphIoProfile {
        kernel_launches: p.kernel_launches,
        sync_points: p.sync_points,
        host_output_bytes: p.host_output_bytes,
        device_traffic_bytes: p.device_traffic_bytes,
    }
}

fn fft_only_graph(batch: usize, n_fft: usize, n_segments: usize) -> rlx_ir::Graph {
    let mut g = rlx_ir::Graph::new("fft_only");
    use rlx_ir::infer::GraphExt;
    let segs = g.input(
        "segs",
        rlx_ir::Shape::new(&[batch * n_segments, n_fft], rlx_ir::DType::F32),
    );
    let zeros = g.sub(segs, segs);
    let block = g.concat_(vec![segs, zeros], 1);
    let y = g.fft(block, false);
    g.set_outputs(vec![y]);
    g
}

/// Net IO gate score (ns) for FFT→`WelchPeaks` fusion; compare to `min_gain_ns`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelchPeaksFusionGateBreakdown {
    pub score_ns: f64,
    pub min_gain_ns: f64,
    pub readback_saved_bytes: i64,
    pub sync_points_saved: isize,
    /// IO readback gate only (`rlx_compile::IoFusionGate`).
    pub should_fuse_io: bool,
    /// IO gate and fused path beats block RLX estimate.
    pub should_fuse: bool,
}

pub(crate) fn welch_peaks_fusion_io_profiles(
    batch: usize,
    n_fft: usize,
    k: usize,
    device: Device,
) -> (
    rlx_compile::fusion_benefit::GraphIoProfile,
    rlx_compile::fusion_benefit::GraphIoProfile,
) {
    let params = WelchPeakParams::fast_for_n_fft(n_fft, k);
    let io_fft = map_fusion_io(profile_graph_io(&fft_only_graph(
        batch,
        n_fft,
        params.welch.n_segments,
    )));
    let mut io_fused = map_fusion_io(profile_graph_io(&build_welch_peaks_fused_graph(
        batch, params,
    )));
    // Phase 5: native in-arena WelchPeaks on discrete GPU at large batch (no tail-host thunk).
    if batch >= 2048
        && matches!(
            device,
            Device::Gpu
                | Device::Vulkan
                | Device::WebGpu
                | Device::DirectX
                | Device::OpenGl
                | Device::Cuda
                | Device::Rocm
        )
    {
        io_fused.sync_points = io_fused.sync_points.saturating_sub(1);
    }
    (io_fft, io_fused)
}

/// Phase-3 IO gate breakdown (readback savings − host-thunk penalty per added sync).
pub fn welch_peaks_fusion_gate_breakdown(
    device: Device,
    batch: usize,
    n_fft: usize,
    k: usize,
) -> WelchPeaksFusionGateBreakdown {
    let (io_fft, io_fused) = welch_peaks_fusion_io_profiles(batch, n_fft, k, device);
    let gate = rlx_compile::io_fusion_gate_for_target(welch_peaks_fusion_target(device));
    let benefit = rlx_compile::fusion_benefit::fusion_benefit(&io_fft, &io_fused);
    let score_ns = gate.score_ns(&benefit);
    let target = welch_peaks_fusion_target(device);
    let should_fuse_io = rlx_compile::should_fuse_with_target(target, &io_fft, &io_fused);
    let compute_ok = welch_peaks_fusion_compute_floor_ok(device, batch, n_fft, k);
    WelchPeaksFusionGateBreakdown {
        score_ns,
        min_gain_ns: gate.min_gain_ns,
        readback_saved_bytes: benefit.host_readback_bytes_saved,
        sync_points_saved: benefit.sync_points_saved,
        should_fuse_io,
        should_fuse: should_fuse_io && compute_ok,
    }
}

/// Block FFT + host streaming top-K (phase-1 path) — Ayala estimate without full spectrum readback.
fn estimate_block_rlx_welch_ns(batch: usize, n_fft: usize, k: usize, device: Device) -> f64 {
    let params = WelchPeakParams::fast_for_n_fft(n_fft, k);
    let g = fft_only_graph(batch, n_fft, params.welch.n_segments);
    let io = profile_graph_io(&g);
    let mut ns = estimate_with_device(&g, &io, device);
    if is_gpu_device(device) {
        // Block layout avoids interleaved spectrum readback; add host top-K work.
        ns *= 0.72;
        ns += (batch * params.k) as f64 * 80.0;
    }
    ns
}

/// Fused path must beat block RLX estimate (IO readback alone can lie on tail-host backends).
pub fn welch_peaks_fusion_compute_floor_ok(
    device: Device,
    batch: usize,
    n_fft: usize,
    k: usize,
) -> bool {
    if !fused_welch_peaks_auto_viable(device) {
        return true;
    }
    // IO gate + host-thunk penalty cover mid batch; floor is for large-batch calibration.
    if batch < 2048 {
        return true;
    }
    let params = WelchPeakParams::fast_for_n_fft(n_fft, k);
    let fused = estimate_fused_graph_ns(batch, params, device, 1.0);
    let block = estimate_block_rlx_welch_ns(batch, n_fft, k, device);
    let slack = if matches!(device, Device::Cuda | Device::Rocm) && batch >= 8192 {
        1.15
    } else {
        1.10
    };
    fused <= block * slack
}

/// Phase-3 IO gate: readback benefit, host-thunk penalty, and compute floor vs block RLX.
pub fn welch_peaks_io_fusion_gate(device: Device, batch: usize, n_fft: usize, k: usize) -> bool {
    welch_peaks_fusion_gate_breakdown(device, batch, n_fft, k).should_fuse
}

/// Auto picker may use fused GPU path (tail-host WGPU is slower than rustfft in practice).
pub fn fused_welch_peaks_auto_viable(device: Device) -> bool {
    matches!(
        device,
        Device::Metal
            | Device::Mlx
            | Device::Ane
            | Device::Cuda
            | Device::Rocm
            | Device::Gpu
            | Device::Vulkan
            | Device::WebGpu
            | Device::DirectX
            | Device::OpenGl
    )
}

/// IO-aware cost estimates for each Welch peaks strategy (Ayala model).
pub fn estimate_welch_peaks_costs(
    device: Device,
    batch: usize,
    n_fft: usize,
    k: usize,
    learned_available: bool,
    learned_active_gates: Option<usize>,
    learned_total_gates: usize,
) -> WelchPeaksCostEstimates {
    let ultra_params = WelchPeakParams::ultra_fast_for_n_fft(n_fft, k);
    let fast_params = WelchPeakParams::fast_for_n_fft(n_fft, k);

    let mut ultra_ns = rustfft_gpu_compare_adjustment(
        batch,
        estimate_rustfft_peaks_ns(batch, ultra_params),
        device,
    );
    // On GPU-class targets, 1-segment ultra is only for small batch (latency floor).
    if is_gpu_device(device) && batch > legacy_ultra_fast_max_batch(device) {
        ultra_ns = f64::INFINITY;
    }
    let fast_ns = rustfft_gpu_compare_adjustment(
        batch,
        estimate_rustfft_peaks_ns(batch, fast_params),
        device,
    );

    let compile_gate_ok = welch_peaks_io_fusion_gate(device, batch, n_fft, k)
        && fused_welch_peaks_auto_viable(device);
    let rlx_ns = if is_gpu_device(device) && compile_gate_ok {
        estimate_fused_graph_ns(batch, fast_params, device, 1.0)
    } else {
        f64::INFINITY
    };

    let sparse_learned = learned_active_gates
        .map(|active| learned_total_gates > 0 && active * 4 < learned_total_gates)
        .unwrap_or(false);
    let learned_ns =
        if learned_available && sparse_learned && is_gpu_device(device) && compile_gate_ok {
            let scale = learned_compute_scale(learned_active_gates, learned_total_gates);
            estimate_fused_graph_ns(batch, fast_params, device, scale)
        } else {
            f64::INFINITY
        };

    WelchPeaksCostEstimates {
        ultra_ns,
        fast_ns,
        rlx_ns,
        learned_ns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_profile_peaks_smaller_than_spectrum() {
        let batch = 8192;
        let params = WelchPeakParams::fast_for_n_fft(256, 16);
        let fused = build_welch_peaks_fused_graph(batch, params);
        let io_fused = profile_graph_io(&fused);
        let io_rust = rustfft_peaks_io_profile(batch, params);
        assert!(io_fused.host_output_bytes < io_rust.host_output_bytes * 4);
    }

    #[test]
    fn fused_peaks_output_smaller_than_full_spectrum() {
        let batch = 8192;
        let params = WelchPeakParams::fast_for_n_fft(256, 16);
        let mut g = rlx_ir::Graph::new("fft_out");
        use rlx_ir::infer::GraphExt;
        let segs = g.input(
            "segs",
            rlx_ir::Shape::new(
                &[batch * params.welch.n_segments, params.welch.n_fft],
                rlx_ir::DType::F32,
            ),
        );
        let zeros = g.sub(segs, segs);
        let block = g.concat_(vec![segs, zeros], 1);
        let spec = g.fft(block, false);
        g.set_outputs(vec![spec]);
        let full_spec = profile_graph_io(&g);
        let fused = profile_graph_io(&build_welch_peaks_fused_graph(batch, params));
        assert!(fused.host_output_bytes < full_spec.host_output_bytes);
    }

    #[test]
    fn fusion_gate_batch_matrix() {
        let small = welch_peaks_fusion_gate_breakdown(Device::Metal, 256, 256, 16);
        assert!(
            !small.should_fuse,
            "small batch: host-thunk penalty dominates (io={})",
            small.should_fuse_io
        );

        for &batch in &[1024usize, 4096, 8192] {
            let metal = welch_peaks_fusion_gate_breakdown(Device::Metal, batch, 256, 16);
            let gpu = welch_peaks_fusion_gate_breakdown(Device::Gpu, batch, 256, 16);
            eprintln!(
                "batch={batch} metal score={:.3}ms fuse={} gpu score={:.3}ms fuse={}",
                metal.score_ns / 1e6,
                metal.should_fuse,
                gpu.score_ns / 1e6,
                gpu.should_fuse,
            );
            assert!(metal.should_fuse, "metal batch={batch}");
            if batch >= 8192 {
                assert!(gpu.should_fuse, "gpu batch={batch} (native WelchPeaks)");
            }
        }
    }

    #[test]
    fn io_gate_favors_fusion_on_metal() {
        assert!(welch_peaks_io_fusion_gate(Device::Metal, 8192, 256, 16));
    }

    #[test]
    fn wgpu_large_batch_native_gpu_profile() {
        let bd = welch_peaks_fusion_gate_breakdown(Device::Gpu, 8192, 256, 16);
        assert!(bd.readback_saved_bytes > 0);
        assert_eq!(
            bd.sync_points_saved, 0,
            "phase 5 large batch: no extra tail-host sync"
        );
        assert!(fused_welch_peaks_auto_viable(Device::Gpu));
    }

    #[test]
    fn wgpu_fused_auto_viable() {
        assert!(fused_welch_peaks_auto_viable(Device::Gpu));
    }

    #[test]
    fn algorithm_bw_positive() {
        let bytes = useful_bytes_touched(32, WelchPeakParams::fast_for_n_fft(256, 16));
        assert!(algorithm_bandwidth_gbps(bytes, 1.0) > 0.0);
    }

    /// Print IO-model components for CUDA calibration (`cargo test … -- --nocapture`).
    #[test]
    #[cfg(feature = "cuda")]
    fn print_cuda_fused_cost_breakdown() {
        use super::estimate_welch_peaks_costs;
        for batch in [256usize, 8192] {
            let fused = estimate_fused_graph_ns(
                batch,
                WelchPeakParams::fast_for_n_fft(256, 16),
                Device::Cuda,
                1.0,
            );
            let block = estimate_block_rlx_welch_ns(batch, 256, 16, Device::Cuda);
            let bd = welch_peaks_fusion_gate_breakdown(Device::Cuda, batch, 256, 16);
            let costs = estimate_welch_peaks_costs(Device::Cuda, batch, 256, 16, false, None, 0);
            eprintln!(
                "cuda batch={batch} scale={:.2} fast={:.3}ms fused={:.3}ms block={:.3}ms io={} floor={} fuse={} rlx={:.3}ms pick={:?}",
                crate::welch_peaks_cost::fused_io_compute_scale_for_calibration(Device::Cuda),
                costs.fast_ns / 1e6,
                fused / 1e6,
                block / 1e6,
                bd.should_fuse_io,
                welch_peaks_fusion_compute_floor_ok(Device::Cuda, batch, 256, 16),
                bd.should_fuse,
                costs.rlx_ns / 1e6,
                crate::welch_peaks_picker::pick_welch_peaks_strategy(
                    Device::Cuda,
                    batch,
                    256,
                    16,
                    false,
                    None,
                    0,
                ),
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_large_batch_fusion_gate() {
        let bd = welch_peaks_fusion_gate_breakdown(Device::Cuda, 8192, 256, 16);
        assert!(bd.readback_saved_bytes > 0);
        assert_eq!(
            bd.sync_points_saved, 0,
            "phase 5: native kernel, no tail-host sync"
        );
        assert!(bd.should_fuse_io, "readback savings dominate at batch 8192");
        assert!(
            welch_peaks_io_fusion_gate(Device::Cuda, 8192, 256, 16),
            "score={:.3}ms io={} floor={}",
            bd.score_ns / 1e6,
            bd.should_fuse_io,
            welch_peaks_fusion_compute_floor_ok(Device::Cuda, 8192, 256, 16),
        );
    }

    /// Print IO-model components for WGPU calibration (`cargo test … -- --nocapture`).
    #[test]
    #[cfg(feature = "gpu")]
    fn print_wgpu_fused_cost_breakdown() {
        use super::estimate_welch_peaks_costs;
        for batch in [256usize, 1024, 4096, 8192] {
            let fused = estimate_fused_graph_ns(
                batch,
                WelchPeakParams::fast_for_n_fft(256, 16),
                Device::Gpu,
                1.0,
            );
            let block = estimate_block_rlx_welch_ns(batch, 256, 16, Device::Gpu);
            let bd = welch_peaks_fusion_gate_breakdown(Device::Gpu, batch, 256, 16);
            let costs = estimate_welch_peaks_costs(Device::Gpu, batch, 256, 16, false, None, 0);
            eprintln!(
                "wgpu batch={batch} fast={:.3}ms fused={:.3}ms block={:.3}ms io={} floor={} fuse={} rlx={:.3}ms pick={:?}",
                costs.fast_ns / 1e6,
                fused / 1e6,
                block / 1e6,
                bd.should_fuse_io,
                welch_peaks_fusion_compute_floor_ok(Device::Gpu, batch, 256, 16),
                bd.should_fuse,
                costs.rlx_ns / 1e6,
                crate::welch_peaks_picker::pick_welch_peaks_strategy(
                    Device::Gpu,
                    batch,
                    256,
                    16,
                    false,
                    None,
                    0,
                ),
            );
        }
    }

    /// Print IO-model components for Metal calibration (`cargo test … -- --nocapture`).
    #[test]
    #[cfg(feature = "metal")]
    fn print_metal_fused_cost_breakdown() {
        use super::estimate_welch_peaks_costs;
        use rlx_runtime::cost::{MetalCostModel, estimate_graph_cost_with_io};
        let model = MetalCostModel::new();
        for batch in [256usize, 1024, 4096, 8192] {
            let params = WelchPeakParams::fast_for_n_fft(256, 16);
            let graph = build_welch_peaks_fused_graph(batch, params);
            let io = profile_graph_io(&graph);
            let io_only = estimate_graph_cost_with_io(&graph, &model, &io);
            let costs = estimate_welch_peaks_costs(Device::Metal, batch, 256, 16, false, None, 0);
            eprintln!(
                "batch={batch} io_only={:.3}ms rlx={:.3}ms fast={:.3}ms pick={:?}",
                io_only / 1e6,
                costs.rlx_ns / 1e6,
                costs.fast_ns / 1e6,
                crate::welch_peaks_picker::pick_welch_peaks_strategy(
                    Device::Metal,
                    batch,
                    256,
                    16,
                    false,
                    None,
                    0,
                ),
            );
        }
    }
}

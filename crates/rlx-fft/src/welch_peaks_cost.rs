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

/// IO-only graph cost under-estimates FFT + host `WelchPeaks` compute; scale from phase-2 bench.
fn fused_io_compute_scale(device: Device) -> f64 {
    match device {
        Device::Metal | Device::Mlx | Device::Ane => 7.5,
        Device::Cuda | Device::Rocm => 9.0,
        Device::Gpu | Device::Vulkan | Device::WebGpu | Device::DirectX | Device::OpenGl => 6.5,
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

fn estimate_fused_graph_ns(
    batch: usize,
    params: WelchPeakParams,
    device: Device,
    compute_scale: f64,
) -> f64 {
    let graph = build_welch_peaks_fused_graph(batch, params);
    let io = profile_graph_io(&graph);
    let io_only = estimate_with_device(&graph, &io, device);
    let scale = fused_io_compute_scale(device) * compute_scale.clamp(0.25, 1.0);
    let small_batch_floor = if is_gpu_device(device) && batch < 512 {
        400_000.0
    } else {
        0.0
    };
    (io_only * scale).max(io_only + small_batch_floor)
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

    let rlx_ns = if is_gpu_device(device) {
        estimate_fused_graph_ns(batch, fast_params, device, 1.0)
    } else {
        f64::INFINITY
    };

    let sparse_learned = learned_active_gates
        .map(|active| learned_total_gates > 0 && active * 4 < learned_total_gates)
        .unwrap_or(false);
    let learned_ns = if learned_available && sparse_learned && is_gpu_device(device) {
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
    fn algorithm_bw_positive() {
        let bytes = useful_bytes_touched(32, WelchPeakParams::fast_for_n_fft(256, 16));
        assert!(algorithm_bandwidth_gbps(bytes, 1.0) > 0.0);
    }

    /// Print IO-model components for WGPU calibration (`cargo test … -- --nocapture`).
    #[test]
    #[cfg(feature = "gpu")]
    fn print_wgpu_fused_cost_breakdown() {
        use super::estimate_welch_peaks_costs;
        for batch in [256usize, 1024, 4096, 8192] {
            let costs = estimate_welch_peaks_costs(Device::Gpu, batch, 256, 16, false, None, 0);
            eprintln!(
                "wgpu batch={batch} fast={:.3}ms rlx={:.3}ms pick={:?}",
                costs.fast_ns / 1e6,
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

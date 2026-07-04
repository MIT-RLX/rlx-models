//! Automatic Welch peaks path — pick fastest strategy from batch size + device.
//!
//! See `crates/rlx-fft/README.md` (Welch peaks section) for CLI flags, strategy
//! table, and examples. Use [`AutoWelchPeaks`] (auto or forced via [`WelchPeaksPickMode`]) or
//! [`parse_welch_peaks_strategy`] for string config.

use crate::exec::device::{ensure_backend_ready, resolve_train_device};
use crate::learned::model::FastLearnedFftModel;
use crate::model::pruned::DEFAULT_GATE_THRESHOLD;
use crate::spectral::peak::{WelchPeakParams, WelchPeaksScratch, welch_peaks_rustfft_with_scratch};
use crate::spectral::welch::WelchParams;
use crate::spectral::welch_peaks_compile::{
    CompiledLearnedWelchPeaks, CompiledRlxWelchPeaksExec, compile_learned_welch_peaks,
    default_welch_peaks_hard_threshold,
};
use crate::spectral::welch_peaks_cost::{
    WelchPeaksCostEstimates, estimate_welch_peaks_costs, is_gpu_device,
};
use anyhow::{Result, bail, ensure};
use rlx_runtime::Device;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WelchPeaksStrategy {
    /// 1 Welch segment + rustfft + streaming top-K.
    UltraFast,
    /// 2 segments + rustfft + streaming top-K.
    FastStreaming,
    /// 2 segments + compiled RLX `Op::Fft` on GPU.
    RlxCompiled,
    /// 2 segments + compiled learned spectrum (sparse gates + large batch).
    LearnedCompiled,
}

impl WelchPeaksStrategy {
    pub fn label(self) -> &'static str {
        match self {
            Self::UltraFast => "ultra_fast_rustfft",
            Self::FastStreaming => "fast_streaming_rustfft",
            Self::RlxCompiled => "rlx_compiled",
            Self::LearnedCompiled => "learned_compiled",
        }
    }
}

/// Batch threshold where RLX compiled peaks beat rustfft (legacy reference; IO picker may differ).
pub fn rlx_crossover_batch(device: Device) -> usize {
    if is_gpu_device(device) {
        8192
    } else {
        usize::MAX
    }
}

/// Small batches use 1-segment ultra-fast path on CPU (legacy reference).
pub fn ultra_fast_max_batch(device: Device) -> usize {
    if is_gpu_device(device) { 128 } else { 256 }
}

/// Per-strategy IO cost estimates + picked strategy (Ayala latency–bandwidth model).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelchPeaksPickBreakdown {
    pub costs: WelchPeaksCostEstimates,
    pub picked: WelchPeaksStrategy,
}

fn legacy_pick_welch_peaks_strategy(
    device: Device,
    batch: usize,
    learned_available: bool,
    learned_active_gates: Option<usize>,
    learned_total_gates: usize,
) -> WelchPeaksStrategy {
    let sparse_learned = learned_active_gates
        .map(|active| learned_total_gates > 0 && active * 4 < learned_total_gates)
        .unwrap_or(false);

    if learned_available
        && sparse_learned
        && batch >= rlx_crossover_batch(device)
        && is_gpu_device(device)
    {
        return WelchPeaksStrategy::LearnedCompiled;
    }

    if batch >= rlx_crossover_batch(device) && is_gpu_device(device) {
        return WelchPeaksStrategy::RlxCompiled;
    }

    if batch <= ultra_fast_max_batch(device) {
        return WelchPeaksStrategy::UltraFast;
    }

    WelchPeaksStrategy::FastStreaming
}

fn pick_from_costs(costs: WelchPeaksCostEstimates) -> WelchPeaksStrategy {
    let mut best = WelchPeaksStrategy::UltraFast;
    let mut best_ns = costs.ultra_ns;

    if costs.fast_ns < best_ns {
        best_ns = costs.fast_ns;
        best = WelchPeaksStrategy::FastStreaming;
    }
    if costs.rlx_ns < best_ns {
        best_ns = costs.rlx_ns;
        best = WelchPeaksStrategy::RlxCompiled;
    }
    if costs.learned_ns < best_ns {
        best = WelchPeaksStrategy::LearnedCompiled;
    }
    best
}

/// IO-aware strategy breakdown (nanoseconds per path + pick).
pub fn pick_welch_peaks_breakdown(
    device: Device,
    batch: usize,
    n_fft: usize,
    k: usize,
    learned_available: bool,
    learned_active_gates: Option<usize>,
    learned_total_gates: usize,
) -> WelchPeaksPickBreakdown {
    if rlx_ir::env::flag("RLX_FFT_LEGACY_PICKER") {
        return WelchPeaksPickBreakdown {
            costs: WelchPeaksCostEstimates {
                ultra_ns: 0.0,
                fast_ns: 0.0,
                rlx_ns: 0.0,
                learned_ns: 0.0,
            },
            picked: legacy_pick_welch_peaks_strategy(
                device,
                batch,
                learned_available,
                learned_active_gates,
                learned_total_gates,
            ),
        };
    }
    let costs = estimate_welch_peaks_costs(
        device,
        batch,
        n_fft,
        k,
        learned_available,
        learned_active_gates,
        learned_total_gates,
    );
    WelchPeaksPickBreakdown {
        picked: pick_from_costs(costs),
        costs,
    }
}

/// Choose the fastest Welch-peaks strategy for `batch` on `device`.
pub fn pick_welch_peaks_strategy(
    device: Device,
    batch: usize,
    n_fft: usize,
    k: usize,
    learned_available: bool,
    learned_active_gates: Option<usize>,
    learned_total_gates: usize,
) -> WelchPeaksStrategy {
    pick_welch_peaks_breakdown(
        device,
        batch,
        n_fft,
        k,
        learned_available,
        learned_active_gates,
        learned_total_gates,
    )
    .picked
}

/// `auto` picks by batch/device; otherwise force a specific path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WelchPeaksPickMode {
    Auto,
    Force(WelchPeaksStrategy),
}

impl WelchPeaksPickMode {
    pub fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }
}

/// Parse CLI / config: `auto`, `ultra`, `fast`, `rlx`, `learned` (aliases accepted).
pub fn parse_welch_peaks_strategy(name: &str) -> Result<WelchPeaksPickMode> {
    match name.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "auto" => Ok(WelchPeaksPickMode::Auto),
        "ultra" | "ultra_fast" | "ultra_fast_rustfft" | "1seg" => {
            Ok(WelchPeaksPickMode::Force(WelchPeaksStrategy::UltraFast))
        }
        "fast" | "streaming" | "fast_streaming" | "fast_streaming_rustfft" | "rustfft" | "2seg" => {
            Ok(WelchPeaksPickMode::Force(WelchPeaksStrategy::FastStreaming))
        }
        "rlx" | "rlx_compiled" | "compiled" | "gpu" => {
            Ok(WelchPeaksPickMode::Force(WelchPeaksStrategy::RlxCompiled))
        }
        "learned" | "learned_compiled" => Ok(WelchPeaksPickMode::Force(
            WelchPeaksStrategy::LearnedCompiled,
        )),
        other => bail!("unknown welch peaks strategy {other:?} (try auto|ultra|fast|rlx|learned)"),
    }
}

pub fn all_welch_peaks_strategy_names() -> &'static [&'static str] {
    &["auto", "ultra", "fast", "rlx", "learned"]
}

/// Resolve auto vs forced strategy.
pub fn resolve_welch_peaks_strategy(
    mode: WelchPeaksPickMode,
    device: Device,
    batch: usize,
    n_fft: usize,
    k: usize,
    learned_available: bool,
    learned_active_gates: Option<usize>,
    learned_total_gates: usize,
) -> WelchPeaksStrategy {
    match mode {
        WelchPeaksPickMode::Force(s) => s,
        WelchPeaksPickMode::Auto => pick_welch_peaks_strategy(
            device,
            batch,
            n_fft,
            k,
            learned_available,
            learned_active_gates,
            learned_total_gates,
        ),
    }
}

/// Stateful picker: auto or forced strategy; compiles GPU path once.
pub struct AutoWelchPeaks {
    pub strategy: WelchPeaksStrategy,
    pub device: Device,
    batch: usize,
    full_frame: usize,
    peak_params: WelchPeakParams,
    scratch: WelchPeaksScratch,
    fast_buf: Vec<f32>,
    rlx: Option<CompiledRlxWelchPeaksExec>,
    learned: Option<CompiledLearnedWelchPeaks>,
}

impl AutoWelchPeaks {
    pub fn new(batch: usize, n_fft: usize, k: usize, device: Option<&str>) -> Result<Self> {
        Self::with_options(batch, n_fft, k, device, None, WelchPeaksPickMode::Auto)
    }

    pub fn with_learned(
        batch: usize,
        n_fft: usize,
        k: usize,
        device: Option<&str>,
        model: Option<&FastLearnedFftModel>,
    ) -> Result<Self> {
        Self::with_options(batch, n_fft, k, device, model, WelchPeaksPickMode::Auto)
    }

    pub fn with_strategy(
        batch: usize,
        n_fft: usize,
        k: usize,
        device: Option<&str>,
        strategy: WelchPeaksStrategy,
    ) -> Result<Self> {
        Self::with_options(
            batch,
            n_fft,
            k,
            device,
            None,
            WelchPeaksPickMode::Force(strategy),
        )
    }

    pub fn with_options(
        batch: usize,
        n_fft: usize,
        k: usize,
        device: Option<&str>,
        model: Option<&FastLearnedFftModel>,
        mode: WelchPeaksPickMode,
    ) -> Result<Self> {
        ensure!(batch >= 1 && k >= 1);
        let device = resolve_train_device(device)?;
        ensure_backend_ready(device)?;

        let learned_available = model.is_some();
        let (active, total) = model
            .map(|m| (Some(m.active_gates(DEFAULT_GATE_THRESHOLD)), m.gates.len()))
            .unwrap_or((None, 0));

        let breakdown =
            pick_welch_peaks_breakdown(device, batch, n_fft, k, learned_available, active, total);
        let strategy = match mode {
            WelchPeaksPickMode::Force(s) => s,
            WelchPeaksPickMode::Auto => breakdown.picked,
        };
        if rlx_ir::env::flag("RLX_FFT_PICKER_TRACE") {
            fn ms(ns: f64) -> f64 {
                if ns.is_finite() {
                    ns / 1e6
                } else {
                    f64::INFINITY
                }
            }
            let gate_bd = crate::spectral::welch_peaks_cost::welch_peaks_fusion_gate_breakdown(
                device, batch, n_fft, k,
            );
            let fused_ok = crate::spectral::welch_peaks_cost::fused_welch_peaks_auto_viable(device);
            eprintln!(
                "[welch-peaks] io pick batch={batch} device={device:?} \
                 ultra={:.2}ms fast={:.2}ms rlx={:.2}ms learned={:.2}ms \
                 gate_score={:.2}ms gate_fuse={} fused_viable={fused_ok} -> {}",
                ms(breakdown.costs.ultra_ns),
                ms(breakdown.costs.fast_ns),
                ms(breakdown.costs.rlx_ns),
                ms(breakdown.costs.learned_ns),
                gate_bd.score_ns / 1e6,
                gate_bd.should_fuse,
                strategy.label(),
            );
        }

        if strategy == WelchPeaksStrategy::LearnedCompiled && model.is_none() {
            bail!("--strategy learned requires a trained model (--train-steps > 0)");
        }

        let peak_params = match strategy {
            WelchPeaksStrategy::UltraFast => WelchPeakParams::ultra_fast_for_n_fft(n_fft, k),
            _ => WelchPeakParams::fast_for_n_fft(n_fft, k),
        };
        let full_frame = WelchParams::for_n_fft(n_fft).frame_len();
        let scratch = WelchPeaksScratch::new(batch, peak_params.n_bins());
        let fast_cap = batch * peak_params.frame_len();
        let fast_buf = Vec::with_capacity(fast_cap);

        let mut rlx = None;
        let mut learned = None;
        match strategy {
            WelchPeaksStrategy::RlxCompiled => {
                rlx = Some(CompiledRlxWelchPeaksExec::compile_adaptive(
                    batch,
                    peak_params,
                    device,
                )?);
                if rlx_ir::env::flag("RLX_FFT_PICKER_TRACE") {
                    eprintln!(
                        "[welch-peaks] rlx exec kind: {:?}",
                        rlx.as_ref().map(|e| e.kind)
                    );
                }
            }
            WelchPeaksStrategy::LearnedCompiled => {
                let m = model.expect("learned model required for LearnedCompiled");
                let mut hard = m.clone();
                hard.hard_gate_threshold = Some(DEFAULT_GATE_THRESHOLD);
                learned = Some(compile_learned_welch_peaks(
                    &hard,
                    batch,
                    peak_params,
                    device,
                    default_welch_peaks_hard_threshold(),
                )?);
            }
            _ => {}
        }

        Ok(Self {
            strategy,
            device,
            batch,
            full_frame,
            peak_params,
            scratch,
            fast_buf,
            rlx,
            learned,
        })
    }

    pub fn strategy_label(&self) -> &'static str {
        self.strategy.label()
    }

    /// Bench / log label including RLX exec kind when applicable.
    pub fn picker_path_label(&self) -> String {
        match self.rlx_exec_kind() {
            Some(kind) => format!("{}_{}", self.strategy.label(), kind.label()),
            None => self.strategy_label().to_string(),
        }
    }

    pub fn peak_params(&self) -> WelchPeakParams {
        self.peak_params
    }

    /// RLX adaptive exec kind when strategy is [`WelchPeaksStrategy::RlxCompiled`].
    pub fn rlx_exec_kind(
        &self,
    ) -> Option<crate::spectral::welch_peaks_compile::RlxWelchPeaksExecKind> {
        self.rlx.as_ref().map(|e| e.kind)
    }

    /// Input `[batch, fast_frame]` when already truncated; skips layout copy.
    pub fn welch_peaks_batch_fast(&mut self, fast_signal: &[f32]) -> Result<Vec<f32>> {
        ensure!(fast_signal.len() == self.batch * self.peak_params.frame_len());
        self.welch_peaks_on_fast(fast_signal)
    }

    /// Input `[batch, full_welch_frame]` or `[batch, fast_frame]`; returns packed top-K peaks.
    pub fn welch_peaks_batch(&mut self, signal: &[f32]) -> Result<Vec<f32>> {
        let fast_len = self.batch * self.peak_params.frame_len();
        if signal.len() == fast_len {
            return self.welch_peaks_on_fast(signal);
        }
        ensure!(signal.len() == self.batch * self.full_frame);
        let mut fast_signal = std::mem::take(&mut self.fast_buf);
        self.peak_params.welch.truncate_batch_into(
            signal,
            self.batch,
            self.full_frame,
            &mut fast_signal,
        )?;
        let out = self.welch_peaks_on_fast(&fast_signal)?;
        self.fast_buf = fast_signal;
        Ok(out)
    }

    fn welch_peaks_on_fast(&mut self, fast_signal: &[f32]) -> Result<Vec<f32>> {
        match self.strategy {
            WelchPeaksStrategy::UltraFast | WelchPeaksStrategy::FastStreaming => {
                welch_peaks_rustfft_with_scratch(
                    fast_signal,
                    self.batch,
                    self.peak_params,
                    Some(&mut self.scratch),
                )
            }
            WelchPeaksStrategy::RlxCompiled => self
                .rlx
                .as_mut()
                .expect("rlx compiled")
                .welch_peaks_batch(fast_signal, &mut self.scratch),
            WelchPeaksStrategy::LearnedCompiled => self
                .learned
                .as_mut()
                .expect("learned compiled")
                .welch_peaks_batch(fast_signal, &mut self.scratch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::Device;

    #[test]
    fn small_batch_picks_ultra_on_cpu() {
        assert_eq!(
            pick_welch_peaks_strategy(Device::Cpu, 32, 256, 16, false, None, 0),
            WelchPeaksStrategy::UltraFast
        );
    }

    #[test]
    fn mid_batch_picks_ultra_on_cpu_io_model() {
        // 1-segment path moves fewer bytes than 2-segment fast path.
        assert_eq!(
            pick_welch_peaks_strategy(Device::Cpu, 512, 256, 16, false, None, 0),
            WelchPeaksStrategy::UltraFast
        );
    }

    #[test]
    fn large_batch_picks_rlx_on_metal() {
        assert_eq!(
            pick_welch_peaks_strategy(Device::Metal, 8192, 256, 16, false, None, 0),
            WelchPeaksStrategy::RlxCompiled
        );
    }

    #[test]
    fn metal_mid_batch_picks_rlx() {
        assert_eq!(
            pick_welch_peaks_strategy(Device::Metal, 1024, 256, 16, false, None, 0),
            WelchPeaksStrategy::RlxCompiled
        );
    }

    #[test]
    fn metal_small_batch_stays_rustfft() {
        assert_eq!(
            pick_welch_peaks_strategy(Device::Metal, 256, 256, 16, false, None, 0),
            WelchPeaksStrategy::FastStreaming
        );
    }

    #[test]
    fn metal_upper_mid_batch_picks_rlx() {
        assert_eq!(
            pick_welch_peaks_strategy(Device::Metal, 4096, 256, 16, false, None, 0),
            WelchPeaksStrategy::RlxCompiled
        );
    }

    #[test]
    fn wgpu_picks_rustfft_small_batch_rlx_large() {
        for batch in [256usize, 1024] {
            assert_eq!(
                pick_welch_peaks_strategy(Device::Gpu, batch, 256, 16, false, None, 0),
                WelchPeaksStrategy::FastStreaming,
                "batch={batch}"
            );
        }
        assert_eq!(
            pick_welch_peaks_strategy(Device::Gpu, 8192, 256, 16, false, None, 0),
            WelchPeaksStrategy::RlxCompiled,
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_large_batch_picks_rlx() {
        assert_eq!(
            pick_welch_peaks_strategy(Device::Cuda, 8192, 256, 16, false, None, 0),
            WelchPeaksStrategy::RlxCompiled,
        );
    }

    #[test]
    fn legacy_picker_matches_old_thresholds() {
        assert_eq!(
            legacy_pick_welch_peaks_strategy(Device::Metal, 8192, false, None, 0),
            WelchPeaksStrategy::RlxCompiled
        );
    }

    #[test]
    fn parse_strategy_aliases() {
        assert!(parse_welch_peaks_strategy("auto").unwrap().is_auto());
        assert_eq!(
            parse_welch_peaks_strategy("ultra-fast").unwrap(),
            WelchPeaksPickMode::Force(WelchPeaksStrategy::UltraFast)
        );
        assert_eq!(
            parse_welch_peaks_strategy("rlx").unwrap(),
            WelchPeaksPickMode::Force(WelchPeaksStrategy::RlxCompiled)
        );
    }

    #[test]
    fn welch_peaks_batch_accepts_fast_layout() {
        let batch = 8;
        let n_fft = 256;
        let k = 16;
        let full = WelchParams::for_n_fft(n_fft);
        let fast_params = WelchPeakParams::fast_for_n_fft(n_fft, k);
        let full_frame = full.frame_len();
        let fast_frame = fast_params.frame_len();
        let signal: Vec<f32> = (0..batch * full_frame).map(|i| i as f32 * 1e-6).collect();
        let fast_signal = fast_params
            .welch
            .truncate_batch(&signal, batch, full_frame)
            .unwrap();
        let mut picker = AutoWelchPeaks::with_strategy(
            batch,
            n_fft,
            k,
            Some("cpu"),
            WelchPeaksStrategy::FastStreaming,
        )
        .unwrap();
        let from_full = picker.welch_peaks_batch(&signal).unwrap();
        let from_fast = picker.welch_peaks_batch(&fast_signal).unwrap();
        let from_fast_api = picker.welch_peaks_batch_fast(&fast_signal).unwrap();
        assert_eq!(from_full, from_fast);
        assert_eq!(from_full, from_fast_api);
        assert_eq!(fast_frame, fast_params.frame_len());
    }

    #[test]
    fn forced_overrides_auto() {
        assert_eq!(
            resolve_welch_peaks_strategy(
                WelchPeaksPickMode::Force(WelchPeaksStrategy::FastStreaming),
                Device::Metal,
                8192,
                256,
                16,
                false,
                None,
                0,
            ),
            WelchPeaksStrategy::FastStreaming
        );
    }
}

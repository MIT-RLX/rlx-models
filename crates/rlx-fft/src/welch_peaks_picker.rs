//! Automatic Welch peaks path — pick fastest strategy from batch size + device.
//!
//! See `crates/rlx-fft/README.md` (Welch peaks section) for CLI flags, strategy
//! table, and examples. Use [`AutoWelchPeaks`] (auto or forced via [`WelchPeaksPickMode`]) or
//! [`parse_welch_peaks_strategy`] for string config.

use crate::device::{ensure_backend_ready, resolve_train_device};
use crate::learned_model::FastLearnedFftModel;
use crate::peak::{WelchPeakParams, WelchPeaksScratch, welch_peaks_rustfft_with_scratch};
use crate::pruned::DEFAULT_GATE_THRESHOLD;
use crate::welch::WelchParams;
use crate::welch_peaks_compile::{
    CompiledLearnedWelchPeaks, CompiledRlxWelchPeaks, compile_learned_welch_peaks,
    compile_rlx_welch_peaks, default_welch_peaks_hard_threshold,
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

fn is_gpu_device(device: Device) -> bool {
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

/// Batch threshold where RLX compiled peaks beat rustfft (Metal n=256 sweep).
pub fn rlx_crossover_batch(device: Device) -> usize {
    if is_gpu_device(device) {
        8192
    } else {
        usize::MAX
    }
}

/// Small batches use 1-segment ultra-fast path on CPU; slightly higher cap on GPU.
pub fn ultra_fast_max_batch(device: Device) -> usize {
    if is_gpu_device(device) { 128 } else { 256 }
}

/// Choose the fastest Welch-peaks strategy for `batch` on `device`.
pub fn pick_welch_peaks_strategy(
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
    learned_available: bool,
    learned_active_gates: Option<usize>,
    learned_total_gates: usize,
) -> WelchPeaksStrategy {
    match mode {
        WelchPeaksPickMode::Force(s) => s,
        WelchPeaksPickMode::Auto => pick_welch_peaks_strategy(
            device,
            batch,
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
    rlx: Option<CompiledRlxWelchPeaks>,
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

        let strategy =
            resolve_welch_peaks_strategy(mode, device, batch, learned_available, active, total);

        if strategy == WelchPeaksStrategy::LearnedCompiled && model.is_none() {
            bail!("--strategy learned requires a trained model (--train-steps > 0)");
        }

        let peak_params = match strategy {
            WelchPeaksStrategy::UltraFast => WelchPeakParams::ultra_fast_for_n_fft(n_fft, k),
            _ => WelchPeakParams::fast_for_n_fft(n_fft, k),
        };
        let full_frame = WelchParams::for_n_fft(n_fft).frame_len();
        let scratch = WelchPeaksScratch::new(batch, peak_params.n_bins());

        let mut rlx = None;
        let mut learned = None;
        match strategy {
            WelchPeaksStrategy::RlxCompiled => {
                rlx = Some(compile_rlx_welch_peaks(batch, peak_params, device)?);
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
            rlx,
            learned,
        })
    }

    pub fn strategy_label(&self) -> &'static str {
        self.strategy.label()
    }

    pub fn peak_params(&self) -> WelchPeakParams {
        self.peak_params
    }

    /// Input `[batch, full_welch_frame]` (8-segment layout); returns packed top-K peaks.
    pub fn welch_peaks_batch(&mut self, signal: &[f32]) -> Result<Vec<f32>> {
        ensure!(signal.len() == self.batch * self.full_frame);
        let fast_signal =
            self.peak_params
                .welch
                .truncate_batch(signal, self.batch, self.full_frame)?;

        match self.strategy {
            WelchPeaksStrategy::UltraFast | WelchPeaksStrategy::FastStreaming => {
                welch_peaks_rustfft_with_scratch(
                    &fast_signal,
                    self.batch,
                    self.peak_params,
                    Some(&mut self.scratch),
                )
            }
            WelchPeaksStrategy::RlxCompiled => self
                .rlx
                .as_mut()
                .expect("rlx compiled")
                .welch_peaks_batch(&fast_signal, &mut self.scratch),
            WelchPeaksStrategy::LearnedCompiled => self
                .learned
                .as_mut()
                .expect("learned compiled")
                .welch_peaks_batch(&fast_signal, &mut self.scratch),
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
            pick_welch_peaks_strategy(Device::Cpu, 32, false, None, 0),
            WelchPeaksStrategy::UltraFast
        );
    }

    #[test]
    fn mid_batch_picks_streaming_on_cpu() {
        assert_eq!(
            pick_welch_peaks_strategy(Device::Cpu, 512, false, None, 0),
            WelchPeaksStrategy::FastStreaming
        );
    }

    #[test]
    fn large_batch_picks_rlx_on_metal() {
        assert_eq!(
            pick_welch_peaks_strategy(Device::Metal, 8192, false, None, 0),
            WelchPeaksStrategy::RlxCompiled
        );
    }

    #[test]
    fn metal_mid_batch_stays_rustfft() {
        assert_eq!(
            pick_welch_peaks_strategy(Device::Metal, 1024, false, None, 0),
            WelchPeaksStrategy::FastStreaming
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
    fn forced_overrides_auto() {
        assert_eq!(
            resolve_welch_peaks_strategy(
                WelchPeaksPickMode::Force(WelchPeaksStrategy::FastStreaming),
                Device::Metal,
                8192,
                false,
                None,
                0,
            ),
            WelchPeaksStrategy::FastStreaming
        );
    }
}

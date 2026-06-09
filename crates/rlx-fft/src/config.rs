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

//! Configuration for learned FFT models.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

/// Supported transform sizes (power-of-two complex FFT length).
pub const SUPPORTED_N_FFT: &[usize] = &[
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
];

/// Full n_fft sweep for limit / capacity studies (same as [`SUPPORTED_N_FFT`]).
pub const FULL_N_FFT_SWEEP: &[usize] = SUPPORTED_N_FFT;

/// Batch sizes attempted in limit sweeps (descending — find max working batch per n_fft).
pub const LIMIT_SWEEP_REQUESTED_BATCHES: &[usize] =
    &[4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1];

/// Max batch to attempt at each n_fft during limit sweeps (push until failure).
pub fn batch_cap_for_limit_sweep(n_fft: usize) -> usize {
    match n_fft {
        n if n <= 128 => 4096,
        n if n <= 256 => 2048,
        n if n <= 512 => 1024,
        n if n <= 1024 => 512,
        n if n <= 2048 => 256,
        n if n <= 4096 => 128,
        n if n <= 8192 => 64,
        n if n <= 16384 => 32,
        n if n <= 32768 => 16,
        n if n <= 65536 => 8,
        _ => 4,
    }
}

/// Cap batch size by n_fft to stay within practical memory / compile limits.
pub fn adaptive_batches_for_n_fft(n_fft: usize, requested: &[usize]) -> Vec<usize> {
    adaptive_batches_with_cap(n_fft, requested, batch_cap_for_limit_sweep(n_fft))
}

pub fn adaptive_batches_with_cap(n_fft: usize, requested: &[usize], cap: usize) -> Vec<usize> {
    let _ = n_fft;
    let mut out: Vec<usize> = requested
        .iter()
        .copied()
        .filter(|&b| b >= 1 && b <= cap)
        .collect();
    if out.is_empty() {
        out.push(cap.max(1));
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Batches used by `--limit-sweep`.
pub fn limit_sweep_batches(n_fft: usize) -> Vec<usize> {
    adaptive_batches_for_n_fft(n_fft, LIMIT_SWEEP_REQUESTED_BATCHES)
}

/// Compiled-graph variants are only attempted up to this n_fft (larger sizes hang on compile).
pub fn compiled_ok_for_n_fft(n_fft: usize) -> bool {
    n_fft <= 1024
}

/// Per-device compiled ceiling for limit sweeps (GPU backends tolerate slightly larger graphs).
pub fn compiled_ok_for_limit_sweep(n_fft: usize, device: &str) -> bool {
    if n_fft > 4096 {
        return false;
    }
    match device.to_ascii_lowercase().as_str() {
        "cpu" => n_fft <= 1024,
        "metal" | "cuda" | "mlx" | "mps" | "rocm" | "wgpu" | "wgu" | "vulkan" | "gpu" => {
            n_fft <= 2048
        }
        _ => n_fft <= 1024,
    }
}

pub fn is_gpu_device_label(device: &str) -> bool {
    matches!(
        device.to_ascii_lowercase().as_str(),
        "metal" | "cuda" | "mlx" | "mps" | "rocm" | "wgpu" | "wgu" | "vulkan" | "gpu"
    )
}

/// Welch PSD path — skip at extreme sizes (segment buffers grow with n_fft).
pub fn welch_ok_for_limit_sweep(n_fft: usize) -> bool {
    n_fft <= 32768
}

/// Welch signal buffer is `batch × frame_len` floats — skip huge combos.
pub fn welch_ok_for_config(n_fft: usize, batch: usize) -> bool {
    if !welch_ok_for_limit_sweep(n_fft) {
        return false;
    }
    let hop = n_fft / 2;
    let frame = n_fft + 7 * hop;
    let bytes = batch.saturating_mul(frame).saturating_mul(4);
    bytes <= 512 * 1024 * 1024
}

/// Reduce training steps at large FFT sizes during sweeps.
pub fn train_steps_for_n_fft(base: usize, n_fft: usize) -> usize {
    match n_fft {
        n if n > 65536 => base.min(2),
        n if n > 32768 => base.min(3),
        n if n > 16384 => base.min(4),
        n if n > 8192 => base.min(5),
        n if n > 4096 => base.min(8),
        n if n > 2048 => base.min(12),
        n if n > 1024 => base.min(15),
        _ => base,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformDir {
    Forward,
    Inverse,
}

impl TransformDir {
    pub fn is_forward(self) -> bool {
        matches!(self, Self::Forward)
    }

    pub fn is_inverse(self) -> bool {
        matches!(self, Self::Inverse)
    }
}

impl std::str::FromStr for TransformDir {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "forward" | "fft" => Ok(Self::Forward),
            "inverse" | "ifft" => Ok(Self::Inverse),
            other => bail!("unknown transform direction: {other} (use fft|ifft)"),
        }
    }
}

pub fn parse_transform_dir(s: &str) -> Result<TransformDir> {
    s.parse()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FftLearnConfig {
    /// Complex FFT length (must be a power of two).
    pub n_fft: usize,
    /// Training / inference batch size.
    pub batch: usize,
}

impl FftLearnConfig {
    pub fn new(n_fft: usize, batch: usize) -> Result<Self> {
        ensure!(
            n_fft.is_power_of_two(),
            "n_fft must be a power of two, got {n_fft}"
        );
        ensure!(n_fft >= 4, "n_fft must be at least 4");
        ensure!(batch >= 1, "batch must be >= 1");
        Ok(Self { n_fft, batch })
    }

    pub fn tiny() -> Self {
        Self {
            n_fft: 64,
            batch: 4,
        }
    }

    pub fn num_stages(&self) -> usize {
        self.n_fft.trailing_zeros() as usize
    }

    pub fn butterflies_per_stage(&self) -> usize {
        self.n_fft / 2
    }

    pub fn twiddle_param_count(&self) -> usize {
        self.num_stages() * self.butterflies_per_stage() * 2
    }

    pub fn validate(&self) -> Result<()> {
        Self::new(self.n_fft, self.batch)?;
        Ok(())
    }
}

pub fn parse_n_fft(s: &str) -> Result<usize> {
    let n: usize = s.parse().context("n_fft: usize")?;
    FftLearnConfig::new(n, 1).map(|_| n)
}

pub fn ensure_supported_n_fft(n_fft: usize) -> Result<()> {
    if SUPPORTED_N_FFT.contains(&n_fft) {
        return Ok(());
    }
    bail!(
        "unsupported n_fft={n_fft}; supported: {}",
        SUPPORTED_N_FFT
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub model: FftLearnConfig,
    pub direction: TransformDir,
    pub steps: usize,
    pub lr: f64,
    pub weight_decay: f32,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub grad_clip: f32,
    pub seed: u64,
    pub log_every: usize,
    pub device: String,
    pub out_dir: Option<std::path::PathBuf>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            model: FftLearnConfig::tiny(),
            direction: TransformDir::Forward,
            steps: 500,
            lr: 1e-3,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_clip: 1.0,
            seed: 42,
            log_every: 50,
            device: "auto".to_string(),
            out_dir: None,
        }
    }
}

/// Three-phase training: encoder only → decoder only → joint fine-tune.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasedTrainConfig {
    pub model: FftLearnConfig,
    pub encoder_steps: usize,
    pub decoder_steps: usize,
    pub joint_steps: usize,
    pub lr: f64,
    pub spectrum_weight: f32,
    pub seed: u64,
    pub log_every: usize,
    pub out_dir: Option<std::path::PathBuf>,
}

impl Default for PhasedTrainConfig {
    fn default() -> Self {
        Self {
            model: FftLearnConfig::tiny(),
            encoder_steps: 300,
            decoder_steps: 300,
            joint_steps: 300,
            lr: 5e-4,
            spectrum_weight: 1.0,
            seed: 42,
            log_every: 50,
            out_dir: None,
        }
    }
}

/// Train encoder (FFT) + decoder (IFFT) jointly on synthetic roundtrip data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncDecTrainConfig {
    pub model: FftLearnConfig,
    pub steps: usize,
    pub lr: f64,
    /// Weight for auxiliary encoder loss vs reference FFT (0 = roundtrip only).
    pub spectrum_weight: f32,
    pub seed: u64,
    pub log_every: usize,
    pub device: String,
    pub out_dir: Option<std::path::PathBuf>,
    #[serde(default = "default_grad_clip")]
    pub grad_clip: f32,
    #[serde(default = "default_project_twiddles")]
    pub project_twiddles: bool,
}

fn default_grad_clip() -> f32 {
    1.0
}

fn default_project_twiddles() -> bool {
    true
}

/// Multi-size encoder–decoder training study (compare schedules across n_fft).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultiTrainSchedule {
    /// Train each n_fft in isolation (`steps` per size).
    Single,
    /// Cycle sizes each step (`steps` total).
    RoundRobin,
    /// Random n_fft each step (`steps` total).
    Random,
    /// Equal step count per size (`steps / n_sizes` each, `steps` total).
    Balanced,
}

impl MultiTrainSchedule {
    pub fn label(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::RoundRobin => "round_robin",
            Self::Random => "random",
            Self::Balanced => "balanced",
        }
    }

    pub fn all() -> &'static [Self] {
        &[Self::Single, Self::RoundRobin, Self::Random, Self::Balanced]
    }

    pub fn parse_csv(s: &str) -> anyhow::Result<Vec<Self>> {
        let mut out = Vec::new();
        for part in s.split(',') {
            let part = part.trim().to_ascii_lowercase();
            if part.is_empty() {
                continue;
            }
            out.push(match part.as_str() {
                "single" => Self::Single,
                "round_robin" | "round-robin" | "rr" => Self::RoundRobin,
                "random" => Self::Random,
                "balanced" => Self::Balanced,
                other => anyhow::bail!(
                    "unknown schedule {other} (use single,round_robin,random,balanced)"
                ),
            });
        }
        anyhow::ensure!(!out.is_empty(), "schedules list is empty");
        Ok(out)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTrainConfig {
    pub n_ffts: Vec<usize>,
    pub batch: usize,
    /// Maximum training steps (per size for `single`, total for mixed schedules).
    pub steps: usize,
    pub schedules: Vec<MultiTrainSchedule>,
    pub lr: f64,
    pub spectrum_weight: f32,
    pub seed: u64,
    pub log_every: usize,
    pub eval_batches: usize,
    pub out_dir: Option<std::path::PathBuf>,
    /// Stop early when holdout loss plateaus (after `min_steps`).
    pub until_converged: bool,
    /// Minimum steps before convergence checks apply.
    pub min_steps: usize,
    /// Re-check holdout loss every N steps.
    pub converge_every: usize,
    /// Consecutive checks without meaningful improvement before stop.
    pub converge_patience: usize,
    /// Relative improvement threshold (fraction of best loss).
    pub converge_delta: f32,
    /// Clip twiddle gradient L2 norm (0 = off).
    pub grad_clip: f32,
    /// Keep twiddles on the unit circle after each step.
    pub project_twiddles: bool,
    /// Use fused enc–dec step (batched ref FFT + shared backward).
    pub use_fused_train: bool,
    pub optimizer: crate::second_order::TwiddleOptimizer,
}

impl Default for MultiTrainConfig {
    fn default() -> Self {
        Self {
            n_ffts: vec![64, 256],
            batch: 8,
            steps: 10_000,
            schedules: MultiTrainSchedule::all().to_vec(),
            lr: 5e-4,
            spectrum_weight: 1.0,
            seed: 42,
            log_every: 50,
            eval_batches: 8,
            out_dir: None,
            until_converged: true,
            min_steps: 300,
            converge_every: 25,
            converge_patience: 5,
            converge_delta: 1e-4,
            grad_clip: 1.0,
            project_twiddles: true,
            use_fused_train: true,
            optimizer: crate::second_order::TwiddleOptimizer::Sgd,
        }
    }
}

impl Default for EncDecTrainConfig {
    fn default() -> Self {
        Self {
            model: FftLearnConfig::tiny(),
            steps: 500,
            lr: 1e-3,
            spectrum_weight: 1.0,
            seed: 42,
            log_every: 50,
            device: "auto".to_string(),
            out_dir: None,
            grad_clip: default_grad_clip(),
            project_twiddles: default_project_twiddles(),
        }
    }
}

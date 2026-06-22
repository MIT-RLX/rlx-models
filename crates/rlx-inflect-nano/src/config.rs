//! Config mirrors of `MicroFastSpeechConfig` and `HifiGanConfig`, plus the
//! bundle `config.json` emitted by `scripts/export_inflect_nano.py`.

use std::collections::BTreeMap;

use serde::Deserialize;

/// `MicroFastSpeechConfig` from `train_inflect_micro_fastspeech_v3_pitch.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct AcousticConfig {
    pub vocab_size: usize,
    pub tone_size: usize,
    pub lang_size: usize,
    pub n_mels: usize,
    pub hidden: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub decoder_ff_mult: usize,
    pub kernel_size: usize,
    pub speaker_count: usize,
    pub speaker_dim: usize,
    #[serde(default)]
    pub dropout: f32,
    pub sample_rate: usize,
    pub max_frames: usize,
    pub postnet_scale: f32,
    pub use_frame_pitch: bool,
    pub abs_frame_bins: usize,
}

/// `HifiGanConfig` (snake_v2mid variant) from `train_hifigan_oracle_v1.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct VocoderConfig {
    pub variant: String,
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_size: usize,
    pub win_size: usize,
    pub num_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub resblock: String,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub activation: String,
    #[serde(default)]
    pub conditioning_channels: usize,
}

/// Top-level `config.json` of the RLX asset bundle.
#[derive(Debug, Clone, Deserialize)]
pub struct BundleConfig {
    #[serde(default)]
    pub model: String,
    pub sample_rate: u32,
    pub n_mels: usize,
    pub add_blank: bool,
    pub language: String,
    pub speakers: BTreeMap<String, i64>,
    pub acoustic: AcousticConfig,
    pub vocoder: VocoderConfig,
}

impl BundleConfig {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Speaker id for "mark" (the released voice), falling back to the first entry.
    pub fn default_speaker(&self) -> i64 {
        self.speakers
            .get("mark")
            .copied()
            .or_else(|| self.speakers.values().next().copied())
            .unwrap_or(0)
    }
}

/// Execution strategy: which compute path the synthesizer uses.
///
/// The acoustic stage is tiny and always runs host-eager; these modes select
/// how the vocoder (the compute core) runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    /// Lowest wall-clock latency: vocoder runs on the fastest available
    /// accelerator (Metal → MLX → wgpu), falling back to the CPU path if none.
    Latency,
    /// Maximum numeric fidelity: pure host-eager f32 — the parity reference,
    /// fully deterministic, no backend-specific kernel approximations.
    #[default]
    Precision,
    /// Smallest memory footprint: host-eager only — no graph compilation, no
    /// on-disk AOT cache, and no compiled-graph residency.
    MemoryFootprint,
    /// iOS-style CPU+GPU split: acoustic on the CPU, vocoder graph on the GPU.
    /// Falls back to the CPU path when no GPU backend is available.
    Hybrid,
}

impl ExecutionMode {
    /// Parse a `--mode` string; unknown values fall back to `Precision`.
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "latency" | "fast" => Self::Latency,
            "memory" | "memory-footprint" | "footprint" => Self::MemoryFootprint,
            "hybrid" | "ios" => Self::Hybrid,
            _ => Self::Precision,
        }
    }
}

/// Inference controls (mirrors `MicroFastSpeech.infer` arguments + `synthesize`).
#[derive(Debug, Clone)]
pub struct InferOpts {
    pub length_scale: f32,
    pub pitch_scale: f32,
    pub energy_scale: f32,
    pub min_duration: i64,
    pub max_duration: i64,
}

impl Default for InferOpts {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            pitch_scale: 1.0,
            energy_scale: 1.0,
            min_duration: 1,
            max_duration: 80,
        }
    }
}

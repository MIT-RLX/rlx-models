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

//! Shared wake-word streaming API and frontend primitives.

pub mod audio;
pub mod bench;
pub mod device;
pub mod mel;
pub mod metrics;
pub mod ops;
pub mod parity;
pub mod cnn;
pub mod ternary;
pub mod train;
pub mod weights_io;

use anyhow::Result;

pub use audio::{SAMPLE_RATE_16K, load_wav_mono_f32, parse_wav_mono_f32, resample_linear};
pub use bench::{BenchStats, bench_engine, print_bench_table};
pub use device::{
    available_device_labels, available_devices, bench_device_label, bind_streaming_device,
    ensure_backend_ready, parse_device_list, resolve_device, streaming_execution_device,
};
pub use mel::{MelConfig, MelFrontend, OWW_CHUNK_SAMPLES};
pub use metrics::{
    DetectionStats, FloatPrecision, best_f1_threshold, detection_stats, float_precision, peak_of,
    print_detection_stats,
};
pub use parity::{
    BackendParityRow, assert_100_percent_parity, max_abs_score_delta, run_backend_parity,
    score_parity_fraction, scores_exact_match,
};
pub use cnn::{WakeCnn, WakeCnnConfig, WakeCnnWeights};
pub use ternary::{TernaryOpts, TernaryStats, is_ternary_f32, ternarize, ternarize_inplace};
pub use train::{
    LabeledClip, MlpConfig, MlpWeights, SgdConfig, CnnTrainConfig, TrainReport,
    load_pos_neg_dirs, synth_pos_neg_dataset, train_mlp, train_wake_cnn, write_synth_corpus,
};

/// Default openWakeWord-style hop (80 ms @ 16 kHz).
pub const DEFAULT_CHUNK_SAMPLES: usize = OWW_CHUNK_SAMPLES;

/// One streaming step after consuming PCM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WakeStep {
    /// Model activation in `[0, 1]`.
    pub score: f32,
    /// True when `score >= threshold` (and optional cooldown allows).
    pub fired: bool,
    /// End time of the consumed audio relative to session start (ms).
    pub t_ms: f32,
}

/// Shared detector configuration.
#[derive(Debug, Clone)]
pub struct WakeConfig {
    pub threshold: f32,
    pub chunk_samples: usize,
    /// Ignore new fires within this many ms after a fire.
    pub cooldown_ms: f32,
    pub keyword: String,
}

impl Default for WakeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            chunk_samples: DEFAULT_CHUNK_SAMPLES,
            cooldown_ms: 750.0,
            keyword: "wake".into(),
        }
    }
}

/// Streaming wake-word engine.
pub trait WakeEngine {
    fn push_pcm(&mut self, samples: &[f32]) -> Result<Vec<WakeStep>>;
    fn reset(&mut self);
    fn config(&self) -> &WakeConfig;
    fn keyword(&self) -> &str {
        &self.config().keyword
    }
}

/// Score an entire mono buffer by feeding fixed-size chunks (zero-pad the tail).
pub fn score_wav<E: WakeEngine>(engine: &mut E, pcm: &[f32]) -> Result<Vec<WakeStep>> {
    let hop = engine.config().chunk_samples.max(1);
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < pcm.len() {
        let end = (i + hop).min(pcm.len());
        let mut chunk = pcm[i..end].to_vec();
        if chunk.len() < hop {
            chunk.resize(hop, 0.0);
        }
        out.extend(engine.push_pcm(&chunk)?);
        i += hop;
    }
    Ok(out)
}

/// Peak score across steps (0 if empty).
pub fn peak_score(steps: &[WakeStep]) -> f32 {
    steps.iter().map(|s| s.score).fold(0.0_f32, f32::max)
}

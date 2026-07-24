// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Session hop / context / VAD / phrase threshold configuration.

use rlx_wakeword_core::SAMPLE_RATE_16K;

/// Default hop: 40 ms @ 16 kHz.
pub const DEFAULT_HOP_SAMPLES: usize = 640;
pub const DEFAULT_CONTEXT_MS: f32 = 1200.0;
pub const DEFAULT_COOLDOWN_MS: f32 = 750.0;
pub const DEFAULT_VAD_THRESHOLD: f32 = 0.35;

#[derive(Debug, Clone)]
pub struct PhraseConfig {
    pub id: String,
    pub threshold: f32,
}

impl PhraseConfig {
    pub fn new(id: impl Into<String>, threshold: f32) -> Self {
        Self {
            id: id.into(),
            threshold,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WakewordConfig {
    pub hop_samples: usize,
    pub context_ms: f32,
    pub vad_gate: bool,
    pub vad_threshold: f32,
    pub cooldown_ms: f32,
    pub phrases: Vec<PhraseConfig>,
    /// When `speaker-id` feature is on and a gate is attached, filter candidates.
    pub speaker_id: bool,
}

impl Default for WakewordConfig {
    fn default() -> Self {
        Self {
            hop_samples: DEFAULT_HOP_SAMPLES,
            context_ms: DEFAULT_CONTEXT_MS,
            vad_gate: cfg!(feature = "earshot"),
            vad_threshold: DEFAULT_VAD_THRESHOLD,
            cooldown_ms: DEFAULT_COOLDOWN_MS,
            phrases: Vec::new(),
            speaker_id: cfg!(feature = "speaker-id"),
        }
    }
}

impl WakewordConfig {
    pub fn with_hop_ms(mut self, hop_ms: u32) -> Self {
        self.hop_samples = hop_ms_to_samples(hop_ms);
        self
    }

    pub fn context_frames(&self, hop_length: usize) -> usize {
        let samples = (self.context_ms / 1000.0 * SAMPLE_RATE_16K as f32) as usize;
        (samples / hop_length.max(1)).max(8)
    }
}

pub fn hop_ms_to_samples(hop_ms: u32) -> usize {
    let sr = SAMPLE_RATE_16K as u32;
    ((hop_ms as u64 * sr as u64) / 1000) as usize
}

pub fn samples_to_hop_ms(samples: usize) -> u32 {
    ((samples as u64 * 1000) / SAMPLE_RATE_16K as u64) as u32
}

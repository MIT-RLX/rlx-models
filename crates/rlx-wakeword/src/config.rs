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
    /// This phrase's window, when it differs from [`WakewordConfig::context_ms`] (a longer
    /// phrase, e.g. a question, needs more than a short call).
    pub context_ms: Option<f32>,
    /// This phrase's confirmation hops, when they differ from [`WakewordConfig::min_hops`] (a
    /// longer phrase holds its score over more hops, and can ask for more).
    pub min_hops: Option<usize>,
    /// This phrase's score smoothing, when it differs from [`WakewordConfig::smooth_hops`].
    pub smooth_hops: Option<usize>,
    /// With a verifier ([`crate::session::WakewordSession::with_verifier`]): the verifier's
    /// score this phrase needs (else the verifier's default), and which of its outputs scores
    /// the phrase (else the detector's output index).
    pub verify_threshold: Option<f32>,
    pub verify_output: Option<usize>,
}

impl PhraseConfig {
    pub fn new(id: impl Into<String>, threshold: f32) -> Self {
        Self {
            id: id.into(),
            threshold,
            context_ms: None,
            min_hops: None,
            smooth_hops: None,
            verify_threshold: None,
            verify_output: None,
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
    /// A phrase fires only after this many hops in a row at or above its threshold: a real
    /// phrase holds its score while the window slides over it, a noise burst rarely does.
    pub min_hops: usize,
    /// Each phrase's score is the mean of its last this many hops before the threshold and
    /// `min_hops` see it: a one-hop dip in a held phrase does not break the run. 1: raw.
    pub smooth_hops: usize,
    /// Score the networks on every this-many hops (the others' frames are queued and folded
    /// into the next score, exactly): 2 halves the work, a detection then comes on an 80 ms
    /// grid. Thresholds, `min_hops` and `smooth_hops` count scored hops.
    pub score_every_hops: usize,
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
            min_hops: 1,
            smooth_hops: 1,
            score_every_hops: 1,
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
        frames_for(self.context_ms, hop_length)
    }

    /// The window of phrase `id` (its own, else the session's), ms.
    pub fn phrase_context_ms(&self, id: &str) -> f32 {
        self.phrases
            .iter()
            .find(|p| p.id == id)
            .and_then(|p| p.context_ms)
            .unwrap_or(self.context_ms)
    }

    /// The longest window any phrase uses, ms: how much audio a caller must keep to fill them.
    pub fn max_context_ms(&self) -> f32 {
        self.phrases
            .iter()
            .filter_map(|p| p.context_ms)
            .fold(self.context_ms, f32::max)
    }
}

/// Mel frames in `ms` of audio.
pub fn frames_for(ms: f32, hop_length: usize) -> usize {
    let samples = (ms / 1000.0 * SAMPLE_RATE_16K as f32) as usize;
    (samples / hop_length.max(1)).max(8)
}

impl WakewordConfig {}

pub fn hop_ms_to_samples(hop_ms: u32) -> usize {
    let sr = SAMPLE_RATE_16K as u32;
    ((hop_ms as u64 * sr as u64) / 1000) as usize
}

pub fn samples_to_hop_ms(samples: usize) -> u32 {
    ((samples as u64 * 1000) / SAMPLE_RATE_16K as u64) as u32
}

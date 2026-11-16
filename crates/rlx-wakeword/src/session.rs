// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Streaming wakeword session: 16 kHz mono PCM → [`WakeEvent`] at hop cadence.
//!
//! Optional Earshot VAD (`earshot`) and speaker-id gate (`speaker-id`).

use anyhow::{Result, bail};
use rlx_wakeword_core::{MelConfig, MelFrontend, SAMPLE_RATE_16K, WakeCnn, WakeCnnWeights};

#[cfg(feature = "speaker-id")]
use crate::cascade::SpeakerGate;
use crate::config::WakewordConfig;

#[derive(Debug, Clone)]
pub enum WakeEvent {
    Idle {
        t_ms: f32,
    },
    Candidate {
        phrase_id: String,
        score: f32,
        t_ms: f32,
        latency_ms: f32,
        /// Populated when `speaker-id` feature + enrolled gate is active.
        speaker_id: Option<String>,
        speaker_score: Option<f32>,
    },
}

/// Scores kept per head for [`WakewordSession::peak_score`]: 12 hops (480 ms at 40 ms).
const SCORE_HISTORY: usize = 12;

struct PhraseHead {
    id: String,
    threshold: f32,
    /// The network ([`WakewordSession::nets`]) and which of its outputs scores this phrase.
    net: usize,
    output: usize,
    /// Its own confirmation hops, if set (else the session's).
    min_hops: Option<usize>,
    last_fire_ms: f32,
    /// Hops in a row at or above the threshold.
    above: usize,
    /// Its score is the mean of its last this many (see [`WakewordConfig::smooth_hops`]).
    smooth: usize,
    /// The verifier's score it needs, and the verifier output scoring it (with a verifier).
    verify_threshold: Option<f32>,
    verify_output: usize,
    /// The verifier said no: it does not ask again until the score drops below the threshold.
    vetoed: bool,
    /// The last [`SCORE_HISTORY`] scores (a ring; 0 when not scored).
    recent: [f32; SCORE_HISTORY],
    next: usize,
}

impl PhraseHead {
    fn record(&mut self, score: f32) {
        self.recent[self.next] = score;
        self.next = (self.next + 1) % SCORE_HISTORY;
    }

    /// The mean of the last `smooth` recorded scores.
    fn smoothed(&self) -> f32 {
        let n = self.smooth.clamp(1, SCORE_HISTORY);
        (1..=n)
            .map(|i| self.recent[(self.next + SCORE_HISTORY - i) % SCORE_HISTORY])
            .sum::<f32>()
            / n as f32
    }
}

/// A second network asked only when a phrase fires (a larger one: run on the robot a few times
/// a minute, not every hop). It is fed every frame without scoring.
struct Verifier {
    net: WakeCnn,
    threshold: f32,
}

pub struct WakewordSession {
    cfg: WakewordConfig,
    mel: MelFrontend,
    /// The networks: one per weights file. A multi-phrase network scores all its phrases in
    /// one (streaming) forward per hop.
    nets: Vec<WakeCnn>,
    heads: Vec<PhraseHead>,
    verifier: Option<Verifier>,
    pcm_buf: Vec<f32>,
    /// Rolling PCM for optional speaker-id scoring (≈ context_ms).
    pcm_ring: Vec<f32>,
    samples_seen: usize,
    device_label: String,
    #[cfg(feature = "earshot")]
    vad: Option<rlx_vad::EarshotDetector>,
    #[cfg(feature = "earshot")]
    vad_pcm: Vec<f32>,
    #[cfg(feature = "speaker-id")]
    speaker: Option<crate::cascade::speaker::SpeakerIdGate>,
}

impl WakewordSession {
    pub fn new(cfg: WakewordConfig, weights: Vec<(String, WakeCnnWeights)>) -> Result<Self> {
        if cfg.phrases.is_empty() && weights.is_empty() {
            bail!("need at least one phrase");
        }
        let shared: Vec<(WakeCnnWeights, Vec<(String, usize)>)> = weights
            .into_iter()
            .map(|(id, w)| (w, vec![(id, 0)]))
            .collect();
        Self::new_shared(cfg, shared)
    }

    /// Networks each scoring one or more phrases: `(weights, [(phrase id, output index)])`.
    /// A network's window is its first phrase's ([`WakewordConfig::phrase_context_ms`]).
    pub fn new_shared(
        cfg: WakewordConfig,
        nets_in: Vec<(WakeCnnWeights, Vec<(String, usize)>)>,
    ) -> Result<Self> {
        if cfg.phrases.is_empty() && nets_in.is_empty() {
            bail!("need at least one phrase");
        }
        let hop_len = MelConfig::default().hop_length;
        let mut nets = Vec::new();
        let mut heads = Vec::new();
        for (w, phrases) in nets_in {
            let Some((first, _)) = phrases.first() else {
                continue;
            };
            let window_frames = crate::config::frames_for(cfg.phrase_context_ms(first), hop_len);
            let n_out = w.n_out();
            for (id, output) in phrases {
                if output >= n_out {
                    bail!("phrase {id}: output {output}, the network has {n_out}");
                }
                let pc = cfg.phrases.iter().find(|p| p.id == id);
                let thr = pc.map(|p| p.threshold).unwrap_or(0.5);
                let min_hops = pc.and_then(|p| p.min_hops);
                let smooth = pc
                    .and_then(|p| p.smooth_hops)
                    .unwrap_or(cfg.smooth_hops)
                    .max(1);
                let verify_threshold = pc.and_then(|p| p.verify_threshold);
                let verify_output = pc.and_then(|p| p.verify_output).unwrap_or(output);
                heads.push(PhraseHead {
                    id,
                    threshold: thr,
                    net: nets.len(),
                    output,
                    min_hops,
                    last_fire_ms: -1.0e9,
                    above: 0,
                    smooth,
                    verify_threshold,
                    verify_output,
                    vetoed: false,
                    recent: [0.0; SCORE_HISTORY],
                    next: 0,
                });
            }
            nets.push(WakeCnn::new(w).with_window_frames(window_frames));
        }
        if heads.is_empty() {
            bail!("no phrase weights loaded");
        }
        #[cfg(feature = "earshot")]
        let vad = if cfg.vad_gate {
            Some(rlx_vad::EarshotDetector::new())
        } else {
            None
        };
        Ok(Self {
            mel: MelFrontend::new(MelConfig::default()),
            cfg,
            nets,
            heads,
            verifier: None,
            pcm_buf: Vec::new(),
            pcm_ring: Vec::new(),
            samples_seen: 0,
            device_label: "cpu".into(),
            #[cfg(feature = "earshot")]
            vad,
            #[cfg(feature = "earshot")]
            vad_pcm: Vec::new(),
            #[cfg(feature = "speaker-id")]
            speaker: None,
        })
    }

    /// Ask `weights` (scoring a `context_ms` window) before any phrase fires: the phrase's
    /// verifier output must reach its `verify_threshold`, else `threshold`.
    pub fn with_verifier(
        mut self,
        weights: WakeCnnWeights,
        threshold: f32,
        context_ms: f32,
    ) -> Self {
        let frames = crate::config::frames_for(context_ms, MelConfig::default().hop_length);
        self.verifier = Some(Verifier {
            net: WakeCnn::new(weights).with_window_frames(frames),
            threshold,
        });
        self
    }

    pub fn with_device_label(mut self, label: impl Into<String>) -> Self {
        self.device_label = label.into();
        self
    }

    #[cfg(feature = "speaker-id")]
    pub fn with_speaker_gate(mut self, gate: crate::cascade::speaker::SpeakerIdGate) -> Self {
        self.cfg.speaker_id = true;
        self.speaker = Some(gate);
        self
    }

    #[cfg(feature = "speaker-id")]
    pub fn speaker_gate_mut(&mut self) -> Option<&mut crate::cascade::speaker::SpeakerIdGate> {
        self.speaker.as_mut()
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }

    pub fn config(&self) -> &WakewordConfig {
        &self.cfg
    }

    pub fn phrase_count(&self) -> usize {
        self.heads.len()
    }

    pub fn set_phrase_threshold(&mut self, id: &str, thr: f32) {
        if let Some(h) = self.heads.iter_mut().find(|h| h.id == id) {
            h.threshold = thr;
        }
        if let Some(p) = self.cfg.phrases.iter_mut().find(|p| p.id == id) {
            p.threshold = thr;
        }
    }

    pub fn upsert_phrase(
        &mut self,
        id: impl Into<String>,
        weights: WakeCnnWeights,
        threshold: f32,
    ) {
        let id = id.into();
        let hop_len = MelConfig::default().hop_length;
        let window_frames = crate::config::frames_for(self.cfg.phrase_context_ms(&id), hop_len);
        let net = WakeCnn::new(weights).with_window_frames(window_frames);
        if let Some(h) = self.heads.iter_mut().find(|h| h.id == id) {
            // Its own network from now on (a shared one keeps serving the other phrases).
            self.nets.push(net);
            h.net = self.nets.len() - 1;
            h.output = 0;
            h.threshold = threshold;
            h.last_fire_ms = -1.0e9;
            h.above = 0;
            h.recent = [0.0; SCORE_HISTORY];
        } else {
            self.nets.push(net);
            self.heads.push(PhraseHead {
                id: id.clone(),
                threshold,
                net: self.nets.len() - 1,
                output: 0,
                min_hops: None,
                last_fire_ms: -1.0e9,
                above: 0,
                smooth: self.cfg.smooth_hops.max(1),
                verify_threshold: None,
                verify_output: 0,
                vetoed: false,
                recent: [0.0; SCORE_HISTORY],
                next: 0,
            });
        }
        if let Some(p) = self.cfg.phrases.iter_mut().find(|p| p.id == id) {
            p.threshold = threshold;
        } else {
            self.cfg
                .phrases
                .push(crate::config::PhraseConfig::new(id, threshold));
        }
    }

    pub fn reset(&mut self) {
        self.mel.reset();
        self.pcm_buf.clear();
        self.pcm_ring.clear();
        self.samples_seen = 0;
        for n in &mut self.nets {
            n.reset();
        }
        if let Some(v) = &mut self.verifier {
            v.net.reset();
        }
        for h in &mut self.heads {
            h.last_fire_ms = -1.0e9;
            h.above = 0;
            h.vetoed = false;
            h.recent = [0.0; SCORE_HISTORY];
        }
        #[cfg(feature = "earshot")]
        {
            if let Some(v) = self.vad.as_mut() {
                v.reset();
            }
            self.vad_pcm.clear();
        }
    }

    /// The highest score `phrase_id` had over the last [`SCORE_HISTORY`] scored hops (`None`
    /// for an unknown phrase). With several phrases, a detection whose rival scored higher
    /// just before or after it is most likely the rival ("Hey <name>" vs "Bye <name>").
    pub fn peak_score(&self, phrase_id: &str) -> Option<f32> {
        self.heads
            .iter()
            .find(|h| h.id == phrase_id)
            .map(|h| h.recent.iter().copied().fold(0.0, f32::max))
    }

    /// See [`WakewordConfig::min_hops`].
    pub fn set_min_hops(&mut self, hops: usize) {
        self.cfg.min_hops = hops.max(1);
    }

    /// The phrase ids, in bundle order.
    pub fn phrase_ids(&self) -> Vec<String> {
        self.heads.iter().map(|h| h.id.clone()).collect()
    }

    /// Fill the mel window from `pcm` (16 kHz mono) without scoring, as if it had streamed in
    /// silence. For a caller that runs the session only while its own VAD hears speech: on the
    /// way in, `reset` and prefill the last `context_ms` it kept, and the phrase onset is in the
    /// window when scoring starts. Bypasses the session's own VAD gate.
    pub fn prefill(&mut self, pcm: &[f32]) {
        let mut buf = core::mem::take(&mut self.pcm_buf);
        buf.extend_from_slice(pcm);
        let hop = self.cfg.hop_samples.max(1);
        let used = buf.len() - buf.len() % hop;
        let frames = self.mel.push(&buf[..used]);
        for n in &mut self.nets {
            n.append_mel_frames(&frames);
        }
        if let Some(v) = &mut self.verifier {
            v.net.append_mel_frames(&frames);
        }
        self.samples_seen += used;
        buf.drain(..used);
        self.pcm_buf = buf;
    }

    /// Push PCM (16 kHz mono). Emits events at `hop_samples` cadence.
    pub fn push(&mut self, pcm: &[f32]) -> Vec<WakeEvent> {
        // Walk the buffered PCM a hop at a time and drop what was used once at the end (no
        // per-hop allocation or shifting).
        let mut buf = core::mem::take(&mut self.pcm_buf);
        buf.extend_from_slice(pcm);
        let hop = self.cfg.hop_samples.max(1);
        let mut events = Vec::new();
        let used = buf.len() - buf.len() % hop;
        for chunk in buf[..used].chunks_exact(hop) {
            self.push_ring(chunk);
            self.samples_seen += hop;
            let t_ms = self.samples_seen as f32 * 1000.0 / SAMPLE_RATE_16K as f32;
            let latency_ms = samples_to_ms(hop);

            // The mel window is fed either way (the VAD needs a few frames to be sure, and
            // the phrase onset must still be in the window then); only scoring is gated.
            let frames = self.mel.push(chunk);
            if let Some(v) = &mut self.verifier {
                v.net.append_mel_frames(&frames);
            }
            let speech = self.vad_allows(chunk);
            if !speech {
                for n in &mut self.nets {
                    n.append_mel_frames(&frames);
                }
                events.push(WakeEvent::Idle { t_ms });
                continue;
            }

            // Each network scores once (streaming); its phrases read their outputs.
            let scores: Vec<Vec<f32>> = self
                .nets
                .iter_mut()
                .map(|n| {
                    let k = n.n_out();
                    n.push_mel_frames_all(&frames)
                        .map_or_else(|| vec![0.0; k], <[f32]>::to_vec)
                })
                .collect();
            let mut fires: Vec<(String, f32)> = Vec::new();
            // The verifier's scores, computed at most once a hop, when something fires.
            let mut verified: Option<Vec<f32>> = None;
            for h in &mut self.heads {
                h.record(scores[h.net][h.output]);
                let score = h.smoothed();
                h.above = if score >= h.threshold { h.above + 1 } else { 0 };
                if h.above == 0 {
                    h.vetoed = false;
                }
                let mut fire =
                    h.above >= h.min_hops.unwrap_or(self.cfg.min_hops).max(1) && !h.vetoed;
                if fire && t_ms - h.last_fire_ms < self.cfg.cooldown_ms {
                    fire = false;
                }
                if fire && let Some(v) = &mut self.verifier {
                    let vs = verified.get_or_insert_with(|| {
                        v.net.score_window().map_or_else(Vec::new, <[f32]>::to_vec)
                    });
                    let need = h.verify_threshold.unwrap_or(v.threshold);
                    if vs.get(h.verify_output).is_none_or(|&p| p < need) {
                        fire = false;
                        h.vetoed = true;
                    }
                }
                if fire {
                    h.last_fire_ms = t_ms;
                    fires.push((h.id.clone(), score));
                }
            }
            let (speaker_id, speaker_score) = self.speaker_annotate();
            if self.speaker_blocks(speaker_score) {
                events.push(WakeEvent::Idle { t_ms });
                continue;
            }
            if fires.is_empty() {
                events.push(WakeEvent::Idle { t_ms });
            } else {
                for (phrase_id, score) in fires {
                    events.push(WakeEvent::Candidate {
                        phrase_id,
                        score,
                        t_ms,
                        latency_ms,
                        speaker_id: speaker_id.clone(),
                        speaker_score,
                    });
                }
            }
        }
        buf.drain(..used);
        self.pcm_buf = buf;
        events
    }

    /// The last `context_ms` of PCM, for the speaker-ID gate only (nothing else reads it).
    /// Compacted when it reaches twice that, so a hop costs no shifting.
    fn push_ring(&mut self, chunk: &[f32]) {
        if !self.wants_ring() {
            return;
        }
        let max = ((self.cfg.context_ms / 1000.0) * SAMPLE_RATE_16K as f32) as usize;
        let max = max.max(chunk.len());
        self.pcm_ring.extend_from_slice(chunk);
        if self.pcm_ring.len() >= 2 * max {
            let drop = self.pcm_ring.len() - max;
            self.pcm_ring.drain(..drop);
        }
    }

    #[cfg(feature = "speaker-id")]
    fn ring(&self) -> &[f32] {
        let max = ((self.cfg.context_ms / 1000.0) * SAMPLE_RATE_16K as f32) as usize;
        &self.pcm_ring[self.pcm_ring.len().saturating_sub(max)..]
    }

    fn wants_ring(&self) -> bool {
        #[cfg(feature = "speaker-id")]
        {
            self.cfg.speaker_id && self.speaker.is_some()
        }
        #[cfg(not(feature = "speaker-id"))]
        {
            false
        }
    }

    fn speaker_annotate(&self) -> (Option<String>, Option<f32>) {
        #[cfg(feature = "speaker-id")]
        {
            if !self.cfg.speaker_id {
                return (None, None);
            }
            let Some(gate) = self.speaker.as_ref() else {
                return (None, None);
            };
            match gate.best_match(self.ring()) {
                Ok((id, score)) => (Some(id), Some(score)),
                Err(_) => (None, None),
            }
        }
        #[cfg(not(feature = "speaker-id"))]
        {
            (None, None)
        }
    }

    fn speaker_blocks(&self, speaker_score: Option<f32>) -> bool {
        #[cfg(feature = "speaker-id")]
        {
            let Some(gate) = self.speaker.as_ref() else {
                return false;
            };
            if !self.cfg.speaker_id || !gate.cfg.require_match {
                return false;
            }
            match speaker_score {
                Some(s) => s < gate.cfg.threshold,
                None => true,
            }
        }
        #[cfg(not(feature = "speaker-id"))]
        {
            let _ = speaker_score;
            false
        }
    }

    #[cfg(feature = "earshot")]
    fn vad_allows(&mut self, chunk: &[f32]) -> bool {
        let Some(det) = self.vad.as_mut() else {
            return true;
        };
        self.vad_pcm.extend_from_slice(chunk);
        let frame = rlx_vad::earshot::FRAME_SAMPLES;
        let mut max_p = 0.0f32;
        while self.vad_pcm.len() >= frame {
            let fr: Vec<f32> = self.vad_pcm.drain(..frame).collect();
            max_p = max_p.max(det.predict_f32(&fr));
        }
        max_p >= self.cfg.vad_threshold
    }

    #[cfg(not(feature = "earshot"))]
    fn vad_allows(&mut self, _chunk: &[f32]) -> bool {
        true
    }
}

fn samples_to_ms(n: usize) -> f32 {
    n as f32 * 1000.0 / SAMPLE_RATE_16K as f32
}

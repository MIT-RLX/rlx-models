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

struct PhraseHead {
    id: String,
    threshold: f32,
    cnn: WakeCnn,
    last_fire_ms: f32,
}

pub struct WakewordSession {
    cfg: WakewordConfig,
    mel: MelFrontend,
    heads: Vec<PhraseHead>,
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
        let hop_len = MelConfig::default().hop_length;
        let window_frames = cfg.context_frames(hop_len);
        let mut heads = Vec::new();
        for (id, w) in weights {
            let thr = cfg
                .phrases
                .iter()
                .find(|p| p.id == id)
                .map(|p| p.threshold)
                .unwrap_or(0.5);
            heads.push(PhraseHead {
                id,
                threshold: thr,
                cnn: WakeCnn::new(w).with_window_frames(window_frames),
                last_fire_ms: -1.0e9,
            });
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
            heads,
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
        let window_frames = self.cfg.context_frames(hop_len);
        if let Some(h) = self.heads.iter_mut().find(|h| h.id == id) {
            h.cnn = WakeCnn::new(weights).with_window_frames(window_frames);
            h.threshold = threshold;
            h.last_fire_ms = -1.0e9;
        } else {
            self.heads.push(PhraseHead {
                id: id.clone(),
                threshold,
                cnn: WakeCnn::new(weights).with_window_frames(window_frames),
                last_fire_ms: -1.0e9,
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
        for h in &mut self.heads {
            h.cnn.reset();
            h.last_fire_ms = -1.0e9;
        }
        #[cfg(feature = "earshot")]
        {
            if let Some(v) = self.vad.as_mut() {
                v.reset();
            }
            self.vad_pcm.clear();
        }
    }

    /// Push PCM (16 kHz mono). Emits events at `hop_samples` cadence.
    pub fn push(&mut self, pcm: &[f32]) -> Vec<WakeEvent> {
        self.pcm_buf.extend_from_slice(pcm);
        let hop = self.cfg.hop_samples.max(1);
        let mut events = Vec::new();
        while self.pcm_buf.len() >= hop {
            let chunk: Vec<f32> = self.pcm_buf.drain(..hop).collect();
            self.push_ring(&chunk);
            self.samples_seen += hop;
            let t_ms = self.samples_seen as f32 * 1000.0 / SAMPLE_RATE_16K as f32;
            let latency_ms = samples_to_ms(hop);

            let speech = self.vad_allows(&chunk);
            if !speech {
                events.push(WakeEvent::Idle { t_ms });
                continue;
            }

            let frames = self.mel.push(&chunk);
            let mut fires: Vec<(String, f32)> = Vec::new();
            for h in &mut self.heads {
                let score = h.cnn.push_mel_frames(&frames);
                let mut fire = score >= h.threshold;
                if fire && t_ms - h.last_fire_ms < self.cfg.cooldown_ms {
                    fire = false;
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
        events
    }

    fn push_ring(&mut self, chunk: &[f32]) {
        let max = ((self.cfg.context_ms / 1000.0) * SAMPLE_RATE_16K as f32) as usize;
        let max = max.max(chunk.len());
        self.pcm_ring.extend_from_slice(chunk);
        if self.pcm_ring.len() > max {
            let drop = self.pcm_ring.len() - max;
            self.pcm_ring.drain(..drop);
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
            match gate.best_match(&self.pcm_ring) {
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

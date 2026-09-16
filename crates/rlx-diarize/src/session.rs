// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

use crate::cluster::cluster_embeddings;
use crate::embed::{EmbedBackend, window_samples};
use anyhow::Result;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeakerTurn {
    pub speaker_id: usize,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone)]
pub struct DiarizeConfig {
    pub window_sec: f32,
    pub hop_sec: f32,
    /// Cosine distance threshold (1 − similarity). Lower → more speakers.
    pub cluster_threshold: f32,
    /// Directory with `graphs/wespeaker.rlxp` or `onnx/wespeaker.onnx`.
    pub wespeaker_dir: Option<PathBuf>,
    /// RLX device for WeSpeaker (ignored for mel-stat).
    pub device: Device,
}

impl Default for DiarizeConfig {
    fn default() -> Self {
        Self {
            // Match WeSpeaker FIXED_FRAMES=148 (~1.5 s @ 16 kHz fbank).
            window_sec: 1.5,
            hop_sec: 0.75,
            cluster_threshold: 0.35,
            wespeaker_dir: None,
            device: Device::Cpu,
        }
    }
}

impl DiarizeConfig {
    /// Resolve WeSpeaker dir from `RLX_WESPEAKER_DIR` / `RLX_WESPEAKER_ONNX` or roots.
    pub fn with_auto_wespeaker(self, search_roots: &[&Path]) -> Self {
        if self.wespeaker_dir.is_some() {
            return self;
        }
        #[cfg(feature = "wespeaker")]
        {
            if let Some(dir) = rlx_wespeaker::resolve_model_dir(search_roots) {
                self.wespeaker_dir = Some(dir);
            }
        }
        #[cfg(not(feature = "wespeaker"))]
        {
            let _ = search_roots;
        }
        self
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
}

pub struct DiarizeSession {
    cfg: DiarizeConfig,
    backend: EmbedBackend,
}

impl DiarizeSession {
    pub fn new(cfg: DiarizeConfig) -> Result<Self> {
        let backend = Self::build_backend(&cfg)?;
        Ok(Self { cfg, backend })
    }

    /// Mel-stat backend only (always succeeds).
    pub fn mel_stat(cfg: DiarizeConfig) -> Self {
        Self {
            cfg: DiarizeConfig {
                wespeaker_dir: None,
                ..cfg
            },
            backend: EmbedBackend::mel_stat(),
        }
    }

    fn build_backend(cfg: &DiarizeConfig) -> Result<EmbedBackend> {
        #[cfg(feature = "wespeaker")]
        if let Some(ref dir) = cfg.wespeaker_dir {
            return EmbedBackend::wespeaker_on(dir, cfg.device);
        }
        let _ = cfg;
        Ok(EmbedBackend::mel_stat())
    }

    pub fn config(&self) -> &DiarizeConfig {
        &self.cfg
    }

    pub fn using_wespeaker(&self) -> bool {
        #[cfg(feature = "wespeaker")]
        {
            matches!(self.backend, EmbedBackend::WeSpeaker(_))
        }
        #[cfg(not(feature = "wespeaker"))]
        {
            false
        }
    }

    pub fn diarize(&mut self, pcm: &[f32]) -> Result<Vec<SpeakerTurn>> {
        let win = window_samples(self.cfg.window_sec);
        let hop = window_samples(self.cfg.hop_sec).max(1);
        if pcm.len() < win / 2 {
            return Ok(vec![SpeakerTurn {
                speaker_id: 0,
                start: 0.0,
                end: pcm.len() as f32 / SAMPLE_RATE as f32,
            }]);
        }

        let mut embeddings = Vec::new();
        let mut times = Vec::new();
        let mut start = 0usize;
        while start + win <= pcm.len() {
            embeddings.push(self.backend.embed_window(&pcm[start..start + win])?);
            times.push((
                start as f32 / SAMPLE_RATE as f32,
                (start + win) as f32 / SAMPLE_RATE as f32,
            ));
            start += hop;
        }

        if embeddings.is_empty() {
            let emb = self.backend.embed_window(pcm)?;
            embeddings.push(emb);
            times.push((0.0, pcm.len() as f32 / SAMPLE_RATE as f32));
        }

        let labels = cluster_embeddings(&embeddings, self.cfg.cluster_threshold);
        merge_turns(&times, &labels)
    }
}

fn merge_turns(times: &[(f32, f32)], labels: &[usize]) -> Result<Vec<SpeakerTurn>> {
    if times.is_empty() || labels.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut cur = labels[0];
    let mut t0 = times[0].0;
    let mut t1 = times[0].1;
    for (i, &lab) in labels.iter().enumerate().skip(1) {
        if lab == cur {
            t1 = times[i].1;
        } else {
            out.push(SpeakerTurn {
                speaker_id: cur,
                start: t0,
                end: t1,
            });
            cur = lab;
            t0 = times[i].0;
            t1 = times[i].1;
        }
    }
    out.push(SpeakerTurn {
        speaker_id: cur,
        start: t0,
        end: t1,
    });
    Ok(out)
}

/// Map a time range to the speaker turn with maximum overlap.
pub fn best_speaker(turns: &[SpeakerTurn], start: f32, end: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for t in turns {
        let overlap = (end.min(t.end) - start.max(t.start)).max(0.0);
        if overlap > 0.0 && best.map(|(_, o)| overlap > o).unwrap_or(true) {
            best = Some((t.speaker_id, overlap));
        }
    }
    best.map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diarize_short_pcm_single_speaker() {
        let pcm = vec![0.01f32; 16_000 * 3];
        let mut session = DiarizeSession::mel_stat(DiarizeConfig::default());
        let turns = session.diarize(&pcm).unwrap();
        assert!(!turns.is_empty());
        assert_eq!(turns[0].speaker_id, 0);
    }
}

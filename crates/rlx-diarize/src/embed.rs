// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Speaker embedding backends for diarization.

use anyhow::Result;

const SAMPLE_RATE: u32 = 16_000;

/// How to turn a PCM window into an embedding.
pub enum EmbedBackend {
    /// Mel energy statistics (no neural weights — always available).
    MelStat,
    /// WeSpeaker ResNet34-LM on native RLX.
    #[cfg(feature = "wespeaker")]
    WeSpeaker(Box<rlx_wespeaker::WeSpeaker>),
}

impl EmbedBackend {
    pub fn mel_stat() -> Self {
        Self::MelStat
    }

    #[cfg(feature = "wespeaker")]
    pub fn wespeaker_on(
        dir: impl AsRef<std::path::Path>,
        device: rlx_runtime::Device,
    ) -> Result<Self> {
        Ok(Self::WeSpeaker(Box::new(
            rlx_wespeaker::WeSpeaker::open_on(dir, device)?,
        )))
    }

    pub fn embed_window(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        match self {
            Self::MelStat => Ok(mel_stat_embed(pcm)),
            #[cfg(feature = "wespeaker")]
            Self::WeSpeaker(model) => model.embed_pcm(pcm),
        }
    }
}

/// Lightweight speaker embedding from mel energy statistics.
pub fn mel_stat_embed(pcm: &[f32]) -> Vec<f32> {
    let n_mels = 80usize;
    let mut emb = vec![0f32; n_mels];
    if pcm.is_empty() {
        return emb;
    }
    let frame = pcm.len() / n_mels.max(1);
    for (i, e) in emb.iter_mut().enumerate().take(n_mels) {
        let start = i * frame;
        let end = ((i + 1) * frame).min(pcm.len());
        if start < end {
            *e = pcm[start..end].iter().map(|x| x * x).sum::<f32>() / (end - start) as f32;
        }
    }
    l2_normalize(&mut emb);
    emb
}

pub fn l2_normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn window_samples(window_sec: f32) -> usize {
    (window_sec * SAMPLE_RATE as f32) as usize
}

/// Back-compat alias used by older callers.
pub fn embed_window(pcm: &[f32]) -> Vec<f32> {
    mel_stat_embed(pcm)
}

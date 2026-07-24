// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_wake::{
    MelConfig, MelFrontend, OWW_CHUNK_SAMPLES, WakeConfig, WakeEngine, WakeStep,
};
use std::collections::VecDeque;
use std::path::Path;

use crate::embedding::{EMBED_DIM, EmbeddingNet, EmbeddingWeights, MEL_STEP, MEL_WINDOW};
use crate::phrase::{EMBED_HISTORY, PhraseWeights, score_phrase};

#[derive(Clone)]
pub struct OpenWakeWordWeights {
    pub embed: EmbeddingWeights,
    pub phrase: PhraseWeights,
}

impl OpenWakeWordWeights {
    pub fn stub(keyword: &str) -> Self {
        Self {
            embed: EmbeddingWeights::stub(32),
            phrase: PhraseWeights::stub(keyword),
        }
    }

    pub fn load_dir(dir: &Path, keyword: &str) -> Result<Self> {
        Ok(Self {
            embed: EmbeddingWeights::load(&dir.join("embedding.safetensors"))?,
            phrase: PhraseWeights::load(&dir.join("phrase.safetensors"), keyword)?,
        })
    }

    pub fn save_dir(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        self.embed.save(&dir.join("embedding.safetensors"))?;
        self.phrase.save(&dir.join("phrase.safetensors"))?;
        Ok(())
    }
}

pub struct OpenWakeWordEngine {
    cfg: WakeConfig,
    mel: MelFrontend,
    embed: EmbeddingNet,
    phrase: PhraseWeights,
    mel_frames: VecDeque<Vec<f32>>,
    embeds: VecDeque<[f32; EMBED_DIM]>,
    samples_seen: usize,
    last_fire_ms: f32,
    device_label: String,
}

impl OpenWakeWordEngine {
    pub fn new(weights: OpenWakeWordWeights, mut cfg: WakeConfig) -> Self {
        cfg.chunk_samples = OWW_CHUNK_SAMPLES;
        cfg.keyword = weights.phrase.keyword.clone();
        Self {
            mel: MelFrontend::new(MelConfig::default()),
            embed: EmbeddingNet::new(weights.embed),
            phrase: weights.phrase,
            mel_frames: VecDeque::new(),
            embeds: VecDeque::new(),
            samples_seen: 0,
            last_fire_ms: -1.0e9,
            device_label: "cpu".into(),
            cfg,
        }
    }

    pub fn with_device_label(mut self, label: impl Into<String>) -> Self {
        self.device_label = label.into();
        self
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }
}

impl WakeEngine for OpenWakeWordEngine {
    fn push_pcm(&mut self, samples: &[f32]) -> Result<Vec<WakeStep>> {
        let flat = self.mel.push(samples);
        let n_mels = self.mel.n_mels();
        let n_new = flat.len() / n_mels;
        for i in 0..n_new {
            let row = flat[i * n_mels..(i + 1) * n_mels].to_vec();
            self.mel_frames.push_back(row);
        }
        // Produce embeddings when we have enough mel frames; slide by MEL_STEP.
        while self.mel_frames.len() >= MEL_WINDOW {
            let mut window = Vec::with_capacity(MEL_WINDOW * n_mels);
            for f in self.mel_frames.iter().take(MEL_WINDOW) {
                window.extend_from_slice(f);
            }
            let emb = self.embed.forward(&window);
            self.embeds.push_back(emb);
            if self.embeds.len() > EMBED_HISTORY {
                self.embeds.pop_front();
            }
            for _ in 0..MEL_STEP.min(self.mel_frames.len()) {
                self.mel_frames.pop_front();
            }
        }

        let score = if self.embeds.len() >= EMBED_HISTORY {
            let mut hist = [[0.0f32; EMBED_DIM]; EMBED_HISTORY];
            for (i, e) in self.embeds.iter().rev().take(EMBED_HISTORY).enumerate() {
                hist[EMBED_HISTORY - 1 - i] = *e;
            }
            score_phrase(&self.phrase, &hist)
        } else {
            0.0
        };

        self.samples_seen += samples.len();
        let t_ms = self.samples_seen as f32 * 1000.0 / 16_000.0;
        let mut fired = score >= self.cfg.threshold;
        if fired && t_ms - self.last_fire_ms < self.cfg.cooldown_ms {
            fired = false;
        }
        if fired {
            self.last_fire_ms = t_ms;
        }
        Ok(vec![WakeStep { score, fired, t_ms }])
    }

    fn reset(&mut self) {
        self.mel.reset();
        self.mel_frames.clear();
        self.embeds.clear();
        self.samples_seen = 0;
        self.last_fire_ms = -1.0e9;
    }

    fn config(&self) -> &WakeConfig {
        &self.cfg
    }
}

// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_wake::{
    MelConfig, MelFrontend, OWW_CHUNK_SAMPLES, WakeCnn, WakeCnnConfig, WakeCnnWeights, WakeConfig,
    WakeEngine, WakeStep,
};
use std::path::Path;

#[derive(Clone)]
pub struct NanoWakeWordWeights {
    pub cnn: WakeCnnWeights,
    pub lite: bool,
}

impl NanoWakeWordWeights {
    pub fn stub(lite: bool, keyword: &str) -> Self {
        let _ = keyword;
        let cfg = if lite {
            WakeCnnConfig::lite()
        } else {
            WakeCnnConfig::full()
        };
        Self {
            cnn: WakeCnnWeights::stub(cfg),
            lite,
        }
    }

    pub fn load(path: &Path, lite: bool) -> Result<Self> {
        Ok(Self {
            cnn: WakeCnnWeights::load(path)?,
            lite,
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.cnn.save(path)
    }
}

pub struct NanoWakeWordEngine {
    cfg: WakeConfig,
    mel: MelFrontend,
    cnn: WakeCnn,
    samples_seen: usize,
    last_fire_ms: f32,
    device_label: String,
    lite: bool,
}

impl NanoWakeWordEngine {
    pub fn new(weights: NanoWakeWordWeights, mut cfg: WakeConfig) -> Self {
        cfg.chunk_samples = OWW_CHUNK_SAMPLES;
        Self {
            mel: MelFrontend::new(MelConfig::default()),
            cnn: WakeCnn::new(weights.cnn),
            samples_seen: 0,
            last_fire_ms: -1.0e9,
            device_label: "cpu".into(),
            lite: weights.lite,
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

    pub fn is_lite(&self) -> bool {
        self.lite
    }
}

impl WakeEngine for NanoWakeWordEngine {
    fn push_pcm(&mut self, samples: &[f32]) -> Result<Vec<WakeStep>> {
        let frames = self.mel.push(samples);
        let score = self.cnn.push_mel_frames(&frames);
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
        self.cnn.reset();
        self.samples_seen = 0;
        self.last_fire_ms = -1.0e9;
    }

    fn config(&self) -> &WakeConfig {
        &self.cfg
    }
}

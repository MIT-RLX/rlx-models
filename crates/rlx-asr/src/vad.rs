// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Energy-based VAD / endpointing (RLX-native; no external speech assets).

use anyhow::{bail, Result};

/// Default RMS energy threshold on float PCM.
pub const ENERGY_THRESH: f32 = 1e-4;

pub struct Vad {
    pub energy_thresh: f32,
    pub activate_ms: u32,
    pub deactivate_ms: u32,
}

impl Default for Vad {
    fn default() -> Self {
        Self {
            energy_thresh: ENERGY_THRESH,
            activate_ms: 30,
            deactivate_ms: 500,
        }
    }
}

impl Vad {
    pub fn is_speech(&self, pcm: &[f32], sample_rate: u32) -> bool {
        let e: f32 = pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len().max(1) as f32;
        e > self.energy_thresh && sample_rate > 0
    }

    pub fn trim(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if pcm.is_empty() {
            bail!("empty pcm");
        }
        let win = (sample_rate as usize * 20) / 1000;
        let mut start = 0;
        while start + win <= pcm.len() && !self.is_speech(&pcm[start..start + win], sample_rate) {
            start += win;
        }
        let mut end = pcm.len();
        while end > start + win && !self.is_speech(&pcm[end - win..end], sample_rate) {
            end -= win;
        }
        Ok(pcm[start..end].to_vec())
    }
}

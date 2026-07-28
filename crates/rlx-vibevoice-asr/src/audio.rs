// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Audio front-end for VibeVoice-ASR — mirrors VibeASR.cpp `utils/audio_io.h`:
// mono float32 at 24 kHz, linear resample, and RMS normalization to −25 dBFS.

use crate::config::{TARGET_DBFS, TARGET_SR};

/// Loaded, preprocessed audio ready for the VAE encoder.
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Mono float32 samples at [`TARGET_SR`].
    pub samples: Vec<f32>,
    pub sample_rate: usize,
    pub duration_sec: f32,
}

/// Linear-interpolation resampler (matches `resample_linear`).
pub fn resample_linear(input: &[f32], src_rate: usize, dst_rate: usize) -> Vec<f32> {
    if src_rate == dst_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = (input.len() as f64 / ratio) as usize;
    let mut out = vec![0f32; out_len];
    for (i, slot) in out.iter_mut().enumerate() {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;
        *slot = if idx + 1 < input.len() {
            ((1.0 - frac) * input[idx] as f64 + frac * input[idx + 1] as f64) as f32
        } else if idx < input.len() {
            input[idx]
        } else {
            0.0
        };
    }
    out
}

/// RMS normalization to `target_db_fs` (matches `normalize_audio`):
/// `scalar = 10^(target/20) / (rms + 1e-6)`.
pub fn normalize_audio(samples: &mut [f32], target_db_fs: f32) {
    if samples.is_empty() {
        return;
    }
    let eps = 1e-6f32;
    let sum_sq: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let rms = (sum_sq / samples.len() as f64).sqrt() as f32;
    if rms < eps {
        return; // silence
    }
    let target_linear = 10f32.powf(target_db_fs / 20.0);
    let scalar = target_linear / (rms + eps);
    for v in samples.iter_mut() {
        *v *= scalar;
    }
}

impl AudioData {
    /// Build from mono samples already at `src_rate`: resample to 24 kHz and
    /// (optionally) RMS-normalize.
    pub fn from_mono(mono: &[f32], src_rate: usize, normalize: bool) -> Self {
        let mut samples = resample_linear(mono, src_rate, TARGET_SR);
        if normalize {
            normalize_audio(&mut samples, TARGET_DBFS);
        }
        let duration_sec = samples.len() as f32 / TARGET_SR as f32;
        Self {
            samples,
            sample_rate: TARGET_SR,
            duration_sec,
        }
    }

    /// Downmix interleaved multi-channel PCM to mono, then [`AudioData::from_mono`].
    pub fn from_interleaved(
        pcm: &[f32],
        channels: usize,
        src_rate: usize,
        normalize: bool,
    ) -> Self {
        let mono = if channels <= 1 {
            pcm.to_vec()
        } else {
            let frames = pcm.len() / channels;
            let mut m = vec![0f32; frames];
            for (i, slot) in m.iter_mut().enumerate() {
                let mut s = 0f32;
                for c in 0..channels {
                    s += pcm[i * channels + c];
                }
                *slot = s / channels as f32;
            }
            m
        };
        Self::from_mono(&mono, src_rate, normalize)
    }

    /// Number of VAE speech frames = `ceil(n_samples / compress_ratio)`.
    pub fn n_frames(&self, compress_ratio: usize) -> usize {
        self.samples.len().div_ceil(compress_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_identity() {
        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(resample_linear(&x, 24000, 24000), x);
    }

    #[test]
    fn normalize_targets_rms() {
        let mut x = vec![0.5f32; 1000];
        normalize_audio(&mut x, -25.0);
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let target = 10f32.powf(-25.0 / 20.0);
        assert!((rms - target).abs() < 1e-3, "rms={rms} target={target}");
    }

    #[test]
    fn frame_count_ceils() {
        let a = AudioData {
            samples: vec![0.0; 3201],
            sample_rate: 24000,
            duration_sec: 0.0,
        };
        assert_eq!(a.n_frames(3200), 2);
    }
}

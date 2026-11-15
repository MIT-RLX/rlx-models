// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! The Gemma 4 audio front end (HF `Gemma4AudioFeatureExtractor`): 16 kHz
//! samples → log-mel frames for [`crate::gemma4_audio`].
//!
//! A semicausal STFT: `frame_length / 2` zeros on the left, frames of
//! `frame_length + 1` samples every `hop_length` of which the first
//! `frame_length` are used (the extra one only feeds pre-emphasis), a periodic
//! Hann window, an `fft_length` real FFT, magnitude (not power), HTK mel
//! triangles without area normalization, then `ln(mel + mel_floor)`.
//!
//! HF pads a batch to a multiple of 128 samples and masks the frames that reach
//! into the padding. Those frames are zeroed and the audio tower ignores them:
//! they never reach a valid output (the sub-sampler zeroes them before each
//! conv, the depthwise conv is causal, attention has no right context). So this
//! returns the valid frames only, and a clip runs on exactly those.

/// Settings of `preprocessor_config.json`, defaulting to the Gemma 4 values.
#[derive(Debug, Clone)]
pub struct AudioFeatureConfig {
    pub sampling_rate: usize,
    pub feature_size: usize,
    pub frame_length: usize,
    pub hop_length: usize,
    pub fft_length: usize,
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub mel_floor: f64,
    pub preemphasis: f64,
    pub preemphasis_htk_flavor: bool,
    pub input_scale_factor: f32,
}

impl Default for AudioFeatureConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 16_000,
            feature_size: 128,
            frame_length: 320,
            hop_length: 160,
            fft_length: 512,
            min_frequency: 0.0,
            max_frequency: 8000.0,
            mel_floor: 1e-3,
            preemphasis: 0.0,
            preemphasis_htk_flavor: true,
            input_scale_factor: 1.0,
        }
    }
}

impl AudioFeatureConfig {
    /// Read the `feature_extractor` block of `processor_config.json` (or a bare
    /// `preprocessor_config.json`); absent keys keep their defaults.
    pub fn from_json(v: &serde_json::Value) -> Self {
        let v = v.get("feature_extractor").unwrap_or(v);
        let mut c = Self::default();
        let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64());
        if let Some(x) = u("sampling_rate") {
            c.sampling_rate = x;
        }
        if let Some(x) = u("feature_size") {
            c.feature_size = x;
        }
        if let Some(x) = u("frame_length") {
            c.frame_length = x;
        }
        if let Some(x) = u("hop_length") {
            c.hop_length = x;
        }
        if let Some(x) = u("fft_length") {
            c.fft_length = x;
        }
        if let Some(x) = f("min_frequency") {
            c.min_frequency = x;
        }
        if let Some(x) = f("max_frequency") {
            c.max_frequency = x;
        }
        if let Some(x) = f("mel_floor") {
            c.mel_floor = x;
        }
        if let Some(x) = f("preemphasis") {
            c.preemphasis = x;
        }
        if let Some(x) = v.get("preemphasis_htk_flavor").and_then(|x| x.as_bool()) {
            c.preemphasis_htk_flavor = x;
        }
        if let Some(x) = f("input_scale_factor") {
            c.input_scale_factor = x as f32;
        }
        c
    }
}

/// The log-mel extractor with its window, DFT and mel tables precomputed.
pub struct AudioFeatureExtractor {
    cfg: AudioFeatureConfig,
    window: Vec<f32>,
    /// `[n_freq, frame_length]` DFT rows (`cos`, `-sin`).
    cos: Vec<f64>,
    sin: Vec<f64>,
    /// `[n_freq, n_mels]`.
    mel: Vec<f64>,
    n_freq: usize,
}

impl AudioFeatureExtractor {
    pub fn new(cfg: AudioFeatureConfig) -> Self {
        let n = cfg.frame_length;
        let n_freq = cfg.fft_length / 2 + 1;
        // `window_function(n)`: periodic Hann, `np.hanning(n + 1)[:-1]`, in f32.
        let window = (0..n)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32)
            .collect();
        let mut cos = vec![0f64; n_freq * n];
        let mut sin = vec![0f64; n_freq * n];
        for k in 0..n_freq {
            for t in 0..n {
                // `(k · t) mod fft_length` keeps the angle small and exact.
                let a = 2.0 * std::f64::consts::PI * ((k * t) % cfg.fft_length) as f64
                    / cfg.fft_length as f64;
                cos[k * n + t] = a.cos();
                sin[k * n + t] = a.sin();
            }
        }
        let mel = mel_filter_bank_htk(
            n_freq,
            cfg.feature_size,
            cfg.min_frequency,
            cfg.max_frequency,
            cfg.sampling_rate,
        );
        Self {
            cfg,
            window,
            cos,
            sin,
            mel,
            n_freq,
        }
    }

    pub fn config(&self) -> &AudioFeatureConfig {
        &self.cfg
    }

    /// Valid frames for `n_samples` samples.
    pub fn num_frames(&self, n_samples: usize) -> usize {
        let padded = n_samples + self.cfg.frame_length / 2;
        let size = self.cfg.frame_length + 1;
        if padded < size {
            0
        } else {
            (padded - size) / self.cfg.hop_length + 1
        }
    }

    /// Log-mel features `[frames · feature_size]` of 16 kHz mono `samples`,
    /// valid frames only.
    pub fn extract(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let c = &self.cfg;
        let n = c.frame_length;
        let pad_left = n / 2;
        let frames = self.num_frames(samples.len());
        let sample = |i: usize| -> f32 {
            // Index into the left-padded signal.
            if i < pad_left {
                0.0
            } else {
                samples.get(i - pad_left).copied().unwrap_or(0.0) * c.input_scale_factor
            }
        };
        let mut out = vec![0f32; frames * c.feature_size];
        let mut frame = vec![0f32; n];
        let mut mag = vec![0f64; self.n_freq];
        for f in 0..frames {
            let start = f * c.hop_length;
            // `frame_length + 1` samples; pre-emphasis may use the extra one.
            let raw: Vec<f32> = (0..=n).map(|t| sample(start + t)).collect();
            if c.preemphasis > 0.0 {
                let p = c.preemphasis as f32;
                if c.preemphasis_htk_flavor {
                    frame[0] = raw[0] * (1.0 - p);
                    for t in 1..n {
                        frame[t] = raw[t] - p * raw[t - 1];
                    }
                } else {
                    for t in 0..n {
                        frame[t] = raw[t + 1] - p * raw[t];
                    }
                }
            } else {
                frame.copy_from_slice(&raw[..n]);
            }
            for (x, w) in frame.iter_mut().zip(&self.window) {
                *x *= w;
            }
            for (k, m) in mag.iter_mut().enumerate() {
                let (cr, sr) = (&self.cos[k * n..(k + 1) * n], &self.sin[k * n..(k + 1) * n]);
                let (mut re, mut im) = (0f64, 0f64);
                for t in 0..n {
                    let x = frame[t] as f64;
                    re += x * cr[t];
                    im -= x * sr[t];
                }
                *m = (re * re + im * im).sqrt();
            }
            let row = &mut out[f * c.feature_size..(f + 1) * c.feature_size];
            for (j, o) in row.iter_mut().enumerate() {
                let mut acc = 0f64;
                for (k, m) in mag.iter().enumerate() {
                    acc += m * self.mel[k * c.feature_size + j];
                }
                *o = (acc + c.mel_floor).ln() as f32;
            }
        }
        (out, frames)
    }
}

/// Audio-tower output length for `frames` mel frames: two stride-2 convs
/// (kernel 3, padding 1), each `ceil(t / 2)`. The processor inserts this many
/// audio placeholder tokens.
pub fn audio_soft_tokens(frames: usize) -> usize {
    let half = |t: usize| t.div_ceil(2);
    half(half(frames))
}

/// `transformers.audio_utils.mel_filter_bank(norm=None, mel_scale="htk")`:
/// `[n_freq, n_mels]` triangles between `n_mels + 2` points evenly spaced on
/// the HTK mel scale.
fn mel_filter_bank_htk(
    n_freq: usize,
    n_mels: usize,
    f_min: f64,
    f_max: f64,
    sr: usize,
) -> Vec<f64> {
    let to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (m_lo, m_hi) = (to_mel(f_min), to_mel(f_max));
    let pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| to_hz(m_lo + (m_hi - m_lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let nyq = (sr / 2) as f64;
    let mut out = vec![0f64; n_freq * n_mels];
    for k in 0..n_freq {
        let fk = nyq * k as f64 / (n_freq - 1) as f64;
        for m in 0..n_mels {
            let down = -(pts[m] - fk) / (pts[m + 1] - pts[m]);
            let up = (pts[m + 2] - fk) / (pts[m + 2] - pts[m + 1]);
            out[k * n_mels + m] = down.min(up).max(0.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_and_token_counts_follow_the_processor() {
        let fx = AudioFeatureExtractor::new(AudioFeatureConfig::default());
        // `(n + 160 - 321) // 160 + 1`.
        assert_eq!(fx.num_frames(83_200), 519);
        assert_eq!(fx.num_frames(11_213), 70);
        assert_eq!(fx.num_frames(160), 0);
        assert_eq!(fx.num_frames(161), 1);
        assert_eq!(audio_soft_tokens(519), 130);
        assert_eq!(audio_soft_tokens(70), 18);
        assert_eq!(audio_soft_tokens(954), 239);
    }

    /// Triangles never exceed one. The lowest HTK filters are narrower than
    /// the 31.25 Hz bin spacing, so some catch no bin at all; HF builds the
    /// same bank and silences the warning about it. Filters above 1 kHz are
    /// wide enough to be non-empty.
    #[test]
    fn mel_triangles_stay_within_one() {
        let m = mel_filter_bank_htk(257, 128, 0.0, 8000.0, 16_000);
        let peaks: Vec<f64> = (0..128)
            .map(|j| (0..257).map(|k| m[k * 128 + j]).fold(0.0, f64::max))
            .collect();
        assert!(peaks.iter().all(|p| *p <= 1.0 + 1e-12));
        assert!(peaks[32..].iter().all(|p| *p > 0.0));
        let empty = peaks.iter().filter(|p| **p == 0.0).count();
        assert!(empty < 32, "{empty} empty filters");
    }
}

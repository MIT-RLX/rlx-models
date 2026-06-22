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

//! Shared audio front-end utilities — a single linear resampler used by every
//! codec / ASR / TTS crate, replacing the byte-identical copies that had been
//! pasted into ~20 `audio.rs` files. Dependency-free (no `hound`) so the core
//! crate stays light; WAV file I/O remains in the leaf crates.

/// Resample a **mono** f32 signal from `from_hz` to `to_hz` with linear
/// interpolation. Identity (clone) when the rates match or the input is empty.
///
/// This is behaviour-identical to the resamplers previously duplicated in the
/// codec crates, so callers can switch to it without changing output.
pub fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from_hz as f64 / to_hz as f64;
    let out_len = (samples.len() as f64 / ratio).ceil() as usize;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * ratio;
            let lo = src.floor() as usize;
            let hi = (lo + 1).min(samples.len() - 1);
            let frac = (src - lo as f64) as f32;
            samples[lo] * (1.0 - frac) + samples[hi] * frac
        })
        .collect()
}

/// Resample an **interleaved** multi-channel f32 signal (frame = one sample per
/// channel) from `from_hz` to `to_hz`, interpolating each channel independently.
/// `channels` must be ≥ 1 and divide `samples.len()`.
pub fn resample_linear_interleaved(
    samples: &[f32],
    channels: usize,
    from_hz: u32,
    to_hz: u32,
) -> Vec<f32> {
    debug_assert!(channels >= 1, "channels must be >= 1");
    if channels <= 1 {
        return resample_linear(samples, from_hz, to_hz);
    }
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let in_frames = samples.len() / channels;
    if in_frames == 0 {
        return Vec::new();
    }
    let ratio = from_hz as f64 / to_hz as f64;
    let out_frames = (in_frames as f64 / ratio).ceil() as usize;
    let mut out = vec![0.0f32; out_frames * channels];
    for of in 0..out_frames {
        let src = of as f64 * ratio;
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(in_frames - 1);
        let frac = (src - lo as f64) as f32;
        for c in 0..channels {
            let a = samples[lo * channels + c];
            let b = samples[hi * channels + c];
            out[of * channels + c] = a * (1.0 - frac) + b * frac;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_rates_match() {
        let x = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(resample_linear(&x, 24_000, 24_000), x);
        assert_eq!(resample_linear_interleaved(&x, 2, 48_000, 48_000), x);
    }

    #[test]
    fn empty_input() {
        assert!(resample_linear(&[], 16_000, 24_000).is_empty());
        assert!(resample_linear_interleaved(&[], 2, 16_000, 24_000).is_empty());
    }

    #[test]
    fn downsample_halves_length() {
        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y = resample_linear(&x, 48_000, 24_000);
        assert_eq!(y.len(), 50);
        // first sample preserved, monotone ramp stays ~linear
        assert!((y[0] - 0.0).abs() < 1e-6);
        assert!((y[10] - 20.0).abs() < 1e-3);
    }

    #[test]
    fn upsample_doubles_length() {
        let x: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let y = resample_linear(&x, 24_000, 48_000);
        assert_eq!(y.len(), 100);
        assert!((y[2] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn interleaved_matches_per_channel() {
        // Two channels: ch0 ramps up, ch1 ramps down.
        let frames = 40usize;
        let mut x = vec![0.0f32; frames * 2];
        for f in 0..frames {
            x[f * 2] = f as f32;
            x[f * 2 + 1] = (frames - f) as f32;
        }
        let y = resample_linear_interleaved(&x, 2, 40_000, 20_000);
        let ch0: Vec<f32> = (0..frames).map(|f| f as f32).collect();
        let ch1: Vec<f32> = (0..frames).map(|f| (frames - f) as f32).collect();
        let y0 = resample_linear(&ch0, 40_000, 20_000);
        let y1 = resample_linear(&ch1, 40_000, 20_000);
        assert_eq!(y.len(), y0.len() * 2);
        for f in 0..y0.len() {
            assert!((y[f * 2] - y0[f]).abs() < 1e-6);
            assert!((y[f * 2 + 1] - y1[f]).abs() < 1e-6);
        }
    }
}

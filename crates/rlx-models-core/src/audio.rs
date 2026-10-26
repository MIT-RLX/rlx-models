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

// ── Band-limited resampling (moved from rlx-tada) ─────────────────────

const SINC_LOWPASS_FILTER_WIDTH: i64 = 6;
const SINC_ROLLOFF: f64 = 0.99;

fn sinc_gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { sinc_gcd(b, a % b) }
}

/// Resample a mono signal from `from_hz` to `to_hz` with a band-limited
/// rational kernel, matching `torchaudio.functional.resample` with its default
/// `sinc_interp_hann` window (`lowpass_filter_width` 6, `rolloff` 0.99).
///
/// Use this instead of [`resample_linear`] when the output feeds a model
/// whose reference preprocessing used torchaudio — a codec encoding a voice
/// prompt, an aligner indexing frames: linear interpolation aliases enough
/// to move features by a frame.
pub fn resample_sinc(x: &[f32], from_hz: usize, to_hz: usize) -> Vec<f32> {
    if from_hz == to_hz || x.is_empty() {
        return x.to_vec();
    }
    let d = sinc_gcd(from_hz, to_hz);
    let orig = (from_hz / d) as i64;
    let new = (to_hz / d) as i64;

    let base_freq = (orig.min(new) as f64) * SINC_ROLLOFF;
    let width = (SINC_LOWPASS_FILTER_WIDTH as f64 * orig as f64 / base_freq).ceil() as i64;
    let scale = base_freq / orig as f64;
    let taps = (2 * width + orig) as usize;

    // kernels[phase][tap]
    let mut kernels = vec![0f64; new as usize * taps];
    for i in 0..new {
        for (j, idx) in (-width..width + orig).enumerate() {
            let mut t = (-(i as f64) / new as f64 + idx as f64 / orig as f64) * base_freq;
            t = t.clamp(
                -(SINC_LOWPASS_FILTER_WIDTH as f64),
                SINC_LOWPASS_FILTER_WIDTH as f64,
            );
            let window = (t * std::f64::consts::PI / SINC_LOWPASS_FILTER_WIDTH as f64 / 2.0)
                .cos()
                .powi(2);
            let t = t * std::f64::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernels[i as usize * taps + j] = sinc * window * scale;
        }
    }

    let target_len = (x.len() as f64 * to_hz as f64 / from_hz as f64).ceil() as usize;
    let mut out = Vec::with_capacity(target_len);
    // The signal is zero-padded by `width` on each side, so output sample
    // `blk * new + i` reads input taps starting at `blk * orig - width`.
    let num_blocks = x.len().div_ceil(orig as usize) + 1;
    'outer: for blk in 0..num_blocks {
        for i in 0..new as usize {
            let mut acc = 0f64;
            let base = blk as i64 * orig - width;
            for (j, k) in kernels[i * taps..(i + 1) * taps].iter().enumerate() {
                let src = base + j as i64;
                if src >= 0 && (src as usize) < x.len() {
                    acc += x[src as usize] as f64 * k;
                }
            }
            out.push(acc as f32);
            if out.len() == target_len {
                break 'outer;
            }
        }
    }
    out.resize(target_len, 0.0);
    out
}

#[cfg(test)]
mod sinc_tests {
    use super::*;

    #[test]
    fn identity_rate_is_a_passthrough() {
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(resample_sinc(&x, 24_000, 24_000), x);
    }

    #[test]
    fn output_length_follows_the_rate_ratio() {
        assert_eq!(
            resample_sinc(&vec![0f32; 24_000], 24_000, 16_000).len(),
            16_000
        );
        assert_eq!(
            resample_sinc(&vec![0f32; 16_000], 16_000, 24_000).len(),
            24_000
        );
        // Non-integer ratios round up, so no sample is silently dropped.
        assert_eq!(resample_sinc(&vec![0f32; 1_001], 24_000, 16_000).len(), 668);
    }

    #[test]
    fn preserves_a_tone_below_the_new_nyquist() {
        // 440 Hz survives a 24 k → 16 k pass essentially untouched.
        let n = 24_000;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24_000.0).sin())
            .collect();
        let y = resample_sinc(&x, 24_000, 16_000);
        // Compare away from the edges, where the kernel is still filling.
        let mut worst = 0f32;
        for i in 200..(y.len() - 200) {
            let want = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin();
            worst = worst.max((y[i] - want).abs());
        }
        assert!(worst < 0.02, "worst deviation {worst}");
    }

    #[test]
    fn rejects_energy_above_the_new_nyquist() {
        // 10 kHz cannot be represented at 16 kHz; it must be filtered out, not
        // folded back as a loud alias.
        let n = 24_000;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 10_000.0 * i as f32 / 24_000.0).sin())
            .collect();
        let y = resample_sinc(&x, 24_000, 16_000);
        let rms = (y[400..y.len() - 400].iter().map(|v| v * v).sum::<f32>()
            / (y.len() - 800) as f32)
            .sqrt();
        assert!(rms < 0.1, "alias RMS {rms}");
    }

    #[test]
    fn a_constant_signal_stays_constant() {
        let x = vec![0.5f32; 8_000];
        let y = resample_sinc(&x, 24_000, 16_000);
        for v in &y[100..y.len() - 100] {
            assert!((v - 0.5).abs() < 1e-3, "{v}");
        }
    }
}

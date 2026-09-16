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

//! Band-limited rational resampling, matching `torchaudio.functional.resample`
//! with its default `sinc_interp_hann` kernel.
//!
//! TADA needs the reference audio at two rates at once: 24 kHz for the codec
//! and 16 kHz for the aligner. The two have to describe the *same* signal at
//! the same times, because the aligner's frame indices are used to index the
//! codec's frames. Linear interpolation aliases enough at a 3:2 ratio to shift
//! CTC peaks by a frame, which moves a token's whole latent — so the upstream
//! kernel is reproduced rather than approximated.

const LOWPASS_FILTER_WIDTH: i64 = 6;
const ROLLOFF: f64 = 0.99;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Resample `x` from `from_hz` to `to_hz`.
pub fn resample(x: &[f32], from_hz: usize, to_hz: usize) -> Vec<f32> {
    if from_hz == to_hz || x.is_empty() {
        return x.to_vec();
    }
    let d = gcd(from_hz, to_hz);
    let orig = (from_hz / d) as i64;
    let new = (to_hz / d) as i64;

    let base_freq = (orig.min(new) as f64) * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH as f64 * orig as f64 / base_freq).ceil() as i64;
    let scale = base_freq / orig as f64;
    let taps = (2 * width + orig) as usize;

    // kernels[phase][tap]
    let mut kernels = vec![0f64; new as usize * taps];
    for i in 0..new {
        for (j, idx) in (-width..width + orig).enumerate() {
            let mut t = (-(i as f64) / new as f64 + idx as f64 / orig as f64) * base_freq;
            t = t.clamp(-(LOWPASS_FILTER_WIDTH as f64), LOWPASS_FILTER_WIDTH as f64);
            let window = (t * std::f64::consts::PI / LOWPASS_FILTER_WIDTH as f64 / 2.0)
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
mod tests {
    use super::*;

    #[test]
    fn identity_rate_is_a_passthrough() {
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(resample(&x, 24_000, 24_000), x);
    }

    #[test]
    fn output_length_follows_the_rate_ratio() {
        assert_eq!(resample(&vec![0f32; 24_000], 24_000, 16_000).len(), 16_000);
        assert_eq!(resample(&vec![0f32; 16_000], 16_000, 24_000).len(), 24_000);
        // Non-integer ratios round up, so no sample is silently dropped.
        assert_eq!(resample(&vec![0f32; 1_001], 24_000, 16_000).len(), 668);
    }

    #[test]
    fn preserves_a_tone_below_the_new_nyquist() {
        // 440 Hz survives a 24 k → 16 k pass essentially untouched.
        let n = 24_000;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24_000.0).sin())
            .collect();
        let y = resample(&x, 24_000, 16_000);
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
        let y = resample(&x, 24_000, 16_000);
        let rms = (y[400..y.len() - 400].iter().map(|v| v * v).sum::<f32>()
            / (y.len() - 800) as f32)
            .sqrt();
        assert!(rms < 0.1, "alias RMS {rms}");
    }

    #[test]
    fn a_constant_signal_stays_constant() {
        let x = vec![0.5f32; 8_000];
        let y = resample(&x, 24_000, 16_000);
        for v in &y[100..y.len() - 100] {
            assert!((v - 0.5).abs() < 1e-3, "{v}");
        }
    }
}

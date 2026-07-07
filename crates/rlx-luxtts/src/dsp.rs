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

//! DSP for LuxTTS: `VocosFbank` log-mel (matches torchaudio
//! `MelSpectrogram(sr=24000, n_fft=1024, hop=256, n_mels=100, center=True,
//! power=1)` then `clamp(1e-7).log()`) and ISTFT (matches `torch.istft` with a
//! periodic Hann window, `center=True`). Both are validated bit-close against
//! Python goldens in `tests/dsp_parity.rs`.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

pub const SR: u32 = 24000;
pub const N_FFT: usize = 1024;
pub const HOP: usize = 256;
pub const N_MELS: usize = 100;
pub const N_FREQ: usize = N_FFT / 2 + 1; // 513

/// Periodic Hann window of length `n` (`torch.hann_window(n, periodic=True)`).
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos() as f32)
        .collect()
}

fn hz_to_mel_htk(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}
fn mel_to_hz_htk(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// torchaudio `melscale_fbanks` (htk, norm=None). Returns `[N_MELS, N_FREQ]`
/// row-major (mel-major).
fn mel_filterbank() -> Vec<f32> {
    let f_min = 0.0;
    let f_max = SR as f64 / 2.0;
    let all_freqs: Vec<f64> = (0..N_FREQ)
        .map(|k| k as f64 * (SR as f64 / 2.0) / (N_FREQ - 1) as f64)
        .collect();
    let m_min = hz_to_mel_htk(f_min);
    let m_max = hz_to_mel_htk(f_max);
    // N_MELS + 2 mel points → hz.
    let f_pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| {
            let m = m_min + (m_max - m_min) * i as f64 / (N_MELS + 1) as f64;
            mel_to_hz_htk(m)
        })
        .collect();
    let mut fb = vec![0f32; N_MELS * N_FREQ];
    for m in 0..N_MELS {
        let (lo, ctr, hi) = (f_pts[m], f_pts[m + 1], f_pts[m + 2]);
        for (k, &f) in all_freqs.iter().enumerate() {
            let down = (f - lo) / (ctr - lo);
            let up = (hi - f) / (hi - ctr);
            let v = down.min(up).max(0.0);
            fb[m * N_FREQ + k] = v as f32;
        }
    }
    fb
}

/// Reusable log-mel extractor.
pub struct VocosFbank {
    window: Vec<f32>,
    fb: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl Default for VocosFbank {
    fn default() -> Self {
        Self::new()
    }
}

impl VocosFbank {
    pub fn new() -> Self {
        Self {
            window: hann_periodic(N_FFT),
            fb: mel_filterbank(),
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    /// `wav` (mono, 24 kHz) → log-mel `[N_MELS, T]` row-major, plus `T`.
    pub fn log_mel(&self, wav: &[f32]) -> (Vec<f32>, usize) {
        // center=True: reflect-pad N_FFT/2 on both ends.
        let pad = N_FFT / 2;
        let padded = reflect_pad(wav, pad);
        let n_frames = if padded.len() >= N_FFT {
            1 + (padded.len() - N_FFT) / HOP
        } else {
            0
        };
        let mut mel = vec![0f32; N_MELS * n_frames];
        let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut mag = vec![0f32; N_FREQ];
        for t in 0..n_frames {
            let start = t * HOP;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for (k, m) in mag.iter_mut().enumerate() {
                *m = buf[k].norm(); // power=1 → magnitude
            }
            for m in 0..N_MELS {
                let row = &self.fb[m * N_FREQ..m * N_FREQ + N_FREQ];
                let mut acc = 0f32;
                for k in 0..N_FREQ {
                    acc += row[k] * mag[k];
                }
                mel[m * n_frames + t] = acc.max(1e-7).ln();
            }
        }
        (mel, n_frames)
    }
}

/// Inverse STFT matching `torch.istft(n_fft, hop, win_length=n_fft,
/// window=hann(periodic), center=True)`. `real`/`imag` are `[N_FREQ, T]`
/// row-major. Returns the time signal of length `(T-1)*HOP`.
pub fn istft(real: &[f32], imag: &[f32], t_frames: usize) -> Vec<f32> {
    let window = hann_periodic(N_FFT);
    let ifft = FftPlanner::<f32>::new().plan_fft_inverse(N_FFT);
    let out_full = (t_frames - 1) * HOP + N_FFT;
    let mut ybuf = vec![0f32; out_full];
    let mut wsum = vec![0f32; out_full];
    let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
    for t in 0..t_frames {
        // Rebuild the full hermitian spectrum from the half-spectrum.
        for k in 0..N_FREQ {
            buf[k] = Complex32::new(real[k * t_frames + t], imag[k * t_frames + t]);
        }
        for k in 1..N_FREQ - 1 {
            buf[N_FFT - k] = buf[k].conj();
        }
        ifft.process(&mut buf);
        let start = t * HOP;
        for i in 0..N_FFT {
            let s = buf[i].re / N_FFT as f32; // rustfft ifft is unnormalized
            ybuf[start + i] += s * window[i];
            wsum[start + i] += window[i] * window[i];
        }
    }
    for i in 0..out_full {
        if wsum[i] > 1e-11 {
            ybuf[i] /= wsum[i];
        }
    }
    // center=True → trim N_FFT/2 from each end.
    let pad = N_FFT / 2;
    ybuf[pad..out_full - pad].to_vec()
}

fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        out.push(x[(pad - i).min(n - 1)]); // reflect (exclude edge sample)
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        out.push(x[n.saturating_sub(2 + i)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filterbank_shape_and_nonneg() {
        let fb = mel_filterbank();
        assert_eq!(fb.len(), N_MELS * N_FREQ);
        assert!(fb.iter().all(|&v| v >= 0.0));
        assert!(fb.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn hann_endpoints() {
        let w = hann_periodic(N_FFT);
        assert!(w[0].abs() < 1e-6);
        assert!((w[N_FFT / 2] - 1.0).abs() < 1e-3);
    }
}

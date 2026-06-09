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

//! CPU mel spectrogram (OpenAI Whisper frontend).

use crate::audio::{MelSpectrogram, SAMPLE_RATE};

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_FREQ: usize = N_FFT / 2 + 1;

/// Build log-mel for one utterance: row-major `[1, n_mels, n_frames]` (padded/truncated to `n_frames`).
pub fn pcm_to_log_mel(pcm: &[f32], n_mels: usize, n_frames: usize) -> MelSpectrogram {
    let filters = mel_filterbank(SAMPLE_RATE as f64, N_FFT, n_mels);
    let window = hann_window(N_FFT);
    let stft = stft_mag(pcm, &window);
    let n_stft_frames = stft.n_frames;
    let mut mel = vec![0f32; n_mels * n_frames];
    for fi in 0..n_stft_frames.min(n_frames) {
        for mi in 0..n_mels {
            let mut acc = 0f32;
            for bin in 0..N_FREQ {
                acc += filters[mi * N_FREQ + bin] * stft.data[fi * N_FREQ + bin];
            }
            mel[mi * n_frames + fi] = acc;
        }
    }
    let mut max = f32::NEG_INFINITY;
    for v in mel.iter_mut() {
        *v = (*v).max(1e-10).log10();
        max = max.max(*v);
    }
    let floor = max - 8.0;
    for v in mel.iter_mut() {
        *v = (*v).max(floor);
        *v = (*v + 4.0) / 4.0;
    }
    MelSpectrogram {
        n_mels,
        n_frames,
        data: mel,
    }
}

/// Stack `N` mels into `[N, n_mels, n_frames]` row-major.
pub fn stack_mels(mels: &[MelSpectrogram]) -> Vec<f32> {
    if mels.is_empty() {
        return Vec::new();
    }
    let n_mels = mels[0].n_mels;
    let n_frames = mels[0].n_frames;
    let plane = n_mels * n_frames;
    let mut out = vec![0f32; mels.len() * plane];
    for (i, m) in mels.iter().enumerate() {
        debug_assert_eq!(m.n_mels, n_mels);
        debug_assert_eq!(m.n_frames, n_frames);
        out[i * plane..(i + 1) * plane].copy_from_slice(&m.data);
    }
    out
}

struct StftMag {
    n_frames: usize,
    data: Vec<f32>,
}

fn stft_mag(pcm: &[f32], window: &[f32]) -> StftMag {
    let pad = N_FFT / 2;
    let mut padded = Vec::with_capacity(pcm.len() + 2 * pad);
    for i in (1..=pad).rev() {
        let j = pcm.len().saturating_sub(i);
        padded.push(pcm[j]);
    }
    padded.extend_from_slice(pcm);
    for i in 1..=pad {
        let j = pcm.len().saturating_sub(i);
        padded.push(pcm[j]);
    }
    let n_frames = if padded.len() >= N_FFT {
        1 + (padded.len() - N_FFT) / HOP_LENGTH
    } else {
        0
    };
    let mut data = vec![0f32; n_frames * N_FREQ];
    for fi in 0..n_frames {
        let start = fi * HOP_LENGTH;
        for bin in 0..N_FREQ {
            let mut re = 0f32;
            let mut im = 0f32;
            let omega = 2.0 * std::f32::consts::PI * bin as f32 / N_FFT as f32;
            for (t, w) in window.iter().enumerate() {
                let s = padded.get(start + t).copied().unwrap_or(0.0);
                let ang = omega * t as f32;
                re += s * w * ang.cos();
                im -= s * w * ang.sin();
            }
            data[fi * N_FREQ + bin] = re * re + im * im;
        }
    }
    StftMag { n_frames, data }
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / n as f32;
            (x.sin()).powi(2)
        })
        .collect()
}

fn hz_to_mel(hz: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - 0.0) / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / (200.0 / 3.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - 0.0) / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * ((logstep * (mel - min_log_mel)).exp())
    } else {
        mel * (200.0 / 3.0)
    }
}

fn mel_filterbank(sample_rate: f64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let fmax = sample_rate * 0.5;
    let n_freq = n_fft / 2 + 1;
    let fftfreqs: Vec<f64> = (0..n_freq)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();
    let n_mel_points = n_mels + 2;
    let mel_pts: Vec<f64> = (0..n_mel_points)
        .map(|i| {
            let mel = hz_to_mel(0.0)
                + (hz_to_mel(fmax) - hz_to_mel(0.0)) * i as f64 / (n_mel_points - 1) as f64;
            mel_to_hz(mel)
        })
        .collect();
    let mut fdiff = vec![0f64; n_mel_points - 1];
    for i in 0..fdiff.len() {
        fdiff[i] = mel_pts[i + 1] - mel_pts[i];
    }
    let mut weights = vec![0f32; n_mels * n_freq];
    for m in 0..n_mels {
        for k in 0..n_freq {
            let f = fftfreqs[k];
            let lower = (f - mel_pts[m]) / fdiff[m];
            let upper = (mel_pts[m + 2] - f) / fdiff[m + 1];
            let v = lower.min(upper).max(0.0) as f32;
            weights[m * n_freq + k] = v;
        }
        let enorm = 2.0 / (mel_pts[m + 2] - mel_pts[m]) as f32;
        for k in 0..n_freq {
            weights[m * n_freq + k] *= enorm;
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::N_FRAMES;

    #[test]
    fn mel_nonempty_for_tone() {
        let sr = SAMPLE_RATE;
        let pcm: Vec<f32> = (0..sr)
            .map(|i| (440.0 * 2.0 * std::f32::consts::PI * i as f32 / sr as f32).sin() * 0.2)
            .collect();
        let mel = pcm_to_log_mel(&pcm, 80, N_FRAMES);
        assert_eq!(mel.n_mels, 80);
        assert_eq!(mel.n_frames, N_FRAMES);
        let energy: f32 = mel.data.iter().map(|x| x.abs()).sum();
        assert!(energy > 0.0);
    }
}

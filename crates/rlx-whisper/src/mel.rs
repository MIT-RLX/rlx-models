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

use crate::audio::MelSpectrogram;
use rlx_spectral::WhisperLogMel;
#[cfg(any(test, feature = "export"))]
use rlx_spectral::{MelBank, Window};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
#[cfg(test)]
const N_FREQ: usize = N_FFT / 2 + 1;

/// Mel frames needed for `n_samples` of 16 kHz PCM (OpenAI hop 160, fft 400).
pub fn mel_frames_from_samples(n_samples: usize) -> usize {
    use crate::audio::{N_FRAMES, N_SAMPLES};
    let n = n_samples.min(N_SAMPLES);
    if n < N_FFT {
        return 1;
    }
    let n_stft = (n - N_FFT) / HOP_LENGTH + 1;
    n_stft.clamp(1, N_FRAMES)
}

/// Bucketed mel frame count for a PCM utterance (compile-cache friendly).
pub fn mel_geometry_frames_for_pcm(pcm: &[f32]) -> usize {
    use crate::audio::pad_or_trim_pcm;
    let pcm = pad_or_trim_pcm(pcm);
    bucket_mel_frames(mel_frames_from_samples(pcm.len()))
}

/// Inverse of [`crate::config::WhisperConfig::encoder_seq_len`] (conv stride-2).
///
/// Used to size align-hidden compile graphs from a sliced encoder sequence length.
pub fn mel_frames_for_enc_seq(enc_seq: usize) -> usize {
    enc_seq.saturating_mul(2).saturating_sub(1).max(1)
}

/// Round up mel length for compile-cache reuse (fewer encoder graphs).
pub fn bucket_mel_frames(n_frames: usize) -> usize {
    use crate::audio::N_FRAMES;
    let mut b = 8usize;
    while b < n_frames {
        b = b.saturating_mul(2);
    }
    b.clamp(8, N_FRAMES)
}

/// PCM sample count that yields at least `mel_frames` STFT columns (OpenAI hop 160, fft 400).
pub fn samples_for_mel_frames(mel_frames: usize) -> usize {
    if mel_frames <= 1 {
        return N_FFT;
    }
    (mel_frames - 1) * HOP_LENGTH + N_FFT
}

/// Pad mel spectrogram to `target` frames (zero-fill trailing columns).
pub fn pad_mel_to_frames(m: &MelSpectrogram, target: usize) -> MelSpectrogram {
    if m.n_frames >= target {
        return m.clone();
    }
    let n_mels = m.n_mels;
    let old_frames = m.n_frames;
    let mut data = vec![0f32; n_mels * target];
    for mi in 0..n_mels {
        for fi in 0..old_frames {
            data[mi * target + fi] = m.data[mi * old_frames + fi];
        }
    }
    MelSpectrogram {
        n_mels,
        n_frames: target,
        data,
    }
}

/// Build log-mel for one utterance: row-major `[1, n_mels, n_frames]` (padded/truncated to `n_frames`).
/// The spectral front-end is rlx-spectral's [`WhisperLogMel`] (f32).
pub fn pcm_to_log_mel(pcm: &[f32], n_mels: usize, n_frames: usize) -> MelSpectrogram {
    let fe = cached_front_end(n_mels);
    MelSpectrogram {
        n_mels,
        n_frames,
        data: fe.compute_frames(pcm, n_frames),
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

/// The front-end (FFT plan, window, sparse mel bank), built once per `n_mels`.
fn cached_front_end(n_mels: usize) -> Arc<WhisperLogMel<f32>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<WhisperLogMel<f32>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    guard
        .entry(n_mels)
        .or_insert_with(|| Arc::new(WhisperLogMel::new(n_mels)))
        .clone()
}

/// Whisper's Slaney mel filters at 16 kHz, `[n_mels, n_fft / 2 + 1]`: what
/// [`pcm_to_log_mel`] weighs each frame's power by.
pub fn mel_filters(n_mels: usize) -> Vec<f32> {
    cached_front_end(n_mels).filters()
}

/// Periodic Hann window (whisper's `torch.hann_window(400)`).
#[cfg(any(test, feature = "export"))]
pub(crate) fn hann_window(n: usize) -> Vec<f32> {
    Window::HannPeriodic.build(n)
}

/// Slaney mel filterbank `[n_mels, n_fft/2 + 1]`.
#[cfg(any(test, feature = "export"))]
pub(crate) fn mel_filterbank(sample_rate: f64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    MelBank::whisper(sample_rate, n_fft, n_mels).weights()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::N_FRAMES;
    use crate::audio::SAMPLE_RATE;

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

    /// Reference naive O(N^2) DFT (the pre-FFT implementation) for parity locking.
    fn stft_mag_naive(pcm: &[f32], window: &[f32]) -> (usize, Vec<f32>) {
        let pad = N_FFT / 2;
        let mut padded = Vec::with_capacity(pcm.len() + 2 * pad);
        let last = pcm.len().saturating_sub(1);
        for i in (1..=pad).rev() {
            padded.push(pcm[i.min(last)]);
        }
        padded.extend_from_slice(pcm);
        for i in 1..=pad {
            padded.push(pcm[last.saturating_sub(i)]);
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
        (n_frames, data)
    }

    #[test]
    fn fft_stft_matches_naive_dft() {
        let sr = SAMPLE_RATE;
        let pcm: Vec<f32> = (0..sr)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.3 * (440.0 * 2.0 * std::f32::consts::PI * t).sin()
                    + 0.1 * (1300.0 * 2.0 * std::f32::consts::PI * t).sin()
            })
            .collect();
        let window = hann_window(N_FFT);
        let st = rlx_spectral::Stft::new(
            window.clone(),
            HOP_LENGTH,
            N_FFT,
            rlx_spectral::Pad::CenterReflect,
            rlx_spectral::FramePrep::default(),
        );
        let mut fast_data = Vec::new();
        st.power_frames(&pcm, |_, p| fast_data.extend_from_slice(p));
        let (n_naive, naive) = stft_mag_naive(&pcm, &window);
        assert_eq!(st.n_frames(pcm.len()), n_naive);
        let peak = naive.iter().cloned().fold(0f32, f32::max).max(1e-12);
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (a, b) in fast_data.iter().zip(naive.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            // Relative error is only meaningful on non-null bins; near-null bins are
            // dominated by the naive DFT's own f32 cancellation noise (the FFT is more
            // accurate there) and are floored out by the downstream log-mel clamp.
            if *b > 1e-4 * peak {
                max_rel = max_rel.max(d / b.abs());
            }
        }
        assert!(
            max_abs < 1e-3 * peak,
            "stft max_abs diff too high: {max_abs}"
        );
        assert!(max_rel < 1e-3, "stft max_rel diff too high: {max_rel}");
    }
}

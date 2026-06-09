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

//! Log-mel frontend from power spectrum (Whisper-style, Tier D validation).

use crate::reference::{fft_real_batch, max_abs_error};
use anyhow::{Result, ensure};

fn hz_to_mel(hz: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / (200.0 / 3.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * ((logstep * (mel - min_log_mel)).exp())
    } else {
        mel * (200.0 / 3.0)
    }
}

/// Slaney-style mel filterbank `[n_mels, n_bins]`.
pub fn mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: f32) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let fmax = sample_rate as f64 * 0.5;
    let fftfreqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();
    let n_pts = n_mels + 2;
    let mel_pts: Vec<f64> = (0..n_pts)
        .map(|i| {
            let mel =
                hz_to_mel(0.0) + (hz_to_mel(fmax) - hz_to_mel(0.0)) * i as f64 / (n_pts - 1) as f64;
            mel_to_hz(mel)
        })
        .collect();
    let mut fb = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let left = mel_pts[m];
        let center = mel_pts[m + 1];
        let right = mel_pts[m + 2];
        for (k, &f) in fftfreqs.iter().enumerate() {
            let v = if f < left || f > right {
                0.0
            } else if f <= center {
                ((f - left) / (center - left).max(1e-8)) as f32
            } else {
                ((right - f) / (right - center).max(1e-8)) as f32
            };
            fb[m * n_bins + k] = v;
        }
    }
    fb
}

pub fn hann_window(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / n_fft as f32;
            (x.sin()).powi(2)
        })
        .collect()
}

/// One-sided power from interleaved complex spectrum.
pub fn spectrum_to_power(interleaved: &[f32], n_fft: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mut p = vec![0f32; n_bins];
    for k in 0..n_bins {
        let re = interleaved[k * 2];
        let im = interleaved[k * 2 + 1];
        p[k] = re * re + im * im;
    }
    p
}

pub fn power_to_log_mel(power: &[f32], filters: &[f32], n_mels: usize, n_bins: usize) -> Vec<f32> {
    let mut mel = vec![0f32; n_mels];
    for m in 0..n_mels {
        let mut acc = 0f32;
        for k in 0..n_bins {
            acc += filters[m * n_bins + k] * power[k];
        }
        mel[m] = (acc.max(1e-10)).log10();
    }
    let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = max - 8.0;
    for v in mel.iter_mut() {
        *v = (*v).max(floor);
        *v = (*v + 4.0) / 4.0;
    }
    mel
}

pub fn log_mel_from_spectrum_batch(
    spectrum: &[f32],
    filters: &[f32],
    batch: usize,
    n_fft: usize,
    n_mels: usize,
) -> Result<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    ensure!(spectrum.len() == batch * n_fft * 2);
    let mut out = vec![0f32; batch * n_mels];
    rlx_ir::audio::log_mel_interleaved_f32(
        spectrum, filters, batch, n_fft, n_bins, n_mels, &mut out,
    );
    Ok(out)
}

/// Log-mel from an already Hann-windowed batch (no extra window pass).
pub fn log_mel_from_windowed_batch(
    windowed: &[f32],
    batch: usize,
    n_fft: usize,
    n_mels: usize,
    sample_rate: f32,
) -> Result<Vec<f32>> {
    ensure!(windowed.len() == batch * n_fft);
    let spec = fft_real_batch(windowed, batch, n_fft)?;
    let filters = mel_filterbank(n_fft, n_mels, sample_rate);
    log_mel_from_spectrum_batch(&spec, &filters, batch, n_fft, n_mels)
}

pub fn ref_log_mel_batch(
    signal: &[f32],
    batch: usize,
    n_fft: usize,
    n_mels: usize,
    sample_rate: f32,
) -> Result<Vec<f32>> {
    let window = hann_window(n_fft);
    let mut windowed = signal.to_vec();
    for b in 0..batch {
        for i in 0..n_fft {
            windowed[b * n_fft + i] *= window[i];
        }
    }
    log_mel_from_windowed_batch(windowed.as_slice(), batch, n_fft, n_mels, sample_rate)
}

pub fn mel_max_err(a: &[f32], b: &[f32]) -> f32 {
    max_abs_error(a, b)
}

/// MSE loss gradient w.r.t. interleaved denoised spectrum (for gate backprop).
pub fn log_mel_loss_grad_wrt_spectrum(
    pred_mel: &[f32],
    target_mel: &[f32],
    spectrum: &[f32],
    filters: &[f32],
    batch: usize,
    n_fft: usize,
    n_mels: usize,
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mel_norm = (batch * n_mels) as f32;
    let mut dy = vec![0f32; batch * n_mels];
    for b in 0..batch {
        let mel_base = b * n_mels;
        for m in 0..n_mels {
            dy[mel_base + m] = 2.0 * (pred_mel[mel_base + m] - target_mel[mel_base + m]) / mel_norm;
        }
    }

    let mut block_spec = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let spec_base = b * n_fft * 2;
        let block_base = spec_base;
        for k in 0..n_fft {
            block_spec[block_base + k] = spectrum[spec_base + k * 2];
            block_spec[block_base + n_fft + k] = spectrum[spec_base + k * 2 + 1];
        }
    }
    let mut grad_block = vec![0f32; batch * n_fft * 2];
    rlx_ir::audio::log_mel_block_vjp(
        &block_spec,
        filters,
        &dy,
        batch,
        n_fft,
        n_bins,
        n_mels,
        &mut grad_block,
    );
    let mut grad = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let spec_base = b * n_fft * 2;
        let block_base = spec_base;
        for k in 0..n_fft {
            grad[spec_base + k * 2] = grad_block[block_base + k];
            grad[spec_base + k * 2 + 1] = grad_block[block_base + n_fft + k];
        }
    }
    grad
}

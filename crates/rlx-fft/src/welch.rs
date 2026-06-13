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

//! Welch PSD — windowed overlapping segments, FFT, averaged one-sided power.

use crate::butterfly::butterfly_forward_real_batch;
use crate::config::{FftLearnConfig, TransformDir};
use crate::reference::{fft_real_batch, max_abs_error};
use crate::rlx_fft::{compile_rlx_fft, rlx_fft_forward};
use anyhow::{Result, ensure};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Welch segment layout: `n_segments` Hann-windowed frames, 50% hop by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WelchParams {
    pub n_fft: usize,
    pub hop: usize,
    pub n_segments: usize,
}

impl WelchParams {
    pub fn for_n_fft(n_fft: usize) -> Self {
        Self {
            n_fft,
            hop: n_fft / 2,
            n_segments: 8,
        }
    }

    pub fn frame_len(self) -> usize {
        self.n_fft + (self.n_segments.saturating_sub(1)) * self.hop
    }

    pub fn n_bins(self) -> usize {
        self.n_fft / 2 + 1
    }

    /// When a longer batch buffer is shared across Welch layouts, take each row's prefix.
    pub fn truncate_batch(
        self,
        signal: &[f32],
        batch: usize,
        full_frame: usize,
    ) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.truncate_batch_into(signal, batch, full_frame, &mut out)?;
        Ok(out)
    }

    /// Like [`Self::truncate_batch`] but reuses `out` capacity (hot-path friendly).
    pub fn truncate_batch_into(
        self,
        signal: &[f32],
        batch: usize,
        full_frame: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let need = self.frame_len();
        ensure!(
            signal.len() == batch * full_frame,
            "welch signal len {} != batch*full_frame {}",
            signal.len(),
            batch * full_frame
        );
        out.clear();
        if need == full_frame {
            out.extend_from_slice(signal);
            return Ok(());
        }
        out.reserve(batch * need);
        for b in 0..batch {
            let base = b * full_frame;
            out.extend_from_slice(&signal[base..base + need]);
        }
        Ok(())
    }

    pub fn output_len(self, batch: usize) -> usize {
        batch * self.n_bins()
    }
}

#[cfg(test)]
mod truncate_tests {
    use super::*;

    #[test]
    fn truncate_batch_into_reuses_capacity() {
        let full = WelchParams::for_n_fft(256);
        let fast = WelchParams {
            n_fft: 256,
            hop: 128,
            n_segments: 2,
        };
        let batch = 4;
        let signal: Vec<f32> = (0..batch * full.frame_len()).map(|i| i as f32).collect();
        let mut buf = Vec::new();
        fast.truncate_batch_into(&signal, batch, full.frame_len(), &mut buf)
            .unwrap();
        assert_eq!(buf.len(), batch * fast.frame_len());
        let cap = buf.capacity();
        fast.truncate_batch_into(&signal, batch, full.frame_len(), &mut buf)
            .unwrap();
        assert_eq!(buf.capacity(), cap);
    }
}

static HANN_CACHE: LazyLock<Mutex<HashMap<usize, Vec<f32>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn hann_window_uncached(n_fft: usize) -> Vec<f32> {
    let mut w = vec![0f32; n_fft];
    if n_fft == 1 {
        w[0] = 1.0;
        return w;
    }
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = 0.5 * (1.0 - (std::f32::consts::TAU * i as f32 / (n_fft - 1) as f32).cos());
    }
    w
}

/// Cached Hann window (hot Welch / peaks paths).
pub fn hann_window(n_fft: usize) -> Vec<f32> {
    let mut cache = HANN_CACHE.lock().expect("hann cache");
    cache
        .entry(n_fft)
        .or_insert_with(|| hann_window_uncached(n_fft))
        .clone()
}

/// Accumulate scaled one-sided power into an existing PSD row.
pub fn accumulate_one_sided_power_row(
    row: &mut [f32],
    interleaved: &[f32],
    n_fft: usize,
    scale: f32,
) {
    let psd = one_sided_power(interleaved, n_fft);
    for (r, p) in row.iter_mut().zip(psd.iter()) {
        *r += *p * scale;
    }
}

/// Flat `[batch, frame_len]` → windowed segments `[batch * n_segments, n_fft]`.
pub fn welch_windowed_segments(
    signal: &[f32],
    batch: usize,
    params: WelchParams,
    window: &[f32],
) -> Result<Vec<f32>> {
    let frame = params.frame_len();
    ensure!(
        signal.len() == batch * frame,
        "welch signal len {} != batch*frame_len {}",
        signal.len(),
        batch * frame
    );
    let n = params.n_fft;
    let hop = params.hop;
    let n_seg = params.n_segments;
    let mut segs = vec![0f32; batch * n_seg * n];
    for b in 0..batch {
        let sig_base = b * frame;
        for s in 0..n_seg {
            let off = s * hop;
            let seg_base = (b * n_seg + s) * n;
            for i in 0..n {
                segs[seg_base + i] = signal[sig_base + off + i] * window[i];
            }
        }
    }
    Ok(segs)
}

/// One-sided power from interleaved complex spectrum `[n_fft, 2]`.
pub fn one_sided_power(interleaved: &[f32], n_fft: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mut psd = vec![0f32; n_bins];
    psd[0] = interleaved[0] * interleaved[0] + interleaved[1] * interleaved[1];
    for k in 1..n_bins.saturating_sub(1) {
        let re = interleaved[k * 2];
        let im = interleaved[k * 2 + 1];
        psd[k] = 2.0 * (re * re + im * im);
    }
    if n_bins > 1 {
        let k = n_bins - 1;
        psd[k] = interleaved[k * 2] * interleaved[k * 2]
            + interleaved[k * 2 + 1] * interleaved[k * 2 + 1];
    }
    psd
}

pub fn average_welch_psd(
    spectrum: &[f32],
    batch: usize,
    n_segments: usize,
    n_fft: usize,
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mut out = vec![0f32; batch * n_bins];
    for b in 0..batch {
        for s in 0..n_segments {
            let spec_base = (b * n_segments + s) * n_fft * 2;
            let psd = one_sided_power(&spectrum[spec_base..spec_base + n_fft * 2], n_fft);
            for k in 0..n_bins {
                out[b * n_bins + k] += psd[k];
            }
        }
        let inv = n_segments as f32;
        for k in 0..n_bins {
            out[b * n_bins + k] /= inv;
        }
    }
    out
}

pub fn welch_rustfft(signal: &[f32], batch: usize, params: WelchParams) -> Result<Vec<f32>> {
    let window = hann_window(params.n_fft);
    let segs = welch_windowed_segments(signal, batch, params, &window)?;
    let spec = fft_real_batch(&segs, batch * params.n_segments, params.n_fft)?;
    Ok(average_welch_psd(
        &spec,
        batch,
        params.n_segments,
        params.n_fft,
    ))
}

pub fn welch_butterfly(
    signal: &[f32],
    twiddles: &[f32],
    batch: usize,
    params: WelchParams,
) -> Result<Vec<f32>> {
    let window = hann_window(params.n_fft);
    let segs = welch_windowed_segments(signal, batch, params, &window)?;
    let spec =
        butterfly_forward_real_batch(&segs, twiddles, batch * params.n_segments, params.n_fft)?;
    Ok(average_welch_psd(
        &spec,
        batch,
        params.n_segments,
        params.n_fft,
    ))
}

pub fn welch_rlx_op_fft(
    exec: &mut CompiledGraph,
    signal: &[f32],
    batch: usize,
    params: WelchParams,
) -> Result<Vec<f32>> {
    let window = hann_window(params.n_fft);
    let segs = welch_windowed_segments(signal, batch, params, &window)?;
    let spec = rlx_fft_forward(exec, &segs, batch * params.n_segments, params.n_fft);
    Ok(average_welch_psd(
        &spec,
        batch,
        params.n_segments,
        params.n_fft,
    ))
}

pub fn compile_welch_rlx_fft(
    batch: usize,
    params: WelchParams,
    device: Device,
) -> Result<CompiledGraph> {
    let cfg = FftLearnConfig::new(params.n_fft, batch * params.n_segments)?;
    compile_rlx_fft(&cfg, TransformDir::Forward, device)
}

pub fn welch_max_error(a: &[f32], b: &[f32]) -> f32 {
    max_abs_error(a, b)
}

fn one_sided_power_grad(interleaved: &[f32], d_psd: &[f32], n_fft: usize, grad: &mut [f32]) {
    let n_bins = n_fft / 2 + 1;
    let base = 0;
    grad[base] += d_psd[0] * 2.0 * interleaved[0];
    grad[base + 1] += d_psd[0] * 2.0 * interleaved[1];
    for k in 1..n_bins.saturating_sub(1) {
        let re = interleaved[k * 2];
        let im = interleaved[k * 2 + 1];
        grad[k * 2] += d_psd[k] * 4.0 * re;
        grad[k * 2 + 1] += d_psd[k] * 4.0 * im;
    }
    if n_bins > 1 {
        let k = n_bins - 1;
        grad[k * 2] += d_psd[k] * 2.0 * interleaved[k * 2];
        grad[k * 2 + 1] += d_psd[k] * 2.0 * interleaved[k * 2 + 1];
    }
}

/// MSE gradient w.r.t. segment spectra for Welch gate training.
pub fn welch_loss_grad_wrt_spectrum(
    pred: &[f32],
    target: &[f32],
    spectrum: &[f32],
    batch: usize,
    n_segments: usize,
    n_fft: usize,
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mut grad = vec![0f32; spectrum.len()];
    let norm = (batch * n_bins) as f32;
    let inv_seg = 1.0 / n_segments as f32;

    for b in 0..batch {
        let mut d_psd = vec![0f32; n_bins];
        for k in 0..n_bins {
            d_psd[k] = 2.0 * (pred[b * n_bins + k] - target[b * n_bins + k]) / norm * inv_seg;
        }
        for s in 0..n_segments {
            let spec_base = (b * n_segments + s) * n_fft * 2;
            one_sided_power_grad(
                &spectrum[spec_base..spec_base + n_fft * 2],
                &d_psd,
                n_fft,
                &mut grad[spec_base..spec_base + n_fft * 2],
            );
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn welch_signal(rng: &mut impl Rng, batch: usize, params: WelchParams) -> Vec<f32> {
        let frame = params.frame_len();
        let mut out = vec![0f32; batch * frame];
        for v in &mut out {
            *v = rng.gen_range(-1.0..1.0);
        }
        out
    }

    #[test]
    fn welch_rustfft_butterfly_match() {
        let params = WelchParams::for_n_fft(64);
        let batch = 4;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let signal = welch_signal(&mut rng, batch, params);
        let tw = crate::twiddle::exact_twiddles(&FftLearnConfig::new(64, batch).unwrap());
        let ref_psd = welch_rustfft(&signal, batch, params).unwrap();
        let bf_psd = welch_butterfly(&signal, &tw, batch, params).unwrap();
        assert!(welch_max_error(&bf_psd, &ref_psd) < 1e-3);
    }

    #[test]
    fn welch_rlx_op_fft_cpu() {
        let params = WelchParams::for_n_fft(64);
        let batch = 2;
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let signal = welch_signal(&mut rng, batch, params);
        let ref_psd = welch_rustfft(&signal, batch, params).unwrap();
        let mut exec = compile_welch_rlx_fft(batch, params, Device::Cpu).unwrap();
        let rlx_psd = welch_rlx_op_fft(&mut exec, &signal, batch, params).unwrap();
        assert!(welch_max_error(&rlx_psd, &ref_psd) < 1e-3);
    }
}

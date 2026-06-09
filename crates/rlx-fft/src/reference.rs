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

//! Exact FFT reference via `rustfft`.

use anyhow::{Result, ensure};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

/// Row-major complex layout: `[batch, n_fft, 2]` (real, imag).
pub fn fft_real_batch(signal: &[f32], batch: usize, n_fft: usize) -> Result<Vec<f32>> {
    ensure!(
        signal.len() == batch * n_fft,
        "signal len {} != batch*n_fft {}",
        signal.len(),
        batch * n_fft
    );
    let mut complex = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        for i in 0..n_fft {
            complex[b * n_fft * 2 + i * 2] = signal[b * n_fft + i];
        }
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    fft_complex_batch(&complex, batch, n_fft, fft)
}

pub fn fft_complex_batch(
    signal_re_im: &[f32],
    batch: usize,
    n_fft: usize,
    fft: Arc<dyn Fft<f32>>,
) -> Result<Vec<f32>> {
    ensure!(
        signal_re_im.len() == batch * n_fft * 2,
        "complex signal len {} != batch*n_fft*2 {}",
        signal_re_im.len(),
        batch * n_fft * 2
    );
    let mut out = vec![0f32; batch * n_fft * 2];
    let mut scratch = vec![Complex::<f32>::default(); fft.get_inplace_scratch_len()];
    for b in 0..batch {
        let mut buf: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| {
                let base = b * n_fft * 2 + i * 2;
                Complex::new(signal_re_im[base], signal_re_im[base + 1])
            })
            .collect();
        fft.process_with_scratch(&mut buf, &mut scratch);
        for (i, c) in buf.into_iter().enumerate() {
            let base = b * n_fft * 2 + i * 2;
            out[base] = c.re;
            out[base + 1] = c.im;
        }
    }
    Ok(out)
}

pub fn make_fft_plan(n_fft: usize) -> Arc<dyn Fft<f32>> {
    FftPlanner::<f32>::new().plan_fft_forward(n_fft)
}

pub fn make_ifft_plan(n_fft: usize) -> Arc<dyn Fft<f32>> {
    FftPlanner::<f32>::new().plan_fft_inverse(n_fft)
}

/// Complex IFFT via `rustfft` (unnormalized — `ifft(fft(x)) == n_fft * x`).
pub fn ifft_complex_batch(spectrum_re_im: &[f32], batch: usize, n_fft: usize) -> Result<Vec<f32>> {
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    ifft_transform_batch(spectrum_re_im, batch, n_fft, ifft)
}

fn ifft_transform_batch(
    signal_re_im: &[f32],
    batch: usize,
    n_fft: usize,
    transform: Arc<dyn Fft<f32>>,
) -> Result<Vec<f32>> {
    ensure!(
        signal_re_im.len() == batch * n_fft * 2,
        "complex spectrum len {} != batch*n_fft*2 {}",
        signal_re_im.len(),
        batch * n_fft * 2
    );
    let mut out = vec![0f32; batch * n_fft * 2];
    let mut scratch = vec![Complex::<f32>::default(); transform.get_inplace_scratch_len()];
    for b in 0..batch {
        let mut buf: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| {
                let base = b * n_fft * 2 + i * 2;
                Complex::new(signal_re_im[base], signal_re_im[base + 1])
            })
            .collect();
        transform.process_with_scratch(&mut buf, &mut scratch);
        for (i, c) in buf.into_iter().enumerate() {
            let base = b * n_fft * 2 + i * 2;
            out[base] = c.re;
            out[base + 1] = c.im;
        }
    }
    Ok(out)
}

/// Roundtrip check helper: `ifft(fft(x))` scaled by `1/n_fft`.
pub fn roundtrip_scale(n_fft: usize) -> f32 {
    n_fft as f32
}

/// RLX `Op::Fft` block layout `[batch, n_fft*2]` (re plane then im plane) → interleaved.
pub fn block_to_interleaved(block: &[f32], batch: usize, n_fft: usize) -> Vec<f32> {
    let mut interleaved = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let base = b * n_fft * 2;
        for i in 0..n_fft {
            interleaved[base + i * 2] = block[base + i];
            interleaved[base + i * 2 + 1] = block[base + n_fft + i];
        }
    }
    interleaved
}

/// Block layout → interleaved with per-bin affine correction (`out = x * gain + bias`).
pub fn block_to_interleaved_correct(
    block: &[f32],
    batch: usize,
    n_fft: usize,
    gain: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let flat = n_fft * 2;
    let mut out = vec![0f32; batch * flat];
    for b in 0..batch {
        let base = b * flat;
        for i in 0..n_fft {
            let re = block[base + i];
            let im = block[base + n_fft + i];
            let gi = i * 2;
            out[base + gi] = re * gain[gi] + bias[gi];
            out[base + gi + 1] = im * gain[gi + 1] + bias[gi + 1];
        }
    }
    out
}

/// Max absolute error between two `[batch, n_fft, 2]` tensors.
pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Mean squared error.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    if n == 0.0 {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        / n
}

//! Welch peak extraction — top-K frequency spikes from PSD (fast path uses fewer segments).

use crate::reference::{fft_real_batch, max_abs_error};
use crate::welch::{
    WelchParams, accumulate_one_sided_power_row, hann_window, welch_windowed_segments,
};
use anyhow::Result;
use std::cmp::Ordering;

/// Default top-K when CLI / bench configs omit `--k` / `--peak-k`.
pub const DEFAULT_PEAK_K: usize = 16;

/// Welch layout + top-K spike output (default: 2 segments for speed vs 8 in full Welch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WelchPeakParams {
    pub welch: WelchParams,
    pub k: usize,
    /// Half-width (bins) for peak-aware training gradients.
    pub band_half_width: usize,
}

impl WelchPeakParams {
    /// Fast student path: 2 segments, top-K spikes.
    pub fn fast_for_n_fft(n_fft: usize, k: usize) -> Self {
        Self {
            welch: WelchParams {
                n_fft,
                hop: n_fft / 2,
                n_segments: 2,
            },
            k: k.max(1),
            band_half_width: 3,
        }
    }

    /// Ultra-fast path: 1 segment (lowest latency, noisier peaks).
    pub fn ultra_fast_for_n_fft(n_fft: usize, k: usize) -> Self {
        Self {
            welch: WelchParams {
                n_fft,
                hop: n_fft / 2,
                n_segments: 1,
            },
            k: k.max(1),
            band_half_width: 3,
        }
    }

    /// Reference teacher peaks from standard Welch (8 segments).
    pub fn reference_for_n_fft(n_fft: usize, k: usize) -> Self {
        Self {
            welch: WelchParams::for_n_fft(n_fft),
            k: k.max(1),
            band_half_width: 3,
        }
    }

    pub fn n_bins(self) -> usize {
        self.welch.n_bins()
    }

    pub fn output_len(self, batch: usize) -> usize {
        batch * self.k * 2
    }

    pub fn frame_len(self) -> usize {
        self.welch.frame_len()
    }
}

/// Reusable PSD buffer for streaming peak extraction (avoids per-call alloc).
#[derive(Debug, Clone, Default)]
pub struct WelchPeaksScratch {
    psd: Vec<f32>,
}

impl WelchPeaksScratch {
    pub fn new(batch: usize, n_bins: usize) -> Self {
        Self {
            psd: vec![0f32; batch * n_bins],
        }
    }

    pub fn ensure(&mut self, batch: usize, n_bins: usize) -> &mut [f32] {
        let need = batch * n_bins;
        if self.psd.len() < need {
            self.psd.resize(need, 0.0);
        }
        &mut self.psd[..need]
    }
}

/// One batch row PSD `[n_bins]` → top-K `(bin, power)` sorted by power descending.
pub fn topk_peaks_one(psd: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n_bins = psd.len();
    let k = k.min(n_bins).max(1);
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (bin, &power) in psd.iter().enumerate() {
        if top.len() < k {
            top.push((bin, power));
            if top.len() == k {
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            }
            continue;
        }
        if power <= top[k - 1].1 {
            continue;
        }
        top[k - 1] = (bin, power);
        let mut i = k - 1;
        while i > 0 && top[i].1 > top[i - 1].1 {
            top.swap(i, i - 1);
            i -= 1;
        }
    }
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    top
}

/// Pack peaks as `[batch, k, 2]` interleaved `(bin, power)` per batch item.
pub fn pack_peaks_batch(peaks_per_row: &[Vec<(usize, f32)>], k: usize) -> Vec<f32> {
    let batch = peaks_per_row.len();
    let mut out = vec![0f32; batch * k * 2];
    for (b, peaks) in peaks_per_row.iter().enumerate() {
        for (i, &(bin, power)) in peaks.iter().take(k).enumerate() {
            let base = (b * k + i) * 2;
            out[base] = bin as f32;
            out[base + 1] = power;
        }
    }
    out
}

/// Dense PSD `[batch, n_bins]` → packed top-K peaks.
pub fn peaks_from_psd_batch(psd: &[f32], batch: usize, n_bins: usize, k: usize) -> Vec<f32> {
    let mut rows = Vec::with_capacity(batch);
    for b in 0..batch {
        let base = b * n_bins;
        rows.push(topk_peaks_one(&psd[base..base + n_bins], k));
    }
    pack_peaks_batch(&rows, k)
}

/// Segment spectra → averaged PSD in `psd_scratch` → top-K (no extra PSD alloc).
pub fn peaks_from_segment_spectrum_streaming(
    spectrum: &[f32],
    batch: usize,
    params: WelchPeakParams,
    psd_scratch: &mut [f32],
) -> Vec<f32> {
    let n_bins = params.n_bins();
    let n_fft = params.welch.n_fft;
    let n_seg = params.welch.n_segments;
    let inv = 1.0 / n_seg as f32;
    psd_scratch.fill(0.0);
    for b in 0..batch {
        let row = &mut psd_scratch[b * n_bins..(b + 1) * n_bins];
        for s in 0..n_seg {
            let spec_base = (b * n_seg + s) * n_fft * 2;
            accumulate_one_sided_power_row(
                row,
                &spectrum[spec_base..spec_base + n_fft * 2],
                n_fft,
                inv,
            );
        }
    }
    peaks_from_psd_batch(psd_scratch, batch, n_bins, params.k)
}

pub fn welch_peaks_rustfft(
    signal: &[f32],
    batch: usize,
    params: WelchPeakParams,
) -> Result<Vec<f32>> {
    welch_peaks_rustfft_with_scratch(signal, batch, params, None)
}

/// Rustfft peaks with optional reusable PSD scratch (avoids alloc when provided).
pub fn welch_peaks_rustfft_with_scratch(
    signal: &[f32],
    batch: usize,
    params: WelchPeakParams,
    scratch: Option<&mut WelchPeaksScratch>,
) -> Result<Vec<f32>> {
    let window = hann_window(params.welch.n_fft);
    let segs = welch_windowed_segments(signal, batch, params.welch, &window)?;
    let spec = fft_real_batch(&segs, batch * params.welch.n_segments, params.welch.n_fft)?;
    if let Some(scratch) = scratch {
        let psd = scratch.ensure(batch, params.n_bins());
        Ok(peaks_from_segment_spectrum_streaming(
            &spec, batch, params, psd,
        ))
    } else {
        Ok(welch_peaks_from_segment_spectrum(&spec, batch, params))
    }
}

/// Peak match loss on packed `(bin, power)` tensors.
pub fn peak_match_loss(pred: &[f32], target: &[f32], batch: usize, k: usize) -> f32 {
    debug_assert_eq!(pred.len(), target.len());
    debug_assert_eq!(pred.len(), batch * k * 2);
    let mut s = 0f32;
    for i in 0..pred.len() {
        let d = pred[i] - target[i];
        s += d * d;
    }
    s / pred.len() as f32
}

pub fn peak_max_err(pred: &[f32], target: &[f32]) -> f32 {
    max_abs_error(pred, target)
}

/// Band mask `[batch, n_bins]`: 1.0 near reference peak bins (for sparse PSD gradients).
pub fn peak_band_mask(
    ref_packed: &[f32],
    batch: usize,
    n_bins: usize,
    k: usize,
    half_width: usize,
) -> Vec<f32> {
    let mut mask = vec![0f32; batch * n_bins];
    for b in 0..batch {
        for i in 0..k {
            let base = (b * k + i) * 2;
            let bin = ref_packed[base].round() as isize;
            if bin < 0 {
                continue;
            }
            let bin = bin as usize;
            let lo = bin.saturating_sub(half_width);
            let hi = (bin + half_width).min(n_bins.saturating_sub(1));
            for j in lo..=hi {
                mask[b * n_bins + j] = 1.0;
            }
        }
    }
    mask
}

/// MSE gradient w.r.t. segment spectrum, masked to bins near reference peaks only.
pub fn peak_loss_grad_wrt_spectrum(
    pred_psd: &[f32],
    ref_psd: &[f32],
    ref_packed: &[f32],
    batch: usize,
    n_bins: usize,
    k: usize,
    half_width: usize,
) -> Vec<f32> {
    let mask = peak_band_mask(ref_packed, batch, n_bins, k, half_width);
    let mut grad = vec![0f32; batch * n_bins];
    let norm = (batch * n_bins) as f32;
    for b in 0..batch {
        for j in 0..n_bins {
            let idx = b * n_bins + j;
            if mask[idx] > 0.0 {
                grad[idx] = 2.0 * (pred_psd[idx] - ref_psd[idx]) / norm;
            }
        }
    }
    grad
}

/// PSD from precomputed segment spectra (shared by learned / ternary paths).
pub fn welch_peaks_from_segment_spectrum(
    spectrum: &[f32],
    batch: usize,
    params: WelchPeakParams,
) -> Vec<f32> {
    let n_bins = params.n_bins();
    let mut scratch = vec![0f32; batch * n_bins];
    peaks_from_segment_spectrum_streaming(spectrum, batch, params, &mut scratch)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn topk_orders_by_power() {
        let psd = vec![0.1, 0.5, 0.2, 0.9, 0.3];
        let peaks = topk_peaks_one(&psd, 2);
        assert_eq!(peaks[0].0, 3);
        assert!((peaks[0].1 - 0.9).abs() < 1e-6);
        assert_eq!(peaks[1].0, 1);
    }

    #[test]
    fn topk_partial_matches_full_sort() {
        let psd: Vec<f32> = (0..129).map(|i| (i as f32 * 0.03).sin().abs()).collect();
        let partial = topk_peaks_one(&psd, 16);
        let mut order: Vec<usize> = (0..psd.len()).collect();
        order.sort_by(|&a, &b| psd[b].partial_cmp(&psd[a]).unwrap_or(Ordering::Equal));
        order.truncate(16);
        let full: Vec<(usize, f32)> = order.into_iter().map(|b| (b, psd[b])).collect();
        assert_eq!(partial, full);
    }

    #[test]
    fn welch_peaks_rustfft_matches_manual_topk() {
        let params = WelchPeakParams::fast_for_n_fft(128, 4);
        let batch = 2;
        let frame = params.frame_len();
        let signal: Vec<f32> = (0..batch * frame)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let psd = crate::welch::welch_rustfft(&signal, batch, params.welch).unwrap();
        let manual = peaks_from_psd_batch(&psd, batch, params.n_bins(), params.k);
        let direct = welch_peaks_rustfft(&signal, batch, params).unwrap();
        assert_eq!(manual, direct);
    }

    #[test]
    fn streaming_matches_dense_psd_path() {
        let params = WelchPeakParams::fast_for_n_fft(128, 8);
        let batch = 4;
        let frame = params.frame_len();
        let signal: Vec<f32> = (0..batch * frame)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let dense = welch_peaks_rustfft(&signal, batch, params).unwrap();
        let mut scratch = WelchPeaksScratch::new(batch, params.n_bins());
        let stream =
            welch_peaks_rustfft_with_scratch(&signal, batch, params, Some(&mut scratch)).unwrap();
        assert_eq!(dense, stream);
    }

    #[test]
    fn peak_band_mask_covers_neighbors() {
        let mut packed = vec![0f32; 2];
        packed[0] = 10.0;
        packed[1] = 1.0;
        let mask = peak_band_mask(&packed, 1, 32, 1, 2);
        for j in 8..=12 {
            assert_eq!(mask[j], 1.0);
        }
        assert_eq!(mask[0], 0.0);
    }
}

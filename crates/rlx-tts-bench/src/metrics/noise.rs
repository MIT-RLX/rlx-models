//! Output noise / quality metrics (always computed when `--noise`).

use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseMetrics {
    pub peak: f64,
    pub rms: f64,
    pub crest_db: f64,
    pub spectral_flatness: f64,
    /// Crude SNR vs peak-normalized energy (higher is cleaner).
    pub snr_db: f64,
}

pub fn noise_metrics(pcm: &[f32]) -> NoiseMetrics {
    let peak = pcm
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs() as f64)
        .fold(0.0, f64::max);
    let n = pcm.len().max(1) as f64;
    let rms = (pcm.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / n).sqrt();
    let crest_db = if rms > 1e-12 {
        20.0 * (peak / rms).log10()
    } else {
        0.0
    };
    let spectral_flatness = spectral_flatness(pcm);
    // Treat RMS energy relative to a peak-normalized unit signal as crude SNR.
    let snr_db = if peak > 1e-12 {
        let norm_rms = rms / peak;
        // Ideal peak-normalized sine has RMS ~0.707; quieter/noisier → lower score.
        20.0 * (norm_rms.max(1e-12)).log10() + 3.0
    } else {
        -80.0
    };
    NoiseMetrics {
        peak,
        rms,
        crest_db,
        spectral_flatness,
        snr_db,
    }
}

fn spectral_flatness(pcm: &[f32]) -> f64 {
    let n = pcm.len().min(4096);
    if n < 64 {
        return 0.0;
    }
    let mut buf: Vec<Complex<f32>> = pcm[..n]
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();
    // zero-pad to power of two
    let mut nfft = 64usize;
    while nfft < n {
        nfft *= 2;
    }
    buf.resize(nfft, Complex { re: 0.0, im: 0.0 });
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);
    fft.process(&mut buf);
    let mags: Vec<f64> = buf[..nfft / 2]
        .iter()
        .map(|c| (c.norm_sqr() as f64).max(1e-20))
        .collect();
    if mags.is_empty() {
        return 0.0;
    }
    let log_mean = mags.iter().map(|m| m.ln()).sum::<f64>() / mags.len() as f64;
    let arith = mags.iter().sum::<f64>() / mags.len() as f64;
    (log_mean.exp() / arith).clamp(0.0, 1.0)
}

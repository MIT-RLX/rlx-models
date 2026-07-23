
use rlx_fft::reference::fft_real_batch;

const COMPARE_SR: u32 = 16_000;
const N_FFT: usize = 512;
const HOP: usize = 160;
const N_MELS: usize = 40;

#[derive(Debug, Clone)]
pub struct SpectralMetrics {
    pub stft_cosine: f64,
    pub logmel_cosine: f64,
    pub band_low_ratio: f64,
    pub band_mid_ratio: f64,
    pub band_high_ratio: f64,
    pub duration_ratio: f64,
    pub peak_a: f32,
    pub peak_b: f32,
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt() + 1e-20)
}

pub fn resample_linear(pcm: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == 0 || to == 0 || pcm.is_empty() {
        return Vec::new();
    }
    if from == to {
        return pcm.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n = ((pcm.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let src = i as f64 * ratio;
        let j = src.floor() as usize;
        let frac = (src - j as f64) as f32;
        let a = pcm[j];
        let b = pcm.get(j + 1).copied().unwrap_or(a);
        out.push(a * (1.0 - frac) + b * frac);
    }
    out
}

pub fn peak(pcm: &[f32]) -> f32 {
    pcm.iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, f32::max)
}

pub fn spectral_vs_ref(pcm: &[f32], sr: u32, ref_pcm: &[f32], ref_sr: u32) -> SpectralMetrics {
    let a = resample_linear(pcm, sr, COMPARE_SR);
    let b = resample_linear(ref_pcm, ref_sr, COMPARE_SR);
    let n = a.len().min(b.len());
    let duration_ratio = if b.is_empty() {
        0.0
    } else {
        a.len() as f64 / b.len() as f64
    };
    if n < N_FFT {
        return SpectralMetrics {
            stft_cosine: 0.0,
            logmel_cosine: 0.0,
            band_low_ratio: 0.0,
            band_mid_ratio: 0.0,
            band_high_ratio: 0.0,
            duration_ratio,
            peak_a: peak(pcm),
            peak_b: peak(ref_pcm),
        };
    }
    let a = &a[..n];
    let b = &b[..n];
    let ma = stft_mag(a);
    let mb = stft_mag(b);
    let stft_cosine = cosine(&ma, &mb);
    let la = log_mel(&ma);
    let lb = log_mel(&mb);
    let logmel_cosine = cosine(&la, &lb);
    let (al, am, ah) = band_energy(&ma);
    let (bl, bm, bh) = band_energy(&mb);
    SpectralMetrics {
        stft_cosine,
        logmel_cosine,
        band_low_ratio: ratio(al, bl),
        band_mid_ratio: ratio(am, bm),
        band_high_ratio: ratio(ah, bh),
        duration_ratio,
        peak_a: peak(pcm),
        peak_b: peak(ref_pcm),
    }
}

fn ratio(a: f64, b: f64) -> f64 {
    if b.abs() < 1e-12 {
        return 0.0;
    }
    a / b
}

/// Hann-windowed STFT magnitudes via `rlx_fft::reference::fft_real_batch`.
fn stft_mag(pcm: &[f32]) -> Vec<f32> {
    let n_bins = N_FFT / 2 + 1;
    let mut frames = Vec::new();
    let mut i = 0;
    while i + N_FFT <= pcm.len() {
        frames.push(i);
        i += HOP;
    }
    if frames.is_empty() {
        return Vec::new();
    }
    let n_frames = frames.len();
    let mut block = vec![0f32; n_frames * N_FFT];
    for (fi, &start) in frames.iter().enumerate() {
        let dst = &mut block[fi * N_FFT..(fi + 1) * N_FFT];
        for k in 0..N_FFT {
            let w =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / (N_FFT as f32 - 1.0)).cos();
            dst[k] = pcm[start + k] * w;
        }
    }
    let Ok(spec) = fft_real_batch(&block, n_frames, N_FFT) else {
        return Vec::new();
    };
    let mut out = vec![0f32; n_frames * n_bins];
    for fi in 0..n_frames {
        let in_base = fi * N_FFT * 2;
        let out_base = fi * n_bins;
        for k in 0..n_bins {
            let re = spec[in_base + k * 2];
            let im = spec[in_base + k * 2 + 1];
            out[out_base + k] = (re * re + im * im).sqrt();
        }
    }
    out
}

fn log_mel(stft: &[f32]) -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    if stft.is_empty() || bins == 0 {
        return Vec::new();
    }
    let frames = stft.len() / bins;
    let mut out = Vec::with_capacity(frames * N_MELS);
    for f in 0..frames {
        let frame = &stft[f * bins..(f + 1) * bins];
        for m in 0..N_MELS {
            let start = m * bins / N_MELS;
            let end = ((m + 1) * bins / N_MELS).max(start + 1);
            let e: f32 =
                frame[start..end.min(frame.len())].iter().sum::<f32>() / (end - start) as f32;
            out.push((e + 1e-10).ln());
        }
    }
    out
}

fn band_energy(stft: &[f32]) -> (f64, f64, f64) {
    let bins = N_FFT / 2 + 1;
    if stft.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let frames = stft.len() / bins;
    let mut low = 0.0f64;
    let mut mid = 0.0f64;
    let mut high = 0.0f64;
    for f in 0..frames {
        let frame = &stft[f * bins..(f + 1) * bins];
        for (i, &v) in frame.iter().enumerate() {
            let e = (v as f64).powi(2);
            let third = bins / 3;
            if i < third {
                low += e;
            } else if i < 2 * third {
                mid += e;
            } else {
                high += e;
            }
        }
    }
    (low, mid, high)
}

/// WaveRNN μ-law expand + one-pole IIR post.
pub fn apply_wavernn_mulaw_iir(pcm: &mut [f32], alpha: f32) {
    if pcm.is_empty() {
        return;
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut prev = 0.0f32;
    let mut i = 0;
    while i + 1 < pcm.len() {
        let c0 = (((pcm[i] + 1.0) * 255.0 / 2.0).round() as i32).clamp(0, 255) as f32;
        let c1 = (((pcm[i + 1] + 1.0) * 255.0 / 2.0).round() as i32).clamp(0, 255) as f32;
        let o0 = (prev * alpha + mulaw_expand_class(c0)).clamp(-1.0, 1.0);
        let o1 = (o0 * alpha + mulaw_expand_class(c1)).clamp(-1.0, 1.0);
        pcm[i] = o0;
        pcm[i + 1] = o1;
        prev = o1;
        i += 2;
    }
}

#[inline]
fn mulaw_expand_class(class: f32) -> f32 {
    let x = (class / 255.0) * 2.0 - 1.0;
    let sign = if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        0.0
    };
    sign * ((8.0 * x.abs()).exp2() - 1.0) * (1.0 / 255.0)
}

pub fn apply_output_volume(
    pcm: &mut [f32],
    global: f32,
    peak_ratio: f32,
    smoothing_window: usize,
) {
    if pcm.is_empty() {
        return;
    }
    for s in pcm.iter_mut() {
        *s *= global;
    }
    let peak_now = peak(pcm).max(1e-8);
    let target = peak_ratio.clamp(0.05, 1.0);
    if peak_now > target {
        let scale = target / peak_now;
        // Smooth gain toward `scale` over `smoothing_window` samples at the head,
        let win = smoothing_window.max(1).min(pcm.len());
        for (i, s) in pcm.iter_mut().enumerate() {
            let g = if i < win {
                let t = i as f32 / win as f32;
                1.0 + (scale - 1.0) * t
            } else {
                scale
            };
            *s *= g;
        }
    }
}

pub fn apply_leading_silence_ms(pcm: &mut Vec<f32>, silence_ms: u32, sample_rate_hz: u32) {
    if silence_ms == 0 || sample_rate_hz == 0 {
        return;
    }
    let n = (silence_ms as u64 * sample_rate_hz as u64 / 1000) as usize;
    if n == 0 {
        return;
    }
    let mut out = vec![0.0f32; n + pcm.len()];
    out[n..].copy_from_slice(pcm);
    *pcm = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_volume_scales_and_limits() {
        let mut pcm = vec![0.5f32; 200];
        apply_output_volume(&mut pcm, 0.8, 0.7, 120);
        assert!((pcm[0] - 0.4).abs() < 1e-5);
        assert!(peak(&pcm) <= 0.7 + 1e-4);
    }

    #[test]
    fn leading_silence_prepends_zeros() {
        let mut pcm = vec![1.0f32, 2.0];
        apply_leading_silence_ms(&mut pcm, 50, 24_000);
        assert_eq!(pcm.len(), 2 + 24_000 / 20);
        assert!(pcm[..1200].iter().all(|&x| x == 0.0));
        assert_eq!(pcm[1200], 1.0);
        assert_eq!(pcm[1201], 2.0);
    }

    #[test]
    fn mulaw_iir_roundtrip_classes() {
        // DualSoftmax lattice for classes 128,128,128,128,128,127
        let classes = [128i32, 128, 128, 128, 128, 127];
        let mut pcm: Vec<f32> = classes
            .iter()
            .map(|&c| (c as f32) * (2.0 / 255.0) - 1.0)
            .collect();
        apply_wavernn_mulaw_iir(&mut pcm, 0.86);
        let i16: Vec<i16> = pcm
            .iter()
            .map(|&x| x.mul_add(32767.0, 0.0).trunc() as i16)
            .collect();
        assert_eq!(i16, vec![2, 5, 7, 9, 10, 6]);
    }
}

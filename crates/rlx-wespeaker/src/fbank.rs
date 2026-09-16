// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Kaldi-style 80-bin log-mel fbank @ 16 kHz (25 ms / 10 ms, povey window).
//!
//! Matches the preprocessing expected by WeSpeaker ResNet34 ONNX
//! (`fbank` `[T, 80]`). No dither (speaker embeddings prefer stable features).

const SAMPLE_RATE: u32 = 16_000;
const FRAME_SHIFT: usize = 160;
const FRAME_LENGTH: usize = 400;
const N_FFT: usize = 512;
const N_MELS: usize = 80;
const PREEMPH: f32 = 0.97;

/// Compute `[n_frames, 80]` log-mel filterbank from mono PCM in `[-1, 1]` @ 16 kHz.
pub fn log_mel_fbank(pcm: &[f32]) -> Vec<[f32; N_MELS]> {
    if pcm.len() < FRAME_LENGTH {
        return Vec::new();
    }
    let mut x: Vec<f32> = pcm.iter().map(|s| s * 32768.0).collect();
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    for v in &mut x {
        *v -= mean;
    }
    if x.len() > 1 {
        for i in (1..x.len()).rev() {
            x[i] -= PREEMPH * x[i - 1];
        }
    }
    let filters = mel_filterbank();
    let window: Vec<f32> = (0..FRAME_LENGTH)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / (FRAME_LENGTH as f32 - 1.0);
            let hann = 0.5 - 0.5 * t.cos();
            hann.powf(0.85)
        })
        .collect();
    let n_frames = 1 + (x.len() - FRAME_LENGTH) / FRAME_SHIFT;
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * FRAME_SHIFT;
        let mut frame = vec![0f32; N_FFT];
        for i in 0..FRAME_LENGTH {
            frame[i] = x[start + i] * window[i];
        }
        let power = rfft_power(&frame);
        let mut mel = [0f32; N_MELS];
        for (b, filt) in filters.iter().enumerate() {
            let mut e = 0f32;
            for (k, &w) in filt.iter().enumerate() {
                e += w * power[k];
            }
            mel[b] = e.max(1e-10).ln();
        }
        out.push(mel);
    }
    out
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

fn mel_filterbank() -> Vec<Vec<f32>> {
    let n_freqs = N_FFT / 2 + 1;
    let fmin = 20.0f32;
    let fmax = SAMPLE_RATE as f32 / 2.0;
    let mmin = hz_to_mel(fmin);
    let mmax = hz_to_mel(fmax);
    let mut mels = Vec::with_capacity(N_MELS + 2);
    for i in 0..N_MELS + 2 {
        mels.push(mmin + (mmax - mmin) * i as f32 / (N_MELS as f32 + 1.0));
    }
    let bins: Vec<usize> = mels
        .iter()
        .map(|&m| {
            ((N_FFT + 1) as f32 * mel_to_hz(m) / SAMPLE_RATE as f32)
                .floor()
                .clamp(0.0, (n_freqs - 1) as f32) as usize
        })
        .collect();
    let mut filters = vec![vec![0f32; n_freqs]; N_MELS];
    for m in 0..N_MELS {
        let l = bins[m];
        let c = bins[m + 1];
        let r = bins[m + 2];
        for j in l..c {
            if c != l {
                filters[m][j] = (j - l) as f32 / (c - l) as f32;
            }
        }
        for j in c..r {
            if r != c {
                filters[m][j] = (r - j) as f32 / (r - c) as f32;
            }
        }
    }
    filters
}

fn rfft_power(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let mut re = frame.to_vec();
    let mut im = vec![0f32; n];
    fft_inplace(&mut re, &mut im);
    let half = n / 2 + 1;
    let mut power = vec![0f32; half];
    for k in 0..half {
        power[k] = re[k] * re[k] + im[k] * im[k];
    }
    power
}

fn fft_inplace(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let (wr0, wi0) = (ang.cos(), ang.sin());
        for i in (0..n).step_by(len) {
            let mut wr = 1.0f32;
            let mut wi = 0.0f32;
            for k in 0..half {
                let u_re = re[i + k];
                let u_im = im[i + k];
                let v_re = re[i + k + half] * wr - im[i + k + half] * wi;
                let v_im = re[i + k + half] * wi + im[i + k + half] * wr;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + half] = u_re - v_re;
                im[i + k + half] = u_im - v_im;
                let nwr = wr * wr0 - wi * wi0;
                wi = wr * wi0 + wi * wr0;
                wr = nwr;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbank_shape_1_5s() {
        let pcm = vec![0.05f32; 16_000 + 8_000];
        let fb = log_mel_fbank(&pcm);
        assert!(fb.len() > 100);
        assert_eq!(fb[0].len(), 80);
    }
}

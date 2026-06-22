// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Host inverse STFT matching WavTokenizer's ISTFTHead (`padding="same"`):
// irfft per frame → window → overlap-add → divide by window-envelope → trim.

use std::f32::consts::PI;

/// `mag`/`phase`: `[n_freq, T]` row-major (n_freq = n_fft/2 + 1). Returns the
/// time-domain waveform of length `hop * T`.
pub fn istft_same(
    mag: &[f32],
    phase: &[f32],
    n_freq: usize,
    t: usize,
    window: &[f32],
    n_fft: usize,
    hop: usize,
) -> Vec<f32> {
    debug_assert_eq!(n_freq, n_fft / 2 + 1);
    let win = n_fft; // win_length == n_fft
    let pad = (win - hop) / 2;
    let output_size = (t - 1) * hop + win;
    let mut y = vec![0f32; output_size];
    let mut env = vec![0f32; output_size];
    let inv_n = 1.0 / n_fft as f32;
    let half = n_fft / 2;

    for ti in 0..t {
        let re = |f: usize| mag[f * t + ti] * phase[f * t + ti].cos();
        let im = |f: usize| mag[f * t + ti] * phase[f * t + ti].sin();
        for n in 0..n_fft {
            // irfft (norm="backward"): real inverse FFT of the hermitian spectrum.
            let mut acc = re(0) + re(half) * (PI * n as f32).cos();
            for k in 1..half {
                let ang = 2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                acc += 2.0 * (re(k) * ang.cos() - im(k) * ang.sin());
            }
            let v = acc * inv_n * window[n];
            let pos = ti * hop + n;
            y[pos] += v;
            env[pos] += window[n] * window[n];
        }
    }

    let mut out = vec![0f32; output_size - 2 * pad];
    for (i, o) in out.iter_mut().enumerate() {
        let e = env[pad + i];
        *o = if e > 1e-11 { y[pad + i] / e } else { 0.0 };
    }
    out
}

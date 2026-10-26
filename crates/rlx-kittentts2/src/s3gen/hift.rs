// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! HiFT: the S3Gen vocoder, 80-bin mel at 50 Hz → 24 kHz waveform.
//!
//! A neural-source-filter HiFi-GAN that predicts an iSTFT spectrum instead of
//! samples:
//!
//! 1. **f0** — five `k=3` convs + ELU, a linear head, `|·|` (graph);
//! 2. **source** — f0 upsampled ×480, nine sine harmonics, voiced/unvoiced
//!    noise, `tanh(Linear(9→1))`, then a 16-point STFT of that (host);
//! 3. **decode** — conv_pre, three upsampling stages (×8, ×5, ×3) each adding
//!    a downsampled copy of the source spectrum and averaging three Snake
//!    resblocks, conv_post to 18 channels (graph);
//! 4. **iSTFT** — `exp` magnitude (≤ 100) and `sin` phase, 16-point / hop 4
//!    overlap-add, clamp to ±0.99 (host).
//!
//! The hops between graph and host move a few hundred kB; the convolutions
//! are the whole cost and run on the device.

use anyhow::{Result, ensure};
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::Activation;

use super::nn::{Ctx, S3GenWeights, dims};
use crate::sampling::Rng;

pub const N_FFT: usize = 16;
pub const HOP: usize = 4;
const N_BINS: usize = N_FFT / 2 + 1;
/// Samples per mel frame (8 · 5 · 3 · 4).
pub const UPSAMPLE: usize = 480;
const UPS: [(usize, usize); 3] = [(8, 16), (5, 11), (3, 7)];
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const SOURCE_KERNELS: [usize; 3] = [7, 7, 11];
const DILATIONS: [usize; 3] = [1, 3, 5];
const HARMONICS: usize = 9;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
const SAMPLE_RATE: f32 = 24_000.0;
const AUDIO_LIMIT: f32 = 0.99;
const M: &str = "mel2wav";

/// `ConvRNNF0Predictor`: mel `[1, 80, L, 1]` → f0 `[L, 1]`.
pub fn build_f0(ctx: &mut Ctx, w: &S3GenWeights, mel: HirNodeId) -> Result<HirNodeId> {
    let mut h = mel;
    for i in [0, 2, 4, 6, 8] {
        h = ctx.conv1d(w, h, &format!("{M}.f0_predictor.condnet.{i}"), (1, 1), 1, 1)?;
        h = ctx.act(Activation::Elu, h);
    }
    let tc = ctx.nct_to_tc(h);
    let y = ctx.linear(w, tc, &format!("{M}.f0_predictor.classifier"))?;
    Ok(ctx.act(Activation::Abs, y))
}

/// `x + sin(αx)² / (α + 1e-9)` with per-channel `α` (`alpha_logscale=False`).
fn snake(ctx: &mut Ctx, w: &S3GenWeights, x: HirNodeId, key: &str) -> Result<HirNodeId> {
    let d = dims(ctx, x);
    let (c, t) = (d[1], d[2]);
    let alpha = w.get(key)?;
    ensure!(
        alpha.len() == c,
        "{key}: {} alphas for {c} channels",
        alpha.len()
    );
    let a = ctx.param(key, &[1, c, 1, 1], || alpha.to_vec());
    let ia = ctx.param(&format!("{key}^inv"), &[1, c, 1, 1], || {
        alpha.iter().map(|&a| 1.0 / (a + 1e-9)).collect()
    });
    let full = vec![1, c as i64, t as i64, 1];
    let a = ctx.g.expand_(a, full.clone());
    let ia = ctx.g.expand_(ia, full);
    let ax = ctx.g.mul(x, a);
    let s = ctx.act(Activation::Sin, ax);
    let s2 = ctx.g.mul(s, s);
    let s2 = ctx.g.mul(s2, ia);
    Ok(ctx.g.add(x, s2))
}

fn resblock(
    ctx: &mut Ctx,
    w: &S3GenWeights,
    mut x: HirNodeId,
    p: &str,
    k: usize,
) -> Result<HirNodeId> {
    for (i, &d) in DILATIONS.iter().enumerate() {
        let pad1 = (k * d - d) / 2;
        let xt = snake(ctx, w, x, &format!("{p}.activations1.{i}.alpha"))?;
        let xt = ctx.conv1d(w, xt, &format!("{p}.convs1.{i}"), (pad1, pad1), 1, d)?;
        let xt = snake(ctx, w, xt, &format!("{p}.activations2.{i}.alpha"))?;
        let pad2 = (k - 1) / 2;
        let xt = ctx.conv1d(w, xt, &format!("{p}.convs2.{i}"), (pad2, pad2), 1, 1)?;
        x = ctx.g.add(xt, x);
    }
    Ok(x)
}

/// `HiFTGenerator.decode` up to `conv_post`: mel `[1, 80, L, 1]` and the
/// source spectrum `[1, 18, 120L + 1, 1]` → `[1, 18, 120L + 1, 1]`.
pub fn build_decode(
    ctx: &mut Ctx,
    w: &S3GenWeights,
    mel: HirNodeId,
    s_stft: HirNodeId,
) -> Result<HirNodeId> {
    let mut x = ctx.conv1d(w, mel, &format!("{M}.conv_pre"), (3, 3), 1, 1)?;
    // Source branch strides: cumulative upsampling still ahead of each stage.
    let down = [15usize, 3, 1];
    for (i, &(u, k)) in UPS.iter().enumerate() {
        x = ctx.leaky_relu(x, 0.1);
        x = ctx.conv_transpose1d(w, x, &format!("{M}.ups.{i}"), u, (k - u) / 2)?;
        if i == UPS.len() - 1 {
            x = ctx.reflect_pad_left(x, 1);
        }
        let r = down[i];
        let si = if r == 1 {
            ctx.conv1d(w, s_stft, &format!("{M}.source_downs.{i}"), (0, 0), 1, 1)?
        } else {
            ctx.conv1d(
                w,
                s_stft,
                &format!("{M}.source_downs.{i}"),
                (r / 2, r / 2),
                r,
                1,
            )?
        };
        let si = resblock(
            ctx,
            w,
            si,
            &format!("{M}.source_resblocks.{i}"),
            SOURCE_KERNELS[i],
        )?;
        x = ctx.g.add(x, si);
        let mut xs: Option<HirNodeId> = None;
        for (j, &rk) in RESBLOCK_KERNELS.iter().enumerate() {
            let y = resblock(ctx, w, x, &format!("{M}.resblocks.{}", i * 3 + j), rk)?;
            xs = Some(match xs {
                None => y,
                Some(acc) => ctx.g.add(acc, y),
            });
        }
        let xs = xs.unwrap();
        let d = dims(ctx, xs);
        let third = ctx.scalar(1.0 / 3.0, &d);
        x = ctx.g.mul(xs, third);
    }
    let x = ctx.leaky_relu(x, 0.01);
    ctx.conv1d(w, x, &format!("{M}.conv_post"), (3, 3), 1, 1)
}

/// Randomness of the sine source. `phases[0]` is always 0 (the fundamental).
#[derive(Debug, Clone, Copy)]
pub enum SourceNoise<'a> {
    /// Zero phase offsets and no additive noise — deterministic, for parity.
    Off,
    /// Upstream's: uniform phases for the overtones, Gaussian noise.
    Sampled(&'a std::cell::RefCell<Rng>),
}

/// f0 (`[L]`, Hz) → the `tanh(Linear(9→1))` source waveform (`[480 L]`).
pub fn sine_source(w: &S3GenWeights, f0: &[f32], noise: SourceNoise<'_>) -> Result<Vec<f32>> {
    let lw = w.get(&format!("{M}.m_source.l_linear.weight"))?;
    let lb = w.get(&format!("{M}.m_source.l_linear.bias"))?[0];
    ensure!(lw.len() == HARMONICS);
    let mut phase = [0f32; HARMONICS];
    if let SourceNoise::Sampled(rng) = noise {
        let mut r = rng.borrow_mut();
        for p in phase.iter_mut().skip(1) {
            *p = ((r.uniform() * 2.0 - 1.0) * std::f64::consts::PI) as f32;
        }
    }
    let n = f0.len() * UPSAMPLE;
    let mut out = vec![0f32; n];
    // torch accumulates f0·(h+1)/sr in f32 before `% 1`; f64 here keeps the
    // phase from drifting over long clips while agreeing to ~1e-6 cycles.
    let mut acc = [0f64; HARMONICS];
    for (i, o) in out.iter_mut().enumerate() {
        let f = f0[i / UPSAMPLE];
        let voiced = f > VOICED_THRESHOLD;
        let mut s = lb;
        for h in 0..HARMONICS {
            acc[h] += (f * (h + 1) as f32 / SAMPLE_RATE) as f64;
            let theta = std::f64::consts::TAU * acc[h].rem_euclid(1.0);
            let mut v = if voiced {
                SINE_AMP * ((theta as f32) + phase[h]).sin()
            } else {
                0.0
            };
            if let SourceNoise::Sampled(rng) = noise {
                let amp = if voiced { NOISE_STD } else { SINE_AMP / 3.0 };
                v += amp * rng.borrow_mut().normal();
            }
            s += lw[h] * v;
        }
        *o = s.tanh();
    }
    Ok(out)
}

/// Periodic Hann window (`scipy.signal.get_window('hann', N, fftbins=True)`).
fn hann() -> [f32; N_FFT] {
    let mut w = [0f32; N_FFT];
    for (i, v) in w.iter_mut().enumerate() {
        *v = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N_FFT as f64).cos() as f32;
    }
    w
}

/// `torch.stft(x, 16, 4, 16, hann, center=True, reflect)` → `[18, F]`
/// (9 real rows then 9 imaginary), `F = len / 4 + 1`.
pub fn stft(x: &[f32]) -> Vec<f32> {
    let half = N_FFT / 2;
    let n = x.len();
    let reflect = |i: isize| -> f32 {
        let j = if i < 0 {
            -i
        } else if i as usize >= n {
            2 * (n as isize - 1) - i
        } else {
            i
        };
        x[j as usize]
    };
    let frames = n / HOP + 1;
    let win = hann();
    let mut out = vec![0f32; 2 * N_BINS * frames];
    for f in 0..frames {
        let start = (f * HOP) as isize - half as isize;
        let seg: Vec<f64> = (0..N_FFT)
            .map(|j| (reflect(start + j as isize) * win[j]) as f64)
            .collect();
        for k in 0..N_BINS {
            let (mut re, mut im) = (0f64, 0f64);
            for (j, &v) in seg.iter().enumerate() {
                let a = std::f64::consts::TAU * (k * j) as f64 / N_FFT as f64;
                re += v * a.cos();
                im -= v * a.sin();
            }
            out[k * frames + f] = re as f32;
            out[(N_BINS + k) * frames + f] = im as f32;
        }
    }
    out
}

/// conv_post output `[18, F]` → waveform: `exp` magnitude (≤ 100) and
/// `sin` phase, iSTFT, clamp to ±0.99.
pub fn istft_head(x: &[f32], frames: usize) -> Vec<f32> {
    let mut spec = vec![0f64; 2 * N_BINS * frames];
    for k in 0..N_BINS {
        for f in 0..frames {
            let mag = x[k * frames + f].exp().min(100.0) as f64;
            let ph = x[(N_BINS + k) * frames + f].sin() as f64;
            spec[k * frames + f] = mag * ph.cos();
            spec[(N_BINS + k) * frames + f] = mag * ph.sin();
        }
    }
    istft(&spec, frames)
        .into_iter()
        .map(|v| v.clamp(-AUDIO_LIMIT, AUDIO_LIMIT))
        .collect()
}

/// `torch.istft(·, 16, 4, 16, hann, center=True)` of a `[18, F]` (re rows,
/// then im rows) one-sided spectrum → `4 (F − 1)` samples.
pub fn istft(spec: &[f64], frames: usize) -> Vec<f32> {
    let win = hann();
    let half = N_FFT / 2;
    let full = N_FFT + HOP * (frames - 1);
    let mut y = vec![0f64; full];
    let mut env = vec![0f64; full];
    for f in 0..frames {
        let re = |k: usize| spec[k * frames + f];
        let im = |k: usize| spec[(N_BINS + k) * frames + f];
        // irfft: the DC and Nyquist imaginary parts are dropped.
        for j in 0..N_FFT {
            let mut s = re(0) + re(N_BINS - 1) * if j % 2 == 0 { 1.0 } else { -1.0 };
            for k in 1..N_BINS - 1 {
                let a = std::f64::consts::TAU * (k * j) as f64 / N_FFT as f64;
                s += 2.0 * (re(k) * a.cos() - im(k) * a.sin());
            }
            let wj = win[j] as f64;
            y[f * HOP + j] += s / N_FFT as f64 * wj;
            env[f * HOP + j] += wj * wj;
        }
    }
    (half..half + HOP * (frames - 1))
        .map(|i| (if env[i] > 1e-11 { y[i] / env[i] } else { y[i] }) as f32)
        .collect()
}

/// S3Gen's fade-in against reference spill-over: 20 ms of silence, then a
/// 20 ms raised-cosine ramp.
pub fn apply_trim_fade(wav: &mut [f32]) {
    let n = (SAMPLE_RATE as usize) / 50;
    for (i, v) in wav.iter_mut().take(2 * n).enumerate() {
        let g = if i < n {
            0.0
        } else {
            // torch.linspace(π, 0, n)[i − n]
            let a = std::f32::consts::PI * (1.0 - (i - n) as f32 / (n - 1) as f32);
            (a.cos() + 1.0) / 2.0
        };
        *v *= g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// iSTFT(STFT(x)) reconstructs `x` — catches window, hop, centering and
    /// sign slips in either half.
    #[test]
    fn stft_round_trips_through_istft() {
        let x: Vec<f32> = (0..400)
            .map(|i| 0.5 * (i as f32 * 0.37).sin() + 0.2 * (i as f32 * 0.051).cos())
            .collect();
        let s: Vec<f64> = stft(&x).into_iter().map(f64::from).collect();
        let y = istft(&s, x.len() / HOP + 1);
        assert_eq!(y.len(), x.len());
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn trim_fade_is_silent_then_ramps_to_one() {
        let mut w = vec![1f32; 2000];
        apply_trim_fade(&mut w);
        assert!(w[..480].iter().all(|&v| v == 0.0));
        assert!((w[959] - 1.0).abs() < 1e-6);
        assert_eq!(w[1500], 1.0);
    }
}

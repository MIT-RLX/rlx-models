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

//! Pure-Rust ChatterBox glue used by the native (`native`) runtime:
//! token layout, sampling, PRNG, resampling. No backend deps.

use std::collections::HashSet;

/// T3 prompt token layout (byte-identical across ort / native).
pub const START_TEXT: i64 = 255;
pub const STOP_TEXT: i64 = 0;
pub const START_SPEECH: i64 = 6561;
/// LM geometry (30-layer Llama, 16 heads × 64).
pub const N_LAYERS: usize = 30;
pub const N_HEADS: usize = 16;
pub const HEAD_DIM: usize = 64;
/// Speech-token vocab (the last slice of the LM logits).
pub const SPEECH_VOCAB: usize = 8194;
pub const SAMPLE_RATE: u32 = 24000;

/// Per-call synthesis options.
#[derive(Debug, Clone, Copy)]
pub struct SynthOpts {
    pub exaggeration: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub max_frames: usize,
    pub seed: u64,
    /// Deterministic decoding: repetition-penalized argmax (no temperature /
    /// top-k / top-p / multinomial). Robust to tiny cross-backend logit
    /// differences (the multinomial draw is what amplifies them) → the same
    /// token stream on CPU/Metal/MLX/wgpu/CoreML. moss/orpheus playbook.
    pub greedy: bool,
}

impl Default for SynthOpts {
    fn default() -> Self {
        Self {
            exaggeration: 0.5,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 1000,
            repetition_penalty: 1.2,
            max_frames: 1000,
            seed: 0,
            greedy: false,
        }
    }
}

/// SplitMix64 → uniform f32 in [0, 1).
pub struct Rng(u64);
impl Rng {
    pub fn new(s: u64) -> Self {
        Self(s.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box–Muller (`z ~ N(0,1)` for the CFM noise init).
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.uniform() as f32).max(1e-12);
        let u2 = self.uniform() as f32;
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

// ── S3Gen CFM flow-matching decoder (loop-based, re-exported graphs) ─────────
/// Classifier-free-guidance rate (`ConditionalCFM.inference_cfg_rate`).
pub const CFG_RATE: f32 = 0.7;
/// Number of Euler solver steps (`n_timesteps`).
pub const N_FLOW_STEPS: usize = 10;
/// HiFT ISTFT params (`istft_params`): tiny n_fft=16 / hop=4 transform.
pub const ISTFT_NFFT: usize = 16;
pub const ISTFT_HOP: usize = 4;

/// Cosine `t_span`: `1 - cos(linspace(0,1,n+1) * π/2)` — the reverse-diffusion
/// time grid used by `solve_euler`.
pub fn cosine_t_span(n: usize) -> Vec<f32> {
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            1.0 - (t * 0.5 * std::f32::consts::PI).cos()
        })
        .collect()
}

/// Periodic Hann window of length `n` (`get_window("hann", n, fftbins=True)`).
pub fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
        .collect()
}

/// Inverse STFT matching `torch.istft(n_fft, hop, win_length=n_fft, window,
/// center=True)`. `mag`/`phase` are row-major `[n_bins, n_frames]` with
/// `n_bins = n_fft/2 + 1`. Returns the center-trimmed real waveform.
pub fn istft(mag: &[f32], phase: &[f32], n_frames: usize, n_fft: usize, hop: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let win = hann_periodic(n_fft);
    let full_len = n_fft + hop * (n_frames - 1);
    let mut out = vec![0f32; full_len];
    let mut wsum = vec![0f32; full_len];
    // Per-frame inverse DFT (hermitian-symmetric real signal).
    for f in 0..n_frames {
        // Reconstruct the length-n_fft real frame via idft over the onesided spectrum.
        let mut frame = vec![0f32; n_fft];
        for (n, fr) in frame.iter_mut().enumerate() {
            let mut acc = 0f32;
            for k in 0..n_fft {
                // `mag`/`phase` are `[n_bins, n_frames]` (bin-major, the ONNX
                // `[B, n_bins, T]` layout) → index `bin * n_frames + frame`.
                // Hermitian: bins > n_bins-1 mirror the conjugate of the lower bins.
                let (re, im) = if k < n_bins {
                    let b = k * n_frames + f;
                    (mag[b] * phase_cos(phase[b]), mag[b] * phase[b].sin())
                } else {
                    let kk = n_fft - k;
                    let b = kk * n_frames + f;
                    (mag[b] * phase_cos(phase[b]), -mag[b] * phase[b].sin())
                };
                let ang = std::f32::consts::TAU * (k * n) as f32 / n_fft as f32;
                acc += re * ang.cos() - im * ang.sin();
            }
            *fr = acc / n_fft as f32;
        }
        let base = f * hop;
        for i in 0..n_fft {
            out[base + i] += frame[i] * win[i];
            wsum[base + i] += win[i] * win[i];
        }
    }
    for (o, w) in out.iter_mut().zip(&wsum) {
        if *w > 1e-8 {
            *o /= *w;
        }
    }
    // Drop the center padding (n_fft/2 each side).
    let pad = n_fft / 2;
    out[pad..full_len - pad].to_vec()
}

#[inline]
fn phase_cos(p: f32) -> f32 {
    p.cos()
}

/// EOS speech tokens.
pub fn is_eos(t: i64) -> bool {
    t == 2 || t == 6562
}

/// Sample a speech token: repetition penalty → temperature → top-k → top-p →
/// multinomial. `logits` is the `SPEECH_VOCAB`-wide tail of the LM output.
pub fn sample(logits: &[f32], seen: &HashSet<i64>, opts: &SynthOpts, rng: &mut Rng) -> i64 {
    let mut l: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    for &t in seen {
        if let Some(v) = l.get_mut(t as usize) {
            *v = if *v > 0.0 {
                *v / opts.repetition_penalty as f64
            } else {
                *v * opts.repetition_penalty as f64
            };
        }
    }
    // Greedy: repetition-penalized argmax — deterministic across backends.
    if opts.greedy {
        let mut best = 0usize;
        for i in 1..l.len() {
            if l[i] > l[best] {
                best = i;
            }
        }
        return best as i64;
    }
    for v in l.iter_mut() {
        *v /= opts.temperature as f64;
    }
    let mut idx: Vec<usize> = (0..l.len()).collect();
    idx.sort_unstable_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
    idx.truncate(opts.top_k.max(1));
    let max = l[idx[0]];
    let mut probs: Vec<f64> = idx.iter().map(|&i| (l[i] - max).exp()).collect();
    let sum: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let mut cum = 0.0;
    let mut k = probs.len();
    for (j, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= opts.top_p as f64 {
            k = j + 1;
            break;
        }
    }
    idx.truncate(k);
    probs.truncate(k);
    let s: f64 = probs.iter().sum();
    let r = rng.uniform() * s;
    let mut acc = 0.0;
    for (j, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return idx[j] as i64;
        }
    }
    idx[idx.len() - 1] as i64
}

/// Peak absolute amplitude.
pub fn peak_amplitude(a: &[f32]) -> f32 {
    a.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

/// Drop the HiFT / CFM startup click + leading hole, then fade in a few ms.
///
/// Native decode often emits a short impulse at t=0 followed by ~200 ms of
/// near-silence before the first phone. Detect that click→gap pattern, advance
/// through the silence to the first speech energy, keep a 15 ms pad, and apply
/// a 5 ms linear fade-in. Soft first phones are preserved (unlike a high
/// sustained-RMS gate that can skip them).
pub fn polish_onset(pcm: &[f32], sr: u32) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let peak = peak_amplitude(pcm);
    if peak < 1e-4 {
        return pcm.to_vec();
    }
    let win = ((sr as usize) / 100).max(1); // 10 ms
    let speech_thresh = (0.08 * peak).max(0.006);
    let silence_thresh = (0.025 * peak).max(0.002);
    let n_wins = pcm.len() / win;
    if n_wins == 0 {
        return pcm.to_vec();
    }
    let rms: Vec<f32> = (0..n_wins)
        .map(|i| {
            let s = &pcm[i * win..(i + 1) * win];
            (s.iter().map(|x| x * x).sum::<f32>() / win as f32).sqrt()
        })
        .collect();

    let mut i = 0usize;
    // Optional leading click: ≤40 ms of energy followed by ≥80 ms of silence.
    if rms[0] > speech_thresh {
        let mut j = 0usize;
        while j < 4 && j < rms.len() && rms[j] > silence_thresh {
            j += 1;
        }
        let mut sil = 0usize;
        while j + sil < rms.len() && rms[j + sil] <= silence_thresh {
            sil += 1;
        }
        if sil >= 8 {
            i = j;
        }
    }
    // Advance through the leading hole (or ordinary leading silence).
    while i < rms.len() && rms[i] <= silence_thresh {
        i += 1;
    }
    // If we never found speech, leave the buffer alone (aside from fade-in).
    if i >= rms.len() {
        i = 0;
    }

    let pad = ((sr as usize) * 15) / 1000;
    let start = (i * win).saturating_sub(pad);
    let mut out = pcm[start..].to_vec();
    let fade = ((sr as usize) * 5) / 1000;
    for i in 0..fade.min(out.len()) {
        out[i] *= i as f32 / fade as f32;
    }
    out
}

/// Linear resample.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = x[idx.min(x.len() - 1)];
            let b = x[(idx + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polish_onset_strips_click_and_gap() {
        let sr = 24_000u32;
        let mut pcm = vec![0.0f32; sr as usize]; // 1 s
        // Click: 15 ms impulse.
        for s in pcm.iter_mut().take((sr as usize * 15) / 1000) {
            *s = 0.3;
        }
        // Gap: silence until 250 ms, then speech.
        let speech = (sr as usize * 250) / 1000;
        for (i, s) in pcm.iter_mut().enumerate().skip(speech) {
            *s = 0.2 * ((i as f32) * 0.05).sin();
        }
        let out = polish_onset(&pcm, sr);
        assert!(out.len() < pcm.len(), "should trim leading click+gap");
        // Onset should land near the speech, not mid-utterance.
        assert!(
            out.len() > (sr as usize * 600) / 1000,
            "must not eat the speech body"
        );
        let head_rms = {
            let n = (sr as usize * 20) / 1000;
            (out[..n].iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt()
        };
        assert!(
            head_rms > 0.01,
            "speech should start promptly, rms={head_rms}"
        );
    }
}

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

//! What an audio model's runner does on the host around its graphs, as
//! graphs a package runs: [`vocos_istft`], the inverse STFT of a Vocos head
//! (`padding="same"`), the one WavTokenizer, XCodec2 and other Vocos-style
//! vocoders end in; [`istft_same`], the same from magnitude and phase;
//! [`istft_center`], `torch.istft(center=True)`; and
//! [`spectrogram`], a magnitude spectrogram; [`log_mel`], a log-mel spectrogram (`torchaudio`'s `MelSpectrogram`,
//! magnitude, then `clamp(1e-7).log()`); [`whisper_log_mel`], Whisper's
//! normalized log-mel; [`nemo_features`], NeMo's
//! `AudioToMelSpectrogramPreprocessor` — the DFTs matmuls by fixed bases.

use std::collections::HashMap;

use anyhow::{Result, bail};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, PadMode};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn scalar(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    )
}

/// The inverse DFT of each frame as two matrices, `[n_freq, n_fft]`, the
/// window and `1 / n_fft` folded in: a frame is `re · cos + im · sin`.
/// Held as params (`istft.cos`, `istft.sin`), not graph constants.
pub fn istft_bases(n_fft: usize, window: &[f32]) -> HashMap<String, Vec<f32>> {
    let half = n_fft / 2;
    let n_freq = half + 1;
    let (mut c, mut s) = (vec![0f32; n_freq * n_fft], vec![0f32; n_freq * n_fft]);
    for k in 0..n_freq {
        // DC and Nyquist once, the others twice (the hermitian half).
        let twice = if k == 0 || k == half { 1.0 } else { 2.0 };
        for n in 0..n_fft {
            let ang = 2.0 * std::f64::consts::PI * (k * n) as f64 / n_fft as f64;
            let w = f64::from(window[n]) / n_fft as f64;
            c[k * n_fft + n] = (twice * ang.cos() * w) as f32;
            // The imaginary part of DC and Nyquist does not enter.
            if k != 0 && k != half {
                s[k * n_fft + n] = (-2.0 * ang.sin() * w) as f32;
            }
        }
    }
    HashMap::from([("istft.cos".to_string(), c), ("istft.sin".to_string(), s)])
}

/// `a` `[t, r, hop]`, overlap-added: block `b` of the output sums frame
/// `b − j`'s chunk `j` → `[(t + r − 1) · hop]`.
fn overlap_add(g: &mut Graph, a: NodeId, t: usize, r: usize, hop: usize) -> NodeId {
    let mut sum = None;
    for j in 0..r {
        let chunk = g.narrow_(a, 1, j, 1);
        let chunk = g.reshape_(chunk, vec![t as i64, hop as i64]);
        let placed = g.pad_(chunk, vec![[j, r - 1 - j], [0, 0]], PadMode::Constant(0.0));
        sum = Some(match sum {
            Some(s) => g.add(s, placed),
            None => placed,
        });
    }
    let blocks = sum.expect("one chunk at least");
    g.reshape_(blocks, vec![((t + r - 1) * hop) as i64])
}

/// A Vocos head's output `spec` `[t, 2 · n_freq]` (each frame's log
/// magnitude, then its phase) → `audio` `[t · hop]`: magnitude `exp`
/// clipped at 100, each frame's inverse DFT windowed, overlap-added,
/// divided by the window's squared overlap (0 where that is below 1e-11),
/// and the `(n_fft − hop) / 2` padding trimmed — as WavTokenizer's
/// `head_to_wav` computes it. `n_fft` must be a multiple of `hop`. Its
/// params are [`istft_bases`].
pub fn vocos_istft(n_fft: usize, hop: usize, window: &[f32], t: usize) -> Result<Graph> {
    if !n_fft.is_multiple_of(hop) || window.len() != n_fft {
        bail!("an ISTFT of {n_fft} points needs a hop that divides it and a window as long");
    }
    let (half, r) = (n_fft / 2, n_fft / hop);
    let n_freq = half + 1;
    let mut g = Graph::new("istft");
    let spec = g.input("spec", f32s(&[t, 2 * n_freq]));
    let logm = g.narrow_(spec, 1, 0, n_freq);
    let phase = g.narrow_(spec, 1, n_freq, n_freq);
    let mag = g.add_node(
        Op::Activation(Activation::Exp),
        vec![logm],
        f32s(&[t, n_freq]),
    );
    let cap = scalar(&mut g, 1e2);
    let mag = g.binary(BinaryOp::Min, mag, cap, f32s(&[t, n_freq]));
    let cos = g.add_node(
        Op::Activation(Activation::Cos),
        vec![phase],
        f32s(&[t, n_freq]),
    );
    let sin = g.add_node(
        Op::Activation(Activation::Sin),
        vec![phase],
        f32s(&[t, n_freq]),
    );
    let re = g.mul(mag, cos);
    let im = g.mul(mag, sin);
    let bc = g.param("istft.cos", f32s(&[n_freq, n_fft]));
    let bs = g.param("istft.sin", f32s(&[n_freq, n_fft]));
    let a = g.mm(re, bc);
    let b = g.mm(im, bs);
    let frames = g.add(a, b);
    let frames = g.reshape_(frames, vec![t as i64, r as i64, hop as i64]);
    let y = overlap_add(&mut g, frames, t, r, hop);
    // The window's squared overlap, the same way.
    let w2: Vec<f32> = window.iter().map(|w| w * w).collect();
    let data = w2.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w2 = g.add_node(Op::Constant { data }, vec![], f32s(&[1, r, hop]));
    let w2 = g.add_node(
        Op::Expand {
            target_shape: vec![t as i64, r as i64, hop as i64],
        },
        vec![w2],
        f32s(&[t, r, hop]),
    );
    let env = overlap_add(&mut g, w2, t, r, hop);
    let len = (t + r - 1) * hop;
    let floor = scalar(&mut g, 1e-11);
    let safe = g.binary(BinaryOp::Max, env, floor, f32s(&[len]));
    let q = g.binary(BinaryOp::Div, y, safe, f32s(&[len]));
    let keep = g.add_node(
        Op::Compare(CmpOp::Gt),
        vec![env, floor],
        Shape::new(&[len], DType::Bool),
    );
    let keep = g.add_node(Op::Cast { to: DType::F32 }, vec![keep], f32s(&[len]));
    let out = g.binary(BinaryOp::Mul, q, keep, f32s(&[len]));
    let out = g.narrow_(out, 0, (n_fft - hop) / 2, t * hop);
    g.node_mut(out).name = Some("audio".into());
    g.set_outputs(vec![out]);
    Ok(g)
}

/// The DFT of each windowed frame as two matrices, `[n_fft, n_freq]`: a
/// frame's real part is `x · cos`, its imaginary part `x · sin`. Held as
/// params (`stft.cos`, `stft.sin`).
pub fn stft_bases(n_fft: usize, window: &[f32]) -> HashMap<String, Vec<f32>> {
    let n_freq = n_fft / 2 + 1;
    let (mut c, mut s) = (vec![0f32; n_fft * n_freq], vec![0f32; n_fft * n_freq]);
    for n in 0..n_fft {
        let w = f64::from(window[n]);
        for k in 0..n_freq {
            let ang = 2.0 * std::f64::consts::PI * ((k * n) % n_fft) as f64 / n_fft as f64;
            c[n * n_freq + k] = (w * ang.cos()) as f32;
            s[n * n_freq + k] = (-w * ang.sin()) as f32;
        }
    }
    HashMap::from([("stft.cos".to_string(), c), ("stft.sin".to_string(), s)])
}

/// `signal` (already padded as `center` pads it, `hop · (t + n_fft/hop −
/// 1)` samples: the `t` frames' and no more) → `log_mel` `[t, n_mels]`,
/// time-major: each frame's DFT magnitude (`power=1`) through the mel
/// filters (the param `mel.filters`, `[n_freq, n_mels]`), clamped at 1e-7
/// and logged. `n_fft` must be a multiple of `hop`. Its other params are
/// [`stft_bases`].
pub fn log_mel(n_fft: usize, hop: usize, n_mels: usize, t: usize) -> Result<Graph> {
    if !n_fft.is_multiple_of(hop) {
        bail!("a log-mel of {n_fft} points needs a hop that divides it");
    }
    let (r, n_freq) = (n_fft / hop, n_fft / 2 + 1);
    let mut g = Graph::new("log_mel");
    let x = g.input("signal", f32s(&[hop * (t + r - 1)]));
    let blocks = g.reshape_(x, vec![(t + r - 1) as i64, hop as i64]);
    let parts: Vec<NodeId> = (0..r).map(|j| g.narrow_(blocks, 0, j, t)).collect();
    let frames = g.concat_(parts, 1);
    let bc = g.param("stft.cos", f32s(&[n_fft, n_freq]));
    let bs = g.param("stft.sin", f32s(&[n_fft, n_freq]));
    let re = g.mm(frames, bc);
    let im = g.mm(frames, bs);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let p = g.add(re2, im2);
    let mag = g.sqrt(p);
    let fb = g.param("mel.filters", f32s(&[n_freq, n_mels]));
    let mel = g.mm(mag, fb);
    let floor = scalar(&mut g, 1e-7);
    let mel = g.binary(BinaryOp::Max, mel, floor, f32s(&[t, n_mels]));
    let out = g.log(mel);
    g.node_mut(out).name = Some("log_mel".into());
    g.set_outputs(vec![out]);
    Ok(g)
}

/// `signal` (reflect-padded by `n_fft / 2` each side, as `center=True`
/// pads, and cut at the `t`-th frame's end: `hop · (t + ⌈n_fft / hop⌉ − 1)`
/// samples) → `mel` `[n_mels, t]`, `WhisperFeatureExtractor`'s: each frame's
/// power through the mel filters (`mel.filters`, `[n_freq, n_mels]`),
/// `log10` over a 1e-10 floor, floored 8 below the whole mel's max, then
/// `(x + 4) / 4`. Its params are [`whisper_mel_params`].
pub fn whisper_log_mel(n_fft: usize, hop: usize, n_mels: usize, t: usize) -> Graph {
    let n_freq = n_fft / 2 + 1;
    let r = n_fft.div_ceil(hop);
    let mut g = Graph::new("mel");
    let x = g.input("signal", f32s(&[hop * (t + r - 1)]));
    let blocks = g.reshape_(x, vec![(t + r - 1) as i64, hop as i64]);
    let parts: Vec<NodeId> = (0..r).map(|j| g.narrow_(blocks, 0, j, t)).collect();
    let wide = g.concat_(parts, 1);
    let frames = g.narrow_(wide, 1, 0, n_fft);
    let bc = g.param("stft.cos", f32s(&[n_fft, n_freq]));
    let bs = g.param("stft.sin", f32s(&[n_fft, n_freq]));
    let re = g.mm(frames, bc);
    let im = g.mm(frames, bs);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let power = g.add(re2, im2);
    let fb = g.param("mel.filters", f32s(&[n_freq, n_mels]));
    let m = g.mm(power, fb);
    let m = g.clamp_(m, 1e-10, f32::MAX);
    let ln = g.log(m);
    let inv_ln10 = scalar(&mut g, (1.0 / std::f64::consts::LN_10) as f32);
    let log10 = g.mul(ln, inv_ln10);
    let max = g.reduce(
        log10,
        rlx_ir::op::ReduceOp::Max,
        vec![0, 1],
        true,
        f32s(&[1, 1]),
    );
    let eight = scalar(&mut g, 8.0);
    let floor = g.sub(max, eight);
    let floored = g.binary(BinaryOp::Max, log10, floor, f32s(&[t, n_mels]));
    let four = scalar(&mut g, 4.0);
    let shifted = g.add(floored, four);
    let quarter = scalar(&mut g, 0.25);
    let scaled = g.mul(shifted, quarter);
    let out = g.transpose_(scaled, vec![1, 0]);
    g.node_mut(out).name = Some("mel".into());
    g.set_outputs(vec![out]);
    g
}

/// [`whisper_log_mel`]'s params: the periodic-Hann-windowed DFT bases and
/// `filters` (`[n_mels, n_freq]`, as Whisper's `mel_filters`) transposed.
pub fn whisper_mel_params(n_fft: usize, filters: &[f32]) -> HashMap<String, Vec<f32>> {
    let window: Vec<f32> = (0..n_fft)
        .map(|i| {
            (std::f32::consts::PI * i as f32 / n_fft as f32)
                .sin()
                .powi(2)
        })
        .collect();
    let mut p = stft_bases(n_fft, &window);
    let n_freq = n_fft / 2 + 1;
    let n_mels = filters.len() / n_freq;
    let mut t = vec![0f32; n_freq * n_mels];
    for m in 0..n_mels {
        for k in 0..n_freq {
            t[k * n_mels + m] = filters[m * n_freq + k];
        }
    }
    p.insert("mel.filters".into(), t);
    p
}

/// `mag`, `phase` `[n_freq, t]` → `audio` `[(t − 1) · hop + n_fft − 2·pad]`
/// (`pad = (n_fft − hop) / 2`): each frame's inverse DFT windowed,
/// overlap-added, divided by the window's squared overlap (0 where that is
/// below 1e-11), `pad` trimmed from each end — `padding="same"`, as
/// XCodec2's and MioCodec's heads end (`rlx_xcodec::istft::istft_same`).
/// `n_fft` must be a multiple of `hop`. Its params are [`istft_bases`].
pub fn istft_same(n_fft: usize, hop: usize, window: &[f32], t: usize) -> Result<Graph> {
    if !n_fft.is_multiple_of(hop) || window.len() != n_fft || t == 0 {
        bail!("an ISTFT of {n_fft} points needs a hop that divides it and a window as long");
    }
    let (r, n_freq) = (n_fft / hop, n_fft / 2 + 1);
    let mut g = Graph::new("istft");
    let mag = g.input("mag", f32s(&[n_freq, t]));
    let phase = g.input("phase", f32s(&[n_freq, t]));
    let cos = g.add_node(
        Op::Activation(Activation::Cos),
        vec![phase],
        f32s(&[n_freq, t]),
    );
    let sin = g.add_node(
        Op::Activation(Activation::Sin),
        vec![phase],
        f32s(&[n_freq, t]),
    );
    let re = g.mul(mag, cos);
    let im = g.mul(mag, sin);
    let re = g.transpose_(re, vec![1, 0]);
    let im = g.transpose_(im, vec![1, 0]);
    let bc = g.param("istft.cos", f32s(&[n_freq, n_fft]));
    let bs = g.param("istft.sin", f32s(&[n_freq, n_fft]));
    let a = g.mm(re, bc);
    let b = g.mm(im, bs);
    let frames = g.add(a, b);
    let frames = g.reshape_(frames, vec![t as i64, r as i64, hop as i64]);
    let y = overlap_add(&mut g, frames, t, r, hop);
    let w2: Vec<f32> = window.iter().map(|w| w * w).collect();
    let data = w2.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w2 = g.add_node(Op::Constant { data }, vec![], f32s(&[1, r, hop]));
    let w2 = g.add_node(
        Op::Expand {
            target_shape: vec![t as i64, r as i64, hop as i64],
        },
        vec![w2],
        f32s(&[t, r, hop]),
    );
    let env = overlap_add(&mut g, w2, t, r, hop);
    let len = (t + r - 1) * hop;
    let floor = scalar(&mut g, 1e-11);
    let safe = g.binary(BinaryOp::Max, env, floor, f32s(&[len]));
    let q = g.binary(BinaryOp::Div, y, safe, f32s(&[len]));
    let keep = g.add_node(
        Op::Compare(CmpOp::Gt),
        vec![env, floor],
        Shape::new(&[len], DType::Bool),
    );
    let zero = scalar(&mut g, 0.0);
    let zero = g.add_node(
        Op::Expand {
            target_shape: vec![len as i64],
        },
        vec![zero],
        f32s(&[len]),
    );
    let out = g.add_node(Op::Where, vec![keep, q, zero], f32s(&[len]));
    let pad = (n_fft - hop) / 2;
    let out = g.narrow_(out, 0, pad, len - 2 * pad);
    g.node_mut(out).name = Some("audio".into());
    g.set_outputs(vec![out]);
    Ok(g)
}

/// `signal` (already padded, `hop · (t + n_fft/hop − 1)` samples) →
/// `spectrogram` `[t, n_freq]`, time-major: each frame's DFT magnitude,
/// `sqrt(re² + im² + eps)` (VITS' and OpenVoice's `spectrogram_torch`).
/// `n_fft` must be a multiple of `hop`. Its params are [`stft_bases`].
pub fn spectrogram(n_fft: usize, hop: usize, eps: f32, t: usize) -> Result<Graph> {
    if !n_fft.is_multiple_of(hop) {
        bail!("a spectrogram of {n_fft} points needs a hop that divides it");
    }
    let (r, n_freq) = (n_fft / hop, n_fft / 2 + 1);
    let mut g = Graph::new("spectrogram");
    let x = g.input("signal", f32s(&[hop * (t + r - 1)]));
    let blocks = g.reshape_(x, vec![(t + r - 1) as i64, hop as i64]);
    let parts: Vec<NodeId> = (0..r).map(|j| g.narrow_(blocks, 0, j, t)).collect();
    let frames = g.concat_(parts, 1);
    let bc = g.param("stft.cos", f32s(&[n_fft, n_freq]));
    let bs = g.param("stft.sin", f32s(&[n_fft, n_freq]));
    let re = g.mm(frames, bc);
    let im = g.mm(frames, bs);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let p = g.add(re2, im2);
    let e = scalar(&mut g, eps);
    let p = g.add(p, e);
    let out = g.sqrt(p);
    g.node_mut(out).name = Some("spectrogram".into());
    g.set_outputs(vec![out]);
    Ok(g)
}

/// `real`, `imag` `[n_freq, t]` → `audio` `[(t − 1) · hop]`, as
/// `torch.istft(n_fft, hop, window, center=True)`: each frame's inverse
/// DFT windowed, overlap-added, divided by the window's squared overlap
/// where that is above 1e-11, and `n_fft / 2` trimmed from each end.
/// `n_fft` must be a multiple of `hop`. Its params are [`istft_bases`].
pub fn istft_center(n_fft: usize, hop: usize, window: &[f32], t: usize) -> Result<Graph> {
    if !n_fft.is_multiple_of(hop) || window.len() != n_fft || t == 0 {
        bail!("an ISTFT of {n_fft} points needs a hop that divides it and a window as long");
    }
    let (r, n_freq) = (n_fft / hop, n_fft / 2 + 1);
    let mut g = Graph::new("istft");
    let real = g.input("real", f32s(&[n_freq, t]));
    let imag = g.input("imag", f32s(&[n_freq, t]));
    let re = g.transpose_(real, vec![1, 0]);
    let im = g.transpose_(imag, vec![1, 0]);
    let bc = g.param("istft.cos", f32s(&[n_freq, n_fft]));
    let bs = g.param("istft.sin", f32s(&[n_freq, n_fft]));
    let a = g.mm(re, bc);
    let b = g.mm(im, bs);
    let frames = g.add(a, b);
    let frames = g.reshape_(frames, vec![t as i64, r as i64, hop as i64]);
    let y = overlap_add(&mut g, frames, t, r, hop);
    let w2: Vec<f32> = window.iter().map(|w| w * w).collect();
    let data = w2.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w2 = g.add_node(Op::Constant { data }, vec![], f32s(&[1, r, hop]));
    let w2 = g.add_node(
        Op::Expand {
            target_shape: vec![t as i64, r as i64, hop as i64],
        },
        vec![w2],
        f32s(&[t, r, hop]),
    );
    let env = overlap_add(&mut g, w2, t, r, hop);
    let len = (t + r - 1) * hop;
    let floor = scalar(&mut g, 1e-11);
    let q = g.binary(BinaryOp::Div, y, env, f32s(&[len]));
    let keep = g.add_node(
        Op::Compare(CmpOp::Gt),
        vec![env, floor],
        Shape::new(&[len], DType::Bool),
    );
    let out = g.add_node(Op::Where, vec![keep, q, y], f32s(&[len]));
    let out = g.narrow_(out, 0, n_fft / 2, (t - 1) * hop);
    g.node_mut(out).name = Some("audio".into());
    g.set_outputs(vec![out]);
    Ok(g)
}

/// NeMo's frontend, as [`nemo_features`] builds it.
#[derive(Debug, Clone)]
pub struct NemoFrontend {
    pub n_fft: usize,
    pub hop: usize,
    pub n_mels: usize,
    /// The analysis window, centred into `n_fft` (zeros either side).
    pub window: Vec<f32>,
    /// `y[t] = x[t] − k·x[t−1]` before the STFT (NeMo: 0.97).
    pub preemphasis: Option<f32>,
    /// `normalize: per_feature`: each mel bin standardized over time.
    pub per_feature: bool,
}

/// How [`nemo_features_with`] pads the audio and masks its frames.
#[derive(Debug, Clone, Copy, Default)]
pub struct NemoPadding {
    /// `pad_mode="constant"`: `n_fft / 2` zeros either side (NeMo's
    /// featurizer and the Hugging Face port), not a reflection.
    pub zeros: bool,
    /// The last frame zeroed: a centred STFT counts `samples / hop + 1`
    /// frames, of which `samples / hop` are the audio's.
    pub mask_tail: bool,
}

/// The audio symbol of [`nemo_features`].
pub const SAMPLES: u32 = 0;

/// The longest audio [`nemo_features`] frames: its frame indices are the
/// param `audio.iota`, this long ([`audio_iota`]).
pub const MAX_FRAMES: usize = 1 << 17;

/// The values of the param `audio.iota`.
pub fn audio_iota() -> Vec<f32> {
    (0..MAX_FRAMES).map(|i| i as f32).collect()
}

fn nemo_features_at(fe: &NemoFrontend, n: usize, padding: NemoPadding) -> Graph {
    let (n_fft, hop, n_mels) = (fe.n_fft, fe.hop, fe.n_mels);
    let n_freq = n_fft / 2 + 1;
    let pad = n_fft / 2;
    let frames = 1 + n / hop;
    let mut g = Graph::new("features");
    let x = g.input("audio", f32s(&[n]));
    // Pre-emphasis, on the samples as given.
    let x = match fe.preemphasis {
        Some(k) => {
            let head = g.narrow_(x, 0, 0, 1);
            let cur = g.narrow_(x, 0, 1, n - 1);
            let prev = g.narrow_(x, 0, 0, n - 1);
            let k = scalar(&mut g, k);
            let kp = g.mul(prev, k);
            let d = g.sub(cur, kp);
            g.concat_(vec![head, d], 0)
        }
        None => x,
    };
    // `center=True`: `pad` samples either side, reflected (the edge
    // excluded) or zeros.
    let reverse = |g: &mut Graph, v: NodeId| {
        let idx: Vec<f32> = (0..pad).rev().map(|i| i as f32).collect();
        let data = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let idx = g.add_node(Op::Constant { data }, vec![], f32s(&[pad]));
        let idx = g.add_node(
            Op::Cast { to: DType::I32 },
            vec![idx],
            Shape::new(&[pad], DType::I32),
        );
        g.gather(v, idx, 0, f32s(&[pad]))
    };
    let padded = if padding.zeros {
        let z = scalar(&mut g, 0.0);
        let z = g.add_node(
            Op::Expand {
                target_shape: vec![pad as i64],
            },
            vec![z],
            f32s(&[pad]),
        );
        g.concat_(vec![z, x, z], 0)
    } else {
        let left = g.narrow_(x, 0, 1, pad);
        let left = reverse(&mut g, left);
        let right = g.narrow_(x, 0, n - 1 - pad, pad);
        let right = reverse(&mut g, right);
        g.concat_(vec![left, x, right], 0)
    };
    // Frame `f` is samples `f·hop … f·hop + n_fft` of the padded signal.
    let iota = g.param("audio.iota", f32s(&[MAX_FRAMES]));
    let f = g.narrow_(iota, 0, 0, frames);
    let f = g.reshape_(f, vec![frames as i64, 1]);
    let h = scalar(&mut g, hop as f32);
    let start = g.mul(f, h);
    let j = g.narrow_(iota, 0, 0, n_fft);
    let j = g.reshape_(j, vec![1, n_fft as i64]);
    let at = g.add(start, j);
    let at = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![at],
        Shape::new(&[frames, n_fft], DType::I32),
    );
    let windows = g.gather(padded, at, 0, f32s(&[frames, n_fft]));
    // Power spectrum (the window in the bases), the mel filters, the log.
    let bc = g.param("stft.cos", f32s(&[n_fft, n_freq]));
    let bs = g.param("stft.sin", f32s(&[n_fft, n_freq]));
    let re = g.mm(windows, bc);
    let im = g.mm(windows, bs);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let power = g.add(re2, im2);
    let fb = g.param("mel.filters", f32s(&[n_freq, n_mels]));
    let mel = g.mm(power, fb);
    let guard = scalar(&mut g, 5.960_464_5e-8);
    let mel = g.add(mel, guard);
    let mel = g.log(mel);
    // [frames, n_mels] → [n_mels, frames].
    let mut mel = g.transpose_(mel, vec![1, 0]);
    if fe.per_feature {
        // `(x − mean) / (std + 1e-5)` per bin, the std unbiased; with
        // `mask_tail` the statistics are the audio's frames' alone.
        let shape = f32s(&[n_mels, 1]);
        let stats = if padding.mask_tail {
            g.narrow_(mel, 1, 0, frames - 1)
        } else {
            mel
        };
        let mean = g.reduce(
            stats,
            rlx_ir::op::ReduceOp::Mean,
            vec![1],
            true,
            shape.clone(),
        );
        let ds = g.sub(stats, mean);
        let d = if padding.mask_tail {
            g.sub(mel, mean)
        } else {
            ds
        };
        let d2 = g.mul(ds, ds);
        let ss = g.reduce(d2, rlx_ir::op::ReduceOp::Sum, vec![1], true, shape.clone());
        // frames − 1, counted from the data: `sum(d·0 + 1) − 1`.
        let zero = scalar(&mut g, 0.0);
        let one = scalar(&mut g, 1.0);
        let ones = g.mul(ds, zero);
        let ones = g.add(ones, one);
        let count = g.reduce(ones, rlx_ir::op::ReduceOp::Sum, vec![1], true, shape);
        let dof = g.sub(count, one);
        let var = g.div(ss, dof);
        let std = g.sqrt(var);
        let eps = scalar(&mut g, 1e-5);
        let denom = g.add(std, eps);
        mel = g.div(d, denom);
    }
    if padding.mask_tail {
        let kept = g.narrow_(mel, 1, 0, frames - 1);
        let z = scalar(&mut g, 0.0);
        let z = g.reshape_(z, vec![1, 1]);
        let z = g.add_node(
            Op::Expand {
                target_shape: vec![n_mels as i64, 1],
            },
            vec![z],
            f32s(&[n_mels, 1]),
        );
        mel = g.concat_(vec![kept, z], 1);
    }
    let out = g.reshape_(mel, vec![1, n_mels as i64, frames as i64]);
    g.node_mut(out).name = Some("mel".into());
    g.set_outputs(vec![out]);
    g
}

/// `audio` `[samples]` (mono, at the model's rate) → `mel` `[1, n_mels,
/// frames]` (`frames = samples / hop + 1`), as NeMo's
/// `AudioToMelSpectrogramPreprocessor` makes it: pre-emphasis, a centred
/// STFT (reflect padding), the power spectrum through the filterbank
/// (`mel.filters`, `[n_freq, n_mels]`), `ln(x + 2^-24)`, and per-bin
/// standardization. Symbolic in [`SAMPLES`]. Its params: `stft.cos` /
/// `stft.sin` ([`stft_bases`] of [`NemoFrontend::window`]),
/// `mel.filters`, `audio.iota` ([`audio_iota`]).
pub fn nemo_features(fe: &NemoFrontend) -> Result<Graph> {
    nemo_features_with(fe, NemoPadding::default())
}

/// [`nemo_features`] padded and masked as `padding` says.
pub fn nemo_features_with(fe: &NemoFrontend, padding: NemoPadding) -> Result<Graph> {
    if fe.window.len() != fe.n_fft || fe.hop == 0 {
        bail!("the window must be n_fft long");
    }
    // The frame count is a quotient: fit it through sizes whose remainders
    // by the hop differ, the smallest and the largest among them, so the
    // offset is pinned (four middling remainders fit a quotient off by one).
    let sizes: Vec<usize> = [0, fe.hop - 1, fe.hop / 2, 1, fe.hop * 3 / 4, fe.hop / 4]
        .iter()
        .enumerate()
        .map(|(i, r)| (100 + 7 * i) * fe.hop + r)
        .collect();
    let graphs: Vec<Graph> = sizes
        .iter()
        .map(|&n| nemo_features_at(fe, n, padding))
        .collect();
    let refs: Vec<&Graph> = graphs.iter().collect();
    // The padded signal is `samples + n_fft` long: an offset past the
    // default (checked at a fifth length below).
    let g = rlx_ir::dynamic::generalize_many(
        &refs,
        &[(SAMPLES, sizes.clone())],
        Some(2 * fe.n_fft as i64),
    )
    .map_err(|e| anyhow::anyhow!("the frontend is not symbolic in the audio's length: {e}"))?;
    let check_len = 77 * fe.hop + fe.hop / 3;
    let check = nemo_features_at(fe, check_len, padding);
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(SAMPLES, check_len);
    let (got, want) = (
        format!("{}", rlx_ir::dynamic::specialize(&g, &bind)),
        format!("{check}"),
    );
    if got != want {
        let (a, b) = got
            .lines()
            .zip(want.lines())
            .find(|(a, b)| a != b)
            .unwrap_or_default();
        bail!(
            "the frontend generalized from six lengths is not the one built at another: \
             `{a}` where it builds `{b}`"
        );
    }
    Ok(g)
}

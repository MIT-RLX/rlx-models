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

//! NeuCodec decoder — eager CPU inference from safetensors weights.
//!
//! ## Architecture (XCodec2-based)
//!
//! ```text
//!  codes [T]  ──►  FSQ lookup  ──►  fc_post_a  ──►  VocosBackbone  ──►  ISTFTHead  ──►  audio
//! (int, 0..65535)  [T, 2048]      [T, 1024]        [T, 1024]                          [T*hop]
//! ```
//!
//! **VocosBackbone**: Conv1d(k=7) → 2×ResnetBlock → 12×TransformerBlock (RoPE) → 2×ResnetBlock → LayerNorm
//!
//! **ISTFTHead**: Linear(1024 → n_fft+2) → split mag/phase → ISTFT
//!
//! ## Setup (one-time)
//!
//! ```sh
//! python scripts/convert_weights.py   # download + extract decoder weights to safetensors
//! ```
//!
//! Weights are loaded at runtime from `NEUTTS_DECODER_PATH` (not bundled in this crate).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, s};
use rustfft::{FftPlanner, num_complex::Complex};
use safetensors::SafeTensors;

// ─── Public constants ─────────────────────────────────────────────────────────

/// Sample rate of the decoder output (24 kHz).
pub const SAMPLE_RATE: u32 = 24_000;

/// Sample rate the encoder expects as input (16 kHz).
pub const ENCODER_SAMPLE_RATE: u32 = 16_000;

/// Decoder audio samples per speech token — assuming 50 tokens/s at 24 kHz.
/// The actual value is detected from the weight shapes at load time.
pub const SAMPLES_PER_TOKEN: usize = 480;

/// Encoder audio samples consumed per speech token (16 000 / 50 = 320).
pub const ENCODER_SAMPLES_PER_TOKEN: usize = 320;

/// Default reference audio length for the encoder: 10 s × 16 000 Hz.
pub const ENCODER_DEFAULT_INPUT_SAMPLES: usize = 16_000 * 10;

// Feature probes live in `crate::features` (re-exported from `decoder`).

// ─── FSQ constants ────────────────────────────────────────────────────────────

/// FSQ levels for NeuCodec: 8 dimensions × 4 levels → 4^8 = 65 536 codes.
pub(crate) const FSQ_LEVELS: [i32; 8] = [4, 4, 4, 4, 4, 4, 4, 4];

/// Cumulative products of FSQ_LEVELS: used to decompose an integer code.
/// basis[j] = product(FSQ_LEVELS[0..j])
pub(crate) const FSQ_BASIS: [i32; 8] = [1, 4, 16, 64, 256, 1_024, 4_096, 16_384];

// ─── Tensor helpers ───────────────────────────────────────────────────────────

pub(crate) fn load_f32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("Missing weight: {name}"))?;
    let raw = view.data();
    use safetensors::tensor::Dtype;
    Ok(match view.dtype() {
        Dtype::F32 => {
            // Fast path: the bytes are already little-endian f32.  On LE
            // hosts (x86, ARM) we can reinterpret directly with no per-byte
            // work — essentially a single memcpy via the Vec allocation.
            assert!(
                raw.len() % 4 == 0,
                "F32 tensor byte length not divisible by 4"
            );
            let n = raw.len() / 4;
            let mut out = Vec::with_capacity(n);
            // SAFETY: raw is valid, aligned to u8 (no alignment requirement
            // for the source), and we write exactly `n` f32 values.
            #[cfg(target_endian = "little")]
            {
                // SAFETY: f32 and u8 have no padding/invalid-bit patterns for
                // this cast; we own `out` and set its length immediately after.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        raw.as_ptr(),
                        out.as_mut_ptr() as *mut u8,
                        raw.len(),
                    );
                    out.set_len(n);
                }
            }
            #[cfg(not(target_endian = "little"))]
            {
                out.extend(
                    raw.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                );
            }
            out
        }
        Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        dt => bail!("Tensor {name}: unsupported dtype {dt:?} (expected F32 or BF16)"),
    })
}

pub(crate) fn shape_of(st: &SafeTensors<'_>, name: &str) -> Result<Vec<usize>> {
    Ok(st
        .tensor(name)
        .with_context(|| format!("Missing weight: {name}"))?
        .shape()
        .to_vec())
}

pub(crate) fn as1d(data: Vec<f32>, n: usize) -> Array1<f32> {
    Array1::from_shape_vec(n, data).expect("1-D shape mismatch")
}

pub(crate) fn as2d(data: Vec<f32>, rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_vec((rows, cols), data).expect("2-D shape mismatch")
}

pub(crate) fn as3d(data: Vec<f32>, d0: usize, d1: usize, d2: usize) -> Array3<f32> {
    Array3::from_shape_vec((d0, d1, d2), data).expect("3-D shape mismatch")
}

// ─── Math primitives ──────────────────────────────────────────────────────────

/// Linear layer: `out = x @ w.T + b`
///
/// * `x`: \[T, in_dim\]
/// * `w`: \[out_dim, in_dim\]  (PyTorch row-major convention)
/// * `b`: \[out_dim\]  (optional)
/// * returns: \[T, out_dim\]
pub(crate) fn linear(
    x: ArrayView2<f32>,
    w: ArrayView2<f32>,
    b: Option<ArrayView1<f32>>,
) -> Array2<f32> {
    let mut out = x.dot(&w.t()); // [T, out_dim]
    if let Some(b) = b {
        out += &b;
    }
    out
}

/// Conv1d with same-length output (zero-padded).
///
/// * `x`: \[c_in, T\]
/// * `w`: \[c_out, c_in, k\]
/// * `b`: \[c_out\]  (optional)
/// * returns: \[c_out, T\]
fn conv1d(
    x: ArrayView2<f32>,
    w: ArrayView3<f32>,
    b: Option<ArrayView1<f32>>,
    pad: usize,
) -> Array2<f32> {
    let (c_in, t) = (x.shape()[0], x.shape()[1]);
    let (c_out, _, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);

    // im2col: build [T, c_in × k] column matrix
    let mut col = Array2::<f32>::zeros((t, c_in * k));
    for ti in 0..t {
        for ci in 0..c_in {
            for ki in 0..k {
                let src = ti + ki;
                if src >= pad && src < t + pad {
                    col[[ti, ci * k + ki]] = x[[ci, src - pad]];
                }
                // else zero-pad (already zeroed)
            }
        }
    }

    // weight: [c_out, c_in × k]
    let w2 = w
        .into_shape_with_order((c_out, c_in * k))
        .expect("conv1d reshape");

    // out_t = col @ w2.T  →  [T, c_out]  then transpose to [c_out, T]
    let out_t = col.dot(&w2.t());
    let mut out = out_t.t().to_owned(); // [c_out, T]

    if let Some(b) = b {
        // Broadcast b [c_out] over [c_out, T] — one ndarray op, no manual loop.
        use ndarray::Axis;
        out += &b.view().insert_axis(Axis(1));
    }
    out
}

/// GroupNorm: `affine=True`, over input \[C, T\].
/// Normalises over (group_size × T) elements per group.
///
/// Uses an iterator-based variance computation to avoid the temporary
/// array that `block.mapv(|v| (v - mean).powi(2))` would allocate.
fn group_norm(
    x: ArrayView2<f32>,
    n_groups: usize,
    w: ArrayView1<f32>,
    b: ArrayView1<f32>,
    eps: f32,
) -> Array2<f32> {
    let (c, t) = (x.shape()[0], x.shape()[1]);
    let group_size = c / n_groups;
    let n = (group_size * t) as f32;
    let mut out = Array2::<f32>::zeros((c, t));

    for g in 0..n_groups {
        let c_start = g * group_size;
        let c_end = c_start + group_size;
        let block = x.slice(s![c_start..c_end, ..]);

        // Mean — no temporary allocation
        let mean = block.iter().sum::<f32>() / n;
        // Variance — single pass, no temporary allocation
        let var = block
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / n;
        let inv_std = 1.0 / (var + eps).sqrt();

        for ci in c_start..c_end {
            let scale = inv_std * w[ci];
            let shift = b[ci];
            for ti in 0..t {
                out[[ci, ti]] = (x[[ci, ti]] - mean) * scale + shift;
            }
        }
    }
    out
}

/// LayerNorm over the last axis of \[T, C\].
///
/// Uses iterator sums to avoid the temporary arrays that `row.mapv(…).sum()`
/// would allocate for each of the T rows.
fn layer_norm(x: ArrayView2<f32>, w: ArrayView1<f32>, b: ArrayView1<f32>, eps: f32) -> Array2<f32> {
    let (t, c) = (x.shape()[0], x.shape()[1]);
    let c_f = c as f32;
    let mut out = Array2::<f32>::zeros((t, c));
    for ti in 0..t {
        let row = x.slice(s![ti, ..]);
        let mean = row.iter().sum::<f32>() / c_f;
        let var = row
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / c_f;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ci in 0..c {
            out[[ti, ci]] = (x[[ti, ci]] - mean) * inv_std * w[ci] + b[ci];
        }
    }
    out
}

/// RMSNorm over the last axis of \[T, C\].
///
/// Uses an iterator sum to avoid the temporary array that `row.mapv(|v|
/// v*v).sum()` would allocate for each of the T rows.
fn rms_norm(x: ArrayView2<f32>, w: ArrayView1<f32>, eps: f32) -> Array2<f32> {
    let (t, c) = (x.shape()[0], x.shape()[1]);
    let c_f = c as f32;
    let mut out = Array2::<f32>::zeros((t, c));
    for ti in 0..t {
        let row = x.slice(s![ti, ..]);
        let ms = row.iter().map(|&v| v * v).sum::<f32>() / c_f;
        let scale = 1.0 / (ms + eps).sqrt();
        for ci in 0..c {
            out[[ti, ci]] = x[[ti, ci]] * scale * w[ci];
        }
    }
    out
}

/// SiLU (swish): `x * σ(x)`.
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Row-wise softmax (in-place) over \[T\].
fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    x.iter_mut().for_each(|v| {
        *v = (*v - max).exp();
        sum += *v;
    });
    x.iter_mut().for_each(|v| *v /= sum);
}

// ─── FSQ decode ───────────────────────────────────────────────────────────────

/// Decode integer FSQ codes → continuous embeddings.
///
/// For each code (0..65535):
/// 1. Decompose into 8 base-4 digits using `FSQ_BASIS`.
/// 2. Scale each digit d ∈ {0,1,2,3} to {−1, −⅓, ⅓, 1} via `(d/1.5) - 1`.
/// 3. Apply the `project_out` linear layer (8 → 2048).
///
/// Returns \[T, fsq_out_dim\].
fn fsq_decode(
    codes: &[i32],
    proj_w: ArrayView2<f32>, // [fsq_out_dim, 8]
    proj_b: ArrayView1<f32>, // [fsq_out_dim]
) -> Array2<f32> {
    let t = codes.len();
    let _out_dim = proj_w.shape()[0];

    // Build [T, 8] matrix of scaled FSQ digits
    let mut digits = Array2::<f32>::zeros((t, FSQ_BASIS.len()));
    for (i, &code) in codes.iter().enumerate() {
        for (j, (&basis, &levels)) in FSQ_BASIS.iter().zip(FSQ_LEVELS.iter()).enumerate() {
            let d = (code / basis) % levels;
            // Scale from {0,1,…,L-1} to {-1, -1/3, 1/3, 1} for L=4
            // Formula: (d / ((L-1)/2)) - 1  =  (d / 1.5) - 1
            digits[[i, j]] = d as f32 / 1.5 - 1.0;
        }
    }

    // project_out: [T, 8] @ [8, out_dim] + [out_dim]
    linear(digits.view(), proj_w, Some(proj_b))
}

/// Encode continuous latents → integer FSQ codes (inverse of [`fsq_decode`]).
///
/// 1. Apply `project_in` linear (2048 → 8).
/// 2. Round each dimension to the nearest FSQ level digit ∈ {0,1,2,3}.
/// 3. Pack digits into a single code with [`FSQ_BASIS`].
///
/// * `latent`: \[T, fsq_in_dim\]  (typically 2048)
/// * `proj_w`: \[8, fsq_in_dim\]  (`generator.quantizer.project_in.weight`)
/// * `proj_b`: \[8\]
pub(crate) fn fsq_encode(
    latent: ArrayView2<f32>,
    proj_w: ArrayView2<f32>,
    proj_b: ArrayView1<f32>,
) -> Vec<i32> {
    let z = linear(latent, proj_w, Some(proj_b)); // [T, 8]
    let t = z.shape()[0];
    let mut codes = Vec::with_capacity(t);
    for ti in 0..t {
        let mut code = 0i32;
        for (j, &levels) in FSQ_LEVELS.iter().enumerate() {
            let scaled = z[[ti, j]];
            // Inverse of digits[[i,j]] = d/1.5 - 1
            let max_d = (levels - 1) as f32;
            let d = ((scaled + 1.0) * (max_d / 2.0)).round().clamp(0.0, max_d) as i32;
            code += d * FSQ_BASIS[j];
        }
        codes.push(code);
    }
    codes
}

// ─── RoPE sin/cos dispatch ────────────────────────────────────────────────────

/// Compute `(sin(x), cos(x))` for use in Rotary Positional Embedding.
///
/// The implementation is selected at compile time by the active feature flag:
///
/// | Feature     | Implementation                          | Max abs. error |
/// |-------------|-----------------------------------------|----------------|
/// | `fast`      | degree-7/6 Horner polynomial + f32 RR   | ~1 × 10⁻⁴     |
/// | `precise`   | `f32::sin_cos()` — correctly rounded    | ~1 × 10⁻⁷     |
/// | *(neither)* | same as `fast` (default)                | ~1 × 10⁻⁴     |
///
/// ### Fast-mode notes
///
/// The polynomial path avoids transcendental function calls entirely: sin and
/// cos are each evaluated with 6 fused multiply-adds (Horner's method).  On
/// platforms where hardware `sin`/`cos` instructions are slow or absent this
/// can be 6–12× faster per value.
///
/// Range reduction to \[−π, π\] uses a single `f32` round-multiply.  For large
/// angles — RoPE dimensions with position ≈ 2 047 and the highest frequency
/// (`inv_freq = 1.0`) give θ ≈ 2 047 rad — floating-point cancellation in the
/// reduction introduces O(2⁻²³ · |θ|) extra absolute error before the
/// polynomial.  At the worst case this is ≈ 2 × 10⁻⁴ rad, which is well
/// within perceptual threshold for speech synthesis.
///
/// Both this function and the Burn GPU path in `codec_burn::load_weights` use
/// the same dispatch, so precomputed RoPE tables and runtime CPU evaluations
/// are always produced by the same algorithm.
#[cfg(not(feature = "precise"))]
#[inline(always)]
pub(crate) fn rope_sin_cos(x: f32) -> (f32, f32) {
    use std::f32::consts::TAU;
    // Range-reduce to [−π, π] with a single round() multiply.
    let x = x - TAU * (x * (1.0 / TAU)).round();
    let x2 = x * x;
    // Horner-form degree-7 sin: x(1 + x²(−1/6 + x²(1/120 − x²/5040)))
    let s = x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 - x2 * (1.0 / 5040.0))));
    // Horner-form degree-6 cos: 1 + x²(−1/2 + x²(1/24 − x²/720))
    let c = 1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 - x2 * (1.0 / 720.0)));
    (s, c)
}

#[cfg(feature = "precise")]
#[inline(always)]
pub(crate) fn rope_sin_cos(x: f32) -> (f32, f32) {
    x.sin_cos()
}

// ─── Rotary positional embedding ──────────────────────────────────────────────

/// Apply split-half RoPE (torchtune convention) to `x` in-place.
///
/// * `x`: \[T, n_heads, head_dim\]
///
/// The outer loop order is `(position, freq_index, head)` so each
/// `sin_cos()` result is computed **once per (position, freq)** and reused
/// across all heads — previously it was computed once per (position, freq,
/// head), allocating a fresh `Vec<f32>` for each position.
fn apply_rope(x: &mut Array3<f32>) {
    let (t, n_heads, head_dim) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let half = head_dim / 2;

    // Inverse frequencies — only `half` f32 values, computed once.
    let inv_freqs: Vec<f32> = (0..half)
        .map(|i| 1.0_f32 / 10_000_f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    for p in 0..t {
        let p_f = p as f32;
        for i in 0..half {
            // Dispatch to rope_sin_cos(): polynomial (fast, default) or
            // stdlib sin_cos (precise feature).
            let (s, c) = rope_sin_cos(p_f * inv_freqs[i]);
            // Apply the same rotation to every head — no per-head recompute.
            for h in 0..n_heads {
                let x1 = x[[p, h, i]];
                let x2 = x[[p, h, i + half]];
                x[[p, h, i]] = x1 * c - x2 * s;
                x[[p, h, i + half]] = x1 * s + x2 * c;
            }
        }
    }
}

// ─── Transformer components ───────────────────────────────────────────────────

pub(crate) struct TransformerWeights {
    pub(crate) att_norm_w: Array1<f32>, // RMSNorm  [D]
    pub(crate) c_attn_w: Array2<f32>,   // Linear   [3D, D]  (no bias)
    pub(crate) c_proj_w: Array2<f32>,   // Linear   [D, D]   (no bias)
    pub(crate) ffn_norm_w: Array1<f32>, // RMSNorm  [D]
    pub(crate) fc1_w: Array2<f32>,      // Linear   [4D, D]  (no bias)
    pub(crate) fc2_w: Array2<f32>,      // Linear   [D, 4D]  (no bias)
}

/// Single Transformer block (RMSNorm → Attention → RMSNorm → MLP), residual.
///
/// * `x`: \[T, D\]  (modified in-place conceptually; returns new array)
fn transformer_block(x: ArrayView2<f32>, w: &TransformerWeights, n_heads: usize) -> Array2<f32> {
    let (t, d) = (x.shape()[0], x.shape()[1]);
    let head_dim = d / n_heads;

    // ── Attention sub-layer ───────────────────────────────────────────────────
    let normed = rms_norm(x, w.att_norm_w.view(), 1e-6);
    // qkv: [T, 3D]  (no bias)
    let qkv = linear(normed.view(), w.c_attn_w.view(), None);

    // Split into Q, K, V each [T, D]
    let q_flat = qkv.slice(s![.., 0..d]).to_owned();
    let k_flat = qkv.slice(s![.., d..2 * d]).to_owned();
    let v_flat = qkv.slice(s![.., 2 * d..]).to_owned();

    // Reshape to [T, n_heads, head_dim]
    let mut q = q_flat
        .into_shape_with_order((t, n_heads, head_dim))
        .expect("q reshape");
    let mut k = k_flat
        .into_shape_with_order((t, n_heads, head_dim))
        .expect("k reshape");
    let v = v_flat
        .into_shape_with_order((t, n_heads, head_dim))
        .expect("v reshape");

    apply_rope(&mut q);
    apply_rope(&mut k);

    // Scaled dot-product attention per head
    let scale = (head_dim as f32).sqrt().recip();
    // attn_out: [T, n_heads, head_dim]
    let mut attn_out = Array3::<f32>::zeros((t, n_heads, head_dim));

    for h in 0..n_heads {
        let qh = q.slice(s![.., h, ..]).to_owned(); // [T, head_dim]
        let kh = k.slice(s![.., h, ..]).to_owned();
        let vh = v.slice(s![.., h, ..]).to_owned();

        // scores = qh @ kh.T * scale  →  [T, T]
        let mut scores = qh.dot(&kh.t());
        scores.mapv_inplace(|v| v * scale);

        // softmax over last dim (per query row)
        for ti in 0..t {
            softmax_inplace(scores.slice_mut(s![ti, ..]).as_slice_mut().unwrap());
        }

        // weighted_v = scores @ vh  →  [T, head_dim]
        let wv = scores.dot(&vh);
        attn_out.slice_mut(s![.., h, ..]).assign(&wv);
    }

    // Reshape [T, n_heads, head_dim] → [T, D]
    let attn_flat = attn_out
        .into_shape_with_order((t, d))
        .expect("attn out reshape");

    // Project: c_proj (no bias)
    let attn_proj = linear(attn_flat.view(), w.c_proj_w.view(), None);

    // Residual
    let x_attn = &x + &attn_proj;

    // ── MLP sub-layer ─────────────────────────────────────────────────────────
    let normed2 = rms_norm(x_attn.view(), w.ffn_norm_w.view(), 1e-6);
    let h1 = linear(normed2.view(), w.fc1_w.view(), None);
    let h1_act = h1.mapv(silu);
    let h2 = linear(h1_act.view(), w.fc2_w.view(), None);

    &x_attn + &h2
}

// ─── ResnetBlock ─────────────────────────────────────────────────────────────

pub(crate) struct ResnetBlockWeights {
    pub(crate) norm1_w: Array1<f32>, // GroupNorm [C]
    pub(crate) norm1_b: Array1<f32>,
    pub(crate) conv1_w: Array3<f32>, // Conv1d [C, C, 3]
    pub(crate) conv1_b: Array1<f32>,
    pub(crate) norm2_w: Array1<f32>,
    pub(crate) norm2_b: Array1<f32>,
    pub(crate) conv2_w: Array3<f32>, // Conv1d [C, C, 3]
    pub(crate) conv2_b: Array1<f32>,
}

/// ResnetBlock: GroupNorm → swish → Conv1d(k=3) → GroupNorm → swish → Conv1d(k=3) + residual.
///
/// * `x`: \[C, T\]  (channels-first)
fn resnet_block(x: ArrayView2<f32>, w: &ResnetBlockWeights) -> Array2<f32> {
    // norm1 → swish → conv1
    let h = group_norm(x, 32, w.norm1_w.view(), w.norm1_b.view(), 1e-6);
    let h = h.mapv(silu);
    let h = conv1d(h.view(), w.conv1_w.view(), Some(w.conv1_b.view()), 1);

    // norm2 → swish → (dropout=no-op at inference) → conv2
    let h = group_norm(h.view(), 32, w.norm2_w.view(), w.norm2_b.view(), 1e-6);
    let h = h.mapv(silu);
    let h = conv1d(h.view(), w.conv2_w.view(), Some(w.conv2_b.view()), 1);

    // residual (in_channels == out_channels so no projection)
    &x + &h
}

// ─── ISTFT ────────────────────────────────────────────────────────────────────

/// Inverse STFT matching PyTorch `torch.istft(..., center=True)`.
///
/// * `mag`: \[n_fft/2+1, T\]  **log**-magnitudes (the model head outputs log-mag)
/// * `phase`: \[n_fft/2+1, T\] phase angles in radians
/// * `hop`: hop length (= n_fft / 4)
/// * `window`: Hann window \[n_fft\]
/// * returns: waveform of exactly `T × hop` samples
///
/// ### Two bugs this function previously had (now fixed)
///
/// 1. **Clamp-before-exp** — the original code did `mag.min(1e2).exp()`, which
///    caps the *log*-magnitude at 100 (meaning `exp(100) ≈ 2.7e43` for large
///    bins).  The correct Python behaviour is `exp(mag).clamp(max=1e2)` — clamp
///    the *linear* magnitude to 100.  Large log-magnitude bins (common for
///    loud/low-frequency speech) therefore blew up, drowning out high-frequency
///    content and causing muffled output.
///
/// 2. **Wrong center trim** — PyTorch's `center=True` removes `n_fft/2` samples
///    from the **start** of the OLA buffer and then takes exactly `T*hop`
///    samples.  The old code instead removed `(n_fft-hop)/2` from **both ends**,
///    which is a 240-sample temporal offset (at 24 kHz with hop=480) and
///    includes partially-overlapped edge frames with poor reconstruction quality.
///
/// `pub(crate)` so the Burn decoder in `codec_burn.rs` can call it after
/// pulling the head output back from the device.
pub(crate) fn istft_burn(
    mag: ArrayView2<f32>,
    phase: ArrayView2<f32>,
    hop: usize,
    window: &[f32],
    ifft: &dyn rustfft::Fft<f32>,
) -> Vec<f32> {
    let n_bins = mag.shape()[0]; // n_fft/2 + 1
    let n_frames = mag.shape()[1];
    let n_fft = (n_bins - 1) * 2;
    debug_assert_eq!(n_fft, window.len());
    debug_assert_eq!(hop, n_fft / 4);

    // Output buffer length before trimming
    let out_size = (n_frames - 1) * hop + n_fft;
    let mut y = vec![0.0f32; out_size];
    let mut env = vec![0.0f32; out_size];

    let mut buf = vec![Complex::<f32>::default(); n_fft];

    for ti in 0..n_frames {
        // Build the complex spectrum from log-magnitude + phase angle.
        //
        // FIX 1: exp() first, then clamp — matching PyTorch's
        //   `mag = torch.exp(mag).clamp(max=1e2)`
        // The old `.min(1e2).exp()` capped the log-magnitude at 100, which
        // effectively allowed linear magnitudes up to exp(100) ≈ 2.7e43.
        for fi in 0..n_bins {
            let m = mag[[fi, ti]].exp().min(1e2); // ← fixed: clamp linear mag
            let p = phase[[fi, ti]];
            buf[fi] = Complex::new(m * p.cos(), m * p.sin());
        }
        // Hermitian symmetry for real IFFT output
        for fi in 1..n_bins - 1 {
            buf[n_fft - fi] = buf[fi].conj();
        }

        // Inverse FFT (rustfft is unnormalized — we divide by n_fft below)
        ifft.process(&mut buf);

        // Normalize + apply synthesis window, then overlap-add
        let norm = n_fft as f32;
        let offset = ti * hop;
        for i in 0..n_fft {
            let sample = buf[i].re / norm * window[i];
            y[offset + i] += sample;
            env[offset + i] += window[i] * window[i];
        }
    }

    // Weighted overlap-add normalization
    for i in 0..out_size {
        if env[i] > 1e-11 {
            y[i] /= env[i];
        }
    }

    // FIX 2: match PyTorch center=True — trim n_fft/2 from the START only,
    // then take exactly T*hop samples.
    //
    // Old code: y[(n_fft-hop)/2 .. out_size-(n_fft-hop)/2]
    //   → 240-sample temporal offset + includes edge frames with 1-2 overlaps.
    // Correct:  y[n_fft/2 .. n_fft/2 + T*hop]
    //   → first fully-overlapped sample (≥4 frames) through end of signal.
    let start = n_fft / 2;
    let length = n_frames * hop;
    y[start..start + length].to_vec()
}

/// Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect()
}

// ─── Decoder weights ──────────────────────────────────────────────────────────

pub(crate) struct DecoderWeights {
    // FSQ
    pub(crate) fsq_proj_w: Array2<f32>, // [2048, 8]
    pub(crate) fsq_proj_b: Array1<f32>, // [2048]

    // fc_post_a: Linear(2048, 1024)
    pub(crate) fc_post_a_w: Array2<f32>, // [1024, 2048]
    pub(crate) fc_post_a_b: Array1<f32>, // [1024]

    // backbone.embed: Conv1d(1024, 1024, k=7, pad=3)
    pub(crate) embed_w: Array3<f32>, // [1024, 1024, 7]
    pub(crate) embed_b: Array1<f32>, // [1024]

    // backbone.prior_net (2 ResnetBlocks)
    pub(crate) prior_net: Vec<ResnetBlockWeights>,

    // backbone.transformers (N TransformerBlocks)
    pub(crate) transformers: Vec<TransformerWeights>,

    // backbone.final_layer_norm: LayerNorm [D]
    pub(crate) final_norm_w: Array1<f32>,
    pub(crate) final_norm_b: Array1<f32>,

    // backbone.post_net (2 ResnetBlocks)
    pub(crate) post_net: Vec<ResnetBlockWeights>,

    // head.out: Linear(D, n_fft+2)
    pub(crate) head_w: Array2<f32>, // [n_fft+2, 1024]
    pub(crate) head_b: Array1<f32>, // [n_fft+2]

    // Hann window
    pub(crate) window: Vec<f32>, // [n_fft]

    // Detected hyper-parameters
    pub(crate) hidden_dim: usize,
    pub(crate) hop_length: usize,
    pub(crate) depth: usize,
    pub(crate) n_heads: usize,

    // Cached IFFT plan — created once at load time so the plan cache is not
    // discarded between decode() calls.
    pub(crate) ifft_plan: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

fn load_resnet_block(st: &SafeTensors<'_>, prefix: &str, c: usize) -> Result<ResnetBlockWeights> {
    Ok(ResnetBlockWeights {
        norm1_w: as1d(load_f32(st, &format!("{prefix}.norm1.weight"))?, c),
        norm1_b: as1d(load_f32(st, &format!("{prefix}.norm1.bias"))?, c),
        conv1_w: as3d(load_f32(st, &format!("{prefix}.conv1.weight"))?, c, c, 3),
        conv1_b: as1d(load_f32(st, &format!("{prefix}.conv1.bias"))?, c),
        norm2_w: as1d(load_f32(st, &format!("{prefix}.norm2.weight"))?, c),
        norm2_b: as1d(load_f32(st, &format!("{prefix}.norm2.bias"))?, c),
        conv2_w: as3d(load_f32(st, &format!("{prefix}.conv2.weight"))?, c, c, 3),
        conv2_b: as1d(load_f32(st, &format!("{prefix}.conv2.bias"))?, c),
    })
}

fn load_transformer(st: &SafeTensors<'_>, prefix: &str, d: usize) -> Result<TransformerWeights> {
    Ok(TransformerWeights {
        att_norm_w: as1d(load_f32(st, &format!("{prefix}.att_norm.weight"))?, d),
        c_attn_w: as2d(
            load_f32(st, &format!("{prefix}.att.c_attn.weight"))?,
            3 * d,
            d,
        ),
        c_proj_w: as2d(load_f32(st, &format!("{prefix}.att.c_proj.weight"))?, d, d),
        ffn_norm_w: as1d(load_f32(st, &format!("{prefix}.ffn_norm.weight"))?, d),
        fc1_w: as2d(load_f32(st, &format!("{prefix}.mlp.fc1.weight"))?, 4 * d, d),
        fc2_w: as2d(load_f32(st, &format!("{prefix}.mlp.fc2.weight"))?, d, 4 * d),
    })
}

fn load_decoder_weights(
    st: &SafeTensors<'_>,
    user_meta: &Option<std::collections::HashMap<String, String>>,
) -> Result<DecoderWeights> {
    // ── Auto-detect hyper-parameters from weight shapes ───────────────────────
    let embed_shape = shape_of(st, "generator.backbone.embed.weight")?;
    let hidden_dim = embed_shape[0]; // c_out

    let head_shape = shape_of(st, "generator.head.out.weight")?;
    let out_dim = head_shape[0]; // n_fft + 2
    let hop_length = (out_dim - 2) / 4;

    // Count transformer blocks by probing for weight keys
    let depth = (0..64)
        .take_while(|&i| {
            st.tensor(&format!(
                "generator.backbone.transformers.{i}.att_norm.weight"
            ))
            .is_ok()
        })
        .count();

    if depth == 0 {
        bail!("No transformer blocks found — is the safetensors file correct?");
    }

    // n_heads: read from safetensors __metadata__ if present, otherwise default to 16
    let n_heads: usize = user_meta
        .as_ref()
        .and_then(|m| m.get("n_heads"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    // FSQ codebook projection
    // Try the nested key first (older exports), fall back to the flat key
    let fsq_proj_key = if st
        .tensor("generator.quantizer.fsqs.0.project_out.weight")
        .is_ok()
    {
        "generator.quantizer.fsqs.0.project_out.weight"
    } else {
        "generator.quantizer.project_out.weight"
    };
    let fsq_bias_key = if st
        .tensor("generator.quantizer.fsqs.0.project_out.bias")
        .is_ok()
    {
        "generator.quantizer.fsqs.0.project_out.bias"
    } else {
        "generator.quantizer.project_out.bias"
    };

    let fsq_shape = shape_of(st, fsq_proj_key)?;
    let fsq_out_dim = fsq_shape[0]; // 2048
    let fsq_in_dim = fsq_shape[1]; // 8

    let fsq_proj_w = as2d(load_f32(st, fsq_proj_key)?, fsq_out_dim, fsq_in_dim);
    let fsq_proj_b = as1d(load_f32(st, fsq_bias_key)?, fsq_out_dim);

    // fc_post_a: [1024, 2048]
    let fc_post_a_w = as2d(load_f32(st, "fc_post_a.weight")?, hidden_dim, fsq_out_dim);
    let fc_post_a_b = as1d(load_f32(st, "fc_post_a.bias")?, hidden_dim);

    // backbone.embed Conv1d
    let embed_k = embed_shape[2];
    let embed_w = as3d(
        load_f32(st, "generator.backbone.embed.weight")?,
        hidden_dim,
        hidden_dim,
        embed_k,
    );
    let embed_b = as1d(load_f32(st, "generator.backbone.embed.bias")?, hidden_dim);

    // prior_net (2 ResnetBlocks)
    let prior_net = (0..2)
        .map(|i| load_resnet_block(st, &format!("generator.backbone.prior_net.{i}"), hidden_dim))
        .collect::<Result<Vec<_>>>()?;

    // transformers
    let transformers = (0..depth)
        .map(|i| {
            load_transformer(
                st,
                &format!("generator.backbone.transformers.{i}"),
                hidden_dim,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    // final_layer_norm
    let final_norm_w = as1d(
        load_f32(st, "generator.backbone.final_layer_norm.weight")?,
        hidden_dim,
    );
    let final_norm_b = as1d(
        load_f32(st, "generator.backbone.final_layer_norm.bias")?,
        hidden_dim,
    );

    // post_net (2 ResnetBlocks)
    let post_net = (0..2)
        .map(|i| load_resnet_block(st, &format!("generator.backbone.post_net.{i}"), hidden_dim))
        .collect::<Result<Vec<_>>>()?;

    // head.out
    let n_fft = hop_length * 4;
    let head_w = as2d(
        load_f32(st, "generator.head.out.weight")?,
        out_dim,
        hidden_dim,
    );
    let head_b = as1d(load_f32(st, "generator.head.out.bias")?, out_dim);

    // Hann window: try to load from safetensors; compute as fallback
    let window = if st.tensor("generator.head.istft.window").is_ok() {
        load_f32(st, "generator.head.istft.window")?
    } else {
        hann_window(n_fft)
    };

    // Build the IFFT plan once at load time — the FftPlanner caches plans
    // internally, but recreating the planner on each decode() call would
    // silently discard that cache and re-plan from scratch every time.
    let ifft_plan = {
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_inverse(n_fft)
    };

    Ok(DecoderWeights {
        fsq_proj_w,
        fsq_proj_b,
        fc_post_a_w,
        fc_post_a_b,
        embed_w,
        embed_b,
        prior_net,
        transformers,
        final_norm_w,
        final_norm_b,
        post_net,
        head_w,
        head_b,
        window,
        hidden_dim,
        hop_length,
        depth,
        n_heads,
        ifft_plan,
    })
}

// ─── Decoder forward pass ─────────────────────────────────────────────────────

pub(crate) fn decode_forward(codes: &[i32], w: &DecoderWeights) -> Vec<f32> {
    let hop = w.hop_length;
    let n_fft = hop * 4;
    let embed_k = w.embed_w.shape()[2];
    let embed_pad = embed_k / 2;

    // 1. FSQ decode: [T] → [T, fsq_out_dim]
    let emb = fsq_decode(codes, w.fsq_proj_w.view(), w.fsq_proj_b.view());

    // 2. fc_post_a: [T, fsq_out_dim] → [T, hidden_dim]
    let x = linear(emb.view(), w.fc_post_a_w.view(), Some(w.fc_post_a_b.view()));

    // 3. backbone.embed Conv1d: [hidden_dim, T]
    let x_ct = x.t().to_owned(); // [hidden_dim, T]
    let x_ct = conv1d(
        x_ct.view(),
        w.embed_w.view(),
        Some(w.embed_b.view()),
        embed_pad,
    );

    // 4. prior_net (ResnetBlocks, channels-first)
    let x_ct = w
        .prior_net
        .iter()
        .fold(x_ct, |acc, rw| resnet_block(acc.view(), rw));

    // 5. Transformers (sequence-first)
    let x_tc = x_ct.t().to_owned(); // [T, hidden_dim]
    let x_tc = w
        .transformers
        .iter()
        .fold(x_tc, |acc, tw| transformer_block(acc.view(), tw, w.n_heads));

    // 6. post_net (channels-first)
    let x_ct = x_tc.t().to_owned(); // [hidden_dim, T]
    let x_ct = w
        .post_net
        .iter()
        .fold(x_ct, |acc, rw| resnet_block(acc.view(), rw));

    // 7. final_layer_norm (sequence-first)
    let x_tc = x_ct.t().to_owned(); // [T, hidden_dim]
    let x_tc = layer_norm(
        x_tc.view(),
        w.final_norm_w.view(),
        w.final_norm_b.view(),
        1e-6,
    );

    // 8. head.out: [T, hidden_dim] → [T, n_fft+2]
    let x_pred = linear(x_tc.view(), w.head_w.view(), Some(w.head_b.view()));

    // 9. Transpose → [n_fft+2, T], split mag and phase
    let x_pred_ct = x_pred.t().to_owned(); // [n_fft+2, T]
    let half = (n_fft / 2) + 1; // n_bins = 641 for n_fft=1280, 961 for n_fft=1920
    let mag = x_pred_ct.slice(s![0..half, ..]).to_owned();
    let phase = x_pred_ct.slice(s![half.., ..]).to_owned();

    // 10. ISTFT — use the pre-built plan from DecoderWeights
    istft_burn(
        mag.view(),
        phase.view(),
        hop,
        &w.window,
        w.ifft_plan.as_ref(),
    )
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// NeuCodec decoder: converts speech token IDs to a 24 kHz audio waveform.
///
/// ## Setup
///
/// Set `NEUTTS_DECODER_PATH` to `neucodec_decoder.safetensors`, then:
/// ```rust,ignore
/// let dec = NeuCodecDecoder::new()?;
/// let audio = dec.decode(&codes)?;
/// ```
///
/// ## Backend selection
///
/// When built with `--features wgpu`, the decoder automatically selects the
/// best available backend on the **first call to [`decode`]** (lazy init):
///
/// | Priority | Backend                   | When used                          |
/// |----------|---------------------------|------------------------------------|
/// | 1        | Burn wgpu (GPU)           | Metal / Vulkan / DX12 adapter found|
/// | 2        | Burn NdArray (CPU)        | No GPU adapter available           |
/// | 3        | Raw ndarray (CPU)         | Burn init failed entirely          |
///
/// The Burn backend is initialised **eagerly** at [`from_file`](Self::from_file)
/// time so the GPU upload cost is part of model loading, not synthesis latency.
pub struct NeuCodecDecoder {
    weights: DecoderWeights,
    path: PathBuf,

    /// Eagerly-initialised Burn backend.
    ///
    /// * `Ready(Some(_))` — Burn backend available and ready.
    /// * `Ready(None)`    — Burn init was attempted but failed; falls through
    ///                      to raw ndarray.
    ///
    /// The `Mutex` provides interior mutability so that `decode(&self)` can
    /// be called without `&mut self`.
    /// `Mutex<T>: Send + Sync` when `T: Send`, so `NeuCodecDecoder` remains
    /// `Send + Sync`.
    #[cfg(feature = "burn")]
    burn_decoder: std::sync::Mutex<LazyBurnDecoder>,
}

#[cfg(feature = "burn")]
enum LazyBurnDecoder {
    /// Burn backend is available and ready.
    Ready(Option<Box<dyn super::burn::BurnDecoder + Send>>),
}

impl NeuCodecDecoder {
    /// Load from `NEUTTS_DECODER_PATH`.
    pub fn new() -> Result<Self> {
        let path = super::decoder_weights_path()?;
        Self::from_file(&path)
    }

    /// Load from an explicit file path.
    pub fn from_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            bail!(
                "NeuCodec decoder weights not found: {}\n\
                 Set NEUTTS_DECODER_PATH or pass an explicit path to NeuCodecDecoder::from_file().",
                path.display()
            );
        }

        // Memory-map the file so the OS pages in tensor data on demand instead
        // of reading all 840 MB into a heap Vec<u8> upfront.  This halves peak
        // RAM usage during loading and avoids a large malloc + full-file copy.
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open {}", path.display()))?;
        // SAFETY: we do not mutate the mapping, and we hold `mmap` for the
        // full lifetime of `st` (both are dropped at the end of this block
        // after `load_decoder_weights` has copied all tensors into ndarray).
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .with_context(|| format!("Failed to mmap {}", path.display()))?
        };
        let bytes: &[u8] = &mmap;

        // Read user-defined metadata (n_heads, depth, etc.) from the file header
        let (_, file_meta) = SafeTensors::read_metadata(bytes)
            .with_context(|| format!("Failed to parse safetensors header: {}", path.display()))?;
        let user_meta = file_meta.metadata().clone();

        let st = SafeTensors::deserialize(bytes)
            .with_context(|| format!("Failed to parse safetensors: {}", path.display()))?;

        let weights = load_decoder_weights(&st, &user_meta)
            .with_context(|| format!("Failed to load decoder weights from {}", path.display()))?;

        // `st` and `mmap` are dropped here — all tensor data is now owned by
        // the ndarray arrays inside `weights`.
        drop(st);
        drop(mmap);

        println!(
            "NeuCodec decoder: hidden={}, depth={}, heads={}, hop={} ({} samples/token = {} tokens/s)",
            weights.hidden_dim,
            weights.depth,
            weights.n_heads,
            weights.hop_length,
            weights.hop_length,
            SAMPLE_RATE as usize / weights.hop_length,
        );

        // ── Eagerly initialise the Burn GPU/CPU backend ───────────────────────
        //
        // Initialising here (at load time) rather than lazily on the first
        // decode() call moves the ~1-2 s GPU upload cost out of synthesis
        // latency and into model loading, which is a better user experience:
        // the "loaded in X s" number is accurate, and "synth took Y s" reflects
        // only the actual forward pass.
        #[cfg(feature = "burn")]
        let burn_decoder = {
            let t0 = std::time::Instant::now();
            let dec = super::burn::make_burn_decoder(&weights);
            println!(
                "NeuCodec: {} backend ready in {:.2} s",
                dec.as_ref().map_or("cpu (ndarray)", |b| b.backend_name()),
                t0.elapsed().as_secs_f32(),
            );
            std::sync::Mutex::new(LazyBurnDecoder::Ready(dec))
        };

        Ok(Self {
            weights,
            path: path.to_path_buf(),
            #[cfg(feature = "burn")]
            burn_decoder,
        })
    }

    /// Decode speech token IDs to a 24 kHz audio waveform.
    ///
    /// * `codes` — integer token IDs in `0..=65535` (NeuCodec FSQ range).
    ///   Out-of-range values are rejected with an error rather than silently
    ///   producing garbage digits from the FSQ decomposition.
    /// * returns — `Vec<f32>` of `codes.len() × hop_length` samples.
    pub fn decode(&self, codes: &[i32]) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Ok(Vec::new());
        }

        // Validate before touching any weights — an out-of-range code would
        // silently produce wrong FSQ digits (e.g. a negative modulo result).
        for (i, &code) in codes.iter().enumerate() {
            if !(0..=65535).contains(&code) {
                anyhow::bail!(
                    "Speech token at index {i} is out of range: {code} \
                     (NeuCodec FSQ codes must be in 0..=65535)"
                );
            }
        }

        // ── Prefer Burn-accelerated path (wgpu GPU or NdArray CPU via Burn) ──
        //
        // The backend is always Ready here: it is initialised eagerly in
        // from_file() so there is no lazy-init stall inside synthesis.
        #[cfg(feature = "burn")]
        {
            let state = self.burn_decoder.lock().unwrap();
            if let LazyBurnDecoder::Ready(Some(ref bd)) = *state {
                return bd.decode(codes);
            }
        }

        // ── RLX path (byte-identical to eager until compiled graph lands) ────
        #[cfg(feature = "rlx")]
        {
            super::rlx::decode(codes, &self.weights)
        }
        #[cfg(not(feature = "rlx"))]
        {
            Ok(decode_forward(codes, &self.weights))
        }
    }

    /// Name of the active inference backend.
    pub fn backend_name(&self) -> &'static str {
        #[cfg(feature = "burn")]
        {
            let state = self.burn_decoder.lock().unwrap();
            if let LazyBurnDecoder::Ready(Some(bd)) = &*state {
                return bd.backend_name();
            }
        }
        if cfg!(feature = "rlx") {
            "rlx/eager-parity"
        } else {
            "codec/eager-ndarray"
        }
    }

    /// Alias for [`from_file`](Self::from_file) — load from an explicit path.
    pub fn load(path: &Path) -> Result<Self> {
        Self::from_file(path)
    }

    /// Path from which the decoder was loaded.
    pub fn weights_path(&self) -> &Path {
        &self.path
    }

    /// Detected `hop_length` (audio samples per speech token).
    pub fn hop_length(&self) -> usize {
        self.weights.hop_length
    }
}

// ─── Encoder weights ──────────────────────────────────────────────────────────

#[cfg(feature = "w2v-bert")]
pub(crate) use super::encoder::W2vSemanticRunner;
pub(crate) use super::encoder::{
    EncoderWeights, acoustic_token_len, encode_forward, load_encoder_weights,
    stub_semantic_features,
};

/// Load mono PCM from a WAV file and resample to `target_hz` when needed.
fn load_mono_wav(path: &Path, target_hz: u32) -> Result<Vec<f32>> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read WAV file {}", path.display()))?;
    let (pcm, sample_rate) = decode_wav_mono(&data)
        .with_context(|| format!("Failed to parse WAV {}", path.display()))?;
    if sample_rate == target_hz {
        Ok(pcm)
    } else {
        Ok(resample(&pcm, sample_rate, target_hz))
    }
}

/// Minimal RIFF/WAVE reader for mono PCM (16-bit int or 32-bit float).
fn decode_wav_mono(data: &[u8]) -> Result<(Vec<f32>, u32)> {
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }

    let mut pos = 12usize;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;
    let mut channels = 0u16;
    let mut pcm_bytes: Option<&[u8]> = None;

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_len = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let chunk_end = pos.saturating_add(chunk_len).min(data.len());
        let chunk = &data[pos..chunk_end];

        if chunk_id == b"fmt " && chunk.len() >= 16 {
            audio_format = u16::from_le_bytes(chunk[0..2].try_into().unwrap());
            channels = u16::from_le_bytes(chunk[2..4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(chunk[14..16].try_into().unwrap());
        } else if chunk_id == b"data" {
            pcm_bytes = Some(chunk);
        }

        pos = chunk_end + (chunk_len % 2);
    }

    if channels != 1 {
        bail!("WAV must be mono (got {channels} channels)");
    }
    let pcm_raw = pcm_bytes.ok_or_else(|| anyhow::anyhow!("WAV missing data chunk"))?;
    if sample_rate == 0 {
        bail!("WAV missing sample rate");
    }

    let pcm = match (audio_format, bits_per_sample) {
        (1, 16) => pcm_raw
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect(),
        (3, 32) => pcm_raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        (af, bps) => bail!("unsupported WAV format: audio_format={af} bits={bps}"),
    };

    Ok((pcm, sample_rate))
}

// ─── Encoder ──────────────────────────────────────────────────────────────────

/// NeuCodec encoder: converts a 16 kHz audio waveform to speech token IDs.
///
/// ## Architecture (encode path)
///
/// ```text
/// 16 kHz PCM ──┬──► Wav2Vec2-BERT (layer 16) ──► SemanticEncoder_module ──┐
///              │                                                           ├──► fc_prior ──► FSQ ──► codes
///              └──► CodecEnc (BigCodec, strides [2,2,4,4,5]) ─────────────┘
/// ```
///
/// Weights: `neucodec_encoder.safetensors` from [`scripts/export_neucodec_encoder.py`].
/// Set `NEUTTS_ENCODER_PATH` before [`NeuCodecEncoder::new`].
///
/// ## Status
///
/// * **Done**: weight load, CodecEnc + SemanticEncoder eager forward, fc_prior fusion,
///   FSQ quantize ([`fsq_encode`]), WAV ingest, optional W2V-BERT tap (`w2v-bert` feature).
pub struct NeuCodecEncoder {
    path: PathBuf,
    weights: EncoderWeights,
    #[cfg(feature = "w2v-bert")]
    w2v: Option<W2vSemanticRunner>,
}

impl NeuCodecEncoder {
    /// Load from `NEUTTS_ENCODER_PATH`.
    pub fn new() -> Result<Self> {
        let path = super::encoder_weights_path()?;
        Self::load(&path)
    }

    /// Load encoder weights from an explicit safetensors path.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            bail!(
                "NeuCodec encoder weights not found: {}\n\
                 Export with scripts/export_neucodec_encoder.py and set NEUTTS_ENCODER_PATH.",
                path.display()
            );
        }

        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open {}", path.display()))?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .with_context(|| format!("Failed to mmap {}", path.display()))?
        };
        let bytes: &[u8] = &mmap;

        let (_, file_meta) = SafeTensors::read_metadata(bytes)
            .with_context(|| format!("Failed to parse safetensors header: {}", path.display()))?;
        let user_meta = file_meta.metadata().clone();

        let st = SafeTensors::deserialize(bytes)
            .with_context(|| format!("Failed to parse safetensors: {}", path.display()))?;

        let weights = load_encoder_weights(&st, &user_meta)
            .with_context(|| format!("Failed to load encoder weights from {}", path.display()))?;

        drop(st);
        drop(mmap);

        println!(
            "NeuCodec encoder: CodecEnc tensors={}, semantic tensors={}, \
             strides={:?}, w2v_layer={}",
            weights.codec_enc_tensors,
            weights.semantic_enc_tensors,
            weights.codec_enc_strides,
            weights.semantic_w2v_layer,
        );

        Ok(Self {
            path: path.to_path_buf(),
            #[cfg(feature = "w2v-bert")]
            w2v: W2vSemanticRunner::try_from_env(weights.semantic_w2v_layer)?,
            weights,
        })
    }

    /// Quantize a pre-fused latent `[T, 2048]` to FSQ speech token IDs.
    ///
    /// Useful once the acoustic + semantic pipeline produces latents; mirrors
    /// Python `generator.quantizer` encode.
    pub fn quantize_latent(&self, latent: ArrayView2<f32>) -> Result<Vec<i32>> {
        let in_dim = self.weights.fsq_proj_in_w.shape()[1];
        if latent.shape()[1] != in_dim {
            bail!(
                "latent dim {} != expected fsq in_dim {}",
                latent.shape()[1],
                in_dim
            );
        }
        Ok(fsq_encode(
            latent,
            self.weights.fsq_proj_in_w.view(),
            self.weights.fsq_proj_in_b.view(),
        ))
    }

    /// Encode 16 kHz mono PCM to speech token IDs.
    pub fn encode_pcm(&mut self, pcm_16k: &[f32]) -> Result<Vec<i32>> {
        if pcm_16k.is_empty() {
            bail!("empty PCM input");
        }
        let semantic = self.semantic_features(pcm_16k)?;
        encode_forward(pcm_16k, &self.weights, semantic.view())
    }

    fn semantic_features(&mut self, pcm_16k: &[f32]) -> Result<Array2<f32>> {
        let token_len = acoustic_token_len(pcm_16k.len());
        let hidden = 1024usize;

        #[cfg(feature = "w2v-bert")]
        if let Some(w2v) = self.w2v.as_mut() {
            return w2v.encode_pcm(pcm_16k);
        }

        if std::env::var("NEUTTS_ENCODER_STUB_SEMANTIC")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!("NeuCodec encoder: NEUTTS_ENCODER_STUB_SEMANTIC=1 — zero semantic features");
            return Ok(stub_semantic_features(token_len, hidden));
        }

        #[cfg(feature = "w2v-bert")]
        {
            bail!(
                "Wav2Vec2-BERT weights required for semantic encoding.\n\
                 Set RLX_W2V_BERT_DIR to a facebook/w2v-bert-2.0 snapshot (config.json + model.safetensors),\n\
                 or NEUTTS_ENCODER_STUB_SEMANTIC=1 to exercise the acoustic path only."
            );
        }
        #[cfg(not(feature = "w2v-bert"))]
        {
            bail!(
                "NeuCodec encoder needs semantic features from Wav2Vec2-BERT layer {}.\n\
                 Rebuild with --features w2v-bert and set RLX_W2V_BERT_DIR,\n\
                 or NEUTTS_ENCODER_STUB_SEMANTIC=1 to exercise the acoustic path only.",
                self.weights.semantic_w2v_layer
            );
        }
    }

    /// Encode a mono WAV file (any common rate; resampled to 16 kHz) to speech tokens.
    pub fn encode_wav(&mut self, path: &Path) -> Result<Vec<i32>> {
        let pcm = load_mono_wav(path, ENCODER_SAMPLE_RATE)?;
        self.encode_pcm(&pcm)
    }

    /// Path from which encoder weights were loaded.
    pub fn weights_path(&self) -> &Path {
        &self.path
    }

    /// BigCodec downsample strides from export metadata.
    pub fn codec_enc_strides(&self) -> [usize; 5] {
        self.weights.codec_enc_strides
    }

    /// Wav2Vec2-BERT layer index for semantic features.
    pub fn semantic_w2v_layer(&self) -> usize {
        self.weights.semantic_w2v_layer
    }

    /// Backend name.
    pub fn backend_name(&self) -> &'static str {
        #[cfg(feature = "w2v-bert")]
        if self.w2v.is_some() {
            return "codec/encoder+w2v-bert";
        }
        "codec/encoder-eager"
    }
}

// ─── Resample helper ──────────────────────────────────────────────────────────

/// Naive linear resampler: changes sample rate of `samples` from `from_hz` to `to_hz`.
pub fn resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz {
        return samples.to_vec();
    }
    let ratio = from_hz as f64 / to_hz as f64;
    let out_len = (samples.len() as f64 / ratio).ceil() as usize;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * ratio;
            let lo = src.floor() as usize;
            let hi = (lo + 1).min(samples.len() - 1);
            let frac = (src - lo as f64) as f32;
            samples[lo] * (1.0 - frac) + samples[hi] * frac
        })
        .collect()
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsq_decode_shape() {
        // Minimal project_out: 4-dim output, 8-dim input
        let w = Array2::ones((4, 8));
        let b = Array1::zeros(4);
        let codes = vec![0i32, 1, 2, 65535];
        let out = fsq_decode(&codes, w.view(), b.view());
        assert_eq!(out.shape(), &[4, 4]);
    }

    #[test]
    fn test_fsq_code_0() {
        // Code 0 → all digits 0 → all scaled to -1.0
        // project_out identity (8×8)
        let w = Array2::eye(8);
        let b = Array1::zeros(8);
        let out = fsq_decode(&[0], w.view(), b.view());
        for v in out.iter() {
            assert!((*v + 1.0).abs() < 1e-5, "expected -1.0, got {v}");
        }
    }

    #[test]
    fn test_fsq_code_max() {
        // Code 65535 = 4^8 - 1 → all digits 3 → all scaled to 1.0
        let w = Array2::eye(8);
        let b = Array1::zeros(8);
        let out = fsq_decode(&[65535], w.view(), b.view());
        for v in out.iter() {
            assert!((*v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_fsq_roundtrip_identity() {
        let w = Array2::eye(8);
        let b = Array1::zeros(8);
        let codes = vec![0i32, 42, 1000, 65535];
        let emb = fsq_decode(&codes, w.view(), b.view());
        let back = fsq_encode(emb.view(), w.view(), b.view());
        assert_eq!(codes, back);
    }

    #[test]
    fn test_decode_wav_mono_int16() {
        let samples: Vec<i16> = vec![0, 16384, -16384, 32767];
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        let data_len = 44 - 8 + samples.len() * 2;
        wav.extend_from_slice(&(data_len as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&(16u32).to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&(32000u32).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        let data_size = (samples.len() * 2) as u32;
        wav.extend_from_slice(&data_size.to_le_bytes());
        for s in &samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        let (pcm, rate) = decode_wav_mono(&wav).expect("parse wav");
        assert_eq!(rate, 16000);
        assert_eq!(pcm.len(), 4);
        assert!((pcm[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_linear_shape() {
        let x = Array2::ones((5, 3));
        let w = Array2::ones((7, 3));
        let b = Array1::zeros(7);
        let out = linear(x.view(), w.view(), Some(b.view()));
        assert_eq!(out.shape(), &[5, 7]);
    }

    #[test]
    fn test_conv1d_same_length() {
        let c_in = 4;
        let c_out = 8;
        let t = 16;
        let k = 3;
        let x = Array2::ones((c_in, t));
        let w = Array3::ones((c_out, c_in, k));
        let b = Array1::zeros(c_out);
        let out = conv1d(x.view(), w.view(), Some(b.view()), 1);
        assert_eq!(out.shape(), &[c_out, t]); // same length
    }

    #[test]
    fn test_group_norm_shape() {
        let c = 64;
        let t = 10;
        let x = Array2::ones((c, t));
        let w = Array1::ones(c);
        let b = Array1::zeros(c);
        let out = group_norm(x.view(), 4, w.view(), b.view(), 1e-6);
        assert_eq!(out.shape(), &[c, t]);
        // All-ones input → mean 1, var 0 → norm 0*w + b = 0
        for &v in out.iter() {
            assert!(
                v.abs() < 1e-4,
                "expected ~0 after group_norm of all-ones, got {v}"
            );
        }
    }

    #[test]
    fn test_layer_norm_shape() {
        let t = 5;
        let c = 32;
        let x = Array2::from_elem((t, c), 2.0f32);
        let w = Array1::ones(c);
        let b = Array1::zeros(c);
        let out = layer_norm(x.view(), w.view(), b.view(), 1e-6);
        assert_eq!(out.shape(), &[t, c]);
        // Constant input → LayerNorm output is 0
        for &v in out.iter() {
            assert!(v.abs() < 1e-4, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_rms_norm_shape() {
        let t = 3;
        let c = 8;
        let x = Array2::ones((t, c));
        let w = Array1::ones(c);
        let out = rms_norm(x.view(), w.view(), 1e-6);
        assert_eq!(out.shape(), &[t, c]);
        // RMSNorm of all-ones → 1/rms(1) * 1 = 1
        for &v in out.iter() {
            assert!((v - 1.0).abs() < 1e-4, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_rope_shape_preserved() {
        let t = 4;
        let n_heads = 2;
        let head_dim = 8;
        let mut x = Array3::ones((t, n_heads, head_dim));
        apply_rope(&mut x);
        assert_eq!(x.shape(), &[t, n_heads, head_dim]);
    }

    #[test]
    fn test_hann_window() {
        let w = hann_window(4);
        assert_eq!(w.len(), 4);
        // Hann window: w[0] = 0, w[n/2] = 1, w[n] = 0
        assert!(w[0].abs() < 1e-6);
        assert!((w[2] - 1.0).abs() < 1e-6);
    }

    fn make_ifft(n_fft: usize) -> std::sync::Arc<dyn rustfft::Fft<f32>> {
        FftPlanner::<f32>::new().plan_fft_inverse(n_fft)
    }

    #[test]
    fn test_istft_length() {
        let hop = 4;
        let n_fft = 16; // hop * 4
        let t = 10;
        let n_bins = n_fft / 2 + 1; // 9
        // Zero mag → exp(0)=1 magnitude, zero phase → cos(0)=1, sin(0)=0
        let mag = Array2::zeros((n_bins, t));
        let phase = Array2::zeros((n_bins, t));
        let win = hann_window(n_fft);
        let ifft = make_ifft(n_fft);
        let audio = istft_burn(mag.view(), phase.view(), hop, &win, ifft.as_ref());
        // center=True: output is exactly T*hop samples
        assert_eq!(
            audio.len(),
            t * hop,
            "expected {} samples, got {}",
            t * hop,
            audio.len()
        );
    }

    #[test]
    fn test_istft_clamp_does_not_blow_up() {
        // Log-magnitudes well above ln(100)≈4.6 must be clamped to 100 (linear),
        // not allowed to reach exp(large) ≈ infinity.
        let hop = 4;
        let n_fft = 16;
        let t = 4;
        let n_bins = n_fft / 2 + 1;
        // All log-magnitudes = 50 (would give exp(50) ≈ 5e21 without the fix)
        let mag = Array2::from_elem((n_bins, t), 50.0f32);
        let phase = Array2::zeros((n_bins, t));
        let win = hann_window(n_fft);
        let ifft = make_ifft(n_fft);
        let audio = istft_burn(mag.view(), phase.view(), hop, &win, ifft.as_ref());
        // All samples must be finite and ≤ some reasonable bound (the clamp
        // limits linear magnitude to 1e2, so waveform values should be bounded)
        for &s in &audio {
            assert!(s.is_finite(), "sample is not finite: {s}");
            assert!(s.abs() < 1e6, "sample magnitude suspiciously large: {s}");
        }
    }

    #[test]
    fn test_burn_feature_fn() {
        let _ = crate::features::burn_feature_enabled();
    }

    #[test]
    fn test_resample_identity() {
        let s: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let r = resample(&s, 16_000, 16_000);
        assert_eq!(r, s);
    }

    /// Active decoder path (eager / burn / rlx) must match the ndarray gold forward.
    #[test]
    fn decode_output_matches_eager_forward() {
        let Some(path) = crate::decoder::decoder_weights_path_if_available() else {
            eprintln!("skip decode_output_matches_eager_forward: set NEUTTS_DECODER_PATH");
            return;
        };

        let codes: Vec<i32> = vec![0, 42, 128, 512, 1023];
        let dec = NeuCodecDecoder::from_file(&path).expect("NeuCodecDecoder::from_file");
        let actual = dec.decode(&codes).expect("decode");
        eprintln!(
            "decode_output_matches_eager_forward: backend={}",
            dec.backend_name()
        );

        let data = std::fs::read(&path).expect("read safetensors");
        let st = safetensors::SafeTensors::deserialize(&data).expect("safetensors");
        let w = load_decoder_weights(&st, &None).expect("load_decoder_weights");
        let expected = decode_forward(&codes, &w);

        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(a.is_finite() && e.is_finite(), "non-finite at {i}");
            let diff = (a - e).abs();
            assert!(
                diff < 1e-3,
                "sample {i}: actual={a} expected={e} diff={diff} backend={}",
                dec.backend_name()
            );
        }
    }
}

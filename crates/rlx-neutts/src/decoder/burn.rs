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

//! Burn wgpu / NdArray backend for the NeuCodec decoder.
//!
//! When the `wgpu` Cargo feature is enabled, this module provides a
//! GPU-accelerated decoder that runs the full NeuCodec forward pass on the
//! device.  The only step that returns to CPU is the final ISTFT (complex FFT
//! via `rustfft`), which happens once per `decode()` call.
//!
//! ## Backend selection (runtime)
//!
//! 1. **Burn wgpu** — tries `WgpuDevice::DefaultDevice`; used when a
//!    compatible Metal / Vulkan / DX12 / WebGPU adapter is found.
//! 2. **Burn NdArray** — pure-CPU Burn backend; used when wgpu init fails.
//! 3. **Raw ndarray** — the original pure-ndarray decoder from `codec.rs`;
//!    used only as a last-resort fallback (weights already loaded).
//!
//! ## Architecture
//!
//! All weights are uploaded to the selected device once at load time as
//! `Tensor<B, D>` tensors.  Each call to [`BurnDecoder::decode`] runs:
//!
//! ```text
//!  codes [i32]
//!    └─► fsq_decode             (Tensor matmul → [T, 2048])
//!    └─► fc_post_a linear       ([T, 1024])
//!    └─► backbone.embed Conv1d  ([1024, T])
//!    └─► 2 × ResnetBlock        (GroupNorm + SiLU + Conv1d)
//!    └─► N × TransformerBlock   (RMSNorm + MHA + SiLU MLP)
//!    └─► 2 × ResnetBlock
//!    └─► LayerNorm
//!    └─► head.out linear        ([T, n_fft+2])
//!    └─► into_data() → CPU
//!    └─► ISTFT                  (rustfft, always CPU)
//!    └─► Vec<f32>
//! ```

use anyhow::Result;

use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    tensor::{
        Tensor, TensorData,
        activation::{self, softmax},
        backend::Backend,
        module::conv1d,
        ops::ConvOptions,
    },
};

use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;

use super::eager::{DecoderWeights, FSQ_BASIS, FSQ_LEVELS, istft_burn};

// ─── Object-safe trait ───────────────────────────────────────────────────────

/// Object-safe interface over a backend-specific Burn decoder.
///
/// Stored as `Box<dyn BurnDecoder + Send>` in [`NeuCodecDecoder`].
pub(crate) trait BurnDecoder {
    /// Decode speech token IDs to 24 kHz audio samples.
    fn decode(&self, codes: &[i32]) -> Result<Vec<f32>>;
    /// Human-readable backend name.
    fn backend_name(&self) -> &'static str;
}

// ─── Per-backend weight structs ───────────────────────────────────────────────

struct BurnResnetBlock<B: Backend> {
    norm1_w: Tensor<B, 1>,
    norm1_b: Tensor<B, 1>,
    conv1_w: Tensor<B, 3>,
    conv1_b: Tensor<B, 1>,
    norm2_w: Tensor<B, 1>,
    norm2_b: Tensor<B, 1>,
    conv2_w: Tensor<B, 3>,
    conv2_b: Tensor<B, 1>,
}

struct BurnTransformer<B: Backend> {
    att_norm_w: Tensor<B, 1>,
    c_attn_w: Tensor<B, 2>, // [3D, D]
    c_proj_w: Tensor<B, 2>, // [D, D]
    ffn_norm_w: Tensor<B, 1>,
    fc1_w: Tensor<B, 2>, // [4D, D]
    fc2_w: Tensor<B, 2>, // [D, 4D]
}

struct BurnWeights<B: Backend> {
    // FSQ codebook projection
    fsq_proj_w: Tensor<B, 2>, // [fsq_out, 8]
    fsq_proj_b: Tensor<B, 1>, // [fsq_out]
    // Dimension reduction after FSQ
    fc_post_a_w: Tensor<B, 2>, // [hidden, fsq_out]
    fc_post_a_b: Tensor<B, 1>, // [hidden]
    // Backbone
    embed_w: Tensor<B, 3>, // [hidden, hidden, k]
    embed_b: Tensor<B, 1>, // [hidden]
    prior_net: Vec<BurnResnetBlock<B>>,
    transformers: Vec<BurnTransformer<B>>,
    final_norm_w: Tensor<B, 1>, // [hidden]
    final_norm_b: Tensor<B, 1>, // [hidden]
    post_net: Vec<BurnResnetBlock<B>>,
    // ISTFT head
    head_w: Tensor<B, 2>, // [n_fft+2, hidden]
    head_b: Tensor<B, 1>, // [n_fft+2]
    // ISTFT (stays on CPU)
    window: Vec<f32>,
    // Pre-built IFFT plan — shared from DecoderWeights so we don't re-plan
    // on every decode() call.
    ifft_plan: std::sync::Arc<dyn rustfft::Fft<f32>>,
    // Hyper-parameters
    hop_length: usize,
    n_heads: usize,
    embed_pad: usize,
}

// ─── Weight loading ───────────────────────────────────────────────────────────

/// Helper: ndarray Array1 → Burn 1-D tensor on `device`.
fn a1_to_t1<B: Backend>(a: &ndarray::Array1<f32>, device: &B::Device) -> Tensor<B, 1> {
    let n = a.len();
    let data = TensorData::new(a.iter().copied().collect::<Vec<_>>(), vec![n]);
    Tensor::from_data(data, device)
}

/// Helper: ndarray Array2 → Burn 2-D tensor on `device`.
fn a2_to_t2<B: Backend>(a: &ndarray::Array2<f32>, device: &B::Device) -> Tensor<B, 2> {
    let [rows, cols] = [a.shape()[0], a.shape()[1]];
    let data = TensorData::new(a.iter().copied().collect::<Vec<_>>(), vec![rows, cols]);
    Tensor::from_data(data, device)
}

/// Helper: ndarray Array3 → Burn 3-D tensor on `device`.
fn a3_to_t3<B: Backend>(a: &ndarray::Array3<f32>, device: &B::Device) -> Tensor<B, 3> {
    let [d0, d1, d2] = [a.shape()[0], a.shape()[1], a.shape()[2]];
    let data = TensorData::new(a.iter().copied().collect::<Vec<_>>(), vec![d0, d1, d2]);
    Tensor::from_data(data, device)
}

fn load_resnet<B: Backend>(
    dw: &super::eager::ResnetBlockWeights,
    device: &B::Device,
) -> BurnResnetBlock<B> {
    BurnResnetBlock {
        norm1_w: a1_to_t1(&dw.norm1_w, device),
        norm1_b: a1_to_t1(&dw.norm1_b, device),
        conv1_w: a3_to_t3(&dw.conv1_w, device),
        conv1_b: a1_to_t1(&dw.conv1_b, device),
        norm2_w: a1_to_t1(&dw.norm2_w, device),
        norm2_b: a1_to_t1(&dw.norm2_b, device),
        conv2_w: a3_to_t3(&dw.conv2_w, device),
        conv2_b: a1_to_t1(&dw.conv2_b, device),
    }
}

fn load_transformer<B: Backend>(
    tw: &super::eager::TransformerWeights,
    device: &B::Device,
) -> BurnTransformer<B> {
    BurnTransformer {
        att_norm_w: a1_to_t1(&tw.att_norm_w, device),
        c_attn_w: a2_to_t2(&tw.c_attn_w, device),
        c_proj_w: a2_to_t2(&tw.c_proj_w, device),
        ffn_norm_w: a1_to_t1(&tw.ffn_norm_w, device),
        fc1_w: a2_to_t2(&tw.fc1_w, device),
        fc2_w: a2_to_t2(&tw.fc2_w, device),
    }
}

fn load_weights<B: Backend>(dw: &DecoderWeights, device: &B::Device) -> BurnWeights<B> {
    let embed_k = dw.embed_w.shape()[2];
    let embed_pad = embed_k / 2;
    BurnWeights {
        fsq_proj_w: a2_to_t2(&dw.fsq_proj_w, device),
        fsq_proj_b: a1_to_t1(&dw.fsq_proj_b, device),
        fc_post_a_w: a2_to_t2(&dw.fc_post_a_w, device),
        fc_post_a_b: a1_to_t1(&dw.fc_post_a_b, device),
        embed_w: a3_to_t3(&dw.embed_w, device),
        embed_b: a1_to_t1(&dw.embed_b, device),
        prior_net: dw
            .prior_net
            .iter()
            .map(|r| load_resnet(r, device))
            .collect(),
        transformers: dw
            .transformers
            .iter()
            .map(|t| load_transformer(t, device))
            .collect(),
        final_norm_w: a1_to_t1(&dw.final_norm_w, device),
        final_norm_b: a1_to_t1(&dw.final_norm_b, device),
        post_net: dw.post_net.iter().map(|r| load_resnet(r, device)).collect(),
        head_w: a2_to_t2(&dw.head_w, device),
        head_b: a1_to_t1(&dw.head_b, device),
        window: dw.window.clone(),
        ifft_plan: std::sync::Arc::clone(&dw.ifft_plan),
        hop_length: dw.hop_length,
        n_heads: dw.n_heads,
        embed_pad,
    }
}

// ─── Math primitives (Burn tensor ops) ───────────────────────────────────────

/// Linear layer: `out = x @ w.T + b`
///
/// * `x`: [T, in_dim]   * `w`: [out_dim, in_dim]   * `b`: [out_dim]
/// * Returns: [T, out_dim]
#[inline]
fn t_linear<B: Backend>(
    x: Tensor<B, 2>,
    w: &Tensor<B, 2>,
    b: Option<&Tensor<B, 1>>,
) -> Tensor<B, 2> {
    let out = x.matmul(w.clone().transpose());
    match b {
        Some(b) => out + b.clone().unsqueeze_dim::<2>(0), // [1, out_dim] → broadcast
        None => out,
    }
}

/// Conv1d with same-length output (zero-padded on each side).
///
/// * `x`: [c_in, T]   * `w`: [c_out, c_in, k]   * `b`: [c_out]
/// * Returns: [c_out, T]
#[inline]
fn t_conv1d<B: Backend>(
    x: Tensor<B, 2>,
    w: &Tensor<B, 3>,
    b: Option<&Tensor<B, 1>>,
    pad: usize,
) -> Tensor<B, 2> {
    let opts = ConvOptions::new([1], [pad], [1], 1);
    // Burn conv1d expects [batch, c_in, T]; we use batch=1.
    let x3 = x.unsqueeze_dim::<3>(0); // [1, c_in, T]
    let out = conv1d(x3, w.clone(), b.cloned(), opts); // [1, c_out, T]
    out.squeeze_dim::<2>(0) // [c_out, T]
}

/// GroupNorm: normalises over (group_size × T) for each group.
///
/// * `x`: [C, T]  (channels-first)
/// * `w`, `b`: [C]
fn t_group_norm<B: Backend>(
    x: Tensor<B, 2>,
    n_groups: usize,
    w: &Tensor<B, 1>,
    b: &Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 2> {
    let [c, t] = x.dims();
    let gs = c / n_groups; // group_size

    // [C, T] → [n_groups, gs × T]: normalise over last dim per group
    let xg = x.reshape([n_groups, gs * t]);

    let mean = xg.clone().mean_dim(1); // [n_groups, 1]
    let xc = xg - mean;
    let var = xc.clone().square().mean_dim(1); // [n_groups, 1]
    let inv_std = (var.add_scalar(eps)).sqrt().recip(); // [n_groups, 1]
    let xn = (xc * inv_std).reshape([c, t]); // [C, T]

    // Affine: w/b are [C]; need [C, 1] for broadcast with [C, T]
    let w2 = w.clone().unsqueeze_dim::<2>(1); // [C, 1]
    let b2 = b.clone().unsqueeze_dim::<2>(1); // [C, 1]
    xn * w2 + b2
}

/// LayerNorm over the last axis of [T, C].
fn t_layer_norm<B: Backend>(
    x: Tensor<B, 2>,
    w: &Tensor<B, 1>,
    b: &Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 2> {
    let mean = x.clone().mean_dim(1); // [T, 1]
    let xc = x - mean;
    let var = xc.clone().square().mean_dim(1); // [T, 1]
    let inv_std = (var.add_scalar(eps)).sqrt().recip(); // [T, 1]
    let xn = xc * inv_std; // [T, C]

    // w/b: [C] → [1, C] for broadcast with [T, C]
    let w2 = w.clone().unsqueeze_dim::<2>(0);
    let b2 = b.clone().unsqueeze_dim::<2>(0);
    xn * w2 + b2
}

/// RMSNorm over the last axis of [T, C].
fn t_rms_norm<B: Backend>(x: Tensor<B, 2>, w: &Tensor<B, 1>, eps: f32) -> Tensor<B, 2> {
    let ms = x.clone().square().mean_dim(1); // [T, 1]
    let scale = (ms.add_scalar(eps)).sqrt().recip(); // [T, 1]
    let xn = x * scale; // [T, C]
    let w2 = w.clone().unsqueeze_dim::<2>(0); // [1, C]
    xn * w2
}

// ─── Building blocks ──────────────────────────────────────────────────────────

fn t_resnet_block<B: Backend>(x: Tensor<B, 2>, rw: &BurnResnetBlock<B>) -> Tensor<B, 2> {
    // Branch 1: GroupNorm → SiLU → Conv1d(k=3,pad=1)
    let h = t_group_norm(x.clone(), 32, &rw.norm1_w, &rw.norm1_b, 1e-6);
    let h = activation::silu(h);
    let h = t_conv1d(h, &rw.conv1_w, Some(&rw.conv1_b), 1);

    // Branch 2: GroupNorm → SiLU → Conv1d(k=3,pad=1)
    let h = t_group_norm(h, 32, &rw.norm2_w, &rw.norm2_b, 1e-6);
    let h = activation::silu(h);
    let h = t_conv1d(h, &rw.conv2_w, Some(&rw.conv2_b), 1);

    // Residual
    x + h
}

fn t_transformer_block<B: Backend>(
    x: Tensor<B, 2>,
    tw: &BurnTransformer<B>,
    n_heads: usize,
) -> Tensor<B, 2> {
    let [t, d] = x.dims();
    let head_dim = d / n_heads;

    // ── Attention sub-layer ───────────────────────────────────────────────────
    let normed = t_rms_norm(x.clone(), &tw.att_norm_w, 1e-6);

    // QKV: [T, D] @ [D, 3D] = [T, 3D]
    let qkv = t_linear(normed, &tw.c_attn_w, None);

    let q_flat = qkv.clone().slice([0..t, 0..d]);
    let k_flat = qkv.clone().slice([0..t, d..2 * d]);
    let v_flat = qkv.slice([0..t, 2 * d..3 * d]);

    // [T, D] → [T, n_heads, head_dim]. No RoPE: NeuCodec rotates q and k by
    // the head's index (`position_ids = arange(num_heads)`), the same
    // rotation for both, which cancels in qᵀk.
    let q = q_flat.reshape([t, n_heads, head_dim]);
    let k = k_flat.reshape([t, n_heads, head_dim]);
    let v = v_flat.reshape([t, n_heads, head_dim]);

    // Batched attention: permute to [n_heads, T, head_dim]
    let q_b = q.permute([1, 0, 2]);
    let k_b = k.permute([1, 0, 2]);
    let v_b = v.permute([1, 0, 2]);

    let scale = (head_dim as f64).sqrt().recip() as f32;

    // scores: [n_heads, T, T]
    let scores = q_b.matmul(k_b.swap_dims(1, 2)).mul_scalar(scale);
    let attn = softmax(scores, 2);

    // weighted values: [n_heads, T, head_dim] → [T, n_heads, head_dim] → [T, D]
    let attn_out = attn.matmul(v_b).permute([1, 0, 2]).reshape([t, d]);

    // Out projection + residual
    let attn_proj = t_linear(attn_out, &tw.c_proj_w, None);
    let x_attn = x + attn_proj;

    // ── MLP sub-layer ─────────────────────────────────────────────────────────
    let normed2 = t_rms_norm(x_attn.clone(), &tw.ffn_norm_w, 1e-6);
    let h1 = t_linear(normed2, &tw.fc1_w, None);
    let h1_act = activation::silu(h1);
    let h2 = t_linear(h1_act, &tw.fc2_w, None);

    x_attn + h2
}

// ─── FSQ decode ───────────────────────────────────────────────────────────────

fn t_fsq_decode<B: Backend>(
    codes: &[i32],
    proj_w: &Tensor<B, 2>, // [fsq_out, 8]
    proj_b: &Tensor<B, 1>, // [fsq_out]
) -> Tensor<B, 2> {
    let t = codes.len();
    let device = proj_w.device();

    // Build [T, 8] digit matrix from integer codes
    let mut digits = vec![0.0f32; t * 8];
    for (i, &code) in codes.iter().enumerate() {
        for (j, (&basis, &levels)) in FSQ_BASIS.iter().zip(FSQ_LEVELS.iter()).enumerate() {
            let d = (code / basis) % levels;
            // `(d − L/2) / (L/2)`: {−1, −½, 0, ½} for L = 4.
            let half = (levels / 2) as f32;
            digits[i * 8 + j] = (d as f32 - half) / half;
        }
    }

    let d_data = TensorData::new(digits, vec![t, 8]);
    let d_t: Tensor<B, 2> = Tensor::from_data(d_data, &device);

    t_linear(d_t, proj_w, Some(proj_b))
}

// ─── Full forward pass ────────────────────────────────────────────────────────

fn burn_decode<B: Backend>(codes: &[i32], w: &BurnWeights<B>) -> Vec<f32> {
    let hop = w.hop_length;
    let n_fft = hop * 4;
    let t = codes.len();

    // 1. FSQ decode: [T, 8] → [T, fsq_out]
    let emb = t_fsq_decode(codes, &w.fsq_proj_w, &w.fsq_proj_b);

    // 2. fc_post_a: [T, fsq_out] → [T, hidden]
    let x = t_linear(emb, &w.fc_post_a_w, Some(&w.fc_post_a_b));

    // 3. backbone.embed Conv1d: [hidden, T]
    let x_ct = t_conv1d(x.transpose(), &w.embed_w, Some(&w.embed_b), w.embed_pad);

    // 4. prior_net (ResnetBlocks, channels-first)
    let x_ct = w
        .prior_net
        .iter()
        .fold(x_ct, |acc, rw| t_resnet_block(acc, rw));

    // 5. Transformers (sequence-first [T, hidden])
    let x_tc = w.transformers.iter().fold(x_ct.transpose(), |acc, tw| {
        t_transformer_block(acc, tw, w.n_heads)
    });

    // 6. post_net (channels-first)
    let x_ct = w
        .post_net
        .iter()
        .fold(x_tc.transpose(), |acc, rw| t_resnet_block(acc, rw));

    // 7. final_layer_norm (sequence-first)
    let x_tc = t_layer_norm(x_ct.transpose(), &w.final_norm_w, &w.final_norm_b, 1e-6);

    // 8. head.out: [T, hidden] → [T, n_fft+2]
    let x_pred = t_linear(x_tc, &w.head_w, Some(&w.head_b));

    // 9. Pull to CPU: x_pred is row-major [T, n_fft+2]
    let [_n_frames, n_out] = x_pred.dims();
    let flat: Vec<f32> = x_pred
        .into_data()
        .into_vec::<f32>()
        .expect("tensor data read");

    // 10. Rearrange to [n_bins, T] for mag and phase (matching CPU codec layout)
    let n_bins = n_fft / 2 + 1;
    let mut mag = vec![0.0f32; n_bins * t];
    let mut phase = vec![0.0f32; n_bins * t];
    for ti in 0..t {
        for fi in 0..n_bins {
            mag[fi * t + ti] = flat[ti * n_out + fi];
            phase[fi * t + ti] = flat[ti * n_out + n_bins + fi];
        }
    }

    let mag_a = ndarray::Array2::from_shape_vec((n_bins, t), mag).expect("mag shape");
    let phase_a = ndarray::Array2::from_shape_vec((n_bins, t), phase).expect("phase shape");

    // 11. ISTFT (CPU, rustfft) — use the pre-built plan stored in DecoderWeights
    istft_burn(
        mag_a.view(),
        phase_a.view(),
        hop,
        &w.window,
        w.ifft_plan.as_ref(),
    )
}

// ─── Concrete decoder implementations ────────────────────────────────────────

struct BurnDecoderImpl<B: Backend> {
    weights: BurnWeights<B>,
    name: &'static str,
}

impl<B: Backend + Send> BurnDecoder for BurnDecoderImpl<B>
where
    BurnWeights<B>: Send,
{
    fn decode(&self, codes: &[i32]) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Ok(Vec::new());
        }
        Ok(burn_decode(codes, &self.weights))
    }

    fn backend_name(&self) -> &'static str {
        self.name
    }
}

// ─── Public factory ───────────────────────────────────────────────────────────

/// Build a boxed [`BurnDecoder`], trying wgpu first and falling back to NdArray.
///
/// Returns `None` if loading fails on both backends (caller should fall back
/// to the raw ndarray decoder).
pub(crate) fn make_burn_decoder(dw: &DecoderWeights) -> Option<Box<dyn BurnDecoder + Send>> {
    // ── Attempt 1: wgpu GPU ───────────────────────────────────────────────────
    //
    // `DecoderWeights` contains `Arc<dyn Fft<f32>>` which doesn't implement
    // `RefUnwindSafe` (the trait object may have interior mutability).  We
    // know the IFFT plan is only read here (no mutation across the unwind
    // boundary), so wrapping in `AssertUnwindSafe` is sound.
    let wgpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = WgpuDevice::DefaultDevice;
        let weights = load_weights::<Wgpu>(dw, &device);
        Box::new(BurnDecoderImpl::<Wgpu> {
            weights,
            name: "burn/wgpu (GPU)",
        }) as Box<dyn BurnDecoder + Send>
    }));

    if let Ok(dec) = wgpu_result {
        println!("NeuCodec: using Burn wgpu (GPU) backend");
        return Some(dec);
    }

    // ── Attempt 2: Burn NdArray CPU ──────────────────────────────────────────
    let ndarray_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = NdArrayDevice::Cpu;
        let weights = load_weights::<NdArray>(dw, &device);
        Box::new(BurnDecoderImpl::<NdArray> {
            weights,
            name: "burn/ndarray (CPU)",
        }) as Box<dyn BurnDecoder + Send>
    }));

    match ndarray_result {
        Ok(dec) => {
            println!("NeuCodec: using Burn NdArray (CPU) backend (wgpu unavailable)");
            Some(dec)
        }
        Err(_) => {
            eprintln!("NeuCodec: Burn NdArray init failed — falling back to raw ndarray");
            None
        }
    }
}

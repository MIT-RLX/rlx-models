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

//! Host (CPU) reference for the generative DiTs.
//!
//! A faithful, batch-1 fp32 port of `SparseStructureFlowModel` /
//! `SLatFlowModel` (`trellis2/models/{sparse_structure_flow,structured_latent_flow}.py`)
//! + `ModulatedTransformerCrossBlock`. It is the parity oracle for the graph
//! lowering and a CPU fallback. The two DiTs differ only in tokenization:
//!   * dense structure DiT — tokens are a fixed `res³` grid (RoPE from grid);
//!   * sparse SLat DiT — tokens are active voxels (RoPE from their coords),
//!     with an optional `concat_cond` appended to the input features.
//!
//! Everything runs in fp32; weights are the bf16 checkpoint promoted to f32.

use crate::config::DitConfig;
use crate::rope;
use anyhow::{Context, Result};
use rlx_core::host_kernels::{gelu_tanh, matmul, matmul_bt, softmax_rows};
use rlx_core::weight_map::WeightMap;

/// Timestep sinusoid → MLP embedding, matching `TimestepEmbedder`.
/// `freq_embed_size = 256`; `emb = [cos(t·f), sin(t·f)]`, `f_i = 10000^{-i/128}`.
pub fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut e = vec![0.0f32; dim];
    for i in 0..half {
        let freq = (10000f32).powf(-(i as f32) / half as f32);
        let a = t * freq;
        e[i] = a.cos();
        e[i + half] = a.sin();
    }
    if dim % 2 == 1 {
        e[dim - 1] = 0.0;
    }
    e
}

fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

/// `y = x · Wᵀ + b`, with PyTorch-native weight layout `W = [out, in]`.
fn linear_raw(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    w: &[f32],
    out_dim: usize,
    b: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_dim];
    matmul_bt(x, w, &mut out, rows, in_dim, out_dim, 1.0);
    if let Some(b) = b {
        for r in 0..rows {
            let row = &mut out[r * out_dim..(r + 1) * out_dim];
            for (o, bv) in row.iter_mut().zip(b) {
                *o += *bv;
            }
        }
    }
    out
}

/// Non-affine LayerNorm over the last `dim`, eps `1e-6`.
fn layer_norm_noaffine(x: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    let eps = 1e-6f32;
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        let o = &mut out[r * dim..(r + 1) * dim];
        for (oo, v) in o.iter_mut().zip(row) {
            *oo = (v - mean) * inv;
        }
    }
    out
}

/// Affine LayerNorm (weight/bias) over the last `dim`, eps `1e-6`.
fn layer_norm_affine(x: &[f32], rows: usize, dim: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = layer_norm_noaffine(x, rows, dim);
    for r in 0..rows {
        let o = &mut out[r * dim..(r + 1) * dim];
        for i in 0..dim {
            o[i] = o[i] * w[i] + b[i];
        }
    }
    out
}

/// Per-head RMS-style norm from `MultiHeadRMSNorm`:
/// `y = normalize(x)·γ·√head_dim` over `head_dim` (F.normalize eps `1e-12`).
fn multihead_rms_norm(x: &mut [f32], n_pos: usize, heads: usize, head_dim: usize, gamma: &[f32]) {
    let scale = (head_dim as f32).sqrt();
    for p in 0..n_pos {
        for h in 0..heads {
            let off = (p * heads + h) * head_dim;
            let seg = &mut x[off..off + head_dim];
            let n2 = seg.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            let g = &gamma[h * head_dim..(h + 1) * head_dim];
            for i in 0..head_dim {
                seg[i] = seg[i] / n2 * g[i] * scale;
            }
        }
    }
}

/// Multi-head attention from `[n, heads, head_dim]` q/k/v (naive SDPA, no mask,
/// scale `1/√head_dim`). Returns `[n_q, heads*head_dim]`.
fn mha(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_k: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let dim = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_q * dim];
    let mut qh = vec![0.0f32; n_q * head_dim];
    let mut kh = vec![0.0f32; n_k * head_dim];
    let mut vh = vec![0.0f32; n_k * head_dim];
    let mut scores = vec![0.0f32; n_q * n_k];
    let mut oh = vec![0.0f32; n_q * head_dim];
    for h in 0..heads {
        for p in 0..n_q {
            qh[p * head_dim..(p + 1) * head_dim].copy_from_slice(
                &q[(p * heads + h) * head_dim..(p * heads + h) * head_dim + head_dim],
            );
        }
        for p in 0..n_k {
            kh[p * head_dim..(p + 1) * head_dim].copy_from_slice(
                &k[(p * heads + h) * head_dim..(p * heads + h) * head_dim + head_dim],
            );
            vh[p * head_dim..(p + 1) * head_dim].copy_from_slice(
                &v[(p * heads + h) * head_dim..(p * heads + h) * head_dim + head_dim],
            );
        }
        // scores = qh @ kh^T * scale   [n_q, n_k]
        matmul_bt(&qh, &kh, &mut scores, n_q, head_dim, n_k, scale);
        softmax_rows(&mut scores, n_q, n_k);
        // oh = scores @ vh   [n_q, head_dim]
        matmul(&scores, &vh, &mut oh, n_q, n_k, head_dim);
        for p in 0..n_q {
            out[p * dim + h * head_dim..p * dim + (h + 1) * head_dim]
                .copy_from_slice(&oh[p * head_dim..(p + 1) * head_dim]);
        }
    }
    out
}

/// Loaded per-block weights (borrowed from the [`WeightMap`]).
struct BlockW<'a> {
    modulation: &'a [f32], // [6*C]
    norm2_w: &'a [f32],
    norm2_b: &'a [f32],
    // self-attn
    qkv_w: &'a [f32], // [3C, C]
    qkv_b: &'a [f32],
    sa_q_gamma: &'a [f32], // [heads, head_dim]
    sa_k_gamma: &'a [f32],
    sa_out_w: &'a [f32],
    sa_out_b: &'a [f32],
    // cross-attn
    ca_q_w: &'a [f32], // [C, C]
    ca_q_b: &'a [f32],
    ca_kv_w: &'a [f32], // [2C, cond]
    ca_kv_b: &'a [f32],
    ca_q_gamma: &'a [f32],
    ca_k_gamma: &'a [f32],
    ca_out_w: &'a [f32],
    ca_out_b: &'a [f32],
    // mlp
    mlp0_w: &'a [f32], // [hidden, C]
    mlp0_b: &'a [f32],
    mlp2_w: &'a [f32], // [C, hidden]
    mlp2_b: &'a [f32],
}

fn get<'a>(wm: &'a WeightMap, key: &str) -> Result<&'a [f32]> {
    wm.get(key)
        .map(|(d, _)| d)
        .with_context(|| format!("missing weight {key}"))
}

/// Intermediate activations captured for parity bisection.
#[derive(Default)]
pub struct DitDump {
    pub after_input: Vec<f32>,
    pub after_block0: Vec<f32>,
    pub after_final_ln: Vec<f32>,
    pub out: Vec<f32>,
}

/// Timestep → shared adaLN modulation base `[6·C]` (or `[C]` when `!share_mod`).
///
/// Matches `t_embedder` (+ optional `adaLN_modulation`) in the upstream DiTs.
/// Used by both the host reference and the compiled flow path.
pub fn shared_modulation(cfg: &DitConfig, wm: &WeightMap, t: f32) -> Result<Vec<f32>> {
    let c = cfg.args.model_channels;
    let tfreq = timestep_embedding(t, 256);
    let te0_w = get(wm, "t_embedder.mlp.0.weight")?;
    let te0_b = get(wm, "t_embedder.mlp.0.bias")?;
    let te2_w = get(wm, "t_embedder.mlp.2.weight")?;
    let te2_b = get(wm, "t_embedder.mlp.2.bias")?;
    let mut te = linear_raw(&tfreq, 1, 256, te0_w, c, Some(te0_b));
    silu_inplace(&mut te);
    let mut t_emb = linear_raw(&te, 1, c, te2_w, c, Some(te2_b));
    if cfg.args.share_mod {
        silu_inplace(&mut t_emb);
        let ada_w = get(wm, "adaLN_modulation.1.weight")?;
        let ada_b = get(wm, "adaLN_modulation.1.bias")?;
        Ok(linear_raw(&t_emb, 1, c, ada_w, 6 * c, Some(ada_b)))
    } else {
        Ok(t_emb)
    }
}

/// Run the DiT torso over pre-tokenized features.
///
/// * `tokens` — `[n_pos * in_channels]` input features (row-major, channels-last).
/// * `coords` — `[n_pos * 3]` voxel coordinates (f32) for RoPE.
/// * `cond`   — `[n_cond * cond_channels]` conditioner features.
///
/// Returns the `[n_pos * out_channels]` velocity prediction (channels-last).
#[allow(clippy::too_many_arguments)]
pub fn dit_forward(
    cfg: &DitConfig,
    wm: &WeightMap,
    tokens: &[f32],
    coords: &[f32],
    n_pos: usize,
    cond: &[f32],
    n_cond: usize,
    t: f32,
    dump: Option<&mut DitDump>,
) -> Result<Vec<f32>> {
    if n_pos == 0 {
        anyhow::bail!("dit_forward requires n_pos > 0");
    }
    let c = cfg.args.model_channels;
    let heads = cfg.num_heads();
    let hd = cfg.head_dim();
    let in_ch = cfg.args.in_channels;
    let out_ch = cfg.args.out_channels;
    let cond_ch = cfg.args.cond_channels;
    let base = cfg.args.rope_freq;

    // input_layer: [in_ch -> C]
    let in_w = get(wm, "input_layer.weight")?;
    let in_b = get(wm, "input_layer.bias")?;
    let mut h = linear_raw(tokens, n_pos, in_ch, in_w, c, Some(in_b));
    if let Some(d) = dump.as_deref() {
        let _ = d;
    }
    let after_input = h.clone();

    let t_mod = shared_modulation(cfg, wm, t)?;

    // precompute RoPE angle tables (interleaved reference) via coords.
    let mut after_block0 = Vec::new();
    for blk in 0..cfg.args.num_blocks {
        let p = format!("blocks.{blk}");
        let bw = BlockW {
            modulation: get(wm, &format!("{p}.modulation"))?,
            norm2_w: get(wm, &format!("{p}.norm2.weight"))?,
            norm2_b: get(wm, &format!("{p}.norm2.bias"))?,
            qkv_w: get(wm, &format!("{p}.self_attn.to_qkv.weight"))?,
            qkv_b: get(wm, &format!("{p}.self_attn.to_qkv.bias"))?,
            sa_q_gamma: get(wm, &format!("{p}.self_attn.q_rms_norm.gamma"))?,
            sa_k_gamma: get(wm, &format!("{p}.self_attn.k_rms_norm.gamma"))?,
            sa_out_w: get(wm, &format!("{p}.self_attn.to_out.weight"))?,
            sa_out_b: get(wm, &format!("{p}.self_attn.to_out.bias"))?,
            ca_q_w: get(wm, &format!("{p}.cross_attn.to_q.weight"))?,
            ca_q_b: get(wm, &format!("{p}.cross_attn.to_q.bias"))?,
            ca_kv_w: get(wm, &format!("{p}.cross_attn.to_kv.weight"))?,
            ca_kv_b: get(wm, &format!("{p}.cross_attn.to_kv.bias"))?,
            ca_q_gamma: get(wm, &format!("{p}.cross_attn.q_rms_norm.gamma"))?,
            ca_k_gamma: get(wm, &format!("{p}.cross_attn.k_rms_norm.gamma"))?,
            ca_out_w: get(wm, &format!("{p}.cross_attn.to_out.weight"))?,
            ca_out_b: get(wm, &format!("{p}.cross_attn.to_out.bias"))?,
            mlp0_w: get(wm, &format!("{p}.mlp.mlp.0.weight"))?,
            mlp0_b: get(wm, &format!("{p}.mlp.mlp.0.bias"))?,
            mlp2_w: get(wm, &format!("{p}.mlp.mlp.2.weight"))?,
            mlp2_b: get(wm, &format!("{p}.mlp.mlp.2.bias"))?,
        };
        block_forward(
            &bw, &mut h, &t_mod, coords, n_pos, cond, n_cond, c, heads, hd, cond_ch, base,
        );
        if blk == 0 {
            after_block0 = h.clone();
        }
    }

    // final non-affine LN + out_layer
    let after_final_ln = layer_norm_noaffine(&h, n_pos, c);
    let out_w = get(wm, "out_layer.weight")?;
    let out_b = get(wm, "out_layer.bias")?;
    let out = linear_raw(&after_final_ln, n_pos, c, out_w, out_ch, Some(out_b));

    if let Some(d) = dump {
        d.after_input = after_input;
        d.after_block0 = after_block0;
        d.after_final_ln = after_final_ln.clone();
        d.out = out.clone();
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn block_forward(
    bw: &BlockW,
    x: &mut [f32],
    t_mod: &[f32],
    coords: &[f32],
    n_pos: usize,
    cond: &[f32],
    n_cond: usize,
    c: usize,
    heads: usize,
    hd: usize,
    cond_ch: usize,
    base: (f32, f32),
) {
    // modulation: (block.modulation + t_mod).chunk(6)
    let mut m = vec![0.0f32; 6 * c];
    for i in 0..6 * c {
        m[i] = bw.modulation[i] + t_mod[i];
    }
    let (shift_msa, rest) = m.split_at(c);
    let (scale_msa, rest) = rest.split_at(c);
    let (gate_msa, rest) = rest.split_at(c);
    let (shift_mlp, rest) = rest.split_at(c);
    let (scale_mlp, gate_mlp) = rest.split_at(c);

    // --- self-attention ---
    let mut h = layer_norm_noaffine(x, n_pos, c);
    modulate(&mut h, n_pos, c, scale_msa, shift_msa);
    let qkv = linear_raw(&h, n_pos, c, bw.qkv_w, 3 * c, Some(bw.qkv_b));
    // split q/k/v: qkv row is [3, heads, hd] -> q=[0..C], k=[C..2C], v=[2C..3C]
    let mut q = vec![0.0f32; n_pos * c];
    let mut k = vec![0.0f32; n_pos * c];
    let mut v = vec![0.0f32; n_pos * c];
    for p in 0..n_pos {
        let row = &qkv[p * 3 * c..(p + 1) * 3 * c];
        q[p * c..(p + 1) * c].copy_from_slice(&row[0..c]);
        k[p * c..(p + 1) * c].copy_from_slice(&row[c..2 * c]);
        v[p * c..(p + 1) * c].copy_from_slice(&row[2 * c..3 * c]);
    }
    multihead_rms_norm(&mut q, n_pos, heads, hd, bw.sa_q_gamma);
    multihead_rms_norm(&mut k, n_pos, heads, hd, bw.sa_k_gamma);
    rope::apply_interleaved_rope(&mut q, coords, n_pos, heads, hd, 3, base);
    rope::apply_interleaved_rope(&mut k, coords, n_pos, heads, hd, 3, base);
    let attn = mha(&q, &k, &v, n_pos, n_pos, heads, hd);
    let mut sa = linear_raw(&attn, n_pos, c, bw.sa_out_w, c, Some(bw.sa_out_b));
    gate_and_add(x, &mut sa, n_pos, c, gate_msa);

    // --- cross-attention ---
    let h2 = layer_norm_affine(x, n_pos, c, bw.norm2_w, bw.norm2_b);
    let mut cq = linear_raw(&h2, n_pos, c, bw.ca_q_w, c, Some(bw.ca_q_b));
    let ckv = linear_raw(cond, n_cond, cond_ch, bw.ca_kv_w, 2 * c, Some(bw.ca_kv_b));
    let mut ck = vec![0.0f32; n_cond * c];
    let mut cv = vec![0.0f32; n_cond * c];
    for p in 0..n_cond {
        let row = &ckv[p * 2 * c..(p + 1) * 2 * c];
        ck[p * c..(p + 1) * c].copy_from_slice(&row[0..c]);
        cv[p * c..(p + 1) * c].copy_from_slice(&row[c..2 * c]);
    }
    multihead_rms_norm(&mut cq, n_pos, heads, hd, bw.ca_q_gamma);
    multihead_rms_norm(&mut ck, n_cond, heads, hd, bw.ca_k_gamma);
    let cattn = mha(&cq, &ck, &cv, n_pos, n_cond, heads, hd);
    let ca = linear_raw(&cattn, n_pos, c, bw.ca_out_w, c, Some(bw.ca_out_b));
    for i in 0..n_pos * c {
        x[i] += ca[i];
    }

    // --- mlp ---
    let mut h3 = layer_norm_noaffine(x, n_pos, c);
    modulate(&mut h3, n_pos, c, scale_mlp, shift_mlp);
    let hidden = bw.mlp0_w.len() / c;
    let mut up = linear_raw(&h3, n_pos, c, bw.mlp0_w, hidden, Some(bw.mlp0_b));
    gelu_tanh(&mut up);
    let mut down = linear_raw(&up, n_pos, hidden, bw.mlp2_w, c, Some(bw.mlp2_b));
    gate_and_add(x, &mut down, n_pos, c, gate_mlp);
}

/// `x = x·(1+scale) + shift` per channel.
fn modulate(x: &mut [f32], rows: usize, dim: usize, scale: &[f32], shift: &[f32]) {
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        for i in 0..dim {
            row[i] = row[i] * (1.0 + scale[i]) + shift[i];
        }
    }
}

/// `x += gate ⊙ h` per channel.
fn gate_and_add(x: &mut [f32], h: &mut [f32], rows: usize, dim: usize, gate: &[f32]) {
    for r in 0..rows {
        for i in 0..dim {
            let idx = r * dim + i;
            x[idx] += gate[i] * h[idx];
        }
    }
}

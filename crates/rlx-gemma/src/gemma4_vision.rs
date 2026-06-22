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

//! Gemma 4 vision tower (`model.vision_tower`) — SigLIP-style 16-layer encoder.
//!
//! Ground truth (HF `Gemma4VisionModel`): hidden 768, 16 layers, 12 heads,
//! head_dim 64, intermediate 3072, gelu_pytorch_tanh, eps 1e-6, **attention
//! scaling = 1.0** (folded into per-head q/k RMSNorm), **2-D RoPE** (theta 100),
//! sandwich norms (input / post_attn / pre_ffn / post_ffn), no post-encoder
//! norm. RMSNorm is the plain `* weight` convention (the loader returns `w-1`
//! so the shared `gemma_rms`'s `1+(w-1)` reproduces it). v_norm has no weight.
//!
//! 2-D RoPE: head_dim 64 split into two 32-wide halves; the first rotates with
//! the patch x-coordinate, the second with y. Per HF, `cos`/`sin` are the
//! `cat(freqs,freqs)` per axis concatenated → width 64, and each 32-half is
//! rotated by the standard `rotate_half`. We precompute the per-patch
//! `[P,64]` cos/sin on the host (they depend only on the patch grid, like the
//! text PLE inputs) and feed them as graph inputs; the rotation itself is
//! expressed with narrow/neg/concat/mul/add so it runs on every backend.
//!
//! Quantization: vision linears are 8-bit (`...{proj}.linear.weight` + scale),
//! handled by [`crate::qat_loader::GemmaQatLoader`]; patch_embedder + projector
//! are unquantized (BF16/F32).

use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use std::collections::HashMap;

/// Fixed Gemma 4 vision-tower hyperparameters (from `vision_config`).
#[derive(Debug, Clone, Copy)]
pub struct VisionConfig {
    pub hidden: usize,       // 768
    pub layers: usize,       // 16
    pub heads: usize,        // 12
    pub head_dim: usize,     // 64
    pub intermediate: usize, // 3072
    pub eps: f32,            // 1e-6
    pub rope_theta: f64,     // 100.0
    pub lm_hidden: usize,    // 1536 (projector output)
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden: 768,
            layers: 16,
            heads: 12,
            head_dim: 64,
            intermediate: 3072,
            eps: 1e-6,
            rope_theta: 100.0,
            lm_hidden: 1536,
        }
    }
}

/// 2-D RoPE inverse frequencies for the vision tower: `spatial_dim = head_dim/2`
/// (=32), `inv_freq[i] = theta^(-2i/spatial_dim)` for `i in 0..spatial_dim/2`
/// (=16 freqs). Used by the host to build per-patch cos/sin.
pub fn vision_inv_freq(cfg: &VisionConfig) -> Vec<f64> {
    let spatial = cfg.head_dim / 2; // 32
    (0..spatial)
        .step_by(2)
        .map(|i| 1.0 / cfg.rope_theta.powf(i as f64 / spatial as f64))
        .collect()
}

/// Build the per-patch 2-D RoPE cos/sin tables `[num_patches, head_dim]`
/// (flattened) for patch grid positions `pos[(x,y)]`. For each patch the
/// 64-wide row is `[ x: cos(x·f)⊗2(16) | y: cos(y·f)⊗2(16) ]` (and sin),
/// matching HF `apply_multidimensional_rope` (cat(freqs,freqs) per axis).
pub fn vision_rope_tables(cfg: &VisionConfig, positions: &[(u32, u32)]) -> (Vec<f32>, Vec<f32>) {
    let inv = vision_inv_freq(cfg); // 16 freqs
    let hd = cfg.head_dim; // 64
    let half = hd / 2; // 32
    let nf = inv.len(); // 16
    let p = positions.len();
    let mut cos = vec![0f32; p * hd];
    let mut sin = vec![0f32; p * hd];
    for (pi, &(x, y)) in positions.iter().enumerate() {
        let base = pi * hd;
        for (axis, coord) in [x, y].into_iter().enumerate() {
            let off = base + axis * half; // x-half at [0..32], y-half at [32..64]
            for j in 0..nf {
                let angle = coord as f64 * inv[j];
                let (c, s) = (angle.cos() as f32, angle.sin() as f32);
                // cat(freqs, freqs): entries j and j+nf share the angle.
                cos[off + j] = c;
                cos[off + nf + j] = c;
                sin[off + j] = s;
                sin[off + nf + j] = s;
            }
        }
    }
    (cos, sin)
}

/// Apply the vision 2-D RoPE in-graph to a packed `[B, P, heads*head_dim]`
/// tensor. `cos`/`sin` are graph inputs of shape `[1, P, head_dim]` (broadcast
/// over batch + heads). Implements `x*cos + rotate2d(x)*sin` where `rotate2d`
/// does `rotate_half` independently on each 32-wide half.
fn apply_vision_rope_2d(
    g: &mut Graph,
    x: NodeId, // [B, P, heads*hd]
    cos: NodeId,
    sin: NodeId,
    batch: usize,
    p: usize,
    heads: usize,
    hd: usize, // 64
    f: DType,
) -> NodeId {
    let half = hd / 2; // 32
    let quarter = half / 2; // 16
    // [B,P,heads,hd]
    let x4 = g.reshape_(x, vec![batch as i64, p as i64, heads as i64, hd as i64]);
    // cos/sin: [1,P,hd] → [1,P,1,hd] to broadcast over heads.
    let cos4 = g.reshape_(cos, vec![1, p as i64, 1, hd as i64]);
    let sin4 = g.reshape_(sin, vec![1, p as i64, 1, hd as i64]);

    // rotate2d(x): for each 32-half, rotate_half = cat(-x[16:32], x[0:16]).
    // x-half lives at [0:32], y-half at [32:64].
    let sh = Shape::new(&[batch, p, heads, quarter], f);
    let nx = |g: &mut Graph, start: usize| g.narrow_(x4, 3, start, quarter);
    let x_lo = nx(g, 0); // x-half first 16
    let x_hi = nx(g, quarter); // x-half second 16
    let y_lo = nx(g, half); // y-half first 16
    let y_hi = nx(g, half + quarter); // y-half second 16
    let neg = |g: &mut Graph, n: NodeId| {
        g.add_node(
            rlx_ir::op::Op::Activation(rlx_ir::op::Activation::Neg),
            vec![n],
            sh.clone(),
        )
    };
    let nx_hi = neg(g, x_hi);
    let ny_hi = neg(g, y_hi);
    // rot = [ -x_hi, x_lo, -y_hi, y_lo ]
    let rot = g.concat_(vec![nx_hi, x_lo, ny_hi, y_lo], 3); // [B,P,heads,hd]

    let xc = g.mul(x4, cos4);
    let rs = g.mul(rot, sin4);
    let out = g.add(xc, rs);
    g.reshape_(out, vec![batch as i64, p as i64, (heads * hd) as i64])
}

/// Build the vision encoder graph: patch-embed → 16 encoder layers, returning
/// the encoder `last_hidden_state` `[B, P, 768]` (pre-pooling). Inputs:
/// `vision_pixels` `[B,P,3*patch^2=768]` (raw [0,1] patch pixels),
/// `vision_pos_embed` `[B,P,768]` (host-precomputed x+y position-embedding sum),
/// `vision_rope_cos`/`vision_rope_sin` `[1,P,64]`. Bidirectional attention
/// (no causal mask) over all patches.
pub fn build_vision_encoder(
    cfg: &VisionConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    num_patches: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("gemma4_vision");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let hs = build_encoder_core(&mut g, &mut params, cfg, weights, batch, num_patches)?;
    g.set_outputs(vec![hs]);
    Ok((g, params))
}

fn load_t(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    w: &mut dyn WeightLoader,
    key: &str,
) -> Result<NodeId> {
    let (data, shape) = w.take_transposed(key)?;
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok(id)
}
fn load_v(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    w: &mut dyn WeightLoader,
    key: &str,
    shape: &[usize],
) -> Result<NodeId> {
    let (data, sh) = w.take(key)?;
    debug_assert_eq!(&sh, shape, "{key} shape");
    let id = g.param(key, Shape::new(&sh, DType::F32));
    params.insert(key.to_string(), data);
    Ok(id)
}
// Gemma RMSNorm (delta-gamma; loader returns w-1 → 1+(w-1)=w plain weight).
fn vrms(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    key: &str,
    w: &mut dyn WeightLoader,
    dim: usize,
    eps: f32,
) -> Result<NodeId> {
    let wv = load_v(g, params, w, key, &[dim])?;
    let ones = {
        let id = g.param(format!("{key}.ones"), Shape::new(&[dim], DType::F32));
        params.insert(format!("{key}.ones"), vec![1.0f32; dim]);
        id
    };
    let gamma = g.add(ones, wv);
    let beta = {
        let id = g.param(format!("{key}.beta"), Shape::new(&[dim], DType::F32));
        params.insert(format!("{key}.beta"), vec![0.0f32; dim]);
        id
    };
    Ok(g.rms_norm(x, gamma, beta, eps))
}
// RMSNorm with no learned scale (gamma=1, beta=0) — `with_scale=False` modules.
fn vrms_noscale(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    name: &str,
    dim: usize,
    eps: f32,
) -> NodeId {
    let gamma = synth(
        g,
        params,
        &format!("{name}.ones"),
        vec![1.0f32; dim],
        &[dim],
    );
    let beta = synth(
        g,
        params,
        &format!("{name}.beta"),
        vec![0.0f32; dim],
        &[dim],
    );
    g.rms_norm(x, gamma, beta, eps)
}

/// Build patch-embed → 16 encoder layers, returning the `last_hidden_state`
/// node `[B,P,768]`. Adds the `vision_pixels`/`vision_pos_embed`/
/// `vision_rope_cos`/`vision_rope_sin` graph inputs. Does **not** set outputs.
fn build_encoder_core(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &VisionConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    num_patches: usize,
) -> Result<NodeId> {
    let f = DType::F32;
    let (h, nh, hd, eps) = (cfg.hidden, cfg.heads, cfg.head_dim, cfg.eps);
    let p = num_patches;

    let pixels = g.input("vision_pixels", Shape::new(&[batch, p, 3 * 16 * 16], f));
    let pos_embed = g.input("vision_pos_embed", Shape::new(&[batch, p, h], f));
    let rope_cos = g.input("vision_rope_cos", Shape::new(&[1, p, hd], f));
    let rope_sin = g.input("vision_rope_sin", Shape::new(&[1, p, hd], f));

    // Patch embed: scale [0,1]→[-1,1], input_proj, + position embedding.
    let two = synth(g, params, "vis.two", vec![2.0], &[1]);
    let one = synth(g, params, "vis.one", vec![1.0], &[1]);
    let px2 = g.mul(pixels, two);
    let scaled = g.sub(px2, one); // 2x - 1
    let ip = load_t(
        g,
        params,
        weights,
        "model.vision_tower.patch_embedder.input_proj.weight",
    )?;
    let mut hs = g.mm(scaled, ip); // [B,P,768]
    hs = g.add(hs, pos_embed);

    for layer in 0..cfg.layers {
        let lp = format!("model.vision_tower.encoder.layers.{layer}");
        // input_layernorm → attn → post_attention_layernorm → +res
        let normed = vrms(
            g,
            params,
            hs,
            &format!("{lp}.input_layernorm.weight"),
            weights,
            h,
            eps,
        )?;
        let q = {
            let w = load_t(
                g,
                params,
                weights,
                &format!("{lp}.self_attn.q_proj.linear.weight"),
            )?;
            g.mm(normed, w)
        };
        let k = {
            let w = load_t(
                g,
                params,
                weights,
                &format!("{lp}.self_attn.k_proj.linear.weight"),
            )?;
            g.mm(normed, w)
        };
        let v = {
            let w = load_t(
                g,
                params,
                weights,
                &format!("{lp}.self_attn.v_proj.linear.weight"),
            )?;
            g.mm(normed, w)
        };
        // per-head q/k norm (with weight), v norm (no weight).
        let q = per_head_norm(
            g,
            params,
            q,
            &format!("{lp}.self_attn.q_norm.weight"),
            weights,
            batch,
            p,
            nh,
            hd,
            eps,
            true,
        )?;
        let k = per_head_norm(
            g,
            params,
            k,
            &format!("{lp}.self_attn.k_norm.weight"),
            weights,
            batch,
            p,
            nh,
            hd,
            eps,
            true,
        )?;
        let v = per_head_norm(
            g,
            params,
            v,
            &format!("{lp}.self_attn.v_norm"),
            weights,
            batch,
            p,
            nh,
            hd,
            eps,
            false,
        )?;
        // 2-D RoPE on q, k.
        let q = apply_vision_rope_2d(g, q, rope_cos, rope_sin, batch, p, nh, hd, f);
        let k = apply_vision_rope_2d(g, k, rope_cos, rope_sin, batch, p, nh, hd, f);
        // bidirectional attention, scale = 1.0 (folded into q/k norm).
        let attn_shape = rlx_ir::shape::attention_shape(g.shape(q));
        let attn = g.attention_kind_opts(
            q,
            k,
            v,
            nh,
            hd,
            rlx_ir::op::MaskKind::None,
            attn_shape,
            Some(1.0),
            None,
        );
        let o = {
            let w = load_t(
                g,
                params,
                weights,
                &format!("{lp}.self_attn.o_proj.linear.weight"),
            )?;
            g.mm(attn, w)
        };
        let o = vrms(
            g,
            params,
            o,
            &format!("{lp}.post_attention_layernorm.weight"),
            weights,
            h,
            eps,
        )?;
        hs = g.add(hs, o);
        // pre_ffn → gelu-glu MLP → post_ffn → +res
        let normed = vrms(
            g,
            params,
            hs,
            &format!("{lp}.pre_feedforward_layernorm.weight"),
            weights,
            h,
            eps,
        )?;
        let gate = {
            let w = load_t(
                g,
                params,
                weights,
                &format!("{lp}.mlp.gate_proj.linear.weight"),
            )?;
            g.mm(normed, w)
        };
        let up = {
            let w = load_t(
                g,
                params,
                weights,
                &format!("{lp}.mlp.up_proj.linear.weight"),
            )?;
            g.mm(normed, w)
        };
        let gact = g.gelu_approx(gate);
        let inner = g.mul(gact, up);
        let down = {
            let w = load_t(
                g,
                params,
                weights,
                &format!("{lp}.mlp.down_proj.linear.weight"),
            )?;
            g.mm(inner, w)
        };
        let down = vrms(
            g,
            params,
            down,
            &format!("{lp}.post_feedforward_layernorm.weight"),
            weights,
            h,
            eps,
        )?;
        hs = g.add(hs, down);
    }

    Ok(hs)
}

/// Build the host-side 2-D average-pooling matrix `[L, P]` for `positions`:
/// each output token `l` averages the `k×k` block of patches whose grid
/// position floors to it. Mirrors HF `_avg_pool_by_positions`
/// (`block = x//k + (max_x//k)*(y//k)`, weight `1/k²`). Returns `(weights, L)`.
pub fn vision_pool_weights(positions: &[(u32, u32)], k: usize) -> (Vec<f32>, usize) {
    let p = positions.len();
    let max_x = positions.iter().map(|&(x, _)| x).max().unwrap_or(0) as usize + 1;
    let bx = max_x / k; // blocks per row
    let l = p / (k * k);
    let inv = 1.0f32 / (k * k) as f32;
    let mut w = vec![0f32; l * p];
    for (i, &(x, y)) in positions.iter().enumerate() {
        let block = (x as usize / k) + bx * (y as usize / k);
        w[block * p + i] = inv;
    }
    (w, l)
}

/// Build the full image-feature path for a single image: encoder →
/// `k×k` spatial average pool → `×√hidden` → `embed_vision`
/// (pre-projection RMSNorm with no scale → `embedding_projection` 768→1536).
/// Returns a graph whose single output is the soft-token embeddings
/// `[1, L, lm_hidden]` that splice into the LM at the image placeholders.
/// `positions` are the per-patch `(x,y)` grid coords (same order as the
/// `vision_pixels`/`vision_pos_embed` rows); `pooling_kernel` is config
/// `pooling_kernel_size` (3). Feed the same inputs as the encoder.
pub fn build_vision_features(
    cfg: &VisionConfig,
    weights: &mut dyn WeightLoader,
    positions: &[(u32, u32)],
    pooling_kernel: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("gemma4_vision_features");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let p = positions.len();
    let h = cfg.hidden;
    let hs = build_encoder_core(&mut g, &mut params, cfg, weights, 1, p)?; // [1,P,h]
    let hs2d = g.reshape_(hs, vec![p as i64, h as i64]); // [P,h]

    // 2-D average pool: pooled[L,h] = poolW[L,P] @ hs[P,h].
    let (pw, l) = vision_pool_weights(positions, pooling_kernel);
    let pool = synth(&mut g, &mut params, "vis.pool_w", pw, &[l, p]);
    let pooled = g.mm(pool, hs2d); // [L,h]

    // ×√hidden (the pooler's float32 magnitude expansion).
    let root = synth(
        &mut g,
        &mut params,
        "vis.root_h",
        vec![(h as f32).sqrt()],
        &[1],
    );
    let scaled = g.mul(pooled, root);

    // embed_vision: pre-projection RMSNorm (no learned scale) → linear 768→1536.
    let normed = vrms_noscale(
        &mut g,
        &mut params,
        scaled,
        "embed_vision.pre_norm",
        h,
        cfg.eps,
    );
    let proj = load_t(
        &mut g,
        &mut params,
        weights,
        "model.embed_vision.embedding_projection.weight",
    )?;
    let feats = g.mm(normed, proj); // [L, lm_hidden]
    let out = g.reshape_(feats, vec![1, l as i64, cfg.lm_hidden as i64]);
    g.set_outputs(vec![out]);
    Ok((g, params))
}

fn synth(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: Vec<f32>,
    shape: &[usize],
) -> NodeId {
    let id = g.param(name, Shape::new(shape, DType::F32));
    params.insert(name.to_string(), data);
    id
}

/// Per-head RMSNorm over `head_dim` of a packed `[B,P,heads*hd]` tensor.
/// `with_weight=false` (v_norm) → gamma=1 (no learned scale).
#[allow(clippy::too_many_arguments)]
fn per_head_norm(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    key: &str,
    weights: &mut dyn WeightLoader,
    batch: usize,
    p: usize,
    heads: usize,
    hd: usize,
    eps: f32,
    with_weight: bool,
) -> Result<NodeId> {
    let x4 = g.reshape_(x, vec![batch as i64, p as i64, heads as i64, hd as i64]);
    let gamma = if with_weight {
        let (data, _) = weights.take(key)?;
        let ones = vec![1.0f32; hd];
        // loader already returned w-1 for norm weights → gamma = 1 + (w-1) = w.
        let g_id = g.param(key, Shape::new(&[hd], DType::F32));
        params.insert(key.to_string(), data);
        let ones_id = synth(g, params, &format!("{key}.ones"), ones, &[hd]);
        g.add(ones_id, g_id)
    } else {
        synth(g, params, &format!("{key}.ones"), vec![1.0f32; hd], &[hd])
    };
    let beta = synth(g, params, &format!("{key}.vbeta"), vec![0.0f32; hd], &[hd]);
    let normed = g.rms_norm(x4, gamma, beta, eps);
    Ok(g.reshape_(normed, vec![batch as i64, p as i64, (heads * hd) as i64]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_shape_and_pos0_identity() {
        let cfg = VisionConfig::default();
        let positions = vec![(0u32, 0u32), (1, 0), (0, 1), (3, 5)];
        let (cos, sin) = vision_rope_tables(&cfg, &positions);
        assert_eq!(cos.len(), positions.len() * cfg.head_dim);
        assert_eq!(sin.len(), cos.len());
        // pos (0,0) → all angles 0 → cos=1, sin=0.
        for j in 0..cfg.head_dim {
            assert!((cos[j] - 1.0).abs() < 1e-6);
            assert!(sin[j].abs() < 1e-6);
        }
        // cat(freqs,freqs): within the x-half, entry j and j+16 share the angle.
        let row = cfg.head_dim; // patch 1 = (1,0)
        let nf = 16;
        assert!((cos[row] - cos[row + nf]).abs() < 1e-6);
    }
}

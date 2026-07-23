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

//! Qwen3-VL vision tower HIR builder — mirrors `tools/mtmd/models/qwen3vl.cpp`.

use super::config::MmProjConfig;
use super::preprocess::build_spatial_merge_gather_idx;
use super::weights::{DeepstackWeights, MmProjWeights, VisionBlockWeights};
use anyhow::{anyhow, Result};
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use std::collections::HashMap;

type NodeId = HirNodeId;

/// Build the Qwen3-VL vision encoder HIR for a fixed `(img_w, img_h)`.
pub fn build_qwen35_vision_hir(
    cfg: &MmProjConfig,
    weights: &MmProjWeights,
    img_w: usize,
    img_h: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_dims(cfg, img_w, img_h)?;
    let mut hir = HirModule::new("qwen35_vision").with_fusion_policy(FusionPolicy::Direct);
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let batch = 1usize;
    let n = cfg.n_embd;
    let n_ff = cfg.n_ff;
    let nh = cfg.n_head;
    let dh = n / nh;
    let eps = cfg.eps as f32;
    let ps = cfg.patch_size;
    let hp = img_h / ps;
    let wp = img_w / ps;
    let n_pos = hp * wp;
    let merge_sq = cfg.n_merge * cfg.n_merge;
    let n_out = n_pos / merge_sq;
    let proj = cfg.llm_hidden_size;

    let image = hir.input("image", Shape::new(&[batch, 3, img_h, img_w], f));
    let rope_cos = hir.input("vision_rope_cos", Shape::new(&[n_pos, dh], f));
    let rope_sin = hir.input("vision_rope_sin", Shape::new(&[n_pos, dh], f));

    let mut g = HirMut::new(&mut hir);

    // Dual patch conv (stride = patch_size).
    let w0 = param_conv(
        &mut g,
        &mut params,
        "patch_embd.0",
        &weights.patch_embd_0,
        n,
        3,
        ps,
    )?;
    let w1 = param_conv(
        &mut g,
        &mut params,
        "patch_embd.1",
        &weights.patch_embd_1,
        n,
        3,
        ps,
    )?;
    let out_h = img_h / ps;
    let out_w = img_w / ps;
    let conv_shape = Shape::new(&[batch, n, out_h, out_w], f);
    let c0 = g.conv2d(image, w0, [ps, ps], [ps, ps], [0, 0], 1, conv_shape.clone());
    let c1 = g.conv2d(image, w1, [ps, ps], [ps, ps], [0, 0], 1, conv_shape);
    let patches = g.add(c0, c1);

    // [B,C,H,W] → [B, n_pos, n_embd] (raster order), then arrange each
    // merge block contiguously before the transformer and projector.
    let seq = flatten_nchw_to_bsn(&mut g, patches, batch, n, out_h, out_w);
    let merge_idx = build_spatial_merge_gather_idx(hp, wp, cfg.n_merge);
    let merge_idx_id = param_vec(&mut g, &mut params, "spatial_merge_idx", &merge_idx, n_pos);
    let merge_idx_2d = g.reshape_(merge_idx_id, vec![batch as i64, n_pos as i64]);
    let seq = g.gather_(seq, merge_idx_2d, 1);

    // Patch bias + learned absolute position embedding (host-resized to n_pos).
    let bias = param_vec(&mut g, &mut params, "patch_bias", &weights.patch_bias, n);
    let bias_bsn = broadcast_bias_bsn(&mut g, bias, batch, n_pos, n);
    let mut h_id = g.add(seq, bias_bsn);

    // `pos` is row-major [n_pos, n] (token-major). Keep that layout —
    // do not treat it as [n, n_pos] then transpose (that scrambles dims).
    let pos = reorder_position_embd(resize_position_embd(cfg, weights, hp, wp), &merge_idx, n);
    let pos_w = param_mat(&mut g, &mut params, "position_embd", &pos, n_pos, n)?;
    let pos_bsn = g.reshape_(pos_w, vec![batch as i64, n_pos as i64, n as i64]);
    h_id = g.add(h_id, pos_bsn);

    // Pre layer-norm (ViT-style LN, not RMS). HF Qwen3.5 has none — skip.
    if weights.has_pre_ln {
        let pre_w = param_vec(&mut g, &mut params, "pre_ln.weight", &weights.pre_ln_w, n);
        let pre_b = param_vec(&mut g, &mut params, "pre_ln.bias", &weights.pre_ln_b, n);
        h_id = g.ln(h_id, pre_w, pre_b, eps);
    }

    // Full-attention mask (bidirectional).
    let mask = param_mask(&mut g, &mut params, "attn_mask", batch, n_pos);

    let mut deepstack_outs: Vec<NodeId> = Vec::new();

    for (il, blk) in weights.blocks.iter().enumerate() {
        h_id = build_encoder_block(
            &mut g,
            &mut params,
            il,
            blk,
            h_id,
            mask,
            batch,
            n_pos,
            n,
            n_ff,
            nh,
            dh,
            eps,
            merge_sq,
            rope_cos,
            rope_sin,
        )?;
        if let Some(ds) = &blk.deepstack {
            let feat = build_deepstack_branch(
                &mut g,
                &mut params,
                il,
                ds,
                h_id,
                batch,
                n_pos,
                n,
                n_ff,
                eps,
                merge_sq,
            )?;
            deepstack_outs.push(feat);
        }
    }

    if weights.has_post_ln {
        let post_w = param_vec(&mut g, &mut params, "post_ln.weight", &weights.post_ln_w, n);
        let post_b = param_vec(&mut g, &mut params, "post_ln.bias", &weights.post_ln_b, n);
        h_id = g.ln(h_id, post_w, post_b, eps);
    }

    // HF merger normalizes patch embeddings before the spatial reshape.
    let mm_norm_w = param_vec(&mut g, &mut params, "mm_norm.weight", &weights.mm_norm_w, n);
    let mm_norm_b = param_vec(&mut g, &mut params, "mm_norm.bias", &weights.mm_norm_b, n);
    h_id = g.ln(h_id, mm_norm_w, mm_norm_b, eps);

    // MM projector: spatial merge along features, then GELU FFN.
    let merged = merge_spatial_tokens(&mut g, h_id, batch, n_pos, n, merge_sq);
    let mm0_w = param_mat(
        &mut g,
        &mut params,
        "mm.0.weight",
        &transpose_2d(&weights.mm_0_w, n_ff, n * merge_sq),
        n * merge_sq,
        n_ff,
    )?;
    let mm0_b = param_vec(&mut g, &mut params, "mm.0.bias", &weights.mm_0_b, n_ff);
    let mm0_mm = g.mm(merged, mm0_w);
    let mm0 = g.add(mm0_mm, mm0_b);
    let mm0_act = g.gelu(mm0);

    let mm1_w = param_mat(
        &mut g,
        &mut params,
        "mm.1.weight",
        &transpose_2d(&weights.mm_1_w, proj, n_ff),
        n_ff,
        proj,
    )?;
    let mm1_b = param_vec(&mut g, &mut params, "mm.1.bias", &weights.mm_1_b, proj);
    let mm1_mm = g.mm(mm0_act, mm1_w);
    let mut embeddings = g.add(mm1_mm, mm1_b);

    if !deepstack_outs.is_empty() {
        let mut cat_in = vec![embeddings];
        cat_in.extend(deepstack_outs);
        embeddings = g.concat_(cat_in, 2);
    }

    // Output [batch, n_out, llm_hidden_size].
    let out = g.reshape_(embeddings, vec![batch as i64, n_out as i64, proj as i64]);
    g.set_outputs(vec![out]);

    let _ = n_out; // used in reshape
    Ok((hir, params))
}

pub fn build_qwen35_vision_graph(
    cfg: &MmProjConfig,
    weights: &MmProjWeights,
    img_w: usize,
    img_h: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    rlx_core::flow_util::graph_from_built(crate::vision::flow::build_qwen35_vision_built(
        cfg, weights, img_w, img_h,
    )?)
}

fn validate_dims(cfg: &MmProjConfig, img_w: usize, img_h: usize) -> Result<()> {
    let ps = cfg.patch_size;
    let m = cfg.n_merge;
    if !img_w.is_multiple_of(ps * 2) || !img_h.is_multiple_of(ps * 2) {
        return Err(anyhow!(
            "vision: image {img_w}x{img_h} must be divisible by patch_size*2={}",
            ps * 2
        ));
    }
    let n_pos = (img_w / ps) * (img_h / ps);
    if !n_pos.is_multiple_of(m * m) {
        return Err(anyhow!("vision: n_pos={n_pos} not divisible by merge²"));
    }
    Ok(())
}

fn build_encoder_block(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    il: usize,
    blk: &VisionBlockWeights,
    h_in: NodeId,
    _mask: NodeId,
    batch: usize,
    seq: usize,
    n: usize,
    n_ff: usize,
    nh: usize,
    dh: usize,
    eps: f32,
    _merge_sq: usize,
    rope_cos: NodeId,
    rope_sin: NodeId,
) -> Result<NodeId> {
    let p = format!("blk.{il}");
    let ln1_w = param_vec(g, params, &format!("{p}.ln1.weight"), &blk.ln1_w, n);
    let ln1_b = param_vec(g, params, &format!("{p}.ln1.bias"), &blk.ln1_b, n);
    let x = g.ln(h_in, ln1_w, ln1_b, eps);

    let x2d = g.reshape_(x, vec![(batch * seq) as i64, n as i64]);
    let qkv_w = param_mat(
        g,
        params,
        &format!("{p}.attn_qkv.weight"),
        &transpose_2d(&blk.qkv_w, 3 * n, n),
        n,
        3 * n,
    )?;
    let qkv_b = param_vec(g, params, &format!("{p}.attn_qkv.bias"), &blk.qkv_b, 3 * n);
    let qkv_mm = g.mm(x2d, qkv_w);
    let qkv = g.add(qkv_mm, qkv_b);
    let qkv_4d = g.reshape_(qkv, vec![batch as i64, seq as i64, 3 * n as i64]);
    let q = g.narrow_(qkv_4d, 2, 0, n);
    let k = g.narrow_(qkv_4d, 2, n, n);
    let v = g.narrow_(qkv_4d, 2, 2 * n, n);
    let q = apply_vision_rope(g, q, rope_cos, rope_sin, batch, seq, nh, dh);
    let k = apply_vision_rope(g, k, rope_cos, rope_sin, batch, seq, nh, dh);

    let attn = g.attention_kind(
        q,
        k,
        v,
        nh,
        dh,
        MaskKind::None,
        Shape::new(&[batch, seq, n], DType::F32),
    );
    let attn2d = g.reshape_(attn, vec![(batch * seq) as i64, n as i64]);
    let out_w = param_mat(
        g,
        params,
        &format!("{p}.attn_out.weight"),
        &transpose_2d(&blk.attn_out_w, n, n),
        n,
        n,
    )?;
    let out_b = param_vec(g, params, &format!("{p}.attn_out.bias"), &blk.attn_out_b, n);
    let attn_mm = g.mm(attn2d, out_w);
    let attn_out = g.add(attn_mm, out_b);
    let attn_bsn = g.reshape_(attn_out, vec![batch as i64, seq as i64, n as i64]);
    let h = g.add(h_in, attn_bsn);

    let ln2_w = param_vec(g, params, &format!("{p}.ln2.weight"), &blk.ln2_w, n);
    let ln2_b = param_vec(g, params, &format!("{p}.ln2.bias"), &blk.ln2_b, n);
    let y = g.ln(h, ln2_w, ln2_b, eps);
    let y2d = g.reshape_(y, vec![(batch * seq) as i64, n as i64]);

    let gate_w = param_mat(
        g,
        params,
        &format!("{p}.ffn_gate.weight"),
        &transpose_2d(&blk.ffn_gate_w, n_ff, n),
        n,
        n_ff,
    )?;
    let gate_b = param_vec(
        g,
        params,
        &format!("{p}.ffn_gate.bias"),
        &blk.ffn_gate_b,
        n_ff,
    );
    let down_w = param_mat(
        g,
        params,
        &format!("{p}.ffn_down.weight"),
        &transpose_2d(&blk.ffn_down_w, n, n_ff),
        n_ff,
        n,
    )?;
    let down_b = param_vec(g, params, &format!("{p}.ffn_down.bias"), &blk.ffn_down_b, n);

    let gate_mm = g.mm(y2d, gate_w);
    let gate = g.add(gate_mm, gate_b);
    // HF `hidden_act: gelu_pytorch_tanh` → approximate GELU (tanh).
    let gate_act = g.gelu_approx(gate);
    let ff = if blk.ffn_gated {
        let up_w = param_mat(
            g,
            params,
            &format!("{p}.ffn_up.weight"),
            &transpose_2d(&blk.ffn_up_w, n_ff, n),
            n,
            n_ff,
        )?;
        let up_b = param_vec(g, params, &format!("{p}.ffn_up.bias"), &blk.ffn_up_b, n_ff);
        let up_mm = g.mm(y2d, up_w);
        let up = g.add(up_mm, up_b);
        g.mul(gate_act, up)
    } else {
        gate_act
    };
    let down_mm = g.mm(ff, down_w);
    let down = g.add(down_mm, down_b);
    let down_bsn = g.reshape_(down, vec![batch as i64, seq as i64, n as i64]);
    Ok(g.add(h, down_bsn))
}

fn build_deepstack_branch(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    il: usize,
    ds: &DeepstackWeights,
    h_in: NodeId,
    batch: usize,
    n_pos: usize,
    n: usize,
    n_ff: usize,
    eps: f32,
    merge_sq: usize,
) -> Result<NodeId> {
    let p = format!("deepstack.{il}");
    let merged = merge_spatial_tokens(g, h_in, batch, n_pos, n, merge_sq);
    let ln_w = param_vec(
        g,
        params,
        &format!("{p}.norm.weight"),
        &ds.norm_w,
        n * merge_sq,
    );
    let ln_b = param_vec(
        g,
        params,
        &format!("{p}.norm.bias"),
        &ds.norm_b,
        n * merge_sq,
    );
    let x = g.ln(merged, ln_w, ln_b, eps);
    let rows = batch * (n_pos / merge_sq);
    let x2d = g.reshape_(x, vec![rows as i64, (n * merge_sq) as i64]);
    let fc1_w = param_mat(
        g,
        params,
        &format!("{p}.fc1.weight"),
        &transpose_2d(&ds.fc1_w, n_ff, n * merge_sq),
        n * merge_sq,
        n_ff,
    )?;
    let fc1_b = param_vec(g, params, &format!("{p}.fc1.bias"), &ds.fc1_b, n_ff);
    let fc1_mm = g.mm(x2d, fc1_w);
    let fc1_pre = g.add(fc1_mm, fc1_b);
    let fc1 = g.gelu(fc1_pre);
    let fc2_w = param_mat(
        g,
        params,
        &format!("{p}.fc2.weight"),
        &transpose_2d(&ds.fc2_w, n * merge_sq, n_ff),
        n_ff,
        n * merge_sq,
    )?;
    let fc2_b = param_vec(g, params, &format!("{p}.fc2.bias"), &ds.fc2_b, n * merge_sq);
    let fc2_mm = g.mm(fc1, fc2_w);
    let out = g.add(fc2_mm, fc2_b);
    Ok(g.reshape_(
        out,
        vec![
            batch as i64,
            (n_pos / merge_sq) as i64,
            (n * merge_sq) as i64,
        ],
    ))
}

fn merge_spatial_tokens(
    g: &mut HirMut,
    x: NodeId,
    batch: usize,
    n_pos: usize,
    n: usize,
    merge_sq: usize,
) -> NodeId {
    let n_out = n_pos / merge_sq;
    g.reshape_(x, vec![(batch * n_out) as i64, (n * merge_sq) as i64])
}

fn apply_vision_rope(
    g: &mut HirMut,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> NodeId {
    let f = DType::F32;
    let half = head_dim / 2;
    let x4 = g.reshape_(
        x,
        vec![batch as i64, seq as i64, heads as i64, head_dim as i64],
    );
    let cos3 = g.reshape_(cos, vec![1, seq as i64, 1, head_dim as i64]);
    let sin3 = g.reshape_(sin, vec![1, seq as i64, 1, head_dim as i64]);
    let half_shape = Shape::new(&[batch, seq, heads, half], f);
    let x_lo = g.narrow_(x4, 3, 0, half);
    let x_hi = g.narrow_(x4, 3, half, half);
    let neg_hi = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Neg),
        vec![x_hi],
        half_shape,
    );
    let rotated = g.concat_(vec![neg_hi, x_lo], 3);
    let target_shape = vec![batch as i64, seq as i64, heads as i64, head_dim as i64];
    let target = vec![batch, seq, heads, head_dim];
    let cos4 = g.add_node(
        Op::Expand {
            target_shape: target_shape.clone(),
        },
        vec![cos3],
        Shape::new(&target, f),
    );
    let sin4 = g.add_node(
        Op::Expand { target_shape },
        vec![sin3],
        Shape::new(&target, f),
    );
    let scaled = g.mul(x4, cos4);
    let rotated_scaled = g.mul(rotated, sin4);
    let out = g.add(scaled, rotated_scaled);
    g.reshape_(
        out,
        vec![batch as i64, seq as i64, (heads * head_dim) as i64],
    )
}

fn flatten_nchw_to_bsn(
    g: &mut HirMut,
    x: NodeId,
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
) -> NodeId {
    // [B, C, H, W] → [B, H*W, C]
    let flat = g.reshape_(x, vec![batch as i64, c as i64, (h * w) as i64]);
    g.transpose_(flat, vec![0, 2, 1])
}

fn resize_position_embd(
    cfg: &MmProjConfig,
    weights: &MmProjWeights,
    hp: usize,
    wp: usize,
) -> Vec<f32> {
    let n = cfg.n_embd;
    let n_pos = hp * wp;
    let src_side = (weights.position_embd.len() / n).isqrt().max(1);
    if src_side * src_side * n != weights.position_embd.len() {
        // Unexpected layout — fall back to truncate/pad copy.
        let mut out = vec![0f32; n_pos * n];
        let copy = (weights.position_embd.len() / n).min(n_pos);
        for i in 0..copy {
            out[i * n..(i + 1) * n]
                .copy_from_slice(&weights.position_embd[i * n..(i + 1) * n]);
        }
        return out;
    }
    if src_side == hp && src_side == wp {
        return weights.position_embd.clone();
    }
    // HF `get_vision_bilinear_indices_and_weights`: linspace onto the
    // square pos table, then bilinear sample (pre-reorder).
    let mut out = vec![0f32; n_pos * n];
    let side = src_side as f32;
    for ty in 0..hp {
        let gy = if hp == 1 {
            0.0
        } else {
            ty as f32 * (side - 1.0) / (hp as f32 - 1.0)
        };
        let y0 = gy.floor() as usize;
        let y1 = (y0 + 1).min(src_side - 1);
        let fy = gy - y0 as f32;
        for tx in 0..wp {
            let gx = if wp == 1 {
                0.0
            } else {
                tx as f32 * (side - 1.0) / (wp as f32 - 1.0)
            };
            let x0 = gx.floor() as usize;
            let x1 = (x0 + 1).min(src_side - 1);
            let fx = gx - x0 as f32;
            let dst = (ty * wp + tx) * n;
            let w00 = (1.0 - fy) * (1.0 - fx);
            let w01 = (1.0 - fy) * fx;
            let w10 = fy * (1.0 - fx);
            let w11 = fy * fx;
            let i00 = (y0 * src_side + x0) * n;
            let i01 = (y0 * src_side + x1) * n;
            let i10 = (y1 * src_side + x0) * n;
            let i11 = (y1 * src_side + x1) * n;
            for d in 0..n {
                out[dst + d] = w00 * weights.position_embd[i00 + d]
                    + w01 * weights.position_embd[i01 + d]
                    + w10 * weights.position_embd[i10 + d]
                    + w11 * weights.position_embd[i11 + d];
            }
        }
    }
    out
}

fn reorder_position_embd(position_embd: Vec<f32>, idx: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; position_embd.len()];
    for (dst, &src) in idx.iter().enumerate() {
        let src = src as usize;
        out[dst * n..(dst + 1) * n].copy_from_slice(&position_embd[src * n..(src + 1) * n]);
    }
    out
}

fn param_conv(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    out_c: usize,
    in_c: usize,
    k: usize,
) -> Result<NodeId> {
    let shape = Shape::new(&[out_c, in_c, k, k], DType::F32);
    Ok(register_param(g, params, name, data.to_vec(), shape))
}

fn param_vec(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    len: usize,
) -> NodeId {
    register_param(
        g,
        params,
        name,
        data.to_vec(),
        Shape::new(&[len], DType::F32),
    )
}

fn param_mat(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<NodeId> {
    if data.len() != in_dim * out_dim {
        return Err(anyhow::anyhow!(
            "{name}: len {} != in {in_dim} * out {out_dim}",
            data.len()
        ));
    }
    Ok(register_param(
        g,
        params,
        name,
        data.to_vec(),
        Shape::new(&[in_dim, out_dim], DType::F32),
    ))
}

fn param_mask(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    batch: usize,
    seq: usize,
) -> NodeId {
    register_param(
        g,
        params,
        name,
        vec![1.0; batch * seq],
        Shape::new(&[batch, seq], DType::F32),
    )
}

fn broadcast_bias_bsn(g: &mut HirMut, bias: NodeId, batch: usize, seq: usize, n: usize) -> NodeId {
    let bsn = g.reshape_(bias, vec![1, 1, n as i64]);
    g.add_node(
        Op::Expand {
            target_shape: vec![batch as i64, seq as i64, n as i64],
        },
        vec![bsn],
        Shape::new(&[batch, seq, n], DType::F32),
    )
}

fn register_param(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: Vec<f32>,
    shape: Shape,
) -> NodeId {
    let id = g.param(name, shape);
    params.insert(name.to_string(), data);
    id
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

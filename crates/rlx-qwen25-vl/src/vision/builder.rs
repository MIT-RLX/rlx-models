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

//! Qwen2.5-VL vision tower HIR — mirrors `tools/mtmd/models/qwen2vl.cpp`.

use super::config::MmProjConfig;
use super::preprocess::vision_rope_feeds;
use super::weights::{MmProjWeights, VisionBlockWeights};
use anyhow::{Result, anyhow};
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Op, Shape};
use std::collections::HashMap;

type NodeId = HirNodeId;

pub fn build_qwen25_vl_vision_hir(
    cfg: &MmProjConfig,
    weights: &MmProjWeights,
    img_w: usize,
    img_h: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_dims(cfg, img_w, img_h)?;
    let mut hir = HirModule::new("qwen25_vl_vision").with_fusion_policy(FusionPolicy::Direct);
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let batch = 1usize;
    let n = cfg.n_embd;
    let enc_ff = if cfg.n_ff > 0 { cfg.n_ff } else { n * 4 };
    let mm_ff = enc_ff.max(n * 4);
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
    let use_window = cfg.n_wa_pattern > 0;

    anyhow::ensure!(
        rope_cos.len() == n_pos * dh,
        "rope_cos len {} != n_pos*head_dim",
        rope_cos.len()
    );
    anyhow::ensure!(
        rope_sin.len() == n_pos * dh,
        "rope_sin len {} != n_pos*head_dim",
        rope_sin.len()
    );

    let image = hir.input("image", Shape::new(&[batch, 3, img_h, img_w], f));
    let rope_cos_id = hir.input("vision_rope_cos", Shape::new(&[n_pos, dh], f));
    let rope_sin_id = hir.input("vision_rope_sin", Shape::new(&[n_pos, dh], f));
    let inv_window_idx = if use_window {
        Some(hir.input("inv_window_idx", Shape::new(&[n_out], f)))
    } else {
        None
    };
    let window_idx = if use_window {
        Some(hir.input("window_idx", Shape::new(&[n_out], f)))
    } else {
        None
    };
    let window_mask_id = if use_window {
        Some(hir.input("window_mask", Shape::new(&[batch, nh, n_pos, n_pos], f)))
    } else {
        None
    };

    let mut g = HirMut::new(&mut hir);

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
    let flat = flatten_nchw_to_bsn(&mut g, patches, batch, n, out_h, out_w);
    let merge_idx = super::preprocess::build_spatial_merge_gather_idx(hp, wp, cfg.n_merge);
    let merge_idx_id = param_vec(&mut g, &mut params, "spatial_merge_idx", &merge_idx, n_pos);
    let merge_idx_2d = g.reshape_(merge_idx_id, vec![batch as i64, n_pos as i64]);
    let mut h_id = g.gather_(flat, merge_idx_2d, 1);

    if !weights.patch_bias.is_empty() {
        let bias = param_vec(&mut g, &mut params, "patch_bias", &weights.patch_bias, n);
        let bias_bsn = broadcast_bias_bsn(&mut g, bias, batch, n_pos, n);
        h_id = g.add(h_id, bias_bsn);
    }

    if !weights.pre_ln_w.is_empty() {
        h_id = apply_norm(
            &mut g,
            &mut params,
            cfg,
            h_id,
            "pre_ln",
            &weights.pre_ln_w,
            weights.pre_ln_b.as_deref(),
            n,
            eps,
        );
    }

    if let Some(inv) = inv_window_idx {
        h_id = rlx_ir::window_token_gather_bsn(&mut g, h_id, inv, batch, n_pos, n, merge_sq);
    }

    for (il, blk) in weights.blocks.iter().enumerate() {
        let full_attn = !use_window || (il + 1) % cfg.n_wa_pattern == 0;
        let mask = if full_attn { None } else { window_mask_id };
        h_id = build_encoder_block(
            &mut g,
            &mut params,
            cfg,
            il,
            blk,
            h_id,
            mask,
            rope_cos_id,
            rope_sin_id,
            batch,
            n_pos,
            n,
            enc_ff,
            nh,
            dh,
            eps,
            full_attn,
        )?;
    }

    h_id = apply_norm(
        &mut g,
        &mut params,
        cfg,
        h_id,
        "post_ln",
        &weights.post_ln_w,
        weights.post_ln_b.as_deref(),
        n,
        eps,
    );

    let merged = merge_spatial_tokens(&mut g, h_id, batch, n_pos, n, merge_sq);
    let mm0_w = param_mat(
        &mut g,
        &mut params,
        "mm.0.weight",
        &transpose_2d(&weights.mm_0_w, mm_ff, n * merge_sq),
        n * merge_sq,
        mm_ff,
    )?;
    let mm0_b = param_vec(&mut g, &mut params, "mm.0.bias", &weights.mm_0_b, mm_ff);
    let mm0_mm = g.mm(merged, mm0_w);
    let mm0 = g.add(mm0_mm, mm0_b);
    let mm0_act = g.gelu(mm0);

    let mm1_w = param_mat(
        &mut g,
        &mut params,
        "mm.1.weight",
        &transpose_2d(&weights.mm_1_w, proj, mm_ff),
        mm_ff,
        proj,
    )?;
    let mm1_b = param_vec(&mut g, &mut params, "mm.1.bias", &weights.mm_1_b, proj);
    let mm1_mm = g.mm(mm0_act, mm1_w);
    let mut embeddings = g.add(mm1_mm, mm1_b);

    if let Some(win) = window_idx {
        embeddings = rlx_ir::window_token_scatter_bsn(&mut g, embeddings, win, batch, n_out, proj);
    }

    let out = g.reshape_(embeddings, vec![batch as i64, n_out as i64, proj as i64]);
    g.set_outputs(vec![out]);

    Ok((hir, params))
}

pub fn build_qwen25_vl_vision_built(
    cfg: &MmProjConfig,
    weights: &MmProjWeights,
    img_w: usize,
    img_h: usize,
) -> Result<rlx_flow::BuiltModel> {
    let position_hw = super::preprocess::build_vision_position_hw(img_w, img_h, cfg);
    let head_dim = cfg.n_embd / cfg.n_head;
    let (rope_cos, rope_sin) = vision_rope_feeds(&position_hw, head_dim);
    let (hir, params) =
        build_qwen25_vl_vision_hir(cfg, weights, img_w, img_h, &rope_cos, &rope_sin)?;
    rlx_core::flow_util::built_from_hir_with_profile(
        hir,
        params,
        rlx_flow::CompileProfile::encoder(),
    )
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
    cfg: &MmProjConfig,
    il: usize,
    blk: &VisionBlockWeights,
    h_in: NodeId,
    window_mask: Option<NodeId>,
    rope_cos: NodeId,
    rope_sin: NodeId,
    batch: usize,
    seq: usize,
    n: usize,
    enc_ff: usize,
    nh: usize,
    dh: usize,
    eps: f32,
    full_attn: bool,
) -> Result<NodeId> {
    let p = format!("blk.{il}");
    let x = apply_norm(
        g,
        params,
        cfg,
        h_in,
        &format!("{p}.ln1"),
        &blk.ln1_w,
        blk.ln1_b.as_deref(),
        n,
        eps,
    );

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

    let out_shape = Shape::new(&[batch, seq, n], DType::F32);
    let attn = if full_attn {
        g.attention_kind(q, k, v, nh, dh, MaskKind::None, out_shape.clone())
    } else {
        g.attention_bias(
            q,
            k,
            v,
            window_mask.expect("window layer requires window_mask"),
            nh,
            dh,
            out_shape,
        )
    };
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

    let y = apply_norm(
        g,
        params,
        cfg,
        h,
        &format!("{p}.ln2"),
        &blk.ln2_w,
        blk.ln2_b.as_deref(),
        n,
        eps,
    );
    let y2d = g.reshape_(y, vec![(batch * seq) as i64, n as i64]);

    let gate_w = param_mat(
        g,
        params,
        &format!("{p}.ffn_gate.weight"),
        &transpose_2d(&blk.ffn_gate_w, enc_ff, n),
        n,
        enc_ff,
    )?;
    let gate_b = param_vec(
        g,
        params,
        &format!("{p}.ffn_gate.bias"),
        &blk.ffn_gate_b,
        enc_ff,
    );
    let up_w = param_mat(
        g,
        params,
        &format!("{p}.ffn_up.weight"),
        &transpose_2d(&blk.ffn_up_w, enc_ff, n),
        n,
        enc_ff,
    )?;
    let up_b = param_vec(
        g,
        params,
        &format!("{p}.ffn_up.bias"),
        &blk.ffn_up_b,
        enc_ff,
    );
    let down_w = param_mat(
        g,
        params,
        &format!("{p}.ffn_down.weight"),
        &transpose_2d(&blk.ffn_down_w, n, enc_ff),
        enc_ff,
        n,
    )?;
    let down_b = param_vec(g, params, &format!("{p}.ffn_down.bias"), &blk.ffn_down_b, n);

    let gate_mm = g.mm(y2d, gate_w);
    let gate = g.add(gate_mm, gate_b);
    let up_mm = g.mm(y2d, up_w);
    let up = g.add(up_mm, up_b);
    let gate_act = if cfg.use_silu {
        g.silu(gate)
    } else {
        g.gelu(gate)
    };
    let ff = g.mul(gate_act, up);
    let down_mm = g.mm(ff, down_w);
    let down = g.add(down_mm, down_b);
    let down_bsn = g.reshape_(down, vec![batch as i64, seq as i64, n as i64]);
    Ok(g.add(h, down_bsn))
}

fn apply_vision_rope(
    g: &mut HirMut,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    hd: usize,
) -> NodeId {
    let f = DType::F32;
    let half = hd / 2;
    let x4 = g.reshape_(x, vec![batch as i64, seq as i64, heads as i64, hd as i64]);
    let cos3 = g.reshape_(cos, vec![1, seq as i64, 1, hd as i64]);
    let sin3 = g.reshape_(sin, vec![1, seq as i64, 1, hd as i64]);
    let sh_half = Shape::new(&[batch, seq, heads, half], f);
    let x_lo = g.narrow_(x4, 3, 0, half);
    let x_hi = g.narrow_(x4, 3, half, half);
    let neg_hi = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Neg),
        vec![x_hi],
        sh_half.clone(),
    );
    let rot = g.concat_(vec![neg_hi, x_lo], 3);
    let cos4 = g.add_node(
        Op::Expand {
            target_shape: vec![batch as i64, seq as i64, heads as i64, hd as i64],
        },
        vec![cos3],
        Shape::new(&[batch, seq, heads, hd], f),
    );
    let sin4 = g.add_node(
        Op::Expand {
            target_shape: vec![batch as i64, seq as i64, heads as i64, hd as i64],
        },
        vec![sin3],
        Shape::new(&[batch, seq, heads, hd], f),
    );
    let xc = g.mul(x4, cos4);
    let rs = g.mul(rot, sin4);
    let out = g.add(xc, rs);
    g.reshape_(out, vec![batch as i64, seq as i64, (heads * hd) as i64])
}

fn apply_norm(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &MmProjConfig,
    x: NodeId,
    name: &str,
    weight: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    eps: f32,
) -> NodeId {
    let w = param_vec(g, params, &format!("{name}.weight"), weight, n);
    if cfg.use_rms_norm {
        let beta = param_vec(g, params, &format!("{name}.beta"), &vec![0.0f32; n], n);
        g.rms_norm(x, w, beta, eps)
    } else {
        let b = param_vec(
            g,
            params,
            &format!("{name}.bias"),
            bias.unwrap_or(&vec![0.0; n]),
            n,
        );
        g.ln(x, w, b, eps)
    }
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

fn flatten_nchw_to_bsn(
    g: &mut HirMut,
    x: NodeId,
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
) -> NodeId {
    let flat = g.reshape_(x, vec![batch as i64, c as i64, (h * w) as i64]);
    g.transpose_(flat, vec![0, 2, 1])
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

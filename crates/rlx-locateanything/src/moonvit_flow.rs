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

//! Compiled MoonViT encoder — HIR graph for patch stem + transformer + final LN.

use crate::compile_support::moonvit_use_decomposed_rope;
use crate::config::MoonVitConfig;
use crate::moonvit::interpolate_pos_emb;
use crate::rope2d::rope_cos_sin_halves_for_grid;
use crate::weights::LocateAnythingWeightPrefix;
use anyhow::{Result, ensure};
use rlx_core::flow_util::built_from_hir_with_profile;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, Op, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

type NodeId = HirNodeId;

pub struct MoonVitBuilt {
    pub model: BuiltModel,
    pub grid_h: usize,
    pub grid_w: usize,
    pub merge: [usize; 2],
}

pub fn build_moonvit_built(
    cfg: &MoonVitConfig,
    weights: &mut WeightMap,
    batch: usize,
    grid_h: usize,
    grid_w: usize,
    device: Device,
) -> Result<MoonVitBuilt> {
    let portable_rope = moonvit_use_decomposed_rope(device);
    let (hir, params) = build_moonvit_hir(cfg, weights, batch, grid_h, grid_w, portable_rope)?;
    let model = built_from_hir_with_profile(hir, params, CompileProfile::encoder())?;
    Ok(MoonVitBuilt {
        model,
        grid_h,
        grid_w,
        merge: cfg.merge_kernel_size,
    })
}

pub fn build_moonvit_hir(
    cfg: &MoonVitConfig,
    weights: &mut WeightMap,
    batch: usize,
    grid_h: usize,
    grid_w: usize,
    portable_rope: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let dh = cfg.head_dim();
    let mlp = cfg.intermediate_size;
    let ps = cfg.patch_size;
    let patch_dim = 3 * ps * ps;
    let seq = grid_h * grid_w;
    let f = DType::F32;
    let eps = 1e-5f32;

    ensure!(dh.is_multiple_of(4), "head_dim must be divisible by 4");
    ensure!(
        grid_h < 512 && grid_w < 512,
        "grid {grid_h}x{grid_w} exceeds position embedding limit"
    );

    let (patch_w, patch_shape) = weights.take(LocateAnythingWeightPrefix::vision_patch_proj_w())?;
    let (patch_b, _) = weights.take(LocateAnythingWeightPrefix::vision_patch_proj_b())?;
    ensure!(
        patch_shape == [h, 3, ps, ps],
        "patch proj shape {:?}",
        patch_shape
    );
    let (pos_emb, pos_shape) = weights.take(LocateAnythingWeightPrefix::vision_pos_emb())?;
    let pos_h = cfg.init_pos_emb_height;
    let pos_w = cfg.init_pos_emb_width;
    ensure!(pos_shape == [pos_h, pos_w, h]);

    let pos = interpolate_pos_emb(&pos_emb, pos_h, pos_w, grid_h, grid_w, h);
    const ROPE_THETA: f32 = 10_000.0;

    let mut hir = HirModule::new("moonvit").with_fusion_policy(rlx_ir::hir::FusionPolicy::Direct);
    let patches = hir.input("patches", Shape::new(&[batch, seq, patch_dim], f));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut g = HirMut::new(&mut hir);

    let patch_w_t = transpose_mat(&patch_w, h, patch_dim);
    let pw = param_mat(
        &mut g,
        &mut params,
        "patch_embed.weight",
        &patch_w_t,
        patch_dim,
        h,
    )?;
    let pb = param_vec(&mut g, &mut params, "patch_embed.bias", &patch_b, h);
    let flat = g.reshape_(patches, vec![(batch * seq) as i64, patch_dim as i64]);
    let mm = g.mm(flat, pw);
    let stem = g.add(mm, pb);
    let mut hidden = g.reshape_(stem, vec![batch as i64, seq as i64, h as i64]);

    let pos_w = param_mat(&mut g, &mut params, "pos_emb", &pos, seq, h)?;
    let pos_sh = g.reshape_(pos_w, vec![1, seq as i64, h as i64]);
    let pos_bc = expand_bsn(&mut g, pos_sh, batch, seq, h);
    hidden = g.add(hidden, pos_bc);

    for i in 0..cfg.num_hidden_layers {
        let lp = format!("blocks.{i}");
        let norm0_w = take_vec(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "norm0.weight"),
        )?;
        let norm0_b = take_vec(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "norm0.bias"),
        )?;
        let wqkv_w = take_mat(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "wqkv.weight"),
        )?;
        let wqkv_b = take_vec(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "wqkv.bias"),
        )?;
        let wo_w = take_mat(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "wo.weight"),
        )?;
        let wo_b = take_vec(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "wo.bias"),
        )?;
        let norm1_w = take_vec(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "norm1.weight"),
        )?;
        let norm1_b = take_vec(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "norm1.bias"),
        )?;
        let mlp0_w = take_mat(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "mlp.fc0.weight"),
        )?;
        let mlp0_b = take_vec(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "mlp.fc0.bias"),
        )?;
        let mlp1_w = take_mat(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "mlp.fc1.weight"),
        )?;
        let mlp1_b = take_vec(
            weights,
            &LocateAnythingWeightPrefix::vision_block(i, "mlp.fc1.bias"),
        )?;

        hidden = build_encoder_block(
            &mut g,
            &mut params,
            &lp,
            cfg,
            hidden,
            &norm0_w,
            &norm0_b,
            &wqkv_w,
            &wqkv_b,
            &wo_w,
            &wo_b,
            &norm1_w,
            &norm1_b,
            &mlp0_w,
            &mlp0_b,
            &mlp1_w,
            &mlp1_b,
            batch,
            seq,
            grid_h,
            grid_w,
            h,
            mlp,
            nh,
            dh,
            eps,
            ROPE_THETA,
            portable_rope,
        )?;
    }

    let final_w = take_vec(weights, LocateAnythingWeightPrefix::vision_final_ln_w())?;
    let final_b = take_vec(weights, LocateAnythingWeightPrefix::vision_final_ln_b())?;
    let fln_w = param_vec(&mut g, &mut params, "final_ln.weight", &final_w, h);
    let fln_b = param_vec(&mut g, &mut params, "final_ln.bias", &final_b, h);
    hidden = g.ln(hidden, fln_w, fln_b, eps);
    let merged = merge_patches_2d(
        &mut g,
        hidden,
        batch,
        grid_h,
        grid_w,
        h,
        cfg.merge_kernel_size,
    );

    g.set_outputs(vec![merged]);
    Ok((hir, params))
}

/// 2×2 (or `merge`) spatial merge — same layout as [`crate::moonvit::patch_merger`].
fn merge_patches_2d(
    g: &mut HirMut,
    hidden: NodeId,
    batch: usize,
    grid_h: usize,
    grid_w: usize,
    h: usize,
    merge: [usize; 2],
) -> NodeId {
    let kh = merge[0];
    let kw = merge[1];
    let nh = grid_h / kh;
    let nw = grid_w / kw;
    let out_dim = h * kh * kw;
    let mut tokens = Vec::new();
    for py in 0..nh {
        for px in 0..nw {
            let mut pieces = Vec::with_capacity(kh * kw);
            for dy in 0..kh {
                for dx in 0..kw {
                    let sy = py * kh + dy;
                    let sx = px * kw + dx;
                    let idx = sy * grid_w + sx;
                    let piece = g.narrow_(hidden, 1, idx, 1);
                    let flat = g.reshape_(piece, vec![batch as i64, h as i64]);
                    pieces.push(flat);
                }
            }
            let cat = g.concat_(pieces, 1);
            tokens.push(g.reshape_(cat, vec![batch as i64, 1, out_dim as i64]));
        }
    }
    let merged = g.concat_(tokens, 1);
    g.reshape_(merged, vec![batch as i64, (nh * nw) as i64, out_dim as i64])
}

fn build_encoder_block(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    p: &str,
    vit: &MoonVitConfig,
    h_in: NodeId,
    norm0_w: &[f32],
    norm0_b: &[f32],
    wqkv_w: &[f32],
    wqkv_b: &[f32],
    wo_w: &[f32],
    wo_b: &[f32],
    norm1_w: &[f32],
    norm1_b: &[f32],
    mlp0_w: &[f32],
    mlp0_b: &[f32],
    mlp1_w: &[f32],
    mlp1_b: &[f32],
    batch: usize,
    seq: usize,
    grid_h: usize,
    grid_w: usize,
    h: usize,
    mlp: usize,
    nh: usize,
    dh: usize,
    eps: f32,
    rope_theta: f32,
    portable_rope: bool,
) -> Result<NodeId> {
    let f = DType::F32;
    let n0w = param_vec(g, params, &format!("{p}.norm0.weight"), norm0_w, h);
    let n0b = param_vec(g, params, &format!("{p}.norm0.bias"), norm0_b, h);
    let x = g.ln(h_in, n0w, n0b, eps);

    let x2d = g.reshape_(x, vec![(batch * seq) as i64, h as i64]);
    let qkv_w = param_mat(
        g,
        params,
        &format!("{p}.wqkv.weight"),
        &transpose_mat(wqkv_w, 3 * h, h),
        h,
        3 * h,
    )?;
    let qkv_b = param_vec(g, params, &format!("{p}.wqkv.bias"), wqkv_b, 3 * h);
    let qkv_mm = g.mm(x2d, qkv_w);
    let qkv = g.add(qkv_mm, qkv_b);
    let qkv_4d = g.reshape_(qkv, vec![batch as i64, seq as i64, 3 * h as i64]);
    let q = g.narrow_(qkv_4d, 2, 0, h);
    let k = g.narrow_(qkv_4d, 2, h, h);
    let v = g.narrow_(qkv_4d, 2, 2 * h, h);

    let (q, k) = if portable_rope {
        apply_axial_rope_decomposed(g, params, p, q, k, vit, batch, seq, grid_h, grid_w, nh, dh)?
    } else {
        let q_shape = g.shape(q);
        let q = g.0.mir(
            Op::AxialRope2d {
                end_x: grid_w,
                end_y: grid_h,
                head_dim: dh,
                num_heads: nh,
                theta: rope_theta,
                repeat_factor: 1,
            },
            vec![q],
            q_shape.clone(),
        );
        let k_shape = g.shape(k);
        let k = g.0.mir(
            Op::AxialRope2d {
                end_x: grid_w,
                end_y: grid_h,
                head_dim: dh,
                num_heads: nh,
                theta: rope_theta,
                repeat_factor: 1,
            },
            vec![k],
            k_shape.clone(),
        );
        (q, k)
    };

    let attn = g.attention_kind(
        q,
        k,
        v,
        nh,
        dh,
        MaskKind::None,
        Shape::new(&[batch, seq, h], f),
    );
    let attn2d = g.reshape_(attn, vec![(batch * seq) as i64, h as i64]);
    let ow = param_mat(
        g,
        params,
        &format!("{p}.wo.weight"),
        &transpose_mat(wo_w, h, h),
        h,
        h,
    )?;
    let ob = param_vec(g, params, &format!("{p}.wo.bias"), wo_b, h);
    let attn_mm = g.mm(attn2d, ow);
    let attn_out = g.add(attn_mm, ob);
    let attn_bsn = g.reshape_(attn_out, vec![batch as i64, seq as i64, h as i64]);
    let h_mid = g.add(h_in, attn_bsn);

    let n1w = param_vec(g, params, &format!("{p}.norm1.weight"), norm1_w, h);
    let n1b = param_vec(g, params, &format!("{p}.norm1.bias"), norm1_b, h);
    let y = g.ln(h_mid, n1w, n1b, eps);
    let y2d = g.reshape_(y, vec![(batch * seq) as i64, h as i64]);
    let m0w = param_mat(
        g,
        params,
        &format!("{p}.mlp0.weight"),
        &transpose_mat(mlp0_w, mlp, h),
        h,
        mlp,
    )?;
    let m0b = param_vec(g, params, &format!("{p}.mlp0.bias"), mlp0_b, mlp);
    let m1w = param_mat(
        g,
        params,
        &format!("{p}.mlp1.weight"),
        &transpose_mat(mlp1_w, h, mlp),
        mlp,
        h,
    )?;
    let m1b = param_vec(g, params, &format!("{p}.mlp1.bias"), mlp1_b, h);
    let m0_mm = g.mm(y2d, m0w);
    let m0 = g.add(m0_mm, m0b);
    let m0_act = g.gelu(m0);
    let m1_mm = g.mm(m0_act, m1w);
    let m1 = g.add(m1_mm, m1b);
    let delta = g.reshape_(m1, vec![batch as i64, seq as i64, h as i64]);
    Ok(g.add(h_mid, delta))
}

fn take_vec(weights: &mut WeightMap, key: &str) -> Result<Vec<f32>> {
    Ok(weights.take(key)?.0)
}

fn take_mat(weights: &mut WeightMap, key: &str) -> Result<Vec<f32>> {
    Ok(weights.take(key)?.0)
}

fn transpose_mat(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = w[r * cols + c];
        }
    }
    out
}

fn param_vec(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    len: usize,
) -> NodeId {
    let shape = Shape::new(&[len], DType::F32);
    params.insert(name.to_string(), data.to_vec());
    g.param(name, shape)
}

fn param_mat(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> Result<NodeId> {
    let shape = Shape::new(&[rows, cols], DType::F32);
    params.insert(name.to_string(), data.to_vec());
    Ok(g.param(name, shape))
}

/// 2D RoPE via two [`HirGraphExt::rope`] passes (X/Y head halves) — MLX / wgpu / Vulkan.
fn apply_axial_rope_decomposed(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    q: NodeId,
    k: NodeId,
    vit: &MoonVitConfig,
    batch: usize,
    seq: usize,
    grid_h: usize,
    grid_w: usize,
    nh: usize,
    dh: usize,
) -> Result<(NodeId, NodeId)> {
    let (cos_x, sin_x, cos_y, sin_y) = rope_cos_sin_halves_for_grid(vit, grid_h, grid_w);
    let quarter = dh / 4;
    let half = dh / 2;

    let q4 = g.reshape_(q, vec![batch as i64, seq as i64, nh as i64, dh as i64]);
    let k4 = g.reshape_(k, vec![batch as i64, seq as i64, nh as i64, dh as i64]);

    let q_x = g.narrow_(q4, 3, 0, half);
    let q_y = g.narrow_(q4, 3, half, half);
    let k_x = g.narrow_(k4, 3, 0, half);
    let k_y = g.narrow_(k4, 3, half, half);

    let cos_x_n = param_mat(
        g,
        params,
        &format!("{prefix}.rope_cos_x"),
        &cos_x,
        seq,
        quarter,
    )?;
    let sin_x_n = param_mat(
        g,
        params,
        &format!("{prefix}.rope_sin_x"),
        &sin_x,
        seq,
        quarter,
    )?;
    let cos_y_n = param_mat(
        g,
        params,
        &format!("{prefix}.rope_cos_y"),
        &cos_y,
        seq,
        quarter,
    )?;
    let sin_y_n = param_mat(
        g,
        params,
        &format!("{prefix}.rope_sin_y"),
        &sin_y,
        seq,
        quarter,
    )?;

    // `Op::Rope` applies along dim -2 — fold heads into batch: [B, S, H, D] → [B*H, S, D].
    let mut rope_bsh = |x: NodeId, cos: NodeId, sin: NodeId| -> NodeId {
        let flat = g.reshape_(x, vec![(batch * nh) as i64, seq as i64, half as i64]);
        let rotated = g.rope(flat, cos, sin, half);
        g.reshape_(
            rotated,
            vec![batch as i64, seq as i64, nh as i64, half as i64],
        )
    };
    let q_xr = rope_bsh(q_x, cos_x_n, sin_x_n);
    let q_yr = rope_bsh(q_y, cos_y_n, sin_y_n);
    let k_xr = rope_bsh(k_x, cos_x_n, sin_x_n);
    let k_yr = rope_bsh(k_y, cos_y_n, sin_y_n);

    let q_cat = g.concat_(vec![q_xr, q_yr], 3);
    let k_cat = g.concat_(vec![k_xr, k_yr], 3);
    let q_out = g.reshape_(q_cat, vec![batch as i64, seq as i64, (nh * dh) as i64]);
    let k_out = g.reshape_(k_cat, vec![batch as i64, seq as i64, (nh * dh) as i64]);
    Ok((q_out, k_out))
}

/// Broadcast `[1, S, N]` → `[B, S, N]` (positional embeddings are per-token, not a single bias vector).
fn expand_bsn(g: &mut HirMut, x: NodeId, batch: usize, seq: usize, n: usize) -> NodeId {
    if batch == 1 {
        return x;
    }
    g.add_node(
        Op::Expand {
            target_shape: vec![batch as i64, seq as i64, n as i64],
        },
        vec![x],
        Shape::new(&[batch, seq, n], DType::F32),
    )
}

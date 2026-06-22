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

//! HIR-native SAM3 ViT-L vision trunk (32 blocks, window + global attention).
//!
//! Mirrors [`super::vision_encoder::encode_image_native`] but expresses the
//! 32 transformer blocks as a single HIR graph so the heavy lifting can run
//! on any backend wired into `rlx-runtime` (Metal, MLX, CUDA, …).
//!
//! Patch embed + `ln_pre` stay on the host (cheap, sub-millisecond) and feed
//! `[1, grid*grid, embed_dim]` tokens into the compiled graph.

use super::config::Sam3VitConfig;
use super::packed_gguf::packed_linear;
use super::preprocess::assemble_patch_tokens;
use super::tensor::layer_norm;
use super::vision_encoder::{Sam3VisionEncoderWeights, Sam3VisionOutput, Sam3VitBlockWeights};
use anyhow::{Result, ensure};
use rlx_flow::CompileProfile;
use rlx_flow::{GgufPackedLinear, GgufPackedParams};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Op, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::{HashMap, HashSet};

const ROPE_THETA: f32 = 10_000.0;

/// Build product: HIR module + F32 params + (packed name, U8 blob, dtype) entries.
pub struct Sam3VisionEncoderHirParts {
    pub hir: HirModule,
    pub params: HashMap<String, Vec<f32>>,
    pub typed_params: Vec<(String, Vec<u8>, DType)>,
}

/// Compiled ViT-L vision trunk pinned to a device.
pub struct Sam3CompiledVisionEncoder {
    pub compiled: CompiledGraph,
    pub batch: usize,
    pub grid: usize,
    pub embed_dim: usize,
}

impl Sam3CompiledVisionEncoder {
    pub fn new(
        weights: &Sam3VisionEncoderWeights,
        cfg: &Sam3VitConfig,
        batch: usize,
        device: Device,
    ) -> Result<Self> {
        Self::new_with_profile(weights, cfg, batch, device, &CompileProfile::sam3())
    }

    pub fn new_with_profile(
        weights: &Sam3VisionEncoderWeights,
        cfg: &Sam3VitConfig,
        batch: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        Self::new_with_profile_and_gguf(weights, cfg, batch, device, profile, None)
    }

    pub fn new_with_profile_and_gguf(
        weights: &Sam3VisionEncoderWeights,
        cfg: &Sam3VitConfig,
        batch: usize,
        device: Device,
        profile: &CompileProfile,
        gguf_packed: Option<&GgufPackedParams>,
    ) -> Result<Self> {
        let parts = build_vision_encoder_hir(weights, cfg, batch, gguf_packed)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_hir_with_profile(device, parts.hir, profile)?;
        rlx_core::flow_util::attach_built_params(&mut compiled, parts.params, &parts.typed_params);
        Ok(Self {
            compiled,
            batch,
            grid: cfg.patch_grid(),
            embed_dim: cfg.embed_dim,
        })
    }

    /// Run with already-tokenized input `[batch, grid*grid, embed_dim]` (i.e.
    /// post patch-embed + `ln_pre`). Output has the same shape.
    pub fn run_tokens(&mut self, tokens: &[f32]) -> Result<Vec<f32>> {
        let expected = self.batch * self.grid * self.grid * self.embed_dim;
        ensure!(
            tokens.len() == expected,
            "vision encoder expects {expected} tokens, got {}",
            tokens.len()
        );
        let outputs = self.compiled.run(&[("tokens", tokens)]);
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("vision encoder graph produced no outputs"))
    }
}

/// Full end-to-end: preprocess image (CPU) → patch embed + ln_pre (CPU) → 32
/// transformer blocks (compiled graph) → tokens `[grid*grid, embed_dim]`.
pub fn encode_image_ir_on_with_profile(
    weights: &Sam3VisionEncoderWeights,
    gguf_packed: Option<&GgufPackedParams>,
    cfg: &Sam3VitConfig,
    image_nchw: &[f32],
    device: Device,
    profile: &CompileProfile,
) -> Result<Sam3VisionOutput> {
    let mut compiled = Sam3CompiledVisionEncoder::new_with_profile_and_gguf(
        weights,
        cfg,
        1,
        device,
        profile,
        gguf_packed,
    )?;
    let tokens_in = host_preroll(weights, cfg, image_nchw)?;
    let out = compiled.run_tokens(&tokens_in)?;
    Ok(Sam3VisionOutput {
        tokens: out,
        grid: cfg.patch_grid(),
        dim: cfg.embed_dim,
    })
}

pub fn encode_image_ir_on(
    weights: &Sam3VisionEncoderWeights,
    gguf_packed: Option<&GgufPackedParams>,
    cfg: &Sam3VitConfig,
    image_nchw: &[f32],
    device: Device,
) -> Result<Sam3VisionOutput> {
    encode_image_ir_on_with_profile(
        weights,
        gguf_packed,
        cfg,
        image_nchw,
        device,
        &CompileProfile::sam3(),
    )
}

/// CPU patch embed + ln_pre (lightweight) producing graph input.
pub fn host_preroll(
    weights: &Sam3VisionEncoderWeights,
    cfg: &Sam3VitConfig,
    image_nchw: &[f32],
) -> Result<Vec<f32>> {
    let mut x = assemble_patch_tokens(&weights.pre, image_nchw)?;
    x = layer_norm(
        &x,
        &weights.ln_pre_w,
        &weights.ln_pre_b,
        cfg.embed_dim,
        cfg.layer_norm_eps as f32,
    )?;
    Ok(x)
}

pub fn build_vision_encoder_hir(
    weights: &Sam3VisionEncoderWeights,
    cfg: &Sam3VitConfig,
    batch: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3VisionEncoderHirParts> {
    let e = cfg.embed_dim;
    let grid = cfg.patch_grid();
    let seq = grid * grid;
    let nh = cfg.num_heads;
    let dh = e / nh;
    ensure!(
        dh * nh == e,
        "embed_dim {e} not divisible by num_heads {nh}"
    );
    ensure!(dh.is_multiple_of(4), "head_dim must be divisible by 4");
    let ws = cfg.window_size;
    ensure!(
        ws > 0 && grid.is_multiple_of(ws),
        "vision IR currently assumes window_size>0 and divides grid (got ws={ws}, grid={grid})"
    );
    let nw_h = grid / ws;
    let nw_w = grid / ws;
    let num_windows = nw_h * nw_w;
    let win_len = ws * ws;
    let hidden = (e as f64 * cfg.mlp_ratio) as usize;

    let mut hir = HirModule::new("sam3_vision_encoder");
    let mut g = HirMut::new(&mut hir);
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut typed_params: Vec<(String, Vec<u8>, DType)> = Vec::new();
    let mut gguf_cache: HashMap<String, HirNodeId> = HashMap::new();
    let f = DType::F32;

    let tokens = g.input("tokens", Shape::new(&[batch, seq, e], f));

    // Two RoPE tables — windowed blocks (24×24, scale=1) and global blocks
    // (grid×grid, scale=ws/grid). Mirrors `build_rope_freqs` from the CPU path.
    let scale_global = ws as f32 / grid as f32;
    let (cos_w_x_v, sin_w_x_v, cos_w_y_v, sin_w_y_v) =
        rope_quarter_tables(dh, ws, ws, ROPE_THETA, 1.0);
    let (cos_g_x_v, sin_g_x_v, cos_g_y_v, sin_g_y_v) =
        rope_quarter_tables(dh, grid, grid, ROPE_THETA, scale_global);

    let quarter = dh / 4;
    let cos_w_x = param_2d(
        &mut g,
        &mut params,
        "rope.win.cos_x",
        &cos_w_x_v,
        win_len,
        quarter,
    );
    let sin_w_x = param_2d(
        &mut g,
        &mut params,
        "rope.win.sin_x",
        &sin_w_x_v,
        win_len,
        quarter,
    );
    let cos_w_y = param_2d(
        &mut g,
        &mut params,
        "rope.win.cos_y",
        &cos_w_y_v,
        win_len,
        quarter,
    );
    let sin_w_y = param_2d(
        &mut g,
        &mut params,
        "rope.win.sin_y",
        &sin_w_y_v,
        win_len,
        quarter,
    );
    let cos_g_x = param_2d(
        &mut g,
        &mut params,
        "rope.glob.cos_x",
        &cos_g_x_v,
        seq,
        quarter,
    );
    let sin_g_x = param_2d(
        &mut g,
        &mut params,
        "rope.glob.sin_x",
        &sin_g_x_v,
        seq,
        quarter,
    );
    let cos_g_y = param_2d(
        &mut g,
        &mut params,
        "rope.glob.cos_y",
        &cos_g_y_v,
        seq,
        quarter,
    );
    let sin_g_y = param_2d(
        &mut g,
        &mut params,
        "rope.glob.sin_y",
        &sin_g_y_v,
        seq,
        quarter,
    );

    let global_set: HashSet<usize> = cfg.global_att_blocks.iter().copied().collect();

    let mut x = tokens;
    for (li, block) in weights.blocks.iter().enumerate() {
        let is_global = global_set.contains(&li);
        let (cos_x, sin_x, cos_y, sin_y) = if is_global {
            (cos_g_x, sin_g_x, cos_g_y, sin_g_y)
        } else {
            (cos_w_x, sin_w_x, cos_w_y, sin_w_y)
        };
        x = emit_block(
            &mut g,
            &mut params,
            &mut typed_params,
            &mut gguf_cache,
            gguf_packed,
            li,
            block,
            x,
            batch,
            seq,
            grid,
            ws,
            nw_h,
            nw_w,
            num_windows,
            win_len,
            e,
            nh,
            dh,
            hidden,
            cfg.layer_norm_eps as f32,
            is_global,
            cos_x,
            sin_x,
            cos_y,
            sin_y,
        )?;
    }
    g.set_outputs(vec![x]);
    Ok(Sam3VisionEncoderHirParts {
        hir,
        params,
        typed_params,
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_block(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    typed_params: &mut Vec<(String, Vec<u8>, DType)>,
    gguf_cache: &mut HashMap<String, HirNodeId>,
    gguf_packed: Option<&GgufPackedParams>,
    li: usize,
    block: &Sam3VitBlockWeights,
    x: HirNodeId,
    batch: usize,
    seq: usize,
    grid: usize,
    ws: usize,
    nw_h: usize,
    nw_w: usize,
    num_windows: usize,
    win_len: usize,
    e: usize,
    nh: usize,
    dh: usize,
    hidden: usize,
    eps: f32,
    is_global: bool,
    cos_x: HirNodeId,
    sin_x: HirNodeId,
    cos_y: HirNodeId,
    sin_y: HirNodeId,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let n1w = param_1d(g, params, &format!("b{li}.norm1.w"), &block.norm1_w, e);
    let n1b = param_1d(g, params, &format!("b{li}.norm1.b"), &block.norm1_b, e);
    let n1 = g.ln(x, n1w, n1b, eps);

    // QKV projection: [B, seq, e] -> [B, seq, 3e]; then split q/k/v on the
    // channel axis.
    let qkv = linear_or_gguf(
        g,
        params,
        typed_params,
        gguf_cache,
        gguf_packed,
        block.qkv_gguf_prefix.as_deref(),
        &format!("b{li}.qkv"),
        n1,
        &block.qkv_w_t,
        &block.qkv_b,
        e,
        3 * e,
    )?;
    let q_flat = g.narrow_(qkv, 2, 0, e);
    let k_flat = g.narrow_(qkv, 2, e, e);
    let v_flat = g.narrow_(qkv, 2, 2 * e, e);

    let (q_eff, k_eff, v_eff, eff_batch, eff_seq) = if is_global {
        (q_flat, k_flat, v_flat, batch, seq)
    } else {
        let q_w = window_partition(g, q_flat, batch, ws, nw_h, nw_w, e);
        let k_w = window_partition(g, k_flat, batch, ws, nw_h, nw_w, e);
        let v_w = window_partition(g, v_flat, batch, ws, nw_h, nw_w, e);
        (q_w, k_w, v_w, batch * num_windows, win_len)
    };

    let q_rot = rope_2d_decomposed(
        g, q_eff, eff_batch, eff_seq, nh, dh, cos_x, sin_x, cos_y, sin_y,
    );
    let k_rot = rope_2d_decomposed(
        g, k_eff, eff_batch, eff_seq, nh, dh, cos_x, sin_x, cos_y, sin_y,
    );

    let attn = g.attention_kind(
        q_rot,
        k_rot,
        v_eff,
        nh,
        dh,
        MaskKind::None,
        Shape::new(&[eff_batch, eff_seq, e], f),
    );

    let attn_full = if is_global {
        attn
    } else {
        window_unpartition(g, attn, batch, grid, ws, nw_h, nw_w, e)
    };

    let proj = linear_or_gguf(
        g,
        params,
        typed_params,
        gguf_cache,
        gguf_packed,
        block.proj_gguf_prefix.as_deref(),
        &format!("b{li}.proj"),
        attn_full,
        &block.proj_w_t,
        &block.proj_b,
        e,
        e,
    )?;
    let x = g.add(x, proj);

    let n2w = param_1d(g, params, &format!("b{li}.norm2.w"), &block.norm2_w, e);
    let n2b = param_1d(g, params, &format!("b{li}.norm2.b"), &block.norm2_b, e);
    let n2 = g.ln(x, n2w, n2b, eps);
    let fc1 = linear_or_gguf(
        g,
        params,
        typed_params,
        gguf_cache,
        gguf_packed,
        block.mlp_fc1_gguf_prefix.as_deref(),
        &format!("b{li}.mlp1"),
        n2,
        &block.mlp_fc1_w_t,
        &block.mlp_fc1_b,
        e,
        hidden,
    )?;
    let act = g.gelu_approx(fc1);
    let fc2 = linear_or_gguf(
        g,
        params,
        typed_params,
        gguf_cache,
        gguf_packed,
        block.mlp_fc2_gguf_prefix.as_deref(),
        &format!("b{li}.mlp2"),
        act,
        &block.mlp_fc2_w_t,
        &block.mlp_fc2_b,
        hidden,
        e,
    )?;
    Ok(g.add(x, fc2))
}

// ---------------------------------------------------------------------------
// Window partitioning: [B, grid*grid, e] <-> [B*num_windows, win_len, e].

fn window_partition(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    batch: usize,
    ws: usize,
    nw_h: usize,
    nw_w: usize,
    e: usize,
) -> HirNodeId {
    let v = g.reshape_(
        x,
        vec![
            batch as i64,
            nw_h as i64,
            ws as i64,
            nw_w as i64,
            ws as i64,
            e as i64,
        ],
    );
    let t = g.transpose_(v, vec![0, 1, 3, 2, 4, 5]);
    g.reshape_(
        t,
        vec![(batch * nw_h * nw_w) as i64, (ws * ws) as i64, e as i64],
    )
}

fn window_unpartition(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    batch: usize,
    grid: usize,
    ws: usize,
    nw_h: usize,
    nw_w: usize,
    e: usize,
) -> HirNodeId {
    let v = g.reshape_(
        x,
        vec![
            batch as i64,
            nw_h as i64,
            nw_w as i64,
            ws as i64,
            ws as i64,
            e as i64,
        ],
    );
    let t = g.transpose_(v, vec![0, 1, 3, 2, 4, 5]);
    g.reshape_(t, vec![batch as i64, (grid * grid) as i64, e as i64])
}

// ---------------------------------------------------------------------------
// 2D RoPE — pair-wise rotation matching `super::vision_encoder::rope_apply_inplace`.
//
// SAM3's convention is to rotate consecutive value pairs `(v[2k], v[2k+1])`
// inside each head's `head_dim` slice — the first `head_dim/2` slots get the X
// rotation, the second `head_dim/2` get the Y rotation. `Op::Rope` does the
// LLaMA-style **half-split** rotation `(v[i], v[i+rot_half])` instead, so we
// can't call `g.rope` directly. Implementing the rotation by hand with
// reshape + mul/add keeps the pairing unambiguous and skips the axis-merging
// trick (`[B, S, nh, dh] → [B*nh, S, dh]`) that only behaves like a no-op
// when `S == nh`.

#[allow(clippy::too_many_arguments)]
fn rope_2d_decomposed(
    g: &mut HirMut<'_>,
    x: HirNodeId,
    batch: usize,
    seq: usize,
    nh: usize,
    dh: usize,
    cos_x: HirNodeId,
    sin_x: HirNodeId,
    cos_y: HirNodeId,
    sin_y: HirNodeId,
) -> HirNodeId {
    let half = dh / 2;
    let quarter = dh / 4;

    let x4 = g.reshape_(x, vec![batch as i64, seq as i64, nh as i64, dh as i64]);
    let x_xh = g.narrow_(x4, 3, 0, half);
    let x_yh = g.narrow_(x4, 3, half, half);

    let xh_rot = pairwise_rope_half(g, x_xh, batch, seq, nh, quarter, cos_x, sin_x);
    let yh_rot = pairwise_rope_half(g, x_yh, batch, seq, nh, quarter, cos_y, sin_y);

    let cat = g.concat_(vec![xh_rot, yh_rot], 3);
    g.reshape_(cat, vec![batch as i64, seq as i64, (nh * dh) as i64])
}

/// Pair-wise complex rotation on `[B, S, nh, 2*quarter]`.
///
/// Treats the last dim as `quarter` consecutive `(real, imag)` pairs, rotates
/// each pair by the `(cos[s, k], sin[s, k])` entry, and returns a tensor of
/// the same shape. `cos`/`sin` are `[S, quarter]` and broadcast over B/nh.
#[allow(clippy::too_many_arguments)]
fn pairwise_rope_half(
    g: &mut HirMut<'_>,
    x: HirNodeId, // [B, S, nh, 2*quarter]
    batch: usize,
    seq: usize,
    nh: usize,
    quarter: usize,
    cos: HirNodeId, // [S, quarter]
    sin: HirNodeId, // [S, quarter]
) -> HirNodeId {
    // Expose the (real, imag) pair axis.
    let pairs = g.reshape_(
        x,
        vec![batch as i64, seq as i64, nh as i64, quarter as i64, 2],
    );
    let x_r5 = g.narrow_(pairs, 4, 0, 1);
    let x_i5 = g.narrow_(pairs, 4, 1, 1);
    // Drop the trailing length-1 axis so we can broadcast `cos`/`sin` cleanly.
    let x_r = g.reshape_(
        x_r5,
        vec![batch as i64, seq as i64, nh as i64, quarter as i64],
    );
    let x_i = g.reshape_(
        x_i5,
        vec![batch as i64, seq as i64, nh as i64, quarter as i64],
    );

    // [S, quarter] → [1, S, 1, quarter] for broadcasting.
    let cos_b = g.reshape_(cos, vec![1, seq as i64, 1, quarter as i64]);
    let sin_b = g.reshape_(sin, vec![1, seq as i64, 1, quarter as i64]);

    let rc = g.mul(x_r, cos_b);
    let is_ = g.mul(x_i, sin_b);
    let rs = g.mul(x_r, sin_b);
    let ic = g.mul(x_i, cos_b);
    let out_r = g.sub(rc, is_);
    let out_i = g.add(rs, ic);

    // Re-pair `(out_r, out_i)` into the original `[..., quarter, 2]` layout.
    let out_r5 = g.reshape_(
        out_r,
        vec![batch as i64, seq as i64, nh as i64, quarter as i64, 1],
    );
    let out_i5 = g.reshape_(
        out_i,
        vec![batch as i64, seq as i64, nh as i64, quarter as i64, 1],
    );
    let pairs_out = g.concat_(vec![out_r5, out_i5], 4);
    g.reshape_(
        pairs_out,
        vec![batch as i64, seq as i64, nh as i64, (2 * quarter) as i64],
    )
}

// ---------------------------------------------------------------------------
// Linear with optional GGUF packed fallback. `w_t` is [in_dim, out_dim] —
// what `FusedMatMulBiasAct` wants.

#[allow(clippy::too_many_arguments)]
fn linear_or_gguf(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    typed_params: &mut Vec<(String, Vec<u8>, DType)>,
    gguf_cache: &mut HashMap<String, HirNodeId>,
    gguf_packed: Option<&GgufPackedParams>,
    gguf_prefix: Option<&str>,
    ir_stem: &str,
    input: HirNodeId,
    w_t: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<HirNodeId> {
    if let Some(p) = gguf_prefix
        .and_then(|pref| gguf_packed.map(|gp| (gp, format!("{pref}.weight"))))
        .and_then(|(gp, key)| packed_linear(gp, &key))
    {
        return linear_gguf_bias(
            g,
            params,
            typed_params,
            gguf_cache,
            ir_stem,
            p,
            input,
            bias,
            in_dim,
            out_dim,
        );
    }
    ensure!(
        !w_t.is_empty(),
        "{ir_stem}: missing F32 weight and no GGUF packed entry"
    );
    Ok(fused_linear(
        g, params, ir_stem, input, w_t, bias, in_dim, out_dim,
    ))
}

fn fused_linear(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    ir_stem: &str,
    input: HirNodeId,
    w_t: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> HirNodeId {
    let f = DType::F32;
    let w_name = format!("{ir_stem}.w");
    let b_name = format!("{ir_stem}.b");
    let w_id = g.param(&w_name, Shape::new(&[in_dim, out_dim], f));
    params.insert(w_name, w_t.to_vec());
    let b_id = g.param(&b_name, Shape::new(&[out_dim], f));
    params.insert(b_name, bias.to_vec());
    let cur_shape = g.shape(input);
    let mut out_dims: Vec<usize> = cur_shape.dims().iter().map(|d| d.unwrap_static()).collect();
    *out_dims.last_mut().unwrap() = out_dim;
    g.add_node(
        Op::FusedMatMulBiasAct { activation: None },
        vec![input, w_id, b_id],
        Shape::new(&out_dims, f),
    )
}

#[allow(clippy::too_many_arguments)]
fn linear_gguf_bias(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    typed_params: &mut Vec<(String, Vec<u8>, DType)>,
    gguf_cache: &mut HashMap<String, HirNodeId>,
    ir_stem: &str,
    p: &GgufPackedLinear,
    input: HirNodeId,
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<HirNodeId> {
    ensure!(
        p.in_dim == in_dim && p.out_dim == out_dim,
        "{ir_stem}: packed linear shape {}x{} vs {in_dim}x{out_dim}",
        p.in_dim,
        p.out_dim
    );
    let w_name = format!("{ir_stem}.w");
    let w_id = if let Some(&id) = gguf_cache.get(&w_name) {
        id
    } else {
        let id = g.param(&w_name, Shape::new(&[p.w_q.len()], DType::U8));
        typed_params.push((w_name.clone(), p.w_q.clone(), DType::U8));
        gguf_cache.insert(w_name, id);
        id
    };
    let cur = g.shape(input);
    let mut dims: Vec<usize> = cur.dims().iter().map(|d| d.unwrap_static()).collect();
    *dims.last_mut().unwrap() = out_dim;
    let out_shape = Shape::new(&dims, DType::F32);
    let mm = g.add_node(
        Op::DequantMatMul { scheme: p.scheme },
        vec![input, w_id],
        out_shape,
    );
    Ok(add_f32_bias(g, params, &format!("{ir_stem}.b"), mm, bias))
}

fn add_f32_bias(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    input: HirNodeId,
    bias: &[f32],
) -> HirNodeId {
    if bias.iter().all(|&v| v == 0.0) {
        return input;
    }
    let b_id = g.param(name, Shape::new(&[bias.len()], DType::F32));
    params.insert(name.to_string(), bias.to_vec());
    g.add(input, b_id)
}

fn param_1d(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    n: usize,
) -> HirNodeId {
    let id = g.param(name, Shape::new(&[n], DType::F32));
    params.insert(name.to_string(), data.to_vec());
    id
}

fn param_2d(
    g: &mut HirMut<'_>,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> HirNodeId {
    let id = g.param(name, Shape::new(&[rows, cols], DType::F32));
    params.insert(name.to_string(), data.to_vec());
    id
}

/// SAM3 2D RoPE quarter tables (cos_x, sin_x, cos_y, sin_y) of shape
/// `[end_x*end_y, head_dim/4]` each. Matches
/// [`super::vision_encoder::build_rope_freqs`] but split into the X and Y
/// halves so each half can be consumed by the scalar `g.rope` op.
fn rope_quarter_tables(
    head_dim: usize,
    end_x: usize,
    end_y: usize,
    theta: f32,
    scale_pos: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(head_dim.is_multiple_of(4));
    let pair_per_axis = head_dim / 4;
    let mut freqs_per_pair = Vec::with_capacity(pair_per_axis);
    for k in 0..pair_per_axis {
        let exp = (4 * k) as f32 / head_dim as f32;
        freqs_per_pair.push(1.0 / theta.powf(exp));
    }
    let seq = end_x * end_y;
    let q = pair_per_axis;
    let mut cos_x = vec![0f32; seq * q];
    let mut sin_x = vec![0f32; seq * q];
    let mut cos_y = vec![0f32; seq * q];
    let mut sin_y = vec![0f32; seq * q];
    for pos in 0..seq {
        let t_x = (pos % end_x) as f32 * scale_pos;
        let t_y = (pos / end_x) as f32 * scale_pos;
        for k in 0..q {
            let ang_x = t_x * freqs_per_pair[k];
            let ang_y = t_y * freqs_per_pair[k];
            cos_x[pos * q + k] = ang_x.cos();
            sin_x[pos * q + k] = ang_x.sin();
            cos_y[pos * q + k] = ang_y.cos();
            sin_y[pos * q + k] = ang_y.sin();
        }
    }
    (cos_x, sin_x, cos_y, sin_y)
}

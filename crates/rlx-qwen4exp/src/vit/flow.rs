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

//! The tower as a [`ModelFlow`] over a [`WeightSource`].

use anyhow::{Result, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow, WeightSource};
use rlx_ir::hir::HirMut;
use rlx_ir::op::Activation;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use super::config::VisionConfig;
use super::preprocess::{position_embd_for_grid, spatial_merge_gather_idx};
use super::{IMAGE_INPUT, VISION_ROPE_COS_INPUT, VISION_ROPE_SIN_INPUT};
use crate::common::{act, linear};

// ───────────────────────────── the graph ─────────────────────────────

/// Build the vision tower for a fixed, already-smart-resized `(img_w, img_h)`.
///
/// Inputs: `image [1, 3, img_h, img_w]` (normalized NCHW; see
/// [`preprocess_rgb`](super::preprocess_rgb)), plus `vision_rope_cos` / `vision_rope_sin`
/// `[n_patch_tokens, head_dim]` from [`vision_rope_tables`](super::vision_rope_tables).
/// Output: `[1, n_out_tokens, llm_hidden_size]`, in the token order the LM's
/// placeholder span expects.
pub fn build_qwen4exp_vision_flow(
    cfg: &VisionConfig,
    weights: &mut WeightMap,
    img_w: usize,
    img_h: usize,
) -> Result<BuiltModel> {
    build_qwen4exp_vision_flow_with_source(cfg, &mut WeightMapSource(weights), img_w, img_h)
}

/// [`build_qwen4exp_vision_flow`] over any [`WeightSource`] — the entry point
/// for a **packed** mmproj.
///
/// Every matmul goes through [`crate::common::linear`], so a `mmproj-Q8_0.gguf`
/// read through `rlx_core::flow_bridge::PackedWeightLoaderSource` becomes fused
/// `DequantMatMul` nodes with no f32 weight materialized. The convolution
/// kernels, biases, LayerNorm gains and the position table stay f32 — they are
/// not 2-D projections.
pub fn build_qwen4exp_vision_flow_with_source(
    cfg: &VisionConfig,
    weights: &mut dyn WeightSource,
    img_w: usize,
    img_h: usize,
) -> Result<BuiltModel> {
    cfg.validate()?;
    let align = cfg.align_size();
    if img_w == 0 || img_h == 0 || !img_w.is_multiple_of(align) || !img_h.is_multiple_of(align) {
        bail!(
            "qwen4exp vision: image {img_w}x{img_h} must be a non-zero multiple \
             of patch_size · spatial_merge_size = {align}; run it through \
             `smart_resize` first"
        );
    }
    let f = DType::F32;
    let (wp, hp) = cfg.grid(img_w, img_h);
    let n_pos = wp * hp;
    let head_dim = cfg.head_dim();
    let n_out = n_pos / cfg.merge_sq();
    let proj = cfg.llm_hidden_size;
    let out_shape = Shape::new(&[1, n_out, proj], f);

    let merge_idx = spatial_merge_gather_idx(hp, wp, cfg.n_merge);
    if merge_idx.len() != n_pos {
        bail!(
            "qwen4exp vision: the merge permutation covers {} of {n_pos} patches \
             — the grid {wp}x{hp} is not tileable by {}",
            merge_idx.len(),
            cfg.n_merge
        );
    }

    let cfg = cfg.clone();
    let flow = ModelFlow::new("qwen4exp_vision")
        .with_profile(CompileProfile::llama32_prefill())
        .input(IMAGE_INPUT, Shape::new(&[1, 3, img_h, img_w], f))
        .input(VISION_ROPE_COS_INPUT, Shape::new(&[n_pos, head_dim], f))
        .input(VISION_ROPE_SIN_INPUT, Shape::new(&[n_pos, head_dim], f))
        .plugin_named("vit", move |emit, _prev| {
            let id = emit_vision_tower(emit, &cfg, img_w, img_h, &merge_idx)?;
            Ok(Some(emit.wrap(id, out_shape.clone())))
        });

    flow.output("vision_embeds").build_with(weights, None)
}

/// The tower itself. Split out so a caller embedding it in a larger graph can
/// reuse it.
pub fn emit_vision_tower(
    emit: &mut Emit<'_>,
    cfg: &VisionConfig,
    img_w: usize,
    img_h: usize,
    merge_idx: &[f32],
) -> Result<HirNodeId> {
    let f = DType::F32;
    let (wp, hp) = cfg.grid(img_w, img_h);
    let n_pos = wp * hp;
    let (n, ps) = (cfg.n_embd, cfg.patch_size);
    let n_out = n_pos / cfg.merge_sq();
    let (ni, pi) = (n as i64, n_pos as i64);

    let cos = emit.flow_input(VISION_ROPE_COS_INPUT)?.hir_id();
    let sin = emit.flow_input(VISION_ROPE_SIN_INPUT)?.hir_id();
    let image = emit.flow_input(IMAGE_INPUT)?.hir_id();

    // ── dual patch convolution ──
    // The GGUF already stores these as `[out_c, in_c, kh, kw]`, which is
    // `Op::Conv`'s weight layout, so neither needs a host transpose.
    let w0 = emit.load_param("v.patch_embd.weight", false)?;
    let w1 = emit.load_param("v.patch_embd.weight.1", false)?;
    let patch_bias = emit.load_param("v.patch_embd.bias", false)?;
    let merge_idx_p = emit.synth_param(
        "qwen4exp.vision.merge_idx",
        merge_idx.to_vec(),
        Shape::new(&[1, n_pos], f),
    );

    // The position table needs a host-side bilinear resize and the same
    // permutation as the tokens, so it is re-registered rather than loaded.
    let pos = {
        let (table, _) = emit.weights.take("v.position_embd.weight", false)?;
        let data = position_embd_for_grid(&table, n, hp, wp, merge_idx)?;
        emit.synth_param(
            "qwen4exp.vision.position_embd",
            data,
            Shape::new(&[n_pos, n], f),
        )
    };

    let mut h = {
        let mut gb = HirMut::new(emit.hir());
        let conv_shape = Shape::new(&[1, n, hp, wp], f);
        let c0 = gb.conv2d(image, w0, [ps, ps], [ps, ps], [0, 0], 1, conv_shape.clone());
        let c1 = gb.conv2d(image, w1, [ps, ps], [ps, ps], [0, 0], 1, conv_shape);
        let patches = gb.add(c0, c1);

        // [1, C, hp, wp] → [1, n_pos, C] in raster order, then permuted so each
        // merge block is contiguous.
        let flat = gb.reshape_(patches, vec![1, ni, pi]);
        let bsn = gb.transpose_(flat, vec![0, 2, 1]);
        let bsn = gb.gather_(bsn, merge_idx_p, 1);

        let bias = gb.reshape_(patch_bias, vec![1, 1, ni]);
        let x = gb.add(bsn, bias);
        let pos3 = gb.reshape_(pos, vec![1, pi, ni]);
        gb.add(x, pos3)
    };

    for il in 0..cfg.n_layer {
        h = emit_vision_block(emit, &format!("v.blk.{il}"), h, cos, sin, cfg, n_pos)?;
    }

    // `v.post_ln` IS the merger norm — applied once, before the spatial
    // reshape. There is no separate post-block LayerNorm in this mmproj.
    let post_w = emit.load_param("v.post_ln.weight", false)?;
    let post_b = emit.load_param("v.post_ln.bias", false)?;
    let merged = {
        let mut gb = HirMut::new(emit.hir());
        let normed = gb.ln(h, post_w, post_b, cfg.eps);
        // The merge is a pure reshape precisely because the tokens were
        // permuted into blocks above.
        gb.reshape_(normed, vec![n_out as i64, cfg.merger_in() as i64])
    };

    // ── projector: mm.0 → GELU(erf) → mm.2 ──
    let fc1 = linear(emit, "mm.0.weight", merged)?;
    let fc1_b = emit.load_param("mm.0.bias", false)?;
    let act1 = {
        let mut gb = HirMut::new(emit.hir());
        let x = gb.add(fc1, fc1_b);
        // `nn.GELU()` with PyTorch's default `approximate='none'` — the exact
        // erf form, NOT the tanh approximation the blocks use.
        act(&mut gb, Activation::Gelu, x)
    };
    let fc2 = linear(emit, "mm.2.weight", act1)?;
    let fc2_b = emit.load_param("mm.2.bias", false)?;
    let mut gb = HirMut::new(emit.hir());
    let out = gb.add(fc2, fc2_b);
    Ok(gb.reshape_(out, vec![1, n_out as i64, cfg.llm_hidden_size as i64]))
}

/// One encoder block: LN → attention → residual, LN → FFN → residual.
fn emit_vision_block(
    emit: &mut Emit<'_>,
    prefix: &str,
    h_in: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    cfg: &VisionConfig,
    seq: usize,
) -> Result<HirNodeId> {
    let (n, nh) = (cfg.n_embd, cfg.n_head);
    let dh = cfg.head_dim();
    let (si, ni) = (seq as i64, n as i64);

    let ln1_w = emit.load_param(&format!("{prefix}.ln1.weight"), false)?;
    let ln1_b = emit.load_param(&format!("{prefix}.ln1.bias"), false)?;
    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        let x = gb.ln(h_in, ln1_w, ln1_b, cfg.eps);
        gb.reshape_(x, vec![si, ni])
    };

    let qkv = linear(emit, &format!("{prefix}.attn_qkv.weight"), x2d)?;
    let qkv_b = emit.load_param(&format!("{prefix}.attn_qkv.bias"), false)?;
    let scale = crate::common::scalar(
        emit,
        &format!("{prefix}.sdpa_scale"),
        1.0 / (dh as f32).sqrt(),
    );

    let attn2d = {
        let mut gb = HirMut::new(emit.hir());
        let qkv = gb.add(qkv, qkv_b);
        let qkv3 = gb.reshape_(qkv, vec![1, si, 3 * ni]);
        let q = gb.narrow_(qkv3, 2, 0, n);
        let k = gb.narrow_(qkv3, 2, n, n);
        let v = gb.narrow_(qkv3, 2, 2 * n, n);
        let q = apply_axial_rope(&mut gb, q, cos, sin, seq, nh, dh);
        let k = apply_axial_rope(&mut gb, k, cos, sin, seq, nh, dh);

        // SDPA as explicit matmul → softmax → matmul rather than the fused
        // `Op::Attention`. Bidirectional ViT attention at ~1024 tokens is a
        // shape the fused kernel handles badly on some backends, while batched
        // GEMM is fast everywhere; the two are numerically the same with the
        // default `1/√dh` score scale, which is folded into `q` here.
        let (s, hh, d) = (si, nh as i64, dh as i64);
        let qh = gb.reshape_(q, vec![1, s, hh, d]);
        let qh = gb.transpose_(qh, vec![0, 2, 1, 3]);
        let qh = gb.mul(qh, scale);
        let kh = gb.reshape_(k, vec![1, s, hh, d]);
        let kh = gb.transpose_(kh, vec![0, 2, 1, 3]);
        let vh = gb.reshape_(v, vec![1, s, hh, d]);
        let vh = gb.transpose_(vh, vec![0, 2, 1, 3]);
        let kh_t = gb.transpose_(kh, vec![0, 1, 3, 2]);
        let scores = gb.mm(qh, kh_t);
        let probs = gb.sm(scores, 3);
        let ctx = gb.mm(probs, vh);
        let ctx = gb.transpose_(ctx, vec![0, 2, 1, 3]);
        gb.reshape_(ctx, vec![si, ni])
    };

    let attn_out = linear(emit, &format!("{prefix}.attn_out.weight"), attn2d)?;
    let attn_out_b = emit.load_param(&format!("{prefix}.attn_out.bias"), false)?;
    let ln2_w = emit.load_param(&format!("{prefix}.ln2.weight"), false)?;
    let ln2_b = emit.load_param(&format!("{prefix}.ln2.bias"), false)?;

    let (h_mid, y2d) = {
        let mut gb = HirMut::new(emit.hir());
        let o = gb.add(attn_out, attn_out_b);
        let o3 = gb.reshape_(o, vec![1, si, ni]);
        let h = gb.add(h_in, o3);
        let y = gb.ln(h, ln2_w, ln2_b, cfg.eps);
        (h, gb.reshape_(y, vec![si, ni]))
    };

    // Non-gated MLP: `ffn_up` is fc1, `ffn_down` is fc2.
    let up = linear(emit, &format!("{prefix}.ffn_up.weight"), y2d)?;
    let up_b = emit.load_param(&format!("{prefix}.ffn_up.bias"), false)?;
    let act_out = {
        let mut gb = HirMut::new(emit.hir());
        let x = gb.add(up, up_b);
        // `hidden_act = gelu_pytorch_tanh` — the tanh approximation, unlike the
        // merger's exact erf GELU.
        act(&mut gb, Activation::GeluApprox, x)
    };
    let down = linear(emit, &format!("{prefix}.ffn_down.weight"), act_out)?;
    let down_b = emit.load_param(&format!("{prefix}.ffn_down.bias"), false)?;

    let mut gb = HirMut::new(emit.hir());
    let d = gb.add(down, down_b);
    let d3 = gb.reshape_(d, vec![1, si, ni]);
    Ok(gb.add(h_mid, d3))
}

/// Axial 2-D RoPE over a packed `[1, seq, heads · head_dim]` tensor.
///
/// Rotate-half at `head_dim / 2`, with the tables from
/// [`vision_rope_tables`](super::vision_rope_tables) — whose layout is what makes that split pair row
/// frequencies with row and column with column.
fn apply_axial_rope(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> HirNodeId {
    let f = DType::F32;
    let half = head_dim / 2;
    let (si, hi, di) = (seq as i64, heads as i64, head_dim as i64);
    let x4 = gb.reshape_(x, vec![1, si, hi, di]);
    let cos4 = gb.reshape_(cos, vec![1, si, 1, di]);
    let sin4 = gb.reshape_(sin, vec![1, si, 1, di]);
    let cos4 = gb.expand_(cos4, vec![1, si, hi, di]);
    let sin4 = gb.expand_(sin4, vec![1, si, hi, di]);

    let lo = gb.narrow_(x4, 3, 0, half);
    let hi_part = gb.narrow_(x4, 3, half, half);
    let neg_hi = gb.add_node(
        rlx_ir::op::Op::Activation(Activation::Neg),
        vec![hi_part],
        Shape::new(&[1, seq, heads, half], f),
    );
    let rotated = gb.concat_(vec![neg_hi, lo], 3);

    let a = gb.mul(x4, cos4);
    let b = gb.mul(rotated, sin4);
    let out = gb.add(a, b);
    gb.reshape_(out, vec![1, si, (heads * head_dim) as i64])
}

/// Every tensor the vision tower reads, as GGUF names.
///
/// The checkpoint contract stated once, so it can be checked against a real
/// mmproj header without downloading the weights.
pub fn vision_tensor_names(cfg: &VisionConfig) -> Vec<String> {
    let mut v: Vec<String> = [
        "v.patch_embd.weight",
        "v.patch_embd.weight.1",
        "v.patch_embd.bias",
        "v.position_embd.weight",
        "v.post_ln.weight",
        "v.post_ln.bias",
        "mm.0.weight",
        "mm.0.bias",
        "mm.2.weight",
        "mm.2.bias",
    ]
    .map(String::from)
    .to_vec();
    for il in 0..cfg.n_layer {
        for suffix in [
            "ln1.weight",
            "ln1.bias",
            "ln2.weight",
            "ln2.bias",
            "attn_qkv.weight",
            "attn_qkv.bias",
            "attn_out.weight",
            "attn_out.bias",
            "ffn_up.weight",
            "ffn_up.bias",
            "ffn_down.weight",
            "ffn_down.bias",
        ] {
            v.push(format!("v.blk.{il}.{suffix}"));
        }
    }
    v.sort();
    v
}

/// Shapes for [`vision_tensor_names`], as the GGUF loader reports them
/// (torch `[out, in]`).
pub fn vision_tensor_shapes(
    cfg: &VisionConfig,
    num_position_embeddings: usize,
) -> Vec<(String, Vec<usize>)> {
    let (n, ps, ff) = (cfg.n_embd, cfg.patch_size, cfg.n_ff);
    let mm_in = cfg.merger_in();
    let mut v: Vec<(String, Vec<usize>)> = vec![
        // `Op::Conv`'s weight layout, straight from the file.
        ("v.patch_embd.weight".into(), vec![n, 3, ps, ps]),
        ("v.patch_embd.weight.1".into(), vec![n, 3, ps, ps]),
        ("v.patch_embd.bias".into(), vec![n]),
        (
            "v.position_embd.weight".into(),
            vec![num_position_embeddings, n],
        ),
        ("v.post_ln.weight".into(), vec![n]),
        ("v.post_ln.bias".into(), vec![n]),
        ("mm.0.weight".into(), vec![mm_in, mm_in]),
        ("mm.0.bias".into(), vec![mm_in]),
        ("mm.2.weight".into(), vec![cfg.llm_hidden_size, mm_in]),
        ("mm.2.bias".into(), vec![cfg.llm_hidden_size]),
    ];
    for il in 0..cfg.n_layer {
        let b = format!("v.blk.{il}");
        v.extend([
            (format!("{b}.ln1.weight"), vec![n]),
            (format!("{b}.ln1.bias"), vec![n]),
            (format!("{b}.ln2.weight"), vec![n]),
            (format!("{b}.ln2.bias"), vec![n]),
            (format!("{b}.attn_qkv.weight"), vec![3 * n, n]),
            (format!("{b}.attn_qkv.bias"), vec![3 * n]),
            (format!("{b}.attn_out.weight"), vec![n, n]),
            (format!("{b}.attn_out.bias"), vec![n]),
            (format!("{b}.ffn_up.weight"), vec![ff, n]),
            (format!("{b}.ffn_up.bias"), vec![ff]),
            (format!("{b}.ffn_down.weight"), vec![n, ff]),
            (format!("{b}.ffn_down.bias"), vec![n]),
        ]);
    }
    v
}

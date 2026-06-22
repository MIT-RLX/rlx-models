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

//! Native SAM3 ViT trunk.
//!
//! Full numerical port of `sam3.model.vitdet.ViT` (the base 1008² model):
//!
//!   - Patch embed (Conv2d k=14, s=14, no bias) → [B, H, W, C] in NHWC.
//!   - Tiled absolute positional embedding (24×24 → 72×72).
//!   - 32 transformer blocks. Blocks 0..32 are window-attention with
//!     `window_size=24` except for `global_att_blocks=[7,15,23,31]` which
//!     run full-resolution attention.
//!   - 2D RoPE applied to Q/K, interpolated when input differs from
//!     `rope_pt_size=(24, 24)`.
//!   - LayerNorm `ln_pre`, identity `ln_post`, `mlp_ratio=4.625`.
//!
//! Output: final block features in NHWC `[1, 72, 72, 1024]`, flattened
//! row-major as `[grid*grid, embed_dim]`.

use super::config::{SAM3_PATCH_GRID, Sam3VitConfig};
use super::preprocess::{Sam3PreprocessWeights, assemble_patch_tokens, extract_preprocess_weights};
use super::tensor::{gelu_tanh, layer_norm, linear, matmul, matmul_bt, softmax_rows};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{GgufPackedLinear, GgufPackedParams};

#[derive(Clone)]
pub struct Sam3VitBlockWeights {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub qkv_w_t: Vec<f32>,
    pub qkv_b: Vec<f32>,
    /// GGUF prefix for `attn.qkv` (no `.weight` suffix) when `qkv_w_t` is empty.
    pub qkv_gguf_prefix: Option<String>,
    pub proj_w_t: Vec<f32>,
    pub proj_b: Vec<f32>,
    pub proj_gguf_prefix: Option<String>,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub mlp_fc1_w_t: Vec<f32>,
    pub mlp_fc1_b: Vec<f32>,
    pub mlp_fc1_gguf_prefix: Option<String>,
    pub mlp_fc2_w_t: Vec<f32>,
    pub mlp_fc2_b: Vec<f32>,
    pub mlp_fc2_gguf_prefix: Option<String>,
}

#[derive(Clone)]
pub struct Sam3VisionEncoderWeights {
    pub pre: Sam3PreprocessWeights,
    pub ln_pre_w: Vec<f32>,
    pub ln_pre_b: Vec<f32>,
    pub blocks: Vec<Sam3VitBlockWeights>,
}

pub struct Sam3VisionOutput {
    pub tokens: Vec<f32>,
    pub grid: usize,
    pub dim: usize,
}

pub fn extract_vision_encoder_weights(
    weights: &mut WeightMap,
    cfg: &Sam3VitConfig,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Sam3VisionEncoderWeights> {
    let pre = extract_preprocess_weights(weights, cfg)?;
    let e = cfg.embed_dim;
    let (ln_pre_w, ln_pre_b) = take_layer_norm(weights, &prefixes("ln_pre"), e)?;
    let hidden = (e as f64 * cfg.mlp_ratio) as usize;
    let mut blocks = Vec::with_capacity(cfg.depth);
    for i in 0..cfg.depth {
        let p = format!("blocks.{i}");
        let pref = prefixes(&p);
        let (norm1_w, norm1_b) = take_layer_norm(weights, &prefixed(&pref, "norm1"), e)?;
        let (qkv_w_t, qkv_gguf_prefix) =
            take_linear_w_or_gguf(weights, gguf_packed, &prefixed(&pref, "attn.qkv"), e, 3 * e)?;
        let qkv_b = take_linear_b(weights, &prefixed(&pref, "attn.qkv"), 3 * e)?;
        let (proj_w_t, proj_gguf_prefix) =
            take_linear_w_or_gguf(weights, gguf_packed, &prefixed(&pref, "attn.proj"), e, e)?;
        let proj_b = take_linear_b(weights, &prefixed(&pref, "attn.proj"), e)?;
        let (norm2_w, norm2_b) = take_layer_norm(weights, &prefixed(&pref, "norm2"), e)?;
        let (mlp_fc1_w_t, mlp_fc1_gguf_prefix) = take_linear_w_any_or_gguf(
            weights,
            gguf_packed,
            &pref,
            &["mlp.fc1", "mlp.lin1"],
            e,
            hidden,
        )?;
        let mlp_fc1_b = take_linear_b_any(weights, &pref, &["mlp.fc1", "mlp.lin1"], hidden)?;
        let (mlp_fc2_w_t, mlp_fc2_gguf_prefix) = take_linear_w_any_or_gguf(
            weights,
            gguf_packed,
            &pref,
            &["mlp.fc2", "mlp.lin2"],
            hidden,
            e,
        )?;
        let mlp_fc2_b = take_linear_b_any(weights, &pref, &["mlp.fc2", "mlp.lin2"], e)?;
        blocks.push(Sam3VitBlockWeights {
            norm1_w,
            norm1_b,
            qkv_w_t,
            qkv_b,
            qkv_gguf_prefix,
            proj_w_t,
            proj_b,
            proj_gguf_prefix,
            norm2_w,
            norm2_b,
            mlp_fc1_w_t,
            mlp_fc1_b,
            mlp_fc1_gguf_prefix,
            mlp_fc2_w_t,
            mlp_fc2_b,
            mlp_fc2_gguf_prefix,
        });
    }
    Ok(Sam3VisionEncoderWeights {
        pre,
        ln_pre_w,
        ln_pre_b,
        blocks,
    })
}

/// Run the 32 ViT-L transformer blocks on already-tokenized input
/// `[grid*grid, embed_dim]`. Intended for parity tests against the IR path,
/// which also operates on post-`ln_pre` tokens.
pub fn forward_blocks_native(
    weights: &Sam3VisionEncoderWeights,
    gguf_packed: Option<&GgufPackedParams>,
    cfg: &Sam3VitConfig,
    tokens_in: &[f32],
) -> Result<Vec<f32>> {
    let e = cfg.embed_dim;
    let grid = cfg.patch_grid();
    let head_dim = e / cfg.num_heads;
    ensure!(
        head_dim * cfg.num_heads == e,
        "embed_dim {e} not divisible by num_heads {}",
        cfg.num_heads
    );
    ensure!(
        tokens_in.len() == grid * grid * e,
        "tokens_in expected {} got {}",
        grid * grid * e,
        tokens_in.len()
    );
    let rope_pt = if cfg.window_size > 0 {
        cfg.window_size
    } else {
        grid
    };
    let global_set: std::collections::HashSet<usize> =
        cfg.global_att_blocks.iter().copied().collect();
    let rope_global = build_rope_freqs(head_dim, grid, grid, 10000.0, rope_pt as f32 / grid as f32);
    let rope_window = build_rope_freqs(head_dim, cfg.window_size, cfg.window_size, 10000.0, 1.0);
    let mut x = tokens_in.to_vec();
    for (i, block) in weights.blocks.iter().enumerate() {
        let is_global = global_set.contains(&i);
        block_forward(
            &mut x,
            block,
            gguf_packed,
            cfg,
            grid,
            if is_global { 0 } else { cfg.window_size },
            if is_global {
                &rope_global
            } else {
                &rope_window
            },
            head_dim,
            cfg.num_heads,
        )?;
    }
    Ok(x)
}

pub fn encode_image_native(
    weights: &Sam3VisionEncoderWeights,
    gguf_packed: Option<&GgufPackedParams>,
    cfg: &Sam3VitConfig,
    image_nchw: &[f32],
) -> Result<Sam3VisionOutput> {
    let e = cfg.embed_dim;
    let grid = cfg.patch_grid();
    ensure!(
        grid == SAM3_PATCH_GRID,
        "SAM3 base grid must be {SAM3_PATCH_GRID}"
    );
    let head_dim = e / cfg.num_heads;
    ensure!(
        head_dim * cfg.num_heads == e,
        "embed_dim {e} not divisible by num_heads {}",
        cfg.num_heads
    );
    let rope_pt = if cfg.window_size > 0 {
        cfg.window_size
    } else {
        grid
    };

    // Patch embed (+ tiled abs pos), flat [grid*grid, embed_dim] NHWC.
    let mut x = assemble_patch_tokens(&weights.pre, image_nchw)?;
    x = layer_norm(
        &x,
        &weights.ln_pre_w,
        &weights.ln_pre_b,
        e,
        cfg.layer_norm_eps as f32,
    )?;

    let global_set: std::collections::HashSet<usize> =
        cfg.global_att_blocks.iter().copied().collect();
    let rope_global = build_rope_freqs(head_dim, grid, grid, 10000.0, rope_pt as f32 / grid as f32);
    let rope_window = build_rope_freqs(head_dim, cfg.window_size, cfg.window_size, 10000.0, 1.0);

    for (i, block) in weights.blocks.iter().enumerate() {
        let is_global = global_set.contains(&i);
        block_forward(
            &mut x,
            block,
            gguf_packed,
            cfg,
            grid,
            if is_global { 0 } else { cfg.window_size },
            if is_global {
                &rope_global
            } else {
                &rope_window
            },
            head_dim,
            cfg.num_heads,
        )?;
    }
    // ln_post is Identity for SAM3 base, no-op.

    Ok(Sam3VisionOutput {
        tokens: x,
        grid,
        dim: e,
    })
}

/// Compute the 2D RoPE frequency table. The layout is `[L, head_dim]` flat,
/// with each `head_dim`-long stride storing `head_dim/2` interleaved
/// `(cos, sin)` pairs — first half from the x axis, second from the y axis.
fn build_rope_freqs(
    head_dim: usize,
    end_x: usize,
    end_y: usize,
    theta: f32,
    scale_pos: f32,
) -> Vec<f32> {
    let half = head_dim / 2;
    assert!(
        head_dim.is_multiple_of(4),
        "RoPE head_dim must be divisible by 4"
    );
    let pair_per_axis = head_dim / 4;
    let mut freqs_per_pair = Vec::with_capacity(pair_per_axis);
    for k in 0..pair_per_axis {
        let exp = (4 * k) as f32 / head_dim as f32;
        freqs_per_pair.push(1.0 / theta.powf(exp));
    }
    let l = end_x * end_y;
    let mut out = vec![0f32; l * head_dim];
    for pos in 0..l {
        let t_x = (pos % end_x) as f32 * scale_pos;
        let t_y = (pos / end_x) as f32 * scale_pos;
        for k in 0..pair_per_axis {
            let ang_x = t_x * freqs_per_pair[k];
            let ang_y = t_y * freqs_per_pair[k];
            out[pos * head_dim + 2 * k] = ang_x.cos();
            out[pos * head_dim + 2 * k + 1] = ang_x.sin();
            out[pos * head_dim + 2 * (k + pair_per_axis)] = ang_y.cos();
            out[pos * head_dim + 2 * (k + pair_per_axis) + 1] = ang_y.sin();
        }
    }
    let _ = half;
    out
}

/// Apply RoPE in-place to `qk` of shape `[batch_eff * num_heads * L, head_dim]`.
/// `freqs_cis` is `[L, head_dim]` (real, imag pairs) and broadcasts over the
/// outer batch×head axis.
fn rope_apply_inplace(
    qk: &mut [f32],
    freqs_cis: &[f32],
    rows: usize,
    seq_len: usize,
    head_dim: usize,
) {
    let pairs = head_dim / 2;
    for r in 0..rows {
        let l = r % seq_len;
        let f = &freqs_cis[l * head_dim..(l + 1) * head_dim];
        let v = &mut qk[r * head_dim..(r + 1) * head_dim];
        for k in 0..pairs {
            let vr = v[2 * k];
            let vi = v[2 * k + 1];
            let fr = f[2 * k];
            let fi = f[2 * k + 1];
            v[2 * k] = vr * fr - vi * fi;
            v[2 * k + 1] = vr * fi + vi * fr;
        }
    }
}

/// One transformer block: norm → (windowed) attention → residual →
/// norm → MLP → residual. `x` is `[grid*grid, embed_dim]` NHWC flat.
fn linear_maybe_gguf(
    x: &[f32],
    m: usize,
    k: usize,
    w_t: &[f32],
    gguf: Option<&GgufPackedLinear>,
    n: usize,
    b: &[f32],
) -> Result<Vec<f32>> {
    let mut out = vec![0f32; m * n];
    if let Some(p) = gguf {
        ensure!(
            p.in_dim == k && p.out_dim == n,
            "packed linear shape {k}x{n} vs gguf {}x{}",
            p.in_dim,
            p.out_dim
        );
        rlx_cpu::gguf_matmul::gguf_matmul_bt(x, &p.w_q, &mut out, m, k, n, p.scheme);
    } else {
        ensure!(
            !w_t.is_empty(),
            "linear: missing F32 weights and no GGUF packed entry"
        );
        return linear(x, m, k, w_t, n, b);
    }
    for row in 0..m {
        for col in 0..n {
            out[row * n + col] += b[col];
        }
    }
    Ok(out)
}

fn packed_for_prefix<'a>(
    packed: Option<&'a GgufPackedParams>,
    prefix: Option<&String>,
) -> Option<&'a GgufPackedLinear> {
    let key = format!("{}.weight", prefix.as_ref()?);
    packed?.get_linear(&key)
}

fn block_forward(
    x: &mut [f32],
    block: &Sam3VitBlockWeights,
    gguf_packed: Option<&GgufPackedParams>,
    cfg: &Sam3VitConfig,
    grid: usize,
    window_size: usize,
    freqs_cis: &[f32],
    head_dim: usize,
    num_heads: usize,
) -> Result<()> {
    let e = cfg.embed_dim;
    let n = grid * grid;
    let eps = cfg.layer_norm_eps as f32;

    // shortcut: x as-is. Compute attention(norm1(x)) in attn_out.
    let n1 = layer_norm(x, &block.norm1_w, &block.norm1_b, e, eps)?;
    let qkv_gguf = packed_for_prefix(gguf_packed, block.qkv_gguf_prefix.as_ref());
    let proj_gguf = packed_for_prefix(gguf_packed, block.proj_gguf_prefix.as_ref());
    let attn_out = if window_size == 0 {
        attention_native(
            &n1,
            1,
            n,
            &block.qkv_w_t,
            qkv_gguf,
            &block.qkv_b,
            &block.proj_w_t,
            proj_gguf,
            &block.proj_b,
            freqs_cis,
            num_heads,
            head_dim,
        )?
    } else {
        attention_windowed(
            &n1,
            grid,
            grid,
            window_size,
            e,
            &block.qkv_w_t,
            qkv_gguf,
            &block.qkv_b,
            &block.proj_w_t,
            proj_gguf,
            &block.proj_b,
            freqs_cis,
            num_heads,
            head_dim,
        )?
    };
    for i in 0..x.len() {
        x[i] += attn_out[i];
    }

    let n2 = layer_norm(x, &block.norm2_w, &block.norm2_b, e, eps)?;
    let hidden = block.mlp_fc1_b.len();
    let fc1_gguf = packed_for_prefix(gguf_packed, block.mlp_fc1_gguf_prefix.as_ref());
    let fc2_gguf = packed_for_prefix(gguf_packed, block.mlp_fc2_gguf_prefix.as_ref());
    let mut mlp = linear_maybe_gguf(
        &n2,
        n,
        e,
        &block.mlp_fc1_w_t,
        fc1_gguf,
        hidden,
        &block.mlp_fc1_b,
    )?;
    gelu_tanh(&mut mlp);
    let ffn = linear_maybe_gguf(
        &mlp,
        n,
        hidden,
        &block.mlp_fc2_w_t,
        fc2_gguf,
        e,
        &block.mlp_fc2_b,
    )?;
    for i in 0..x.len() {
        x[i] += ffn[i];
    }
    Ok(())
}

fn attention_windowed(
    x: &[f32],
    h: usize,
    w: usize,
    ws: usize,
    e: usize,
    qkv_w_t: &[f32],
    qkv_gguf: Option<&GgufPackedLinear>,
    qkv_b: &[f32],
    proj_w_t: &[f32],
    proj_gguf: Option<&GgufPackedLinear>,
    proj_b: &[f32],
    freqs_cis: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let pad_h = (ws - h % ws) % ws;
    let pad_w = (ws - w % ws) % ws;
    let hp = h + pad_h;
    let wp = w + pad_w;
    let nh = hp / ws;
    let nw = wp / ws;
    let num_windows = nh * nw;
    let win_len = ws * ws;

    // Partition: produce [num_windows, ws, ws, e].
    let mut win = vec![0f32; num_windows * win_len * e];
    for y in 0..hp {
        for xc in 0..wp {
            let wy = y / ws;
            let wx = xc / ws;
            let ry = y % ws;
            let rx = xc % ws;
            let widx = wy * nw + wx;
            let dst = ((widx * ws + ry) * ws + rx) * e;
            if y < h && xc < w {
                let src = (y * w + xc) * e;
                win[dst..dst + e].copy_from_slice(&x[src..src + e]);
            }
            // else: padding stays zero (matches F.pad with 0).
        }
    }

    let attn = attention_native(
        &win,
        num_windows,
        win_len,
        qkv_w_t,
        qkv_gguf,
        qkv_b,
        proj_w_t,
        proj_gguf,
        proj_b,
        freqs_cis,
        num_heads,
        head_dim,
    )?;

    // Unpartition: stitch [num_windows, ws, ws, e] back into [h, w, e],
    // dropping padding.
    let mut out = vec![0f32; h * w * e];
    for y in 0..h {
        for xc in 0..w {
            let wy = y / ws;
            let wx = xc / ws;
            let ry = y % ws;
            let rx = xc % ws;
            let widx = wy * nw + wx;
            let src = ((widx * ws + ry) * ws + rx) * e;
            let dst = (y * w + xc) * e;
            out[dst..dst + e].copy_from_slice(&attn[src..src + e]);
        }
    }
    Ok(out)
}

/// Multi-head self-attention with 2D RoPE for `b` independent sequences of
/// length `l`. `x` is `[b, l, e]`; output is `[b, l, e]`.
fn attention_native(
    x: &[f32],
    b: usize,
    l: usize,
    qkv_w_t: &[f32],
    qkv_gguf: Option<&GgufPackedLinear>,
    qkv_b: &[f32],
    proj_w_t: &[f32],
    proj_gguf: Option<&GgufPackedLinear>,
    proj_b: &[f32],
    freqs_cis: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let e = num_heads * head_dim;
    let rows = b * l;
    let qkv = linear_maybe_gguf(x, rows, e, qkv_w_t, qkv_gguf, 3 * e, qkv_b)?;

    // Split into [b, num_heads, l, head_dim] for q, k, v. We keep them as
    // [b*num_heads, l, head_dim] = [bh, l, head_dim] to feed sgemm.
    let bh = b * num_heads;
    let mut q = vec![0f32; bh * l * head_dim];
    let mut k = vec![0f32; bh * l * head_dim];
    let mut v = vec![0f32; bh * l * head_dim];
    for bi in 0..b {
        for li in 0..l {
            let src = (bi * l + li) * 3 * e;
            for hd in 0..num_heads {
                let qd_src = src + hd * head_dim;
                let kd_src = src + e + hd * head_dim;
                let vd_src = src + 2 * e + hd * head_dim;
                let dst = ((bi * num_heads + hd) * l + li) * head_dim;
                q[dst..dst + head_dim].copy_from_slice(&qkv[qd_src..qd_src + head_dim]);
                k[dst..dst + head_dim].copy_from_slice(&qkv[kd_src..kd_src + head_dim]);
                v[dst..dst + head_dim].copy_from_slice(&qkv[vd_src..vd_src + head_dim]);
            }
        }
    }

    rope_apply_inplace(&mut q, freqs_cis, bh * l, l, head_dim);
    rope_apply_inplace(&mut k, freqs_cis, bh * l, l, head_dim);

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0f32; bh * l * head_dim];
    let mut scores = vec![0f32; l * l];

    for bhi in 0..bh {
        let q_h = &q[bhi * l * head_dim..(bhi + 1) * l * head_dim];
        let k_h = &k[bhi * l * head_dim..(bhi + 1) * l * head_dim];
        let v_h = &v[bhi * l * head_dim..(bhi + 1) * l * head_dim];
        // scores[l, l] = scale * Q[l, hd] @ K[l, hd]^T
        matmul_bt(q_h, k_h, &mut scores, l, head_dim, l, scale);
        softmax_rows(&mut scores, l, l);
        // out[l, hd] = scores[l, l] @ V[l, hd]
        let out_h = &mut attn_out[bhi * l * head_dim..(bhi + 1) * l * head_dim];
        matmul(&scores, v_h, out_h, l, l, head_dim);
    }

    // Repack [b, num_heads, l, head_dim] → [b, l, num_heads*head_dim] for proj.
    let mut packed = vec![0f32; rows * e];
    for bi in 0..b {
        for li in 0..l {
            for hd in 0..num_heads {
                let src = ((bi * num_heads + hd) * l + li) * head_dim;
                let dst = (bi * l + li) * e + hd * head_dim;
                packed[dst..dst + head_dim].copy_from_slice(&attn_out[src..src + head_dim]);
            }
        }
    }
    linear_maybe_gguf(&packed, rows, e, proj_w_t, proj_gguf, e, proj_b)
}

fn prefixes(suffix: &str) -> Vec<String> {
    [
        "detector.backbone.vision_backbone.trunk",
        "detector.backbone.visual.trunk",
        "backbone.vision_backbone.trunk",
        "backbone.visual.trunk",
        "visual.trunk",
        "trunk",
    ]
    .iter()
    .map(|p| format!("{p}.{suffix}"))
    .collect()
}

fn prefixed(prefixes: &[String], suffix: &str) -> Vec<String> {
    prefixes.iter().map(|p| format!("{p}.{suffix}")).collect()
}

fn take_layer_norm(
    weights: &mut WeightMap,
    bases: &[String],
    dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let w = take_shape(weights, &suffixes(bases, "weight"), &[dim])?;
    let b = take_shape(weights, &suffixes(bases, "bias"), &[dim])?;
    Ok((w, b))
}

fn take_linear_w_or_gguf(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
    bases: &[String],
    in_dim: usize,
    out_dim: usize,
) -> Result<(Vec<f32>, Option<String>)> {
    let keys = suffixes(bases, "weight");
    for key in &keys {
        if weights.has(key) {
            let w = take_linear_w(weights, bases, in_dim, out_dim)?;
            return Ok((w, None));
        }
        if let Some(packed) = gguf_packed {
            if let Some(prefix) = key.strip_suffix(".weight") {
                if packed.get_linear(key).is_some() {
                    return Ok((Vec::new(), Some(prefix.to_string())));
                }
            }
        }
    }
    anyhow::bail!("none of the SAM3 linear weight keys were found: {keys:?}")
}

fn take_linear_w_any_or_gguf(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
    block_prefixes: &[String],
    names: &[&str],
    in_dim: usize,
    out_dim: usize,
) -> Result<(Vec<f32>, Option<String>)> {
    let bases: Vec<String> = block_prefixes
        .iter()
        .flat_map(|p| names.iter().map(move |name| format!("{p}.{name}")))
        .collect();
    take_linear_w_or_gguf(weights, gguf_packed, &bases, in_dim, out_dim)
}

fn take_linear_w(
    weights: &mut WeightMap,
    bases: &[String],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    let keys = suffixes(bases, "weight");
    for key in &keys {
        if weights.has(key) {
            let (data, shape) = weights.take_transposed(key)?;
            ensure!(
                shape == vec![in_dim, out_dim],
                "{key} expected [{in_dim}, {out_dim}], got {shape:?}"
            );
            return Ok(data);
        }
    }
    anyhow::bail!("none of the SAM3 linear weight keys were found: {keys:?}")
}

#[allow(dead_code)]
fn take_linear_w_any(
    weights: &mut WeightMap,
    block_prefixes: &[String],
    names: &[&str],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    let bases: Vec<String> = block_prefixes
        .iter()
        .flat_map(|p| names.iter().map(move |name| format!("{p}.{name}")))
        .collect();
    take_linear_w(weights, &bases, in_dim, out_dim)
}

fn take_linear_b(weights: &mut WeightMap, bases: &[String], dim: usize) -> Result<Vec<f32>> {
    take_shape(weights, &suffixes(bases, "bias"), &[dim])
}

fn take_linear_b_any(
    weights: &mut WeightMap,
    block_prefixes: &[String],
    names: &[&str],
    dim: usize,
) -> Result<Vec<f32>> {
    let bases: Vec<String> = block_prefixes
        .iter()
        .flat_map(|p| names.iter().map(move |name| format!("{p}.{name}")))
        .collect();
    take_linear_b(weights, &bases, dim)
}

fn suffixes(bases: &[String], suffix: &str) -> Vec<String> {
    bases.iter().map(|b| format!("{b}.{suffix}")).collect()
}

fn take_shape(weights: &mut WeightMap, keys: &[String], expected: &[usize]) -> Result<Vec<f32>> {
    for key in keys {
        if weights.has(key) {
            let (data, shape) = weights.take(key)?;
            ensure!(
                shape == expected,
                "{key} expected {expected:?}, got {shape:?}"
            );
            return Ok(data);
        }
    }
    anyhow::bail!("none of the SAM3 weight keys were found: {keys:?}")
}

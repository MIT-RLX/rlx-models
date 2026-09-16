// RLX — versatile ML compiler + runtime. GPLv3.
//! **DeepSeek-V4.1 vision tower** — the DeepSeek-ViT encoder plus the aligner
//! that maps its patch grid into the language model's embedding space.
//!
//! The ViT is a plain pre-norm transformer over one image's patches with **full
//! bidirectional attention** and **2-D RoPE**: the first half of each head's
//! rotary pairs is driven by the patch's row, the second half by its column. The
//! aligner then folds a `downsample_ratio × downsample_ratio` neighbourhood of
//! patches into one language token (a strided unfold, zero-padded to a whole
//! number of blocks) and runs it through a two-layer GELU MLP.
//!
//! The result replaces the `IMAGE` slots of the prompt in row-major order; the
//! three span delimiters (`image_start` / `image_end` / `image_newline`) are
//! learned embeddings the caller splices in.
//!
//! Reference: `deepseek-ai/DeepSeek-V4.1-Flash` `inference/vision.py`.

use crate::dsv41::VisionSpec;
use crate::standard_decoder::{load_norm, load_p, synth_const, synth_zero};
use crate::weight_loader::WeightLoader;
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// The ViT's RMSNorm epsilon. It is the `vision.py` default and is **not** the
/// language model's `norm_eps` (1e-20 in the GA checkpoint) — using that one
/// here would divide by an unregularized norm.
pub const VISION_NORM_EPS: f32 = 1e-6;

/// 2-D rotary tables for an `n_h × n_w` patch grid.
///
/// `rope_dim = vision_dim / n_heads / 2`, and each position's frequency vector is
/// `[row · inv_freq, col · inv_freq]` — so the concatenated table is `head_dim/2`
/// wide, exactly the half-split NeoX rotation needs.
fn vision_rope_tables(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    n_h: usize,
    n_w: usize,
    rope_dim: usize,
    theta: f64,
) -> (NodeId, NodeId) {
    let half = rope_dim / 2;
    let n = n_h * n_w;
    let width = 2 * half;
    let mut cos = vec![0f32; n * width];
    let mut sin = vec![0f32; n * width];
    for r in 0..n_h {
        for c in 0..n_w {
            let p = r * n_w + c;
            for i in 0..half {
                let inv = 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64);
                let (sh, ch) = (r as f64 * inv).sin_cos();
                let (sw, cw) = (c as f64 * inv).sin_cos();
                cos[p * width + i] = ch as f32;
                sin[p * width + i] = sh as f32;
                cos[p * width + half + i] = cw as f32;
                sin[p * width + half + i] = sw as f32;
            }
        }
    }
    (
        synth_const(g, params, "v41.vit.rope.cos", cos, &[n, width]),
        synth_const(g, params, "v41.vit.rope.sin", sin, &[n, width]),
    )
}

/// `cat([x1·cos - x2·sin, x2·cos + x1·sin])` over the two halves of each head —
/// the reference `apply_rotary`. `x` is `[n, heads·head_dim]`.
fn vision_rope(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    n: usize,
    heads: usize,
    head_dim: usize,
) -> NodeId {
    let half = head_dim / 2;
    let x3 = g.reshape_(x, vec![n as i64, heads as i64, head_dim as i64]);
    let x1 = g.narrow_(x3, 2, 0, half);
    let x2 = g.narrow_(x3, 2, half, half);
    // the table is per-position, shared across heads
    let cos3 = g.reshape_(cos, vec![n as i64, 1, half as i64]);
    let sin3 = g.reshape_(sin, vec![n as i64, 1, half as i64]);
    let a = g.mul(x1, cos3);
    let b = g.mul(x2, sin3);
    let lo = g.sub(a, b);
    let c = g.mul(x2, cos3);
    let d = g.mul(x1, sin3);
    let hi = g.add(c, d);
    let cat = g.concat_(vec![lo, hi], 2);
    g.reshape_(cat, vec![n as i64, (heads * head_dim) as i64])
}

/// A `[out, in]` weight + optional `[out]` bias applied to `[n, in]`.
fn linear(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
    x: NodeId,
    n: usize,
    out: usize,
    bias: bool,
) -> Result<NodeId> {
    let w = load_p(g, params, weights, &format!("{key}.weight"), true)?;
    let mut y = g.mm(x, w);
    if bias {
        let b = load_p(g, params, weights, &format!("{key}.bias"), false)?;
        let b2 = g.reshape_(b, vec![1, out as i64]);
        y = g.add(y, b2);
    }
    Ok(g.reshape_(y, vec![n as i64, out as i64]))
}

/// Encode one image: `patches [n_h·n_w, 3·patch·patch]` → language-model
/// embeddings `[n_tokens, dim]`, where `n_tokens = ceil(n_h/r) · ceil(n_w/r)`.
///
/// `patches` must already be flattened per patch, in row-major grid order — the
/// reference's `PatchEmbed` does `proj(x.flatten(1))` on an `[N, 3, p, p]` stack,
/// which is the same memory.
pub fn build_v41_vision(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    spec: &VisionSpec,
    lm_dim: usize,
    patches: NodeId,
    n_h: usize,
    n_w: usize,
) -> Result<NodeId> {
    if spec.n_heads == 0 || !spec.dim.is_multiple_of(spec.n_heads) {
        return Err(anyhow!(
            "deepseek_v41 vision: dim {} is not divisible by n_heads {}",
            spec.dim,
            spec.n_heads
        ));
    }
    let n = n_h * n_w;
    let vd = spec.dim;
    let head_dim = vd / spec.n_heads;
    let rope_dim = head_dim / 2;
    let eps = VISION_NORM_EPS;
    let zb = synth_zero(g, params, "v41.vit.zb", vd);

    let mut x = linear(
        g,
        params,
        weights,
        "vision.patch_embed.proj",
        patches,
        n,
        vd,
        true,
    )?;
    let (cos, sin) = vision_rope_tables(g, params, n_h, n_w, rope_dim, spec.rope_theta);

    for il in 0..spec.n_layers {
        let bp = format!("vision.blocks.{il}");
        let n1 = load_norm(g, params, weights, &format!("{bp}.norm1.weight"), 0.0)?;
        let h = g.rms_norm(x, n1, zb, eps);
        let qkv = linear(g, params, weights, &format!("{bp}.attn.wqkv"), h, n, 3 * vd, true)?;
        let q = g.narrow_(qkv, 1, 0, vd);
        let k = g.narrow_(qkv, 1, vd, vd);
        let v = g.narrow_(qkv, 1, 2 * vd, vd);
        let q = vision_rope(g, q, cos, sin, n, spec.n_heads, head_dim);
        let k = vision_rope(g, k, cos, sin, n, spec.n_heads, head_dim);
        // full bidirectional attention over the single image's patches
        let attn = g.attention_kind(
            q,
            k,
            v,
            spec.n_heads,
            head_dim,
            MaskKind::None,
            Shape::new(&[n, vd], DType::F32),
        );
        let o = linear(g, params, weights, &format!("{bp}.attn.wo"), attn, n, vd, true)?;
        x = g.add(x, o);

        let n2 = load_norm(g, params, weights, &format!("{bp}.norm2.weight"), 0.0)?;
        let h = g.rms_norm(x, n2, zb, eps);
        // w1 emits gate and up fused: [2·inter, dim]
        let gu = linear(
            g,
            params,
            weights,
            &format!("{bp}.mlp.w1"),
            h,
            n,
            2 * spec.inter_dim,
            false,
        )?;
        let gate = g.narrow_(gu, 1, 0, spec.inter_dim);
        let up = g.narrow_(gu, 1, spec.inter_dim, spec.inter_dim);
        let act = g.silu(gate);
        let glu = g.mul(act, up);
        let down = linear(g, params, weights, &format!("{bp}.mlp.w2"), glu, n, vd, false)?;
        x = g.add(x, down);
    }
    let nf = load_norm(g, params, weights, "vision.norm.weight", 0.0)?;
    let x = g.rms_norm(x, nf, zb, eps);

    build_v41_aligner(g, params, weights, spec, lm_dim, x, n_h, n_w)
}

/// The aligner: fold each `r × r` patch neighbourhood into one token, then a
/// two-layer GELU MLP into the language model's width.
///
/// The grid is zero-padded on the right and bottom to a whole number of blocks
/// (`F.pad(x, (0, -n_w % r, 0, -n_h % r))`), and the unfold lays each block out
/// **channel-major then row-major within the block**, which is what
/// `F.unfold`'s `[C·r·r, L]` ordering means.
#[allow(clippy::too_many_arguments)]
pub fn build_v41_aligner(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    spec: &VisionSpec,
    lm_dim: usize,
    x: NodeId, // [n_h·n_w, vision_dim]
    n_h: usize,
    n_w: usize,
) -> Result<NodeId> {
    let r = spec.downsample_ratio.max(1);
    let vd = spec.dim;
    let (ph, pw) = (n_h.div_ceil(r) * r, n_w.div_ceil(r) * r);
    let (bh, bw) = (ph / r, pw / r);
    let tokens = bh * bw;
    let in_dim = vd * r * r;

    // Zero-pad the grid, then gather the unfold pattern in one shot: entry
    // (block, c·r·r + dy·r + dx) reads patch (by·r+dy, bx·r+dx), channel c.
    let padded = if ph != n_h || pw != n_w {
        // build a [ph·pw, vd] grid: real rows where inside, zero row otherwise
        let zero = synth_const(g, params, "v41.aligner.zero", vec![0f32; vd], &[1, vd]);
        let mut rows: Vec<NodeId> = Vec::with_capacity(ph * pw);
        for y in 0..ph {
            for xx in 0..pw {
                rows.push(if y < n_h && xx < n_w {
                    g.narrow_(x, 0, y * n_w + xx, 1)
                } else {
                    zero
                });
            }
        }
        g.concat_(rows, 0)
    } else {
        x
    };

    // [tokens, r·r, vd] → transpose to channel-major → [tokens, vd·r·r]
    let mut picks: Vec<NodeId> = Vec::with_capacity(tokens * r * r);
    for by in 0..bh {
        for bx in 0..bw {
            for dy in 0..r {
                for dx in 0..r {
                    let src = (by * r + dy) * pw + (bx * r + dx);
                    picks.push(g.narrow_(padded, 0, src, 1));
                }
            }
        }
    }
    let gathered = g.concat_(picks, 0); // [tokens·r·r, vd]
    let g3 = g.reshape_(gathered, vec![tokens as i64, (r * r) as i64, vd as i64]);
    let g3 = g.transpose_(g3, vec![0, 2, 1]); // [tokens, vd, r·r] — channel-major
    let flat = g.reshape_(g3, vec![tokens as i64, in_dim as i64]);

    let h = linear(g, params, weights, "aligner.w1", flat, tokens, lm_dim, true)?;
    let h = g.gelu(h);
    linear(g, params, weights, "aligner.w2", h, tokens, lm_dim, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_are_row_then_column() {
        let mut g = Graph::new("t");
        let mut p = HashMap::new();
        let (cos, _sin) = vision_rope_tables(&mut g, &mut p, 2, 3, 4, 10000.0);
        let _ = cos;
        let c = &p["v41.vit.rope.cos"];
        // rope_dim 4 → half 2 → width 4; 6 positions
        assert_eq!(c.len(), 6 * 4);
        // position (0,0) is the identity rotation
        assert!(c[..4].iter().all(|v| (*v - 1.0).abs() < 1e-6));
        // position (0,2): row 0 leaves the first half at cos 0 = 1, column 2
        // drives the second half
        let p02 = &c[2 * 4..3 * 4];
        assert!((p02[0] - 1.0).abs() < 1e-6 && (p02[1] - 1.0).abs() < 1e-6);
        assert!((p02[2] - 2f32.cos()).abs() < 1e-5);
        // position (1,0): row 1 drives the first half, column 0 the second
        let p10 = &c[3 * 4..4 * 4];
        assert!((p10[0] - 1f32.cos()).abs() < 1e-5);
        assert!((p10[2] - 1.0).abs() < 1e-6 && (p10[3] - 1.0).abs() < 1e-6);
    }
}

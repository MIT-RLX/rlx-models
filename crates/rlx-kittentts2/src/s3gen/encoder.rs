// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! S3Gen's `UpsampleConformerEncoder` + `encoder_proj`: S3 tokens (25 Hz) →
//! the flow's 50 Hz conditioning `mu` (`[80, 2T]`).
//!
//! ```text
//! embed(tokens) ─ Linear+LN ─ ×√512 ─ PreLookahead(3) ─ 6 × conformer ─┐
//!   ┌──────────────────────────────────────────────────────────────────┘
//!   └ nearest ×2 ─ causal conv k5 ─ Linear+LN ─ ×√512 ─ 4 × conformer ─ LN ─ Linear(512→80)
//! ```
//!
//! Conformer layers here are attention + FFN only (no macaron, no conv
//! module), pre-norm with LayerNorm eps 1e-12, SiLU FFN, and ESPnet relative
//! positional attention: `softmax(((q+u)·kᵀ + rel_shift((q+v)·pᵀ)) / √d)`
//! with `p = linear_pos(pe)` over the `2T − 1` relative offsets. The `(q+u)·kᵀ`
//! half rides the fused attention op; the positional half is its additive
//! bias. The encoder attends over the whole sequence (no chunk mask).

use anyhow::Result;
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::Activation;
use rlx_ir::{HirGraphExt, Shape};

use super::nn::{Ctx, F32, S3GenWeights, dims};

pub const PRE_LOOKAHEAD: usize = 3;

/// Widths, read from the weights: 512 / 8 × 64 for chatterbox-turbo, 368 /
/// 8 × 46 for the distilled student.
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub d: usize,
    pub heads: usize,
    pub head_dim: usize,
}

impl Dims {
    pub fn of(w: &S3GenWeights) -> Result<Self> {
        let d = w.shape("flow.encoder.embed.out.0.weight")?[0];
        let u = w.shape("flow.encoder.encoders.0.self_attn.pos_bias_u")?;
        anyhow::ensure!(
            u.len() == 2 && u[0] * u[1] == d,
            "pos_bias_u {u:?} vs width {d}"
        );
        Ok(Self {
            d,
            heads: u[0],
            head_dim: u[1],
        })
    }
}

/// ESPnet `RelPositionalEncoding` (new implementation): `[2T − 1, d]`, row `k`
/// encoding relative offset `T − 1 − k`, computed in f32 like torch.
pub fn rel_pos_table(t: usize, d: usize) -> Vec<f32> {
    let div: Vec<f32> = (0..d / 2)
        .map(|i| ((2 * i) as f32 * -(10000f32.ln() / d as f32)).exp())
        .collect();
    let mut out = vec![0f32; (2 * t - 1) * d];
    for (k, row) in out.chunks_exact_mut(d).enumerate() {
        let pos = t as f32 - 1.0 - k as f32;
        for (i, &dv) in div.iter().enumerate() {
            let a = pos * dv;
            row[2 * i] = a.sin();
            row[2 * i + 1] = a.cos();
        }
    }
    out
}

fn conformer_layer(
    ctx: &mut Ctx,
    w: &S3GenWeights,
    x: HirNodeId,
    p: &str,
    pos: HirNodeId,
    dm: Dims,
) -> Result<HirNodeId> {
    let t = dims(ctx, x)[0];
    let (h, hd) = (dm.heads as i64, dm.head_dim as i64);
    let ti = t as i64;

    let xn = ctx.layer_norm(w, x, &format!("{p}.norm_mha"), 1e-12)?;
    let q = ctx.linear(w, xn, &format!("{p}.self_attn.linear_q"))?;
    let k = ctx.linear(w, xn, &format!("{p}.self_attn.linear_k"))?;
    let v = ctx.linear(w, xn, &format!("{p}.self_attn.linear_v"))?;
    let pp = ctx.linear(w, pos, &format!("{p}.self_attn.linear_pos"))?;

    let bias_of = |ctx: &mut Ctx, name: &str| -> Result<HirNodeId> {
        let key = format!("{p}.self_attn.{name}");
        let b = ctx.param(&key, &[1, dm.d], || w.get(&key).unwrap().to_vec());
        Ok(ctx.g.expand_(b, vec![ti, dm.d as i64]))
    };
    let u = bias_of(ctx, "pos_bias_u")?;
    let vb = bias_of(ctx, "pos_bias_v")?;
    let qu = ctx.g.add(q, u);
    let qv = ctx.g.add(q, vb);

    // Positional scores: (q+v)·pᵀ per head → [H, T, 2T−1], then rel_shift.
    let s = 2 * t - 1;
    let qv_h = ctx.g.reshape_(qv, vec![ti, h, hd]);
    let qv_h = ctx.g.transpose_(qv_h, vec![1, 0, 2]);
    let p_h = ctx.g.reshape_(pp, vec![s as i64, h, hd]);
    let p_h = ctx.g.transpose_(p_h, vec![1, 2, 0]);
    let bd = ctx.g.mm(qv_h, p_h);
    // rel_shift: pad one zero column, view as [H, 2T, T], drop the first row,
    // view back as [H, T, 2T−1] and keep the first T columns.
    let z = ctx.param(
        &format!("zeros:{}x{t}x1", dm.heads),
        &[dm.heads, t, 1],
        || vec![0.0; dm.heads * t],
    );
    let padded = ctx.g.concat_(vec![z, bd], 2);
    let r = ctx.g.reshape_(padded, vec![h, 2 * ti, ti]);
    let r = ctx.g.narrow_(r, 1, 1, s);
    let r = ctx.g.reshape_(r, vec![h, ti, s as i64]);
    let bd = ctx.g.narrow_(r, 2, 0, t);
    let scale = ctx.scalar(1.0 / (dm.head_dim as f32).sqrt(), &[dm.heads, t, t]);
    let bd = ctx.g.mul(bd, scale);
    let bias = ctx.g.reshape_(bd, vec![1, h, ti, ti]);

    let q4 = ctx.g.reshape_(qu, vec![1, ti, h, hd]);
    let k4 = ctx.g.reshape_(k, vec![1, ti, h, hd]);
    let v4 = ctx.g.reshape_(v, vec![1, ti, h, hd]);
    let att = ctx.g.attention_bias(
        q4,
        k4,
        v4,
        bias,
        dm.heads,
        dm.head_dim,
        Shape::new(&[1, t, dm.heads, dm.head_dim], F32),
    );
    let att = ctx.g.reshape_(att, vec![ti, dm.d as i64]);
    let out = ctx.linear(w, att, &format!("{p}.self_attn.linear_out"))?;
    let x = ctx.g.add(x, out);

    let xn = ctx.layer_norm(w, x, &format!("{p}.norm_ff"), 1e-12)?;
    let hdn = ctx.linear(w, xn, &format!("{p}.feed_forward.w_1"))?;
    let hdn = ctx.act(Activation::Silu, hdn);
    let ff = ctx.linear(w, hdn, &format!("{p}.feed_forward.w_2"))?;
    Ok(ctx.g.add(x, ff))
}

/// `LinearNoSubsampling` + positional scaling: `LN(Linear(x)) · √d`.
fn embed(ctx: &mut Ctx, w: &S3GenWeights, x: HirNodeId, key: &str, d: usize) -> Result<HirNodeId> {
    let y = ctx.linear(w, x, &format!("{key}.out.0"))?;
    let y = ctx.layer_norm(w, y, &format!("{key}.out.1"), 1e-5)?;
    let t = dims(ctx, y)[0];
    let s = ctx.scalar((d as f32).sqrt(), &[t, d]);
    Ok(ctx.g.mul(y, s))
}

fn pos_param(ctx: &mut Ctx, t: usize, d: usize) -> HirNodeId {
    ctx.param(&format!("rel_pos:{t}x{d}"), &[2 * t - 1, d], || {
        rel_pos_table(t, d)
    })
}

/// Emit the encoder over token embeddings `x` (`[T, d]`). Returns `mu`
/// as `[1, 80, 2T, 1]`.
pub fn build(ctx: &mut Ctx, w: &S3GenWeights, x: HirNodeId) -> Result<HirNodeId> {
    let dm = Dims::of(w)?;
    let t = dims(ctx, x)[0];
    let e = "flow.encoder";
    let mut h = embed(ctx, w, x, &format!("{e}.embed"), dm.d)?;

    // PreLookaheadLayer: look 3 tokens ahead, then a causal k=3 conv, residual.
    let n = ctx.tc_to_nct(h);
    let c1 = ctx.conv1d(
        w,
        n,
        &format!("{e}.pre_lookahead_layer.conv1"),
        (0, PRE_LOOKAHEAD),
        1,
        1,
    )?;
    let c1 = ctx.leaky_relu(c1, 0.01);
    let c2 = ctx.conv1d(
        w,
        c1,
        &format!("{e}.pre_lookahead_layer.conv2"),
        (2, 0),
        1,
        1,
    )?;
    let c2 = ctx.nct_to_tc(c2);
    h = ctx.g.add(h, c2);

    let pos = pos_param(ctx, t, dm.d);
    for i in 0..count_layers(w, &format!("{e}.encoders")) {
        h = conformer_layer(ctx, w, h, &format!("{e}.encoders.{i}"), pos, dm)?;
    }

    // Upsample1D: nearest ×2, left-pad 4, k=5 conv.
    let n = ctx.tc_to_nct(h);
    let n4 = ctx.g.reshape_(n, vec![1, dm.d as i64, t as i64, 1]);
    let dup = ctx.g.concat_(vec![n4, n4], 3);
    let up = ctx.g.reshape_(dup, vec![1, dm.d as i64, (2 * t) as i64, 1]);
    let up = ctx.conv1d(w, up, &format!("{e}.up_layer.conv"), (4, 0), 1, 1)?;
    let up = ctx.nct_to_tc(up);

    let mut h = embed(ctx, w, up, &format!("{e}.up_embed"), dm.d)?;
    let pos2 = pos_param(ctx, 2 * t, dm.d);
    for i in 0..count_layers(w, &format!("{e}.up_encoders")) {
        h = conformer_layer(ctx, w, h, &format!("{e}.up_encoders.{i}"), pos2, dm)?;
    }
    let h = ctx.layer_norm(w, h, &format!("{e}.after_norm"), 1e-5)?;
    let mu = ctx.linear(w, h, "flow.encoder_proj")?;
    Ok(ctx.tc_to_nct(mu))
}

/// Number of `{prefix}.{i}.…` layers present in the weights.
pub fn count_layers(w: &S3GenWeights, prefix: &str) -> usize {
    (0..)
        .take_while(|i| w.has(&format!("{prefix}.{i}.norm_mha.weight")))
        .count()
}

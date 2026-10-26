// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! The meanflow `ConditionalDecoder`: the velocity field the flow integrates.
//!
//! Input is the 320-channel stack `[x, mu, spks, cond]` over `L` mel frames;
//! output the 80-channel velocity. One causal U-Net level:
//!
//! ```text
//! resnet(320→256) + 4 transformers ─ causal conv ─┐ (skip)
//! 12 × (resnet(256→256) + 4 transformers)          │
//! concat skip ─ resnet(512→256) + 4 transformers ─ causal conv
//! causal block ─ conv1x1(256→80)
//! ```
//!
//! Every resnet adds `Linear(Mish(temb))`; `temb` mixes the embeddings of
//! the step's start `t` and end `r` (`time_embed_mixer`, the meanflow part)
//! and is computed on the host by [`time_embedding`]. Transformer blocks are
//! diffusers `BasicTransformerBlock`s (8 heads × 64, erf-GELU FFN ×4) with
//! full attention. With a single unpadded utterance every mask is all-ones,
//! so none is applied.

use anyhow::Result;
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::{HirGraphExt, Shape};

use super::nn::{Ctx, F32, S3GenWeights, dims};

/// Attention heads (both released estimators use 8; the head width follows
/// from `to_q`).
const HEADS: usize = 8;
const IN_CHANNELS: usize = 320;
const EST: &str = "flow.decoder.estimator";

/// `TimestepEmbedding(SinusoidalPosEmb(v))` for one scalar time.
fn embed_time(w: &S3GenWeights, v: f32) -> Result<Vec<f32>> {
    let half = IN_CHANNELS / 2;
    let step = 10000f32.ln() / (half - 1) as f32;
    let mut s = vec![0f32; IN_CHANNELS];
    for i in 0..half {
        let a = 1000.0 * v * (i as f32 * -step).exp();
        s[i] = a.sin();
        s[half + i] = a.cos();
    }
    let h = host_linear(w, &format!("{EST}.time_mlp.linear_1"), &s)?;
    let h: Vec<f32> = h.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    host_linear(w, &format!("{EST}.time_mlp.linear_2"), &h)
}

/// Width of the time conditioning (`4 × channels`).
pub fn time_dim(w: &S3GenWeights) -> Result<usize> {
    Ok(w.shape(&format!("{EST}.time_mlp.linear_2.weight"))?[0])
}

fn host_linear(w: &S3GenWeights, key: &str, x: &[f32]) -> Result<Vec<f32>> {
    let wk = format!("{key}.weight");
    let quantised;
    let x = match w.bit_layer(&wk) {
        Some(bit) => {
            quantised = super::nn::bit_input_host(x, bit);
            quantised.as_slice()
        }
        None => x,
    };
    let wt = w.get(&wk)?;
    let out = wt.len() / x.len();
    let bias = w.get(&format!("{key}.bias")).ok();
    Ok((0..out)
        .map(|o| {
            let row = &wt[o * x.len()..(o + 1) * x.len()];
            row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + bias.map_or(0.0, |b| b[o])
        })
        .collect())
}

/// The estimator's time conditioning for a step from `t` to `r`:
/// `mixer([emb(t), emb(r)])`, `[time_dim]`.
pub fn time_embedding(w: &S3GenWeights, t: f32, r: f32) -> Result<Vec<f32>> {
    let mut cat = embed_time(w, t)?;
    cat.extend(embed_time(w, r)?);
    host_linear(w, &format!("{EST}.time_embed_mixer"), &cat)
}

/// `CausalBlock1D`: causal k=3 conv → LayerNorm over channels → Mish.
fn causal_block(ctx: &mut Ctx, w: &S3GenWeights, x: HirNodeId, p: &str) -> Result<HirNodeId> {
    let y = ctx.conv1d(w, x, &format!("{p}.block.0"), (2, 0), 1, 1)?;
    let tc = ctx.nct_to_tc(y);
    let tc = ctx.layer_norm(w, tc, &format!("{p}.block.2"), 1e-5)?;
    let y = ctx.tc_to_nct(tc);
    Ok(ctx.act(Activation::Mish, y))
}

/// `CausalResnetBlock1D` on `[1, C_in, L, 1]` with `temb` (`[1, time_dim]`).
fn resnet(
    ctx: &mut Ctx,
    w: &S3GenWeights,
    x: HirNodeId,
    temb_mish: HirNodeId,
    p: &str,
) -> Result<HirNodeId> {
    let h = causal_block(ctx, w, x, &format!("{p}.block1"))?;
    let d = dims(ctx, h);
    let (c, l) = (d[1], d[2]);
    let m = ctx.linear(w, temb_mish, &format!("{p}.mlp.1"))?;
    let m = ctx.g.reshape_(m, vec![1, c as i64, 1, 1]);
    let m = ctx.g.expand_(m, vec![1, c as i64, l as i64, 1]);
    let h = ctx.g.add(h, m);
    let h = causal_block(ctx, w, h, &format!("{p}.block2"))?;
    let r = ctx.conv1d(w, x, &format!("{p}.res_conv"), (0, 0), 1, 1)?;
    Ok(ctx.g.add(h, r))
}

/// diffusers `BasicTransformerBlock` on `[L, channels]`.
fn transformer(ctx: &mut Ctx, w: &S3GenWeights, x: HirNodeId, p: &str) -> Result<HirNodeId> {
    let l = dims(ctx, x)[0];
    let head_dim = w.shape(&format!("{p}.attn1.to_q.weight"))?[0] / HEADS;
    let (li, h, hd) = (l as i64, HEADS as i64, head_dim as i64);
    let n = ctx.layer_norm(w, x, &format!("{p}.norm1"), 1e-5)?;
    let q = ctx.linear(w, n, &format!("{p}.attn1.to_q"))?;
    let k = ctx.linear(w, n, &format!("{p}.attn1.to_k"))?;
    let v = ctx.linear(w, n, &format!("{p}.attn1.to_v"))?;
    let q = ctx.g.reshape_(q, vec![1, li, h, hd]);
    let k = ctx.g.reshape_(k, vec![1, li, h, hd]);
    let v = ctx.g.reshape_(v, vec![1, li, h, hd]);
    let a = ctx.g.attention_kind(
        q,
        k,
        v,
        HEADS,
        head_dim,
        MaskKind::None,
        Shape::new(&[1, l, HEADS, head_dim], F32),
    );
    let a = ctx.g.reshape_(a, vec![li, h * hd]);
    let a = ctx.linear(w, a, &format!("{p}.attn1.to_out.0"))?;
    let x = ctx.g.add(x, a);
    let n = ctx.layer_norm(w, x, &format!("{p}.norm3"), 1e-5)?;
    let f = ctx.linear(w, n, &format!("{p}.ff.net.0.proj"))?;
    let f = ctx.act(Activation::Gelu, f);
    let f = ctx.linear(w, f, &format!("{p}.ff.net.2"))?;
    Ok(ctx.g.add(x, f))
}

fn transformers(ctx: &mut Ctx, w: &S3GenWeights, x: HirNodeId, p: &str) -> Result<HirNodeId> {
    let mut tc = ctx.nct_to_tc(x);
    let n = (0..)
        .take_while(|j| w.has(&format!("{p}.{j}.norm1.weight")))
        .count();
    for j in 0..n {
        tc = transformer(ctx, w, tc, &format!("{p}.{j}"))?;
    }
    Ok(ctx.tc_to_nct(tc))
}

/// Emit the estimator. Inputs are `[1, 80, L, 1]` (`x`, `mu`, `cond`),
/// `spks` `[1, 80, 1, 1]` and `temb` `[1, time_dim]`; output the velocity
/// `[1, 80, L, 1]`.
pub fn build(
    ctx: &mut Ctx,
    w: &S3GenWeights,
    x: HirNodeId,
    mu: HirNodeId,
    spks: HirNodeId,
    cond: HirNodeId,
    temb: HirNodeId,
) -> Result<HirNodeId> {
    let l = dims(ctx, x)[2];
    let sp = ctx.g.expand_(spks, vec![1, 80, l as i64, 1]);
    let h = ctx.g.concat_(vec![x, mu, sp, cond], 1);
    let tm = ctx.act(Activation::Mish, temb);

    let h = resnet(ctx, w, h, tm, &format!("{EST}.down_blocks.0.0"))?;
    let h = transformers(ctx, w, h, &format!("{EST}.down_blocks.0.1"))?;
    let skip = h;
    let mut h = ctx.conv1d(w, h, &format!("{EST}.down_blocks.0.2"), (2, 0), 1, 1)?;

    let n_mid = (0..)
        .take_while(|i| w.has(&format!("{EST}.mid_blocks.{i}.0.res_conv.weight")))
        .count();
    for i in 0..n_mid {
        h = resnet(ctx, w, h, tm, &format!("{EST}.mid_blocks.{i}.0"))?;
        h = transformers(ctx, w, h, &format!("{EST}.mid_blocks.{i}.1"))?;
    }

    let h = ctx.g.concat_(vec![h, skip], 1);
    let h = resnet(ctx, w, h, tm, &format!("{EST}.up_blocks.0.0"))?;
    let h = transformers(ctx, w, h, &format!("{EST}.up_blocks.0.1"))?;
    let h = ctx.conv1d(w, h, &format!("{EST}.up_blocks.0.2"), (2, 0), 1, 1)?;

    let h = causal_block(ctx, w, h, &format!("{EST}.final_block"))?;
    ctx.conv1d(w, h, &format!("{EST}.final_proj"), (0, 0), 1, 1)
}

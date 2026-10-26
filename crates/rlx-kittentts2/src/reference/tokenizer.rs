// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! `S3TokenizerV2` (`speech_tokenizer_v2_25hz`): 16 kHz audio → 25 Hz codes
//! in `[0, 6561)`, the vocabulary both the LM and S3Gen speak.
//!
//! A Whisper-shaped encoder — two stride-2 `k=3` convs + GELU over a 128-bin
//! log-mel, then six pre-norm blocks of 20-head attention with rotate-half
//! RoPE, an FSMN memory (depthwise `k=31` conv over `v`, plus `v`) added to
//! the attention output, and a GELU MLP — followed by FSQ: `Linear(1280→8)`,
//! `tanh · 0.999`, rounded to `{−1, 0, 1}` and read as eight base-3 digits.

use anyhow::{Context, Result, ensure};
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::{Activation, MaskKind, Op};
use rlx_ir::{HirGraphExt, Shape};

use crate::s3gen::nn::{Ctx, F32, S3GenWeights, dims};

pub const N_MELS: usize = 128;
const D: usize = 1280;
const HEADS: usize = 20;
const HEAD_DIM: usize = D / HEADS;
const LAYERS: usize = 6;
const FSMN_K: usize = 31;
const P: &str = "tokenizer.encoder";

/// Frames after the two stride-2 convs.
pub fn encoded_len(mel_frames: usize) -> usize {
    let c1 = (mel_frames - 1) / 2 + 1;
    (c1 - 1) / 2 + 1
}

/// `[T, 64]` cos/sin of `precompute_freqs_cis(64, ·)` duplicated over both
/// halves, as `apply_rotary_emb` uses them.
fn rope_tables(t: usize) -> (Vec<f32>, Vec<f32>) {
    let half = HEAD_DIM / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / HEAD_DIM as f32))
        .collect();
    let mut cos = vec![0f32; t * HEAD_DIM];
    let mut sin = vec![0f32; t * HEAD_DIM];
    for p in 0..t {
        for (i, &f) in freqs.iter().enumerate() {
            let a = p as f32 * f;
            let (s, c) = a.sin_cos();
            for j in [i, i + half] {
                cos[p * HEAD_DIM + j] = c;
                sin[p * HEAD_DIM + j] = s;
            }
        }
    }
    (cos, sin)
}

/// `x·cos + rotate_half(x)·sin` on `[T, H, 64]`.
fn rope(ctx: &mut Ctx, x: HirNodeId, cos: HirNodeId, sin: HirNodeId, t: usize) -> HirNodeId {
    let (ti, h) = (t as i64, HEADS as i64);
    let x = ctx.g.reshape_(x, vec![ti, h, HEAD_DIM as i64]);
    let lo = ctx.g.narrow_(x, 2, 0, HEAD_DIM / 2);
    let hi = ctx.g.narrow_(x, 2, HEAD_DIM / 2, HEAD_DIM / 2);
    let neg = ctx.g.neg(hi);
    let rot = ctx.g.concat_(vec![neg, lo], 2);
    let c = ctx.g.reshape_(cos, vec![ti, 1, HEAD_DIM as i64]);
    let c = ctx.g.expand_(c, vec![ti, h, HEAD_DIM as i64]);
    let s = ctx.g.reshape_(sin, vec![ti, 1, HEAD_DIM as i64]);
    let s = ctx.g.expand_(s, vec![ti, h, HEAD_DIM as i64]);
    let a = ctx.g.mul(x, c);
    let b = ctx.g.mul(rot, s);
    ctx.g.add(a, b)
}

/// Depthwise `k=31` conv over time with "same" zero padding, on `[T, D]`.
fn fsmn(ctx: &mut Ctx, w: &S3GenWeights, v: HirNodeId, key: &str) -> Result<HirNodeId> {
    let t = dims(ctx, v)[0];
    let wk = format!("{key}.weight");
    ensure!(w.shape(&wk)? == [D, 1, FSMN_K], "{wk}: unexpected shape");
    let n = ctx.tc_to_nct(v);
    let pad = (FSMN_K - 1) / 2;
    let np = ctx.pad_len(n, pad, FSMN_K - 1 - pad);
    let wid = ctx.param(&wk, &[D, 1, FSMN_K, 1], || w.get(&wk).unwrap().to_vec());
    let y = ctx.g.add_node(
        Op::Conv {
            kernel_size: vec![FSMN_K, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: D,
        },
        vec![np, wid],
        Shape::new(&[1, D, t, 1], F32),
    );
    let y = ctx.nct_to_tc(y);
    Ok(ctx.g.add(y, v))
}

fn block(
    ctx: &mut Ctx,
    w: &S3GenWeights,
    x: HirNodeId,
    i: usize,
    cos: HirNodeId,
    sin: HirNodeId,
) -> Result<HirNodeId> {
    let p = format!("{P}.blocks.{i}");
    let t = dims(ctx, x)[0];
    let n = ctx.layer_norm(w, x, &format!("{p}.attn_ln"), 1e-5)?;
    let q = ctx.linear(w, n, &format!("{p}.attn.query"))?;
    let k = ctx.linear(w, n, &format!("{p}.attn.key"))?;
    let v = ctx.linear(w, n, &format!("{p}.attn.value"))?;
    let memory = fsmn(ctx, w, v, &format!("{p}.attn.fsmn_block"))?;
    let q = rope(ctx, q, cos, sin, t);
    let k = rope(ctx, k, cos, sin, t);
    let (ti, h, hd) = (t as i64, HEADS as i64, HEAD_DIM as i64);
    let q = ctx.g.reshape_(q, vec![1, ti, h, hd]);
    let k = ctx.g.reshape_(k, vec![1, ti, h, hd]);
    let v4 = ctx.g.reshape_(v, vec![1, ti, h, hd]);
    let a = ctx.g.attention_kind(
        q,
        k,
        v4,
        HEADS,
        HEAD_DIM,
        MaskKind::None,
        Shape::new(&[1, t, HEADS, HEAD_DIM], F32),
    );
    let a = ctx.g.reshape_(a, vec![ti, D as i64]);
    let a = ctx.linear(w, a, &format!("{p}.attn.out"))?;
    let a = ctx.g.add(a, memory);
    let x = ctx.g.add(x, a);
    let n = ctx.layer_norm(w, x, &format!("{p}.mlp_ln"), 1e-5)?;
    let m = ctx.linear(w, n, &format!("{p}.mlp.0"))?;
    let m = ctx.act(Activation::Gelu, m);
    let m = ctx.linear(w, m, &format!("{p}.mlp.2"))?;
    Ok(ctx.g.add(x, m))
}

/// Emit the encoder over a log-mel `[1, 128, F, 1]`; returns `[T, 1280]`.
pub fn build(ctx: &mut Ctx, w: &S3GenWeights, mel: HirNodeId) -> Result<HirNodeId> {
    let h = ctx.conv1d(w, mel, &format!("{P}.conv1"), (1, 1), 2, 1)?;
    let h = ctx.act(Activation::Gelu, h);
    let h = ctx.conv1d(w, h, &format!("{P}.conv2"), (1, 1), 2, 1)?;
    let h = ctx.act(Activation::Gelu, h);
    let mut x = ctx.nct_to_tc(h);
    let t = dims(ctx, x)[0];
    let (c, s) = rope_tables(t);
    let cos = ctx.param(&format!("rope_cos:{t}"), &[t, HEAD_DIM], || c);
    let sin = ctx.param(&format!("rope_sin:{t}"), &[t, HEAD_DIM], || s);
    for i in 0..LAYERS {
        x = block(ctx, w, x, i, cos, sin)?;
    }
    Ok(x)
}

/// FSQ over encoder output `[T, 1280]` → codes.
pub fn quantize(w: &S3GenWeights, hidden: &[f32]) -> Result<Vec<u32>> {
    let pw = w.get("tokenizer.quantizer._codebook.project_down.weight")?;
    let pb = w.get("tokenizer.quantizer._codebook.project_down.bias")?;
    ensure!(pw.len() == 8 * D && pb.len() == 8, "FSQ projection shape");
    hidden
        .as_chunks::<D>()
        .0
        .iter()
        .map(|row| {
            let mut code = 0u32;
            let mut place = 1u32;
            for j in 0..8 {
                let z: f32 = pw[j * D..(j + 1) * D]
                    .iter()
                    .zip(row)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    + pb[j];
                let digit = (z.tanh() * 0.999_f32).round_ties_even() + 1.0;
                code += digit as u32 * place;
                place *= 3;
            }
            Ok(code)
        })
        .collect::<Result<Vec<u32>>>()
        .context("FSQ")
}

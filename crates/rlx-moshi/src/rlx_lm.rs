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

//! Native RLX graphs for the Moshi temporal transformer (Helium), built directly
//! on the rlx-ir / rlx-runtime op library — no candle, no `moshi` crate.
//!
//! Moshi's temporal stack is Llama/Helium-shaped: pre-norm RMSNorm, fused-QKV
//! multi-head attention with rotary embeddings (rotate-half, θ=10000), and a
//! fused gate+up SwiGLU MLP, all bias-free. The input is a single pre-summed
//! embedding vector per step (text + up to 16 audio codebooks summed outside the
//! transformer), so the graph consumes `inputs_embeds [batch, seq, d_model]`.
//!
//! This module currently builds the forward up to the text-logits head and is
//! validated for numeric parity against the eager [`crate::lm::LmModel`].

use crate::config::{DepFormerConfig, TransformerConfig};
use crate::sampling::LogitsProcessor;
use anyhow::{Context, Result};
use ndarray::ArrayView1;
use rayon::prelude::*;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

const RMS_EPS: f32 = 1e-8;

/// Static dims of the Helium temporal transformer, derived from its config.
#[derive(Debug, Clone, Copy)]
pub struct HeliumDims {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_layers: usize,
    pub ffn: usize, // SwiGLU hidden width
    pub vocab_out: usize,
    pub rope_theta: f32,
}

impl HeliumDims {
    pub fn from_cfg(cfg: &TransformerConfig, vocab_out: usize) -> Self {
        Self {
            d_model: cfg.d_model,
            n_heads: cfg.num_heads,
            head_dim: cfg.d_model / cfg.num_heads,
            n_layers: cfg.num_layers,
            ffn: cfg.swiglu_hidden(),
            vocab_out,
            rope_theta: cfg.max_period as f32,
        }
    }
}

fn p(li: usize, name: &str) -> String {
    format!("L{li}.{name}")
}

/// Apply rotate-half RoPE (Moshi/HF convention) to `x [1, seq, heads, head_dim]`
/// in-graph: pairs `(i, i+half)` rotate together. `cos`/`sin` are `[1, seq, 1, half]`.
fn apply_rope(g: &mut Graph, x: NodeId, cos: NodeId, sin: NodeId, half: usize) -> NodeId {
    let x1 = g.narrow_(x, 3, 0, half);
    let x2 = g.narrow_(x, 3, half, half);
    let x1c = g.mul(x1, cos);
    let x2s = g.mul(x2, sin);
    let x2c = g.mul(x2, cos);
    let x1s = g.mul(x1, sin);
    let r1 = g.sub(x1c, x2s);
    let r2 = g.add(x2c, x1s);
    g.concat_(vec![r1, r2], 3)
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // Transpose `[rows, cols]` → `[cols, rows]` in parallel over output rows (each
    // gathers one source column). Called per-weight at load → matters for the 7B.
    let mut out = vec![0.0f32; rows * cols];
    out.par_chunks_mut(rows)
        .enumerate()
        .for_each(|(c, out_row)| {
            for (r, slot) in out_row.iter_mut().enumerate() {
                *slot = data[r * cols + c];
            }
        });
    out
}

/// `RmsNorm` node over `x` with a per-layer `[d]` weight named `name` (eps 1e-8).
fn rms(g: &mut Graph, x: NodeId, name: &str, d: usize, shape: Shape, zero_beta: NodeId) -> NodeId {
    let w = g.param(name, Shape::new(&[d], DType::F32));
    g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: RMS_EPS,
        },
        vec![x, w, zero_beta],
        shape,
    )
}

/// Project `n1` to per-head `q`/`k`/`v` via the (split, transposed) `q`/`k`/`v`
/// params and reshape each to `heads` (`[1, seq, nh, hd]`). RoPE is applied by
/// the caller (temporal) or skipped (DepFormer).
fn qkv_proj(
    g: &mut Graph,
    n1: NodeId,
    li: usize,
    d: usize,
    heads: &[i64],
) -> (NodeId, NodeId, NodeId) {
    let qw = g.param(p(li, "q"), Shape::new(&[d, d], DType::F32));
    let kw = g.param(p(li, "k"), Shape::new(&[d, d], DType::F32));
    let vw = g.param(p(li, "v"), Shape::new(&[d, d], DType::F32));
    let q = g.mm(n1, qw);
    let k = g.mm(n1, kw);
    let v = g.mm(n1, vw);
    let q = g.reshape_(q, heads.to_vec());
    let k = g.reshape_(k, heads.to_vec());
    let v = g.reshape_(v, heads.to_vec());
    (q, k, v)
}

/// Pre-norm SwiGLU MLP block with residual: `x + down(silu(gate(n2)) * up(n2))`
/// where `n2 = RmsNorm(x)`. Identical across all temporal/DepFormer builders.
fn swiglu_block(
    g: &mut Graph,
    x: NodeId,
    li: usize,
    d: usize,
    ffn: usize,
    shape: Shape,
    zero_beta: NodeId,
) -> NodeId {
    let n2 = rms(g, x, &p(li, "n2"), d, shape, zero_beta);
    let gw = g.param(p(li, "gate"), Shape::new(&[d, ffn], DType::F32));
    let uw = g.param(p(li, "up"), Shape::new(&[d, ffn], DType::F32));
    let gate = g.mm(n2, gw);
    let up = g.mm(n2, uw);
    let gate = g.silu(gate);
    let h = g.mul(gate, up);
    let dw = g.param(p(li, "down"), Shape::new(&[ffn, d], DType::F32));
    let mlp = g.mm(h, dw);
    g.add(x, mlp)
}

/// Emit each per-layer transformer param (name → transposed `[in, out]` data) for
/// a stack under `base` (e.g. `""` for the temporal stack, `"depformer.{si}."`
/// for a DepFormer slice), splitting fused QKV (`in_proj_weight [3d,d]`) and fused
/// gate+up (`gating.linear_in [2·ffn,d]`). Shared by the param builders below.
fn for_each_transformer_param(
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    base: &str,
    d: usize,
    ffn: usize,
    n_layers: usize,
    mut emit: impl FnMut(String, Vec<f32>),
) -> Result<()> {
    let get = |k: &str| -> Result<&Vec<f32>> {
        weights
            .get(k)
            .map(|(v, _)| v)
            .with_context(|| format!("missing weight {k}"))
    };
    for li in 0..n_layers {
        let pre = format!("{base}transformer.layers.{li}");
        let inproj = get(&format!("{pre}.self_attn.in_proj_weight"))?;
        emit(p(li, "q"), transpose(&inproj[0..d * d], d, d));
        emit(p(li, "k"), transpose(&inproj[d * d..2 * d * d], d, d));
        emit(p(li, "v"), transpose(&inproj[2 * d * d..3 * d * d], d, d));
        emit(
            p(li, "o"),
            transpose(get(&format!("{pre}.self_attn.out_proj.weight"))?, d, d),
        );
        emit(p(li, "n1"), get(&format!("{pre}.norm1.alpha"))?.clone());
        emit(p(li, "n2"), get(&format!("{pre}.norm2.alpha"))?.clone());
        let gate_up = get(&format!("{pre}.gating.linear_in.weight"))?;
        emit(p(li, "gate"), transpose(&gate_up[0..ffn * d], ffn, d));
        emit(
            p(li, "up"),
            transpose(&gate_up[ffn * d..2 * ffn * d], ffn, d),
        );
        emit(
            p(li, "down"),
            transpose(get(&format!("{pre}.gating.linear_out.weight"))?, d, ffn),
        );
    }
    Ok(())
}

/// Build the temporal-transformer forward graph over `seq` tokens, consuming
/// `inputs_embeds [1, seq, d_model]` and producing `logits [1, seq, vocab_out]`.
///
/// Inputs: `inputs_embeds`, `rope_cos`/`rope_sin` `[seq, head_dim/2]`.
/// Params: per-layer `L{i}.{q,k,v,o}` `[d,d]`, `L{i}.{n1,n2}` `[d]`,
/// `L{i}.{gate,up}` `[ffn,d]`, `L{i}.down` `[d,ffn]`; `out_norm` `[d]`,
/// `text_linear` `[vocab,d]`; plus `zero_beta` `[d]` for the RMSNorm op.
pub fn build_temporal_graph(dims: &HeliumDims, seq: usize) -> Graph {
    let HeliumDims {
        d_model: d,
        n_heads: nh,
        head_dim: hd,
        n_layers,
        ffn,
        vocab_out,
        ..
    } = *dims;
    let mut g = Graph::new("moshi_temporal");
    let bsd = Shape::new(&[1, seq, d], DType::F32);

    let half = hd / 2;
    let mut x = g.input("inputs_embeds", bsd.clone());
    let cos = g.input("rope_cos", Shape::new(&[seq, half], DType::F32));
    let sin = g.input("rope_sin", Shape::new(&[seq, half], DType::F32));
    // Reshape rope tables to [1, seq, 1, half] so they broadcast over heads.
    let cos4 = g.reshape_(cos, vec![1, seq as i64, 1, half as i64]);
    let sin4 = g.reshape_(sin, vec![1, seq as i64, 1, half as i64]);
    let zero_beta = g.param("zero_beta", Shape::new(&[d], DType::F32));

    let heads = [1i64, seq as i64, nh as i64, hd as i64];
    for li in 0..n_layers {
        // ── attention block (pre-norm), RoPE + full causal mask ──
        let n1 = rms(&mut g, x, &p(li, "n1"), d, bsd.clone(), zero_beta);
        let (q, k, v) = qkv_proj(&mut g, n1, li, d, &heads);
        let q = apply_rope(&mut g, q, cos4, sin4, half);
        let k = apply_rope(&mut g, k, cos4, sin4, half);
        let attn = g.attention_kind(
            q,
            k,
            v,
            nh,
            hd,
            MaskKind::Causal,
            Shape::new(&[1, seq, nh, hd], DType::F32),
        );
        let attn = g.reshape_(attn, vec![1i64, seq as i64, d as i64]);
        let ow = g.param(p(li, "o"), Shape::new(&[d, d], DType::F32));
        let attn = g.mm(attn, ow);
        x = g.add(x, attn);

        // ── SwiGLU MLP block (pre-norm) ──
        x = swiglu_block(&mut g, x, li, d, ffn, bsd.clone(), zero_beta);
    }

    // ── final norm + text logits head ──
    let xn = rms(&mut g, x, "out_norm", d, bsd, zero_beta);
    let tl = g.param("text_linear", Shape::new(&[d, vocab_out], DType::F32));
    let logits = g.mm(xn, tl);
    g.set_outputs(vec![logits]);
    g
}

/// Build the single-token KV-**decode** graph: consume `inputs_embeds [1,1,d]` plus
/// per-layer `past_k_{i}`/`past_v_{i}` `[1, past, nh, hd]`, and produce
/// `logits [1,1,vocab]` + per-layer `new_k_{i}`/`new_v_{i}` `[1, past+1, nh, hd]`.
/// Same params as [`build_temporal_graph`]. RoPE position is `past` (passed via
/// the `rope_cos`/`rope_sin` inputs, shape `[1, half]`).
pub fn build_temporal_decode_graph(dims: &HeliumDims, past: usize) -> Graph {
    let HeliumDims {
        d_model: d,
        n_heads: nh,
        head_dim: hd,
        n_layers,
        ffn,
        vocab_out,
        ..
    } = *dims;
    let half = hd / 2;
    let p1d = Shape::new(&[1, 1, d], DType::F32);
    let mut g = Graph::new("moshi_temporal_decode");

    let mut x = g.input("inputs_embeds", p1d.clone());
    let cos = g.input("rope_cos", Shape::new(&[1, half], DType::F32));
    let sin = g.input("rope_sin", Shape::new(&[1, half], DType::F32));
    let cos4 = g.reshape_(cos, vec![1, 1, 1, half as i64]);
    let sin4 = g.reshape_(sin, vec![1, 1, 1, half as i64]);
    let zero_beta = g.param("zero_beta", Shape::new(&[d], DType::F32));
    let kv_shape = Shape::new(&[1, past, nh, hd], DType::F32);
    let heads1 = vec![1i64, 1, nh as i64, hd as i64];
    let mut kv_outputs = Vec::with_capacity(2 * n_layers);

    for li in 0..n_layers {
        let n1 = rms(&mut g, x, &p(li, "n1"), d, p1d.clone(), zero_beta);
        let (q, k, v) = qkv_proj(&mut g, n1, li, d, &heads1);
        let q = apply_rope(&mut g, q, cos4, sin4, half);
        let k = apply_rope(&mut g, k, cos4, sin4, half);

        // At the first step there is no cache; otherwise prepend past_k/past_v.
        let (k_full, v_full) = if past == 0 {
            (k, v)
        } else {
            let past_k = g.input(format!("past_k_{li}"), kv_shape.clone());
            let past_v = g.input(format!("past_v_{li}"), kv_shape.clone());
            (g.concat_(vec![past_k, k], 1), g.concat_(vec![past_v, v], 1))
        };
        kv_outputs.push(k_full);
        kv_outputs.push(v_full);

        let attn = g.attention_kind(
            q,
            k_full,
            v_full,
            nh,
            hd,
            MaskKind::Causal,
            Shape::new(&[1, 1, nh, hd], DType::F32),
        );
        let attn = g.reshape_(attn, vec![1i64, 1, d as i64]);
        let ow = g.param(p(li, "o"), Shape::new(&[d, d], DType::F32));
        let attn = g.mm(attn, ow);
        x = g.add(x, attn);

        x = swiglu_block(&mut g, x, li, d, ffn, p1d.clone(), zero_beta);
    }

    // `x` (pre-out_norm transformer output) is the DepFormer's conditioning input.
    let hidden = x;
    let xn = rms(&mut g, x, "out_norm", d, p1d, zero_beta);
    let tl = g.param("text_linear", Shape::new(&[d, vocab_out], DType::F32));
    let logits = g.mm(xn, tl);
    let mut outs = vec![logits, hidden];
    outs.extend(kv_outputs);
    g.set_outputs(outs);
    g
}

/// Binary keep-mask for one bucketed decode step (length `upper + 1`):
/// attend the `past_seq` real cached keys plus the new token at slot `upper`;
/// mask the `[past_seq, upper)` zero padding. Matches rlx-runtime's
/// `bucket_decode_mask` convention (`MaskKind::Custom` is `m[ki] < 0.5` → drop).
pub fn bucket_decode_mask(past_seq: usize, upper: usize) -> Vec<f32> {
    (0..=upper)
        .map(|i| if i < past_seq || i == upper { 1.0 } else { 0.0 })
        .collect()
}

/// Bucketed single-token decode graph: like [`build_temporal_decode_graph`] but
/// the past KV is padded to a fixed `upper`, so one compiled graph is reused
/// across many frames. Attention uses a Custom keep-mask (built per frame by
/// [`bucket_decode_mask`]) to exclude the padding. The emitted `new_k`/`new_v`
/// are the **single** new token's K/V (append to the real cache), not the
/// padded concat. Outputs: `[logits, hidden, new_k_0, new_v_0, …]`.
pub fn build_temporal_decode_graph_bucketed(dims: &HeliumDims, upper: usize) -> Graph {
    let HeliumDims {
        d_model: d,
        n_heads: nh,
        head_dim: hd,
        n_layers,
        ffn,
        vocab_out,
        ..
    } = *dims;
    let half = hd / 2;
    let p1d = Shape::new(&[1, 1, d], DType::F32);
    let mut g = Graph::new("moshi_temporal_decode_bucketed");

    let mut x = g.input("inputs_embeds", p1d.clone());
    let cos = g.input("rope_cos", Shape::new(&[1, half], DType::F32));
    let sin = g.input("rope_sin", Shape::new(&[1, half], DType::F32));
    let cos4 = g.reshape_(cos, vec![1, 1, 1, half as i64]);
    let sin4 = g.reshape_(sin, vec![1, 1, 1, half as i64]);
    let mask = g.input("attn_mask", Shape::new(&[1, upper + 1], DType::F32));
    let zero_beta = g.param("zero_beta", Shape::new(&[d], DType::F32));
    let one = g.param("kv_one", Shape::new(&[1], DType::F32));
    let kv_shape = Shape::new(&[1, upper, nh, hd], DType::F32);
    let heads1 = vec![1i64, 1, nh as i64, hd as i64];
    let mut kv_outputs = Vec::with_capacity(2 * n_layers);

    for li in 0..n_layers {
        let n1 = rms(&mut g, x, &p(li, "n1"), d, p1d.clone(), zero_beta);
        let (q, k, v) = qkv_proj(&mut g, n1, li, d, &heads1);
        let q = apply_rope(&mut g, q, cos4, sin4, half);
        let k = apply_rope(&mut g, k, cos4, sin4, half);

        // Emit just the new token's K/V (materialized) for the real cache.
        let new_k = g.mul(k, one);
        let new_v = g.mul(v, one);
        kv_outputs.push(new_k);
        kv_outputs.push(new_v);

        // Attend over [padded past || new token] with the custom keep-mask.
        let past_k = g.input(format!("past_k_{li}"), kv_shape.clone());
        let past_v = g.input(format!("past_v_{li}"), kv_shape.clone());
        let k_full = g.concat_(vec![past_k, k], 1);
        let v_full = g.concat_(vec![past_v, v], 1);
        let attn = g.attention_(q, k_full, v_full, mask, nh, hd);
        let attn = g.reshape_(attn, vec![1i64, 1, d as i64]);
        let ow = g.param(p(li, "o"), Shape::new(&[d, d], DType::F32));
        let attn = g.mm(attn, ow);
        x = g.add(x, attn);

        x = swiglu_block(&mut g, x, li, d, ffn, p1d.clone(), zero_beta);
    }

    let hidden = x;
    let xn = rms(&mut g, x, "out_norm", d, p1d, zero_beta);
    let tl = g.param("text_linear", Shape::new(&[d, vocab_out], DType::F32));
    let logits = g.mm(xn, tl);
    let mut outs = vec![logits, hidden];
    outs.extend(kv_outputs);
    g.set_outputs(outs);
    g
}

/// Re-key the Moshi weight map into the temporal graph's param names, splitting
/// the fused QKV (`in_proj_weight [3d,d]` → q/k/v) and fused gate+up
/// (`gating.linear_in [2·ffn,d]` → gate/up).
pub fn temporal_params(
    dims: &HeliumDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<HashMap<String, Vec<f32>>> {
    let HeliumDims {
        d_model: d,
        ffn,
        n_layers,
        ..
    } = *dims;
    let get = |k: &str| -> Result<&Vec<f32>> {
        weights
            .get(k)
            .map(|(v, _)| v)
            .with_context(|| format!("missing weight {k}"))
    };
    let mut out = HashMap::new();
    out.insert("zero_beta".to_string(), vec![0.0f32; d]);
    for_each_transformer_param(weights, "", d, ffn, n_layers, |name, data| {
        out.insert(name, data);
    })?;
    out.insert("out_norm".to_string(), get("out_norm.alpha")?.clone());
    out.insert(
        "text_linear".to_string(),
        transpose(get("text_linear.weight")?, dims.vocab_out, d),
    );
    Ok(out)
}

/// Memory-efficient counterpart of [`temporal_params`] for the real 7B: transpose
/// each weight and `set_param` it **one at a time** so the full transposed map
/// (~27 GB at 7B) never coexists with the weights map *and* the compiled graph's
/// copy. Peak ≈ weights + graph instead of weights + map + graph.
pub fn set_temporal_params(
    compiled: &mut rlx_runtime::CompiledGraph,
    dims: &HeliumDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<()> {
    let HeliumDims {
        d_model: d,
        ffn,
        n_layers,
        vocab_out,
        ..
    } = *dims;
    let get = |k: &str| -> Result<&Vec<f32>> {
        weights
            .get(k)
            .map(|(v, _)| v)
            .with_context(|| format!("missing weight {k}"))
    };
    compiled.set_param("zero_beta", &vec![0.0f32; d]);
    compiled.set_param("kv_one", &[1.0]);
    for_each_transformer_param(weights, "", d, ffn, n_layers, |name, data| {
        compiled.set_param(&name, &data);
    })?;
    compiled.set_param("out_norm", get("out_norm.alpha")?);
    compiled.set_param(
        "text_linear",
        &transpose(get("text_linear.weight")?, vocab_out, d),
    );
    Ok(())
}

// ───────────────────────────── DepFormer ──────────────────────────────
//
// The depth decoder is `num_slices` independent mini-transformers (no RoPE).
// Slice `si` is conditioned on the temporal hidden state + an embedding of the
// previous token, and inherits slice `si-1`'s KV cache, so it attends to
// positions `0..si`. Since `context == num_slices`, the cache never wraps —
// this is exactly the KV-decode primitive with `past = si` and RoPE disabled.

/// Static dims of one DepFormer slice transformer (+ its in/out projections).
#[derive(Debug, Clone, Copy)]
pub struct DepDims {
    pub d_main: usize, // temporal d_model feeding linear_in
    pub d_dep: usize,  // slice transformer width
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_layers: usize,
    pub ffn: usize,
    pub audio_vocab: usize,
    pub num_slices: usize,
}

impl DepDims {
    pub fn from_cfg(cfg: &DepFormerConfig, d_main: usize, audio_vocab: usize) -> Self {
        let t = &cfg.transformer;
        Self {
            d_main,
            d_dep: t.d_model,
            n_heads: t.num_heads,
            head_dim: t.d_model / t.num_heads,
            n_layers: t.num_layers,
            ffn: t.swiglu_hidden(),
            audio_vocab,
            num_slices: cfg.num_slices,
        }
    }
}

/// Build one DepFormer slice's decode graph (no RoPE). Inputs: `temporal_hidden
/// [1,1,d_main]`, `emb_vec [1,1,d_dep]` (looked-up token embedding, or zeros),
/// and per-layer `past_k_{i}`/`past_v_{i} [1, past, nh, hd]`. Produces
/// `logits [1,1,audio_vocab]` + per-layer `new_k_{i}`/`new_v_{i}`.
pub fn build_depformer_slice_graph(dd: &DepDims, past: usize) -> Graph {
    let DepDims {
        d_main,
        d_dep: d,
        n_heads: nh,
        head_dim: hd,
        n_layers,
        ffn,
        audio_vocab,
        ..
    } = *dd;
    let p1d = Shape::new(&[1, 1, d], DType::F32);
    let mut g = Graph::new("moshi_depformer_slice");

    let hidden = g.input("temporal_hidden", Shape::new(&[1, 1, d_main], DType::F32));
    let emb_vec = g.input("emb_vec", p1d.clone());
    let zero_beta = g.param("zero_beta", Shape::new(&[d], DType::F32));
    // Multiply-by-one to materialize the emitted KV: a bare reshape/concat node
    // used directly as a graph output can alias an internal buffer (reads back as
    // zeros in the compiled path); a binary op forces a real output buffer.
    let one = g.param("kv_one", Shape::new(&[1], DType::F32));
    let kv_shape = Shape::new(&[1, past, nh, hd], DType::F32);
    let heads1 = vec![1i64, 1, nh as i64, hd as i64];

    // h = linear_in(temporal_hidden) + emb(prev_token)
    let lin_in = g.param("linear_in", Shape::new(&[d_main, d], DType::F32));
    let proj = g.mm(hidden, lin_in);
    let mut x = g.add(proj, emb_vec);

    let mut kv_outputs = Vec::with_capacity(2 * n_layers);
    for li in 0..n_layers {
        // No RoPE for the depth transformer.
        let n1 = rms(&mut g, x, &p(li, "n1"), d, p1d.clone(), zero_beta);
        let (q, k, v) = qkv_proj(&mut g, n1, li, d, &heads1);

        let (k_full, v_full) = if past == 0 {
            (k, v)
        } else {
            let past_k = g.input(format!("past_k_{li}"), kv_shape.clone());
            let past_v = g.input(format!("past_v_{li}"), kv_shape.clone());
            (g.concat_(vec![past_k, k], 1), g.concat_(vec![past_v, v], 1))
        };
        // Emit materialized copies for the inherited cache; attend on the originals.
        let k_out = g.mul(k_full, one);
        let v_out = g.mul(v_full, one);
        kv_outputs.push(k_out);
        kv_outputs.push(v_out);

        let attn = g.attention_kind(
            q,
            k_full,
            v_full,
            nh,
            hd,
            MaskKind::Causal,
            Shape::new(&[1, 1, nh, hd], DType::F32),
        );
        let attn = g.reshape_(attn, vec![1i64, 1, d as i64]);
        let ow = g.param(p(li, "o"), Shape::new(&[d, d], DType::F32));
        let attn = g.mm(attn, ow);
        x = g.add(x, attn);

        x = swiglu_block(&mut g, x, li, d, ffn, p1d.clone(), zero_beta);
    }

    // Audio logits head (no out_norm in the depth decoder).
    let lo = g.param("linear_out", Shape::new(&[d, audio_vocab], DType::F32));
    let logits = g.mm(x, lo);
    let mut outs = vec![logits];
    outs.extend(kv_outputs);
    g.set_outputs(outs);
    g
}

/// Build the param map for DepFormer slice `si` (per-slice transformer weights +
/// linear_in/linear_out), splitting fused QKV / gate+up like the temporal stack.
pub fn depformer_slice_params(
    dd: &DepDims,
    si: usize,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<HashMap<String, Vec<f32>>> {
    let DepDims {
        d_main,
        d_dep: d,
        ffn,
        n_layers,
        audio_vocab,
        ..
    } = *dd;
    let get = |k: &str| -> Result<&Vec<f32>> {
        weights
            .get(k)
            .map(|(v, _)| v)
            .with_context(|| format!("missing weight {k}"))
    };
    let mut out = HashMap::new();
    out.insert("zero_beta".to_string(), vec![0.0f32; d]);
    out.insert("kv_one".to_string(), vec![1.0f32]);
    out.insert(
        "linear_in".to_string(),
        transpose(get(&format!("depformer.{si}.linear_in.weight"))?, d, d_main),
    );
    out.insert(
        "linear_out".to_string(),
        transpose(
            get(&format!("depformer.{si}.linear_out.weight"))?,
            audio_vocab,
            d,
        ),
    );
    for_each_transformer_param(
        weights,
        &format!("depformer.{si}."),
        d,
        ffn,
        n_layers,
        |name, data| {
            out.insert(name, data);
        },
    )?;
    Ok(out)
}

/// Compile + set params for DepFormer slice `si` at past size `past` (= `si` with
/// full inheritance). Reuse the returned graph across frames via [`depformer_slice_run`].
pub fn compile_depformer_slice(
    dd: &DepDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    si: usize,
    past: usize,
    device: Device,
) -> Result<rlx_runtime::CompiledGraph> {
    let mut c = Session::new(device).compile(build_depformer_slice_graph(dd, past));
    for (name, data) in &depformer_slice_params(dd, si, weights)? {
        c.set_param(name, data);
    }
    Ok(c)
}

/// Run a (pre-compiled) DepFormer slice graph: looks up the `last_token`
/// embedding, feeds the inherited `past_kv`, returns `(audio_logits, new_kv)`.
pub fn depformer_slice_run(
    compiled: &mut rlx_runtime::CompiledGraph,
    dd: &DepDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    temporal_hidden: &[f32],
    si: usize,
    last_token: Option<u32>,
    past_kv: &[(Vec<f32>, Vec<f32>)],
) -> Result<(Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>)> {
    let emb_vec = match last_token {
        Some(t) => {
            let (data, shape) = weights
                .get(&format!("depformer.{si}.emb.weight"))
                .with_context(|| format!("missing depformer.{si}.emb.weight"))?;
            let row = shape[1];
            data[t as usize * row..(t as usize + 1) * row].to_vec()
        }
        None => vec![0.0f32; dd.d_dep],
    };
    let mut inputs: Vec<(String, &[f32])> = vec![
        ("temporal_hidden".to_string(), temporal_hidden),
        ("emb_vec".to_string(), emb_vec.as_slice()),
    ];
    if !past_kv.is_empty() {
        for (li, (k, v)) in past_kv.iter().enumerate() {
            inputs.push((format!("past_k_{li}"), k.as_slice()));
            inputs.push((format!("past_v_{li}"), v.as_slice()));
        }
    }
    let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, dt)| (n.as_str(), *dt)).collect();
    let mut it = compiled.run(&refs).into_iter();
    let logits = it.next().context("depformer slice produced no logits")?;
    let mut new_kv = Vec::with_capacity(dd.n_layers);
    for _ in 0..dd.n_layers {
        let k = it.next().context("missing new_k")?;
        let v = it.next().context("missing new_v")?;
        new_kv.push((k, v));
    }
    Ok((logits, new_kv))
}

/// Run one DepFormer slice graph (fresh compile): returns `(audio_logits, new_kv)`
/// for slice `si`. The past size is derived from the inherited cache, not `si`.
pub fn depformer_run_slice(
    dd: &DepDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    temporal_hidden: &[f32],
    si: usize,
    last_token: Option<u32>,
    past_kv: &[(Vec<f32>, Vec<f32>)],
    device: Device,
) -> Result<(Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>)> {
    let past = if past_kv.is_empty() {
        0
    } else {
        past_kv[0].0.len() / (dd.n_heads * dd.head_dim)
    };
    let mut compiled = compile_depformer_slice(dd, weights, si, past, device)?;
    depformer_slice_run(
        &mut compiled,
        dd,
        weights,
        temporal_hidden,
        si,
        last_token,
        past_kv,
    )
}

/// Sample all `num_slices` audio codebook tokens via the native RLX DepFormer,
/// mirroring the eager [`crate::depformer::DepFormer::sample`]: per-slice KV
/// inheritance, `linear_in(hidden) + emb(prev)` conditioning, greedy/top-k head.
pub fn depformer_sample_rlx(
    dd: &DepDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    temporal_hidden: &[f32],
    text_token: Option<u32>,
    forced: &[Option<u32>],
    lp: &mut LogitsProcessor,
    device: Device,
) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(dd.num_slices);
    let mut last_token = text_token;
    let mut past_kv: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for si in 0..dd.num_slices {
        let (logits, new_kv) = depformer_run_slice(
            dd,
            weights,
            temporal_hidden,
            si,
            last_token,
            &past_kv,
            device,
        )?;
        past_kv = new_kv;
        let token = lp.sample(ArrayView1::from(&logits))?;
        tokens.push(token);
        let next = forced.get(si).copied().flatten().unwrap_or(token);
        last_token = Some(next);
    }
    Ok(tokens)
}

/// Forced-conditioning per-slice logits (no sampling) — mirror of
/// [`crate::depformer::DepFormer::forced_logits`] for exact numeric parity.
pub fn depformer_forced_logits_rlx(
    dd: &DepDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    temporal_hidden: &[f32],
    text_token: Option<u32>,
    forced: &[u32],
    inherit: bool,
    device: Device,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(dd.num_slices);
    let mut last_token = text_token;
    let mut past_kv: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for si in 0..dd.num_slices {
        let (logits, new_kv) = depformer_run_slice(
            dd,
            weights,
            temporal_hidden,
            si,
            last_token,
            &past_kv,
            device,
        )?;
        past_kv = if inherit { new_kv } else { Vec::new() };
        out.push(logits);
        last_token = forced.get(si).copied();
    }
    Ok(out)
}

/// RoPE cos/sin tables `[seq, head_dim/2]` for absolute positions `0..seq`.
pub fn rope_tables(dims: &HeliumDims, seq: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dims.head_dim / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for pos in 0..seq {
        for i in 0..half {
            let inv_freq = 1.0f32 / dims.rope_theta.powf(i as f32 / half as f32);
            let f = pos as f32 * inv_freq;
            cos[pos * half + i] = f.cos();
            sin[pos * half + i] = f.sin();
        }
    }
    (cos, sin)
}

/// Compile + run the temporal graph for one batch of `inputs_embeds [seq, d]`,
/// returning logits `[seq, vocab_out]` flattened. Used for parity validation.
pub fn temporal_logits_rlx(
    dims: &HeliumDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    inputs_embeds: &[f32],
    seq: usize,
    device: Device,
) -> Result<Vec<f32>> {
    let graph = build_temporal_graph(dims, seq);
    let params = temporal_params(dims, weights)?;
    let mut compiled = Session::new(device).compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    let (cos, sin) = rope_tables(dims, seq);
    let outs = compiled.run(&[
        ("inputs_embeds", inputs_embeds),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
    ]);
    outs.into_iter()
        .next()
        .context("temporal graph produced no output")
}

/// Run one streaming KV-decode step at absolute position `pos`. `past_kv` holds
/// per-layer `(k, v)` from the previous step, each flattened `[1, pos, nh, hd]`
/// (empty when `pos == 0`). Returns `(logits [vocab_out], new_kv)` where each
/// `new_kv` entry is `[1, pos+1, nh, hd]` flattened.
pub fn temporal_decode_step_rlx(
    dims: &HeliumDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    inputs_embeds: &[f32],
    past_kv: &[(Vec<f32>, Vec<f32>)],
    pos: usize,
    device: Device,
) -> Result<(Vec<f32>, Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>)> {
    let graph = build_temporal_decode_graph(dims, pos);
    let params = temporal_params(dims, weights)?;
    let mut compiled = Session::new(device).compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    // RoPE tables for the single absolute position `pos`.
    let half = dims.head_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for i in 0..half {
        let inv_freq = 1.0f32 / dims.rope_theta.powf(i as f32 / half as f32);
        let f = pos as f32 * inv_freq;
        cos[i] = f.cos();
        sin[i] = f.sin();
    }
    let mut inputs: Vec<(String, &[f32])> = vec![
        ("inputs_embeds".to_string(), inputs_embeds),
        ("rope_cos".to_string(), cos.as_slice()),
        ("rope_sin".to_string(), sin.as_slice()),
    ];
    if pos > 0 {
        for (li, (k, v)) in past_kv.iter().enumerate() {
            inputs.push((format!("past_k_{li}"), k.as_slice()));
            inputs.push((format!("past_v_{li}"), v.as_slice()));
        }
    }
    let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, d)| (n.as_str(), *d)).collect();
    let mut it = compiled.run(&refs).into_iter();
    let logits = it.next().context("decode produced no logits")?;
    let hidden = it.next().context("decode produced no hidden")?;
    let mut new_kv = Vec::with_capacity(dims.n_layers);
    for _ in 0..dims.n_layers {
        let k = it.next().context("missing new_k")?;
        let v = it.next().context("missing new_v")?;
        new_kv.push((k, v));
    }
    Ok((logits, hidden, new_kv))
}

/// One bucketed decode step (fresh compile). `real_past_kv` is per-layer `(k, v)`
/// each flattened `[past_seq, nh, hd]` (real, unpadded; empty when `past_seq==0`);
/// it is zero-padded to `upper`. Returns `(logits, hidden, single-token new_kv)` —
/// append the `new_kv` to the real cache. Requires `past_seq < upper` (room for
/// padding) and the new token attends at slot `upper`.
pub fn temporal_decode_bucketed_rlx(
    dims: &HeliumDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    inputs_embeds: &[f32],
    real_past_kv: &[(Vec<f32>, Vec<f32>)],
    past_seq: usize,
    upper: usize,
    device: Device,
) -> Result<(Vec<f32>, Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>)> {
    let mut compiled =
        Session::new(device).compile(build_temporal_decode_graph_bucketed(dims, upper));
    for (name, data) in &temporal_params(dims, weights)? {
        compiled.set_param(name, data);
    }
    compiled.set_param("kv_one", &[1.0]);
    decode_bucketed_run(
        &mut compiled,
        dims,
        inputs_embeds,
        real_past_kv,
        past_seq,
        upper,
    )
}

/// Drive a (pre-compiled, params-set) bucketed decode graph for one frame.
pub fn decode_bucketed_run(
    compiled: &mut rlx_runtime::CompiledGraph,
    dims: &HeliumDims,
    inputs_embeds: &[f32],
    real_past_kv: &[(Vec<f32>, Vec<f32>)],
    past_seq: usize,
    upper: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>)> {
    let half = dims.head_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for i in 0..half {
        let inv_freq = 1.0f32 / dims.rope_theta.powf(i as f32 / half as f32);
        let f = past_seq as f32 * inv_freq;
        cos[i] = f.cos();
        sin[i] = f.sin();
    }
    let mask = bucket_decode_mask(past_seq, upper);
    let kvw = dims.n_heads * dims.head_dim;
    let pad_len = upper * kvw;
    let padded: Vec<(Vec<f32>, Vec<f32>)> = (0..dims.n_layers)
        .map(|li| {
            let (rk, rv) = real_past_kv
                .get(li)
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .unwrap_or((&[], &[]));
            let mut pk = rk.to_vec();
            pk.resize(pad_len, 0.0);
            let mut pv = rv.to_vec();
            pv.resize(pad_len, 0.0);
            (pk, pv)
        })
        .collect();

    let mut inputs: Vec<(String, &[f32])> = vec![
        ("inputs_embeds".to_string(), inputs_embeds),
        ("rope_cos".to_string(), cos.as_slice()),
        ("rope_sin".to_string(), sin.as_slice()),
        ("attn_mask".to_string(), mask.as_slice()),
    ];
    for (li, (k, v)) in padded.iter().enumerate() {
        inputs.push((format!("past_k_{li}"), k.as_slice()));
        inputs.push((format!("past_v_{li}"), v.as_slice()));
    }
    let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, d)| (n.as_str(), *d)).collect();
    let mut it = compiled.run(&refs).into_iter();
    let logits = it.next().context("bucketed decode produced no logits")?;
    let hidden = it.next().context("bucketed decode produced no hidden")?;
    let mut new_kv = Vec::with_capacity(dims.n_layers);
    for _ in 0..dims.n_layers {
        let k = it.next().context("missing new_k")?;
        let v = it.next().context("missing new_v")?;
        new_kv.push((k, v));
    }
    Ok((logits, hidden, new_kv))
}

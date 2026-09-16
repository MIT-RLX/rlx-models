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

//! DFlash drafter graphs.
//!
//! DFlash is **not** an encoder feeding a decoder stack. It is three separate
//! passes over two different kinds of batch, and conflating them silently
//! produces a drafter that runs, loads every weight, and proposes nonsense:
//!
//! 1. [`build_encoder_graph`] — `concat(target taps) -> fc -> enc.output_norm`.
//!    That is the *whole* encoder. It never touches a decoder block.
//! 2. [`build_kv_inject_graph`] — the fused features from (1) projected
//!    through each layer's `wk`/`wv` (K getting `attn_k_norm` + RoPE, V
//!    neither) to become that layer's KV-cache entries for the committed
//!    tokens. No attention, no FFN, no residual.
//! 3. [`build_decoder_graph`] — the noise block `[anchor, MASK, MASK, …]`
//!    embedded from the token table and run through the decoder blocks with
//!    **non-causal** attention over `[injected cache ‖ this block]`. Every
//!    block position is denoised in one pass, which is what makes DFlash a
//!    block drafter rather than an autoregressive one.
//!
//! The drafter has no embedding and no LM head of its own — it reuses the
//! target's. That is what makes it Eagle-style rather than a small standalone
//! model, and it is why `token_embd.weight` / `output.weight` fall back to
//! caller-supplied params when the drafter's GGUF omits them.
//!
//! DFlash2 checkpoints additionally carry a dynamic depthwise convolution
//! around each sublayer ([`crate::conv`]) and a candidate selector on the
//! output ([`crate::selector`]); both switch on from
//! [`DflashConfig::dflash2`].

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::config::{Dflash2Config, DflashConfig};
use crate::conv::{ConvSide, emit_dyn_conv};
use crate::selector::emit_selector_lattice;

type Packed = HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>;
type Params = HashMap<String, Vec<f32>>;

/// Load a norm gain (F32, 1-D).
fn load_norm(
    g: &mut Graph,
    params: &mut Params,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<NodeId> {
    if let Some(id) = g.param_id(key) {
        return Ok(id);
    }
    let (data, shape) = weights
        .take(key)
        .map_err(|e| anyhow!("dflash: missing norm {key}: {e}"))?;
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok(id)
}

/// Load a dense F32 tensor verbatim, keeping the on-disk shape.
fn load_raw(
    g: &mut Graph,
    params: &mut Params,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<NodeId> {
    if let Some(id) = g.param_id(key) {
        return Ok(id);
    }
    let (data, shape) = weights
        .take(key)
        .map_err(|e| anyhow!("dflash: missing tensor {key}: {e}"))?;
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok(id)
}

/// Load a projection, keeping K-quant weights packed when the loader offers it.
fn load_proj(
    g: &mut Graph,
    params: &mut Params,
    packed: &mut Packed,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<(NodeId, Option<rlx_ir::quant::QuantScheme>)> {
    if let Some(id) = g.param_id(key) {
        return Ok((id, packed.get(key).map(|(s, _)| *s)));
    }
    if let Some((scheme, shape)) = weights.packed_meta(key) {
        let nbytes = weights
            .tensor_bytes_borrowed(key)
            .ok_or_else(|| anyhow!("dflash: packed {key} has no bytes"))?
            .len();
        let id = g.param(key, Shape::new(&[nbytes], DType::U8));
        packed.insert(key.to_string(), (scheme, shape));
        return Ok((id, Some(scheme)));
    }
    // F32 fallback: store transposed so a plain `mm` works.
    let (data, shape) = weights
        .take_transposed(key)
        .map_err(|e| anyhow!("dflash: missing weight {key}: {e}"))?;
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok((id, None))
}

/// A projection the drafter may not ship, because it shares the target's.
///
/// Declared as a plain param either way, so the caller uploads the target's
/// tensor under the same name when the drafter's GGUF omits it. Silently
/// skipping it would leave the graph unable to embed or score anything.
fn load_shared_proj(
    g: &mut Graph,
    params: &mut Params,
    packed: &mut Packed,
    weights: &mut dyn WeightLoader,
    key: &str,
    fallback: Shape,
    missing: &mut Vec<String>,
) -> (NodeId, Option<rlx_ir::quant::QuantScheme>) {
    match load_proj(g, params, packed, weights, key) {
        Ok(hit) => hit,
        Err(_) => {
            missing.push(key.to_string());
            (g.param(key, fallback), None)
        }
    }
}

/// [`load_shared_proj`] for a tensor consumed verbatim rather than transposed.
fn load_shared_raw(
    g: &mut Graph,
    params: &mut Params,
    weights: &mut dyn WeightLoader,
    key: &str,
    fallback: Shape,
    missing: &mut Vec<String>,
) -> NodeId {
    load_raw(g, params, weights, key).unwrap_or_else(|_| {
        missing.push(key.to_string());
        g.param(key, fallback)
    })
}

fn emit_proj(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    scheme: Option<rlx_ir::quant::QuantScheme>,
    out: Shape,
) -> NodeId {
    match scheme {
        Some(s) => g.add_node(Op::DequantMatMul { scheme: s }, vec![x, w], out),
        None => g.mm(x, w),
    }
}

/// RMSNorm each head over `head_dim` with a shared `[head_dim]` gain.
#[allow(clippy::too_many_arguments)]
fn per_head_rms(
    g: &mut Graph,
    x: NodeId,
    gamma: NodeId,
    beta: NodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> NodeId {
    let flat = (batch * seq * heads) as i64;
    let r = g.reshape_(x, vec![flat, head_dim as i64]);
    let n = g.rms_norm(r, gamma, beta, eps);
    g.reshape_(n, vec![batch as i64, seq as i64, (heads * head_dim) as i64])
}

/// Expand `n_kv` KV heads to `n_kv * group` by broadcasting each head.
///
/// `Op::Expand` rather than narrow+concat: the broadcast is one node instead
/// of `n_kv * group`, which on a decode-shaped graph is the difference between
/// a handful of launches and hundreds.
fn repeat_kv(
    g: &mut Graph,
    x: NodeId,
    batch: usize,
    seq: usize,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> NodeId {
    if group == 1 {
        return x;
    }
    let (b, s, nkv, dh) = (
        batch as i64,
        seq as i64,
        num_kv_heads as i64,
        head_dim as i64,
    );
    let r = g.reshape_(x, vec![b, s, nkv, 1, dh]);
    let e = g.add_node(
        Op::Expand {
            target_shape: vec![b, s, nkv, group as i64, dh],
        },
        vec![r],
        Shape::new(&[batch, seq, num_kv_heads, group, head_dim], DType::F32),
    );
    g.reshape_(e, vec![b, s, (num_kv_heads * group * head_dim) as i64])
}

/// RoPE tables for an explicit list of absolute positions.
///
/// The decoder's block sits at `[n, n + block)` for a running `n`, and the
/// injection pass rotates whatever committed positions it is fed, so neither
/// can bake a `0..seq` table into the graph the way a prefill-only pass can.
pub fn rope_tables(positions: &[usize], head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; positions.len() * half];
    let mut sin = vec![0f32; positions.len() * half];
    for (row, &p) in positions.iter().enumerate() {
        for i in 0..half {
            let inv = 1.0f64 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let a = p as f64 * inv;
            cos[row * half + i] = a.cos() as f32;
            sin[row * half + i] = a.sin() as f32;
        }
    }
    (cos, sin)
}

fn zero_beta(g: &mut Graph, params: &mut Params, name: &str, n: usize) -> NodeId {
    if let Some(id) = g.param_id(name) {
        return id;
    }
    let id = g.param(name, Shape::new(&[n], DType::F32));
    params.insert(name.to_string(), vec![0f32; n]);
    id
}

/// A DFlash checkpoint with a reduced draft vocabulary needs `d2t` to map its
/// rows back to target token ids. Not supported yet, and drafting target ids
/// straight out of a reduced-vocab head would emit wrong tokens rather than
/// fail, so refuse up front.
fn reject_d2t(weights: &mut dyn WeightLoader) -> Result<()> {
    if weights.packed_meta("d2t").is_some() || weights.tensor_bytes_borrowed("d2t").is_some() {
        bail!(
            "dflash: this checkpoint has a reduced draft vocab (`d2t`), which is not \
             supported yet — its draft rows would be read as target token ids"
        );
    }
    Ok(())
}

// ── 1. Encoder ──────────────────────────────────────────────────────────

/// `concat(target taps) -> fc -> enc.output_norm`.
///
/// Input  `dflash_taps`: `[batch, seq, n_taps * hidden]` — the target model's
/// residual streams at `cfg.target_layers`, concatenated in that order.
/// Output `[batch, seq, hidden]`, consumed by [`build_kv_inject_graph`].
pub fn build_encoder_graph(
    cfg: &DflashConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    packed: &mut Packed,
) -> Result<(Graph, Params)> {
    let mut g = Graph::new("dflash_encoder");
    let mut params: Params = HashMap::new();
    let f = DType::F32;
    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;

    let zero_h = zero_beta(&mut g, &mut params, "dflash.zero_beta.hidden", h);
    let taps = g.input(
        "dflash_taps",
        Shape::new(&[batch, seq, cfg.fused_input_dim()], f),
    );
    let (fc_w, fc_s) = load_proj(&mut g, &mut params, packed, weights, "fc.weight")?;
    let fused = emit_proj(&mut g, taps, fc_w, fc_s, Shape::new(&[batch, seq, h], f));
    // The norm comes AFTER fc — dflash.cpp calls this "encoder hidden_norm
    // (after fc)". Reversing the two is silent garbage.
    let enc_n = load_norm(&mut g, &mut params, weights, "enc.output_norm.weight")?;
    let out = g.rms_norm(fused, enc_n, zero_h, eps);

    g.set_outputs(vec![out]);
    Ok((g, params))
}

// ── 2. KV injection ─────────────────────────────────────────────────────

/// Project fused features into every layer's KV-cache entries.
///
/// Inputs:
/// * `dflash_fused` — `[batch, seq, hidden]`, the encoder's output.
/// * `rope_cos` / `rope_sin` — `[seq, head_dim/2]` at the tokens' absolute
///   positions (see [`rope_tables`]).
///
/// Outputs, in order: `inject_k_0, inject_v_0, …, inject_k_{L-1},
/// inject_v_{L-1}`, each `[batch, seq, kv_proj_dim]`. The runner appends these
/// to its per-layer cache; [`build_decoder_graph`] reads them back as
/// `past_k_{i}` / `past_v_{i}`.
///
/// Note what is *absent*: no `attn_norm` before the projections (the encoder's
/// own norm already ran), no attention, no FFN, no residual. K gets
/// `attn_k_norm` and RoPE; V gets neither.
pub fn build_kv_inject_graph(
    cfg: &DflashConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    packed: &mut Packed,
) -> Result<(Graph, Params)> {
    let mut g = Graph::new("dflash_kv_inject");
    let mut params: Params = HashMap::new();
    let f = DType::F32;
    let (h, dh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.num_key_value_heads);
    let eps = cfg.rms_norm_eps as f32;
    let half = dh / 2;

    let zero_dh = zero_beta(&mut g, &mut params, "dflash.zero_beta.head_dim", dh);
    let fused = g.input("dflash_fused", Shape::new(&[batch, seq, h], f));
    let cos = g.input("rope_cos", Shape::new(&[seq, half], f));
    let sin = g.input("rope_sin", Shape::new(&[seq, half], f));

    let mut outs = Vec::with_capacity(cfg.num_hidden_layers * 2);
    for il in 0..cfg.num_hidden_layers {
        let p = format!("blk.{il}");
        let kv_shape = Shape::new(&[batch, seq, cfg.kv_proj_dim()], f);

        let (kw, ks) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.attn_k.weight"),
        )?;
        let (vw, vs) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.attn_v.weight"),
        )?;
        let k = emit_proj(&mut g, fused, kw, ks, kv_shape.clone());
        let v = emit_proj(&mut g, fused, vw, vs, kv_shape);

        let kn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{p}.attn_k_norm.weight"),
        )?;
        let k = per_head_rms(&mut g, k, kn, zero_dh, batch, seq, nkv, dh, eps);
        let k = g.rope_styled(k, cos, sin, dh, cfg.rope_style);

        outs.push(k);
        outs.push(v);
    }

    g.set_outputs(outs);
    Ok((g, params))
}

// ── 3. Decoder ──────────────────────────────────────────────────────────

/// What [`build_decoder_graph`] put in the graph's output list.
///
/// Output order is fixed: the head outputs first, then this block's K/V per
/// layer so the runner can append the accepted prefix to its cache.
#[derive(Debug, Clone)]
pub struct DecoderOutputs {
    /// Index of `logits [batch, block, vocab]`. `None` on a DFlash2
    /// checkpoint, which returns the selector lattice instead — the whole
    /// point of the selector is that full-vocab logits never leave the device.
    pub logits: Option<usize>,
    /// Index of `hidden [batch, block, hidden]`, the post-`output_norm` states.
    pub hidden: usize,
    /// DFlash2 only: indices of `cand [batch, block, top_k]` and
    /// `scores [batch, block-1, top_k, top_k]`. Feed to
    /// [`crate::selector::walk_lattice`].
    pub selector: Option<(usize, usize)>,
    /// Index of the first `block_k_0`; layer `i`'s K/V are at
    /// `kv_base + 2*i` and `kv_base + 2*i + 1`.
    pub kv_base: usize,
    /// Params this checkpoint does NOT ship because it shares the target's —
    /// `token_embd.weight` and/or `output.weight`. They are declared in the
    /// graph but hold no data; the caller must upload the TARGET's copies.
    ///
    /// This is surfaced rather than left implicit because an unset param is
    /// silently wrong, not loudly broken: the graph runs and drafts noise.
    /// Every released DFlash/DFlash2 checkpoint hits this — they are
    /// Eagle-style heads with no embedding and no LM head of their own.
    pub shared_params: Vec<String>,
}

/// The noise-block decoder.
///
/// Inputs:
/// * `noise_tokens` — `[batch, block]`, f32-encoded ids laid out
///   `[anchor, MASK, MASK, …]`. The anchor is the last token the target
///   committed; the rest are `cfg.mask_token_id`.
/// * `past_k_{i}` / `past_v_{i}` — `[batch, past_cap, kv_proj_dim]`, the cache
///   filled by [`build_kv_inject_graph`], **padded** up to `past_cap`.
/// * `attn_mask` — `[batch, past_cap + block]`, `1.0` for a live key and `0.0`
///   for padding. The cache grows by up to `block + 1` every round, so a graph
///   keyed on the exact `past_seq` would recompile O(N) times over a
///   generation; padding to a power-of-two capacity and masking the slack
///   makes it O(log N).
/// * `rope_cos` / `rope_sin` — `[block, head_dim/2]` at the block's absolute
///   positions.
/// * `anchor_ids` — `[batch, 1]`, DFlash2 only: the anchor token id again, as
///   the seed predecessor for the selector lattice.
///
/// Attention is **non-causal** over `[past ‖ block]`: every block position sees
/// every other. That is the diffusion step — a causal mask here would turn the
/// block drafter back into an autoregressive one that only ever proposes its
/// first token correctly. `attn_mask` is a *padding* mask applied to all
/// queries alike, so it hides slack without reintroducing an ordering.
pub fn build_decoder_graph(
    cfg: &DflashConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    block: usize,
    past_seq: usize,
    packed: &mut Packed,
) -> Result<(Graph, Params, DecoderOutputs)> {
    reject_d2t(weights)?;

    let mut g = Graph::new("dflash_decoder");
    let mut params: Params = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let dh = cfg.head_dim;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let group = cfg.kv_group_size();
    let eps = cfg.rms_norm_eps as f32;
    let half = dh / 2;
    let total = past_seq + block;
    let v_sz = cfg.vocab_size;

    let zero_h = zero_beta(&mut g, &mut params, "dflash.zero_beta.hidden", h);
    let zero_dh = zero_beta(&mut g, &mut params, "dflash.zero_beta.head_dim", dh);

    let tokens = g.input("noise_tokens", Shape::new(&[batch, block], f));
    let cos = g.input("rope_cos", Shape::new(&[block, half], f));
    let sin = g.input("rope_sin", Shape::new(&[block, half], f));
    let attn_mask = g.input("attn_mask", Shape::new(&[batch, total], f));
    let past: Vec<(NodeId, NodeId)> = (0..cfg.num_hidden_layers)
        .map(|i| {
            let s = Shape::new(&[batch, past_seq, cfg.kv_proj_dim()], f);
            (
                g.input(format!("past_k_{i}"), s.clone()),
                g.input(format!("past_v_{i}"), s),
            )
        })
        .collect();

    // Token table: shared with the target when the drafter omits its own.
    // Loaded verbatim, NOT transposed — `Gather` indexes rows, so it needs
    // `[vocab, hidden]`. (`output.weight` below is the opposite: `mm` wants
    // `[hidden, vocab]`, so that one does get transposed.)
    let mut shared_params: Vec<String> = Vec::new();
    let tok_embd = load_shared_raw(
        &mut g,
        &mut params,
        weights,
        "token_embd.weight",
        Shape::new(&[v_sz, h], f),
        &mut shared_params,
    );
    let flat_tokens = g.reshape_(tokens, vec![(batch * block) as i64]);
    let embedded = g.gather_(tok_embd, flat_tokens, 0);
    let mut x = g.reshape_(embedded, vec![batch as i64, block as i64, h as i64]);
    if let Some(scale) = cfg.embedding_scale {
        let s = g.constant(scale as f64, f);
        x = g.mul(x, s);
    }

    let d2 = cfg.dflash2;
    let mut block_kv: Vec<NodeId> = Vec::with_capacity(cfg.num_hidden_layers * 2);

    for il in 0..cfg.num_hidden_layers {
        let p = format!("blk.{il}");
        let res = x;

        let an = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{p}.attn_norm.weight"),
        )?;
        let mut xn = g.rms_norm(x, an, zero_h, eps);

        // DFlash2: one projection per sublayer feeds both conv sides.
        let attn_delta = match d2 {
            None => None,
            Some(c) => {
                let (w, s) = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{p}.attn_conv_proj.weight"),
                )?;
                let base = load_raw(&mut g, &mut params, weights, &format!("{p}.attn_conv_base"))?;
                let d = emit_proj(
                    &mut g,
                    xn,
                    w,
                    s,
                    Shape::new(&[batch, block, c.dynamic_dim(h)], f),
                );
                xn = emit_dyn_conv(&mut g, xn, d, base, ConvSide::Pre, &c);
                Some((d, base, c))
            }
        };

        let (qw, qs) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.attn_q.weight"),
        )?;
        let (kw, ks) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.attn_k.weight"),
        )?;
        let (vw, vs) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.attn_v.weight"),
        )?;
        let q = emit_proj(
            &mut g,
            xn,
            qw,
            qs,
            Shape::new(&[batch, block, cfg.q_proj_dim()], f),
        );
        let k = emit_proj(
            &mut g,
            xn,
            kw,
            ks,
            Shape::new(&[batch, block, cfg.kv_proj_dim()], f),
        );
        let v = emit_proj(
            &mut g,
            xn,
            vw,
            vs,
            Shape::new(&[batch, block, cfg.kv_proj_dim()], f),
        );

        // Per-head QK-norm, then RoPE — same order as the reference.
        let qn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{p}.attn_q_norm.weight"),
        )?;
        let kn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{p}.attn_k_norm.weight"),
        )?;
        let q = per_head_rms(&mut g, q, qn, zero_dh, batch, block, nh, dh, eps);
        let k = per_head_rms(&mut g, k, kn, zero_dh, batch, block, nkv, dh, eps);

        // Pairing flavor comes from the checkpoint format (see
        // `DflashConfig::rope_style`), not from the architecture.
        let q = g.rope_styled(q, cos, sin, dh, cfg.rope_style);
        let k = g.rope_styled(k, cos, sin, dh, cfg.rope_style);

        // Hand the runner this block's K/V before they are GQA-expanded.
        block_kv.push(k);
        block_kv.push(v);

        let k_all = g.concat_(vec![past[il].0, k], 1);
        let v_all = g.concat_(vec![past[il].1, v], 1);
        let k_rep = repeat_kv(&mut g, k_all, batch, total, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v_all, batch, total, nkv, dh, group);

        let attn_shape = Shape::new(&[batch, block, cfg.q_proj_dim()], f);
        let attn = g.attention(q, k_rep, v_rep, attn_mask, nh, dh, attn_shape);

        let (ow, os) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.attn_output.weight"),
        )?;
        let mut attn_out = emit_proj(&mut g, attn, ow, os, Shape::new(&[batch, block, h], f));
        if let Some((d, base, c)) = attn_delta {
            attn_out = emit_dyn_conv(&mut g, attn_out, d, base, ConvSide::Post, &c);
        }
        let ffn_inp = g.add(res, attn_out);

        let fn_ = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{p}.ffn_norm.weight"),
        )?;
        let mut normed = g.rms_norm(ffn_inp, fn_, zero_h, eps);

        let ffn_delta = match d2 {
            None => None,
            Some(c) => {
                let (w, s) = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{p}.ffn_conv_proj.weight"),
                )?;
                let base = load_raw(&mut g, &mut params, weights, &format!("{p}.ffn_conv_base"))?;
                let d = emit_proj(
                    &mut g,
                    normed,
                    w,
                    s,
                    Shape::new(&[batch, block, c.dynamic_dim(h)], f),
                );
                normed = emit_dyn_conv(&mut g, normed, d, base, ConvSide::Pre, &c);
                Some((d, base, c))
            }
        };

        let inter = cfg.intermediate_size;
        let (gw, gs) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.ffn_gate.weight"),
        )?;
        let (uw, us) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.ffn_up.weight"),
        )?;
        let gate = emit_proj(
            &mut g,
            normed,
            gw,
            gs,
            Shape::new(&[batch, block, inter], f),
        );
        let up = emit_proj(
            &mut g,
            normed,
            uw,
            us,
            Shape::new(&[batch, block, inter], f),
        );
        let act = {
            let s = g.silu(gate);
            g.mul(s, up)
        };
        let (dw, ds) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{p}.ffn_down.weight"),
        )?;
        let mut ffn_out = emit_proj(&mut g, act, dw, ds, Shape::new(&[batch, block, h], f));
        if let Some((d, base, c)) = ffn_delta {
            ffn_out = emit_dyn_conv(&mut g, ffn_out, d, base, ConvSide::Post, &c);
        }
        x = g.add(ffn_inp, ffn_out);
    }

    let on = load_norm(&mut g, &mut params, weights, "output_norm.weight")?;
    let hidden = g.rms_norm(x, on, zero_h, eps);

    // LM head: shared with the target when the drafter omits its own.
    let (lm_w, lm_s) = load_shared_proj(
        &mut g,
        &mut params,
        packed,
        weights,
        "output.weight",
        Shape::new(&[h, v_sz], f),
        &mut shared_params,
    );
    let mut logits = emit_proj(
        &mut g,
        hidden,
        lm_w,
        lm_s,
        Shape::new(&[batch, block, v_sz], f),
    );
    if let Some(scale) = cfg.logit_scale {
        let s = g.constant(scale as f64, f);
        logits = g.mul(logits, s);
    }
    if let Some(cap) = cfg.final_logit_softcapping {
        let inv = g.constant(1.0 / cap as f64, f);
        let c = g.constant(cap as f64, f);
        let t = g.mul(logits, inv);
        let t = g.tanh(t);
        logits = g.mul(t, c);
    }

    // Output list. DFlash2 keeps the logits on-device and ships the lattice.
    let mut outs: Vec<NodeId> = Vec::new();
    let mut meta = DecoderOutputs {
        logits: None,
        hidden: 0,
        selector: None,
        kv_base: 0,
        shared_params,
    };
    match d2 {
        Some(c) => {
            let (cand, scores) = emit_selector_weights(
                &mut g,
                &mut params,
                packed,
                weights,
                cfg,
                logits,
                hidden,
                &c,
            )?;
            outs.push(cand);
            outs.push(scores);
            meta.selector = Some((0, 1));
            meta.hidden = 2;
            outs.push(hidden);
        }
        None => {
            meta.logits = Some(0);
            meta.hidden = 1;
            outs.push(logits);
            outs.push(hidden);
        }
    }
    meta.kv_base = outs.len();
    outs.extend(block_kv);

    g.set_outputs(outs);
    Ok((g, params, meta))
}

/// Load the three selector tensors and emit the lattice.
#[allow(clippy::too_many_arguments)]
fn emit_selector_weights(
    g: &mut Graph,
    params: &mut Params,
    packed: &mut Packed,
    weights: &mut dyn WeightLoader,
    cfg: &DflashConfig,
    logits: NodeId,
    hidden: NodeId,
    d2: &Dflash2Config,
) -> Result<(NodeId, NodeId)> {
    let f = DType::F32;
    let ls = g.shape(logits).clone();
    let (batch, block) = (ls.dim(0).unwrap_static(), ls.dim(1).unwrap_static());
    let anchor = g.input("anchor_ids", Shape::new(&[batch, 1], f));
    // The codebooks are row-gathered by token id, so they must be dense.
    let a = load_raw(g, params, weights, "selector_predecessor.weight")?;
    let b = load_raw(g, params, weights, "selector_successor.weight")?;
    // `H` may ship quantized — it does in every released checkpoint — so it
    // reaches the graph as a packed U8 blob. Project through `emit_proj` and
    // hand the lattice the *result*; a plain `mm` on the packed weight is a
    // rank-1-vs-rank-3 shape error at build time.
    let (hw, hs) = load_proj(g, params, packed, weights, "selector_hidden.weight")?;
    let hgate = emit_proj(
        g,
        hidden,
        hw,
        hs,
        Shape::new(&[batch, block, d2.selector_rank], f),
    );
    let _ = cfg;
    Ok(emit_selector_lattice(g, logits, hgate, anchor, a, b, d2))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use rlx_ir::op::OpKind;
    use rlx_runtime::{Device, Session};

    /// In-memory checkpoint. Tensors are stored in GGUF's reversed row-major
    /// order (`[out, in]`), so `take_transposed` yields the `[in, out]` a plain
    /// `mm` wants — the same convention the real `GgufLoader` presents.
    pub(crate) struct SynthLoader(HashMap<String, (Vec<f32>, Vec<usize>)>);

    impl SynthLoader {
        fn put(&mut self, key: &str, shape: &[usize]) {
            let n: usize = shape.iter().product();
            let mut st = key.bytes().fold(0x9e37_79b9_7f4a_7c15u64, |a, b| {
                a.rotate_left(5) ^ u64::from(b)
            });
            let data = (0..n)
                .map(|_| {
                    st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (((st >> 33) as f64 / (1u64 << 30) as f64) - 1.0) as f32 * 0.3
                })
                .collect();
            self.0.insert(key.to_string(), (data, shape.to_vec()));
        }
    }

    impl WeightLoader for SynthLoader {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| anyhow!("synth: no tensor {key}"))
        }
        fn remaining_keys(&self) -> Vec<String> {
            self.0.keys().cloned().collect()
        }
        fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            let (data, shape) = self.take(key)?;
            let r = shape.len();
            assert!(r >= 2, "cannot transpose rank-{r} {key}");
            let (rows, cols) = (shape[r - 2], shape[r - 1]);
            let batch = data.len() / (rows * cols);
            let mut out = vec![0f32; data.len()];
            for b in 0..batch {
                for i in 0..rows {
                    for j in 0..cols {
                        out[b * rows * cols + j * rows + i] = data[b * rows * cols + i * cols + j];
                    }
                }
            }
            let mut s = shape.clone();
            s.swap(r - 2, r - 1);
            Ok((out, s))
        }
    }

    const V: usize = 12;
    const H: usize = 8;
    const FF: usize = 16;
    const NH: usize = 2;
    const NKV: usize = 1;
    const DH: usize = 4;
    const L: usize = 2;

    pub(crate) fn cfg(dflash2: Option<Dflash2Config>) -> DflashConfig {
        DflashConfig {
            vocab_size: V,
            hidden_size: H,
            intermediate_size: FF,
            num_hidden_layers: L,
            num_attention_heads: NH,
            num_key_value_heads: NKV,
            head_dim: DH,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 128,
            block_size: 4,
            target_layers: vec![0, 1],
            sliding_window: None,
            rope_style: rlx_ir::RopeStyle::GptJ,
            mask_token_id: Some(11),
            sample_from_anchor: false,
            dflash2,
            logit_scale: None,
            final_logit_softcapping: None,
            embedding_scale: None,
        }
    }

    pub(crate) fn synth(c: &DflashConfig) -> SynthLoader {
        let mut w = SynthLoader(HashMap::new());
        w.put("fc.weight", &[H, c.fused_input_dim()]);
        w.put("enc.output_norm.weight", &[H]);
        w.put("output_norm.weight", &[H]);
        w.put("token_embd.weight", &[V, H]);
        w.put("output.weight", &[V, H]);
        for i in 0..L {
            let p = format!("blk.{i}");
            w.put(&format!("{p}.attn_norm.weight"), &[H]);
            w.put(&format!("{p}.ffn_norm.weight"), &[H]);
            w.put(&format!("{p}.attn_q.weight"), &[NH * DH, H]);
            w.put(&format!("{p}.attn_k.weight"), &[NKV * DH, H]);
            w.put(&format!("{p}.attn_v.weight"), &[NKV * DH, H]);
            w.put(&format!("{p}.attn_output.weight"), &[H, NH * DH]);
            w.put(&format!("{p}.attn_q_norm.weight"), &[DH]);
            w.put(&format!("{p}.attn_k_norm.weight"), &[DH]);
            w.put(&format!("{p}.ffn_gate.weight"), &[FF, H]);
            w.put(&format!("{p}.ffn_up.weight"), &[FF, H]);
            w.put(&format!("{p}.ffn_down.weight"), &[H, FF]);
        }
        if let Some(d2) = c.dflash2 {
            let proj = d2.dynamic_dim(H);
            for i in 0..L {
                let p = format!("blk.{i}");
                w.put(&format!("{p}.attn_conv_base"), &[2, d2.conv_kernel_size, H]);
                w.put(&format!("{p}.attn_conv_proj.weight"), &[proj, H]);
                w.put(&format!("{p}.ffn_conv_base"), &[2, d2.conv_kernel_size, H]);
                w.put(&format!("{p}.ffn_conv_proj.weight"), &[proj, H]);
            }
            w.put("selector_predecessor.weight", &[V, d2.selector_rank]);
            w.put("selector_successor.weight", &[V, d2.selector_rank]);
            w.put("selector_hidden.weight", &[d2.selector_rank, H]);
        }
        w
    }

    fn kinds(g: &Graph) -> Vec<OpKind> {
        g.nodes().iter().map(|n| n.op.kind()).collect()
    }

    /// The encoder is `fc` + a norm. Nothing else. The old builder ran the
    /// whole decoder stack here, which is the bug this pins down.
    #[test]
    fn encoder_is_only_fc_and_norm() {
        let c = cfg(None);
        let mut w = synth(&c);
        let mut packed = HashMap::new();
        let (g, _) = build_encoder_graph(&c, &mut w, 1, 3, &mut packed).unwrap();
        let ks = kinds(&g);
        assert!(
            !ks.contains(&OpKind::Attention),
            "encoder must not attend: {ks:?}"
        );
        assert!(
            !ks.contains(&OpKind::Rope),
            "encoder has no positions to rotate: {ks:?}"
        );
        assert_eq!(ks.iter().filter(|k| **k == OpKind::RmsNorm).count(), 1);
    }

    /// Injection projects K/V only — no attention, no FFN, no residual — and
    /// hands back one K and one V per layer.
    #[test]
    fn kv_inject_emits_k_and_v_per_layer() {
        let c = cfg(None);
        let mut w = synth(&c);
        let mut packed = HashMap::new();
        let (g, params) = build_kv_inject_graph(&c, &mut w, 1, 3, &mut packed).unwrap();
        let ks = kinds(&g);
        assert!(
            !ks.contains(&OpKind::Attention),
            "injection must not attend"
        );
        assert_eq!(g.outputs.len(), L * 2);

        let mut s = Session::new(Device::Cpu).compile(g);
        for (k, v) in &params {
            s.set_param(k, v);
        }
        let (cos, sin) = rope_tables(&[0, 1, 2], DH, c.rope_theta);
        let fused = vec![0.1f32; 3 * H];
        let out = s.run(&[
            ("dflash_fused", fused.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ]);
        for o in &out {
            assert_eq!(o.len(), 3 * c.kv_proj_dim());
            assert!(o.iter().all(|v| v.is_finite()));
        }
    }

    fn run_decoder(c: &DflashConfig, tokens: &[f32], past_seq: usize) -> Vec<Vec<f32>> {
        let mut w = synth(c);
        let mut packed = HashMap::new();
        let block = tokens.len();
        let (g, params, _) =
            build_decoder_graph(c, &mut w, 1, block, past_seq, &mut packed).unwrap();
        let mut s = Session::new(Device::Cpu).compile(g);
        for (k, v) in &params {
            s.set_param(k, v);
        }
        let positions: Vec<usize> = (past_seq..past_seq + block).collect();
        let (cos, sin) = rope_tables(&positions, c.head_dim, c.rope_theta);
        let past = vec![0.05f32; past_seq * c.kv_proj_dim()];
        let anchor = [tokens[0]];
        let mask = vec![1.0f32; past_seq + block];

        let mut inputs: Vec<(String, &[f32])> = vec![
            ("noise_tokens".into(), tokens),
            ("rope_cos".into(), cos.as_slice()),
            ("rope_sin".into(), sin.as_slice()),
            ("attn_mask".into(), mask.as_slice()),
        ];
        if c.dflash2.is_some() {
            inputs.push(("anchor_ids".into(), &anchor));
        }
        let names: Vec<String> = (0..c.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        for n in &names {
            inputs.push((n.clone(), past.as_slice()));
        }
        let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, d)| (n.as_str(), *d)).collect();
        s.run(&refs)
    }

    /// **The regression guard for the restructure.** Perturbing the LAST slot
    /// of the noise block must change position 0's hidden state. Under the
    /// causal mask the previous builder used, it could not — which is exactly
    /// how a block drafter degrades into proposing one good token and then
    /// noise.
    #[test]
    fn decoder_attention_is_non_causal_across_the_block() {
        let c = cfg(None);
        let out_a = run_decoder(&c, &[3.0, 11.0, 11.0, 11.0], 2);
        let out_b = run_decoder(&c, &[3.0, 11.0, 11.0, 7.0], 2);

        let (ha, hb) = (&out_a[1], &out_b[1]); // hidden [1, block, H]
        let pos0_delta: f32 = ha[..H]
            .iter()
            .zip(&hb[..H])
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(
            pos0_delta > 1e-4,
            "position 0 did not react to the last block token (delta {pos0_delta}) — \
             attention is still causal"
        );
    }

    /// The block's own K/V come back so the runner can append the accepted
    /// prefix to its cache without a second forward.
    #[test]
    fn decoder_returns_block_kv_per_layer() {
        let c = cfg(None);
        let mut w = synth(&c);
        let mut packed = HashMap::new();
        let (g, _, meta) = build_decoder_graph(&c, &mut w, 1, 4, 2, &mut packed).unwrap();
        assert_eq!(meta.logits, Some(0));
        assert_eq!(meta.hidden, 1);
        assert_eq!(meta.kv_base, 2);
        assert_eq!(g.outputs.len(), 2 + L * 2);

        let out = run_decoder(&c, &[3.0, 11.0, 11.0, 11.0], 2);
        assert_eq!(out[meta.logits.unwrap()].len(), 4 * V);
        for i in 0..L {
            assert_eq!(out[meta.kv_base + 2 * i].len(), 4 * c.kv_proj_dim());
            assert_eq!(out[meta.kv_base + 2 * i + 1].len(), 4 * c.kv_proj_dim());
        }
    }

    /// DFlash2 ships the lattice instead of the logits — the D2H saving is the
    /// point of the selector, so a build that still returned full-vocab logits
    /// would have thrown it away.
    #[test]
    fn dflash2_decoder_returns_lattice_not_logits() {
        let d2 = Dflash2Config {
            conv_kernel_size: 2,
            conv_group_size: 2,
            selector_rank: 3,
            selector_top_k: 2,
        };
        let c = cfg(Some(d2));
        let mut w = synth(&c);
        let mut packed = HashMap::new();
        let (_, _, meta) = build_decoder_graph(&c, &mut w, 1, 4, 2, &mut packed).unwrap();
        assert!(meta.logits.is_none(), "DFlash2 must keep logits on-device");
        assert_eq!(meta.selector, Some((0, 1)));

        let out = run_decoder(&c, &[3.0, 11.0, 11.0, 11.0], 2);
        assert_eq!(out[0].len(), 4 * d2.selector_top_k);
        assert_eq!(out[1].len(), 3 * d2.selector_top_k * d2.selector_top_k);
        assert!(out[1].iter().all(|v| v.is_finite()));
        // Candidate ids must be real vocab rows, not garbage from a bad gather.
        assert!(out[0].iter().all(|v| *v >= 0.0 && (*v as usize) < V));
    }

    /// A reduced-vocab checkpoint would emit draft-local ids as if they were
    /// target ids. Refuse rather than mistranslate.
    #[test]
    fn reduced_draft_vocab_is_refused_not_mistranslated() {
        struct D2t(SynthLoader);
        impl WeightLoader for D2t {
            fn len(&self) -> usize {
                self.0.len()
            }
            fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                self.0.take(k)
            }
            fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                self.0.take_transposed(k)
            }
            fn remaining_keys(&self) -> Vec<String> {
                self.0.remaining_keys()
            }
            fn tensor_bytes_borrowed(&self, k: &str) -> Option<&[u8]> {
                (k == "d2t").then_some(&[][..])
            }
        }
        let c = cfg(None);
        let mut w = D2t(synth(&c));
        let mut packed = HashMap::new();
        let err = build_decoder_graph(&c, &mut w, 1, 4, 2, &mut packed).unwrap_err();
        assert!(format!("{err}").contains("d2t"), "got: {err}");
    }
}

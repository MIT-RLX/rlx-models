// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// LLaMA-3.2 graph builder — GQA + RoPE + SwiGLU, no QK-norm.

use crate::config::{DenseArch, Llama32Config, NormKind};
use crate::rope::{build_rope_tables, resolve_inv_freq, rope_slice};
use anyhow::{Result, anyhow, bail};
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::shape::{self};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

fn apply_qk_rope(
    g: &mut Graph,
    q: NodeId,
    k: NodeId,
    cos_id: NodeId,
    sin_id: NodeId,
    cfg: &Llama32Config,
) -> (NodeId, NodeId) {
    let dh = cfg.head_dim();
    let n_rot = cfg.n_rot();
    if n_rot < dh {
        (
            g.rope_n_styled(q, cos_id, sin_id, dh, n_rot, cfg.rope_style),
            g.rope_n_styled(k, cos_id, sin_id, dh, n_rot, cfg.rope_style),
        )
    } else {
        (
            g.rope_styled(q, cos_id, sin_id, dh, cfg.rope_style),
            g.rope_styled(k, cos_id, sin_id, dh, cfg.rope_style),
        )
    }
}

/// Build a HIR-stage LLaMA-3.2 forward module (fusion-first).
pub fn build_llama32_hir_sized(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    with_kv_outputs: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_llama32_hir_sized_impl(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        with_kv_outputs,
        false,
        false,
    )
}

/// Static-seq prefill HIR (last-position logits + optional KV tap).
pub fn build_llama32_prefill_hir_sized_ext(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_llama32_hir_sized_impl(cfg, weights, batch, seq, true, with_kv_outputs, true, false)
}

/// Prefill HIR with symbolic seq dim (`sym::SEQ`) for dynamic compile cache.
pub fn build_llama32_prefill_hir_dynamic_ext(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    max_seq: usize,
    with_kv_outputs: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_llama32_hir_sized_impl(
        cfg,
        weights,
        batch,
        max_seq,
        true,
        with_kv_outputs,
        true,
        true,
    )
}

pub fn build_llama32_graph_sized(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_llama32_graph_sized_impl(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        with_kv_outputs,
        false,
    )
}

pub fn build_llama32_graph_sized_last_logits(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_llama32_graph_sized_impl(cfg, weights, batch, seq, true, with_kv_outputs, true)
}

/// Prefill graph with per-layer KV side outputs only (no LM head).
pub fn build_llama32_graph_sized_kv_tap(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_llama32_graph_sized_impl(cfg, weights, batch, seq, false, true, false)
}

fn build_llama32_graph_sized_impl(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    with_kv_outputs: bool,
    last_logits_only: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::Llama32PrefillOpts {
        batch,
        seq,
        dynamic_seq: false,
        with_lm_head,
        with_kv_outputs,
        last_logits_only,
        inputs_embeds: false,
        profile: None,
    };
    rlx_core::flow_util::graph_from_built(crate::flow::build_llama32_prefill_built(
        cfg, weights, &opts,
    )?)
}

fn build_llama32_hir_sized_impl(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    with_kv_outputs: bool,
    last_logits_only: bool,
    dynamic_seq: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    if dynamic_seq && batch != 1 {
        return Err(anyhow!("llama32: dynamic_seq prefill requires batch=1"));
    }

    use crate::flow::{Llama32PrefillOpts, build_llama32_prefill_flow};

    let opts = Llama32PrefillOpts {
        batch,
        seq,
        dynamic_seq,
        with_lm_head,
        with_kv_outputs,
        last_logits_only,
        inputs_embeds: false,
        profile: None,
    };
    build_llama32_prefill_flow(cfg, weights, &opts)
}

pub fn build_llama32_decode_graph_sized(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_llama32_decode_graph_sized_ext(cfg, weights, batch, past_seq, false)
}

/// HIR-stage decode graph (KV-cache concat + causal/custom-mask attention).
pub fn build_llama32_decode_hir_sized(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_llama32_decode_hir_sized_ext(cfg, weights, batch, past_seq, false)
}

pub fn build_llama32_decode_hir_sized_ext(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_llama32_decode_hir_sized_impl(cfg, weights, batch, past_seq, use_custom_mask, false)
}

/// Decode HIR with symbolic past length (`sym::PAST_SEQ`).
pub fn build_llama32_decode_hir_dynamic_ext(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    max_past_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_llama32_decode_hir_sized_impl(cfg, weights, batch, max_past_seq, false, true)
}

fn build_llama32_decode_hir_sized_impl(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
    dynamic_past: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;

    use crate::flow::{Llama32DecodeOpts, build_llama32_decode_flow};

    let opts = Llama32DecodeOpts {
        batch,
        past_seq,
        dynamic_past,
        use_custom_mask,
        profile: None,
    };
    build_llama32_decode_flow(cfg, weights, &opts)
}

pub fn build_llama32_decode_graph_sized_ext(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    use crate::flow::{Llama32DecodeOpts, build_llama32_decode_graph};

    let opts = Llama32DecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask,
        profile: None,
    };
    build_llama32_decode_graph(cfg, weights, &opts)
}

#[allow(dead_code)]
fn gather_last_token(
    g: &mut HirMut,
    h: HirNodeId,
    batch: usize,
    last_token_idx: HirNodeId,
) -> HirNodeId {
    let idx_2d = g.reshape_(last_token_idx, vec![batch as i64, 1]);
    g.gather_(h, idx_2d, 1)
}

fn validate_cfg(cfg: &Llama32Config) -> Result<()> {
    if !cfg
        .num_attention_heads
        .is_multiple_of(cfg.num_key_value_heads)
    {
        return Err(anyhow!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            cfg.num_attention_heads,
            cfg.num_key_value_heads
        ));
    }
    if cfg.attention_bias {
        return Err(anyhow!("attention_bias=true not yet wired for llama32"));
    }
    Ok(())
}

fn take_rope_freqs(weights: &mut dyn WeightLoader) -> Option<Vec<f32>> {
    weights.take("rope_freqs.weight").ok().map(|(data, _)| data)
}

#[allow(dead_code)]
fn repeat_kv_hir(
    g: &mut HirMut,
    x: HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> HirNodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces: Vec<HirNodeId> = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

fn repeat_kv(
    g: &mut Graph,
    x: NodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> NodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces: Vec<NodeId> = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

#[allow(dead_code)]
fn load_p_hir(
    hir: &mut HirModule,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
    transpose: bool,
) -> Result<HirNodeId> {
    let (data, shape) = if transpose {
        weights.take_transposed(key)?
    } else {
        weights.take(key)?
    };
    let ir_shape = Shape::new(&shape, DType::F32);
    let id = hir.param(key, ir_shape);
    params.insert(key.to_string(), data);
    Ok(id)
}

#[allow(dead_code)]
fn synth_zero_hir(
    hir: &mut HirModule,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    len: usize,
) -> HirNodeId {
    let id = hir.param(name, Shape::new(&[len], DType::F32));
    params.insert(name.to_string(), vec![0f32; len]);
    id
}

fn load_p(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
    transpose: bool,
) -> Result<NodeId> {
    // Shared across Looped-Transformer iterations (Nanbeige `num_loops` > 1).
    if let Some(id) = g.param_id(key) {
        return Ok(id);
    }
    let (data, shape) = if transpose {
        weights.take_transposed(key)?
    } else {
        weights.take(key)?
    };
    let ir_shape = Shape::new(&shape, DType::F32);
    let id = g.param(key, ir_shape);
    params.insert(key.to_string(), data);
    Ok(id)
}

fn synth_zero(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    len: usize,
) -> NodeId {
    let id = g.param(name, Shape::new(&[len], DType::F32));
    params.insert(name.to_string(), vec![0f32; len]);
    id
}

/// Multiply a graph node by a constant scalar via a broadcast `[1]` param
/// (mirrors rlx-flow `EmbedScaleStage`). Used for the Granite embedding /
/// residual / logit multipliers. `name` must be unique per scale site.
fn scale_by(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    name: &str,
    scale: f32,
) -> NodeId {
    let sid = if let Some(id) = g.param_id(name) {
        id
    } else {
        let id = g.param(name, Shape::new(&[1], DType::F32));
        params.insert(name.to_string(), vec![scale]);
        id
    };
    g.mul(x, sid)
}

/// Granite attention score scale (`attention_multiplier`) → the `score_scale`
/// argument of `Op::Attention`. `None` for non-Granite (op uses its default
/// `1/sqrt(head_dim)`).
fn attn_causal(
    g: &mut Graph,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    nh: usize,
    dh: usize,
    score_scale: Option<f32>,
    shape: Shape,
) -> NodeId {
    g.attention_kind_opts(q, k, v, nh, dh, MaskKind::Causal, shape, score_scale, None)
}

/// Dequant a single embed row from packed GGUF bytes (Q4K / Q6K).
pub(crate) fn gather_embed_row(
    packed_bytes: &[u8],
    scheme: rlx_ir::quant::QuantScheme,
    hidden: usize,
    token_id: usize,
    out: &mut [f32],
) -> Result<()> {
    use rlx_ir::quant::QuantScheme;
    debug_assert_eq!(out.len(), hidden);
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    if block_elems == 0 || !hidden.is_multiple_of(block_elems) {
        bail!(
            "gather_embed_row: scheme {scheme:?} block_elems={block_elems} doesn't divide hidden={hidden}"
        );
    }
    let blocks_per_row = hidden / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    let off = token_id * row_bytes;
    if off + row_bytes > packed_bytes.len() {
        bail!(
            "gather_embed_row: row offset {off}+{row_bytes} past packed bytes len {}",
            packed_bytes.len()
        );
    }
    let row = &packed_bytes[off..off + row_bytes];
    let dequant = match scheme {
        QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(row, hidden)?,
        QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k(row, hidden)?,
        _ => bail!("gather_embed_row: unsupported scheme {scheme:?}"),
    };
    out.copy_from_slice(&dequant);
    Ok(())
}

/// Host-gather prompt-token embedding rows for lazy packed prefill/decode.
pub(crate) fn gather_embed_rows(
    packed_bytes: &[u8],
    scheme: rlx_ir::quant::QuantScheme,
    hidden: usize,
    token_ids: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let n = token_ids.len();
    anyhow::ensure!(
        out.len() == n * hidden,
        "gather_embed_rows: out len {} != {} tokens × hidden {}",
        out.len(),
        n,
        hidden
    );
    for (i, &tok) in token_ids.iter().enumerate() {
        gather_embed_row(
            packed_bytes,
            scheme,
            hidden,
            tok as usize,
            &mut out[i * hidden..(i + 1) * hidden],
        )?;
    }
    Ok(())
}

/// loader exposes K-quant bytes for `key`, else as a transposed F32 param
/// (plain `g.mm`). Shared by the packed prefill and decode builders.
fn proj_available(
    packed: &HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &dyn WeightLoader,
    key: &str,
) -> bool {
    packed.contains_key(key) || weights.tensor_bytes_borrowed(key).is_some()
}

fn load_self_attn_qkv(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    cfg: &Llama32Config,
    lp: &str,
    batch: usize,
    seq: usize,
    normed_in: NodeId,
    f: DType,
) -> Result<(NodeId, NodeId, NodeId)> {
    let fused_key = format!("{lp}.self_attn.qkv.weight");
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    if cfg.is_phi_arch() || proj_available(packed, &*weights, &fused_key) {
        let (w, s, _) = load_proj(g, params, packed, weights, &fused_key)?;
        let total = q_dim + kv_dim + kv_dim;
        let combined = emit_proj(g, normed_in, w, s, Shape::new(&[batch, seq, total], f));
        let q = g.narrow_(combined, 2, 0, q_dim);
        let k = g.narrow_(combined, 2, q_dim, kv_dim);
        let v = g.narrow_(combined, 2, q_dim + kv_dim, kv_dim);
        Ok((q, k, v))
    } else {
        let (q_w, q_s, _) = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        let q = emit_proj(g, normed_in, q_w, q_s, Shape::new(&[batch, seq, q_dim], f));
        let (k_w, k_s, _) = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{lp}.self_attn.k_proj.weight"),
        )?;
        let k = emit_proj(g, normed_in, k_w, k_s, Shape::new(&[batch, seq, kv_dim], f));
        let (v_w, v_s, _) = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{lp}.self_attn.v_proj.weight"),
        )?;
        let v = emit_proj(g, normed_in, v_w, v_s, Shape::new(&[batch, seq, kv_dim], f));
        Ok((q, k, v))
    }
}

fn load_swiglu_ffn(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    cfg: &Llama32Config,
    lp: &str,
    batch: usize,
    seq: usize,
    normed_post: NodeId,
    f: DType,
) -> Result<(NodeId, NodeId)> {
    let inter = cfg.intermediate_size;
    let gate_up_key = format!("{lp}.mlp.gate_up.weight");
    if cfg.is_phi_arch() || proj_available(packed, &*weights, &gate_up_key) {
        let (gu_w, gu_s, _) = load_proj(g, params, packed, weights, &gate_up_key)?;
        let combined = emit_proj(
            g,
            normed_post,
            gu_w,
            gu_s,
            Shape::new(&[batch, seq, inter * 2], f),
        );
        let gate = g.narrow_(combined, 2, 0, inter);
        let up = g.narrow_(combined, 2, inter, inter);
        Ok((gate, up))
    } else {
        let (gate_w, gate_s, _) = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{lp}.mlp.gate_proj.weight"),
        )?;
        let gate = emit_proj(
            g,
            normed_post,
            gate_w,
            gate_s,
            Shape::new(&[batch, seq, inter], f),
        );
        let (up_w, up_s, _) = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{lp}.mlp.up_proj.weight"),
        )?;
        let up = emit_proj(
            g,
            normed_post,
            up_w,
            up_s,
            Shape::new(&[batch, seq, inter], f),
        );
        Ok((gate, up))
    }
}

fn load_proj(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<(NodeId, Option<rlx_ir::quant::QuantScheme>, Vec<usize>)> {
    if let Some(id) = g.param_id(key) {
        if let Some((scheme, shape)) = packed.get(key) {
            return Ok((id, Some(*scheme), shape.clone()));
        }
        return Ok((id, None, Vec::new()));
    }
    // Zero-copy packed path: read (scheme, shape) non-destructively and reserve
    // the U8 param slot by byte length, WITHOUT materializing the quantized
    // bytes. The bytes are uploaded straight from the loader's mmap after
    // compile (see `upload_packed_borrowed`), so the model isn't duplicated in
    // RSS. Loaders without a zero-copy path (`packed_meta` → None) fall through
    // to the F32 dequant branch.
    if let Some((scheme, shape)) = weights.packed_meta(key) {
        let nbytes = weights
            .tensor_bytes_borrowed(key)
            .ok_or_else(|| anyhow!("packed weight {key}: metadata present but bytes unavailable"))?
            .len();
        let id = g.param(key, Shape::new(&[nbytes], DType::U8));
        packed.insert(key.to_string(), (scheme, shape.clone()));
        Ok((id, Some(scheme), shape))
    } else {
        let nid = load_p(g, params, weights, key, true)?;
        Ok((nid, None, Vec::new()))
    }
}

/// Emit a (possibly quantized) projection: `Op::DequantMatMul` when a scheme
/// is present, else a plain matmul.
fn emit_proj(
    g: &mut Graph,
    input: NodeId,
    w: NodeId,
    scheme: Option<rlx_ir::quant::QuantScheme>,
    out_shape: Shape,
) -> NodeId {
    match scheme {
        Some(s) => g.add_node(Op::DequantMatMul { scheme: s }, vec![input, w], out_shape),
        None => g.mm(input, w),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Per-arch block deltas (OLMo-2 / Nemotron / Cohere / GLM-4 / ChatGLM).
//
// Each dense arch reuses the Llama qkv-proj + RoPE + attention + o_proj core;
// only the *normalization placement/kind*, *FFN shape*, and *residual wiring*
// differ. The packed prefill and decode builders share these helpers so the
// arch topology is defined once. `DenseArch::Llama` reproduces the stock
// Llama/Granite/Phi block byte-for-byte.
// ─────────────────────────────────────────────────────────────────────────

/// Fetch (or create once) a zero bias node of length `len` under `name`.
fn zero_beta_named(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    len: usize,
) -> NodeId {
    if let Some(id) = g.param_id(name) {
        return id;
    }
    synth_zero(g, params, name, len)
}

/// Arch-aware per-layer normalization over the last (hidden) axis: RMSNorm for
/// Llama / OLMo-2 / GLM, mean-subtracting LayerNorm for Nemotron (`bias_key`)
/// and Cohere (no bias → zero beta).
#[allow(clippy::too_many_arguments)]
fn emit_norm(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    cfg: &Llama32Config,
    weight_key: &str,
    bias_key: Option<&str>,
    x: NodeId,
    eps: f32,
    zero_beta: NodeId,
) -> Result<NodeId> {
    let w = load_p(g, params, weights, weight_key, false)?;
    match cfg.norm_kind() {
        NormKind::Rms => Ok(g.rms_norm(x, w, zero_beta, eps)),
        NormKind::LayerNorm => {
            let beta = match bias_key {
                Some(bk) => load_p(g, params, weights, bk, false)?,
                None => zero_beta,
            };
            Ok(g.ln(x, w, beta, eps))
        }
    }
}

/// OLMo-2 applies an RMSNorm over the FULL Q/K projection (`[n_heads·head_dim]`)
/// between the q/k projections and RoPE. No-op for every other arch.
fn emit_qk_norm(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    cfg: &Llama32Config,
    weight_idx: usize,
    q: NodeId,
    k: NodeId,
    eps: f32,
) -> Result<(NodeId, NodeId)> {
    if cfg.dense_arch() != DenseArch::Olmo2 {
        return Ok((q, k));
    }
    let zb_q = zero_beta_named(g, params, "olmo2.zero_beta.q", cfg.q_proj_dim());
    let zb_kv = zero_beta_named(g, params, "olmo2.zero_beta.kv", cfg.kv_proj_dim());
    let qn = load_p(
        g,
        params,
        weights,
        &format!("blk.{weight_idx}.attn_q_norm.weight"),
        false,
    )?;
    let kn = load_p(
        g,
        params,
        weights,
        &format!("blk.{weight_idx}.attn_k_norm.weight"),
        false,
    )?;
    Ok((g.rms_norm(q, qn, zb_q, eps), g.rms_norm(k, kn, zb_kv, eps)))
}

/// Arch-aware FFN (through `down_proj`): SwiGLU for Llama/OLMo-2/Cohere (split
/// gate/up), fused gate∥up SwiGLU for GLM, gate-less squared-ReLU for Nemotron.
#[allow(clippy::too_many_arguments)]
fn emit_arch_ffn(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    cfg: &Llama32Config,
    lp: &str,
    batch: usize,
    seq: usize,
    ffn_in: NodeId,
    f: DType,
) -> Result<NodeId> {
    let inter = cfg.intermediate_size;
    let hh = cfg.hidden_size;
    let act = match cfg.dense_arch() {
        DenseArch::Nemotron => {
            // Gate-less squared-ReLU: down(relu(up(x))²).
            let (up_w, up_s, _) = load_proj(
                g,
                params,
                packed,
                weights,
                &format!("{lp}.mlp.up_proj.weight"),
            )?;
            let up = emit_proj(g, ffn_in, up_w, up_s, Shape::new(&[batch, seq, inter], f));
            let r = g.relu(up);
            g.mul(r, r)
        }
        DenseArch::Glm4 | DenseArch::ChatGlm => {
            // GLM fuses gate∥up into a single `ffn_up` of width 2·inter; the
            // first half is the gate (SiLU), the second is the up projection
            // (matches llama.cpp `LLM_FFN_SWIGLU`'s split order).
            let (gu_w, gu_s, _) = load_proj(
                g,
                params,
                packed,
                weights,
                &format!("{lp}.mlp.up_proj.weight"),
            )?;
            let combined = emit_proj(
                g,
                ffn_in,
                gu_w,
                gu_s,
                Shape::new(&[batch, seq, inter * 2], f),
            );
            let gate = g.narrow_(combined, 2, 0, inter);
            let up = g.narrow_(combined, 2, inter, inter);
            let ga = g.silu(gate);
            g.mul(ga, up)
        }
        _ => {
            let (gate, up) =
                load_swiglu_ffn(g, params, packed, weights, cfg, lp, batch, seq, ffn_in, f)?;
            let ga = g.silu(gate);
            g.mul(ga, up)
        }
    };
    let (down_w, down_s, _) = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.mlp.down_proj.weight"),
    )?;
    Ok(emit_proj(
        g,
        act,
        down_w,
        down_s,
        Shape::new(&[batch, seq, hh], f),
    ))
}

/// Emit the pre-attention normalization. Returns `(attn_in, ffn_parallel)`:
/// `attn_in` feeds the q/k/v projections; `ffn_parallel` is `Some` only for
/// Cohere's parallel residual (the same normed input also feeds the MLP).
#[allow(clippy::too_many_arguments)]
fn emit_input_stage(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    cfg: &Llama32Config,
    lp: &str,
    weight_idx: usize,
    h_id: NodeId,
    eps: f32,
    zero_beta_hidden: NodeId,
) -> Result<(NodeId, Option<NodeId>)> {
    match cfg.dense_arch() {
        // OLMo-2 has no pre-attention norm — attention reads the residual stream.
        DenseArch::Olmo2 => Ok((h_id, None)),
        DenseArch::Cohere => {
            let n = emit_norm(
                g,
                params,
                weights,
                cfg,
                &format!("{lp}.input_layernorm.weight"),
                None,
                h_id,
                eps,
                zero_beta_hidden,
            )?;
            Ok((n, Some(n)))
        }
        _ => {
            let bias = (cfg.dense_arch() == DenseArch::Nemotron)
                .then(|| format!("blk.{weight_idx}.attn_norm.bias"));
            let n = emit_norm(
                g,
                params,
                weights,
                cfg,
                &format!("{lp}.input_layernorm.weight"),
                bias.as_deref(),
                h_id,
                eps,
                zero_beta_hidden,
            )?;
            Ok((n, None))
        }
    }
}

/// Emit the post-attention norm + FFN + residual wiring for one block, returning
/// the new residual-stream value. `attn_out` is the raw `o_proj` output (before
/// any residual scaling); `ffn_parallel` carries Cohere's shared normed input.
#[allow(clippy::too_many_arguments)]
fn emit_output_stage(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    cfg: &Llama32Config,
    lp: &str,
    weight_idx: usize,
    batch: usize,
    seq: usize,
    h_id: NodeId,
    attn_out: NodeId,
    ffn_parallel: Option<NodeId>,
    eps: f32,
    zero_beta_hidden: NodeId,
    f: DType,
) -> Result<NodeId> {
    match cfg.dense_arch() {
        DenseArch::Cohere => {
            // Parallel residual: h = x + attn(ln(x)) + mlp(ln(x)).
            let ffn_in = ffn_parallel
                .ok_or_else(|| anyhow!("cohere parallel residual missing shared norm"))?;
            let ffn_out =
                emit_arch_ffn(g, params, packed, weights, cfg, lp, batch, seq, ffn_in, f)?;
            let s1 = g.add(h_id, attn_out);
            Ok(g.add(s1, ffn_out))
        }
        DenseArch::Olmo2 => {
            // post-attn RMSNorm on the attention output, then residual; NO
            // pre-FFN norm; post-FFN RMSNorm before the FFN residual add.
            let post_attn_n = emit_norm(
                g,
                params,
                weights,
                cfg,
                &format!("blk.{weight_idx}.post_attention_norm.weight"),
                None,
                attn_out,
                eps,
                zero_beta_hidden,
            )?;
            let post_attn = g.add(h_id, post_attn_n);
            let ffn_raw = emit_arch_ffn(
                g, params, packed, weights, cfg, lp, batch, seq, post_attn, f,
            )?;
            let ffn_n = emit_norm(
                g,
                params,
                weights,
                cfg,
                &format!("blk.{weight_idx}.post_ffw_norm.weight"),
                None,
                ffn_raw,
                eps,
                zero_beta_hidden,
            )?;
            Ok(g.add(post_attn, ffn_n))
        }
        DenseArch::Glm4 => {
            // 4 RMSNorms: (pre-attn already applied) → post-attn → pre-ffn → post-ffn.
            let post_attn_n = emit_norm(
                g,
                params,
                weights,
                cfg,
                &format!("blk.{weight_idx}.attn_post_norm.weight"),
                None,
                attn_out,
                eps,
                zero_beta_hidden,
            )?;
            let ffn_inp = g.add(h_id, post_attn_n);
            let normed = emit_norm(
                g,
                params,
                weights,
                cfg,
                &format!("{lp}.post_attention_layernorm.weight"),
                None,
                ffn_inp,
                eps,
                zero_beta_hidden,
            )?;
            let ffn_raw =
                emit_arch_ffn(g, params, packed, weights, cfg, lp, batch, seq, normed, f)?;
            let ffn_n = emit_norm(
                g,
                params,
                weights,
                cfg,
                &format!("blk.{weight_idx}.ffn_post_norm.weight"),
                None,
                ffn_raw,
                eps,
                zero_beta_hidden,
            )?;
            Ok(g.add(ffn_inp, ffn_n))
        }
        DenseArch::Nemotron => {
            let post_attn = g.add(h_id, attn_out);
            let bias = format!("blk.{weight_idx}.ffn_norm.bias");
            let normed = emit_norm(
                g,
                params,
                weights,
                cfg,
                &format!("{lp}.post_attention_layernorm.weight"),
                Some(&bias),
                post_attn,
                eps,
                zero_beta_hidden,
            )?;
            let ffn_out =
                emit_arch_ffn(g, params, packed, weights, cfg, lp, batch, seq, normed, f)?;
            Ok(g.add(post_attn, ffn_out))
        }
        DenseArch::Llama | DenseArch::ChatGlm => {
            // Stock Llama block (+ Granite residual multipliers). ChatGLM reuses
            // it (pre-norm + pre-ffn norm); its fused gate∥up MLP is handled in
            // `emit_arch_ffn`.
            let attn_out = match cfg.residual_scale {
                Some(rs) => scale_by(
                    g,
                    params,
                    attn_out,
                    &format!("granite.res_scale.attn.{weight_idx}"),
                    rs,
                ),
                None => attn_out,
            };
            let post_attn = g.add(h_id, attn_out);
            let normed_post = emit_norm(
                g,
                params,
                weights,
                cfg,
                &format!("{lp}.post_attention_layernorm.weight"),
                None,
                post_attn,
                eps,
                zero_beta_hidden,
            )?;
            let ffn_out = emit_arch_ffn(
                g,
                params,
                packed,
                weights,
                cfg,
                lp,
                batch,
                seq,
                normed_post,
                f,
            )?;
            let ffn_out = match cfg.residual_scale {
                Some(rs) => scale_by(
                    g,
                    params,
                    ffn_out,
                    &format!("granite.res_scale.ffn.{weight_idx}"),
                    rs,
                ),
                None => ffn_out,
            };
            Ok(g.add(post_attn, ffn_out))
        }
    }
}

/// Arch-aware final norm (before the LM head) — same kinds as [`emit_norm`],
/// with Nemotron's `output_norm.bias`.
fn emit_final_norm(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    cfg: &Llama32Config,
    h_id: NodeId,
    eps: f32,
    zero_beta_hidden: NodeId,
) -> Result<NodeId> {
    let bias = (cfg.dense_arch() == DenseArch::Nemotron).then_some("output_norm.bias");
    emit_norm(
        g,
        params,
        weights,
        cfg,
        "model.norm.weight",
        bias,
        h_id,
        eps,
        zero_beta_hidden,
    )
}

/// Apply the arch's final logit scaling (Cohere multiplies, Granite divides).
fn apply_logit_scale(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &Llama32Config,
    logits: NodeId,
) -> NodeId {
    match cfg.final_logit_multiplier() {
        Some(m) => scale_by(g, params, logits, "arch.logit_scale", m),
        None => logits,
    }
}

/// Load the input-embedding table for a packed builder.
///
/// Mirrors rlx-gemma's lazy-embed pattern: when the GGUF stores
/// `model.embed_tokens.weight` as K-quant bytes AND the checkpoint ties the LM
/// head to it, the table stays PACKED. The input embedding is then gathered
/// host-side at run time (see `generator.rs::dequant_embed_row`) and fed as the
/// `input_embeddings` graph input, while the tied LM head reuses the same packed
/// bytes via `Op::DequantMatMul`. This eliminates both the ~2 GB F32 embed table
/// (vocab×hidden) and its ~2 GB transposed copy per session.
///
/// Returns `(h_in, tied_head)`:
/// - `h_in` feeds the first transformer block: either the `input_embeddings`
///   input (lazy path) or `gather(embed_f32, input_ids)` (F32 path).
/// - `tied_head` is `Some((embed_node, scheme))` only on the lazy path AND when
///   `want_head` — the U8 embed param node + its quant scheme for the tied
///   head's `Op::DequantMatMul`. `None` otherwise; the caller's tied path then
///   builds the transposed F32 copy (unchanged).
///
/// Gated on `tie_word_embeddings`: non-tied checkpoints keep the legacy F32
/// gather + separate `lm_head.weight`, untouched.
#[allow(clippy::too_many_arguments)]
fn load_packed_embed(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    cfg: &Llama32Config,
    batch: usize,
    seq: usize,
    want_head: bool,
    embed_host: &mut Option<(Vec<u8>, rlx_ir::quant::QuantScheme)>,
) -> Result<(NodeId, Option<(NodeId, rlx_ir::quant::QuantScheme)>)> {
    let key = "model.embed_tokens.weight";
    if let Some((scheme, shape)) = weights.packed_meta(key) {
        // The embed stays packed. Keep ONE owned copy of its bytes for the
        // host-side lazy gather (`embed_host`): the prefill loader is dropped
        // after build, so this single tensor can't be borrowed at runtime. The
        // tied LM-head param (if any) carries metadata only and is uploaded
        // zero-copy from the loader at attach time, like every other matmul.
        let bytes = weights
            .tensor_bytes_borrowed(key)
            .ok_or_else(|| anyhow!("packed embed {key}: bytes unavailable"))?
            .to_vec();
        let nbytes = bytes.len();
        *embed_host = Some((bytes, scheme));
        let h_in = g.input(
            "input_embeddings",
            Shape::new(&[batch, seq, cfg.hidden_size], DType::F32),
        );
        let tied = if want_head && cfg.tie_word_embeddings {
            let id = g.param(key, Shape::new(&[nbytes], DType::U8));
            packed.insert(key.to_string(), (scheme, shape));
            Some((id, scheme))
        } else {
            None
        };
        Ok((h_in, tied))
    } else {
        let embed_w = load_p(g, params, weights, key, false)?;
        let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
        Ok((g.gather_(embed_w, input_ids, 0), None))
    }
}

/// Packed-weights prefill graph — K-quant matmuls stay in the arena via
/// `Op::DequantMatMul` (mirrors [`rlx_qwen3::build_qwen3_graph_sized_packed`]).
#[allow(clippy::too_many_arguments)]
pub fn build_llama32_graph_sized_packed(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    with_kv_outputs: bool,
    packed: &mut HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    embed_host: &mut Option<(Vec<u8>, rlx_ir::quant::QuantScheme)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;

    let mut g = Graph::new("llama32_packed");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim();
    let group = cfg.kv_group_size();
    let eps = cfg.rms_norm_eps as f32;

    let zero_beta_hidden = synth_zero(&mut g, &mut params, "llama32.zero_beta.hidden", h);

    let rope_factors = take_rope_freqs(weights);
    let inv_freq = resolve_inv_freq(cfg, rope_factors.as_deref());
    let rope_rows = seq;
    let (cos_data, sin_data) = build_rope_tables(&inv_freq, rope_rows);
    let half = inv_freq.len();
    let cos_id = g.param("rope.cos", Shape::new(&[rope_rows, half], f));
    params.insert("rope.cos".into(), cos_data);
    let sin_id = g.param("rope.sin", Shape::new(&[rope_rows, half], f));
    params.insert("rope.sin".into(), sin_data);

    // Keep the embed table PACKED (Q-quant) for tied checkpoints: input
    // embedding gathered host-side (`input_embeddings`), tied LM head via
    // `Op::DequantMatMul` on the packed bytes — no F32 embed/transpose copies.
    // Non-tied / uncompressed embeds fall back to the F32 gather (unchanged).
    let (mut h_id, tied_embed_head) = load_packed_embed(
        &mut g,
        &mut params,
        packed,
        weights,
        cfg,
        batch,
        seq,
        with_lm_head,
        embed_host,
    )?;
    // Granite: multiply input embeddings by `embedding_multiplier`.
    if let Some(es) = cfg.embedding_scale {
        h_id = scale_by(&mut g, &mut params, h_id, "granite.embed_scale", es);
    }
    let score_scale = cfg.attn_score_scale();
    let last_token_idx = if last_logits_only {
        Some(g.input("last_token_idx", Shape::new(&[batch], DType::F32)))
    } else {
        None
    };
    let mut kv_outputs: Vec<(NodeId, NodeId)> = Vec::new();
    let physical = cfg.physical_layers();
    let kv_layers = cfg.kv_layers();

    for exec_idx in 0..kv_layers {
        let weight_idx = cfg.weight_layer_index(exec_idx);
        let lp = format!("model.layers.{weight_idx}");

        // Arch-aware pre-attention norm (OLMo-2 has none; Cohere shares it with
        // the MLP for the parallel residual).
        let (attn_in, ffn_parallel) = emit_input_stage(
            &mut g,
            &mut params,
            weights,
            cfg,
            &lp,
            weight_idx,
            h_id,
            eps,
            zero_beta_hidden,
        )?;

        let (q, k, v) = load_self_attn_qkv(
            &mut g,
            &mut params,
            packed,
            weights,
            cfg,
            &lp,
            batch,
            seq,
            attn_in,
            f,
        )?;
        // OLMo-2: RMSNorm the full Q/K projection before RoPE (no-op otherwise).
        let (q, k) = emit_qk_norm(&mut g, &mut params, weights, cfg, weight_idx, q, k, eps)?;

        // GGUF Llama → interleaved/GPT-J RoPE flavor (mirror the decode-packed
        // builder `build_llama32_decode_graph_sized_packed`). Plain `g.rope`
        // applies NeoX rotation, which corrupts packed-prefill KV for GGUF
        // checkpoints and makes Metal-decode diverge from the CPU F32 reference.
        // Cohere2 skips RoPE on its global (full-attention) layers (NoPE).
        let cohere2_nope = cfg
            .cohere2_nope_pattern()
            .is_some_and(|p| (weight_idx + 1) % p == 0);
        let (q_rope, k_rope) = if cohere2_nope {
            (q, k)
        } else {
            apply_qk_rope(&mut g, q, k, cos_id, sin_id, cfg)
        };
        if with_kv_outputs {
            kv_outputs.push((k_rope, v));
        }

        let k_rep = repeat_kv(&mut g, k_rope, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v, nkv, dh, group);

        let attn_shape = shape::attention_shape(g.shape(q_rope));
        let attn = attn_causal(
            &mut g,
            q_rope,
            k_rep,
            v_rep,
            nh,
            dh,
            score_scale,
            attn_shape,
        );

        let (o_w, o_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, o_w, o_s, Shape::new(&[batch, seq, h], f));

        // Arch-aware post-attention norm + FFN + residual wiring.
        h_id = emit_output_stage(
            &mut g,
            &mut params,
            packed,
            weights,
            cfg,
            &lp,
            weight_idx,
            batch,
            seq,
            h_id,
            attn_out,
            ffn_parallel,
            eps,
            zero_beta_hidden,
            f,
        )?;

        let loop_end = physical > 0 && (exec_idx + 1) % physical == 0;
        let last_exec = exec_idx + 1 == kv_layers;
        if loop_end && !cfg.skip_loop_final_norm && !last_exec {
            let mid_ln = load_p(&mut g, &mut params, weights, "model.norm.weight", false)?;
            h_id = g.rms_norm(h_id, mid_ln, zero_beta_hidden, eps);
        }
    }

    let hidden = emit_final_norm(
        &mut g,
        &mut params,
        weights,
        cfg,
        h_id,
        eps,
        zero_beta_hidden,
    )?;

    let out = if with_lm_head {
        let head_input = if last_logits_only {
            let idx = last_token_idx.expect("last_token_idx input");
            let idx_2d = g.reshape_(idx, vec![batch as i64, 1]);
            g.gather_(hidden, idx_2d, 1)
        } else {
            hidden
        };
        let (lm_head_w, lm_head_scheme) = if cfg.tie_word_embeddings {
            if let Some((embed_node, scheme)) = tied_embed_head {
                // Packed tied LM head: DequantMatMul against the Q-quant embed
                // bytes (no ~2 GB transposed F32 copy).
                (embed_node, Some(scheme))
            } else {
                // F32 tied LM head: transposed copy of the embed (unchanged).
                let embed = params
                    .get("model.embed_tokens.weight")
                    .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?;
                let vocab = cfg.vocab_size;
                let hidden_size = cfg.hidden_size;
                let mut transposed = vec![0f32; embed.len()];
                for v in 0..vocab {
                    for hi in 0..hidden_size {
                        transposed[hi * vocab + v] = embed[v * hidden_size + hi];
                    }
                }
                let name = "llama32.lm_head.tied_t";
                let id = g.param(name, Shape::new(&[hidden_size, vocab], DType::F32));
                params.insert(name.to_string(), transposed);
                (id, None)
            }
        } else {
            let (id, scheme, _) =
                load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
            (id, scheme)
        };
        let logits = emit_proj(
            &mut g,
            head_input,
            lm_head_w,
            lm_head_scheme,
            Shape::new(
                &[
                    batch,
                    if last_logits_only { 1 } else { seq },
                    cfg.vocab_size,
                ],
                f,
            ),
        );
        // Granite divides by `logits_scaling`; Cohere multiplies by `logit_scale`.
        apply_logit_scale(&mut g, &mut params, cfg, logits)
    } else {
        hidden
    };

    let mut outputs = Vec::new();
    if with_lm_head || !with_kv_outputs {
        outputs.push(out);
    } else if last_logits_only {
        // Host greedy prefill: last-token hidden + per-layer KV (skip vocab matmul).
        let idx = last_token_idx.expect("last_token_idx input");
        let idx_2d = g.reshape_(idx, vec![batch as i64, 1]);
        outputs.push(g.gather_(hidden, idx_2d, 1));
    }
    if with_kv_outputs {
        for (k, v) in kv_outputs {
            outputs.push(k);
            outputs.push(v);
        }
    }
    g.set_outputs(outputs);
    Ok((g, params))
}

/// Packed-weights *decode* graph — single-token KV-cache step that keeps
/// K-quant matmuls in the arena via `Op::DequantMatMul`. Structural mirror of
/// the F32 decode flow (`build_llama32_decode_flow`): `input_ids` of shape
/// `[batch, 1]`, per-layer `past_k_{i}`/`past_v_{i}` inputs, single-row decode
/// RoPE at position `past_seq`, `concat(past, new)` along the sequence axis,
/// `MaskKind::Causal` attention, SwiGLU MLP, and LM head.
///
/// Outputs match the F32 flow decode: `[logits, k0_full, v0_full, k1_full, …]`
/// where each `k*/v*` is the full `concat(past, new)` KV (length `past_seq+1`),
/// so the generator's existing `split_decode_outputs` + cache-replace logic is
/// reused verbatim.
///
/// GGUF Llama needs the interleaved/GPT-J RoPE flavor, so this uses
/// `g.rope_styled(.., cfg.rope_style)` (not the NeoX `g.rope` used by the
/// packed prefill builder).
///
/// `use_custom_mask` switches the graph from a per-position oneshot graph to a
/// reusable bucketed-decode graph (mirrors `build_llama32_decode_flow`'s
/// `use_custom_mask`):
/// - cos/sin become runtime **inputs** (`cos`/`sin`, shape `[1, half]`) instead
///   of baked params, so one compiled graph serves every position in a bucket.
/// - a `mask` input of shape `[batch, past_seq + 1]` drives `MaskKind::Custom`
///   attention (binary keep mask; see `bucket_decode_mask`), zeroing the
///   `past_k`/`past_v` padding rows. `past_seq` is the bucket's upper bound.
///
/// With `use_custom_mask = false` the graph bakes the cos/sin row for the exact
/// `past_seq` and uses `MaskKind::Causal` (the slow but always-correct fallback).
/// Like [`build_llama32_decode_graph_sized_packed`] but can omit the in-graph
/// vocab matmul (`with_lm_head = false` → post-norm hidden + KV outputs).
#[allow(clippy::too_many_arguments)]
pub fn build_llama32_decode_graph_sized_packed_ext(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
    with_lm_head: bool,
    packed: &mut HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;

    let mut g = Graph::new("llama32_decode_packed");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim();
    let group = cfg.kv_group_size();
    let eps = cfg.rms_norm_eps as f32;
    let kv_dim = cfg.kv_proj_dim();

    let zero_beta_hidden = synth_zero(&mut g, &mut params, "llama32.zero_beta.hidden", h);

    // Single-row decode RoPE at absolute position `past_seq`.
    // Bucketed decode feeds the cos/sin row as runtime inputs so one compiled
    // graph serves every position in the bucket; oneshot bakes the exact row.
    let rope_factors = take_rope_freqs(weights);
    let inv_freq = resolve_inv_freq(cfg, rope_factors.as_deref());
    let half = inv_freq.len();
    let (cos_id, sin_id) = if use_custom_mask {
        let cos_id = g.input("cos", Shape::new(&[1, half], f));
        let sin_id = g.input("sin", Shape::new(&[1, half], f));
        (cos_id, sin_id)
    } else {
        let (cos_row, sin_row) = rope_slice(&inv_freq, past_seq);
        let cos_id = g.param("decode.rope.cos", Shape::new(&[1, half], f));
        params.insert("decode.rope.cos".into(), cos_row);
        let sin_id = g.param("decode.rope.sin", Shape::new(&[1, half], f));
        params.insert("decode.rope.sin".into(), sin_row);
        (cos_id, sin_id)
    };

    // Keep the embed table PACKED for tied checkpoints (see `load_packed_embed`):
    // single-token input embedding gathered host-side (`input_embeddings`),
    // tied LM head via `Op::DequantMatMul`. Decode always emits the LM head.
    let mut embed_host = None;
    let (mut h_id, tied_embed_head) = load_packed_embed(
        &mut g,
        &mut params,
        packed,
        weights,
        cfg,
        batch,
        1,
        with_lm_head,
        &mut embed_host,
    )?;
    // Granite: multiply input embeddings by `embedding_multiplier`.
    if let Some(es) = cfg.embedding_scale {
        h_id = scale_by(&mut g, &mut params, h_id, "granite.embed_scale", es);
    }
    let score_scale = cfg.attn_score_scale();

    // Per-layer past K/V cache inputs (unrolled across loops).
    let kv_layers = cfg.kv_layers();
    let physical = cfg.physical_layers();
    let mut past_k_ids: Vec<NodeId> = Vec::with_capacity(kv_layers);
    let mut past_v_ids: Vec<NodeId> = Vec::with_capacity(kv_layers);
    for i in 0..kv_layers {
        past_k_ids.push(g.input(
            format!("past_k_{i}"),
            Shape::new(&[batch, past_seq, kv_dim], f),
        ));
        past_v_ids.push(g.input(
            format!("past_v_{i}"),
            Shape::new(&[batch, past_seq, kv_dim], f),
        ));
    }

    // Bucketed decode: binary keep mask over `concat(past_k, new_k)` positions.
    let mask_id = if use_custom_mask {
        Some(g.input("mask", Shape::new(&[batch, past_seq + 1], f)))
    } else {
        None
    };

    let mut kv_outputs: Vec<(NodeId, NodeId)> = Vec::new();

    for exec_idx in 0..kv_layers {
        let weight_idx = cfg.weight_layer_index(exec_idx);
        let lp = format!("model.layers.{weight_idx}");

        // Arch-aware pre-attention norm (OLMo-2 has none; Cohere shares it).
        let (attn_in, ffn_parallel) = emit_input_stage(
            &mut g,
            &mut params,
            weights,
            cfg,
            &lp,
            weight_idx,
            h_id,
            eps,
            zero_beta_hidden,
        )?;

        let (q, k, v) = load_self_attn_qkv(
            &mut g,
            &mut params,
            packed,
            weights,
            cfg,
            &lp,
            batch,
            1,
            attn_in,
            f,
        )?;
        // OLMo-2: RMSNorm the full Q/K projection before RoPE (no-op otherwise).
        let (q, k) = emit_qk_norm(&mut g, &mut params, weights, cfg, weight_idx, q, k, eps)?;

        // GGUF Llama → interleaved/GPT-J RoPE flavor. Cohere2 global layers = NoPE.
        let cohere2_nope = cfg
            .cohere2_nope_pattern()
            .is_some_and(|p| (weight_idx + 1) % p == 0);
        let (q_rope, k_rope) = if cohere2_nope {
            (q, k)
        } else {
            apply_qk_rope(&mut g, q, k, cos_id, sin_id, cfg)
        };

        // Append the new token to the cached KV, export the full buffers.
        let new_k = g.concat_(vec![past_k_ids[exec_idx], k_rope], 1);
        let new_v = g.concat_(vec![past_v_ids[exec_idx], v], 1);
        kv_outputs.push((new_k, new_v));

        let k_rep = repeat_kv(&mut g, new_k, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, new_v, nkv, dh, group);

        let attn_shape = shape::attention_shape(g.shape(q_rope));
        let attn = match mask_id {
            Some(m) => g.attention_opts(
                q_rope,
                k_rep,
                v_rep,
                m,
                nh,
                dh,
                attn_shape,
                score_scale,
                None,
            ),
            None => attn_causal(
                &mut g,
                q_rope,
                k_rep,
                v_rep,
                nh,
                dh,
                score_scale,
                attn_shape,
            ),
        };

        let (o_w, o_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, o_w, o_s, Shape::new(&[batch, 1, h], f));

        // Arch-aware post-attention norm + FFN + residual wiring.
        h_id = emit_output_stage(
            &mut g,
            &mut params,
            packed,
            weights,
            cfg,
            &lp,
            weight_idx,
            batch,
            1,
            h_id,
            attn_out,
            ffn_parallel,
            eps,
            zero_beta_hidden,
            f,
        )?;

        let loop_end = physical > 0 && (exec_idx + 1) % physical == 0;
        let last_exec = exec_idx + 1 == kv_layers;
        if loop_end && !cfg.skip_loop_final_norm && !last_exec {
            let mid_ln = load_p(&mut g, &mut params, weights, "model.norm.weight", false)?;
            h_id = g.rms_norm(h_id, mid_ln, zero_beta_hidden, eps);
        }
    }

    let hidden = emit_final_norm(
        &mut g,
        &mut params,
        weights,
        cfg,
        h_id,
        eps,
        zero_beta_hidden,
    )?;

    let out = if with_lm_head {
        // Decode is always last-position (seq == 1), so no gather is needed.
        let (lm_head_w, lm_head_scheme) = if cfg.tie_word_embeddings {
            if let Some((embed_node, scheme)) = tied_embed_head {
                // Packed tied LM head: DequantMatMul against the Q-quant embed bytes.
                (embed_node, Some(scheme))
            } else {
                // F32 tied LM head: transposed copy of the embed (unchanged).
                let embed = params
                    .get("model.embed_tokens.weight")
                    .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?;
                let vocab = cfg.vocab_size;
                let hidden_size = cfg.hidden_size;
                let mut transposed = vec![0f32; embed.len()];
                for v in 0..vocab {
                    for hi in 0..hidden_size {
                        transposed[hi * vocab + v] = embed[v * hidden_size + hi];
                    }
                }
                let name = "llama32.lm_head.tied_t";
                let id = g.param(name, Shape::new(&[hidden_size, vocab], DType::F32));
                params.insert(name.to_string(), transposed);
                (id, None)
            }
        } else {
            let (id, scheme, _) =
                load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
            (id, scheme)
        };
        let logits = emit_proj(
            &mut g,
            hidden,
            lm_head_w,
            lm_head_scheme,
            Shape::new(&[batch, 1, cfg.vocab_size], f),
        );
        // Granite divides by `logits_scaling`; Cohere multiplies by `logit_scale`.
        apply_logit_scale(&mut g, &mut params, cfg, logits)
    } else {
        hidden
    };

    let mut outputs = vec![out];
    for (k, v) in kv_outputs {
        outputs.push(k);
        outputs.push(v);
    }
    g.set_outputs(outputs);
    Ok((g, params))
}

#[allow(clippy::too_many_arguments)]
pub fn build_llama32_decode_graph_sized_packed(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
    packed: &mut HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_llama32_decode_graph_sized_packed_ext(
        cfg,
        weights,
        batch,
        past_seq,
        use_custom_mask,
        true,
        packed,
    )
}

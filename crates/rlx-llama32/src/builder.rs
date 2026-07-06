// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// LLaMA-3.2 graph builder — GQA + RoPE + SwiGLU, no QK-norm.

use crate::config::Llama32Config;
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
    packed: &HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &dyn WeightLoader,
    key: &str,
) -> bool {
    packed.contains_key(key) || weights.tensor_bytes_borrowed(key).is_some()
}

fn load_self_attn_qkv(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
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
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
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
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<(NodeId, Option<rlx_ir::quant::QuantScheme>, Vec<usize>)> {
    if let Some((bytes, scheme, shape)) = weights.take_packed(key)? {
        let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
        packed.insert(key.to_string(), (bytes, scheme, shape.clone()));
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
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    cfg: &Llama32Config,
    batch: usize,
    seq: usize,
    want_head: bool,
    embed_host: &mut Option<(Vec<u8>, rlx_ir::quant::QuantScheme)>,
) -> Result<(NodeId, Option<(NodeId, rlx_ir::quant::QuantScheme)>)> {
    let key = "model.embed_tokens.weight";
    if let Some((bytes, scheme, shape)) = weights.take_packed(key)? {
        *embed_host = Some((bytes.clone(), scheme));
        let h_in = g.input(
            "input_embeddings",
            Shape::new(&[batch, seq, cfg.hidden_size], DType::F32),
        );
        let tied = if want_head && cfg.tie_word_embeddings {
            let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
            packed.insert(key.to_string(), (bytes, scheme, shape));
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
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
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
    let last_token_idx = if last_logits_only {
        Some(g.input("last_token_idx", Shape::new(&[batch], DType::F32)))
    } else {
        None
    };
    let mut kv_outputs: Vec<(NodeId, NodeId)> = Vec::new();

    for layer_idx in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer_idx}");

        let in_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            false,
        )?;
        let normed_in = g.rms_norm(h_id, in_ln_g, zero_beta_hidden, eps);

        let (q, k, v) = load_self_attn_qkv(
            &mut g,
            &mut params,
            packed,
            weights,
            cfg,
            &lp,
            batch,
            seq,
            normed_in,
            f,
        )?;

        // GGUF Llama → interleaved/GPT-J RoPE flavor (mirror the decode-packed
        // builder `build_llama32_decode_graph_sized_packed`). Plain `g.rope`
        // applies NeoX rotation, which corrupts packed-prefill KV for GGUF
        // checkpoints and makes Metal-decode diverge from the CPU F32 reference.
        let (q_rope, k_rope) = apply_qk_rope(&mut g, q, k, cos_id, sin_id, cfg);
        if with_kv_outputs {
            kv_outputs.push((k_rope, v));
        }

        let k_rep = repeat_kv(&mut g, k_rope, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v, nkv, dh, group);

        let attn_shape = shape::attention_shape(g.shape(q_rope));
        let attn = g.attention_kind(q_rope, k_rep, v_rep, nh, dh, MaskKind::Causal, attn_shape);

        let (o_w, o_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, o_w, o_s, Shape::new(&[batch, seq, h], f));
        let post_attn = g.add(h_id, attn_out);

        let post_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            false,
        )?;
        let normed_post = g.rms_norm(post_attn, post_ln_g, zero_beta_hidden, eps);

        let (gate, up) = load_swiglu_ffn(
            &mut g,
            &mut params,
            packed,
            weights,
            cfg,
            &lp,
            batch,
            seq,
            normed_post,
            f,
        )?;
        let (down_w, down_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.down_proj.weight"),
        )?;
        let gate_act = g.silu(gate);
        let swiglu = g.mul(gate_act, up);
        let ffn_out = emit_proj(
            &mut g,
            swiglu,
            down_w,
            down_s,
            Shape::new(&[batch, seq, h], f),
        );
        h_id = g.add(post_attn, ffn_out);
    }

    let final_ln_g = load_p(&mut g, &mut params, weights, "model.norm.weight", false)?;
    let hidden = g.rms_norm(h_id, final_ln_g, zero_beta_hidden, eps);

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
        emit_proj(
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
        )
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
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
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

    // Per-layer past K/V cache inputs.
    let mut past_k_ids: Vec<NodeId> = Vec::with_capacity(cfg.num_hidden_layers);
    let mut past_v_ids: Vec<NodeId> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
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

    for layer_idx in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer_idx}");

        let in_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            false,
        )?;
        let normed_in = g.rms_norm(h_id, in_ln_g, zero_beta_hidden, eps);

        let (q, k, v) = load_self_attn_qkv(
            &mut g,
            &mut params,
            packed,
            weights,
            cfg,
            &lp,
            batch,
            1,
            normed_in,
            f,
        )?;

        // GGUF Llama → interleaved/GPT-J RoPE flavor.
        let (q_rope, k_rope) = apply_qk_rope(&mut g, q, k, cos_id, sin_id, cfg);

        // Append the new token to the cached KV, export the full buffers.
        let new_k = g.concat_(vec![past_k_ids[layer_idx], k_rope], 1);
        let new_v = g.concat_(vec![past_v_ids[layer_idx], v], 1);
        kv_outputs.push((new_k, new_v));

        let k_rep = repeat_kv(&mut g, new_k, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, new_v, nkv, dh, group);

        let attn_shape = shape::attention_shape(g.shape(q_rope));
        let attn = match mask_id {
            Some(m) => g.attention(q_rope, k_rep, v_rep, m, nh, dh, attn_shape),
            None => g.attention_kind(q_rope, k_rep, v_rep, nh, dh, MaskKind::Causal, attn_shape),
        };

        let (o_w, o_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, o_w, o_s, Shape::new(&[batch, 1, h], f));
        let post_attn = g.add(h_id, attn_out);

        let post_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            false,
        )?;
        let normed_post = g.rms_norm(post_attn, post_ln_g, zero_beta_hidden, eps);

        let (gate, up) = load_swiglu_ffn(
            &mut g,
            &mut params,
            packed,
            weights,
            cfg,
            &lp,
            batch,
            1,
            normed_post,
            f,
        )?;
        let (down_w, down_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.down_proj.weight"),
        )?;
        let gate_act = g.silu(gate);
        let swiglu = g.mul(gate_act, up);
        let ffn_out = emit_proj(
            &mut g,
            swiglu,
            down_w,
            down_s,
            Shape::new(&[batch, 1, h], f),
        );
        h_id = g.add(post_attn, ffn_out);
    }

    let final_ln_g = load_p(&mut g, &mut params, weights, "model.norm.weight", false)?;
    let hidden = g.rms_norm(h_id, final_ln_g, zero_beta_hidden, eps);

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
        emit_proj(
            &mut g,
            hidden,
            lm_head_w,
            lm_head_scheme,
            Shape::new(&[batch, 1, cfg.vocab_size], f),
        )
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
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
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

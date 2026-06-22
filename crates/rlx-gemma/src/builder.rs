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

//! Gemma graph builders — thin wrappers over [`crate::flow::GemmaFlow`].

use crate::config::GemmaConfig;
use anyhow::{Result, anyhow};
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::Graph;
use rlx_ir::hir::HirModule;
use rlx_ir::infer::GraphExt;
use rlx_ir::quant::QuantScheme;
use std::collections::HashMap;

type F32WeightMap = HashMap<String, Vec<f32>>;
type PackedWeightMap = HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>;
type PackedDrainResult = (F32WeightMap, PackedWeightMap);

pub fn build_gemma_graph_sized(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::GemmaPrefillOpts {
        batch,
        seq,
        dynamic_seq: false,
        prefill_hidden: false,
        media_attn_bias: false,
        with_lm_head,
        with_kv_outputs,
        last_logits_only: false,
        profile: None,
    };
    rlx_core::flow_util::graph_from_built(crate::flow::build_gemma_prefill_built(
        cfg, weights, &opts,
    )?)
}

pub fn build_gemma_graph_sized_last_logits(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::GemmaPrefillOpts {
        batch,
        seq,
        dynamic_seq: false,
        prefill_hidden: false,
        media_attn_bias: false,
        with_lm_head: true,
        with_kv_outputs,
        last_logits_only: true,
        profile: None,
    };
    rlx_core::flow_util::graph_from_built(crate::flow::build_gemma_prefill_built(
        cfg, weights, &opts,
    )?)
}

pub fn build_gemma_prefill_hir_dynamic_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    max_seq: usize,
    with_kv_outputs: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_gemma_prefill_hir_dynamic_ext_inner(cfg, weights, batch, max_seq, with_kv_outputs, false)
}

/// Dynamic-seq prefill from fused `inputs_embeds` (multimodal).
pub fn build_gemma_prefill_hidden_hir_dynamic_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    max_seq: usize,
    with_kv_outputs: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_gemma_prefill_hir_dynamic_ext_inner(cfg, weights, batch, max_seq, with_kv_outputs, true)
}

fn build_gemma_prefill_hir_dynamic_ext_inner(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    max_seq: usize,
    with_kv_outputs: bool,
    prefill_hidden: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    if batch != 1 {
        return Err(anyhow!("gemma: dynamic_seq prefill requires batch=1"));
    }
    let opts = crate::flow::GemmaPrefillOpts {
        batch,
        seq: max_seq,
        dynamic_seq: true,
        prefill_hidden,
        media_attn_bias: prefill_hidden && cfg.use_bidirectional_vision(),
        with_lm_head: true,
        with_kv_outputs,
        last_logits_only: true,
        profile: None,
    };
    crate::flow::build_gemma_prefill_flow(cfg, weights, &opts)
}

pub fn build_gemma_graph_sized_last_logits_hidden(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::GemmaPrefillOpts {
        batch,
        seq,
        dynamic_seq: false,
        prefill_hidden: true,
        media_attn_bias: cfg.use_bidirectional_vision(),
        with_lm_head: true,
        with_kv_outputs,
        last_logits_only: true,
        profile: None,
    };
    rlx_core::flow_util::graph_from_built(crate::flow::build_gemma_prefill_built(
        cfg, weights, &opts,
    )?)
}

pub fn build_gemma_decode_graph_sized(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_gemma_decode_graph_sized_ext(cfg, weights, batch, past_seq, false)
}

pub fn build_gemma_decode_graph_sized_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::GemmaDecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask,
        profile: None,
        aux_hidden_layer_ids: Vec::new(),
    };
    crate::flow::build_gemma_decode_graph(cfg, weights, &opts)
}

pub fn build_gemma_decode_hir_sized(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_gemma_decode_hir_sized_ext(cfg, weights, batch, past_seq, false)
}

pub fn build_gemma_decode_hir_sized_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let opts = crate::flow::GemmaDecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask,
        profile: None,
        aux_hidden_layer_ids: Vec::new(),
    };
    crate::flow::build_gemma_decode_flow(cfg, weights, &opts)
}

pub fn build_gemma_decode_hir_dynamic_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    max_past_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let opts = crate::flow::GemmaDecodeOpts {
        batch,
        past_seq: max_past_seq,
        dynamic_past: true,
        use_custom_mask: false,
        profile: None,
        aux_hidden_layer_ids: Vec::new(),
    };
    crate::flow::build_gemma_decode_flow(cfg, weights, &opts)
}

/// Packed K-quant prefill graph. Mirrors `build_gemma_graph_sized`
/// but emits `Op::DequantMatMul` against the GGUF-quantized weight
/// buffers (kept in their on-disk layout in `packed`) for every
/// projection that has a quantization scheme. Tensors that come back
/// from the loader as F32 (norms, embed) still go through `MatMul`.
/// Memory cost: O(quantized weight bytes) per layer in the arena
/// instead of O(F32 bytes).
///
/// Supports the full Gemma 1/2/3/4 surface area exposed by
/// `GemmaConfig`: per-layer head_dim / num_kv_heads / n_rot,
/// `attention_k_eq_v`, split RoPE per layer kind, alternating
/// sliding-window vs full-attention masks, attention soft-cap, and
/// final logit soft-cap. Layer-norm names follow the version-aware
/// pattern (`post_attention_layernorm` for V1, pre+post feedforward
/// for V2+).
///
/// `last_token_from_input`: when true, adds a `last_token_idx` graph input
/// and gathers one hidden row before `lm_head` (1× vocab logits). Required
/// for autoregressive packed generation — do not use fixed `seq-1` narrow.
fn gather_last_token_packed(
    g: &mut Graph,
    hidden: rlx_ir::NodeId,
    batch: usize,
    last_token_idx: rlx_ir::NodeId,
) -> rlx_ir::NodeId {
    let idx_2d = g.reshape_(last_token_idx, vec![batch as i64, 1]);
    g.gather_(hidden, idx_2d, 1)
}

fn slice_rope_table(table: &[f32], half: usize, rows: usize) -> Vec<f32> {
    let need = rows * half;
    if table.len() >= need {
        table[..need].to_vec()
    } else {
        table.to_vec()
    }
}

/// Drain GGUF weights + RoPE tables for packed session init (no layer graph).
pub fn drain_gemma_packed_weights(
    cfg: &GemmaConfig,
    loader: &mut rlx_core::weight_loader::GgufLoader,
) -> Result<PackedDrainResult> {
    drain_gemma_packed_weights_ext(cfg, loader, None)
}

/// Same as [`drain_gemma_packed_weights`] but with an optional cap on the
/// number of RoPE-table rows materialised at LOAD. Default (`None`) preserves
/// the legacy behaviour (`cfg.max_position_embeddings` rows, up to ~1 GB for
/// Gemma 4 12B); the session caller passes `Some(max_seq + 16)` so we only
/// allocate the rows the prefill bucket can actually read (a ≥ 99% saving for
/// typical `max_seq=128`). Decode uses single-row `rope_slice` so the table
/// size doesn't gate decode either.
pub fn drain_gemma_packed_weights_ext(
    cfg: &GemmaConfig,
    loader: &mut rlx_core::weight_loader::GgufLoader,
    max_rope_rows: Option<usize>,
) -> Result<PackedDrainResult> {
    use crate::rope::{build_rope_tables, resolve_global_inv_freq, resolve_inv_freq};
    use rlx_core::weight_map::{WeightDrainPolicy, WeightMap};

    let rope_rows = max_rope_rows.unwrap_or(cfg.max_position_embeddings);
    let rope_factors = loader.take("rope_freqs.weight").ok().map(|(d, _)| d);
    let inv_freq = resolve_inv_freq(cfg, rope_factors.as_deref());
    let (cos_data, sin_data) = build_rope_tables(&inv_freq, rope_rows);

    let arch = loader.arch_hint().unwrap_or("gemma").to_string();

    let mut f32_params: HashMap<String, Vec<f32>> = HashMap::new();
    // Force-dequant the embed table to F32 so the input-embedding gather can
    // continue using a host-side f32 path. We do NOT pre-transpose it for the
    // tied LM head — that step is replaced by DequantMatMul on the original
    // Q4K-packed `token_embd.weight` bytes (see graph builder below). On
    // Gemma 4 31B Q4_K_M that saves the ~5.6 GB f32 transpose plus the same
    // amount cloned into each prefill/decode graph constant.
    //
    // Task #36: prefer `take_packed` so we ALSO retain the original Q4K bytes
    // for the LM head's `DequantMatMul`. Falls back to `take()` for f16/f32
    // embed tables (small models) or if the scheme isn't dequant-supported.
    let mut embed_packed: Option<(Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> = None;
    // RLX_GEMMA_LAZY_EMBED_DISABLE=1 forces the legacy `take()`+f32-gather path
    // even when packed bytes are available, for parity bisection.
    let lazy_disabled = std::env::var("RLX_GEMMA_LAZY_EMBED_DISABLE")
        .ok()
        .as_deref()
        == Some("1");
    if lazy_disabled {
        if let Ok((data, _shape)) = loader.take("model.embed_tokens.weight") {
            f32_params.insert("model.embed_tokens.weight".into(), data);
        }
    } else {
        match loader.take_packed("model.embed_tokens.weight") {
            Ok(Some((bytes, scheme, shape))) => {
                // Task #37: when packed Q4K bytes are available the runtime
                // host-gathers embedding rows on demand (see
                // `packed_session.rs::gather_embed_rows`), so we DO NOT dequant
                // the full vocab×hidden f32 table at LOAD. For 12B Q4_K_M that's
                // ~3.8 GB never allocated.
                embed_packed = Some((bytes, scheme, shape));
            }
            _ => {
                if let Ok((data, _shape)) = loader.take("model.embed_tokens.weight") {
                    f32_params.insert("model.embed_tokens.weight".into(), data);
                }
            }
        }
    }

    let (mut wm, packed_list) =
        WeightMap::drain_loader(loader, WeightDrainPolicy::AllF32WarnUnused)?;
    for key in wm.keys().map(str::to_string).collect::<Vec<_>>() {
        let (data, _shape) = wm.take(&key)?;
        let canonical = rlx_core::weight_loader::gguf_to_hf_name_for_arch(&key, &arch)
            .unwrap_or_else(|| key.clone());
        f32_params.insert(canonical, data);
    }
    f32_params.insert("rope.cos".into(), cos_data);
    f32_params.insert("rope.sin".into(), sin_data);
    if let Some(global_inv) = resolve_global_inv_freq(cfg, rope_factors.as_deref()) {
        let (gcd, gsd) = build_rope_tables(&global_inv, rope_rows);
        f32_params.insert("rope.global.cos".into(), gcd);
        f32_params.insert("rope.global.sin".into(), gsd);
    }

    let mut packed = HashMap::new();
    for (key, bytes, scheme, shape) in packed_list {
        let canonical = rlx_core::weight_loader::gguf_to_hf_name_for_arch(&key, &arch)
            .unwrap_or_else(|| key.clone());
        packed.insert(canonical, (bytes, scheme, shape));
    }
    // Task #36: keep the original Q4K bytes of `embed_tokens` so the LM-head
    // builder can issue a DequantMatMul on them instead of materialising a
    // ~4 GB transposed f32 constant per session.
    if let Some((bytes, scheme, shape)) = embed_packed {
        packed.insert("model.embed_tokens.weight".into(), (bytes, scheme, shape));
    }

    // Per-layer projection fusion. Each row of a Q4K weight is an
    // independent block sequence, so collapsing N proj weights into one
    // by byte-concat along the output (`n`) axis lets the runtime issue
    // one matmul + N narrows instead of N matmuls — saves the dispatch
    // overhead llama.cpp hides via fused MSL kernels per layer. Every
    // backend (rlx-cpu / rlx-metal / rlx-mlx) gets the win for free via
    // its existing matmul + narrow lowerings.
    let fuse_gate_up = std::env::var("RLX_GEMMA_NO_FUSE_GATE_UP").as_deref() != Ok("1");
    let fuse_qkv = std::env::var("RLX_GEMMA_NO_FUSE_QKV").as_deref() != Ok("1");
    let num_layers = cfg.num_hidden_layers;
    for layer in 0..num_layers {
        // FFN: gate_proj || up_proj.
        if fuse_gate_up {
            let gk = format!("model.layers.{layer}.mlp.gate_proj.weight");
            let uk = format!("model.layers.{layer}.mlp.up_proj.weight");
            if let (Some(gate_entry), Some(up_entry)) =
                (packed.get(&gk).cloned(), packed.get(&uk).cloned())
            {
                let (gate_bytes, gate_scheme, gate_shape) = gate_entry;
                let (up_bytes, up_scheme, up_shape) = up_entry;
                let gate_n = gate_shape.first().copied().unwrap_or(0);
                let gate_k = gate_shape.get(1).copied().unwrap_or(0);
                let up_n = up_shape.first().copied().unwrap_or(0);
                let up_k = up_shape.get(1).copied().unwrap_or(0);
                if gate_scheme == up_scheme && gate_k > 0 && gate_k == up_k {
                    let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
                    fused.extend_from_slice(&gate_bytes);
                    fused.extend_from_slice(&up_bytes);
                    packed.insert(
                        format!("model.layers.{layer}.mlp.gate_up.weight"),
                        (fused, gate_scheme, vec![gate_n + up_n, gate_k]),
                    );
                    packed.remove(&gk);
                    packed.remove(&uk);
                }
            }
        }
        // Attention: q_proj || k_proj [|| v_proj]. With attention_k_eq_v
        // the v stream aliases k so the GGUF has no v_proj.weight; the
        // fused key is q||k only and the graph builder uses k as v.
        if fuse_qkv {
            let qk_key = format!("model.layers.{layer}.self_attn.q_proj.weight");
            let kk_key = format!("model.layers.{layer}.self_attn.k_proj.weight");
            let vk_key = format!("model.layers.{layer}.self_attn.v_proj.weight");
            let q_entry = packed.get(&qk_key).cloned();
            let k_entry = packed.get(&kk_key).cloned();
            let v_entry = packed.get(&vk_key).cloned();
            if let (Some(q_e), Some(k_e)) = (q_entry, k_entry) {
                let (q_bytes, q_scheme, q_shape) = q_e;
                let (k_bytes, k_scheme, k_shape) = k_e;
                let q_n = q_shape.first().copied().unwrap_or(0);
                let q_k_dim = q_shape.get(1).copied().unwrap_or(0);
                let k_n = k_shape.first().copied().unwrap_or(0);
                let k_k_dim = k_shape.get(1).copied().unwrap_or(0);
                if q_scheme == k_scheme && q_k_dim > 0 && q_k_dim == k_k_dim {
                    let mut fused = Vec::with_capacity(
                        q_bytes.len()
                            + k_bytes.len()
                            + v_entry.as_ref().map_or(0, |(b, _, _)| b.len()),
                    );
                    fused.extend_from_slice(&q_bytes);
                    fused.extend_from_slice(&k_bytes);
                    let (mut total_n, has_v) = (q_n + k_n, v_entry.is_some());
                    if let Some((v_bytes, v_scheme, v_shape)) = v_entry.as_ref() {
                        let v_n = v_shape.first().copied().unwrap_or(0);
                        let v_k_dim = v_shape.get(1).copied().unwrap_or(0);
                        if *v_scheme == q_scheme && v_k_dim == q_k_dim {
                            fused.extend_from_slice(v_bytes);
                            total_n += v_n;
                        } else {
                            // Mixed quant — skip the fusion for this
                            // layer rather than emit a broken tensor.
                            continue;
                        }
                    }
                    packed.insert(
                        format!("model.layers.{layer}.self_attn.qkv.weight"),
                        (fused, q_scheme, vec![total_n, q_k_dim]),
                    );
                    packed.remove(&qk_key);
                    packed.remove(&kk_key);
                    if has_v {
                        packed.remove(&vk_key);
                    }
                }
            }
        }
    }

    Ok((f32_params, packed))
}

#[allow(clippy::too_many_arguments)]
pub fn build_gemma_graph_sized_packed(
    cfg: &GemmaConfig,
    weights: &mut rlx_core::weight_loader::GgufLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_token_from_input: bool,
    with_kv_outputs: bool,
    packed: &mut PackedWeightMap,
) -> Result<(Graph, F32WeightMap)> {
    build_gemma_graph_sized_packed_ext(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        last_token_from_input,
        with_kv_outputs,
        packed,
        None,
        None,
    )
}

/// Like [`build_gemma_graph_sized_packed`] but can rebuild from session weight caches.
#[allow(clippy::too_many_arguments)]
pub fn build_gemma_graph_sized_packed_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn rlx_core::weight_loader::WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_token_from_input: bool,
    with_kv_outputs: bool,
    packed: &mut PackedWeightMap,
    known_packed: Option<&PackedWeightMap>,
    known_f32: Option<&F32WeightMap>,
) -> Result<(Graph, F32WeightMap)> {
    use crate::config::GemmaArch;
    use crate::rope::{build_rope_tables, resolve_inv_freq};
    use rlx_core::weight_loader::WeightLoader;
    use rlx_ir::op::{Activation, Op};
    use rlx_ir::quant::QuantScheme;
    use rlx_ir::{DType, NodeId, Shape};

    validate_cfg(cfg)?;

    let mut g = Graph::new("gemma_packed");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let eps = cfg.rms_norm_eps as f32;
    let num_layers = cfg.active_num_layers();

    // ── Helpers ────────────────────────────────────────────────────
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
    fn synth_const(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        name: &str,
        data: Vec<f32>,
        shape: &[usize],
    ) -> NodeId {
        let id = g.param(name, Shape::new(shape, DType::F32));
        params.insert(name.to_string(), data);
        id
    }
    fn load_p_cached(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        weights: &mut dyn WeightLoader,
        known_f32: Option<&HashMap<String, Vec<f32>>>,
        key: &str,
        shape: &[usize],
        transpose: bool,
    ) -> Result<NodeId> {
        let (data, out_shape) = if let Some(cached) = known_f32.and_then(|m| m.get(key)) {
            if transpose {
                let rows = shape[0];
                let cols = shape[1];
                let mut t = vec![0f32; cached.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        t[c * rows + r] = cached[r * cols + c];
                    }
                }
                (t, vec![cols, rows])
            } else {
                (cached.clone(), shape.to_vec())
            }
        } else if transpose {
            weights.take_transposed(key)?
        } else {
            weights.take(key)?
        };
        let id = g.param(key, Shape::new(&out_shape, DType::F32));
        params.insert(key.to_string(), data);
        Ok(id)
    }
    fn load_proj(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        packed: &mut PackedWeightMap,
        weights: &mut dyn WeightLoader,
        known_packed: Option<&PackedWeightMap>,
        known_f32: Option<&F32WeightMap>,
        key: &str,
    ) -> Result<(NodeId, Option<QuantScheme>)> {
        if let Some((bytes, scheme, shape)) = known_packed.and_then(|m| m.get(key)) {
            if bytes.is_empty() {
                let cached = known_f32
                    .and_then(|m| m.get(key))
                    .ok_or_else(|| anyhow::anyhow!("f32 cache miss for drained proj {key}"))?;
                let id = g.param(key, Shape::new(shape, DType::F32));
                params.insert(key.to_string(), cached.clone());
                return Ok((id, None));
            }
            let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
            return Ok((id, Some(*scheme)));
        }
        if let Some((bytes, scheme, shape)) = weights.take_packed(key)? {
            let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
            packed.insert(key.to_string(), (bytes, scheme, shape));
            Ok((id, Some(scheme)))
        } else {
            let (data, shape) = weights.take_transposed(key)?;
            let id = g.param(key, Shape::new(&shape, DType::F32));
            params.insert(key.to_string(), data);
            // Sentinel: proj was materialized to F32 on drain (e.g. Q4_0); rebuild from cache.
            packed.insert(key.to_string(), (Vec::new(), QuantScheme::GgufQ4_0, shape));
            Ok((id, None))
        }
    }
    fn emit_proj(
        g: &mut Graph,
        input: NodeId,
        w: NodeId,
        scheme: Option<QuantScheme>,
        out_shape: Shape,
    ) -> NodeId {
        match scheme {
            Some(s) => g.add_node(Op::DequantMatMul { scheme: s }, vec![input, w], out_shape),
            None => g.mm(input, w),
        }
    }
    /// Delta-gamma RMS norm: gamma = 1 + loaded_weight.
    fn gemma_rms(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        x: NodeId,
        weight_key: &str,
        weights: &mut dyn WeightLoader,
        known_f32: Option<&HashMap<String, Vec<f32>>>,
        _zero_beta: NodeId,
        h: usize,
        eps: f32,
    ) -> Result<NodeId> {
        let w = load_p_cached(g, params, weights, known_f32, weight_key, &[h], false)?;
        let ones = synth_const(
            g,
            params,
            &format!("{weight_key}.ones"),
            vec![1.0f32; h],
            &[h],
        );
        let gamma = g.add(ones, w);
        // Build a beta sized to `h` rather than reusing the caller's
        // hidden-sized zero_beta — otherwise Q/K norm (h=head_dim) gets
        // a beta of size hidden_size and the RmsNorm op's gamma/beta
        // shapes disagree, propagating NaN through the network. This
        // was the root cause for task #50's persistent all-NaN logits.
        let beta = synth_const(
            g,
            params,
            &format!("{weight_key}.beta"),
            vec![0.0f32; h],
            &[h],
        );
        Ok(g.rms_norm(x, gamma, beta, eps))
    }

    let zero_beta = synth_const(
        &mut g,
        &mut params,
        "gemma.packed.zero_beta",
        vec![0.0f32; h],
        &[h],
    );

    // ── Default RoPE table (sliding layers + non-Gemma-4) ─────────
    let inv_freq = if known_f32.is_some() {
        resolve_inv_freq(cfg, None)
    } else {
        let rope_factors = weights.take("rope_freqs.weight").ok().map(|(d, _)| d);
        resolve_inv_freq(cfg, rope_factors.as_deref())
    };
    let half = inv_freq.len();
    let rope_len = seq;
    let (cos_id, sin_id) = if let (Some(cos), Some(sin)) = (
        known_f32.and_then(|m| m.get("rope.cos")),
        known_f32.and_then(|m| m.get("rope.sin")),
    ) {
        (
            synth_const(
                &mut g,
                &mut params,
                "rope.cos",
                slice_rope_table(cos, half, rope_len),
                &[rope_len, half],
            ),
            synth_const(
                &mut g,
                &mut params,
                "rope.sin",
                slice_rope_table(sin, half, rope_len),
                &[rope_len, half],
            ),
        )
    } else {
        let rope_factors = weights.take("rope_freqs.weight").ok().map(|(d, _)| d);
        let inv = resolve_inv_freq(cfg, rope_factors.as_deref());
        let (cos_data, sin_data) = build_rope_tables(&inv, rope_len);
        (
            synth_const(&mut g, &mut params, "rope.cos", cos_data, &[rope_len, half]),
            synth_const(&mut g, &mut params, "rope.sin", sin_data, &[rope_len, half]),
        )
    };

    // ── Secondary "global" RoPE table for Gemma 4 full-attention ──
    let (global_cos, global_sin) = if let (Some(cos), Some(sin)) = (
        known_f32.and_then(|m| m.get("rope.global.cos")),
        known_f32.and_then(|m| m.get("rope.global.sin")),
    ) {
        // Drain sizes the cached cos/sin table to `rope_rows` (= `max_seq + 16`
        // per `drain_gemma_packed_weights_ext`), NOT `max_position_embeddings`.
        // The old inference `half_g = cos.len() / max_position_embeddings`
        // gave 0 for capped tables (18432 / 262144 = 0), registering
        // rope.global.cos as 0-sized → empty tensors propagated through every
        // FULL-attention layer's RoPE.
        //
        // The full-attention head_dim can differ from the SWA head_dim (Gemma
        // 4 12B: SWA=256 → half=128, FULL=512 → half_g=256), so `half` is the
        // wrong upper bound for the FULL layer. Recover the FULL half_g from
        // the model config the same way drain did: ask
        // `resolve_global_inv_freq` for the global inverse-frequency table —
        // its length is the FULL-layer `head_dim / 2`.
        let half_g = crate::rope::resolve_global_inv_freq(cfg, None)
            .map(|v| v.len())
            .unwrap_or(half);
        (
            Some(synth_const(
                &mut g,
                &mut params,
                "rope.global.cos",
                slice_rope_table(cos, half_g, rope_len),
                &[rope_len, half_g],
            )),
            Some(synth_const(
                &mut g,
                &mut params,
                "rope.global.sin",
                slice_rope_table(sin, half_g, rope_len),
                &[rope_len, half_g],
            )),
        )
    } else if let Some(global_inv) = crate::rope::resolve_global_inv_freq(cfg, None) {
        let half_g = global_inv.len();
        let (cd, sd) = build_rope_tables(&global_inv, rope_len);
        let c = synth_const(
            &mut g,
            &mut params,
            "rope.global.cos",
            cd,
            &[rope_len, half_g],
        );
        let s = synth_const(
            &mut g,
            &mut params,
            "rope.global.sin",
            sd,
            &[rope_len, half_g],
        );
        (Some(c), Some(s))
    } else {
        (None, None)
    };

    let vocab = cfg.vocab_size;
    // Task #37: when the embed table is in `known_packed` (Q4K bytes), bypass
    // the in-graph `gather(embed_w, input_ids)` and accept the gathered rows
    // directly. The runtime (`packed_session.rs::gather_embed_rows`) dequants
    // only the prompt-token rows host-side. Skips the ~3.8 GB f32 embed cache
    // and the per-bucket embed param upload.
    let embed_lazy = known_packed
        .map(|m| m.contains_key("model.embed_tokens.weight"))
        .unwrap_or(false);
    // Multimodal: when the caller marks `__media_bias__`, attention uses an
    // additive bias tensor `[batch, heads, seq, seq]` (`MaskKind::Bias`) instead
    // of the fused causal/sliding `MaskKind` — letting bidirectional image/audio
    // blocks open up. Valid for `seq <= sliding_window` (sliding == causal then),
    // so one bias serves all layer types. Gated entirely on the sentinel: callers
    // that don't set it are byte-identical to before.
    let media_bias_id = if known_packed
        .map(|m| m.contains_key("__media_bias__"))
        .unwrap_or(false)
    {
        Some(g.input("attn_bias", Shape::new(&[batch, nh, seq, seq], DType::F32)))
    } else {
        None
    };
    let last_token_idx = if with_lm_head && last_token_from_input {
        Some(g.input("last_token_idx", Shape::new(&[batch], DType::F32)))
    } else {
        None
    };
    let mut h_id = if embed_lazy {
        g.input("input_embeddings", Shape::new(&[batch, seq, h], DType::F32))
    } else {
        let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
        let embed_w = load_p_cached(
            &mut g,
            &mut params,
            weights,
            known_f32,
            "model.embed_tokens.weight",
            &[vocab, h],
            false,
        )?;
        g.gather_(embed_w, input_ids, 0)
    };
    // Gemma embed-scale (sqrt(hidden_size)) — emitted as a single
    // scalar multiply; matches GemmaFlow's EmbedScaleStage.
    let scale_val = (h as f32).sqrt();
    let embed_scale = synth_const(
        &mut g,
        &mut params,
        "gemma.packed.embed_scale",
        vec![scale_val],
        &[1],
    );
    h_id = g.mul(h_id, embed_scale);

    let attn_score_scale = cfg.attn_score_scale();
    let attn_softcap = cfg.attn_logit_softcapping;
    let mut kv_outputs: Vec<(NodeId, NodeId)> = Vec::new();
    // Gemma 4 E2B KV sharing: the last fresh layer of each attention type
    // (sliding / full) stores its post-norm/post-RoPE K and post-v_norm V;
    // shared layers (>= first_kv_shared_layer) reuse them instead of computing
    // their own. Index 0 = sliding, 1 = full. Empty for non-E2B configs.
    let mut shared_k: [Option<NodeId>; 2] = [None, None];
    let mut shared_v: [Option<NodeId>; 2] = [None, None];
    // Gemma 4 E2B Per-Layer Embeddings: the runner precomputes the per-layer
    // input slices [batch, seq, num_layers * ple_w] (gather + dequant + project
    // + combine) and feeds them as a graph input; each layer slices its block.
    let per_layer_inputs = if cfg.has_ple() {
        Some(g.input(
            "per_layer_inputs",
            Shape::new(&[batch, seq, num_layers * cfg.ple_width()], f),
        ))
    } else {
        None
    };
    // Diagnostic tap: when RLX_TAP_L0=1, surface layer-0 intermediates
    // as additional graph outputs so we can bisect Metal's all-NaN bug.
    // Order (consumed in `packed_session::predict_logits`):
    //   1: embed*scale (h_id at layer-0 entry)
    //   2: input_layernorm(x)
    //   3: Q after per-head q_norm + reshape back to [B,S,q_dim]
    //   4: K after per-head k_norm + reshape back to [B,S,kv_dim]
    //   5: V after v_norm + reshape back
    //   6: Q after RoPE
    //   7: K after RoPE
    //   8: Attention output (pre-o_proj)
    //   9: attn_out after post_attention_norm
    //  10: residual h_id + attn_out
    //  11: layer 0 final h (post-FFN add)
    let tap_l0 = std::env::var("RLX_TAP_L0").ok().is_some();
    let tap_layer: usize = std::env::var("RLX_TAP_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut l0_taps: Vec<NodeId> = Vec::new();
    // RLX_TAP_ALL: append every layer's final hidden state as a graph output,
    // for full-trajectory parity bisection against HF hidden_states.
    let tap_all = std::env::var("RLX_TAP_ALL").ok().is_some();
    let mut all_layer_taps: Vec<NodeId> = Vec::new();

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        let layer_dh = cfg.layer_head_dim(layer);
        let layer_kv = cfg.layer_num_kv_heads(layer);
        let layer_nrot = cfg.layer_n_rot(layer);
        let q_dim = nh * layer_dh;
        let kv_dim = layer_kv * layer_dh;
        let group = nh / layer_kv;
        // Gemma 4 E2B: KV-shared layers use a double-wide MLP
        // (intermediate_size × 2). Non-E2B configs return the base size,
        // leaving flagship/legacy behavior unchanged.
        let int_dim = cfg.layer_intermediate_size(layer);
        let is_shared_kv = cfg.is_kv_shared_layer(layer);
        let kv_type_idx = cfg.is_full_attention_layer(layer) as usize;

        if tap_l0 && layer == tap_layer {
            l0_taps.push(h_id); // tap 1: embed*scale (layer-0 input)
        }

        // input_layernorm.
        let normed_in = gemma_rms(
            &mut g,
            &mut params,
            h_id,
            &format!("{lp}.input_layernorm.weight"),
            weights,
            known_f32,
            zero_beta,
            h,
            eps,
        )?;
        if tap_l0 && layer == tap_layer {
            l0_taps.push(normed_in); // tap 2: input_layernorm(x)
        }

        // Q/K/V projections. v_proj is skipped when k_eq_v.
        let fused_qkv_key = format!("{lp}.self_attn.qkv.weight");
        let has_fused_qkv = known_packed
            .map(|m| m.contains_key(&fused_qkv_key))
            .unwrap_or(false);
        let (q, k, v) = if has_fused_qkv {
            let (qkv_w, qkv_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &fused_qkv_key,
            )?;
            let v_present = !cfg.layer_k_eq_v(layer);
            let total_n = q_dim + kv_dim + if v_present { kv_dim } else { 0 };
            let combined = emit_proj(
                &mut g,
                normed_in,
                qkv_w,
                qkv_s,
                Shape::new(&[batch, seq, total_n], f),
            );
            let q = g.narrow_(combined, 2, 0, q_dim);
            let k = g.narrow_(combined, 2, q_dim, kv_dim);
            let v = if v_present {
                g.narrow_(combined, 2, q_dim + kv_dim, kv_dim)
            } else {
                k
            };
            (q, k, v)
        } else {
            let (q_w, q_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.self_attn.q_proj.weight"),
            )?;
            let q = emit_proj(
                &mut g,
                normed_in,
                q_w,
                q_s,
                Shape::new(&[batch, seq, q_dim], f),
            );
            // KV-shared layers (Gemma 4 E2B, >= first_kv_shared_layer) reuse
            // K/V from an earlier same-type layer and ship no usable
            // k_norm/k_proj/v_proj — skip their projections entirely; the real
            // K/V are substituted after the RoPE step below. `q` is a harmless
            // placeholder for k/v here.
            let (k, v) = if is_shared_kv {
                (q, q)
            } else {
                let (k_w, k_s) = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    known_packed,
                    known_f32,
                    &format!("{lp}.self_attn.k_proj.weight"),
                )?;
                let k = emit_proj(
                    &mut g,
                    normed_in,
                    k_w,
                    k_s,
                    Shape::new(&[batch, seq, kv_dim], f),
                );
                let v = if cfg.layer_k_eq_v(layer) {
                    k
                } else {
                    let (v_w, v_s) = load_proj(
                        &mut g,
                        &mut params,
                        packed,
                        weights,
                        known_packed,
                        known_f32,
                        &format!("{lp}.self_attn.v_proj.weight"),
                    )?;
                    emit_proj(
                        &mut g,
                        normed_in,
                        v_w,
                        v_s,
                        Shape::new(&[batch, seq, kv_dim], f),
                    )
                };
                (k, v)
            };
            (q, k, v)
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(q); // tap A (was 3): Q POST-projection, PRE per-head norm
            l0_taps.push(k); // tap B: K post-projection, pre per-head norm
            l0_taps.push(v); // tap C: V post-projection
        }

        // Gemma 4 per-head Q/K RMS norms + V RMS norm (see decode
        // builder for full rationale + task #50). Same fix applied
        // here so predict_logits (prefill path) doesn't produce
        // all-NaN logits.
        let (q, k, v) = if matches!(cfg.arch, GemmaArch::Gemma4) {
            let q_norm_key = format!("{lp}.self_attn.q_norm.weight");
            let k_norm_key = format!("{lp}.self_attn.k_norm.weight");
            let q_4d = g.reshape_(
                q,
                vec![batch as i64, seq as i64, nh as i64, layer_dh as i64],
            );
            if tap_l0 && layer == tap_layer {
                l0_taps.push(q_4d); // tap D: Q reshape-only (4D pre-norm)
            }
            let q_normed = gemma_rms(
                &mut g,
                &mut params,
                q_4d,
                &q_norm_key,
                weights,
                known_f32,
                zero_beta,
                layer_dh,
                eps,
            )?;
            if tap_l0 && layer == tap_layer {
                l0_taps.push(q_normed); // tap E: Q after per-head RMS norm (4D)
            }
            let q = g.reshape_(q_normed, vec![batch as i64, seq as i64, q_dim as i64]);

            // KV-shared layers don't carry k_norm/v_norm and reuse stored K/V,
            // so skip the K/V norm entirely (the `k`/`v` placeholders are
            // discarded after RoPE below).
            let (k, v) = if is_shared_kv {
                (k, v)
            } else {
                let k_4d = g.reshape_(
                    k,
                    vec![batch as i64, seq as i64, layer_kv as i64, layer_dh as i64],
                );
                let k_normed = gemma_rms(
                    &mut g,
                    &mut params,
                    k_4d,
                    &k_norm_key,
                    weights,
                    known_f32,
                    zero_beta,
                    layer_dh,
                    eps,
                )?;
                let k = g.reshape_(k_normed, vec![batch as i64, seq as i64, kv_dim as i64]);
                // V RMS-norm with no learnable scale — matches llama.cpp
                // gemma4.cpp:256 `ggml_rms_norm(Vcur, f_norm_rms_eps)`.
                // Without this V grows unbounded → attention output blows
                // up over 48 layers → NaN logits.
                let v_4d = g.reshape_(
                    v,
                    vec![batch as i64, seq as i64, layer_kv as i64, layer_dh as i64],
                );
                let v_ones = synth_const(
                    &mut g,
                    &mut params,
                    &format!("{lp}.self_attn.v_norm.ones"),
                    vec![1.0f32; layer_dh],
                    &[layer_dh],
                );
                let v_zeros = synth_const(
                    &mut g,
                    &mut params,
                    &format!("{lp}.self_attn.v_norm.zeros"),
                    vec![0.0f32; layer_dh],
                    &[layer_dh],
                );
                let v_normed = g.rms_norm(v_4d, v_ones, v_zeros, eps);
                let v = g.reshape_(v_normed, vec![batch as i64, seq as i64, kv_dim as i64]);
                (k, v)
            };
            (q, k, v)
        } else {
            (q, k, v)
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(q); // tap 3: Q post-norm (after per-head q_norm)
            l0_taps.push(k); // tap 4: K post-norm
            l0_taps.push(v); // tap 5: V post-norm
        }

        // RoPE — pick global slot for full-attention layers when split.
        let (layer_cos, layer_sin) = if cfg.is_full_attention_layer(layer) {
            match (global_cos, global_sin) {
                (Some(gc), Some(gs)) => (gc, gs),
                _ => (cos_id, sin_id),
            }
        } else {
            (cos_id, sin_id)
        };
        let q_rope = g.rope_n(q, layer_cos, layer_sin, layer_dh, layer_nrot);
        // E2B KV sharing: shared layers reuse the stored same-type (post-norm,
        // post-RoPE) K and (post-v_norm) V; fresh layers RoPE their own K and,
        // when sharing is active, store it for later same-type layers.
        let (k_rope, v) = if is_shared_kv {
            let sk = shared_k[kv_type_idx].ok_or_else(|| {
                anyhow!("KV-shared layer {layer}: no stored source K for type {kv_type_idx}")
            })?;
            let sv = shared_v[kv_type_idx]
                .ok_or_else(|| anyhow!("KV-shared layer {layer}: no stored source V"))?;
            (sk, sv)
        } else {
            let k_rope = g.rope_n(k, layer_cos, layer_sin, layer_dh, layer_nrot);
            if cfg.num_kv_shared_layers > 0 {
                shared_k[kv_type_idx] = Some(k_rope);
                shared_v[kv_type_idx] = Some(v);
            }
            (k_rope, v)
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(q_rope); // tap 6: Q post-RoPE
            l0_taps.push(k_rope); // tap 7: K post-RoPE
        }
        if with_kv_outputs {
            kv_outputs.push((k_rope, v));
        }

        let k_rep = repeat_kv_packed(&mut g, k_rope, layer_kv, layer_dh, group);
        let v_rep = repeat_kv_packed(&mut g, v, layer_kv, layer_dh, group);
        if tap_l0 && layer == 0 {
            l0_taps.push(k_rep); // tap F: K_rep
            l0_taps.push(v_rep); // tap G: V_rep
        }

        // Per-layer mask.
        let (mask_kind, _, _) = cfg.layer_attn_options(layer);
        let attn_shape = rlx_ir::shape::attention_shape(g.shape(q_rope));
        let attn = if let Some(bias) = media_bias_id {
            g.attention_bias_opts(
                q_rope,
                k_rep,
                v_rep,
                bias,
                nh,
                layer_dh,
                attn_shape,
                attn_score_scale,
                attn_softcap,
            )
        } else {
            g.attention_kind_opts(
                q_rope,
                k_rep,
                v_rep,
                nh,
                layer_dh,
                mask_kind,
                attn_shape,
                attn_score_scale,
                attn_softcap,
            )
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(attn); // tap 8: attention output (pre-o_proj)
        }

        let (o_w, o_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, o_w, o_s, Shape::new(&[batch, seq, h], f));
        // Gemma 3/4 sandwich-norm: post_attention_layernorm applied to
        // attn_out BEFORE the residual add. Without this the attention
        // contribution grows unbounded across 48 layers — task #50 NaN
        // root cause. Maps GGUF blk.N.post_attention_norm.weight via
        // gguf_to_hf_name_for_arch.
        let attn_out = if matches!(cfg.arch, GemmaArch::Gemma3 | GemmaArch::Gemma4) {
            gemma_rms(
                &mut g,
                &mut params,
                attn_out,
                &format!("{lp}.post_attention_layernorm.weight"),
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?
        } else {
            attn_out
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(attn_out); // tap 9: attn_out after post_attn_norm
        }
        let post_attn = g.add(h_id, attn_out);
        if tap_l0 && layer == tap_layer {
            l0_taps.push(post_attn); // tap 10: residual h + attn_out
        }

        // Pre-FFN norm — Gemma 1 uses `post_attention_layernorm`;
        // Gemma 2/3/4 use `pre_feedforward_layernorm`.
        let pre_ffn_key = if cfg.arch == GemmaArch::Gemma {
            format!("{lp}.post_attention_layernorm.weight")
        } else {
            format!("{lp}.pre_feedforward_layernorm.weight")
        };
        let normed_post = gemma_rms(
            &mut g,
            &mut params,
            post_attn,
            &pre_ffn_key,
            weights,
            known_f32,
            zero_beta,
            h,
            eps,
        )?;
        if tap_l0 && layer == tap_layer {
            l0_taps.push(normed_post); // tap 10b: pre-FFN rms norm
        }

        // GeGLU MLP.
        // Try fused {gate_up}_proj first — set up at drain time by
        // `drain_gemma_packed_weights`. One matmul + narrow halves
        // replaces two matmuls on the same input.
        let fused_gate_up_key = format!("{lp}.mlp.gate_up.weight");
        let fused_gate_up = known_packed
            .map(|m| m.contains_key(&fused_gate_up_key))
            .unwrap_or(false);
        let (gate, up) = if fused_gate_up {
            let (gu_w, gu_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &fused_gate_up_key,
            )?;
            let combined = emit_proj(
                &mut g,
                normed_post,
                gu_w,
                gu_s,
                Shape::new(&[batch, seq, int_dim * 2], f),
            );
            let gate = g.narrow_(combined, 2, 0, int_dim);
            let up = g.narrow_(combined, 2, int_dim, int_dim);
            (gate, up)
        } else {
            let (gate_w, gate_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.mlp.gate_proj.weight"),
            )?;
            let (up_w, up_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.mlp.up_proj.weight"),
            )?;
            let gate = emit_proj(
                &mut g,
                normed_post,
                gate_w,
                gate_s,
                Shape::new(&[batch, seq, int_dim], f),
            );
            let up = emit_proj(
                &mut g,
                normed_post,
                up_w,
                up_s,
                Shape::new(&[batch, seq, int_dim], f),
            );
            (gate, up)
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(gate); // tap 10c: gate proj
            l0_taps.push(up); // tap 10d: up proj
        }
        let (down_w, down_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.mlp.down_proj.weight"),
        )?;
        let gate_act = g.gelu_approx(gate);
        if tap_l0 && layer == tap_layer {
            l0_taps.push(gate_act); // tap 10e: gelu(gate)
        }
        let mlp_inner = g.mul(gate_act, up);
        if tap_l0 && layer == tap_layer {
            l0_taps.push(mlp_inner); // tap 10f: gate*up
        }
        let mut ffn_out = emit_proj(
            &mut g,
            mlp_inner,
            down_w,
            down_s,
            Shape::new(&[batch, seq, h], f),
        );
        if tap_l0 && layer == tap_layer {
            l0_taps.push(ffn_out); // tap 10g: down proj (pre post_ffn norm)
        }

        // Post-FFN norm for Gemma 2/3/4.
        if cfg.arch != GemmaArch::Gemma {
            let post_ffn_key = format!("{lp}.post_feedforward_layernorm.weight");
            ffn_out = gemma_rms(
                &mut g,
                &mut params,
                ffn_out,
                &post_ffn_key,
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?;
        }

        h_id = g.add(post_attn, ffn_out);
        if tap_l0 && layer == tap_layer {
            l0_taps.push(h_id); // tap 12: residual after FFN (pre-PLE)
        }

        // Gemma 4 E2B Per-Layer Embeddings: inject this layer's per-layer input
        // slice after the MLP residual (HF `Gemma4TextDecoderLayer.forward`):
        //   res = h; h = post_norm( per_layer_projection(
        //             gelu(per_layer_input_gate(h)) * ple_slice ) ); h = res + h
        if let Some(ple_all) = per_layer_inputs {
            let pw = cfg.ple_width();
            let ple_slice = g.narrow_(ple_all, 2, layer * pw, pw); // [B,S,ple_w]
            let (gate_w, gate_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.per_layer_input_gate.weight"),
            )?;
            let gated = emit_proj(
                &mut g,
                h_id,
                gate_w,
                gate_s,
                Shape::new(&[batch, seq, pw], f),
            );
            let gated = g.gelu_approx(gated);
            let gated = g.mul(gated, ple_slice);
            let (proj_w, proj_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.per_layer_projection.weight"),
            )?;
            let projected = emit_proj(
                &mut g,
                gated,
                proj_w,
                proj_s,
                Shape::new(&[batch, seq, h], f),
            );
            let projected = gemma_rms(
                &mut g,
                &mut params,
                projected,
                &format!("{lp}.post_per_layer_input_norm.weight"),
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?;
            h_id = g.add(h_id, projected);
            if tap_l0 && layer == tap_layer {
                l0_taps.push(h_id); // tap 13: after PLE (pre layer_scalar)
            }
        }

        // Gemma 4 per-layer output scalar: E2B uses `layer_scalar`; flagship
        // GGUF ships `layer_output_scale` (≈0.02–0.05). llama.cpp multiplies
        // every layer output by this before the next residual; skipping it lets
        // the hidden stream grow ~20–50× per layer → softcap-saturated garbage
        // logits on prefill (decode already applied this in the builder below).
        if cfg.has_ple() {
            let ls = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.layer_scalar"),
                false,
            )?;
            h_id = g.mul(h_id, ls);
        } else if matches!(cfg.arch, GemmaArch::Gemma4) {
            let scale_w = load_p_cached(
                &mut g,
                &mut params,
                weights,
                known_f32,
                &format!("{lp}.self_attn.output_scale.weight"),
                &[1],
                false,
            )?;
            h_id = g.mul(h_id, scale_w);
        }
        if tap_l0 && layer == tap_layer {
            l0_taps.push(h_id); // tap 11: layer 0 final h
        }
        if tap_all {
            all_layer_taps.push(h_id);
        }
    }

    // model.norm + lm_head.
    let hidden = gemma_rms(
        &mut g,
        &mut params,
        h_id,
        "model.norm.weight",
        weights,
        known_f32,
        zero_beta,
        h,
        eps,
    )?;

    let out = if with_lm_head {
        // lm_head is normally tied to embed_tokens for Gemma. Use the
        // F32 transposed embed; quantized lm_head isn't ubiquitous.
        let head_input = if let Some(idx) = last_token_idx {
            gather_last_token_packed(&mut g, hidden, batch, idx)
        } else {
            hidden
        };
        let logit_rows = if last_token_from_input { 1 } else { seq };
        let vocab = cfg.vocab_size;
        // Task #36: when the embed bytes are still packed (Q4K/Q6K), issue a
        // single `Op::DequantMatMul` against them instead of building the
        // ~4 GB transposed f32 constant (and the ~4 GB clone into the graph).
        let packed_embed_scheme = known_packed
            .and_then(|m| m.get("model.embed_tokens.weight"))
            .map(|(_, scheme, _)| *scheme);
        let mut logits = if let Some(scheme) =
            packed_embed_scheme.filter(|_| cfg.tie_word_embeddings && with_lm_head)
        {
            let (w_id, _) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                "model.embed_tokens.weight",
            )?;
            let logits_shape = Shape::new(&[batch, logit_rows, vocab], f);
            g.add_node(
                Op::DequantMatMul { scheme },
                vec![head_input, w_id],
                logits_shape,
            )
        } else {
            let lm_head_w = if cfg.tie_word_embeddings {
                let embed = params
                    .get("model.embed_tokens.weight")
                    .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?;
                let mut transposed = vec![0f32; embed.len()];
                for v in 0..vocab {
                    for hi in 0..h {
                        transposed[hi * vocab + v] = embed[v * h + hi];
                    }
                }
                synth_const(
                    &mut g,
                    &mut params,
                    "gemma.packed.lm_head.tied_t",
                    transposed,
                    &[h, vocab],
                )
            } else {
                load_p(&mut g, &mut params, weights, "lm_head.weight", true)?
            };
            g.mm(head_input, lm_head_w)
        };
        // Final logit soft-cap: tanh(x / cap) * cap.
        if let Some(cap) = cfg.final_logit_softcapping {
            let inv = synth_const(
                &mut g,
                &mut params,
                &format!("gemma.packed.softcap.inv.{cap}"),
                vec![1.0 / cap],
                &[1],
            );
            let cap_id = synth_const(
                &mut g,
                &mut params,
                &format!("gemma.packed.softcap.cap.{cap}"),
                vec![cap],
                &[1],
            );
            let scaled = g.mul(logits, inv);
            let scaled_shape = g.shape(scaled).clone();
            let t = g.add_node(Op::Activation(Activation::Tanh), vec![scaled], scaled_shape);
            logits = g.mul(t, cap_id);
        }
        let _ = logit_rows;
        logits
    } else {
        hidden
    };

    let mut outputs = vec![out];
    if with_kv_outputs {
        for (k, v) in kv_outputs {
            outputs.push(k);
            outputs.push(v);
        }
    }
    // Append layer-0 taps last so existing kv-output indexing isn't disturbed.
    // packed_session reads outputs in [logits, ...kv, ...taps] order.
    if tap_l0 {
        outputs.extend(l0_taps.iter().copied());
        eprintln!(
            "[rlx-gemma] RLX_TAP_L0: appended {} layer-0 taps as graph outputs",
            l0_taps.len()
        );
    }
    if tap_all {
        outputs.extend(all_layer_taps.iter().copied());
        eprintln!(
            "[rlx-gemma] RLX_TAP_ALL: appended {} per-layer hidden taps",
            all_layer_taps.len()
        );
    }
    g.set_outputs(outputs);
    Ok((g, params))
}

/// Single-token decode graph with packed `Op::DequantMatMul` projections.
///
/// Expects runtime `rope_cos` / `rope_sin` rows (and optional global rows for
/// Gemma 4 full-attention layers). When `use_custom_mask` is true, supply a
/// bucketed mask of length `past_seq + 1`.
///
/// Precompute tied lm_head transpose once for packed decode bucket builds.
pub fn precompute_packed_decode_tied_lm_head(cfg: &GemmaConfig, embed: &[f32]) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    if embed.len() != vocab * h {
        return Err(anyhow!(
            "embed_tokens.weight len {} != vocab*hidden ({vocab}*{h})",
            embed.len()
        ));
    }
    let mut transposed = vec![0f32; embed.len()];
    for v in 0..vocab {
        for hi in 0..h {
            transposed[hi * vocab + v] = embed[v * h + hi];
        }
    }
    Ok(transposed)
}

#[allow(clippy::too_many_arguments)]
pub fn build_gemma_decode_graph_sized_packed(
    cfg: &GemmaConfig,
    weights: &mut dyn rlx_core::weight_loader::WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
    packed: &mut PackedWeightMap,
) -> Result<(Graph, F32WeightMap)> {
    build_gemma_decode_graph_sized_packed_ext(
        cfg,
        weights,
        batch,
        past_seq,
        use_custom_mask,
        packed,
        None,
        None,
    )
}

/// Like [`build_gemma_decode_graph_sized_packed`] but reuses cached packed/F32 tensors
/// (avoids GGUF reload and tied-lm_head re-transpose on each decode bucket).
#[allow(clippy::too_many_arguments)]
pub fn build_gemma_decode_graph_sized_packed_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn rlx_core::weight_loader::WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
    packed: &mut PackedWeightMap,
    known_packed: Option<&PackedWeightMap>,
    known_f32: Option<&F32WeightMap>,
) -> Result<(Graph, F32WeightMap)> {
    use crate::config::GemmaArch;
    use crate::rope::resolve_inv_freq;
    use rlx_core::weight_loader::WeightLoader;
    use rlx_ir::op::{Activation, Op};
    use rlx_ir::quant::QuantScheme;
    use rlx_ir::{DType, NodeId, Shape};

    validate_cfg(cfg)?;
    if batch != 1 {
        return Err(anyhow!("gemma packed decode requires batch=1"));
    }
    let seq = 1usize;

    let mut g = Graph::new("gemma_packed_decode");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let eps = cfg.rms_norm_eps as f32;
    let num_layers = cfg.active_num_layers();

    fn synth_const(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        name: &str,
        data: Vec<f32>,
        shape: &[usize],
    ) -> NodeId {
        let id = g.param(name, Shape::new(shape, DType::F32));
        params.insert(name.to_string(), data);
        id
    }
    fn load_p_cached(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        weights: &mut dyn WeightLoader,
        known_f32: Option<&HashMap<String, Vec<f32>>>,
        key: &str,
        shape: &[usize],
        transpose: bool,
    ) -> Result<NodeId> {
        let (data, out_shape) = if let Some(cached) = known_f32.and_then(|m| m.get(key)) {
            if transpose {
                let rows = shape[0];
                let cols = shape[1];
                let mut t = vec![0f32; cached.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        t[c * rows + r] = cached[r * cols + c];
                    }
                }
                (t, vec![cols, rows])
            } else {
                (cached.clone(), shape.to_vec())
            }
        } else if transpose {
            weights.take_transposed(key)?
        } else {
            weights.take(key)?
        };
        let id = g.param(key, Shape::new(&out_shape, DType::F32));
        params.insert(key.to_string(), data);
        Ok(id)
    }
    fn load_proj(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        packed: &mut PackedWeightMap,
        weights: &mut dyn WeightLoader,
        known_packed: Option<&PackedWeightMap>,
        known_f32: Option<&F32WeightMap>,
        key: &str,
    ) -> Result<(NodeId, Option<QuantScheme>)> {
        if let Some((bytes, scheme, shape)) = known_packed.and_then(|m| m.get(key)) {
            if bytes.is_empty() {
                let cached = known_f32
                    .and_then(|m| m.get(key))
                    .ok_or_else(|| anyhow::anyhow!("f32 cache miss for drained proj {key}"))?;
                let id = g.param(key, Shape::new(shape, DType::F32));
                params.insert(key.to_string(), cached.clone());
                return Ok((id, None));
            }
            let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
            return Ok((id, Some(*scheme)));
        }
        if let Some((bytes, scheme, shape)) = weights.take_packed(key)? {
            let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
            packed.insert(key.to_string(), (bytes, scheme, shape));
            Ok((id, Some(scheme)))
        } else {
            let (data, shape) = weights.take_transposed(key)?;
            let id = g.param(key, Shape::new(&shape, DType::F32));
            params.insert(key.to_string(), data);
            // Sentinel: proj was materialized to F32 on drain (e.g. Q4_0); rebuild from cache.
            packed.insert(key.to_string(), (Vec::new(), QuantScheme::GgufQ4_0, shape));
            Ok((id, None))
        }
    }
    fn emit_proj(
        g: &mut Graph,
        input: NodeId,
        w: NodeId,
        scheme: Option<QuantScheme>,
        out_shape: Shape,
    ) -> NodeId {
        match scheme {
            Some(s) => g.add_node(Op::DequantMatMul { scheme: s }, vec![input, w], out_shape),
            None => g.mm(input, w),
        }
    }
    fn gemma_rms(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        x: NodeId,
        weight_key: &str,
        weights: &mut dyn WeightLoader,
        known_f32: Option<&HashMap<String, Vec<f32>>>,
        _zero_beta: NodeId,
        h: usize,
        eps: f32,
    ) -> Result<NodeId> {
        let w = load_p_cached(g, params, weights, known_f32, weight_key, &[h], false)?;
        let ones = synth_const(
            g,
            params,
            &format!("{weight_key}.ones"),
            vec![1.0f32; h],
            &[h],
        );
        let gamma = g.add(ones, w);
        // Build a beta sized to `h` rather than reusing the caller's
        // hidden-sized zero_beta — otherwise Q/K norm (h=head_dim) gets
        // a beta of size hidden_size and the RmsNorm op's gamma/beta
        // shapes disagree, propagating NaN through the network. This
        // was the root cause for task #50's persistent all-NaN logits.
        let beta = synth_const(
            g,
            params,
            &format!("{weight_key}.beta"),
            vec![0.0f32; h],
            &[h],
        );
        Ok(g.rms_norm(x, gamma, beta, eps))
    }

    let zero_beta = synth_const(
        &mut g,
        &mut params,
        "gemma.packed.decode.zero_beta",
        vec![0.0f32; h],
        &[h],
    );

    let inv_freq = resolve_inv_freq(cfg, None);
    let half = inv_freq.len();
    let rope_cos = g.input("rope_cos", Shape::new(&[1, half], f));
    let rope_sin = g.input("rope_sin", Shape::new(&[1, half], f));

    let global_rope = crate::rope::resolve_global_inv_freq(cfg, None);
    let (global_cos_in, global_sin_in) = if let Some(global_inv) = global_rope {
        let half_g = global_inv.len();
        (
            Some(g.input("rope_cos_global", Shape::new(&[1, half_g], f))),
            Some(g.input("rope_sin_global", Shape::new(&[1, half_g], f))),
        )
    } else {
        (None, None)
    };

    let mask_id = if use_custom_mask {
        Some(g.input("mask", Shape::new(&[batch, past_seq + seq], f)))
    } else {
        None
    };

    let vocab = cfg.vocab_size;
    // Task #37: lazy-embed path — runtime host-gathers the single decode
    // token's row from packed Q4K bytes and feeds it as `input_embeddings`.
    // Skips the per-decode-bucket embed param upload (≈ 540 MB Q4K for 12B).
    let embed_lazy = known_packed
        .map(|m| m.contains_key("model.embed_tokens.weight"))
        .unwrap_or(false);
    let mut h_id = if embed_lazy {
        g.input("input_embeddings", Shape::new(&[batch, seq, h], DType::F32))
    } else {
        let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
        let embed_w = load_p_cached(
            &mut g,
            &mut params,
            weights,
            known_f32,
            "model.embed_tokens.weight",
            &[vocab, h],
            false,
        )?;
        g.gather_(embed_w, input_ids, 0)
    };
    let scale_val = (h as f32).sqrt();
    let embed_scale = synth_const(
        &mut g,
        &mut params,
        "gemma.packed.decode.embed_scale",
        vec![scale_val],
        &[1],
    );
    h_id = g.mul(h_id, embed_scale);

    let attn_score_scale = cfg.attn_score_scale();
    let attn_softcap = cfg.attn_logit_softcapping;
    let mut new_kv_outputs: Vec<(NodeId, NodeId)> = Vec::with_capacity(num_layers);

    // Gemma 4 E2B decode: Per-Layer-Embedding inputs for the current token
    // (seq=1 in decode), fed by the runner via `compute_per_layer_inputs`; and
    // the shared-KV store (last fresh same-type layer's post-cache K/V). Both
    // empty for non-E2B configs, leaving the flagship/legacy decode untouched.
    let per_layer_inputs_dec = if cfg.has_ple() {
        Some(g.input(
            "per_layer_inputs",
            Shape::new(&[batch, seq, num_layers * cfg.ple_width()], f),
        ))
    } else {
        None
    };
    let mut shared_k_dec: [Option<NodeId>; 2] = [None, None];
    let mut shared_v_dec: [Option<NodeId>; 2] = [None, None];

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        let layer_dh = cfg.layer_head_dim(layer);
        let layer_kv = cfg.layer_num_kv_heads(layer);
        let layer_nrot = cfg.layer_n_rot(layer);
        let q_dim = nh * layer_dh;
        let kv_dim = layer_kv * layer_dh;
        let group = nh / layer_kv;
        // Gemma 4 E2B: KV-shared layers use a double-wide MLP
        // (intermediate_size × 2). Non-E2B configs return the base size,
        // leaving flagship/legacy behavior unchanged.
        let int_dim = cfg.layer_intermediate_size(layer);
        let is_shared_kv = cfg.is_kv_shared_layer(layer);
        let kv_type_idx = cfg.is_full_attention_layer(layer) as usize;

        let past_k = g.input(
            format!("past_k_{layer}"),
            Shape::new(&[batch, past_seq, kv_dim], f),
        );
        let past_v = g.input(
            format!("past_v_{layer}"),
            Shape::new(&[batch, past_seq, kv_dim], f),
        );

        let normed_in = gemma_rms(
            &mut g,
            &mut params,
            h_id,
            &format!("{lp}.input_layernorm.weight"),
            weights,
            known_f32,
            zero_beta,
            h,
            eps,
        )?;

        let fused_qkv_key = format!("{lp}.self_attn.qkv.weight");
        let has_fused_qkv = known_packed
            .map(|m| m.contains_key(&fused_qkv_key))
            .unwrap_or(false);
        let (q, k, v) = if has_fused_qkv {
            let (qkv_w, qkv_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &fused_qkv_key,
            )?;
            let v_present = !cfg.layer_k_eq_v(layer);
            let total_n = q_dim + kv_dim + if v_present { kv_dim } else { 0 };
            let combined = emit_proj(
                &mut g,
                normed_in,
                qkv_w,
                qkv_s,
                Shape::new(&[batch, seq, total_n], f),
            );
            let q = g.narrow_(combined, 2, 0, q_dim);
            let k = g.narrow_(combined, 2, q_dim, kv_dim);
            let v = if v_present {
                g.narrow_(combined, 2, q_dim + kv_dim, kv_dim)
            } else {
                k
            };
            (q, k, v)
        } else {
            let (q_w, q_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.self_attn.q_proj.weight"),
            )?;
            let q = emit_proj(
                &mut g,
                normed_in,
                q_w,
                q_s,
                Shape::new(&[batch, seq, q_dim], f),
            );
            // KV-shared layers (Gemma 4 E2B) reuse an earlier same-type layer's
            // cached K/V and ship no usable k_norm/k_proj/v_proj — skip them; the
            // real K/V are substituted after RoPE below. `q` is a placeholder.
            let (k, v) = if is_shared_kv {
                (q, q)
            } else {
                let (k_w, k_s) = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    known_packed,
                    known_f32,
                    &format!("{lp}.self_attn.k_proj.weight"),
                )?;
                let k = emit_proj(
                    &mut g,
                    normed_in,
                    k_w,
                    k_s,
                    Shape::new(&[batch, seq, kv_dim], f),
                );
                let v = if cfg.layer_k_eq_v(layer) {
                    k
                } else {
                    let (v_w, v_s) = load_proj(
                        &mut g,
                        &mut params,
                        packed,
                        weights,
                        known_packed,
                        known_f32,
                        &format!("{lp}.self_attn.v_proj.weight"),
                    )?;
                    emit_proj(
                        &mut g,
                        normed_in,
                        v_w,
                        v_s,
                        Shape::new(&[batch, seq, kv_dim], f),
                    )
                };
                (k, v)
            };
            (q, k, v)
        };

        // Gemma 4 specific per-head Q/K RMS norms + plain V RMS norm.
        // Tensors:
        //   blk.N.attn_q_norm.weight [head_dim]
        //   blk.N.attn_k_norm.weight [head_dim]
        //   (V uses ggml_rms_norm with NO learnable weight)
        // Applied AFTER q/k/v projection, BEFORE RoPE. Without them,
        // Q·K·V grows unbounded → softmax overflow → NaN at layer 0.
        // V-norm matches llama.cpp gemma4.cpp:256 — plain RMSNorm with
        // gamma=1, beta=0 (just normalizes V to unit-RMS per head).
        let (q, k, v) = if matches!(cfg.arch, GemmaArch::Gemma4) {
            let q_norm_key = format!("{lp}.self_attn.q_norm.weight");
            let k_norm_key = format!("{lp}.self_attn.k_norm.weight");
            // q: [B, S, nh*dh] → [B, S, nh, dh] for per-head norm.
            let q_4d = g.reshape_(
                q,
                vec![batch as i64, seq as i64, nh as i64, layer_dh as i64],
            );
            let q_normed = gemma_rms(
                &mut g,
                &mut params,
                q_4d,
                &q_norm_key,
                weights,
                known_f32,
                zero_beta,
                layer_dh,
                eps,
            )?;
            let q = g.reshape_(q_normed, vec![batch as i64, seq as i64, q_dim as i64]);

            // Shared-KV layers carry no k_norm/v_norm and reuse cached K/V; skip.
            let (k, v) = if is_shared_kv {
                (k, v)
            } else {
                let k_4d = g.reshape_(
                    k,
                    vec![batch as i64, seq as i64, layer_kv as i64, layer_dh as i64],
                );
                let k_normed = gemma_rms(
                    &mut g,
                    &mut params,
                    k_4d,
                    &k_norm_key,
                    weights,
                    known_f32,
                    zero_beta,
                    layer_dh,
                    eps,
                )?;
                let k = g.reshape_(k_normed, vec![batch as i64, seq as i64, kv_dim as i64]);
                // V RMS-norm with no learnable scale — matches llama.cpp
                // gemma4.cpp:256 `ggml_rms_norm(Vcur, f_norm_rms_eps)`.
                let v_4d = g.reshape_(
                    v,
                    vec![batch as i64, seq as i64, layer_kv as i64, layer_dh as i64],
                );
                let v_ones = synth_const(
                    &mut g,
                    &mut params,
                    &format!("{lp}.self_attn.v_norm.ones"),
                    vec![1.0f32; layer_dh],
                    &[layer_dh],
                );
                let v_zeros = synth_const(
                    &mut g,
                    &mut params,
                    &format!("{lp}.self_attn.v_norm.zeros"),
                    vec![0.0f32; layer_dh],
                    &[layer_dh],
                );
                let v_normed = g.rms_norm(v_4d, v_ones, v_zeros, eps);
                let v = g.reshape_(v_normed, vec![batch as i64, seq as i64, kv_dim as i64]);
                (k, v)
            };
            (q, k, v)
        } else {
            (q, k, v)
        };

        let (layer_cos, layer_sin) = if cfg.is_full_attention_layer(layer) {
            match (global_cos_in, global_sin_in) {
                (Some(gc), Some(gs)) => (gc, gs),
                _ => (rope_cos, rope_sin),
            }
        } else {
            (rope_cos, rope_sin)
        };
        let q_rope = g.rope_n(q, layer_cos, layer_sin, layer_dh, layer_nrot);
        // E2B KV sharing: a shared layer reuses the last fresh same-type layer's
        // full (post-cache) K/V — no own RoPE/concat. Its `past_k_{L}`/cache slot
        // still exists (kept as a redundant copy so the cache stays uniform and
        // packed_session needs no per-layer special-casing) but is unused here.
        // Fresh layers RoPE + concat their own K and, when sharing is active,
        // store the result for later same-type shared layers.
        let (new_k, new_v) = if is_shared_kv {
            let sk = shared_k_dec[kv_type_idx].ok_or_else(|| {
                anyhow!("decode KV-shared layer {layer}: no source K for type {kv_type_idx}")
            })?;
            let sv = shared_v_dec[kv_type_idx]
                .ok_or_else(|| anyhow!("decode KV-shared layer {layer}: no source V"))?;
            (sk, sv)
        } else {
            let k_rope = g.rope_n(k, layer_cos, layer_sin, layer_dh, layer_nrot);
            let nk = g.concat_(vec![past_k, k_rope], 1);
            let nv = g.concat_(vec![past_v, v], 1);
            if cfg.num_kv_shared_layers > 0 {
                shared_k_dec[kv_type_idx] = Some(nk);
                shared_v_dec[kv_type_idx] = Some(nv);
            }
            (nk, nv)
        };
        new_kv_outputs.push((new_k, new_v));

        let k_rep = repeat_kv_packed(&mut g, new_k, layer_kv, layer_dh, group);
        let v_rep = repeat_kv_packed(&mut g, new_v, layer_kv, layer_dh, group);

        let attn = if let Some(mask) = mask_id {
            g.attention_(q_rope, k_rep, v_rep, mask, nh, layer_dh)
        } else {
            let (mask_kind, _, _) = cfg.layer_attn_options(layer);
            let attn_shape = rlx_ir::shape::attention_shape(g.shape(q_rope));
            g.attention_kind_opts(
                q_rope,
                k_rep,
                v_rep,
                nh,
                layer_dh,
                mask_kind,
                attn_shape,
                attn_score_scale,
                attn_softcap,
            )
        };

        let (o_w, o_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, o_w, o_s, Shape::new(&[batch, seq, h], f));
        // Gemma 3/4 sandwich-norm: post_attention_layernorm applied to
        // attn_out BEFORE the residual add. Without this the attention
        // contribution grows unbounded across 48 layers — task #50 NaN
        // root cause. Maps GGUF blk.N.post_attention_norm.weight via
        // gguf_to_hf_name_for_arch.
        let attn_out = if matches!(cfg.arch, GemmaArch::Gemma3 | GemmaArch::Gemma4) {
            gemma_rms(
                &mut g,
                &mut params,
                attn_out,
                &format!("{lp}.post_attention_layernorm.weight"),
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?
        } else {
            attn_out
        };
        let post_attn = g.add(h_id, attn_out);

        let pre_ffn_key = if cfg.arch == GemmaArch::Gemma {
            format!("{lp}.post_attention_layernorm.weight")
        } else {
            format!("{lp}.pre_feedforward_layernorm.weight")
        };
        let normed_post = gemma_rms(
            &mut g,
            &mut params,
            post_attn,
            &pre_ffn_key,
            weights,
            known_f32,
            zero_beta,
            h,
            eps,
        )?;

        // Try fused {gate_up}_proj first — set up at drain time by
        // `drain_gemma_packed_weights`. One matmul + narrow halves
        // replaces two matmuls on the same input.
        let fused_gate_up_key = format!("{lp}.mlp.gate_up.weight");
        let fused_gate_up = known_packed
            .map(|m| m.contains_key(&fused_gate_up_key))
            .unwrap_or(false);
        let (gate, up) = if fused_gate_up {
            let (gu_w, gu_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &fused_gate_up_key,
            )?;
            let combined = emit_proj(
                &mut g,
                normed_post,
                gu_w,
                gu_s,
                Shape::new(&[batch, seq, int_dim * 2], f),
            );
            let gate = g.narrow_(combined, 2, 0, int_dim);
            let up = g.narrow_(combined, 2, int_dim, int_dim);
            (gate, up)
        } else {
            let (gate_w, gate_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.mlp.gate_proj.weight"),
            )?;
            let (up_w, up_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.mlp.up_proj.weight"),
            )?;
            let gate = emit_proj(
                &mut g,
                normed_post,
                gate_w,
                gate_s,
                Shape::new(&[batch, seq, int_dim], f),
            );
            let up = emit_proj(
                &mut g,
                normed_post,
                up_w,
                up_s,
                Shape::new(&[batch, seq, int_dim], f),
            );
            (gate, up)
        };
        let (down_w, down_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.mlp.down_proj.weight"),
        )?;
        let gate_act = g.gelu_approx(gate);
        let mlp_inner = g.mul(gate_act, up);
        let mut ffn_out = emit_proj(
            &mut g,
            mlp_inner,
            down_w,
            down_s,
            Shape::new(&[batch, seq, h], f),
        );

        if cfg.arch != GemmaArch::Gemma {
            let post_ffn_key = format!("{lp}.post_feedforward_layernorm.weight");
            ffn_out = gemma_rms(
                &mut g,
                &mut params,
                ffn_out,
                &post_ffn_key,
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?;
        }

        let mut layer_out = g.add(post_attn, ffn_out);

        // Gemma 4 E2B Per-Layer Embeddings: inject the current token's per-layer
        // slice after the MLP residual (mirrors prefill / HF decoder forward).
        if let Some(ple_all) = per_layer_inputs_dec {
            let pw = cfg.ple_width();
            let ple_slice = g.narrow_(ple_all, 2, layer * pw, pw);
            let (gate_w, gate_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.per_layer_input_gate.weight"),
            )?;
            let gated = emit_proj(
                &mut g,
                layer_out,
                gate_w,
                gate_s,
                Shape::new(&[batch, seq, pw], f),
            );
            let gated = g.gelu_approx(gated);
            let gated = g.mul(gated, ple_slice);
            let (proj_w, proj_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.per_layer_projection.weight"),
            )?;
            let projected = emit_proj(
                &mut g,
                gated,
                proj_w,
                proj_s,
                Shape::new(&[batch, seq, h], f),
            );
            let projected = gemma_rms(
                &mut g,
                &mut params,
                projected,
                &format!("{lp}.post_per_layer_input_norm.weight"),
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?;
            layer_out = g.add(layer_out, projected);
        }

        // Per-layer output scalar that multiplies the combined layer output
        // before it becomes the next layer's residual (without it the stream
        // grows unbounded → inf/NaN at the LM head). Gemma 4 E2B ships this as
        // `layer_scalar`; the flagship GGUF path uses `self_attn.output_scale`.
        h_id = if cfg.has_ple() {
            let ls = load_p_cached(
                &mut g,
                &mut params,
                weights,
                known_f32,
                &format!("{lp}.layer_scalar"),
                &[1],
                false,
            )?;
            g.mul(layer_out, ls)
        } else if matches!(cfg.arch, GemmaArch::Gemma4) {
            let scale_w = load_p_cached(
                &mut g,
                &mut params,
                weights,
                known_f32,
                &format!("{lp}.self_attn.output_scale.weight"),
                &[1],
                false,
            )?;
            g.mul(layer_out, scale_w)
        } else {
            layer_out
        };
    }

    let hidden = gemma_rms(
        &mut g,
        &mut params,
        h_id,
        "model.norm.weight",
        weights,
        known_f32,
        zero_beta,
        h,
        eps,
    )?;

    const TIED_LM_HEAD: &str = "gemma.packed.decode.lm_head.tied_t";
    // Task #36: prefer packed DequantMatMul on the original Q4K-packed embed
    // bytes — skips the ~4 GB transposed-f32 constant per decode bucket
    // (≈ 4 GB × num_decode_buckets if recompiled per bucket).
    let packed_embed_scheme = known_packed
        .and_then(|m| m.get("model.embed_tokens.weight"))
        .map(|(_, scheme, _)| *scheme);
    let mut logits = if let Some(scheme) = packed_embed_scheme.filter(|_| cfg.tie_word_embeddings) {
        let (w_id, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            "model.embed_tokens.weight",
        )?;
        let logits_shape = Shape::new(&[batch, seq, vocab], f);
        g.add_node(
            Op::DequantMatMul { scheme },
            vec![hidden, w_id],
            logits_shape,
        )
    } else {
        let lm_head_w = if cfg.tie_word_embeddings {
            if let Some(tied) = known_f32.and_then(|m| m.get(TIED_LM_HEAD)) {
                synth_const(&mut g, &mut params, TIED_LM_HEAD, tied.clone(), &[h, vocab])
            } else {
                let embed = params
                    .get("model.embed_tokens.weight")
                    .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?
                    .clone();
                synth_const(
                    &mut g,
                    &mut params,
                    TIED_LM_HEAD,
                    precompute_packed_decode_tied_lm_head(cfg, &embed)?,
                    &[h, vocab],
                )
            }
        } else {
            load_p_cached(
                &mut g,
                &mut params,
                weights,
                known_f32,
                "lm_head.weight",
                &[vocab, h],
                true,
            )?
        };
        g.mm(hidden, lm_head_w)
    };
    if let Some(cap) = cfg.final_logit_softcapping {
        let inv = synth_const(
            &mut g,
            &mut params,
            &format!("gemma.packed.decode.softcap.inv.{cap}"),
            vec![1.0 / cap],
            &[1],
        );
        let cap_id = synth_const(
            &mut g,
            &mut params,
            &format!("gemma.packed.decode.softcap.cap.{cap}"),
            vec![cap],
            &[1],
        );
        let scaled = g.mul(logits, inv);
        let scaled_shape = g.shape(scaled).clone();
        let t = g.add_node(Op::Activation(Activation::Tanh), vec![scaled], scaled_shape);
        logits = g.mul(t, cap_id);
    }

    let mut outputs = vec![logits];
    for (k, v) in new_kv_outputs {
        outputs.push(k);
        outputs.push(v);
    }
    g.set_outputs(outputs);
    Ok((g, params))
}

fn repeat_kv_packed(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> rlx_ir::NodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces: Vec<rlx_ir::NodeId> = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

fn validate_cfg(cfg: &GemmaConfig) -> Result<()> {
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
        return Err(anyhow!("attention_bias=true not yet wired for gemma"));
    }
    Ok(())
}

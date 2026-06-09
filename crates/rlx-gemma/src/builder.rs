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
    use crate::rope::{build_rope_tables, resolve_global_inv_freq, resolve_inv_freq};
    use rlx_core::weight_map::{WeightDrainPolicy, WeightMap};

    let rope_factors = loader.take("rope_freqs.weight").ok().map(|(d, _)| d);
    let inv_freq = resolve_inv_freq(cfg, rope_factors.as_deref());
    let (cos_data, sin_data) = build_rope_tables(&inv_freq, cfg.max_position_embeddings);

    let arch = loader.arch_hint().unwrap_or("gemma").to_string();

    let mut f32_params: HashMap<String, Vec<f32>> = HashMap::new();
    if let Ok((data, _shape)) = loader.take("model.embed_tokens.weight") {
        f32_params.insert("model.embed_tokens.weight".into(), data);
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
        let (gcd, gsd) = build_rope_tables(&global_inv, cfg.max_position_embeddings);
        f32_params.insert("rope.global.cos".into(), gcd);
        f32_params.insert("rope.global.sin".into(), gsd);
    }

    let mut packed = HashMap::new();
    for (key, bytes, scheme, shape) in packed_list {
        let canonical = rlx_core::weight_loader::gguf_to_hf_name_for_arch(&key, &arch)
            .unwrap_or_else(|| key.clone());
        packed.insert(canonical, (bytes, scheme, shape));
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
    let int_dim = cfg.intermediate_size;
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
        zero_beta: NodeId,
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
        Ok(g.rms_norm(x, gamma, zero_beta, eps))
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
        let half_g = if cos.len() >= rope_len && cfg.max_position_embeddings > 0 {
            cos.len() / cfg.max_position_embeddings
        } else {
            half
        };
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

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
    let last_token_idx = if with_lm_head && last_token_from_input {
        Some(g.input("last_token_idx", Shape::new(&[batch], DType::F32)))
    } else {
        None
    };
    let vocab = cfg.vocab_size;
    let embed_w = load_p_cached(
        &mut g,
        &mut params,
        weights,
        known_f32,
        "model.embed_tokens.weight",
        &[vocab, h],
        false,
    )?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);
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

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        let layer_dh = cfg.layer_head_dim(layer);
        let layer_kv = cfg.layer_num_kv_heads(layer);
        let layer_nrot = cfg.layer_n_rot(layer);
        let q_dim = nh * layer_dh;
        let kv_dim = layer_kv * layer_dh;
        let group = nh / layer_kv;

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

        // Q/K/V projections. v_proj is skipped when k_eq_v.
        let (q_w, q_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        let (k_w, k_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.self_attn.k_proj.weight"),
        )?;
        let q = emit_proj(
            &mut g,
            normed_in,
            q_w,
            q_s,
            Shape::new(&[batch, seq, q_dim], f),
        );
        let k = emit_proj(
            &mut g,
            normed_in,
            k_w,
            k_s,
            Shape::new(&[batch, seq, kv_dim], f),
        );
        let v = if cfg.attention_k_eq_v {
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
        let k_rope = g.rope_n(k, layer_cos, layer_sin, layer_dh, layer_nrot);
        if with_kv_outputs {
            kv_outputs.push((k_rope, v));
        }

        let k_rep = repeat_kv_packed(&mut g, k_rope, layer_kv, layer_dh, group);
        let v_rep = repeat_kv_packed(&mut g, v, layer_kv, layer_dh, group);

        // Per-layer mask.
        let (mask_kind, _, _) = cfg.layer_attn_options(layer);
        let attn_shape = rlx_ir::shape::attention_shape(g.shape(q_rope));
        let attn = g.attention_kind_opts(
            q_rope,
            k_rep,
            v_rep,
            nh,
            layer_dh,
            mask_kind,
            attn_shape,
            attn_score_scale,
            attn_softcap,
        );

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
        let post_attn = g.add(h_id, attn_out);

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

        // GeGLU MLP.
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
        let (down_w, down_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.mlp.down_proj.weight"),
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
        let gate_act = g.gelu_approx(gate);
        let mlp_inner = g.mul(gate_act, up);
        let mut ffn_out = emit_proj(
            &mut g,
            mlp_inner,
            down_w,
            down_s,
            Shape::new(&[batch, seq, h], f),
        );

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
        let lm_head_w = if cfg.tie_word_embeddings {
            let embed = params
                .get("model.embed_tokens.weight")
                .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?;
            let vocab = cfg.vocab_size;
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
        let mut logits = g.mm(head_input, lm_head_w);
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
    let int_dim = cfg.intermediate_size;
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
        zero_beta: NodeId,
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
        Ok(g.rms_norm(x, gamma, zero_beta, eps))
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

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
    let vocab = cfg.vocab_size;
    let embed_w = load_p_cached(
        &mut g,
        &mut params,
        weights,
        known_f32,
        "model.embed_tokens.weight",
        &[vocab, h],
        false,
    )?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);
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

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        let layer_dh = cfg.layer_head_dim(layer);
        let layer_kv = cfg.layer_num_kv_heads(layer);
        let layer_nrot = cfg.layer_n_rot(layer);
        let q_dim = nh * layer_dh;
        let kv_dim = layer_kv * layer_dh;
        let group = nh / layer_kv;

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

        let (q_w, q_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        let (k_w, k_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.self_attn.k_proj.weight"),
        )?;
        let q = emit_proj(
            &mut g,
            normed_in,
            q_w,
            q_s,
            Shape::new(&[batch, seq, q_dim], f),
        );
        let k = emit_proj(
            &mut g,
            normed_in,
            k_w,
            k_s,
            Shape::new(&[batch, seq, kv_dim], f),
        );
        let v = if cfg.attention_k_eq_v {
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

        let (layer_cos, layer_sin) = if cfg.is_full_attention_layer(layer) {
            match (global_cos_in, global_sin_in) {
                (Some(gc), Some(gs)) => (gc, gs),
                _ => (rope_cos, rope_sin),
            }
        } else {
            (rope_cos, rope_sin)
        };
        let q_rope = g.rope_n(q, layer_cos, layer_sin, layer_dh, layer_nrot);
        let k_rope = g.rope_n(k, layer_cos, layer_sin, layer_dh, layer_nrot);

        let new_k = g.concat_(vec![past_k, k_rope], 1);
        let new_v = g.concat_(vec![past_v, v], 1);
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
        let (down_w, down_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.mlp.down_proj.weight"),
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

        h_id = g.add(post_attn, ffn_out);
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
    let mut logits = g.mm(hidden, lm_head_w);
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

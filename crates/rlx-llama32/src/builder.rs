// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// LLaMA-3.2 graph builder — GQA + RoPE + SwiGLU, no QK-norm.

use crate::config::Llama32Config;
use crate::rope::{build_rope_tables, resolve_inv_freq};
use anyhow::{Result, anyhow};
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::shape::{self};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

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

/// Packed-weights prefill graph — K-quant matmuls stay in the arena via
/// `Op::DequantMatMul` (mirrors [`rlx_qwen3::build_qwen3_graph_sized_packed`]).
#[allow(clippy::too_many_arguments)]
pub fn build_llama32_graph_sized_packed(
    cfg: &Llama32Config,
    weights: &mut rlx_core::weight_loader::GgufLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    use rlx_ir::quant::QuantScheme;

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
    let (cos_data, sin_data) = build_rope_tables(&inv_freq, cfg.max_position_embeddings);
    let half = inv_freq.len();
    let cos_id = g.param(
        "rope.cos",
        Shape::new(&[cfg.max_position_embeddings, half], f),
    );
    params.insert("rope.cos".into(), cos_data);
    let sin_id = g.param(
        "rope.sin",
        Shape::new(&[cfg.max_position_embeddings, half], f),
    );
    params.insert("rope.sin".into(), sin_data);

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
    let embed_w = load_p(
        &mut g,
        &mut params,
        weights,
        "model.embed_tokens.weight",
        false,
    )?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    fn load_proj(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
        weights: &mut rlx_core::weight_loader::GgufLoader,
        key: &str,
    ) -> Result<(NodeId, Option<QuantScheme>, Vec<usize>)> {
        if let Some((bytes, scheme, shape)) = weights.take_packed(key)? {
            let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
            packed.insert(key.to_string(), (bytes, scheme, shape.clone()));
            Ok((id, Some(scheme), shape))
        } else {
            let nid = load_p(g, params, weights, key, true)?;
            Ok((nid, None, Vec::new()))
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

        let q_dim = nh * dh;
        let kv_dim = nkv * dh;
        let (q_w, q_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        let (k_w, k_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.k_proj.weight"),
        )?;
        let (v_w, v_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.v_proj.weight"),
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
        let v = emit_proj(
            &mut g,
            normed_in,
            v_w,
            v_s,
            Shape::new(&[batch, seq, kv_dim], f),
        );

        let q_rope = g.rope(q, cos_id, sin_id, dh);
        let k_rope = g.rope(k, cos_id, sin_id, dh);

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

        let inter = cfg.intermediate_size;
        let (gate_w, gate_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.gate_proj.weight"),
        )?;
        let (up_w, up_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.up_proj.weight"),
        )?;
        let (down_w, down_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.down_proj.weight"),
        )?;
        let gate = emit_proj(
            &mut g,
            normed_post,
            gate_w,
            gate_s,
            Shape::new(&[batch, seq, inter], f),
        );
        let up = emit_proj(
            &mut g,
            normed_post,
            up_w,
            up_s,
            Shape::new(&[batch, seq, inter], f),
        );
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
            g.narrow_(hidden, 1, seq - 1, 1)
        } else {
            hidden
        };
        let (lm_head_w, lm_head_scheme) = if cfg.tie_word_embeddings {
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

    g.set_outputs(vec![out]);
    Ok((g, params))
}

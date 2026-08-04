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

//! Compiled DeepSeek-V2-style MoE LM graphs for Unlimited-OCR.
//!
//! Dense layer 0 (through [`UnlimitedOcrConfig::first_k_dense_replace`]), then
//! routed + ungated shared-expert MoE layers. MHA attention (no bias, no
//! qk_norm), BHSD RoPE for MLX-safe multi-head packing, causal mask on prefill
//! and decode.
//!
//! When [`PackedLmWeights::keeps_quants_in_ir`] is true (Q8_0 / Q4_0 host pack),
//! large mats stay U8 params with `DequantMatMul` / `DequantGroupedMatMul`
//! instead of widening into F32 `MatMul` / `GroupedMatMul` params.

use crate::config::UnlimitedOcrConfig;
use crate::expert_pack::{
    PackedLmWeights, expert_down_exps_key, expert_gate_exps_key, expert_up_exps_key,
};
use crate::nn;
use anyhow::{Context, Result, ensure};
use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::blocks::{LmHeadStage, RopeTablesStage};
use rlx_flow::escape::Emit;
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, FlowValue, ModelFlow, SideOutputs};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::shape;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Op, QuantScheme, Shape};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// `MoEGate.routed_scaling_factor` — HF default when absent from checkpoint.
const ROUTED_SCALING_FACTOR: f32 = 1.0;

/// Collects U8 GGUF blobs for `compile_built` → `set_param_typed`.
#[derive(Clone, Default)]
struct TypedParamSink {
    typed: Arc<Mutex<Vec<(String, Vec<u8>, DType)>>>,
    /// `cache_key` → scheme for packed mats (absent ⇒ F32 MatMul).
    schemes: Arc<Mutex<HashMap<String, QuantScheme>>>,
}

impl TypedParamSink {
    fn new() -> Self {
        Self::default()
    }

    fn drain(self) -> Vec<(String, Vec<u8>, DType)> {
        std::mem::take(&mut *self.typed.lock().expect("typed params"))
    }
}

#[derive(Clone)]
struct LayerSpec {
    layer_idx: usize,
    batch: usize,
    seq: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_group_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    eps: f32,
    hidden_shape: Shape,
    is_dense: bool,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    n_shared_experts: usize,
    keep_packed: bool,
    pack: Option<Arc<PackedLmWeights>>,
    typed: TypedParamSink,
}

struct MatParam {
    id: HirNodeId,
    scheme: Option<QuantScheme>,
}

/// Prefill from fused `inputs_embeds` — last-token logits + per-layer K/V export.
pub fn build_unlimited_ocr_prefill_built(
    cfg: &UnlimitedOcrConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    build_prefill_inner(cfg, weights, None, batch, seq)
}

/// Prefill using a host pack — keeps Q8_0/Q4_0 in IR when applicable.
pub fn build_unlimited_ocr_prefill_built_from_pack(
    cfg: &UnlimitedOcrConfig,
    pack: &Arc<PackedLmWeights>,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    let mut loader = pack.loader();
    let pack_opt = if pack.keeps_quants_in_ir() {
        Some(Arc::clone(pack))
    } else {
        None
    };
    build_prefill_inner(cfg, &mut loader, pack_opt, batch, seq)
}

fn build_prefill_inner(
    cfg: &UnlimitedOcrConfig,
    weights: &mut dyn WeightLoader,
    pack: Option<Arc<PackedLmWeights>>,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    cfg.validate().context("unlimited-ocr prefill config")?;
    validate_heads(cfg)?;

    let keep_packed = pack.is_some();
    let profile = {
        let mut p = CompileProfile::llama32_prefill();
        if keep_packed {
            // Packed GGUF Dequant* — skip residual/RMSNorm fusion (same as
            // `compile_options_for_packed_gguf_*`).
            p.fusion.skip = true;
        }
        p
    };
    let f = DType::F32;
    let h = cfg.hidden_size;
    let dh = cfg.head_dim();
    let eps = cfg.rms_norm_eps as f32;
    let half = dh / 2;
    let typed = TypedParamSink::new();

    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let (cos_data, sin_data) = nn::rope_tables(cfg.max_position_embeddings, dh, cfg.rope_theta);

    let kv_sink = SideOutputs::new();

    let mut flow = ModelFlow::new("unlimited_ocr_prefill")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape.clone())
        .rope_tables(RopeTablesStage::param(
            cfg.max_position_embeddings,
            half,
            cos_data,
            sin_data,
        ))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);

    for layer_idx in 0..cfg.num_hidden_layers {
        let spec = layer_spec(
            cfg,
            layer_idx,
            batch,
            seq,
            eps,
            hidden_shape.clone(),
            keep_packed,
            pack.clone(),
            typed.clone(),
        );
        let sink = kv_sink.clone();
        flow = flow.plugin_named(
            format!("unlimited_ocr.prefill_layer_{layer_idx}"),
            move |emit, hidden| run_layer(emit, hidden, &spec, &sink, false),
        );
    }

    let typed_head = typed.clone();
    let pack_head = pack.clone();
    let vocab = cfg.vocab_size;
    let mut built = if keep_packed {
        flow.gather_last_token_at(batch, seq)
            .final_norm(eps)
            .plugin_named("unlimited_ocr.lm_head", move |emit, hidden| {
                emit_lm_head(
                    emit,
                    hidden,
                    vocab,
                    keep_packed,
                    pack_head.as_ref(),
                    &typed_head,
                )
            })
            .output("logits")
            .build(&mut WeightLoaderSource(weights))?
            .with_extra_hir_outputs(kv_sink.drain())
    } else {
        flow.gather_last_token_at(batch, seq)
            .final_norm(eps)
            .raw_stage(FlowStage::LmHead(LmHeadStage::separate(
                "lm_head.weight",
                cfg.vocab_size,
                h,
            )))
            .output("logits")
            .build(&mut WeightLoaderSource(weights))?
            .with_extra_hir_outputs(kv_sink.drain())
    };

    built.typed_params = typed.drain();
    Ok(built)
}

/// Single-token decode from `inputs_embeds` — logits + updated K/V side outputs.
pub fn build_unlimited_ocr_decode_built(
    cfg: &UnlimitedOcrConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<BuiltModel> {
    build_decode_inner(cfg, weights, None, batch, past_seq)
}

/// Decode using a host pack — keeps Q8_0/Q4_0 in IR when applicable.
pub fn build_unlimited_ocr_decode_built_from_pack(
    cfg: &UnlimitedOcrConfig,
    pack: &Arc<PackedLmWeights>,
    batch: usize,
    past_seq: usize,
) -> Result<BuiltModel> {
    let mut loader = pack.loader();
    let pack_opt = if pack.keeps_quants_in_ir() {
        Some(Arc::clone(pack))
    } else {
        None
    };
    build_decode_inner(cfg, &mut loader, pack_opt, batch, past_seq)
}

fn build_decode_inner(
    cfg: &UnlimitedOcrConfig,
    weights: &mut dyn WeightLoader,
    pack: Option<Arc<PackedLmWeights>>,
    batch: usize,
    past_seq: usize,
) -> Result<BuiltModel> {
    cfg.validate().context("unlimited-ocr decode config")?;
    validate_heads(cfg)?;

    let keep_packed = pack.is_some();
    let profile = {
        let mut p = CompileProfile::llama32_decode();
        if keep_packed {
            p.fusion.skip = true;
        }
        p
    };
    let f = DType::F32;
    let h = cfg.hidden_size;
    let dh = cfg.head_dim();
    let eps = cfg.rms_norm_eps as f32;
    let half = dh / 2;
    let kv_dim = cfg.num_key_value_heads * dh;
    let typed = TypedParamSink::new();

    let hidden_shape = Shape::new(&[batch, 1, h], f);
    let past_kv_shape = Shape::new(&[batch, past_seq, kv_dim], f);

    let kv_sink = SideOutputs::new();

    let mut flow = ModelFlow::new("unlimited_ocr_decode")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape.clone())
        .input("rope_cos", Shape::new(&[1, half], f))
        .input("rope_sin", Shape::new(&[1, half], f));

    for layer_idx in 0..cfg.num_hidden_layers {
        flow = flow
            .input(format!("past_k_{layer_idx}"), past_kv_shape.clone())
            .input(format!("past_v_{layer_idx}"), past_kv_shape.clone());
    }

    flow = flow
        .bind_decode_inputs(cfg.num_hidden_layers, false, true)
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);

    for layer_idx in 0..cfg.num_hidden_layers {
        let spec = layer_spec(
            cfg,
            layer_idx,
            batch,
            1,
            eps,
            hidden_shape.clone(),
            keep_packed,
            pack.clone(),
            typed.clone(),
        );
        let sink = kv_sink.clone();
        flow = flow.plugin_named(
            format!("unlimited_ocr.decode_layer_{layer_idx}"),
            move |emit, hidden| run_layer(emit, hidden, &spec, &sink, true),
        );
    }

    let typed_head = typed.clone();
    let pack_head = pack.clone();
    let vocab = cfg.vocab_size;
    let mut built = if keep_packed {
        flow.final_norm(eps)
            .plugin_named("unlimited_ocr.lm_head", move |emit, hidden| {
                emit_lm_head(
                    emit,
                    hidden,
                    vocab,
                    keep_packed,
                    pack_head.as_ref(),
                    &typed_head,
                )
            })
            .output("logits")
            .build(&mut WeightLoaderSource(weights))?
            .with_extra_hir_outputs(kv_sink.drain())
    } else {
        flow.final_norm(eps)
            .raw_stage(FlowStage::LmHead(LmHeadStage::separate(
                "lm_head.weight",
                cfg.vocab_size,
                h,
            )))
            .output("logits")
            .build(&mut WeightLoaderSource(weights))?
            .with_extra_hir_outputs(kv_sink.drain())
    };

    built.typed_params = typed.drain();
    Ok(built)
}

/// Single-position RoPE cos/sin slice for decode (`[half]`, `[half]`).
pub fn compute_rope_slice(cfg: &UnlimitedOcrConfig, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim();
    let half = dh / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for i in 0..half {
        let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
        let angle = pos as f64 * freq;
        let (s, c) = angle.sin_cos();
        cos[i] = c as f32;
        sin[i] = s as f32;
    }
    (cos, sin)
}

fn layer_spec(
    cfg: &UnlimitedOcrConfig,
    layer_idx: usize,
    batch: usize,
    seq: usize,
    eps: f32,
    hidden_shape: Shape,
    keep_packed: bool,
    pack: Option<Arc<PackedLmWeights>>,
    typed: TypedParamSink,
) -> LayerSpec {
    LayerSpec {
        layer_idx,
        batch,
        seq,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        kv_group_size: cfg.kv_group_size(),
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        eps,
        hidden_shape,
        is_dense: cfg.is_dense_layer(layer_idx),
        num_experts_per_tok: cfg.num_experts_per_tok,
        moe_intermediate_size: cfg.moe_intermediate_size,
        n_shared_experts: cfg.n_shared_experts.max(1),
        keep_packed,
        pack,
        typed,
    }
}

fn param_cache_key(key: &str, transpose: bool) -> String {
    if transpose {
        format!("{key}\0t")
    } else {
        key.to_string()
    }
}

fn load_mat(
    emit: &mut Emit<'_>,
    key: &str,
    transpose: bool,
    keep_packed: bool,
    pack: Option<&Arc<PackedLmWeights>>,
    typed: &TypedParamSink,
) -> Result<MatParam> {
    let cache_key = param_cache_key(key, transpose);
    if let Some(&id) = emit.state.loaded_params.get(&cache_key) {
        let scheme = typed
            .schemes
            .lock()
            .expect("schemes")
            .get(&cache_key)
            .copied();
        return Ok(MatParam { id, scheme });
    }

    if keep_packed {
        if let Some(pack) = pack {
            if let Some(blob) = pack.ir_mat_blob(key, transpose)? {
                let id = emit
                    .hir()
                    .param(key, Shape::new(&[blob.bytes.len()], DType::U8));
                typed
                    .typed
                    .lock()
                    .expect("typed")
                    .push((key.to_string(), blob.bytes, DType::U8));
                typed
                    .schemes
                    .lock()
                    .expect("schemes")
                    .insert(cache_key.clone(), blob.scheme);
                emit.state.loaded_params.insert(cache_key, id);
                return Ok(MatParam {
                    id,
                    scheme: Some(blob.scheme),
                });
            }
        }
    }

    let id = emit.load_param(key, transpose)?;
    Ok(MatParam { id, scheme: None })
}

fn emit_proj(g: &mut HirMut, input: HirNodeId, weight: &MatParam, out_shape: Shape) -> HirNodeId {
    match weight.scheme {
        None => g.mm(input, weight.id),
        Some(scheme) => g.add_node(
            Op::DequantMatMul { scheme },
            vec![input, weight.id],
            out_shape,
        ),
    }
}

fn emit_grouped_proj(
    g: &mut HirMut,
    input: HirNodeId,
    weight: &MatParam,
    expert_idx: HirNodeId,
    out_shape: Shape,
) -> HirNodeId {
    match weight.scheme {
        None => g.add_node(
            Op::GroupedMatMul,
            vec![input, weight.id, expert_idx],
            out_shape,
        ),
        Some(scheme) => g.add_node(
            Op::DequantGroupedMatMul { scheme },
            vec![input, weight.id, expert_idx],
            out_shape,
        ),
    }
}

fn emit_lm_head(
    emit: &mut Emit<'_>,
    hidden: Option<FlowValue>,
    vocab: usize,
    keep_packed: bool,
    pack: Option<&Arc<PackedLmWeights>>,
    typed: &TypedParamSink,
) -> Result<Option<FlowValue>> {
    let hidden = hidden.ok_or_else(|| anyhow::anyhow!("lm_head requires hidden"))?;
    let w = load_mat(emit, "lm_head.weight", true, keep_packed, pack, typed)?;
    let mut gb = HirMut::new(emit.hir());
    let dims = hidden.shape.dims();
    let out_shape = Shape::from_dims(&[dims[0], dims[1], rlx_ir::Dim::Static(vocab)], DType::F32);
    let id = emit_proj(&mut gb, hidden.hir_id(), &w, out_shape.clone());
    Ok(Some(emit.wrap(id, out_shape)))
}

fn run_layer(
    emit: &mut Emit<'_>,
    hidden: Option<FlowValue>,
    spec: &LayerSpec,
    kv_sink: &SideOutputs,
    decode: bool,
) -> Result<Option<FlowValue>> {
    let hidden =
        hidden.ok_or_else(|| anyhow::anyhow!("layer {} requires hidden", spec.layer_idx))?;
    let lp = format!("model.layers.{}", spec.layer_idx);
    let zero_beta_h = emit
        .state
        .zero_beta
        .ok_or_else(|| anyhow::anyhow!("layer {} requires zero_beta", spec.layer_idx))?;

    let decode_state = if decode {
        Some(emit.state.decode.clone().ok_or_else(|| {
            anyhow::anyhow!("decode layer {} requires BindDecodeInputs", spec.layer_idx)
        })?)
    } else {
        None
    };
    let rope_cos =
        if decode {
            None
        } else {
            Some(emit.state.rope_cos.ok_or_else(|| {
                anyhow::anyhow!("prefill layer {} requires rope cos", spec.layer_idx)
            })?)
        };
    let rope_sin =
        if decode {
            None
        } else {
            Some(emit.state.rope_sin.ok_or_else(|| {
                anyhow::anyhow!("prefill layer {} requires rope sin", spec.layer_idx)
            })?)
        };

    let pack = spec.pack.as_ref();
    let in_ln_g = emit.load_param(&format!("{lp}.input_layernorm.weight"), false)?;
    let q_w = load_mat(
        emit,
        &format!("{lp}.self_attn.q_proj.weight"),
        true,
        spec.keep_packed,
        pack,
        &spec.typed,
    )?;
    let k_w = load_mat(
        emit,
        &format!("{lp}.self_attn.k_proj.weight"),
        true,
        spec.keep_packed,
        pack,
        &spec.typed,
    )?;
    let v_w = load_mat(
        emit,
        &format!("{lp}.self_attn.v_proj.weight"),
        true,
        spec.keep_packed,
        pack,
        &spec.typed,
    )?;
    let o_w = load_mat(
        emit,
        &format!("{lp}.self_attn.o_proj.weight"),
        true,
        spec.keep_packed,
        pack,
        &spec.typed,
    )?;
    let post_ln_g = emit.load_param(&format!("{lp}.post_attention_layernorm.weight"), false)?;

    let dense_ffn = if spec.is_dense {
        Some(DenseFfnParams {
            gate_w: load_mat(
                emit,
                &format!("{lp}.mlp.gate_proj.weight"),
                true,
                spec.keep_packed,
                pack,
                &spec.typed,
            )?,
            up_w: load_mat(
                emit,
                &format!("{lp}.mlp.up_proj.weight"),
                true,
                spec.keep_packed,
                pack,
                &spec.typed,
            )?,
            down_w: load_mat(
                emit,
                &format!("{lp}.mlp.down_proj.weight"),
                true,
                spec.keep_packed,
                pack,
                &spec.typed,
            )?,
        })
    } else {
        None
    };
    let moe_ffn = if spec.is_dense {
        None
    } else {
        Some(MoeFfnParams {
            router_w: emit.load_param(&format!("{lp}.mlp.gate.weight"), true)?,
            gate_exps: load_mat(
                emit,
                &expert_gate_exps_key(spec.layer_idx),
                false,
                spec.keep_packed,
                pack,
                &spec.typed,
            )?,
            up_exps: load_mat(
                emit,
                &expert_up_exps_key(spec.layer_idx),
                false,
                spec.keep_packed,
                pack,
                &spec.typed,
            )?,
            down_exps: load_mat(
                emit,
                &expert_down_exps_key(spec.layer_idx),
                false,
                spec.keep_packed,
                pack,
                &spec.typed,
            )?,
            s_gate_w: load_mat(
                emit,
                &format!("{lp}.mlp.shared_experts.gate_proj.weight"),
                true,
                spec.keep_packed,
                pack,
                &spec.typed,
            )?,
            s_up_w: load_mat(
                emit,
                &format!("{lp}.mlp.shared_experts.up_proj.weight"),
                true,
                spec.keep_packed,
                pack,
                &spec.typed,
            )?,
            s_down_w: load_mat(
                emit,
                &format!("{lp}.mlp.shared_experts.down_proj.weight"),
                true,
                spec.keep_packed,
                pack,
                &spec.typed,
            )?,
        })
    };

    let mut gb = HirMut::new(emit.hir());
    let skip = hidden.hir_id();
    let normed_in = gb.rms_norm(skip, in_ln_g, zero_beta_h, spec.eps);
    let q_out = Shape::new(
        &[spec.batch, spec.seq, spec.num_heads * spec.head_dim],
        DType::F32,
    );
    let kv_out = Shape::new(
        &[spec.batch, spec.seq, spec.num_kv_heads * spec.head_dim],
        DType::F32,
    );
    let q = emit_proj(&mut gb, normed_in, &q_w, q_out);
    let k = emit_proj(&mut gb, normed_in, &k_w, kv_out.clone());
    let v = emit_proj(&mut gb, normed_in, &v_w, kv_out);

    let (q_rope, k_for_cache, v_for_cache) = if let Some(decode) = decode_state {
        let past_k = decode.past_k[spec.layer_idx];
        let past_v = decode.past_v[spec.layer_idx];

        let q_rope = apply_rope_bhsd(
            &mut gb,
            q,
            decode.cos,
            decode.sin,
            spec.batch,
            spec.seq,
            spec.num_heads,
            spec.head_dim,
        );
        let k_rope = apply_rope_bhsd(
            &mut gb,
            k,
            decode.cos,
            decode.sin,
            spec.batch,
            spec.seq,
            spec.num_kv_heads,
            spec.head_dim,
        );
        let new_k = gb.concat_(vec![past_k, k_rope], 1);
        let new_v = gb.concat_(vec![past_v, v], 1);
        kv_sink.inner().lock().expect("kv sink").push(new_k);
        kv_sink.inner().lock().expect("kv sink").push(new_v);
        (q_rope, new_k, new_v)
    } else {
        let cos = rope_cos.expect("prefill rope cos");
        let sin = rope_sin.expect("prefill rope sin");
        let q_rope = apply_rope_bhsd(
            &mut gb,
            q,
            cos,
            sin,
            spec.batch,
            spec.seq,
            spec.num_heads,
            spec.head_dim,
        );
        let k_rope = apply_rope_bhsd(
            &mut gb,
            k,
            cos,
            sin,
            spec.batch,
            spec.seq,
            spec.num_kv_heads,
            spec.head_dim,
        );
        kv_sink.inner().lock().expect("kv sink").push(k_rope);
        kv_sink.inner().lock().expect("kv sink").push(v);
        (q_rope, k_rope, v)
    };

    let k_rep = repeat_kv(
        &mut gb,
        k_for_cache,
        spec.num_kv_heads,
        spec.head_dim,
        spec.kv_group_size,
    );
    let v_rep = repeat_kv(
        &mut gb,
        v_for_cache,
        spec.num_kv_heads,
        spec.head_dim,
        spec.kv_group_size,
    );

    let attn_shape = shape::attention_shape(gb.shape(q_rope));
    let attn = gb.add_node(
        Op::Attention {
            num_heads: spec.num_heads,
            head_dim: spec.head_dim,
            v_head_dim: None,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q_rope, k_rep, v_rep],
        attn_shape,
    );
    let attn_out = emit_proj(
        &mut gb,
        attn,
        &o_w,
        Shape::new(&[spec.batch, spec.seq, spec.hidden_size], DType::F32),
    );
    let post_attn = gb.add(skip, attn_out);
    let normed_post = gb.rms_norm(post_attn, post_ln_g, zero_beta_h, spec.eps);

    let ffn_out = if let Some(dense) = dense_ffn {
        build_dense_swiglu(&mut gb, spec, normed_post, &dense)
    } else {
        build_moe_ffn(&mut gb, spec, normed_post, moe_ffn.expect("moe params"))?
    };
    let out = gb.add(post_attn, ffn_out);
    Ok(Some(emit.wrap(out, spec.hidden_shape.clone())))
}

struct DenseFfnParams {
    gate_w: MatParam,
    up_w: MatParam,
    down_w: MatParam,
}

struct MoeFfnParams {
    router_w: HirNodeId,
    gate_exps: MatParam,
    up_exps: MatParam,
    down_exps: MatParam,
    s_gate_w: MatParam,
    s_up_w: MatParam,
    s_down_w: MatParam,
}

fn build_dense_swiglu(
    g: &mut HirMut,
    spec: &LayerSpec,
    normed_post: HirNodeId,
    params: &DenseFfnParams,
) -> HirNodeId {
    let ff = spec.intermediate_size;
    let gate = emit_proj(
        g,
        normed_post,
        &params.gate_w,
        Shape::new(&[spec.batch, spec.seq, ff], DType::F32),
    );
    let up = emit_proj(
        g,
        normed_post,
        &params.up_w,
        Shape::new(&[spec.batch, spec.seq, ff], DType::F32),
    );
    let gate_act = g.silu(gate);
    let swiglu = g.mul(gate_act, up);
    emit_proj(
        g,
        swiglu,
        &params.down_w,
        Shape::new(&[spec.batch, spec.seq, spec.hidden_size], DType::F32),
    )
}

fn build_moe_ffn(
    g: &mut HirMut,
    spec: &LayerSpec,
    normed_post: HirNodeId,
    params: MoeFfnParams,
) -> Result<HirNodeId> {
    let h = spec.hidden_size;
    let rows = spec.batch * spec.seq;
    let top_k = spec.num_experts_per_tok.max(1);
    let moe_ff = spec.moe_intermediate_size;
    let s_ff = spec.moe_intermediate_size * spec.n_shared_experts;

    let h_2d = g.reshape_(normed_post, vec![rows as i64, h as i64]);

    let logits = g.mm(h_2d, params.router_w);
    let probs = g.sm(logits, -1);
    let probs = if (ROUTED_SCALING_FACTOR - 1.0).abs() > f32::EPSILON {
        let scale = g.add_node(
            Op::Constant {
                data: ROUTED_SCALING_FACTOR.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        g.mul(probs, scale)
    } else {
        probs
    };

    let top_idx_2d = g.add_node(
        Op::TopK { k: top_k },
        vec![probs],
        Shape::new(&[rows, top_k], DType::F32),
    );
    let top_probs_2d = g.gather_(probs, top_idx_2d, 1);

    let mut moe_acc: Option<HirNodeId> = None;
    for ki in 0..top_k {
        let expert_col = g.narrow_(top_idx_2d, 1, ki, 1);
        let expert_idx = g.reshape_(expert_col, vec![rows as i64]);
        let prob_col = g.narrow_(top_probs_2d, 1, ki, 1);
        let prob_2d = g.reshape_(prob_col, vec![rows as i64, 1]);

        // Some backends (MLX) lower GroupedMatMul to a rank-3 runtime tensor;
        // reshape back to the declared `[rows, N]` layout before elementwise ops.
        let gate = emit_grouped_proj(
            g,
            h_2d,
            &params.gate_exps,
            expert_idx,
            Shape::new(&[rows, moe_ff], DType::F32),
        );
        let gate = g.reshape_(gate, vec![rows as i64, moe_ff as i64]);
        let up = emit_grouped_proj(
            g,
            h_2d,
            &params.up_exps,
            expert_idx,
            Shape::new(&[rows, moe_ff], DType::F32),
        );
        let up = g.reshape_(up, vec![rows as i64, moe_ff as i64]);
        let gate_act = g.silu(gate);
        let swiglu = g.mul(gate_act, up);
        let down = emit_grouped_proj(
            g,
            swiglu,
            &params.down_exps,
            expert_idx,
            Shape::new(&[rows, h], DType::F32),
        );
        let down = g.reshape_(down, vec![rows as i64, h as i64]);
        let weighted = g.mul(down, prob_2d);
        moe_acc = Some(match moe_acc {
            None => weighted,
            Some(acc) => g.add(acc, weighted),
        });
    }
    let moe_flat = moe_acc.context("num_experts_per_tok must be >= 1")?;

    let s_gate = emit_proj(
        g,
        h_2d,
        &params.s_gate_w,
        Shape::new(&[rows, s_ff], DType::F32),
    );
    let s_up = emit_proj(
        g,
        h_2d,
        &params.s_up_w,
        Shape::new(&[rows, s_ff], DType::F32),
    );
    let s_gate_act = g.silu(s_gate);
    let s_swiglu = g.mul(s_gate_act, s_up);
    let shared_out = emit_proj(
        g,
        s_swiglu,
        &params.s_down_w,
        Shape::new(&[rows, h], DType::F32),
    );

    let combined = g.add(moe_flat, shared_out);
    Ok(g.reshape_(combined, vec![spec.batch as i64, spec.seq as i64, h as i64]))
}

/// Run RoPE on BHSD layout (MLX-safe), restoring `[B, S, H*D]`.
fn apply_rope_bhsd(
    g: &mut HirMut,
    x: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    batch: usize,
    seq: usize,
    n_heads: usize,
    head_dim: usize,
) -> HirNodeId {
    let x4 = g.reshape_(
        x,
        vec![batch as i64, seq as i64, n_heads as i64, head_dim as i64],
    );
    let bhsd = g.transpose_(x4, vec![0, 2, 1, 3]);
    let roped = g.rope(bhsd, cos, sin, head_dim);
    let bshd = g.transpose_(roped, vec![0, 2, 1, 3]);
    g.reshape_(
        bshd,
        vec![batch as i64, seq as i64, (n_heads * head_dim) as i64],
    )
}

fn repeat_kv(
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
    let mut pieces = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

fn validate_heads(cfg: &UnlimitedOcrConfig) -> Result<()> {
    ensure!(
        cfg.num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads),
        "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
        cfg.num_attention_heads,
        cfg.num_key_value_heads
    );
    ensure!(
        cfg.hidden_size == cfg.num_attention_heads * cfg.head_dim(),
        "hidden_size ({}) must equal num_attention_heads * head_dim ({})",
        cfg.hidden_size,
        cfg.num_attention_heads * cfg.head_dim()
    );
    Ok(())
}

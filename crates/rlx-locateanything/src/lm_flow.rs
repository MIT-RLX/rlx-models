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

//! Qwen2.5 LM prefill/decode with fused `inputs_embeds`.

use crate::config::LocateAnythingConfig;
use crate::weights::LanguageModelPrefixLoader;
use anyhow::Result;
use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::blocks::{
    BindDecodeInputsStage, LmHeadStage, Qwen3DecoderSpec, RopeTablesStage,
    qwen3_prefill_layer_fused_kv,
};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{MaskKind, Op};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_qwen3::Qwen3Config;
use rlx_qwen3::flow::{Qwen3DecodeOpts, build_qwen3_decode_built};

/// `inputs_embeds` prefill skips the embed stage; tied `LmHead` still needs the table in `params`.
fn with_tied_embed_seed(mut flow: ModelFlow, tie: bool) -> ModelFlow {
    if tie {
        flow = flow.plugin_named("locateanything.seed_tied_embed", move |emit, hidden| {
            let v = hidden.ok_or_else(|| anyhow::anyhow!("seed_tied_embed needs activations"))?;
            let _ = emit.load_param("model.embed_tokens.weight", false)?;
            Ok(Some(v))
        });
    }
    flow
}

pub fn build_locateanything_prefill_built(
    cfg: &LocateAnythingConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
    last_logits_only: bool,
) -> Result<BuiltModel> {
    build_locateanything_prefill_built_ext(
        cfg,
        weights,
        batch,
        seq,
        with_kv_outputs,
        last_logits_only,
    )
}

pub fn build_locateanything_prefill_all_logits_built(
    cfg: &LocateAnythingConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    build_locateanything_prefill_built_ext(cfg, weights, batch, seq, false, false)
}

/// Prefill with additive 2D MTP mask (`attn_bias` input, `MaskKind::Bias` per layer).
pub fn build_locateanything_prefill_mtp_built(
    cfg: &LocateAnythingConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    last_logits_only: bool,
) -> Result<BuiltModel> {
    let qcfg = cfg.text_config.to_qwen3_config();
    validate_cfg(&qcfg)?;
    let profile = CompileProfile::llama32_prefill();
    let f = DType::F32;
    let h = qcfg.hidden_size;
    let nh = qcfg.num_attention_heads;
    let nkv = qcfg.num_key_value_heads;
    let dh = qcfg.head_dim;
    let eps = qcfg.rms_norm_eps as f32;
    let group = nh / nkv;

    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let bias_shape = Shape::new(&[batch, nh, seq, seq], f);
    let (cos_data, sin_data) = rope_tables(&qcfg);
    let decoder_spec = Qwen3DecoderSpec {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: dh,
        eps,
        hidden_shape: hidden_shape.clone(),
        batch,
        seq,
        qk_norm: qcfg.qk_norm,
        attention_bias: qcfg.attention_bias,
        mask: MaskKind::Causal,
    };

    let mut flow = ModelFlow::new("locateanything_mtp_prefill")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape.clone());
    flow = with_tied_embed_seed(flow, qcfg.tie_word_embeddings);
    flow = flow
        .input("attn_bias", bias_shape)
        .rope_tables(RopeTablesStage::param(
            qcfg.max_position_embeddings,
            dh / 2,
            cos_data,
            sin_data,
        ))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh)
        .plugin_named("locateanything.bind_mtp_bias", |emit, hidden| {
            let bias = emit.flow_input("attn_bias")?;
            emit.set_named("mtp_attn_bias", bias.hir_id());
            Ok(hidden)
        });

    for layer_idx in 0..qcfg.num_hidden_layers {
        let spec = decoder_spec.clone();
        flow = flow.plugin_named(
            format!("locateanything.mtp_layer_{layer_idx}"),
            move |emit, hidden| {
                let hidden = hidden.ok_or_else(|| anyhow::anyhow!("mtp prefill needs hidden"))?;
                let attn_bias = emit.named("mtp_attn_bias")?;
                let lp = format!("model.layers.{layer_idx}");
                let zero_beta_h = emit
                    .state
                    .zero_beta
                    .ok_or_else(|| anyhow::anyhow!("mtp layer requires zero_beta"))?;
                let cos = emit
                    .state
                    .rope_cos
                    .ok_or_else(|| anyhow::anyhow!("mtp layer requires rope cos"))?;
                let sin = emit
                    .state
                    .rope_sin
                    .ok_or_else(|| anyhow::anyhow!("mtp layer requires rope sin"))?;

                let in_ln_g = emit.load_param(&format!("{lp}.input_layernorm.weight"), false)?;
                let q_w = emit.load_param(&format!("{lp}.self_attn.q_proj.weight"), true)?;
                let k_w = emit.load_param(&format!("{lp}.self_attn.k_proj.weight"), true)?;
                let v_w = emit.load_param(&format!("{lp}.self_attn.v_proj.weight"), true)?;
                let o_w = emit.load_param(&format!("{lp}.self_attn.o_proj.weight"), true)?;
                let post_ln_g =
                    emit.load_param(&format!("{lp}.post_attention_layernorm.weight"), false)?;
                let gate_w = emit.load_param(&format!("{lp}.mlp.gate_proj.weight"), true)?;
                let up_w = emit.load_param(&format!("{lp}.mlp.up_proj.weight"), true)?;
                let down_w = emit.load_param(&format!("{lp}.mlp.down_proj.weight"), true)?;
                let (q_bias, k_bias, v_bias) = if spec.attention_bias {
                    (
                        Some(emit.load_param(&format!("{lp}.self_attn.q_proj.bias"), false)?),
                        Some(emit.load_param(&format!("{lp}.self_attn.k_proj.bias"), false)?),
                        Some(emit.load_param(&format!("{lp}.self_attn.v_proj.bias"), false)?),
                    )
                } else {
                    (None, None, None)
                };

                let mut gb = HirMut::new(emit.hir());
                let skip = hidden.hir_id();
                let normed_in = gb.rms_norm(skip, in_ln_g, zero_beta_h, spec.eps);
                let mut q = gb.mm(normed_in, q_w);
                let mut k = gb.mm(normed_in, k_w);
                let mut v = gb.mm(normed_in, v_w);
                if let (Some(qb), Some(kb), Some(vb)) = (q_bias, k_bias, v_bias) {
                    q = gb.add(q, qb);
                    k = gb.add(k, kb);
                    v = gb.add(v, vb);
                }
                // Work around MLX `Op::Rope` rank-3 multi-head packing expecting [B, H, S, D].
                // Keep the external layout rank-3 [B, S, H*D] for attention, but run rope on BHSD.
                let q4 = gb.reshape_(
                    q,
                    vec![spec.batch as i64, spec.seq as i64, nh as i64, dh as i64],
                );
                let q_bhsd = gb.transpose_(q4, vec![0, 2, 1, 3]);
                let q_rope_bhsd = gb.rope(q_bhsd, cos, sin, dh);
                let q_rope_bshd = gb.transpose_(q_rope_bhsd, vec![0, 2, 1, 3]);
                let q_rope = gb.reshape_(
                    q_rope_bshd,
                    vec![spec.batch as i64, spec.seq as i64, (nh * dh) as i64],
                );

                let k4 = gb.reshape_(
                    k,
                    vec![spec.batch as i64, spec.seq as i64, nkv as i64, dh as i64],
                );
                let k_bhsd = gb.transpose_(k4, vec![0, 2, 1, 3]);
                let k_rope_bhsd = gb.rope(k_bhsd, cos, sin, dh);
                let k_rope_bshd = gb.transpose_(k_rope_bhsd, vec![0, 2, 1, 3]);
                let k_rope = gb.reshape_(
                    k_rope_bshd,
                    vec![spec.batch as i64, spec.seq as i64, (nkv * dh) as i64],
                );
                let k_rep = mtp_repeat_kv(&mut gb, k_rope, nkv, dh, group);
                let v_rep = mtp_repeat_kv(&mut gb, v, nkv, dh, group);
                let attn = mtp_attention_bias(&mut gb, q_rope, k_rep, v_rep, attn_bias, nh, dh);
                let attn_out = gb.mm(attn, o_w);
                let post_attn = gb.add(skip, attn_out);
                let normed_post = gb.rms_norm(post_attn, post_ln_g, zero_beta_h, spec.eps);
                let gate = gb.mm(normed_post, gate_w);
                let up = gb.mm(normed_post, up_w);
                let gate_act = gb.silu(gate);
                let swiglu = gb.mul(gate_act, up);
                let ffn_out = gb.mm(swiglu, down_w);
                let out = gb.add(post_attn, ffn_out);
                Ok(Some(emit.wrap(out, spec.hidden_shape.clone())))
            },
        );
    }

    if last_logits_only {
        flow = flow.gather_last_token_at(batch, seq);
    }
    flow = flow.final_norm(eps);

    let mut prefixed = LanguageModelPrefixLoader::new(weights);
    flow.raw_stage(lm_head_stage(&qcfg))
        .output("logits")
        .build(&mut WeightLoaderSource(&mut prefixed))
}

/// MTP block prefill reusing cached prefix KV (`past_k`/`past_v`) + 2D `attn_bias`.
pub fn build_locateanything_mtp_kv_built(
    cfg: &LocateAnythingConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    q_seq: usize,
) -> Result<BuiltModel> {
    let qcfg = cfg.text_config.to_qwen3_config();
    validate_cfg(&qcfg)?;
    let profile = CompileProfile::llama32_decode();
    let f = DType::F32;
    let h = qcfg.hidden_size;
    let nh = qcfg.num_attention_heads;
    let nkv = qcfg.num_key_value_heads;
    let dh = qcfg.head_dim;
    let eps = qcfg.rms_norm_eps as f32;
    let group = nh / nkv;
    let half = dh / 2;
    let kv_dim = qcfg.kv_proj_dim();
    let k_len = past_seq + q_seq;

    let hidden_shape = Shape::new(&[batch, q_seq, h], f);
    let decoder_spec = Qwen3DecoderSpec {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: dh,
        eps,
        hidden_shape: hidden_shape.clone(),
        batch,
        seq: q_seq,
        qk_norm: qcfg.qk_norm,
        attention_bias: qcfg.attention_bias,
        mask: MaskKind::Causal,
    };

    let mut flow = ModelFlow::new("locateanything_mtp_kv")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape.clone());
    flow = with_tied_embed_seed(flow, qcfg.tie_word_embeddings);
    flow = flow
        .input("attn_bias", Shape::new(&[batch, nh, q_seq, k_len], f))
        .input("rope_cos", Shape::new(&[q_seq, half], f))
        .input("rope_sin", Shape::new(&[q_seq, half], f));

    for layer_idx in 0..qcfg.num_hidden_layers {
        flow = flow
            .input(
                format!("past_k_{layer_idx}"),
                Shape::new(&[batch, past_seq, kv_dim], f),
            )
            .input(
                format!("past_v_{layer_idx}"),
                Shape::new(&[batch, past_seq, kv_dim], f),
            );
    }

    let kv_sink = SideOutputs::new();

    flow = flow
        .raw_stage(FlowStage::BindDecodeInputs(BindDecodeInputsStage {
            num_layers: qcfg.num_hidden_layers,
            use_custom_mask: false,
            need_past_kv: true,
        }))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh)
        .plugin_named("locateanything.bind_mtp_kv_bias", |emit, hidden| {
            let bias = emit.flow_input("attn_bias")?;
            emit.set_named("mtp_attn_bias", bias.hir_id());
            Ok(hidden)
        });

    for layer_idx in 0..qcfg.num_hidden_layers {
        let spec = decoder_spec.clone();
        let sink = kv_sink.clone();
        flow = flow.plugin_named(
            format!("locateanything.mtp_kv_layer_{layer_idx}"),
            move |emit, hidden| {
                let hidden = hidden.ok_or_else(|| anyhow::anyhow!("mtp kv needs hidden"))?;
                let decode = emit
                    .state
                    .decode
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("mtp kv requires BindDecodeInputs"))?;
                let attn_bias = emit.named("mtp_attn_bias")?;
                let lp = format!("model.layers.{layer_idx}");
                let zero_beta_h = emit
                    .state
                    .zero_beta
                    .ok_or_else(|| anyhow::anyhow!("mtp kv layer requires zero_beta"))?;
                let past_k = decode.past_k[layer_idx];
                let past_v = decode.past_v[layer_idx];

                let in_ln_g = emit.load_param(&format!("{lp}.input_layernorm.weight"), false)?;
                let q_w = emit.load_param(&format!("{lp}.self_attn.q_proj.weight"), true)?;
                let k_w = emit.load_param(&format!("{lp}.self_attn.k_proj.weight"), true)?;
                let v_w = emit.load_param(&format!("{lp}.self_attn.v_proj.weight"), true)?;
                let o_w = emit.load_param(&format!("{lp}.self_attn.o_proj.weight"), true)?;
                let post_ln_g =
                    emit.load_param(&format!("{lp}.post_attention_layernorm.weight"), false)?;
                let gate_w = emit.load_param(&format!("{lp}.mlp.gate_proj.weight"), true)?;
                let up_w = emit.load_param(&format!("{lp}.mlp.up_proj.weight"), true)?;
                let down_w = emit.load_param(&format!("{lp}.mlp.down_proj.weight"), true)?;
                let (q_bias, k_bias, v_bias) = if spec.attention_bias {
                    (
                        Some(emit.load_param(&format!("{lp}.self_attn.q_proj.bias"), false)?),
                        Some(emit.load_param(&format!("{lp}.self_attn.k_proj.bias"), false)?),
                        Some(emit.load_param(&format!("{lp}.self_attn.v_proj.bias"), false)?),
                    )
                } else {
                    (None, None, None)
                };

                let mut gb = HirMut::new(emit.hir());
                let skip = hidden.hir_id();
                let normed_in = gb.rms_norm(skip, in_ln_g, zero_beta_h, spec.eps);
                let mut q = gb.mm(normed_in, q_w);
                let mut k = gb.mm(normed_in, k_w);
                let mut v = gb.mm(normed_in, v_w);
                if let (Some(qb), Some(kb), Some(vb)) = (q_bias, k_bias, v_bias) {
                    q = gb.add(q, qb);
                    k = gb.add(k, kb);
                    v = gb.add(v, vb);
                }
                // MLX `Op::Rope` rank-3 multi-head packing expects BHSD; run rope on BHSD then restore.
                let q4 = gb.reshape_(
                    q,
                    vec![spec.batch as i64, spec.seq as i64, nh as i64, dh as i64],
                );
                let q_bhsd = gb.transpose_(q4, vec![0, 2, 1, 3]);
                let q_rope_bhsd = gb.rope(q_bhsd, decode.cos, decode.sin, dh);
                let q_rope_bshd = gb.transpose_(q_rope_bhsd, vec![0, 2, 1, 3]);
                let q_rope = gb.reshape_(
                    q_rope_bshd,
                    vec![spec.batch as i64, spec.seq as i64, (nh * dh) as i64],
                );

                let k4 = gb.reshape_(
                    k,
                    vec![spec.batch as i64, spec.seq as i64, nkv as i64, dh as i64],
                );
                let k_bhsd = gb.transpose_(k4, vec![0, 2, 1, 3]);
                let k_rope_bhsd = gb.rope(k_bhsd, decode.cos, decode.sin, dh);
                let k_rope_bshd = gb.transpose_(k_rope_bhsd, vec![0, 2, 1, 3]);
                let k_rope = gb.reshape_(
                    k_rope_bshd,
                    vec![spec.batch as i64, spec.seq as i64, (nkv * dh) as i64],
                );
                let new_k = gb.concat_(vec![past_k, k_rope], 1);
                let new_v = gb.concat_(vec![past_v, v], 1);
                sink.inner().lock().expect("mtp kv sink").push(new_k);
                sink.inner().lock().expect("mtp kv sink").push(new_v);
                let k_rep = mtp_repeat_kv(&mut gb, new_k, nkv, dh, group);
                let v_rep = mtp_repeat_kv(&mut gb, new_v, nkv, dh, group);
                let attn = mtp_attention_bias(&mut gb, q_rope, k_rep, v_rep, attn_bias, nh, dh);
                let attn_out = gb.mm(attn, o_w);
                let post_attn = gb.add(skip, attn_out);
                let normed_post = gb.rms_norm(post_attn, post_ln_g, zero_beta_h, spec.eps);
                let gate = gb.mm(normed_post, gate_w);
                let up = gb.mm(normed_post, up_w);
                let gate_act = gb.silu(gate);
                let swiglu = gb.mul(gate_act, up);
                let ffn_out = gb.mm(swiglu, down_w);
                let out = gb.add(post_attn, ffn_out);
                Ok(Some(emit.wrap(out, spec.hidden_shape.clone())))
            },
        );
    }

    flow = flow.final_norm(eps);
    let mut prefixed = LanguageModelPrefixLoader::new(weights);
    let built = flow
        .raw_stage(lm_head_stage(&qcfg))
        .output("logits")
        .build(&mut WeightLoaderSource(&mut prefixed))?;
    Ok(built.with_extra_hir_outputs(kv_sink.drain()))
}

fn build_locateanything_prefill_built_ext(
    cfg: &LocateAnythingConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
    last_logits_only: bool,
) -> Result<BuiltModel> {
    let qcfg = cfg.text_config.to_qwen3_config();
    validate_cfg(&qcfg)?;
    let profile = CompileProfile::llama32_prefill();
    let f = DType::F32;
    let h = qcfg.hidden_size;
    let nh = qcfg.num_attention_heads;
    let nkv = qcfg.num_key_value_heads;
    let dh = qcfg.head_dim;
    let eps = qcfg.rms_norm_eps as f32;

    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let (cos_data, sin_data) = rope_tables(&qcfg);
    let decoder_spec = Qwen3DecoderSpec {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: dh,
        eps,
        hidden_shape: hidden_shape.clone(),
        batch,
        seq,
        qk_norm: qcfg.qk_norm,
        attention_bias: qcfg.attention_bias,
        mask: MaskKind::Causal,
    };

    let kv_sink = SideOutputs::new();

    let mut flow = ModelFlow::new("locateanything_prefill")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape.clone());
    flow = with_tied_embed_seed(flow, qcfg.tie_word_embeddings);
    flow = flow
        .rope_tables(RopeTablesStage::param(
            qcfg.max_position_embeddings,
            dh / 2,
            cos_data,
            sin_data,
        ))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);

    flow = flow.repeat_layers(qcfg.num_hidden_layers, {
        let spec = decoder_spec.clone();
        let sink = kv_sink.clone();
        move |i| qwen3_prefill_layer_fused_kv(i, spec.clone(), sink.inner())
    });

    if last_logits_only {
        flow = flow.gather_last_token_at(batch, seq);
    }

    flow = flow.final_norm(eps);

    let mut prefixed = LanguageModelPrefixLoader::new(weights);
    let mut built = flow
        .raw_stage(lm_head_stage(&qcfg))
        .output("logits")
        .build(&mut WeightLoaderSource(&mut prefixed))?;

    if with_kv_outputs {
        built = built.with_extra_hir_outputs(kv_sink.drain());
    }
    Ok(built)
}

pub fn build_locateanything_decode_built(
    cfg: &LocateAnythingConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<BuiltModel> {
    build_locateanything_decode_built_ext(cfg, weights, batch, past_seq, use_custom_mask, false)
}

/// `dynamic_past` — one graph for all past lengths (KV uses `Dim::Dynamic(PAST_SEQ)`).
/// When `use_custom_mask` is true, `past_seq` sets the mask width (`past_seq + 1` keys).
pub fn build_locateanything_decode_built_ext(
    cfg: &LocateAnythingConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
    dynamic_past: bool,
) -> Result<BuiltModel> {
    let qcfg = cfg.text_config.to_qwen3_config();
    let opts = Qwen3DecodeOpts {
        batch,
        past_seq,
        dynamic_past,
        use_custom_mask,
        ragged_rope: false,
        profile: None,
    };
    let mut prefixed = LanguageModelPrefixLoader::new(weights);
    build_qwen3_decode_built(&qcfg, &mut prefixed, &opts)
}

pub fn qwen3_config(cfg: &LocateAnythingConfig) -> Qwen3Config {
    cfg.text_config.to_qwen3_config()
}

fn lm_head_stage(cfg: &Qwen3Config) -> FlowStage {
    if cfg.tie_word_embeddings {
        FlowStage::LmHead(LmHeadStage {
            weight_key: None,
            tie_word_embeddings: true,
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            tied_param_name: "qwen3.lm_head.tied_t".into(),
        })
    } else {
        FlowStage::LmHead(LmHeadStage::separate(
            "lm_head.weight",
            cfg.vocab_size,
            cfg.hidden_size,
        ))
    }
}

fn validate_cfg(cfg: &Qwen3Config) -> Result<()> {
    if !cfg
        .num_attention_heads
        .is_multiple_of(cfg.num_key_value_heads)
    {
        anyhow::bail!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            cfg.num_attention_heads,
            cfg.num_key_value_heads
        );
    }
    Ok(())
}

/// RoPE tables for `len` consecutive positions starting at `start`.
pub fn compute_rope_chunk(cfg: &Qwen3Config, start: usize, len: usize) -> (Vec<f32>, Vec<f32>) {
    let half = cfg.head_dim / 2;
    let mut cos = vec![0f32; len * half];
    let mut sin = vec![0f32; len * half];
    for t in 0..len {
        let (c, s) = compute_rope_slice(cfg, start + t);
        cos[t * half..(t + 1) * half].copy_from_slice(&c);
        sin[t * half..(t + 1) * half].copy_from_slice(&s);
    }
    (cos, sin)
}

/// Single-position RoPE slice for decode (matches `rlx_qwen3` generator).
pub fn compute_rope_slice(cfg: &Qwen3Config, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
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

fn mtp_attention_bias(
    g: &mut HirMut,
    q: rlx_ir::HirNodeId,
    k: rlx_ir::HirNodeId,
    v: rlx_ir::HirNodeId,
    bias: rlx_ir::HirNodeId,
    num_heads: usize,
    head_dim: usize,
) -> rlx_ir::HirNodeId {
    let attn_shape = rlx_ir::shape::attention_shape(g.shape(q));
    g.add_node(
        Op::Attention {
            num_heads,
            head_dim,
            mask_kind: MaskKind::Bias,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v, bias],
        attn_shape,
    )
}

fn mtp_repeat_kv(
    g: &mut HirMut,
    x: rlx_ir::HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> rlx_ir::HirNodeId {
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

fn rope_tables(cfg: &Qwen3Config) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos_data = vec![0f32; cfg.max_position_embeddings * half];
    let mut sin_data = vec![0f32; cfg.max_position_embeddings * half];
    for pos in 0..cfg.max_position_embeddings {
        for i in 0..half {
            let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
            let angle = pos as f64 * freq;
            let (s, c) = angle.sin_cos();
            cos_data[pos * half + i] = c as f32;
            sin_data[pos * half + i] = s as f32;
        }
    }
    (cos_data, sin_data)
}

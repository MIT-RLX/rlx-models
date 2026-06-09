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

//! Compiled Ministral graphs (`inputs_embeds` prefill/decode, no LM head).

use crate::config::TextConfig;
use crate::decode_shard_layer::tts_decode_shard_layer_from_sink;
use crate::weights::BackbonePrefixLoader;
use anyhow::{Result, ensure};
use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::blocks::{
    LlamaDecodeLayerSpec, LlamaDecodeLayerStage, LlamaDecoderSpec, LlamaKvTapStage,
    RopeTablesStage, llama_prefill_layer_composed, llama_prefill_layer_fused,
};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_ir::dynamic::sym;
use rlx_ir::op::MaskKind;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Shape};
use rlx_llama32::flow::Llama32DecodeOpts;
use rlx_llama32::rope::{build_rope_tables, resolve_inv_freq};
use std::sync::Arc;

pub fn build_tts_backbone_prefill_built(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
) -> Result<BuiltModel> {
    build_tts_backbone_prefill_built_opts(cfg, weights, batch, seq, with_kv_outputs, false)
}

pub fn build_tts_backbone_prefill_built_opts(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
    dynamic_seq: bool,
) -> Result<BuiltModel> {
    let llama = cfg.llama_config();
    let profile = CompileProfile::llama32_prefill();
    let f = DType::F32;
    let h = llama.hidden_size;
    let eps = llama.rms_norm_eps as f32;
    let dh = llama.head_dim();

    let hidden_shape = if dynamic_seq {
        Shape::from_dims(
            &[Dim::Static(batch), Dim::Dynamic(sym::SEQ), Dim::Static(h)],
            f,
        )
    } else {
        Shape::new(&[batch, seq, h], f)
    };
    let rope_factors = weights.take("rope_freqs.weight").ok().map(|(d, _)| d);
    let inv_freq = resolve_inv_freq(&llama, rope_factors.as_deref());
    let (cos_data, sin_data) = build_rope_tables(&inv_freq, llama.max_position_embeddings);

    let decoder_spec = LlamaDecoderSpec {
        num_heads: llama.num_attention_heads,
        head_dim: dh,
        num_kv_heads: llama.num_key_value_heads,
        eps,
        mask: MaskKind::Causal,
        hidden_shape: hidden_shape.clone(),
    };

    let kv_sink = SideOutputs::new();
    let export_kv = with_kv_outputs;

    let mut flow = ModelFlow::new("voxtral_tts_backbone_prefill")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape)
        .rope_tables(RopeTablesStage::param(
            llama.max_position_embeddings,
            inv_freq.len(),
            cos_data,
            sin_data,
        ))
        .zero_beta_named("voxtral_tts.zero_beta.hidden", h);

    flow = flow.repeat_layers(llama.num_hidden_layers, {
        let spec = decoder_spec.clone();
        let sink = kv_sink.clone();
        move |i| {
            let mut stages = Vec::new();
            if export_kv {
                stages.push(FlowStage::LlamaKvTap(LlamaKvTapStage::layer(
                    i,
                    dh,
                    eps,
                    sink.inner(),
                )));
            }
            stages.push(llama_prefill_layer_fused(i, spec.clone()));
            if stages.len() == 1 {
                stages.into_iter().next().unwrap()
            } else {
                FlowStage::Sequence(stages)
            }
        }
    });

    flow = flow.final_norm(eps);

    let mut prefixed = BackbonePrefixLoader::new(weights);
    let mut built = flow.build(&mut WeightLoaderSource(&mut prefixed))?;
    if export_kv {
        built = built.with_extra_hir_outputs(kv_sink.drain());
    }
    Ok(built)
}

/// Prefill HIR with symbolic seq dim (`sym::SEQ`) for dynamic compile cache.
pub fn build_tts_backbone_prefill_hir_dynamic_ext(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    batch: usize,
    max_seq: usize,
    with_kv_outputs: bool,
) -> Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    build_tts_backbone_prefill_built_opts(cfg, weights, batch, max_seq, with_kv_outputs, true)?
        .into_parts()
}

pub fn build_tts_backbone_decode_built_opts(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    opts: &Llama32DecodeOpts,
) -> Result<BuiltModel> {
    let llama = cfg.llama_config();
    let profile = opts
        .profile
        .clone()
        .unwrap_or_else(CompileProfile::llama32_decode);
    let f = DType::F32;
    let h = llama.hidden_size;
    let eps = llama.rms_norm_eps as f32;
    let dh = llama.head_dim();
    let kv_dim = llama.kv_proj_dim();
    let half = dh / 2;

    let hidden_shape = Shape::new(&[opts.batch, 1, h], f);
    let past_kv_shape = if opts.dynamic_past {
        Shape::from_dims(
            &[
                Dim::Static(opts.batch),
                Dim::Dynamic(sym::PAST_SEQ),
                Dim::Static(kv_dim),
            ],
            f,
        )
    } else {
        Shape::new(&[opts.batch, opts.past_seq, kv_dim], f)
    };

    let decode_spec = LlamaDecodeLayerSpec {
        num_heads: llama.num_attention_heads,
        head_dim: dh,
        num_kv_heads: llama.num_key_value_heads,
        kv_group_size: llama.kv_group_size(),
        eps,
        use_custom_mask: opts.use_custom_mask,
        hidden_shape: hidden_shape.clone(),
    };

    let kv_out = SideOutputs::new();

    let mut flow = ModelFlow::new("voxtral_tts_backbone_decode")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape)
        .input("rope_cos", Shape::new(&[1, half], f))
        .input("rope_sin", Shape::new(&[1, half], f));

    if opts.use_custom_mask {
        flow = flow.input("mask", Shape::new(&[opts.batch, opts.past_seq + 1], f));
    }

    for layer_idx in 0..llama.num_hidden_layers {
        flow = flow
            .input(format!("past_k_{layer_idx}"), past_kv_shape.clone())
            .input(format!("past_v_{layer_idx}"), past_kv_shape.clone());
    }

    flow = flow
        .bind_decode_inputs(llama.num_hidden_layers, opts.use_custom_mask)
        .zero_beta_named("voxtral_tts.zero_beta.hidden", h)
        .repeat_layers(llama.num_hidden_layers, {
            let spec = decode_spec.clone();
            let sink = kv_out.clone();
            move |i| FlowStage::Named {
                name: format!("layer{i}"),
                inner: Arc::new(FlowStage::LlamaDecodeLayer(LlamaDecodeLayerStage::layer(
                    i,
                    spec.clone(),
                    sink.inner(),
                ))),
            }
        })
        .final_norm(eps);

    let mut prefixed = BackbonePrefixLoader::new(weights);
    let built = flow
        .build(&mut WeightLoaderSource(&mut prefixed))?
        .with_extra_hir_outputs(kv_out.drain());
    Ok(built)
}

pub fn build_tts_backbone_decode_built(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    batch: usize,
    past_seq: usize,
) -> Result<BuiltModel> {
    let opts = Llama32DecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask: false,
        profile: None,
    };
    build_tts_backbone_decode_built_opts(cfg, weights, &opts)
}

/// Decode HIR with symbolic past length (`sym::PAST_SEQ`) for dynamic specialization.
pub fn build_tts_backbone_decode_hir_dynamic_ext(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    max_past_seq: usize,
) -> Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    let opts = Llama32DecodeOpts {
        batch: 1,
        past_seq: max_past_seq,
        dynamic_past: true,
        use_custom_mask: false,
        profile: None,
    };
    build_tts_backbone_decode_built_opts(cfg, weights, &opts)?.into_parts()
}

/// Layer range for wgpu/Vulkan (each shard must stay under `max_storage_buffer_binding_size`).
pub fn build_tts_backbone_prefill_shard_built_opts(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq_cap: usize,
    layer_start: usize,
    layer_end: usize,
    with_kv_outputs: bool,
    dynamic_seq: bool,
) -> Result<BuiltModel> {
    let llama = cfg.llama_config();
    ensure!(
        layer_start < layer_end && layer_end <= llama.num_hidden_layers,
        "invalid layer shard [{layer_start}, {layer_end})"
    );
    let n_shard = layer_end - layer_start;
    let profile = CompileProfile::llama32_prefill();
    let f = DType::F32;
    let h = llama.hidden_size;
    let eps = llama.rms_norm_eps as f32;
    let dh = llama.head_dim();

    let hidden_shape = if dynamic_seq {
        Shape::from_dims(
            &[Dim::Static(batch), Dim::Dynamic(sym::SEQ), Dim::Static(h)],
            f,
        )
    } else {
        Shape::new(&[batch, seq_cap, h], f)
    };
    let rope_factors = weights.take("rope_freqs.weight").ok().map(|(d, _)| d);
    let inv_freq = resolve_inv_freq(&llama, rope_factors.as_deref());
    let (cos_data, sin_data) = build_rope_tables(&inv_freq, llama.max_position_embeddings);

    let decoder_spec = LlamaDecoderSpec {
        num_heads: llama.num_attention_heads,
        head_dim: dh,
        num_kv_heads: llama.num_key_value_heads,
        eps,
        mask: MaskKind::Causal,
        hidden_shape: hidden_shape.clone(),
    };

    let kv_sink = SideOutputs::new();
    let export_kv = with_kv_outputs;
    let input_name = if layer_start == 0 {
        "inputs_embeds"
    } else {
        "hidden_in"
    };

    let mut flow = ModelFlow::new(format!(
        "voxtral_tts_backbone_prefill_{layer_start}_{layer_end}"
    ))
    .with_profile(profile)
    .input(input_name, hidden_shape)
    .rope_tables(RopeTablesStage::param(
        llama.max_position_embeddings,
        inv_freq.len(),
        cos_data,
        sin_data,
    ))
    .zero_beta_named("voxtral_tts.zero_beta.hidden", h);

    flow = flow.repeat_layers(n_shard, {
        let spec = decoder_spec.clone();
        let sink = kv_sink.clone();
        move |j| {
            let i = layer_start + j;
            let mut stages = Vec::new();
            if export_kv {
                stages.push(FlowStage::LlamaKvTap(LlamaKvTapStage::layer(
                    i,
                    dh,
                    eps,
                    sink.inner(),
                )));
            }
            stages.push(llama_prefill_layer_composed(i, spec.clone()));
            if stages.len() == 1 {
                stages.into_iter().next().unwrap()
            } else {
                FlowStage::Sequence(stages)
            }
        }
    });

    if layer_end == llama.num_hidden_layers {
        flow = flow.final_norm(eps);
    }

    let mut prefixed = BackbonePrefixLoader::new(weights);
    let mut built = flow.build(&mut WeightLoaderSource(&mut prefixed))?;
    if export_kv {
        built = built.with_extra_hir_outputs(kv_sink.drain());
    }
    Ok(built)
}

pub fn build_tts_backbone_prefill_shard_hir_dynamic_ext(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    batch: usize,
    max_seq: usize,
    layer_start: usize,
    layer_end: usize,
    with_kv_outputs: bool,
) -> Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    build_tts_backbone_prefill_shard_built_opts(
        cfg,
        weights,
        batch,
        max_seq,
        layer_start,
        layer_end,
        with_kv_outputs,
        true,
    )?
    .into_parts()
}

pub fn build_tts_backbone_decode_shard_built_opts(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    opts: &Llama32DecodeOpts,
    layer_start: usize,
    layer_end: usize,
) -> Result<BuiltModel> {
    let llama = cfg.llama_config();
    ensure!(
        layer_start < layer_end && layer_end <= llama.num_hidden_layers,
        "invalid layer shard [{layer_start}, {layer_end})"
    );
    let n_shard = layer_end - layer_start;
    let profile = opts
        .profile
        .clone()
        .unwrap_or_else(CompileProfile::llama32_decode);
    let f = DType::F32;
    let h = llama.hidden_size;
    let eps = llama.rms_norm_eps as f32;
    let dh = llama.head_dim();
    let kv_dim = llama.kv_proj_dim();
    let half = dh / 2;

    let input_name = if layer_start == 0 {
        "inputs_embeds"
    } else {
        "hidden_in"
    };
    let hidden_shape = Shape::new(&[opts.batch, 1, h], f);
    let past_kv_shape = Shape::new(&[opts.batch, opts.past_seq, kv_dim], f);

    let decode_spec = LlamaDecodeLayerSpec {
        num_heads: llama.num_attention_heads,
        head_dim: dh,
        num_kv_heads: llama.num_key_value_heads,
        kv_group_size: llama.kv_group_size(),
        eps,
        use_custom_mask: opts.use_custom_mask,
        hidden_shape: hidden_shape.clone(),
    };

    let kv_out = SideOutputs::new();

    let mut flow = ModelFlow::new(format!(
        "voxtral_tts_backbone_decode_{layer_start}_{layer_end}"
    ))
    .with_profile(profile)
    .input(input_name, hidden_shape)
    .input("rope_cos", Shape::new(&[1, half], f))
    .input("rope_sin", Shape::new(&[1, half], f));

    if opts.use_custom_mask {
        flow = flow.input("mask", Shape::new(&[opts.batch, opts.past_seq + 1], f));
    }

    for local_i in 0..n_shard {
        flow = flow
            .input(format!("past_k_{local_i}"), past_kv_shape.clone())
            .input(format!("past_v_{local_i}"), past_kv_shape.clone());
    }

    flow = flow
        .bind_decode_inputs(n_shard, opts.use_custom_mask)
        .zero_beta_named("voxtral_tts.zero_beta.hidden", h)
        .repeat_layers(n_shard, {
            let spec = decode_spec.clone();
            let sink = kv_out.clone();
            move |j| {
                let global = layer_start + j;
                tts_decode_shard_layer_from_sink(global, j, spec.clone(), &sink)
            }
        });

    if layer_end == llama.num_hidden_layers {
        flow = flow.final_norm(eps);
    }

    let mut prefixed = BackbonePrefixLoader::new(weights);
    let built = flow
        .build(&mut WeightLoaderSource(&mut prefixed))?
        .with_extra_hir_outputs(kv_out.drain());
    Ok(built)
}

pub fn build_tts_backbone_decode_shard_hir_sized_ext(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    batch: usize,
    past_seq: usize,
    layer_start: usize,
    layer_end: usize,
    use_custom_mask: bool,
) -> Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    let opts = Llama32DecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask,
        profile: None,
    };
    build_tts_backbone_decode_shard_built_opts(cfg, weights, &opts, layer_start, layer_end)?
        .into_parts()
}

/// Sized decode HIR (optionally with custom mask for bucketed KV padding).
pub fn build_tts_backbone_decode_hir_sized_ext(
    cfg: &TextConfig,
    weights: &mut WeightMap,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    let opts = Llama32DecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask,
        profile: None,
    };
    build_tts_backbone_decode_built_opts(cfg, weights, &opts)?.into_parts()
}

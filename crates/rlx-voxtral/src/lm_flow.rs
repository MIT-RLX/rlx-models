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

//! Llama decoder prefill/decode with `inputs_embeds` (audio+text fusion).

use crate::config::VoxtralConfig;
use crate::weights::LanguageModelPrefixLoader;
use anyhow::Result;
use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::blocks::{
    LlamaDecoderSpec, RopeTablesStage, llama_prefill_layer_attn_only, llama_prefill_layer_composed,
    llama_prefill_layer_fused, llama_prefill_layer_mlp_only,
};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Shape};
use rlx_llama32::flow::{Llama32DecodeOpts, build_llama32_decode_built};
use rlx_llama32::rope::{build_rope_tables, resolve_inv_freq};

pub fn build_voxtral_prefill_built(
    cfg: &VoxtralConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
    last_logits_only: bool,
) -> Result<BuiltModel> {
    let llama = cfg.llama_config();
    let profile = CompileProfile::llama32_prefill();
    let f = DType::F32;
    let h = llama.hidden_size;
    let eps = llama.rms_norm_eps as f32;
    let dh = llama.head_dim();

    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let rope_factors = weights.take("rope_freqs.weight").ok().map(|(d, _)| d);
    let inv_freq = resolve_inv_freq(llama, rope_factors.as_deref());
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

    let mut flow = ModelFlow::new("voxtral_prefill")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape.clone())
        .rope_tables(RopeTablesStage::param(
            llama.max_position_embeddings,
            inv_freq.len(),
            cos_data,
            sin_data,
        ))
        .zero_beta_named("voxtral.zero_beta.hidden", h);

    // Debug bisection: RLX_VOXTRAL_NLAYERS caps the decoder layer count so we can
    // split "layers" from the "final_norm + lm_head" tail (0 = head-only).
    let n_layers = std::env::var("RLX_VOXTRAL_NLAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| n.min(llama.num_hidden_layers))
        .unwrap_or(llama.num_hidden_layers);
    let export = with_kv_outputs;
    flow = flow.repeat_layers(n_layers, {
        let spec = decoder_spec.clone();
        let sink = kv_sink.clone();
        move |i| {
            let mut stages = Vec::new();
            if export {
                stages.push(FlowStage::LlamaKvTap(
                    rlx_flow::blocks::LlamaKvTapStage::layer(i, dh, eps, sink.inner()),
                ));
            }
            // NOTE: the Metal garbage-logits bug is NOT in the fusion — the composed
            // small-block path (RLX_VOXTRAL_COMPOSED) produces byte-identical wrong
            // logits, so the culprit is a shared LM op on Metal (RoPE / RMSNorm /
            // causal-mask SDPA), which the Whisper encoder doesn't use. CPU + MLX are
            // correct. Kept the toggle for op-level bisection of the rlx-metal kernel.
            if std::env::var("RLX_VOXTRAL_ATTNONLY").is_ok() {
                stages.push(llama_prefill_layer_attn_only(i, spec.clone()));
            } else if std::env::var("RLX_VOXTRAL_MLPONLY").is_ok() {
                stages.push(llama_prefill_layer_mlp_only(i, spec.clone()));
            } else if std::env::var("RLX_VOXTRAL_COMPOSED").is_ok() {
                stages.push(llama_prefill_layer_composed(i, spec.clone()));
            } else {
                stages.push(llama_prefill_layer_fused(i, spec.clone()));
            }
            if stages.len() == 1 {
                stages.into_iter().next().unwrap()
            } else {
                FlowStage::Sequence(stages)
            }
        }
    });

    if last_logits_only {
        flow = flow.gather_last_token_at(batch, seq);
    }

    flow = flow.final_norm(eps);

    let mut prefixed = LanguageModelPrefixLoader::new(weights);
    let mut built = flow
        .lm_head(llama.vocab_size, h, false)
        .build(&mut WeightLoaderSource(&mut prefixed))?;

    if with_kv_outputs {
        built = built.with_extra_hir_outputs(kv_sink.drain());
    }
    Ok(built)
}

pub fn build_voxtral_decode_built(
    cfg: &VoxtralConfig,
    weights: &mut WeightMap,
    batch: usize,
    past_seq: usize,
    dynamic_past: bool,
    use_custom_mask: bool,
) -> Result<BuiltModel> {
    let opts = Llama32DecodeOpts {
        batch,
        past_seq,
        dynamic_past,
        use_custom_mask,
        profile: None,
    };
    let mut prefixed = LanguageModelPrefixLoader::new(weights);
    build_llama32_decode_built(cfg.llama_config(), &mut prefixed, &opts)
}

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

//! Qwen3 decoder driving Qwen3-ASR: prefill from fused `inputs_embeds`
//! (audio + text) with KV export + last-token logits, plus token-id decode.

use anyhow::Result;
use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::blocks::{
    LmHeadStage, Qwen3DecoderSpec, RopeTablesStage, qwen3_prefill_layer_fused_kv,
};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_ir::{DType, Shape};
use rlx_qwen3::{Qwen3Config, Qwen3DecodeOpts, build_qwen3_decode_built};

/// Prefill the Qwen3 trunk from fused `inputs_embeds` `[batch, seq, hidden]`.
///
/// Outputs `[logits[batch, vocab], k0, v0, k1, v1, …]` — last-token logits
/// followed by per-layer K/V caches (full prompt), ready to seed decode.
pub fn build_asr_prefill_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    skip_fusion: bool,
) -> Result<BuiltModel> {
    // The Metal fused decoder kernels accumulate fp drift that flips argmax
    // over the 28 decoder layers (CPU/MLX fused paths are exact). Callers pass
    // `skip_fusion = true` on Metal to use the bit-exact unfused path.
    let mut profile = CompileProfile::llama32_prefill();
    if skip_fusion {
        profile.fusion.skip = true;
    }
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;

    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let (cos_data, sin_data) = rope_tables(cfg);

    let spec = Qwen3DecoderSpec {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: dh,
        eps,
        hidden_shape: hidden_shape.clone(),
        batch,
        seq,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
        mask: rlx_ir::op::MaskKind::Causal,
    };

    let kv_sink = SideOutputs::new();

    let mut flow = ModelFlow::new("qwen3_asr_prefill")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape)
        .rope_tables(RopeTablesStage::param(
            cfg.max_position_embeddings,
            dh / 2,
            cos_data,
            sin_data,
        ))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);

    flow = flow.repeat_layers(cfg.num_hidden_layers, {
        let spec = spec.clone();
        let sink = kv_sink.clone();
        move |i| qwen3_prefill_layer_fused_kv(i, spec.clone(), sink.inner())
    });

    flow = flow.gather_last_token_at(batch, seq).final_norm(eps);

    // Use the checkpoint's explicit (tied-value) `lm_head.weight`: the tied
    // LmHead path needs `embed_tokens` already in graph params, which the
    // `inputs_embeds` prefill never loads.
    let built = flow
        .raw_stage(FlowStage::LmHead(LmHeadStage::separate(
            "lm_head.weight",
            cfg.vocab_size,
            h,
        )))
        .output("logits")
        .build(&mut WeightLoaderSource(weights))?
        .with_extra_hir_outputs(kv_sink.drain());

    Ok(built)
}

/// One token-id decode step with KV cache (reuses the Qwen3 decode flow).
pub fn build_asr_decode_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    skip_fusion: bool,
) -> Result<BuiltModel> {
    build_asr_decode_built_opts(cfg, weights, batch, past_seq, skip_fusion, false)
}

/// As [`build_asr_decode_built`], but with `use_custom_mask` so one graph
/// compiled at a fixed (bucket) `past_seq` serves every actual past length via a
/// per-step mask — lets the decode loop reuse a single compiled model (weights
/// resident once, no per-token reload; one pipeline ⇒ no Metal OOM/cache growth).
pub fn build_asr_decode_built_opts(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    skip_fusion: bool,
    use_custom_mask: bool,
) -> Result<BuiltModel> {
    let profile = if skip_fusion {
        let mut p = CompileProfile::llama32_decode();
        p.fusion.skip = true;
        Some(p)
    } else {
        None
    };
    let opts = Qwen3DecodeOpts {
        batch,
        past_seq,
        use_custom_mask,
        profile,
        ..Default::default()
    };
    build_qwen3_decode_built(cfg, weights, &opts)
}

/// RoPE cos/sin tables `[max_pos, head_dim/2]` (matches the Qwen3 flow).
pub fn rope_tables(cfg: &Qwen3Config) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos = vec![0f32; cfg.max_position_embeddings * half];
    let mut sin = vec![0f32; cfg.max_position_embeddings * half];
    for pos in 0..cfg.max_position_embeddings {
        for i in 0..half {
            let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos[pos * half + i] = c as f32;
            sin[pos * half + i] = s as f32;
        }
    }
    (cos, sin)
}

/// RoPE cos/sin row for a single position (`[1, head_dim/2]` flattened).
pub fn rope_slice(cfg: &Qwen3Config, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for i in 0..half {
        let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
        let (s, c) = (pos as f64 * freq).sin_cos();
        cos[i] = c as f32;
        sin[i] = s as f32;
    }
    (cos, sin)
}

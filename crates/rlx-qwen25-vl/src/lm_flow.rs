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

//! Qwen2.5-VL LM flow — runtime mRoPE prefill/decode on `rlx-qwen3` dense layers.

use crate::config::Qwen25VlLmConfig;
use crate::mrope::mrope_prefill_feeds;
use anyhow::Result;
use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::blocks::{LmHeadStage, Qwen3DecoderSpec, qwen3_prefill_layer_side};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Shape};
use rlx_qwen3::Qwen3Config;
use rlx_qwen3::flow::Qwen3DecodeOpts;

#[derive(Debug, Clone)]
pub struct Qwen25VlPrefillOpts {
    pub batch: usize,
    pub seq: usize,
    pub with_lm_head: bool,
    pub last_logits_only: bool,
    /// Export post-RoPE Q and expanded K side outputs per layer (AIF probe).
    pub export_aif_qk: bool,
    pub profile: Option<CompileProfile>,
}

impl Qwen25VlPrefillOpts {
    pub fn vlm_prefill(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            with_lm_head: false,
            last_logits_only: true,
            export_aif_qk: false,
            profile: None,
        }
    }

    pub fn vlm_prefill_aif_probe(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            with_lm_head: false,
            last_logits_only: true,
            export_aif_qk: true,
            profile: None,
        }
    }
}

/// Prefill from host-spliced hidden states + per-token runtime mRoPE tables.
pub fn build_qwen25_vl_prefill_mrope_built(
    vl_cfg: &Qwen25VlLmConfig,
    weights: &mut dyn WeightLoader,
    opts: &Qwen25VlPrefillOpts,
    section_positions: Option<&[[usize; 4]]>,
) -> Result<BuiltModel> {
    let cfg = &vl_cfg.lm;
    let profile = opts
        .profile
        .clone()
        .unwrap_or_else(CompileProfile::qwen3_prefill);
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let batch = opts.batch;
    let seq = opts.seq;
    let half = dh / 2;

    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let rope_shape = Shape::new(&[seq, half], f);

    let decoder_spec = Qwen3DecoderSpec {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: dh,
        eps,
        hidden_shape: hidden_shape.clone(),
        batch,
        seq,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
        mask: MaskKind::Causal,
    };

    let kv_sink = SideOutputs::new();
    let qk_sink = SideOutputs::new();

    let mut flow = ModelFlow::new("qwen25_vl_prefill_mrope")
        .with_profile(profile)
        .input("prefill_hidden", hidden_shape)
        .input("rope_cos", rope_shape.clone())
        .input("rope_sin", rope_shape)
        .plugin_named("qwen25vl.rope.bind", move |emit, _| {
            let cos = emit.flow_input("rope_cos")?.hir_id();
            let sin = emit.flow_input("rope_sin")?.hir_id();
            emit.state.rope_cos = Some(cos);
            emit.state.rope_sin = Some(sin);
            Ok(None)
        })
        .plugin_named("qwen25vl.adopt_hidden", move |emit, _| {
            if emit.weights.has("model.embed_tokens.weight") {
                let _ = emit.load_param("model.embed_tokens.weight", false)?;
            }
            Ok(Some(emit.flow_input("prefill_hidden")?))
        })
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);

    flow = flow.repeat_layers(cfg.num_hidden_layers, {
        let spec = decoder_spec.clone();
        let sink = kv_sink.clone();
        let qk = qk_sink.clone();
        let export_kv = true;
        let export_qk = opts.export_aif_qk;
        move |i| qwen3_prefill_layer_side(i, spec.clone(), &sink, &qk, export_kv, export_qk)
    });

    if opts.last_logits_only {
        flow = flow.gather_last_token_at(batch, seq);
    }

    flow = flow.final_norm(eps);

    if opts.with_lm_head {
        flow = flow.raw_stage(lm_head_stage(cfg)).output("logits");
    } else {
        flow = flow.output("hidden_states");
    }

    let built = flow.build(&mut WeightLoaderSource(weights))?;
    let mut extra = kv_sink.drain();
    if opts.export_aif_qk {
        extra.extend(qk_sink.drain());
    }
    let built = built.with_extra_hir_outputs(extra);

    let _ = mrope_prefill_feeds(vl_cfg, seq, section_positions, half);
    Ok(built)
}

pub fn build_qwen25_vl_decode_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3DecodeOpts,
) -> Result<BuiltModel> {
    rlx_qwen3::flow::build_qwen3_decode_built(cfg, weights, opts)
}

pub fn mrope_decode_feeds(vl_cfg: &Qwen25VlLmConfig, abs_pos: usize) -> (Vec<f32>, Vec<f32>) {
    crate::mrope::mrope_slice_at_pos(vl_cfg, abs_pos, vl_cfg.head_half())
}

fn lm_head_stage(cfg: &Qwen3Config) -> FlowStage {
    if cfg.tie_word_embeddings {
        FlowStage::LmHead(LmHeadStage {
            weight_key: None,
            tie_word_embeddings: true,
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            tied_param_name: "qwen25vl.lm_head.tied_t".into(),
        })
    } else {
        FlowStage::LmHead(LmHeadStage::separate(
            "lm_head.weight",
            cfg.vocab_size,
            cfg.hidden_size,
        ))
    }
}

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

//! Tier-0 Qwen3 flow — native [`ModelFlow`] assembly (QK-norm + GQA + RoPE).
//!
//! ```rust,ignore
//! use rlx_models::qwen3::Qwen3Flow;
//!
//! // Prefill hidden states
//! let built = Qwen3Flow::for_prefill(&cfg, 1, 128).build(&mut weights)?;
//!
//! // Decode step with KV side outputs
//! let built = Qwen3Flow::for_decode(&cfg, 1, 256)
//!     .custom_mask()
//!     .build(&mut weights)?;
//! ```

use anyhow::{Result, anyhow};
use rlx_flow::blocks::{
    LmHeadStage, Qwen3DecodeLayerSpec, Qwen3DecoderSpec, RopeTablesStage, qwen3_decode_layer_fused,
    qwen3_decode_layer_side, qwen3_prefill_layer_side,
};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_ir::dynamic::sym;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Shape};

use super::config::Qwen3Config;
use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::weight_loader::WeightLoader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3Mode {
    Prefill,
    Decode,
}

#[derive(Debug, Clone)]
pub struct Qwen3PrefillOpts {
    pub batch: usize,
    pub seq: usize,
    pub with_lm_head: bool,
    pub with_kv_outputs: bool,
    /// Export post-RoPE Q and GQA-expanded K per layer (AIF / attention probe).
    pub with_qk_outputs: bool,
    pub last_logits_only: bool,
    pub profile: Option<CompileProfile>,
    /// When set, use these tables (`[seq, head_dim/2]`) instead of config-derived RoPE.
    pub rope_cos: Option<Vec<f32>>,
    pub rope_sin: Option<Vec<f32>>,
}

impl Qwen3PrefillOpts {
    pub fn static_prefill(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            with_lm_head: false,
            with_kv_outputs: false,
            with_qk_outputs: false,
            last_logits_only: false,
            profile: None,
            rope_cos: None,
            rope_sin: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3DecodeOpts {
    pub batch: usize,
    pub past_seq: usize,
    pub dynamic_past: bool,
    pub use_custom_mask: bool,
    /// When `true`, declare the RoPE inputs as `[batch, head_dim/2]` (one row
    /// per sequence) instead of the default `[1, head_dim/2]` shared row. This
    /// lets **ragged** batched decode give each sequence its own absolute
    /// position — the rows of the cos/sin tables are indexed per batch element.
    /// Requires `batch > 1` to differ from the shared default.
    pub ragged_rope: bool,
    /// Export post-RoPE Q and GQA-expanded K side outputs per layer (AIF decode probe).
    pub export_qk: bool,
    pub profile: Option<CompileProfile>,
}

impl Default for Qwen3DecodeOpts {
    fn default() -> Self {
        Self {
            batch: 1,
            past_seq: 0,
            dynamic_past: false,
            use_custom_mask: false,
            ragged_rope: false,
            export_qk: false,
            profile: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3Flow<'a> {
    cfg: &'a Qwen3Config,
    mode: Qwen3Mode,
    batch: usize,
    seq: usize,
    past_seq: usize,
    dynamic_past: bool,
    with_lm_head: bool,
    with_kv_outputs: bool,
    with_qk_outputs: bool,
    last_logits_only: bool,
    use_custom_mask: bool,
    profile: Option<CompileProfile>,
}

impl<'a> Qwen3Flow<'a> {
    pub fn new(cfg: &'a Qwen3Config) -> Self {
        Self {
            cfg,
            mode: Qwen3Mode::Prefill,
            batch: 1,
            seq: 128,
            past_seq: 0,
            dynamic_past: false,
            with_lm_head: false,
            with_kv_outputs: false,
            with_qk_outputs: false,
            last_logits_only: false,
            use_custom_mask: false,
            profile: None,
        }
    }

    pub fn for_prefill(cfg: &'a Qwen3Config, batch: usize, seq: usize) -> Self {
        Self::new(cfg).prefill().batch(batch).seq(seq)
    }

    pub fn for_decode(cfg: &'a Qwen3Config, batch: usize, past_seq: usize) -> Self {
        Self::new(cfg)
            .decode()
            .batch(batch)
            .past(past_seq)
            .lm_head()
    }

    pub fn prefill(mut self) -> Self {
        self.mode = Qwen3Mode::Prefill;
        self
    }

    pub fn decode(mut self) -> Self {
        self.mode = Qwen3Mode::Decode;
        self
    }

    pub fn batch(mut self, batch: usize) -> Self {
        self.batch = batch;
        self
    }

    pub fn seq(mut self, seq: usize) -> Self {
        self.seq = seq;
        self
    }

    pub fn past(mut self, past_seq: usize) -> Self {
        self.past_seq = past_seq;
        self
    }

    pub fn dynamic_past(mut self) -> Self {
        self.dynamic_past = true;
        self
    }

    pub fn lm_head(mut self) -> Self {
        self.with_lm_head = true;
        self
    }

    pub fn last_token_logits(mut self) -> Self {
        self.with_lm_head = true;
        self.last_logits_only = true;
        self
    }

    pub fn export_kv(mut self) -> Self {
        self.with_kv_outputs = true;
        self
    }

    /// Export per-layer Q/K side outputs (requires [`Self::export_kv`]).
    pub fn export_qk(mut self) -> Self {
        self.with_kv_outputs = true;
        self.with_qk_outputs = true;
        self
    }

    pub fn custom_mask(mut self) -> Self {
        self.use_custom_mask = true;
        self
    }

    pub fn profile(mut self, profile: CompileProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    pub fn build(self, weights: &mut dyn WeightLoader) -> Result<BuiltModel> {
        match self.mode {
            Qwen3Mode::Prefill => {
                build_qwen3_prefill_built(self.cfg, weights, &self.into_prefill_opts())
            }
            Qwen3Mode::Decode => {
                build_qwen3_decode_built(self.cfg, weights, &self.into_decode_opts())
            }
        }
    }
}

impl Qwen3Flow<'_> {
    fn into_prefill_opts(self) -> Qwen3PrefillOpts {
        Qwen3PrefillOpts {
            batch: self.batch,
            seq: self.seq,
            with_lm_head: self.with_lm_head,
            with_kv_outputs: self.with_kv_outputs,
            with_qk_outputs: self.with_qk_outputs,
            last_logits_only: self.last_logits_only,
            profile: self.profile,
            rope_cos: None,
            rope_sin: None,
        }
    }

    fn into_decode_opts(self) -> Qwen3DecodeOpts {
        Qwen3DecodeOpts {
            batch: self.batch,
            past_seq: self.past_seq,
            dynamic_past: self.dynamic_past,
            use_custom_mask: self.use_custom_mask,
            ragged_rope: false,
            export_qk: false,
            profile: self.profile,
        }
    }
}

pub fn build_qwen3_prefill_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3PrefillOpts,
) -> Result<BuiltModel> {
    validate_cfg(cfg)?;

    let profile = opts
        .profile
        .clone()
        .unwrap_or_else(CompileProfile::llama32_prefill);
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let batch = opts.batch;
    let seq = opts.seq;

    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let (cos_data, sin_data) = rope_tables(cfg);
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
        mask: crate::builder::attn_mask_kind(cfg),
    };

    let kv_sink = SideOutputs::new();
    let qk_sink = SideOutputs::new();

    let mut flow = ModelFlow::new("qwen3")
        .with_profile(profile)
        // input_ids declared I32 — backends convert from f32 host buffers at
        // the input boundary (Metal arena.write_from_f32, MLX mc::astype).
        // Declaring F32 here bit-reinterpreted the float bytes as ints inside
        // the gather kernel, producing garbled-token streams.
        .input("input_ids", Shape::new(&[batch, seq], DType::I32))
        .rope_tables(RopeTablesStage::param(
            cfg.max_position_embeddings,
            dh / 2,
            cos_data,
            sin_data,
        ))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh)
        .token_embed();

    flow = flow.repeat_layers(cfg.num_hidden_layers, {
        let spec = decoder_spec.clone();
        let sink = kv_sink.clone();
        let qk = qk_sink.clone();
        let export_kv = opts.with_kv_outputs;
        let export_qk = opts.with_qk_outputs;
        move |i| qwen3_prefill_layer_side(i, spec.clone(), &sink, &qk, export_kv, export_qk)
    });

    if opts.with_lm_head && opts.last_logits_only {
        flow = flow.gather_last_token_at(batch, seq);
    }

    flow = flow.final_norm(eps);

    let mut built = if opts.with_lm_head {
        flow.raw_stage(qwen3_lm_head_stage(cfg))
            .output("logits")
            .build(&mut WeightLoaderSource(weights))?
    } else {
        flow.output("hidden_states")
            .build(&mut WeightLoaderSource(weights))?
    };

    if opts.with_kv_outputs {
        let mut extra = kv_sink.drain();
        if opts.with_qk_outputs {
            extra.extend(qk_sink.drain());
        }
        built = built.with_extra_hir_outputs(extra);
    }
    Ok(built)
}

pub fn build_qwen3_decode_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3DecodeOpts,
) -> Result<BuiltModel> {
    validate_cfg(cfg)?;

    let profile = opts
        .profile
        .clone()
        .unwrap_or_else(CompileProfile::llama32_decode);
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let batch = opts.batch;
    let half = dh / 2;
    let kv_dim = cfg.kv_proj_dim();

    let hidden_shape = Shape::new(&[batch, 1, h], f);
    // F16-resident KV cache matches the F16 K/V the decode layer stores.
    let kv_dt = if rlx_ir::env::flag("RLX_QWEN3_F16_KV") {
        DType::F16
    } else {
        f
    };
    let past_kv_shape = if opts.dynamic_past {
        Shape::from_dims(
            &[
                Dim::Static(batch),
                Dim::Dynamic(sym::PAST_SEQ),
                Dim::Static(kv_dim),
            ],
            kv_dt,
        )
    } else {
        Shape::new(&[batch, opts.past_seq, kv_dim], kv_dt)
    };

    let decode_spec = Qwen3DecodeLayerSpec {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: dh,
        kv_group_size: cfg.kv_group_size(),
        eps,
        use_custom_mask: opts.use_custom_mask,
        hidden_shape: hidden_shape.clone(),
        batch,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
    };

    let kv_out = SideOutputs::new();
    let qk_out = SideOutputs::new();

    // Ragged decode supplies one RoPE row per sequence (`[batch, half]`); the
    // default shares a single row across the batch (`[1, half]`).
    let rope_rows = if opts.ragged_rope { batch } else { 1 };
    let mut flow = ModelFlow::new("qwen3_decode")
        .with_profile(profile)
        .input("input_ids", Shape::new(&[batch, 1], DType::I32))
        .input("rope_cos", Shape::new(&[rope_rows, half], f))
        .input("rope_sin", Shape::new(&[rope_rows, half], f));

    if opts.use_custom_mask {
        flow = flow.input("mask", Shape::new(&[batch, opts.past_seq + 1], f));
    }

    for layer_idx in 0..cfg.num_hidden_layers {
        flow = flow
            .input(format!("past_k_{layer_idx}"), past_kv_shape.clone())
            .input(format!("past_v_{layer_idx}"), past_kv_shape.clone());
    }

    let export_qk = opts.export_qk;
    let built = flow
        .bind_decode_inputs(cfg.num_hidden_layers, opts.use_custom_mask, true)
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh)
        .token_embed()
        .repeat_layers(cfg.num_hidden_layers, {
            let spec = decode_spec.clone();
            let kv = kv_out.clone();
            let qk = qk_out.clone();
            move |i| qwen3_decode_layer_side(i, spec.clone(), &kv, &qk, export_qk)
        })
        .final_norm(eps)
        .raw_stage(qwen3_lm_head_stage(cfg))
        .output("logits")
        .build(&mut WeightLoaderSource(weights))?;
    let mut extra = kv_out.drain();
    if opts.export_qk {
        extra.extend(qk_out.drain());
    }
    Ok(built.with_extra_hir_outputs(extra))
}

/// Decode one step from a precomputed embedding (`[batch, 1, hidden]`).
/// Outputs `hidden_states` and per-layer K/V (no LM head — use an external codec head for TTS).
pub fn build_qwen3_decode_embeds_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3DecodeOpts,
) -> Result<BuiltModel> {
    validate_cfg(cfg)?;

    let profile = opts
        .profile
        .clone()
        .unwrap_or_else(CompileProfile::llama32_decode);
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let batch = opts.batch;
    let half = dh / 2;
    let kv_dim = cfg.kv_proj_dim();

    let hidden_shape = Shape::new(&[batch, 1, h], f);
    // F16-resident KV cache matches the F16 K/V the decode layer stores.
    let kv_dt = if rlx_ir::env::flag("RLX_QWEN3_F16_KV") {
        DType::F16
    } else {
        f
    };
    let past_kv_shape = if opts.dynamic_past {
        Shape::from_dims(
            &[
                Dim::Static(batch),
                Dim::Dynamic(sym::PAST_SEQ),
                Dim::Static(kv_dim),
            ],
            kv_dt,
        )
    } else {
        Shape::new(&[batch, opts.past_seq, kv_dim], kv_dt)
    };

    let decode_spec = Qwen3DecodeLayerSpec {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: dh,
        kv_group_size: cfg.kv_group_size(),
        eps,
        use_custom_mask: opts.use_custom_mask,
        hidden_shape: hidden_shape.clone(),
        batch,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
    };

    let kv_out = SideOutputs::new();

    let mut flow = ModelFlow::new("qwen3_decode_embeds")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape)
        .input("rope_cos", Shape::new(&[1, half], f))
        .input("rope_sin", Shape::new(&[1, half], f));

    if opts.use_custom_mask {
        flow = flow.input("mask", Shape::new(&[batch, opts.past_seq + 1], f));
    }

    for layer_idx in 0..cfg.num_hidden_layers {
        flow = flow
            .input(format!("past_k_{layer_idx}"), past_kv_shape.clone())
            .input(format!("past_v_{layer_idx}"), past_kv_shape.clone());
    }

    let built = flow
        .bind_decode_inputs(cfg.num_hidden_layers, opts.use_custom_mask, true)
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh)
        .repeat_layers(cfg.num_hidden_layers, {
            let spec = decode_spec.clone();
            let sink = kv_out.clone();
            move |i| qwen3_decode_layer_fused(i, spec.clone(), sink.inner())
        })
        .final_norm(eps)
        .output("hidden_states")
        .build(&mut WeightLoaderSource(weights))?
        .with_extra_hir_outputs(kv_out.drain());

    Ok(built)
}

/// Prefill from `inputs_embeds` (`[batch, seq, hidden]`) with optional K/V export.
pub fn build_qwen3_prefill_embeds_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3PrefillOpts,
) -> Result<BuiltModel> {
    validate_cfg(cfg)?;

    let profile = opts
        .profile
        .clone()
        .unwrap_or_else(CompileProfile::llama32_prefill);
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let batch = opts.batch;
    let seq = opts.seq;

    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let half = dh / 2;
    let (cos_data, sin_data) = match (&opts.rope_cos, &opts.rope_sin) {
        (Some(c), Some(s)) => (c.clone(), s.clone()),
        _ => rope_tables(cfg),
    };
    let rope_max_pos = cfg.max_position_embeddings;
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
        mask: crate::builder::attn_mask_kind(cfg),
    };

    let kv_sink = SideOutputs::new();
    let qk_sink = SideOutputs::new();

    let mut flow = ModelFlow::new("qwen3_prefill_embeds")
        .with_profile(profile)
        .input("inputs_embeds", hidden_shape)
        .rope_tables(RopeTablesStage::param(
            rope_max_pos,
            half,
            cos_data,
            sin_data,
        ))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);

    flow = flow.repeat_layers(cfg.num_hidden_layers, {
        let spec = decoder_spec.clone();
        let sink = kv_sink.clone();
        let qk = qk_sink.clone();
        let export_kv = opts.with_kv_outputs;
        let export_qk = opts.with_qk_outputs;
        move |i| qwen3_prefill_layer_side(i, spec.clone(), &sink, &qk, export_kv, export_qk)
    });

    if opts.last_logits_only {
        flow = flow.gather_last_token_at(batch, seq);
    }

    flow = flow.final_norm(eps);

    let mut built = flow
        .output("hidden_states")
        .build(&mut WeightLoaderSource(weights))?;

    if opts.with_kv_outputs {
        let mut extra = kv_sink.drain();
        if opts.with_qk_outputs {
            extra.extend(qk_sink.drain());
        }
        built = built.with_extra_hir_outputs(extra);
    }
    Ok(built)
}

pub fn build_qwen3_prefill_flow(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3PrefillOpts,
) -> Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    build_qwen3_prefill_built(cfg, weights, opts)?.into_parts()
}

pub fn build_qwen3_decode_flow(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3DecodeOpts,
) -> Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    build_qwen3_decode_built(cfg, weights, opts)?.into_parts()
}

pub fn build_qwen3_prefill_graph(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3PrefillOpts,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    rlx_core::flow_util::graph_from_built(build_qwen3_prefill_built(cfg, weights, opts)?)
}

pub fn build_qwen3_decode_graph(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3DecodeOpts,
) -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)> {
    rlx_core::flow_util::graph_from_built(build_qwen3_decode_built(cfg, weights, opts)?)
}

pub fn build_qwen3_prefill_hir(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3PrefillOpts,
) -> Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    build_qwen3_prefill_flow(cfg, weights, opts)
}

pub fn build_qwen3_decode_hir(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    opts: &Qwen3DecodeOpts,
) -> Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    build_qwen3_decode_flow(cfg, weights, opts)
}

pub(crate) fn qwen3_lm_head_stage(cfg: &Qwen3Config) -> FlowStage {
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

pub(crate) fn validate_cfg(cfg: &Qwen3Config) -> Result<()> {
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
    // Qwen 2 / 2.5 ship `attention_bias=true` (explicit bias on Q/K/V);
    // the builder loads + adds the bias vectors. Qwen 3 sets `false`.
    Ok(())
}

pub(crate) fn rope_tables(cfg: &Qwen3Config) -> (Vec<f32>, Vec<f32>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    fn tiny_cfg() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            qk_norm: true,
            sliding_window: None,
            max_window_layers: usize::MAX,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    fn synthetic_weights(cfg: &Qwen3Config) -> WeightMap {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_proj_dim();
        let kv_dim = cfg.kv_proj_dim();
        let int_dim = cfg.intermediate_size;
        let dh = cfg.head_dim;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];
        t.insert(
            "model.embed_tokens.weight".into(),
            (z(cfg.vocab_size * h), vec![cfg.vocab_size, h]),
        );
        let lp = "model.layers.0";
        t.insert(format!("{lp}.input_layernorm.weight"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (z(h), vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (z(q_dim * h), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (z(kv_dim * h), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (z(kv_dim * h), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (z(h * q_dim), vec![h, q_dim]),
        );
        t.insert(format!("{lp}.self_attn.q_norm.weight"), (z(dh), vec![dh]));
        t.insert(format!("{lp}.self_attn.k_norm.weight"), (z(dh), vec![dh]));
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (z(int_dim * h), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (z(int_dim * h), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (z(h * int_dim), vec![h, int_dim]),
        );
        t.insert("model.norm.weight".into(), (z(h), vec![h]));
        t.insert(
            "lm_head.weight".into(),
            (z(cfg.vocab_size * h), vec![cfg.vocab_size, h]),
        );
        WeightMap::from_tensors(t)
    }

    #[test]
    fn prefill_flow_builds() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let built = Qwen3Flow::for_prefill(&cfg, 1, 4).build(&mut wm).unwrap();
        assert_eq!(built.primary_shape().rank(), 3);
    }

    #[test]
    fn prefill_flow_export_kv_qk() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let built = Qwen3Flow::for_prefill(&cfg, 1, 4)
            .export_qk()
            .build(&mut wm)
            .unwrap();
        let hir = built.into_hir().unwrap();
        // primary + K/V + Q/K per layer
        assert_eq!(hir.outputs.len(), 1 + 4 * cfg.num_hidden_layers);
    }

    #[test]
    fn prefill_flow_export_kv() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let built = Qwen3Flow::for_prefill(&cfg, 1, 4)
            .export_kv()
            .build(&mut wm)
            .unwrap();
        let hir = built.into_hir().unwrap();
        assert!(hir.outputs.len() >= 3);
    }

    #[test]
    fn decode_flow_builds() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let built = Qwen3Flow::for_decode(&cfg, 1, 4).build(&mut wm).unwrap();
        let hir = built.into_hir().unwrap();
        assert!(hir.outputs.len() >= 3, "logits + new K/V");
    }

    #[test]
    fn decode_flow_export_qk() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let opts = Qwen3DecodeOpts {
            batch: 1,
            past_seq: 4,
            export_qk: true,
            ..Default::default()
        };
        let built = build_qwen3_decode_built(&cfg, &mut wm, &opts).unwrap();
        let hir = built.into_hir().unwrap();
        assert_eq!(hir.outputs.len(), 1 + 4 * cfg.num_hidden_layers);
    }
}

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

//! Fluent Gemma model assembly — tier-0 reference for `rlx-flow`.
//!
//! ```rust,ignore
//! use rlx_models::gemma::GemmaFlow;
//!
//! // Prefill logits for the last token
//! let built = GemmaFlow::for_prefill(&cfg, 1, 128)
//!     .last_token_logits()
//!     .profile_near(&weights_path)
//!     .build(&mut weights)?;
//!
//! // Decode step with KV side outputs
//! let built = GemmaFlow::for_decode(&cfg, 1, 256)
//!     .custom_mask()
//!     .profile_decode()
//!     .build(&mut weights)?;
//!
//! // Override one layer while keeping the rest of the recipe
//! let built = GemmaFlow::for_prefill(&cfg, 1, 128)
//!     .layer(|ctx| {
//!         if ctx.index() == 0 {
//!             ctx.default_stage() // or FlowStage::Custom(...)
//!         } else {
//!             ctx.default_stage()
//!         }
//!     })
//!     .build(&mut weights)?;
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use rlx_flow::blocks::{
    DecodeRopeParamsStage, EmbedScaleStage, GemmaDecodeLayerSpec, GemmaDecodeLayerStage,
    GemmaLayerStyle, GemmaRmsNormStage, LmHeadStage, LogitSoftcapStage, RopeTablesStage,
    gemma_attn_spec, gemma_prefill_layer_composed,
};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_ir::dynamic::sym;
use rlx_ir::hir::HirModule;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, Shape};

use super::config::{GemmaArch, GemmaConfig};
use super::rope::{build_rope_tables, resolve_inv_freq};
use rlx_core::flow_bridge::{WeightLoaderSource, load_compile_profile};
use rlx_core::weight_loader::WeightLoader;

/// Tier-1 profile file name colocated with weights.
pub const GEMMA_PROFILE_FILE: &str = "gemma.rlx.toml";

/// Resolve compile profile from `gemma.rlx.toml` in the weights directory.
pub fn gemma_profile_near_weights(weights: &Path, decode: bool) -> CompileProfile {
    let default = if decode {
        CompileProfile::gemma_decode()
    } else {
        CompileProfile::gemma_prefill()
    };
    let dir = weights.parent().unwrap_or_else(|| Path::new("."));
    load_compile_profile(&dir.join(GEMMA_PROFILE_FILE), default)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaMode {
    Prefill,
    Decode,
}

/// Per-layer context for `.layer()` overrides — defaults preserve stock Gemma blocks.
pub enum GemmaLayerCtx<'a> {
    Prefill {
        index: usize,
        style: GemmaLayerStyle,
        attn: rlx_flow::blocks::SelfAttnPrefillSpec,
        kv_sink: &'a SideOutputs,
        export_kv: bool,
        head_dim: usize,
        eps: f32,
    },
    Decode {
        index: usize,
        spec: GemmaDecodeLayerSpec,
        kv_out: &'a SideOutputs,
        /// EAGLE3-style pre-attention-norm input tap. `Some` only on
        /// layers requested via [`GemmaFlow::with_aux_hidden_outputs`].
        aux_in: Option<&'a SideOutputs>,
    },
}

impl GemmaLayerCtx<'_> {
    pub fn index(&self) -> usize {
        match self {
            Self::Prefill { index, .. } | Self::Decode { index, .. } => *index,
        }
    }

    pub fn default_stage(&self) -> FlowStage {
        match self {
            Self::Prefill {
                index,
                style,
                attn,
                kv_sink,
                export_kv,
                head_dim: _,
                eps,
            } => gemma_prefill_layer_composed(
                *index,
                *style,
                attn.clone(),
                *eps,
                if *export_kv {
                    Some(kv_sink.inner())
                } else {
                    None
                },
            ),
            Self::Decode {
                index,
                spec,
                kv_out,
                aux_in,
            } => {
                let mut stage = GemmaDecodeLayerStage::layer(*index, spec.clone(), kv_out.inner());
                if let Some(sink) = aux_in {
                    stage = stage.with_aux_input_tap(sink.inner());
                }
                FlowStage::Named {
                    name: format!("layer{index}"),
                    inner: Arc::new(FlowStage::GemmaDecodeLayer(stage)),
                }
            }
        }
    }
}

type LayerFn = Arc<dyn Fn(GemmaLayerCtx<'_>) -> FlowStage + Send + Sync>;
type FlowPatchFn = Arc<dyn Fn(ModelFlow) -> ModelFlow + Send + Sync>;

/// Fluent Gemma flow builder — reads config once, chain modifiers, then `build`.
///
/// ```rust,ignore
/// use rlx_models::gemma::{GemmaConfig, GemmaFlow};
///
/// let built = GemmaFlow::new(&cfg)
///     .prefill()
///     .batch(1)
///     .seq(128)
///     .lm_head()
///     .last_token_logits()
///     .build(&mut weights)?;
/// ```
#[derive(Clone)]
pub struct GemmaFlow<'a> {
    cfg: &'a GemmaConfig,
    mode: GemmaMode,
    batch: usize,
    seq: usize,
    past_seq: usize,
    dynamic_seq: bool,
    dynamic_past: bool,
    with_lm_head: bool,
    with_kv_outputs: bool,
    /// EAGLE3 layer-input tap: sorted, unique layer indices whose
    /// pre-attention-norm hidden states should be exported as extra
    /// graph outputs (one tensor per layer, in ascending order).
    aux_hidden_layer_ids: Vec<usize>,
    last_logits_only: bool,
    use_custom_mask: bool,
    profile: Option<CompileProfile>,
    before_layers: Vec<FlowStage>,
    after_layers: Vec<FlowStage>,
    layer_fn: Option<LayerFn>,
    flow_patch: Option<FlowPatchFn>,
    /// Prefill from fused `inputs_embeds` (`prefill_hidden` input) instead of token ids.
    prefill_hidden: bool,
    /// Sliding layers read additive `attn_bias` for vision bidirectional spans.
    media_attn_bias: bool,
}

impl fmt::Debug for GemmaFlow<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GemmaFlow")
            .field("mode", &self.mode)
            .field("batch", &self.batch)
            .field("seq", &self.seq)
            .field("past_seq", &self.past_seq)
            .field("dynamic_seq", &self.dynamic_seq)
            .field("dynamic_past", &self.dynamic_past)
            .field("with_lm_head", &self.with_lm_head)
            .field("with_kv_outputs", &self.with_kv_outputs)
            .field("last_logits_only", &self.last_logits_only)
            .field("use_custom_mask", &self.use_custom_mask)
            .field("profile", &self.profile)
            .field("before_layers", &self.before_layers.len())
            .field("after_layers", &self.after_layers.len())
            .field("layer_fn", &self.layer_fn.is_some())
            .field("flow_patch", &self.flow_patch.is_some())
            .finish_non_exhaustive()
    }
}

impl<'a> GemmaFlow<'a> {
    pub fn new(cfg: &'a GemmaConfig) -> Self {
        Self {
            cfg,
            mode: GemmaMode::Prefill,
            batch: 1,
            seq: 128,
            past_seq: 0,
            dynamic_seq: false,
            dynamic_past: false,
            with_lm_head: false,
            with_kv_outputs: false,
            aux_hidden_layer_ids: Vec::new(),
            last_logits_only: false,
            use_custom_mask: false,
            profile: None,
            before_layers: Vec::new(),
            after_layers: Vec::new(),
            layer_fn: None,
            flow_patch: None,
            prefill_hidden: false,
            media_attn_bias: false,
        }
    }

    /// Skip token embedding — feed pre-scaled hidden states at `prefill_hidden`.
    pub fn prefill_from_hidden(mut self) -> Self {
        self.prefill_hidden = true;
        self
    }

    /// Add `attn_bias` input and bidirectional self-attn on sliding layers.
    pub fn prefill_media_attn_bias(mut self) -> Self {
        self.media_attn_bias = true;
        self
    }

    /// Prefill recipe with common batch/seq defaults.
    pub fn for_prefill(cfg: &'a GemmaConfig, batch: usize, seq: usize) -> Self {
        Self::new(cfg).prefill().batch(batch).seq(seq)
    }

    /// Decode recipe with common batch/past defaults (includes LM head).
    pub fn for_decode(cfg: &'a GemmaConfig, batch: usize, past_seq: usize) -> Self {
        Self::new(cfg)
            .decode()
            .batch(batch)
            .past(past_seq)
            .lm_head()
    }

    pub fn prefill(mut self) -> Self {
        self.mode = GemmaMode::Prefill;
        self
    }

    pub fn decode(mut self) -> Self {
        self.mode = GemmaMode::Decode;
        self
    }

    pub fn batch(mut self, batch: usize) -> Self {
        self.batch = batch;
        self
    }

    /// Prefill sequence length (ignored in decode mode).
    pub fn seq(mut self, seq: usize) -> Self {
        self.seq = seq;
        self
    }

    /// Decode past length (ignored in prefill mode).
    pub fn past(mut self, past_seq: usize) -> Self {
        self.past_seq = past_seq;
        self
    }

    /// Symbolic sequence dim (`sym::SEQ`) for dynamic prefill specialization.
    pub fn dynamic_seq(mut self) -> Self {
        self.dynamic_seq = true;
        self
    }

    /// Symbolic past dim (`sym::PAST_SEQ`) for dynamic decode specialization.
    pub fn dynamic_past(mut self) -> Self {
        self.dynamic_past = true;
        self
    }

    pub fn lm_head(mut self) -> Self {
        self.with_lm_head = true;
        self
    }

    /// Hidden states only — skip LM head (default for prefill unless `.lm_head()`).
    pub fn hidden_only(mut self) -> Self {
        self.with_lm_head = false;
        self.last_logits_only = false;
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

    /// EAGLE3 layer-input tap. For each requested layer index, the
    /// pre-attention-norm hidden state (`inpL`) is appended as an
    /// extra graph output. Outputs come **after** the KV-cache
    /// outputs, in **ascending layer-index order**, regardless of
    /// the order the caller passes them in.
    ///
    /// Decode-only for now; calling this with `prefill()` has no
    /// effect (the prefill path uses the composed
    /// `gemma_prefill_layer_composed` recipe, which doesn't yet plumb
    /// the tap — see PLAN.md).
    ///
    /// Layer indices ≥ `num_hidden_layers` are silently dropped at
    /// build time.
    pub fn with_aux_hidden_outputs(mut self, ids: impl IntoIterator<Item = usize>) -> Self {
        let mut v: Vec<usize> = ids.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        self.aux_hidden_layer_ids = v;
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

    /// Fusion-first prefill profile preset.
    pub fn profile_prefill(mut self) -> Self {
        self.profile = Some(CompileProfile::gemma_prefill());
        self
    }

    pub fn profile_decode(mut self) -> Self {
        self.profile = Some(CompileProfile::gemma_decode());
        self
    }

    pub fn profile_near(mut self, weights_path: &Path) -> Self {
        let decode = self.mode == GemmaMode::Decode;
        self.profile = Some(gemma_profile_near_weights(weights_path, decode));
        self
    }

    /// Insert custom stages after embedding, before the layer stack.
    pub fn before_layers(mut self, stages: impl IntoIterator<Item = FlowStage>) -> Self {
        self.before_layers.extend(stages);
        self
    }

    /// Insert custom stages after the layer stack, before final norm / LM head.
    pub fn after_layers(mut self, stages: impl IntoIterator<Item = FlowStage>) -> Self {
        self.after_layers.extend(stages);
        self
    }

    /// Override per-layer construction (prefill or decode depending on mode).
    ///
    /// Call [`GemmaLayerCtx::default_stage`] to keep stock blocks for unmodified layers.
    pub fn layer<F>(mut self, f: F) -> Self
    where
        F: Fn(GemmaLayerCtx<'_>) -> FlowStage + Send + Sync + 'static,
    {
        self.layer_fn = Some(Arc::new(f));
        self
    }

    /// Patch the assembled [`ModelFlow`] before build — full flexibility escape hatch.
    pub fn patch_flow<F>(mut self, f: F) -> Self
    where
        F: Fn(ModelFlow) -> ModelFlow + Send + Sync + 'static,
    {
        self.flow_patch = Some(Arc::new(f));
        self
    }

    pub fn build(self, weights: &mut dyn WeightLoader) -> Result<BuiltModel> {
        match self.mode {
            GemmaMode::Prefill => self.build_prefill(weights),
            GemmaMode::Decode => self.build_decode(weights),
        }
    }

    fn build_prefill(self, weights: &mut dyn WeightLoader) -> Result<BuiltModel> {
        if self.dynamic_seq && self.batch != 1 {
            anyhow::bail!("gemma: dynamic_seq prefill requires batch=1");
        }

        let cfg = self.cfg;
        let profile = self.profile.unwrap_or_else(CompileProfile::gemma_prefill);
        let f = DType::F32;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        let layer_style = cfg.layer_style();

        let hidden_shape = prefill_hidden_shape(self.batch, self.seq, h, self.dynamic_seq, f);
        let input_shape = prefill_input_shape(self.batch, self.seq, self.dynamic_seq);

        let rope_factors = weights.take("rope_freqs.weight").ok().map(|(data, _)| data);
        let inv_freq = resolve_inv_freq(cfg, rope_factors.as_deref());
        let (cos_data, sin_data) = build_rope_tables(&inv_freq, cfg.max_position_embeddings);

        // When Gemma 4 ships split rope_parameters with a distinct
        // full-attention theta and/or partial_rotary_factor, build a
        // second cos/sin table under the "global" slot. The per-layer
        // closure below opts full-attention layers into it via
        // SelfAttnPrefillSpec::with_rope_table("global").
        let global_rope =
            secondary_rope_tables(cfg, cfg.max_position_embeddings, rope_factors.as_deref());

        let kv_sink = SideOutputs::new();

        let mut flow = ModelFlow::new("gemma").with_profile(profile);
        if self.prefill_hidden {
            flow = flow.input("prefill_hidden", hidden_shape.clone());
        } else {
            flow = flow.input("input_ids", input_shape);
        }

        if self.dynamic_seq && self.with_lm_head && self.last_logits_only {
            flow = flow.input("last_token_idx", Shape::new(&[self.batch], DType::F32));
        }

        if self.media_attn_bias {
            let nh = cfg.num_attention_heads;
            if self.dynamic_seq {
                flow = flow.input(
                    "attn_bias",
                    Shape::from_dims(
                        &[
                            rlx_ir::shape::Dim::Static(self.batch),
                            rlx_ir::shape::Dim::Static(nh),
                            rlx_ir::shape::Dim::Dynamic(rlx_ir::sym::SEQ),
                            rlx_ir::shape::Dim::Dynamic(rlx_ir::sym::SEQ),
                        ],
                        f,
                    ),
                );
            } else {
                flow = flow.input(
                    "attn_bias",
                    Shape::new(&[self.batch, nh, self.seq, self.seq], f),
                );
            }
        }

        flow = flow
            .rope_tables(RopeTablesStage::param(
                cfg.max_position_embeddings,
                inv_freq.len(),
                cos_data,
                sin_data,
            ))
            .zero_beta_named("gemma.zero_beta.hidden", h);

        if self.prefill_hidden {
            flow = flow.plugin_named("gemma.prefill_hidden_bind", move |emit, _| {
                let hidden = emit
                    .flow_input("prefill_hidden")
                    .map_err(|e| anyhow::anyhow!("prefill_hidden input: {e}"))?;
                // Tied LM head still needs the embedding table in params.
                let _ = emit.load_param("model.embed_tokens.weight", false)?;
                Ok(Some(hidden))
            });
        } else {
            flow = flow
                .token_embed()
                .raw_stage(FlowStage::EmbedScale(EmbedScaleStage::new(h)));
        }

        flow = flow.raw_stages(self.before_layers.iter().cloned());

        if let Some(g) = &global_rope {
            flow = flow.raw_stage(FlowStage::RopeTables(RopeTablesStage::param_named(
                "global",
                cfg.max_position_embeddings,
                g.half_dim,
                g.cos.clone(),
                g.sin.clone(),
            )));
        }

        let layer_fn = self.layer_fn.clone();
        let export = self.with_kv_outputs;
        let media_bias = self.media_attn_bias;
        let num_heads = cfg.num_attention_heads;
        let num_layers = cfg.active_num_layers();
        let layer_attn: Vec<_> = (0..num_layers).map(|i| cfg.layer_attn_options(i)).collect();
        // PLAN.md M2 — Gemma 4 MoE (`gemma4-26b-a4b`) routes the FFN
        // through `MoeFfnStage` via the upstream
        // `gemma_moe_prefill_layer_composed` helper. Dense Gemma
        // (`is_moe() == false`) keeps the existing default stage.
        let is_moe = cfg.is_moe();
        let moe_num_experts = cfg.num_experts;
        let moe_top_k = cfg.num_experts_used;
        let moe_n_embd = cfg.hidden_size;
        let moe_n_ff = cfg.expert_ffn_dim();
        // Gemma 4 12B varies (head_dim, num_kv_heads, n_rot) across
        // layers — sliding layers stay at the base shape, global
        // (full-attention) layers may override. For Gemma <=3 every
        // accessor returns the uniform value so the closure is a
        // no-op shape-wise.
        let per_layer: Vec<PerLayerAttn> = (0..num_layers)
            .map(|i| PerLayerAttn {
                head_dim: cfg.layer_head_dim(i),
                num_kv_heads: cfg.layer_num_kv_heads(i),
                n_rot: cfg.layer_n_rot(i),
                rope_table: if cfg.is_full_attention_layer(i) && global_rope.is_some() {
                    Some("global".to_string())
                } else {
                    None
                },
                k_eq_v: cfg.attention_k_eq_v,
            })
            .collect();
        flow = flow.repeat_layers(num_layers, {
            let style = layer_style;
            let sink = kv_sink.clone();
            move |i| {
                let (mask, score_scale, softcap) = layer_attn[i];
                let pl = &per_layer[i];
                let lh = pl.head_dim;
                let mut attn = gemma_attn_spec(
                    i,
                    num_heads,
                    pl.head_dim,
                    pl.num_kv_heads,
                    pl.n_rot,
                    mask,
                    score_scale,
                    softcap,
                );
                if let Some(name) = pl.rope_table.as_ref() {
                    attn = attn.with_rope_table(name);
                }
                if pl.k_eq_v {
                    attn = attn.with_k_eq_v();
                }
                if let Some(ref f) = layer_fn {
                    return f(GemmaLayerCtx::Prefill {
                        index: i,
                        style,
                        attn: attn.clone(),
                        kv_sink: &sink,
                        export_kv: export,
                        head_dim: lh,
                        eps,
                    });
                }
                if media_bias {
                    return crate::multimodal_flow::multimodal_layer_override(
                        GemmaLayerCtx::Prefill {
                            index: i,
                            style,
                            attn,
                            kv_sink: &sink,
                            export_kv: export,
                            head_dim: lh,
                            eps,
                        },
                        true,
                    );
                }
                if is_moe {
                    let prefix = format!("model.layers.{i}");
                    let moe = rlx_flow::blocks::MoeFfnStage::hf(
                        prefix,
                        moe_num_experts,
                        moe_top_k,
                        moe_n_embd,
                        moe_n_ff,
                    );
                    let kv = if export { Some(sink.inner()) } else { None };
                    return rlx_flow::blocks::gemma_moe_prefill_layer_composed(
                        i, style, attn, eps, kv, moe,
                    );
                }
                GemmaLayerCtx::Prefill {
                    index: i,
                    style,
                    attn,
                    kv_sink: &sink,
                    export_kv: export,
                    head_dim: lh,
                    eps,
                }
                .default_stage()
            }
        });

        flow = flow.raw_stages(self.after_layers.iter().cloned());

        if self.with_lm_head && self.last_logits_only {
            flow = if self.dynamic_seq {
                flow.gather_last_token_dynamic(self.batch)
            } else {
                flow.gather_last_token_at(self.batch, self.seq)
            };
        }

        flow = flow.raw_stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            "model.norm",
            eps,
        )));

        if let Some(patch) = self.flow_patch {
            flow = patch(flow);
        }

        let mut built = if self.with_lm_head {
            let lm = if cfg.tie_word_embeddings {
                FlowStage::LmHead(LmHeadStage::tied(cfg.vocab_size, h))
            } else {
                FlowStage::LmHead(LmHeadStage::separate("lm_head.weight", cfg.vocab_size, h))
            };
            flow = flow.raw_stage(lm);
            if let Some(cap) = cfg.final_logit_softcapping {
                flow = flow.raw_stage(FlowStage::LogitSoftcap(LogitSoftcapStage::new(cap)));
            }
            flow.output("logits")
                .build(&mut WeightLoaderSource(weights))?
        } else {
            flow.output("hidden")
                .build(&mut WeightLoaderSource(weights))?
        };

        if self.with_kv_outputs {
            built = built.with_extra_hir_outputs(kv_sink.drain());
        }
        Ok(built)
    }

    fn build_decode(self, weights: &mut dyn WeightLoader) -> Result<BuiltModel> {
        let cfg = self.cfg;
        let profile = self.profile.unwrap_or_else(CompileProfile::gemma_decode);
        let f = DType::F32;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        let dh = cfg.head_dim();
        let half = dh / 2;

        let hidden_shape = Shape::new(&[self.batch, 1, h], f);

        let decode_style = cfg.layer_style();
        let decode_score_scale = cfg.attn_score_scale();
        let decode_softcap = cfg.attn_logit_softcapping;
        let decode_arch = cfg.arch;
        let decode_sliding = cfg.sliding_window;

        let kv_out = SideOutputs::new();
        // EAGLE3 aux tap. One SideOutputs sink shared across all tapped
        // layers. Layer-construction order is ascending, which combined
        // with `aux_hidden_layer_ids` being sorted gives ascending push
        // order — so the drained vec is naturally indexed by `(idx of
        // layer_id in aux_hidden_layer_ids)`.
        let aux_in = SideOutputs::new();
        let aux_layer_ids: Vec<usize> = self
            .aux_hidden_layer_ids
            .iter()
            .copied()
            .filter(|i| *i < cfg.num_hidden_layers)
            .collect();
        let aux_in_active = !aux_layer_ids.is_empty();
        if aux_in_active && cfg.is_moe() {
            anyhow::bail!(
                "gemma: with_aux_hidden_outputs is not yet supported on MoE configs \
                 (`is_moe()=true`). The MoE decode layer goes through \
                 `gemma_moe_decode_layer_composed`, which does not yet plumb the tap. \
                 Validate EAGLE3 against the dense Gemma 4 31B target."
            );
        }

        let rope_factors = weights.take("rope_freqs.weight").ok().map(|(data, _)| data);
        let inv_freq = resolve_inv_freq(cfg, rope_factors.as_deref());
        let (rope_cos, rope_sin) = if self.dynamic_past {
            (Vec::new(), Vec::new())
        } else {
            crate::rope::rope_slice(&inv_freq, self.past_seq)
        };

        // Static-past mode bakes the per-step cos/sin row as a const.
        // Dynamic-past mode promotes both default and (Gemma 4) global
        // rope rows to graph inputs so the runner can supply them at
        // step-time.
        let global_rope_row = if !self.dynamic_past {
            secondary_rope_row(cfg, self.past_seq, rope_factors.as_deref())
        } else {
            None
        };
        let global_params = needs_secondary_rope_params(cfg);

        let mut flow = ModelFlow::new("gemma_decode")
            .with_profile(profile)
            .input("input_ids", Shape::new(&[self.batch, 1], DType::F32));

        if self.dynamic_past {
            flow = flow
                .input("rope_cos", Shape::new(&[1, half], f))
                .input("rope_sin", Shape::new(&[1, half], f));
            if let Some(gp) = global_params {
                let half_global =
                    crate::rope::resolve_global_inv_freq(cfg, rope_factors.as_deref())
                        .map(|v| v.len())
                        .unwrap_or_else(|| crate::rope::default_inv_freq(gp.theta, gp.n_rot).len());
                flow = flow
                    .input("rope_cos_global", Shape::new(&[1, half_global], f))
                    .input("rope_sin_global", Shape::new(&[1, half_global], f))
                    .raw_stage(FlowStage::Custom(rlx_flow::blocks::CustomStage::named(
                        "gemma.bind_global_decode_rope",
                        |emit, val| {
                            // Find the freshly declared inputs in the
                            // HIR and publish them under the "global"
                            // slot so per-layer dispatch resolves
                            // state.named["global_cos"]/_sin.
                            let cos = find_hir_input(emit.hir(), "rope_cos_global")?;
                            let sin = find_hir_input(emit.hir(), "rope_sin_global")?;
                            emit.set_named("global_cos", cos);
                            emit.set_named("global_sin", sin);
                            Ok(val)
                        },
                    )));
            }
        }

        if self.use_custom_mask {
            flow = flow.input("mask", Shape::new(&[self.batch, self.past_seq + 1], f));
        }

        // Per-layer past-K/V shapes — sliding layers ship the base
        // num_kv_heads * head_dim, full-attention layers may ship a
        // smaller (Gemma 4 12B: 1 * 512 = 512 instead of 8 * 256 =
        // 2048) cache slot.
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_kv_dim = cfg.layer_num_kv_heads(layer_idx) * cfg.layer_head_dim(layer_idx);
            let shape = if self.dynamic_past {
                Shape::from_dims(
                    &[
                        Dim::Static(self.batch),
                        Dim::Dynamic(sym::PAST_SEQ),
                        Dim::Static(layer_kv_dim),
                    ],
                    f,
                )
            } else {
                Shape::new(&[self.batch, self.past_seq, layer_kv_dim], f)
            };
            flow = flow
                .input(format!("past_k_{layer_idx}"), shape.clone())
                .input(format!("past_v_{layer_idx}"), shape);
        }

        if !self.dynamic_past {
            flow = flow.raw_stage(FlowStage::DecodeRopeParams(DecodeRopeParamsStage::new(
                rope_cos, rope_sin, half,
            )));
            if let Some(g) = &global_rope_row {
                flow = flow.raw_stage(FlowStage::DecodeRopeParams(DecodeRopeParamsStage::named(
                    "global",
                    g.cos.clone(),
                    g.sin.clone(),
                    g.half_dim,
                )));
            }
        }

        flow = flow
            .bind_decode_inputs(cfg.num_hidden_layers, self.use_custom_mask, true)
            .zero_beta_named("gemma.zero_beta.hidden", h)
            .token_embed()
            .raw_stage(FlowStage::EmbedScale(EmbedScaleStage::new(h)))
            .raw_stages(self.before_layers.iter().cloned());

        let layer_fn = self.layer_fn.clone();
        let use_custom_mask = self.use_custom_mask;
        let num_heads = cfg.num_attention_heads;
        let num_layers = cfg.active_num_layers();
        // Per-layer (head_dim, kv_heads, n_rot) — uniform on Gemma <=3,
        // diverges on Gemma 4 12B's full-attention layers.
        let secondary_rope_active = global_rope_row.is_some();
        let per_layer_decode: Vec<PerLayerAttn> = (0..num_layers)
            .map(|i| PerLayerAttn {
                head_dim: cfg.layer_head_dim(i),
                num_kv_heads: cfg.layer_num_kv_heads(i),
                n_rot: cfg.layer_n_rot(i),
                rope_table: if cfg.is_full_attention_layer(i) && secondary_rope_active {
                    Some("global".to_string())
                } else {
                    None
                },
                k_eq_v: cfg.attention_k_eq_v,
            })
            .collect();
        // PLAN.md M2 — Gemma 4 MoE (`gemma4-26b-a4b`) decode-side dispatch.
        let is_moe = cfg.is_moe();
        let moe_num_experts = cfg.num_experts;
        let moe_top_k = cfg.num_experts_used;
        let moe_n_embd = cfg.hidden_size;
        let moe_n_ff = cfg.expert_ffn_dim();
        flow = flow.repeat_layers(num_layers, {
            let sink = kv_out.clone();
            let aux_sink = aux_in.clone();
            let aux_ids = aux_layer_ids.clone();
            let hidden_shape = hidden_shape.clone();
            move |i| {
                let aux_for_layer: Option<&SideOutputs> = if aux_ids.binary_search(&i).is_ok() {
                    Some(&aux_sink)
                } else {
                    None
                };
                let mask = if use_custom_mask {
                    rlx_ir::op::MaskKind::Causal
                } else {
                    match (decode_arch, decode_sliding) {
                        (GemmaArch::Gemma2, Some(w)) => rlx_flow::blocks::gemma2_layer_mask(i, w),
                        // PLAN.md M2 — Gemma 3 / 4 use the strided
                        // `sliding_window_pattern` (5 sliding + 1
                        // full for stride 6).
                        (GemmaArch::Gemma3 | GemmaArch::Gemma4, Some(w)) => {
                            rlx_flow::blocks::gemma_strided_layer_mask(
                                i,
                                w,
                                decode_arch.sliding_window_stride(),
                            )
                        }
                        _ => rlx_ir::op::MaskKind::Causal,
                    }
                };
                let pl = &per_layer_decode[i];
                let kv_group_size = num_heads / pl.num_kv_heads;
                let spec = GemmaDecodeLayerSpec {
                    style: decode_style,
                    num_heads,
                    head_dim: pl.head_dim,
                    num_kv_heads: pl.num_kv_heads,
                    kv_group_size,
                    n_rot: pl.n_rot,
                    rope_table: pl.rope_table.clone(),
                    k_eq_v: pl.k_eq_v,
                    eps,
                    use_custom_mask,
                    hidden_shape: hidden_shape.clone(),
                    mask,
                    score_scale: decode_score_scale,
                    attn_logit_softcap: decode_softcap,
                };
                if let Some(ref f) = layer_fn {
                    return f(GemmaLayerCtx::Decode {
                        index: i,
                        spec: spec.clone(),
                        kv_out: &sink,
                        aux_in: aux_for_layer,
                    });
                }
                if is_moe {
                    let prefix = format!("model.layers.{i}");
                    let moe = rlx_flow::blocks::MoeFfnStage::hf(
                        prefix,
                        moe_num_experts,
                        moe_top_k,
                        moe_n_embd,
                        moe_n_ff,
                    );
                    return rlx_flow::blocks::gemma_moe_decode_layer_composed(
                        i,
                        spec,
                        sink.inner(),
                        moe,
                    );
                }
                GemmaLayerCtx::Decode {
                    index: i,
                    spec,
                    kv_out: &sink,
                    aux_in: aux_for_layer,
                }
                .default_stage()
            }
        });

        flow = flow.raw_stages(self.after_layers.iter().cloned());

        if let Some(patch) = self.flow_patch {
            flow = patch(flow);
        }

        let mut flow = flow.raw_stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            "model.norm",
            eps,
        )));
        let lm = if cfg.tie_word_embeddings {
            FlowStage::LmHead(LmHeadStage::tied(cfg.vocab_size, h))
        } else {
            FlowStage::LmHead(LmHeadStage::separate("lm_head.weight", cfg.vocab_size, h))
        };
        flow = flow.raw_stage(lm);
        if let Some(cap) = cfg.final_logit_softcapping {
            flow = flow.raw_stage(FlowStage::LogitSoftcap(LogitSoftcapStage::new(cap)));
        }
        // Build must run first — stages only push their KV / aux
        // HirNodeIds into the shared sinks during stage `emit()`,
        // which fires inside `flow.build()`.
        let built = flow
            .output("logits")
            .build(&mut WeightLoaderSource(weights))?;
        let mut extra_outputs = kv_out.drain();
        if aux_in_active {
            let aux = aux_in.drain();
            debug_assert_eq!(
                aux.len(),
                aux_layer_ids.len(),
                "aux tap pushed {} hidden states but {} layer ids were requested",
                aux.len(),
                aux_layer_ids.len()
            );
            extra_outputs.extend(aux);
        }
        let built = built.with_extra_hir_outputs(extra_outputs);

        Ok(built)
    }
}

fn prefill_hidden_shape(
    batch: usize,
    seq: usize,
    hidden: usize,
    dynamic: bool,
    dtype: DType,
) -> Shape {
    if dynamic {
        Shape::from_dims(
            &[
                Dim::Static(batch),
                Dim::Dynamic(sym::SEQ),
                Dim::Static(hidden),
            ],
            dtype,
        )
    } else {
        Shape::new(&[batch, seq, hidden], dtype)
    }
}

fn prefill_input_shape(batch: usize, seq: usize, dynamic: bool) -> Shape {
    if dynamic {
        Shape::from_dims(&[Dim::Static(batch), Dim::Dynamic(sym::SEQ)], DType::F32)
    } else {
        Shape::new(&[batch, seq], DType::F32)
    }
}

/// Per-layer attention dimensions cached at flow-build time. Uniform
/// across layers for Gemma <=3; diverges on Gemma 4 unified where
/// full-attention layers may carry different (head_dim, kv_heads,
/// n_rot) and a secondary RoPE table.
#[derive(Debug, Clone)]
struct PerLayerAttn {
    head_dim: usize,
    num_kv_heads: usize,
    n_rot: usize,
    rope_table: Option<String>,
    k_eq_v: bool,
}

#[derive(Debug, Clone)]
struct GlobalRopeTables {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half_dim: usize,
}

/// Build the Gemma 4 "global" (full-attention) RoPE table when the
/// unified config carries a distinct rope_theta or
/// partial_rotary_factor for full-attention layers. Returns `None`
/// for Gemma <=3 and for Gemma 4 configs that omit the split.
fn secondary_rope_tables(
    cfg: &GemmaConfig,
    max_pos: usize,
    factors: Option<&[f32]>,
) -> Option<GlobalRopeTables> {
    let inv = crate::rope::resolve_global_inv_freq(cfg, factors)?;
    let (cos, sin) = crate::rope::build_rope_tables(&inv, max_pos);
    Some(GlobalRopeTables {
        cos,
        sin,
        half_dim: inv.len(),
    })
}

/// One-position decode row for the global RoPE.
fn secondary_rope_row(
    cfg: &GemmaConfig,
    pos: usize,
    factors: Option<&[f32]>,
) -> Option<GlobalRopeTables> {
    let inv = crate::rope::resolve_global_inv_freq(cfg, factors)?;
    let (cos, sin) = crate::rope::rope_slice(&inv, pos);
    Some(GlobalRopeTables {
        cos,
        sin,
        half_dim: inv.len(),
    })
}

fn needs_secondary_rope_params(cfg: &GemmaConfig) -> Option<GlobalRopeParams> {
    crate::rope::global_rope_params(cfg).map(|(theta, n_rot)| GlobalRopeParams { theta, n_rot })
}

#[derive(Debug, Clone, Copy)]
struct GlobalRopeParams {
    theta: f64,
    n_rot: usize,
}

fn find_hir_input(hir: &HirModule, name: &str) -> anyhow::Result<rlx_ir::HirNodeId> {
    use rlx_ir::hir::HirOp;
    for node in hir.nodes() {
        if let HirOp::Input { name: n } = &node.op {
            if n == name {
                return Ok(node.id);
            }
        }
    }
    Err(anyhow::anyhow!("gemma decode flow missing input: {name}"))
}

// ── Legacy opt structs + thin wrappers (backward compatible) ─────────

impl<'a> GemmaFlow<'a> {
    fn from_prefill_opts(cfg: &'a GemmaConfig, o: &GemmaPrefillOpts) -> Self {
        let mut f = GemmaFlow::new(cfg).prefill().batch(o.batch).seq(o.seq);
        if o.dynamic_seq {
            f = f.dynamic_seq();
        }
        if o.prefill_hidden {
            f = f.prefill_from_hidden();
        }
        if o.media_attn_bias {
            f = f.prefill_media_attn_bias();
        }
        if o.with_lm_head {
            f = f.lm_head();
        }
        if o.with_kv_outputs {
            f = f.export_kv();
        }
        if o.last_logits_only {
            f = f.last_token_logits();
        }
        if let Some(p) = o.profile.clone() {
            f = f.profile(p);
        }
        f
    }

    fn from_decode_opts(cfg: &'a GemmaConfig, o: &GemmaDecodeOpts) -> Self {
        let mut f = GemmaFlow::new(cfg)
            .decode()
            .batch(o.batch)
            .past(o.past_seq)
            .lm_head();
        if o.dynamic_past {
            f = f.dynamic_past();
        }
        if o.use_custom_mask {
            f = f.custom_mask();
        }
        if let Some(p) = o.profile.clone() {
            f = f.profile(p);
        }
        if !o.aux_hidden_layer_ids.is_empty() {
            f = f.with_aux_hidden_outputs(o.aux_hidden_layer_ids.iter().copied());
        }
        f
    }
}

/// Options for the tier-0 Gemma prefill assembly line.
#[derive(Debug, Clone)]
pub struct GemmaPrefillOpts {
    pub batch: usize,
    pub seq: usize,
    pub dynamic_seq: bool,
    pub prefill_hidden: bool,
    pub media_attn_bias: bool,
    pub with_lm_head: bool,
    pub with_kv_outputs: bool,
    pub last_logits_only: bool,
    pub profile: Option<CompileProfile>,
}

impl GemmaPrefillOpts {
    pub fn static_prefill(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            dynamic_seq: false,
            prefill_hidden: false,
            media_attn_bias: false,
            with_lm_head: false,
            with_kv_outputs: false,
            last_logits_only: false,
            profile: None,
        }
    }
}

/// Options for tier-0 Gemma decode (KV-cache) assembly line.
#[derive(Debug, Clone)]
pub struct GemmaDecodeOpts {
    pub batch: usize,
    pub past_seq: usize,
    pub dynamic_past: bool,
    pub use_custom_mask: bool,
    pub profile: Option<CompileProfile>,
    /// EAGLE3 layer-input tap: target layer indices whose
    /// pre-attention-norm hidden states should be emitted as extra
    /// graph outputs after the KV outputs. Empty ⇒ disabled.
    /// Mirrors [`GemmaFlow::with_aux_hidden_outputs`].
    #[doc(alias = "eagle3")]
    pub aux_hidden_layer_ids: Vec<usize>,
}

impl GemmaDecodeOpts {
    /// Default decode opts (no aux tap). Mirrors what was the old
    /// in-struct default before the EAGLE3 tap landed.
    pub fn new(batch: usize, past_seq: usize) -> Self {
        Self {
            batch,
            past_seq,
            dynamic_past: false,
            use_custom_mask: false,
            profile: None,
            aux_hidden_layer_ids: Vec::new(),
        }
    }

    /// Enable EAGLE3 aux hidden-state outputs on this decode build.
    pub fn with_aux_hidden_layer_ids(mut self, ids: impl IntoIterator<Item = usize>) -> Self {
        self.aux_hidden_layer_ids = ids.into_iter().collect();
        self
    }
}

pub fn build_gemma_prefill_flow(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    opts: &GemmaPrefillOpts,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_gemma_prefill_built(cfg, weights, opts)?.into_parts()
}

pub fn build_gemma_prefill_built(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    opts: &GemmaPrefillOpts,
) -> Result<BuiltModel> {
    GemmaFlow::from_prefill_opts(cfg, opts).build(weights)
}

pub fn build_gemma_decode_flow(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    opts: &GemmaDecodeOpts,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_gemma_decode_built(cfg, weights, opts)?.into_parts()
}

pub fn build_gemma_decode_graph(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    opts: &GemmaDecodeOpts,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    rlx_core::flow_util::graph_from_built(build_gemma_decode_built(cfg, weights, opts)?)
}

pub fn build_gemma_decode_built(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    opts: &GemmaDecodeOpts,
) -> Result<BuiltModel> {
    GemmaFlow::from_decode_opts(cfg, opts).build(weights)
}

#[cfg(test)]
mod gemma4_tests {
    use super::*;
    use crate::config::{
        GemmaArch, GemmaLayerType, GemmaRopeKind, GemmaRopeMap, GemmaRopeParameters,
    };

    fn gemma4_12b_like() -> GemmaConfig {
        let mut cfg = GemmaConfig::tiny_test();
        cfg.arch = GemmaArch::Gemma4;
        cfg.hidden_size = 3840;
        cfg.intermediate_size = 15_360;
        cfg.num_hidden_layers = 12; // small for tests
        cfg.num_attention_heads = 16;
        cfg.num_key_value_heads = 8;
        cfg.head_dim = Some(256);
        cfg.global_head_dim = Some(512);
        cfg.num_global_key_value_heads = Some(1);
        cfg.attention_k_eq_v = true;
        cfg.sliding_window = Some(1024);
        cfg.final_logit_softcapping = Some(30.0);
        cfg.tie_word_embeddings = true;
        cfg.max_position_embeddings = 4096;
        cfg.rope_theta = 10_000.0;
        // Stride-6 pattern: every 6th layer (1-indexed) is full.
        cfg.layer_types = (0..cfg.num_hidden_layers)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    GemmaLayerType::FullAttention
                } else {
                    GemmaLayerType::SlidingAttention
                }
            })
            .collect();
        cfg.rope_parameters = GemmaRopeMap {
            sliding_attention: Some(GemmaRopeParameters {
                rope_theta: Some(10_000.0),
                rope_type: Some(GemmaRopeKind::Default),
                partial_rotary_factor: None,
            }),
            full_attention: Some(GemmaRopeParameters {
                rope_theta: Some(1_000_000.0),
                rope_type: Some(GemmaRopeKind::Proportional),
                partial_rotary_factor: Some(0.25),
            }),
        };
        cfg
    }

    #[test]
    fn secondary_rope_emits_distinct_table_for_full_attention() {
        let cfg = gemma4_12b_like();
        let tables = secondary_rope_tables(&cfg, cfg.max_position_embeddings, None)
            .expect("Gemma 4 split rope_parameters should produce a secondary table");
        // The cos/sin table row stride is head_dim/2 = 256 (the RoPE kernel's
        // `tab_half`); only the leading n_rot/2 = 64 entries are rotated freqs,
        // the rest are zeroed NoPE dims.
        assert_eq!(tables.half_dim, 256);
        assert_eq!(tables.cos.len(), cfg.max_position_embeddings * 256);
        assert_eq!(tables.sin.len(), tables.cos.len());

        // pos=0 row is always (1, 0) regardless of theta.
        assert!((tables.cos[0] - 1.0).abs() < 1e-6);
        assert!(tables.sin[0].abs() < 1e-6);
        // The frequency exponent kicks in for dim>=1: at pos=1, dim=5
        // the two thetas should produce different cos values.
        //
        // Proportional partial RoPE: the global inv_freq exponent uses the
        // FULL head_dim (512) as denominator even though only 128 dims rotate
        // (HF `_compute_proportional_rope_parameters`). So the expected sample
        // is from `default_inv_freq(theta, 512)`, NOT the rotary count 128.
        let global_inv = crate::rope::default_inv_freq(1_000_000.0, 512);
        let sliding_inv = crate::rope::default_inv_freq(10_000.0, 256);
        assert!((global_inv[5] - sliding_inv[5]).abs() > 1e-3);
        let global_cos_p1_d5 = (1.0 * global_inv[5]).cos();
        let global_sample = tables.cos[256 + 5]; // pos=1, dim=5 (stride head_dim/2=256)
        assert!((global_sample as f64 - global_cos_p1_d5).abs() < 1e-5);
        // The rotated freqs occupy the first 64 cols; col 64+ is zeroed → cos=1.
        assert!((tables.cos[256 + 64] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn per_layer_kv_dims_diverge_on_full_attention() {
        let cfg = gemma4_12b_like();
        // Sliding: 8 heads * 256 = 2048.
        assert_eq!(cfg.layer_num_kv_heads(0) * cfg.layer_head_dim(0), 2048);
        // Full: 1 head * 512 = 512.
        assert_eq!(cfg.layer_num_kv_heads(5) * cfg.layer_head_dim(5), 512);
        assert_eq!(cfg.layer_num_kv_heads(11) * cfg.layer_head_dim(11), 512);
    }

    #[test]
    fn no_secondary_table_when_params_match() {
        // Gemma 3-shape (uniform rope) — secondary table should not
        // be emitted even if arch is Gemma4 (e.g. a tuned variant
        // with collapsed rope_parameters).
        let mut cfg = gemma4_12b_like();
        cfg.rope_parameters.full_attention = cfg.rope_parameters.sliding_attention;
        cfg.global_head_dim = None;
        cfg.num_global_key_value_heads = None;
        assert!(secondary_rope_tables(&cfg, cfg.max_position_embeddings, None).is_none());
    }
}

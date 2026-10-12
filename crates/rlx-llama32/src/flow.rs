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

//! Fluent LLaMA-3.2 model assembly — tier-0 reference for `rlx-flow`.
//!
//! ```rust,ignore
//! use rlx_models::llama32::Llama32Flow;
//!
//! // Prefill logits for the last token
//! let built = Llama32Flow::for_prefill(&cfg, 1, 128)
//!     .last_token_logits()
//!     .profile_near(&weights_path)
//!     .build(&mut weights)?;
//!
//! // Decode step with KV side outputs
//! let built = Llama32Flow::for_decode(&cfg, 1, 256)
//!     .custom_mask()
//!     .profile_decode()
//!     .build(&mut weights)?;
//!
//! // Override one layer while keeping the rest of the recipe
//! let built = Llama32Flow::for_prefill(&cfg, 1, 128)
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
    DecodeRopeParamsStage, LlamaDecodeLayerSpec, LlamaDecoderSpec, LlamaDecoderStage, RmsNormStage,
    RopeTablesStage, llama_prefill_layer_composed, llama_prefill_layer_fused,
};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_ir::dynamic::sym;
use rlx_ir::hir::HirModule;
use rlx_ir::op::MaskKind;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, Shape};

use super::config::Llama32Config;
use super::rope::{build_rope_tables, resolve_inv_freq, rope_slice};
use rlx_core::flow_bridge::{WeightLoaderSource, load_compile_profile};
use rlx_core::weight_loader::WeightLoader;

/// Tier-1 profile file name colocated with weights.
pub const LLAMA32_PROFILE_FILE: &str = "llama32.rlx.toml";

/// Resolve compile profile from `llama32.rlx.toml` in the weights directory.
pub fn llama32_profile_near_weights(weights: &Path, decode: bool) -> CompileProfile {
    let default = if decode {
        CompileProfile::llama32_decode()
    } else {
        CompileProfile::llama32_prefill()
    };
    let dir = weights.parent().unwrap_or_else(|| Path::new("."));
    load_compile_profile(&dir.join(LLAMA32_PROFILE_FILE), default)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Llama32Mode {
    Prefill,
    Decode,
}

/// Per-layer context for `.layer()` overrides — defaults preserve stock LLaMA blocks.
pub enum LlamaLayerCtx<'a> {
    Prefill {
        /// Execution / KV-cache slot index (unrolled across loops).
        index: usize,
        /// Weight prefix index (`model.layers.{weight_index}`).
        weight_index: usize,
        spec: &'a LlamaDecoderSpec,
        kv_sink: &'a SideOutputs,
        export_kv: bool,
        head_dim: usize,
        eps: f32,
    },
    Decode {
        /// Execution / KV-cache slot index (unrolled across loops).
        index: usize,
        /// Weight prefix index (`model.layers.{weight_index}`).
        weight_index: usize,
        spec: &'a LlamaDecodeLayerSpec,
        kv_out: &'a SideOutputs,
    },
}

impl LlamaLayerCtx<'_> {
    pub fn index(&self) -> usize {
        match self {
            Self::Prefill { index, .. } | Self::Decode { index, .. } => *index,
        }
    }

    pub fn weight_index(&self) -> usize {
        match self {
            Self::Prefill { weight_index, .. } | Self::Decode { weight_index, .. } => *weight_index,
        }
    }

    /// Stock fused LLaMA layer for this mode (what `.layer()` falls back to).
    pub fn default_stage(&self) -> FlowStage {
        match self {
            Self::Prefill {
                index,
                weight_index,
                spec,
                kv_sink,
                export_kv,
                head_dim,
                eps,
            } => {
                let mut stages = Vec::new();
                if *export_kv {
                    let mut tap = rlx_flow::blocks::LlamaKvTapStage::layer(
                        *weight_index,
                        *head_dim,
                        *eps,
                        kv_sink.inner(),
                        spec.rope_style,
                    );
                    // Keep weight prefix on the physical layer; execution order
                    // still appends K/V into the side-output sink sequentially.
                    tap.layer_prefix = format!("model.layers.{weight_index}");
                    stages.push(FlowStage::LlamaKvTap(tap));
                }
                let mut decoder = LlamaDecoderStage::layer(*weight_index, (*spec).clone());
                decoder.layer_prefix = format!("model.layers.{weight_index}");
                stages.push(FlowStage::Named {
                    name: format!("layer{index}"),
                    inner: Arc::new(FlowStage::LlamaDecoder(decoder)),
                });
                FlowStage::Sequence(stages)
            }
            Self::Decode {
                index,
                weight_index,
                spec,
                kv_out,
            } => {
                let mut decode = rlx_flow::blocks::LlamaDecodeLayerStage::layer(
                    *index,
                    (*spec).clone(),
                    kv_out.inner(),
                );
                decode.layer_prefix = format!("model.layers.{weight_index}");
                FlowStage::Named {
                    name: format!("layer{index}"),
                    inner: Arc::new(FlowStage::LlamaDecodeLayer(decode)),
                }
            }
        }
    }
}

fn loop_mid_norm_stage(eps: f32) -> FlowStage {
    FlowStage::RmsNorm(RmsNormStage::new("model.norm.weight", eps))
}

type LayerFn = Arc<dyn Fn(LlamaLayerCtx<'_>) -> FlowStage + Send + Sync>;
type FlowPatchFn = Arc<dyn Fn(ModelFlow) -> ModelFlow + Send + Sync>;

/// Fluent LLaMA-3.2 flow builder — reads config once, chain modifiers, then `build`.
///
/// ```rust,ignore
/// use rlx_models::llama32::{Llama32Config, Llama32Flow};
///
/// let built = Llama32Flow::new(&cfg)
///     .prefill()
///     .batch(1)
///     .seq(128)
///     .lm_head()
///     .last_token_logits()
///     .build(&mut weights)?;
/// ```
#[derive(Clone)]
pub struct Llama32Flow<'a> {
    cfg: &'a Llama32Config,
    mode: Llama32Mode,
    batch: usize,
    seq: usize,
    past_seq: usize,
    dynamic_seq: bool,
    dynamic_past: bool,
    with_lm_head: bool,
    with_kv_outputs: bool,
    last_logits_only: bool,
    use_custom_mask: bool,
    inputs_embeds: bool,
    profile: Option<CompileProfile>,
    before_layers: Vec<FlowStage>,
    after_layers: Vec<FlowStage>,
    layer_fn: Option<LayerFn>,
    flow_patch: Option<FlowPatchFn>,
    /// Build only this slice of the model's layers. `None` builds all of
    /// them. See [`Llama32Flow::layer_range`].
    layer_range: Option<std::ops::Range<usize>>,
    /// Apply the model's final norm. See
    /// [`Llama32Flow::without_final_norm`].
    final_norm: bool,
}

impl fmt::Debug for Llama32Flow<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Llama32Flow")
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
            .field("inputs_embeds", &self.inputs_embeds)
            .field("profile", &self.profile)
            .field("before_layers", &self.before_layers.len())
            .field("after_layers", &self.after_layers.len())
            .field("layer_fn", &self.layer_fn.is_some())
            .field("flow_patch", &self.flow_patch.is_some())
            .finish_non_exhaustive()
    }
}

impl<'a> Llama32Flow<'a> {
    pub fn new(cfg: &'a Llama32Config) -> Self {
        Self {
            cfg,
            mode: Llama32Mode::Prefill,
            batch: 1,
            seq: 128,
            past_seq: 0,
            dynamic_seq: false,
            dynamic_past: false,
            with_lm_head: false,
            with_kv_outputs: false,
            last_logits_only: false,
            use_custom_mask: false,
            inputs_embeds: false,
            profile: None,
            before_layers: Vec::new(),
            after_layers: Vec::new(),
            layer_fn: None,
            flow_patch: None,
            layer_range: None,
            final_norm: true,
        }
    }

    /// Prefill recipe with common batch/seq defaults.
    pub fn for_prefill(cfg: &'a Llama32Config, batch: usize, seq: usize) -> Self {
        Self::new(cfg).prefill().batch(batch).seq(seq)
    }

    /// Decode recipe with common batch/past defaults (includes LM head).
    pub fn for_decode(cfg: &'a Llama32Config, batch: usize, past_seq: usize) -> Self {
        Self::new(cfg)
            .decode()
            .batch(batch)
            .past(past_seq)
            .lm_head()
    }

    pub fn prefill(mut self) -> Self {
        self.mode = Llama32Mode::Prefill;
        self
    }

    pub fn decode(mut self) -> Self {
        self.mode = Llama32Mode::Decode;
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

    /// Feed precomputed `inputs_embeds [batch, seq, hidden]` directly instead of
    /// `input_ids`, skipping the token-embedding gather. For embeds-driven LMs
    /// (ChatterBox T3, Orpheus/Sesame-style prompts) whose entry is a
    /// concatenation of audio / text / conditioning embeddings rather than token
    /// ids. Works in both prefill and decode modes; the first graph input becomes
    /// `inputs_embeds` with the hidden shape.
    pub fn inputs_embeds(mut self) -> Self {
        self.inputs_embeds = true;
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

    /// Build only layers `range`, with execution indices `0..range.len()`
    /// and weights taken from `model.layers.{range.start + i}`.
    ///
    /// For running one model across several machines: each rank holds a
    /// contiguous block of layers and needs a graph over just those, while
    /// the weights keep the names they have in the checkpoint. The two
    /// indices were already separate here — a layer-sharing model reuses one
    /// weight set across several execution slots — so a block is expressed
    /// by shifting the weight index rather than by renaming anything.
    ///
    /// An empty or out-of-range slice is refused at `build` rather than
    /// quietly producing a model with no layers in it.
    pub fn layer_range(mut self, range: std::ops::Range<usize>) -> Self {
        self.layer_range = Some(range);
        self
    }

    /// Leave off the model's final norm.
    ///
    /// For a block that is not the end of the model: the final norm belongs
    /// once, immediately before the head, and a block in the middle that
    /// applied it would hand the next block an already-normalised hidden
    /// state — and the next one would normalise it again.
    ///
    /// Deliberately *not* folded into [`Self::hidden_only`]. That one says
    /// "give me hidden states instead of logits", which is what a caller
    /// extracting whole-model embeddings wants — and for them the final norm
    /// is part of the model, not something to skip.
    pub fn without_final_norm(mut self) -> Self {
        self.final_norm = false;
        self
    }

    pub fn export_kv(mut self) -> Self {
        self.with_kv_outputs = true;
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
        self.profile = Some(CompileProfile::llama32_prefill());
        self
    }

    /// Decode / KV-cache profile preset (`Fusable` lowering).
    pub fn profile_decode(mut self) -> Self {
        self.profile = Some(CompileProfile::llama32_decode());
        self
    }

    pub fn profile_near(mut self, weights_path: &Path) -> Self {
        let decode = self.mode == Llama32Mode::Decode;
        self.profile = Some(llama32_profile_near_weights(weights_path, decode));
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
    /// Call [`LlamaLayerCtx::default_stage`] to keep stock blocks for unmodified layers.
    pub fn layer<F>(mut self, f: F) -> Self
    where
        F: Fn(LlamaLayerCtx<'_>) -> FlowStage + Send + Sync + 'static,
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
            Llama32Mode::Prefill => self.build_prefill(weights),
            Llama32Mode::Decode => self.build_decode(weights),
        }
    }

    fn build_prefill(self, weights: &mut dyn WeightLoader) -> Result<BuiltModel> {
        if self.dynamic_seq && self.batch != 1 {
            anyhow::bail!("llama32: dynamic_seq prefill requires batch=1");
        }

        let cfg = self.cfg;
        let profile = self.profile.unwrap_or_else(CompileProfile::llama32_prefill);
        let f = DType::F32;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        let dh = cfg.head_dim();
        let n_rot = cfg.n_rot();

        let hidden_shape = prefill_hidden_shape(self.batch, self.seq, h, self.dynamic_seq, f);
        let input_shape = prefill_input_shape(self.batch, self.seq, self.dynamic_seq);

        let rope_factors = weights.take("rope_freqs.weight").ok().map(|(data, _)| data);
        let inv_freq = resolve_inv_freq(cfg, rope_factors.as_deref());
        let (cos_data, sin_data) = build_rope_tables(&inv_freq, cfg.max_position_embeddings);

        let decoder_spec = LlamaDecoderSpec {
            num_heads: cfg.num_attention_heads,
            head_dim: dh,
            n_rot,
            num_kv_heads: cfg.num_key_value_heads,
            eps,
            mask: MaskKind::Causal,
            hidden_shape: hidden_shape.clone(),
            rope_style: cfg.rope_style,
        };

        let kv_sink = SideOutputs::new();

        let mut flow = ModelFlow::new("llama32").with_profile(profile);
        flow = if self.inputs_embeds {
            flow.input("inputs_embeds", hidden_shape.clone())
        } else {
            flow.input("input_ids", input_shape)
        };

        if self.dynamic_seq && self.with_lm_head && self.last_logits_only {
            flow = flow.input("last_token_idx", Shape::new(&[self.batch], DType::F32));
        }

        flow = flow
            .rope_tables(RopeTablesStage::param(
                cfg.max_position_embeddings,
                inv_freq.len(),
                cos_data,
                sin_data,
            ))
            .zero_beta_named("llama32.zero_beta.hidden", h);
        if !self.inputs_embeds {
            flow = flow.token_embed();
        }
        flow = flow.raw_stages(self.before_layers.iter().cloned());

        let layer_fn = self.layer_fn.clone();
        let export = self.with_kv_outputs;
        let physical = cfg.physical_layers();
        let (kv_layers, weight_base) = block_layers(&self.layer_range, cfg.kv_layers(), physical)?;
        let skip_loop_final_norm = cfg.skip_loop_final_norm;
        flow = flow.repeat_layers(kv_layers, {
            let spec = decoder_spec.clone();
            let sink = kv_sink.clone();
            move |exec_idx| {
                let weight_idx = if physical == 0 {
                    0
                } else {
                    weight_base + (exec_idx % physical)
                };
                let base = if let Some(ref f) = layer_fn {
                    f(LlamaLayerCtx::Prefill {
                        index: exec_idx,
                        weight_index: weight_idx,
                        spec: &spec,
                        kv_sink: &sink,
                        export_kv: export,
                        head_dim: dh,
                        eps,
                    })
                } else {
                    let mut stages = Vec::new();
                    if export {
                        let mut tap = rlx_flow::blocks::LlamaKvTapStage::layer(
                            weight_idx,
                            dh,
                            eps,
                            sink.inner(),
                            spec.rope_style,
                        );
                        tap.layer_prefix = format!("model.layers.{weight_idx}");
                        stages.push(FlowStage::LlamaKvTap(tap));
                    }
                    // Partial RoPE (Phi) needs the composed path; fused HIR
                    // composite covers the full-RoPE Llama case.
                    let layer = if n_rot < dh {
                        llama_prefill_layer_composed(weight_idx, spec.clone())
                    } else {
                        llama_prefill_layer_fused(weight_idx, spec.clone())
                    };
                    stages.push(FlowStage::Named {
                        name: format!("layer{exec_idx}"),
                        inner: Arc::new(layer),
                    });
                    if stages.len() == 1 {
                        stages.into_iter().next().unwrap()
                    } else {
                        FlowStage::Sequence(stages)
                    }
                };

                // Nanbeige-style loop norm after each completed loop except the
                // last (stock `final_norm` covers the final pass).
                let loop_end = physical > 0 && (exec_idx + 1) % physical == 0;
                let last_exec = exec_idx + 1 == kv_layers;
                if loop_end && !skip_loop_final_norm && !last_exec {
                    FlowStage::Sequence(vec![base, loop_mid_norm_stage(eps)])
                } else {
                    base
                }
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

        if self.final_norm {
            flow = flow.final_norm(eps);
        }

        if let Some(patch) = self.flow_patch {
            flow = patch(flow);
        }

        let mut built = if self.with_lm_head {
            flow.lm_head(cfg.vocab_size, h, cfg.tie_word_embeddings)
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
        let profile = self.profile.unwrap_or_else(CompileProfile::llama32_decode);
        let f = DType::F32;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        let dh = cfg.head_dim();
        let n_rot = cfg.n_rot();
        let kv_dim = cfg.kv_proj_dim();
        let half = n_rot / 2;

        let hidden_shape = Shape::new(&[self.batch, 1, h], f);
        let past_kv_shape = if self.dynamic_past {
            Shape::from_dims(
                &[
                    Dim::Static(self.batch),
                    Dim::Dynamic(sym::PAST_SEQ),
                    Dim::Static(kv_dim),
                ],
                f,
            )
        } else {
            Shape::new(&[self.batch, self.past_seq, kv_dim], f)
        };

        let decode_spec = LlamaDecodeLayerSpec {
            num_heads: cfg.num_attention_heads,
            head_dim: dh,
            n_rot,
            num_kv_heads: cfg.num_key_value_heads,
            kv_group_size: cfg.kv_group_size(),
            eps,
            use_custom_mask: self.use_custom_mask,
            hidden_shape,
            rope_style: cfg.rope_style,
        };

        let rope_factors = weights.take("rope_freqs.weight").ok().map(|(data, _)| data);
        let inv_freq = resolve_inv_freq(cfg, rope_factors.as_deref());
        let (cos_data, sin_data) = build_rope_tables(&inv_freq, cfg.max_position_embeddings);

        let kv_out = SideOutputs::new();

        let mut flow = ModelFlow::new("llama32_decode").with_profile(profile);
        flow = if self.inputs_embeds {
            flow.input("inputs_embeds", Shape::new(&[self.batch, 1, h], f))
        } else {
            flow.input("input_ids", Shape::new(&[self.batch, 1], DType::F32))
        };

        if self.use_custom_mask {
            flow = flow.input("mask", Shape::new(&[self.batch, self.past_seq + 1], f));
        }

        for layer_idx in 0..cfg.kv_layers() {
            if self.past_seq > 0 || self.dynamic_past || self.use_custom_mask {
                flow = flow
                    .input(format!("past_k_{layer_idx}"), past_kv_shape.clone())
                    .input(format!("past_v_{layer_idx}"), past_kv_shape.clone());
            }
        }

        if self.dynamic_past || self.use_custom_mask {
            flow = flow.input("position", Shape::new(&[1], DType::F32));
            flow = flow
                .rope_tables(RopeTablesStage::param(
                    cfg.max_position_embeddings,
                    half,
                    cos_data,
                    sin_data,
                ))
                .gather_decode_rope(half);
        } else {
            let (cos_row, sin_row) = rope_slice(&inv_freq, self.past_seq);
            flow = flow.raw_stage(FlowStage::DecodeRopeParams(DecodeRopeParamsStage::new(
                cos_row, sin_row, half,
            )));
        }

        flow = flow
            .bind_decode_inputs(
                cfg.kv_layers(),
                self.use_custom_mask,
                self.past_seq > 0 || self.dynamic_past || self.use_custom_mask,
            )
            .zero_beta_named("llama32.zero_beta.hidden", h);
        if !self.inputs_embeds {
            flow = flow.token_embed();
        }
        flow = flow.raw_stages(self.before_layers.iter().cloned());

        let layer_fn = self.layer_fn.clone();
        let physical = cfg.physical_layers();
        let (kv_layers, weight_base) = block_layers(&self.layer_range, cfg.kv_layers(), physical)?;
        let skip_loop_final_norm = cfg.skip_loop_final_norm;
        flow = flow.repeat_layers(kv_layers, {
            let spec = decode_spec.clone();
            let sink = kv_out.clone();
            move |exec_idx| {
                let weight_idx = if physical == 0 {
                    0
                } else {
                    weight_base + (exec_idx % physical)
                };
                let base = if let Some(ref f) = layer_fn {
                    f(LlamaLayerCtx::Decode {
                        index: exec_idx,
                        weight_index: weight_idx,
                        spec: &spec,
                        kv_out: &sink,
                    })
                } else {
                    LlamaLayerCtx::Decode {
                        index: exec_idx,
                        weight_index: weight_idx,
                        spec: &spec,
                        kv_out: &sink,
                    }
                    .default_stage()
                };

                let loop_end = physical > 0 && (exec_idx + 1) % physical == 0;
                let last_exec = exec_idx + 1 == kv_layers;
                if loop_end && !skip_loop_final_norm && !last_exec {
                    FlowStage::Sequence(vec![base, loop_mid_norm_stage(eps)])
                } else {
                    base
                }
            }
        });

        flow = flow.raw_stages(self.after_layers.iter().cloned());

        if let Some(patch) = self.flow_patch {
            flow = patch(flow);
        }

        // Honor `hidden_only()` here as prefill does. It used to be ignored,
        // so a decode flow always built (and, when tied, always demanded)
        // `model.embed_tokens.weight` even for callers that only wanted hidden
        // states — which is a hard load failure for a checkpoint whose
        // embedding table lives outside the decoder's weight map.
        let flow = if self.final_norm {
            flow.final_norm(eps)
        } else {
            flow
        };
        let flow = if self.with_lm_head {
            flow.lm_head(cfg.vocab_size, h, cfg.tie_word_embeddings)
        } else {
            flow.output("hidden")
        };
        let built = flow
            .build(&mut WeightLoaderSource(weights))?
            .with_extra_hir_outputs(kv_out.drain());

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

// ── Legacy opt structs + thin wrappers (backward compatible) ─────────

impl<'a> Llama32Flow<'a> {
    fn from_prefill_opts(cfg: &'a Llama32Config, o: &Llama32PrefillOpts) -> Self {
        let mut f = Llama32Flow::new(cfg).prefill().batch(o.batch).seq(o.seq);
        if o.dynamic_seq {
            f = f.dynamic_seq();
        }
        if o.inputs_embeds {
            f = f.inputs_embeds();
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

    fn from_decode_opts(cfg: &'a Llama32Config, o: &Llama32DecodeOpts) -> Self {
        let mut f = Llama32Flow::new(cfg)
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
        f
    }
}

/// Options for the tier-0 LLaMA-3.2 prefill assembly line.
#[derive(Debug, Clone)]
pub struct Llama32PrefillOpts {
    pub batch: usize,
    pub seq: usize,
    pub dynamic_seq: bool,
    pub with_lm_head: bool,
    pub with_kv_outputs: bool,
    pub last_logits_only: bool,
    /// Feed `inputs_embeds [batch, seq, hidden]` instead of token ids.
    pub inputs_embeds: bool,
    pub profile: Option<CompileProfile>,
}

impl Llama32PrefillOpts {
    pub fn static_prefill(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            dynamic_seq: false,
            with_lm_head: false,
            with_kv_outputs: false,
            last_logits_only: false,
            inputs_embeds: false,
            profile: None,
        }
    }
}

/// Options for tier-0 LLaMA-3.2 decode (KV-cache) assembly line.
#[derive(Debug, Clone)]
pub struct Llama32DecodeOpts {
    pub batch: usize,
    pub past_seq: usize,
    pub dynamic_past: bool,
    pub use_custom_mask: bool,
    pub profile: Option<CompileProfile>,
}

/// How many layers to execute, and what to add to each execution index to
/// reach its weights.
///
/// `(all, 0)` for a whole model; a slice of it when a layer range was given.
/// A free function rather than a method because `build` has already moved
/// parts of the builder by the time it needs this.
fn block_layers(
    range: &Option<std::ops::Range<usize>>,
    all: usize,
    physical: usize,
) -> Result<(usize, usize)> {
    let Some(range) = range.clone() else {
        return Ok((all, 0));
    };
    if range.is_empty() || range.end > physical {
        anyhow::bail!("layer range {range:?} is not a slice of this model's {physical} layers");
    }
    Ok((range.len(), range.start))
}

pub fn build_llama32_prefill_flow(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    opts: &Llama32PrefillOpts,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_llama32_prefill_built(cfg, weights, opts)?.into_parts()
}

pub fn build_llama32_prefill_built(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    opts: &Llama32PrefillOpts,
) -> Result<BuiltModel> {
    Llama32Flow::from_prefill_opts(cfg, opts).build(weights)
}

pub fn build_llama32_decode_flow(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    opts: &Llama32DecodeOpts,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_llama32_decode_built(cfg, weights, opts)?.into_parts()
}

pub fn build_llama32_decode_graph(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    opts: &Llama32DecodeOpts,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    rlx_core::flow_util::graph_from_built(build_llama32_decode_built(cfg, weights, opts)?)
}

pub fn build_llama32_decode_built(
    cfg: &Llama32Config,
    weights: &mut dyn WeightLoader,
    opts: &Llama32DecodeOpts,
) -> Result<BuiltModel> {
    Llama32Flow::from_decode_opts(cfg, opts).build(weights)
}

#[cfg(test)]
mod layer_range_tests {
    use super::*;
    use crate::config::Llama32Config;
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    fn cfg(layers: usize) -> Llama32Config {
        Llama32Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            head_dim: None,
            rope_scaling: None,
            embedding_scale: None,
            residual_scale: None,
            attention_scale: None,
            logit_scale: None,
            num_loops: 1,
            skip_loop_final_norm: false,
            rope_style: rlx_ir::RopeStyle::NeoX,
            gguf_arch: None,
            rope_dim: None,
            sliding_window: None,
            sliding_window_pattern: None,
            final_logit_softcap: None,
        }
    }

    /// Every weight a whole model of `layers` layers needs, so a build can
    /// be asked which of them it actually took.
    fn weights(c: &Llama32Config) -> WeightMap {
        let h = c.hidden_size;
        let dh = c.head_dim();
        let q = c.num_attention_heads * dh;
        let kv = c.num_key_value_heads * dh;
        let im = c.intermediate_size;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut put = |k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            // Small distinct values; the graph only has to build, not be right.
            let data = (0..n).map(|i| ((i % 7) as f32 + 1.0) * 0.01).collect();
            t.insert(k, (data, shape));
        };
        put("model.embed_tokens.weight".into(), vec![c.vocab_size, h]);
        for l in 0..c.num_hidden_layers {
            let p = format!("model.layers.{l}");
            put(format!("{p}.input_layernorm.weight"), vec![h]);
            put(format!("{p}.post_attention_layernorm.weight"), vec![h]);
            put(format!("{p}.self_attn.q_proj.weight"), vec![q, h]);
            put(format!("{p}.self_attn.k_proj.weight"), vec![kv, h]);
            put(format!("{p}.self_attn.v_proj.weight"), vec![kv, h]);
            put(format!("{p}.self_attn.o_proj.weight"), vec![h, q]);
            put(format!("{p}.mlp.gate_proj.weight"), vec![im, h]);
            put(format!("{p}.mlp.up_proj.weight"), vec![im, h]);
            put(format!("{p}.mlp.down_proj.weight"), vec![h, im]);
        }
        put("model.norm.weight".into(), vec![h]);
        put("lm_head.weight".into(), vec![c.vocab_size, h]);
        WeightMap::from_tensors(t)
    }

    /// Which `model.layers.N` indices a built graph pulled weights for.
    fn layers_taken(built: &BuiltModel) -> Vec<usize> {
        let mut seen: Vec<usize> = built
            .params()
            .keys()
            .filter_map(|k| k.strip_prefix("model.layers."))
            .filter_map(|rest| rest.split('.').next()?.parse().ok())
            .collect();
        seen.sort_unstable();
        seen.dedup();
        seen
    }

    #[test]
    fn a_block_takes_its_own_layers_weights_and_no_others() {
        // The point of the range: rank 1 of a 2-way split over 8 layers
        // builds four layers, and they are the checkpoint's layers 4..8 —
        // not 0..4 renamed.
        let c = cfg(8);
        let built = Llama32Flow::for_prefill(&c, 1, 3)
            .layer_range(4..8)
            .inputs_embeds()
            .hidden_only()
            .export_kv()
            .build(&mut weights(&c))
            .expect("a middle block should build");
        assert_eq!(layers_taken(&built), vec![4, 5, 6, 7]);
    }

    #[test]
    fn the_first_block_starts_at_zero() {
        let c = cfg(8);
        let built = Llama32Flow::for_prefill(&c, 1, 3)
            .layer_range(0..4)
            .hidden_only()
            .export_kv()
            .build(&mut weights(&c))
            .expect("a first block should build");
        assert_eq!(layers_taken(&built), vec![0, 1, 2, 3]);
    }

    #[test]
    fn no_range_still_builds_the_whole_model() {
        // The default must be untouched: every existing caller passes no
        // range at all.
        let c = cfg(4);
        let built = Llama32Flow::for_prefill(&c, 1, 3)
            .export_kv()
            .build(&mut weights(&c))
            .expect("a whole model should build");
        assert_eq!(layers_taken(&built), vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_decode_block_takes_the_same_layers_as_its_prefill() {
        // They must agree, or the KV a block seeded would be read by a graph
        // over different weights.
        let c = cfg(8);
        let built = Llama32Flow::for_decode(&c, 1, 5)
            .layer_range(4..8)
            .inputs_embeds()
            .hidden_only()
            .build(&mut weights(&c))
            .expect("a middle decode block should build");
        assert_eq!(layers_taken(&built), vec![4, 5, 6, 7]);
    }

    #[test]
    fn a_range_that_is_not_a_slice_of_the_model_is_refused() {
        // Each of these would otherwise build a graph quietly missing
        // layers, which has no shape error to catch it.
        let c = cfg(4);
        for bad in [0..0, 2..2, 0..5, 4..8] {
            let err = Llama32Flow::for_prefill(&c, 1, 3)
                .layer_range(bad.clone())
                .export_kv()
                .build(&mut weights(&c))
                .expect_err(&format!("{bad:?} should be refused"));
            assert!(
                format!("{err:#}").contains("not a slice"),
                "{bad:?}: {err:#}"
            );
        }
    }
}

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

//! Fluent Qwen3.5 assembly — GDN + full-attn trunk via `rlx-flow` plugins.
//!
//! Arch-specific blocks live here; generic stream/GDN primitives stay in `rlx-flow`.

use std::fmt;
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, plugin_named};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut};
use rlx_ir::{DType, Dim, HirNodeId, Shape, sym};

/// Whether to gather token embeddings on the host and feed them as
/// `inputs_embeds`, instead of uploading the full `[vocab, hidden]` F32 table
/// as a resident device param.
///
/// - `RLX_QWEN35_HOST_EMBED=1` / `0` — force on / off
/// - unset — auto-on when `table_nbytes ≥ 1 GiB` (Bonsai-27B: 4.7 GiB), so
///   16 GB CUDA boxes don't OOM parking the table in the arena
///
/// The runner must feed `inputs_embeds` when this is set.
#[allow(dead_code)] // public helper; call sites prefer `_for_bytes`
pub(crate) fn host_embed_enabled() -> bool {
    host_embed_enabled_for_bytes(0)
}

/// Like [`host_embed_enabled`], but passes the F32 table byte size so auto
/// mode can kick in without an env override.
pub(crate) fn host_embed_enabled_for_bytes(table_nbytes: usize) -> bool {
    match rlx_ir::env::var("RLX_QWEN35_HOST_EMBED").as_deref() {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") | Some("yes") => true,
        Some(_) => true,
        None => table_nbytes >= (1usize << 30),
    }
}

use super::builder::{
    PackedParams, Qwen35BsLayout, emit_qwen35_decode_trunk_layer,
    emit_qwen35_full_attn_prefill_layer, emit_qwen35_gather_last_token,
    emit_qwen35_gdn_prefill_layer, emit_qwen35_layer_probe_layer,
    emit_qwen35_prefill_cache_trunk_layer, emit_qwen35_prefill_tail,
};
use super::config::Qwen35Config;
use super::rope;
use super::weights::{Qwen35FullAttnLayer, Qwen35LinearLayer, Qwen35TrunkLayer, Qwen35Weights};
use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::weight_loader::WeightLoader;

type LayerFn = Arc<dyn Fn(Qwen35LayerCtx<'_>) -> FlowStage + Send + Sync>;

const ROPE_COS: &str = "qwen35.rope.cos";
const ROPE_SIN: &str = "qwen35.rope.sin";
const H_PRE_NORM: &str = "qwen35.h_pre_norm";

/// Wire HIR outputs as `[primary, (optional MTP), …layer_side]` for [`crate::cache`]
/// parsers (`logits` / MTP head, then recurrent states in layer order).
fn finish_hir_side_outputs(
    mut built: BuiltModel,
    layer_side: Vec<HirNodeId>,
    mtp: Option<HirNodeId>,
) -> BuiltModel {
    if layer_side.is_empty() && mtp.is_none() {
        return built;
    }
    let hir = built
        .module
        .as_hir_mut()
        .expect("finish_hir_side_outputs requires HIR stage");
    let primary = hir.outputs[0];
    let mut outputs = vec![primary];
    if let Some(m) = mtp {
        outputs.push(m);
    }
    outputs.extend(layer_side);
    hir.set_outputs(outputs);
    built
}

/// Per-layer override hook — default emits GDN or full-attn trunk blocks.
pub struct Qwen35LayerCtx<'a> {
    pub index: usize,
    pub cfg: &'a Qwen35Config,
    pub weights: &'a Qwen35Weights,
    pub batch: usize,
    pub seq: usize,
}

impl Qwen35LayerCtx<'_> {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn default_stage(&self) -> FlowStage {
        self.default_stage_with_packed(None)
    }

    /// `packed_arc` variant of [`Self::default_stage`] — forwards the
    /// shared sink into the per-layer plugins so K-quant trunk weights
    /// reach the runner's packed-upload step. Without it, packed
    /// `MatWeight` references inside `Qwen35Weights` get registered
    /// but the byte payloads are never uploaded → all-zero matmul
    /// outputs at runtime.
    pub fn default_stage_with_packed(
        &self,
        packed_arc: Option<Arc<Mutex<PackedParams>>>,
    ) -> FlowStage {
        let bs = Qwen35BsLayout::new(self.batch, self.seq, false);
        let out = hidden_shape(self.batch, self.seq, self.cfg.hidden_size);
        match self.weights.trunk_layers.get(self.index) {
            Some(Qwen35TrunkLayer::Linear(lin)) => {
                gdn_layer_plugin_with_packed(self.index, self.cfg, lin.clone(), bs, out, packed_arc)
            }
            Some(Qwen35TrunkLayer::FullAttn(fa)) => full_attn_layer_plugin_with_packed(
                self.index,
                self.cfg,
                fa.clone(),
                bs,
                out,
                packed_arc,
            ),
            None => plugin_named(format!("qwen35.empty{}", self.index), |_emit, input| {
                Ok(input)
            }),
        }
    }
}

/// Tier-0 Qwen3.5 flow builder.
#[derive(Clone)]
pub struct Qwen35Flow<'a> {
    cfg: &'a Qwen35Config,
    weights: Option<&'a Qwen35Weights>,
    batch: usize,
    seq: usize,
    with_embed: bool,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    runtime_mrope: bool,
    fast_mtp: bool,
    export_normed_hidden: bool,
    single_gdn: Option<(usize, Qwen35LinearLayer)>,
    layer_fn: Option<LayerFn>,
    profile: Option<CompileProfile>,
}

impl fmt::Debug for Qwen35Flow<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Qwen35Flow")
            .field("batch", &self.batch)
            .field("seq", &self.seq)
            .field("with_embed", &self.with_embed)
            .field("with_lm_head", &self.with_lm_head)
            .field("last_logits_only", &self.last_logits_only)
            .field("enable_mtp_head", &self.enable_mtp_head)
            .field("single_gdn", &self.single_gdn.as_ref().map(|(i, _)| i))
            .field("layer_fn", &self.layer_fn.is_some())
            .field("profile", &self.profile)
            .finish_non_exhaustive()
    }
}

impl<'a> Qwen35Flow<'a> {
    pub fn new(cfg: &'a Qwen35Config) -> Self {
        Self {
            cfg,
            weights: None,
            batch: 1,
            seq: 128,
            with_embed: false,
            with_lm_head: false,
            last_logits_only: false,
            enable_mtp_head: false,
            runtime_mrope: false,
            fast_mtp: false,
            export_normed_hidden: false,
            single_gdn: None,
            layer_fn: None,
            profile: None,
        }
    }

    pub fn prefill(
        cfg: &'a Qwen35Config,
        weights: &'a Qwen35Weights,
        batch: usize,
        seq: usize,
    ) -> Self {
        Self::new(cfg)
            .weights(weights)
            .batch(batch)
            .seq(seq)
            .with_embed()
    }

    /// Build a single GDN trunk layer (hidden states in → hidden out).
    pub fn one_gdn_layer(
        cfg: &'a Qwen35Config,
        lin: Qwen35LinearLayer,
        layer_idx: usize,
        batch: usize,
        seq: usize,
    ) -> Self {
        Self {
            cfg,
            weights: None,
            batch,
            seq,
            with_embed: false,
            with_lm_head: false,
            last_logits_only: false,
            enable_mtp_head: false,
            runtime_mrope: false,
            fast_mtp: false,
            export_normed_hidden: false,
            single_gdn: Some((layer_idx, lin)),
            layer_fn: None,
            profile: None,
        }
    }

    pub fn weights(mut self, weights: &'a Qwen35Weights) -> Self {
        self.weights = Some(weights);
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

    pub fn with_embed(mut self) -> Self {
        self.with_embed = true;
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

    pub fn mtp_head(mut self) -> Self {
        self.enable_mtp_head = true;
        self
    }

    /// Use runtime `rope_cos` / `rope_sin` graph inputs instead of baked tables.
    pub fn runtime_mrope(mut self) -> Self {
        self.runtime_mrope = true;
        self
    }

    pub fn fast_mtp(mut self) -> Self {
        self.fast_mtp = true;
        self
    }

    /// Last-token index input without enabling the LM head.
    pub fn last_token_index(mut self) -> Self {
        self.last_logits_only = true;
        self
    }

    /// Export final RMS-normed hidden (fast-greedy LM on host) instead of logits.
    pub fn export_normed_hidden(mut self) -> Self {
        self.export_normed_hidden = true;
        self.with_lm_head = false;
        self
    }

    pub fn profile(mut self, profile: CompileProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    pub fn layer<F>(mut self, f: F) -> Self
    where
        F: Fn(Qwen35LayerCtx<'_>) -> FlowStage + Send + Sync + 'static,
    {
        self.layer_fn = Some(Arc::new(f));
        self
    }

    pub fn build(self, weights: &mut dyn WeightLoader) -> Result<(BuiltModel, PackedParams)> {
        self.build_inner(&mut WeightLoaderSource(weights))
    }

    pub fn build_with_weights(
        self,
        weights: &'a Qwen35Weights,
    ) -> Result<(BuiltModel, PackedParams)> {
        let mut inline = InlineQwen35Weights {
            weights,
            cfg: self.cfg,
        };
        self.build_inner(&mut inline)
    }

    fn build_inner(
        self,
        source: &mut dyn rlx_flow::WeightSource,
    ) -> Result<(BuiltModel, PackedParams)> {
        let cfg = self.cfg.clone();
        let h = cfg.hidden_size;
        let hidden = hidden_shape(self.batch, self.seq, h);
        let profile = self.profile.unwrap_or_default();
        let batch = self.batch;
        let seq = self.seq;
        let with_lm_head = self.with_lm_head;
        let last_logits_only = self.last_logits_only;
        let enable_mtp_head = self.enable_mtp_head;
        let runtime_mrope = self.runtime_mrope;
        let fast_mtp = self.fast_mtp;
        let export_normed_hidden = self.export_normed_hidden;
        let need_last_idx =
            last_logits_only && (with_lm_head || enable_mtp_head || export_normed_hidden);

        if let Some((layer_idx, lin)) = self.single_gdn {
            let built = ModelFlow::new("qwen35_gdn")
                .with_profile(profile)
                .input("hidden", hidden.clone())
                .raw_stage(gdn_layer_plugin(
                    layer_idx,
                    &cfg,
                    lin,
                    Qwen35BsLayout::new(batch, seq, false),
                    hidden,
                ))
                .output("hidden")
                .build(source)?;
            return Ok((built, PackedParams::new()));
        }

        let weights_ref = self
            .weights
            .ok_or_else(|| anyhow::anyhow!("Qwen35Flow: call .weights() for prefill builds"))?;
        let weights = weights_ref.clone();

        let trunk_count = cfg
            .num_hidden_layers
            .saturating_sub(cfg.nextn_predict_layers);

        let head_half = cfg.key_length / 2;
        let max_pos = cfg.max_position_embeddings;

        let flow_name = if runtime_mrope {
            "qwen35_runtime_mrope_prefill"
        } else {
            "qwen35_prefill"
        };
        let f = DType::F32;
        let rope_shape = Shape::new(&[seq, head_half], f);

        let mut flow = ModelFlow::new(flow_name)
            .with_profile(profile)
            .input("input_ids", Shape::new(&[batch, seq], f));

        if runtime_mrope {
            flow = flow
                .input("rope_cos", rope_shape.clone())
                .input("rope_sin", rope_shape);
        }

        if need_last_idx {
            flow = flow.input("last_token_idx", Shape::new(&[batch], f));
        }

        if self.with_embed {
            // Off-load the (large-vocab) F32 embedding table to host RAM and
            // feed gathered rows as `inputs_embeds` when host-embed is on
            // (Bonsai-27B: token_embd [248320,5120] F32 = 4.7 GiB — auto).
            let emb_bytes = cfg.vocab_size.saturating_mul(h).saturating_mul(4);
            flow = if host_embed_enabled_for_bytes(emb_bytes) {
                flow.embed_host("token_embd.weight", h)
            } else {
                flow.embed("token_embd.weight")
            };
        } else {
            flow = flow.input("hidden", hidden.clone());
        }

        if runtime_mrope {
            flow = flow.plugin_named("qwen35.rope.bind", move |emit, input| {
                emit.set_named(ROPE_COS, emit.flow_input("rope_cos")?.hir_id());
                emit.set_named(ROPE_SIN, emit.flow_input("rope_sin")?.hir_id());
                Ok(input)
            });
        } else {
            flow = flow.plugin_named("qwen35.rope", {
                let cfg = cfg.clone();
                move |emit, input| {
                    let (cos_data, sin_data) = rope::build_mrope_tables(&cfg, max_pos, head_half);
                    let cos_shape = Shape::new(&[max_pos, head_half], f);
                    let sin_shape = cos_shape.clone();
                    let cos_id = emit.synth_param(ROPE_COS, cos_data, cos_shape);
                    let sin_id = emit.synth_param(ROPE_SIN, sin_data, sin_shape);
                    emit.set_named(ROPE_COS, cos_id);
                    emit.set_named(ROPE_SIN, sin_id);
                    Ok(input)
                }
            });
        }

        flow = flow.plugin_named("qwen35.snapshot", |emit, input| {
            let h = input.ok_or_else(|| anyhow::anyhow!("snapshot requires hidden"))?;
            emit.set_named(H_PRE_NORM, h.hir_id());
            Ok(Some(h))
        });

        let layer_fn = self.layer_fn.clone();
        let cfg_layers = cfg.clone();
        let weights_layers = weights.clone();
        // Reserve the shared packed sink early so the trunk plugins
        // populate it from inside the closure (Arc<Mutex<...>> lets us
        // drain it after `flow.build(...)` below). Without this every
        // K-quant trunk weight registered via `proj_mat(.., &mut local)`
        // gets dropped at the closure boundary.
        let packed_sink: Arc<Mutex<PackedParams>> = Arc::new(Mutex::new(PackedParams::new()));
        let packed_for_layers = packed_sink.clone();
        flow = flow.repeat_layers(trunk_count, move |i| {
            if let Some(ref f) = layer_fn {
                return f(Qwen35LayerCtx {
                    index: i,
                    cfg: &cfg_layers,
                    weights: &weights_layers,
                    batch,
                    seq,
                });
            }
            Qwen35LayerCtx {
                index: i,
                cfg: &cfg_layers,
                weights: &weights_layers,
                batch,
                seq,
            }
            .default_stage_with_packed(Some(packed_for_layers.clone()))
        });

        if need_last_idx {
            flow = flow.gather_last_token_dynamic(batch);
        }

        let mtp_sink: Arc<Mutex<Vec<HirNodeId>>> = Arc::new(Mutex::new(Vec::new()));
        let mtp_out = mtp_sink.clone();
        // Reuse the same `packed_sink` created earlier for the trunk
        // layers — the tail closure (LM head, MTP head) adds its own
        // packed K-quant references to the same sink so a single
        // drain at the end covers the whole graph.
        let packed_out = packed_sink.clone();
        let n_vocab = if weights.token_embd.is_empty() {
            cfg.vocab_size
        } else {
            weights.token_embd.len() / h
        };

        flow = flow.plugin_named("qwen35.tail", {
            let cfg = cfg.clone();
            let weights = weights.clone();
            let n_embd = h;
            move |emit, hidden_in| {
                let h_for_lm = hidden_in.ok_or_else(|| anyhow::anyhow!("tail requires hidden"))?;
                let h_pre = emit.named(H_PRE_NORM)?;
                let input_ids = emit.flow_input("input_ids")?.hir_id();
                let cos = emit.named(ROPE_COS)?;
                let sin = emit.named(ROPE_SIN)?;
                let last_idx = if need_last_idx {
                    Some(emit.flow_input("last_token_idx")?.hir_id())
                } else {
                    None
                };
                let mut packed_guard = packed_out.lock().expect("packed sink");
                let hir = emit
                    .module
                    .as_hir_mut()
                    .expect("qwen35 prefill flow requires HIR stage");
                let mut gb = HirMut::new(hir);
                let (logits, mtp, normed) = emit_qwen35_prefill_tail(
                    &mut gb,
                    emit.params,
                    &mut packed_guard,
                    &cfg,
                    &weights,
                    batch,
                    seq,
                    h_for_lm.hir_id(),
                    h_pre,
                    input_ids,
                    cos,
                    sin,
                    with_lm_head,
                    enable_mtp_head,
                    export_normed_hidden,
                    fast_mtp,
                    last_idx,
                )?;
                if let Some(mtp_id) = mtp {
                    mtp_out.lock().expect("mtp sink").push(mtp_id);
                }
                let primary = if export_normed_hidden {
                    normed.expect("export_normed_hidden requires normed output")
                } else {
                    logits.unwrap_or(h_for_lm.hir_id())
                };
                let logit_rows = if last_logits_only { 1 } else { seq };
                let out_shape = if export_normed_hidden {
                    Shape::new(&[batch, logit_rows, n_embd], DType::F32)
                } else if with_lm_head {
                    Shape::new(&[batch, logit_rows, n_vocab], DType::F32)
                } else {
                    h_for_lm.shape.clone()
                };
                Ok(Some(emit.wrap(primary, out_shape)))
            }
        });

        if export_normed_hidden || !with_lm_head {
            flow = flow.output("hidden");
        } else {
            flow = flow.output("logits");
        }

        let mut built = flow.build(source)?;
        let extra: Vec<HirNodeId> = mtp_sink.lock().expect("mtp sink").clone();
        if !extra.is_empty() {
            built = built.with_extra_hir_outputs(extra);
        }
        let packed = std::mem::take(&mut *packed_sink.lock().expect("packed sink"));
        Ok((built, packed))
    }
}

/// Shared [`Qwen35Flow`] prefill builder (standard / runtime MRoPE / export normed hidden).
#[allow(clippy::too_many_arguments)]
pub fn build_qwen35_prefill_flow_built(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    runtime_mrope: bool,
    fast_mtp: bool,
    export_normed_hidden: bool,
) -> Result<(BuiltModel, PackedParams)> {
    let mut flow = Qwen35Flow::prefill(cfg, weights, batch, seq);
    if runtime_mrope {
        flow = flow.runtime_mrope();
    }
    if export_normed_hidden {
        flow = flow.export_normed_hidden();
    } else if with_lm_head {
        flow = flow.lm_head();
    }
    if fast_mtp {
        flow = flow.fast_mtp();
    }
    if enable_mtp_head {
        flow = flow.mtp_head();
    }
    if last_logits_only {
        flow = if export_normed_hidden {
            flow.last_token_index()
        } else if with_lm_head {
            flow.last_token_logits()
        } else {
            flow.last_token_index()
        };
    }
    flow.build_with_weights(weights)
}

/// Standard prefill HIR via [`Qwen35Flow`].
#[allow(clippy::too_many_arguments)]
pub fn build_qwen35_prefill_flow_ext(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    runtime_mrope: bool,
    fast_mtp: bool,
    export_normed_hidden: bool,
) -> Result<(
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    let (built, packed) = build_qwen35_prefill_flow_built(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        last_logits_only,
        enable_mtp_head,
        runtime_mrope,
        fast_mtp,
        export_normed_hidden,
    )?;
    built
        .into_parts()
        .map(|(hir, params)| (hir, params, packed))
}

/// Standard prefill [`BuiltModel`] via [`Qwen35Flow`].
pub fn build_qwen35_prefill_built(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
) -> Result<BuiltModel> {
    build_qwen35_prefill_flow_built(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        last_logits_only,
        enable_mtp_head,
        false,
        false,
        false,
    )
    .map(|(built, _)| built)
}

/// Standard prefill assembly via [`Qwen35Flow`] (no decode / dynamic / export flags).
pub fn build_qwen35_prefill_flow(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
) -> Result<(
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    build_qwen35_prefill_flow_ext(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        last_logits_only,
        enable_mtp_head,
        false,
        false,
        false,
    )
}

/// Options for a single-token Qwen3.5 decode step (GDN recurrent + full-attn KV).
#[derive(Debug, Clone)]
pub struct Qwen35DecodeOpts {
    pub batch: usize,
    pub past_seq: usize,
    pub dynamic_past: bool,
    pub use_custom_mask: bool,
    pub enable_mtp_head: bool,
    pub fast_mtp: bool,
    pub fast_greedy_lm_head: bool,
    pub profile: Option<CompileProfile>,
}

impl Qwen35DecodeOpts {
    pub fn step(batch: usize, past_seq: usize) -> Self {
        Self {
            batch,
            past_seq,
            dynamic_past: false,
            use_custom_mask: false,
            enable_mtp_head: false,
            fast_mtp: false,
            fast_greedy_lm_head: false,
            profile: None,
        }
    }
}

/// Decode-mode fluent builder — native decode HIR assembly (GDN + KV cache).
#[derive(Clone)]
pub struct Qwen35DecodeFlow<'a> {
    cfg: &'a Qwen35Config,
    weights: &'a Qwen35Weights,
    opts: Qwen35DecodeOpts,
}

impl<'a> Qwen35DecodeFlow<'a> {
    pub fn new(cfg: &'a Qwen35Config, weights: &'a Qwen35Weights, past_seq: usize) -> Self {
        Self {
            cfg,
            weights,
            opts: Qwen35DecodeOpts::step(1, past_seq),
        }
    }

    pub fn batch(mut self, batch: usize) -> Self {
        self.opts.batch = batch;
        self
    }

    pub fn custom_mask(mut self) -> Self {
        self.opts.use_custom_mask = true;
        self
    }

    pub fn dynamic_past(mut self) -> Self {
        self.opts.dynamic_past = true;
        self
    }

    pub fn mtp_head(mut self) -> Self {
        self.opts.enable_mtp_head = true;
        self
    }

    pub fn profile(mut self, profile: CompileProfile) -> Self {
        self.opts.profile = Some(profile);
        self
    }

    pub fn build(self) -> Result<(BuiltModel, PackedParams)> {
        build_qwen35_decode_built(
            self.cfg,
            std::sync::Arc::new(self.weights.clone()),
            &self.opts,
        )
    }
}

/// Native decode assembly via [`ModelFlow`] plugins + shared [`super::builder`] emit helpers.
pub fn build_qwen35_decode_model_flow(
    cfg: &Qwen35Config,
    weights: std::sync::Arc<Qwen35Weights>,
    opts: &Qwen35DecodeOpts,
) -> Result<(
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    let batch = opts.batch;
    let seq = 1usize;
    let past_len = opts.past_seq;
    let with_lm_head = !opts.fast_greedy_lm_head;
    let head_half = cfg.key_length / 2;
    let f = DType::F32;
    let hidden = hidden_shape(batch, seq, cfg.hidden_size);
    let ids_shape = Shape::new(&[batch, seq], f);
    let rope_shape = Shape::new(&[1, head_half], f);

    let cfg_c = cfg.clone();
    let weights_c = weights.clone();
    let recur_sink: Arc<Mutex<Vec<HirNodeId>>> = Arc::new(Mutex::new(Vec::new()));
    let mtp_sink: Arc<Mutex<Option<HirNodeId>>> = Arc::new(Mutex::new(None));
    let packed_sink: Arc<Mutex<PackedParams>> = Arc::new(Mutex::new(PackedParams::new()));
    let use_mask = opts.use_custom_mask;
    let dynamic_past = opts.dynamic_past;
    let enable_mtp = opts.enable_mtp_head;
    let fast_mtp = opts.fast_mtp;

    let mut flow = ModelFlow::new("qwen35_decode")
        .input("rope_cos", rope_shape.clone())
        .input("rope_sin", rope_shape)
        .input("input_ids", ids_shape);

    if use_mask {
        flow = flow.input("mask", Shape::new(&[batch, past_len + seq], f));
    }

    // Host-gathered embeddings: decode's new-token embedding is fed as
    // `inputs_embeds` instead of registering the full F32 token_embd table
    // (Bonsai-27B 4.7 GiB) into the decode arena — which, being >4 GiB,
    // otherwise mis-addresses the u32 gather offset.
    let host_embed = host_embed_enabled_for_bytes(weights_c.token_embd.len() * 4);
    if host_embed {
        flow = flow.input("inputs_embeds", hidden.clone());
    }

    let weights_embed = weights_c.clone();
    let cfg_embed = cfg_c.clone();
    flow = flow.plugin_named("qwen35.decode.embed", move |emit, _| {
        let ids = emit.flow_input("input_ids")?.hir_id();
        if host_embed {
            let e = emit.flow_input("inputs_embeds")?.hir_id();
            emit.set_named("qwen35.decode.input_ids", ids);
            return Ok(Some(emit.wrap(e, hidden.clone())));
        }
        let hir = emit
            .module
            .as_hir_mut()
            .expect("qwen35 decode flow requires HIR stage");
        let mut gb = HirMut::new(hir);
        let n_embd = cfg_embed.hidden_size;
        let n_vocab = weights_embed.lm_vocab_size(&cfg_embed);
        let embed_w = super::builder::register_param(
            &mut gb,
            emit.params,
            "token_embd.weight",
            weights_embed.token_embd.to_vec(),
            Shape::new(&[n_vocab, n_embd], f),
        );
        let h = gb.gather_(embed_w, ids, 0);
        emit.set_named("qwen35.decode.input_ids", ids);
        Ok(Some(emit.wrap(h, hidden.clone())))
    });

    let trunk_count = weights_c.trunk_layers.len();
    let weights_layers = weights_c.clone();
    let cfg_layers = cfg_c.clone();
    let recur = recur_sink.clone();
    let packed_arc = packed_sink.clone();
    flow = flow.repeat_layers(trunk_count, move |il| {
        let cfg = cfg_layers.clone();
        let weights = weights_layers.clone();
        let recur = recur.clone();
        let packed_arc = packed_arc.clone();
        // Borrow layer via Arc — avoid cloning the full weight bundle per layer.
        let layer_il = il;
        let out_shape = hidden_shape(batch, seq, cfg.hidden_size);
        plugin_named(format!("qwen35.decode.l{il}"), move |emit, input| {
            let hidden = input.ok_or_else(|| anyhow::anyhow!("decode layer requires hidden"))?;
            let cos = emit.flow_input("rope_cos")?.hir_id();
            let sin = emit.flow_input("rope_sin")?.hir_id();
            let mask = if use_mask {
                Some(emit.flow_input("mask")?.hir_id())
            } else {
                None
            };
            let hir = emit
                .module
                .as_hir_mut()
                .expect("qwen35 decode flow requires HIR stage");
            let mut gb = HirMut::new(hir);
            let mut layer_recur = Vec::new();
            let mut packed = packed_arc.lock().expect("packed sink");
            let h = emit_qwen35_decode_trunk_layer(
                &mut gb,
                emit.params,
                &mut packed,
                &cfg,
                layer_il,
                &weights.trunk_layers[layer_il],
                Qwen35BsLayout::new(batch, seq, false),
                hidden.hir_id(),
                cos,
                sin,
                past_len,
                dynamic_past,
                mask,
                &mut layer_recur,
            )?;
            recur.lock().expect("recur sink").extend(layer_recur);
            Ok(Some(emit.wrap(h, out_shape.clone())))
        })
    });

    let weights_tail = weights_c.clone();
    let cfg_tail = cfg_c.clone();
    let mtp_out = mtp_sink.clone();
    let packed_tail = packed_sink.clone();
    flow = flow.plugin_named("qwen35.decode.tail", move |emit, input| {
        let hidden = input.ok_or_else(|| anyhow::anyhow!("decode tail requires hidden"))?;
        let cos = emit.flow_input("rope_cos")?.hir_id();
        let sin = emit.flow_input("rope_sin")?.hir_id();
        let input_ids = emit.named("qwen35.decode.input_ids")?;
        let hir = emit
            .module
            .as_hir_mut()
            .expect("qwen35 decode flow requires HIR stage");
        let mut gb = HirMut::new(hir);
        let mut packed = packed_tail.lock().expect("packed sink");
        let h_pre = hidden.hir_id();
        let h_lm = if with_lm_head {
            gb.narrow_(h_pre, 1, seq - 1, 1)
        } else {
            h_pre
        };
        let (logits, mtp, _) = emit_qwen35_prefill_tail(
            &mut gb,
            emit.params,
            &mut packed,
            &cfg_tail,
            &weights_tail,
            batch,
            seq,
            h_lm,
            h_pre,
            input_ids,
            cos,
            sin,
            with_lm_head,
            enable_mtp,
            false,
            fast_mtp,
            None,
        )?;
        if let Some(mtp_id) = mtp {
            *mtp_out.lock().expect("mtp sink") = Some(mtp_id);
        }
        let primary = logits.unwrap_or(h_lm);
        let out_shape = if with_lm_head {
            Shape::new(&[batch, 1, weights_tail.lm_vocab_size(&cfg_tail)], f)
        } else {
            hidden.shape.clone()
        };
        Ok(Some(emit.wrap(primary, out_shape)))
    });

    if with_lm_head {
        flow = flow.output("logits");
    } else {
        flow = flow.output("hidden");
    }

    let mut built = flow.build(&mut InlineQwen35Weights {
        weights: &weights_c,
        cfg: &cfg_c,
    })?;
    let layer_side = recur_sink.lock().expect("recur sink").clone();
    let mtp = mtp_sink.lock().expect("mtp sink").take();
    built = finish_hir_side_outputs(built, layer_side, mtp);
    let packed = packed_sink.lock().expect("packed sink").clone();
    built
        .into_parts()
        .map(|(hir, params)| (hir, params, packed))
}

pub fn build_qwen35_decode_flow(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    opts: &Qwen35DecodeOpts,
) -> Result<(
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    build_qwen35_decode_model_flow(cfg, weights.into(), opts)
}

pub fn build_qwen35_decode_built(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    opts: &Qwen35DecodeOpts,
) -> Result<(BuiltModel, PackedParams)> {
    let (hir, params, packed) = build_qwen35_decode_flow(cfg, weights, opts)?;
    let mut built = rlx_core::flow_util::built_from_hir(hir, params)?;
    if let Some(profile) = opts.profile.clone() {
        built.profile = profile;
    }
    Ok((built, packed))
}

/// Options for a Qwen3.5 prefill-cache graph (seeds decode recurrent state).
#[derive(Debug, Clone)]
pub struct Qwen35PrefillCacheOpts {
    pub batch: usize,
    pub seq: usize,
    pub with_lm_head: bool,
    pub runtime_mrope: bool,
    pub dynamic_seq: bool,
    pub prefill_from_hidden: bool,
    pub enable_mtp_head: bool,
    pub fast_mtp: bool,
    pub fast_greedy_lm_head: bool,
    pub profile: Option<CompileProfile>,
}

impl Qwen35PrefillCacheOpts {
    pub fn static_cache(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            with_lm_head: true,
            runtime_mrope: false,
            dynamic_seq: false,
            prefill_from_hidden: false,
            enable_mtp_head: false,
            fast_mtp: false,
            fast_greedy_lm_head: false,
            profile: None,
        }
    }
}

#[derive(Clone)]
pub struct Qwen35PrefillCacheFlow<'a> {
    cfg: &'a Qwen35Config,
    weights: std::sync::Arc<Qwen35Weights>,
    opts: Qwen35PrefillCacheOpts,
}

impl<'a> Qwen35PrefillCacheFlow<'a> {
    pub fn new(
        cfg: &'a Qwen35Config,
        weights: impl Into<std::sync::Arc<Qwen35Weights>>,
        seq: usize,
    ) -> Self {
        Self {
            cfg,
            weights: weights.into(),
            opts: Qwen35PrefillCacheOpts::static_cache(1, seq),
        }
    }

    pub fn runtime_mrope(mut self) -> Self {
        self.opts.runtime_mrope = true;
        self
    }

    pub fn dynamic_seq(mut self) -> Self {
        self.opts.dynamic_seq = true;
        self
    }

    pub fn from_hidden(mut self) -> Self {
        self.opts.prefill_from_hidden = true;
        self
    }

    pub fn mtp_head(mut self) -> Self {
        self.opts.enable_mtp_head = true;
        self
    }

    pub fn build(self) -> Result<(BuiltModel, PackedParams)> {
        build_qwen35_prefill_cache_built(self.cfg, self.weights, &self.opts)
    }
}

/// Native prefill-cache assembly via [`ModelFlow`] (GDN + full-attn K/V export).
pub fn build_qwen35_prefill_cache_model_flow(
    cfg: &Qwen35Config,
    weights: std::sync::Arc<Qwen35Weights>,
    opts: &Qwen35PrefillCacheOpts,
) -> Result<(
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    let batch = opts.batch;
    let seq = opts.seq;
    let dynamic_seq = opts.dynamic_seq;
    let runtime_mrope = opts.runtime_mrope;
    let prefill_from_hidden = opts.prefill_from_hidden;
    let with_lm_head = opts.with_lm_head;
    let export_normed_hidden = opts.fast_greedy_lm_head;
    let enable_mtp = opts.enable_mtp_head;
    let fast_mtp = opts.fast_mtp;
    let need_last_idx = with_lm_head || export_normed_hidden;

    let head_half = cfg.key_length / 2;
    let max_pos = cfg.max_position_embeddings;
    let f = DType::F32;
    let n_embd = cfg.hidden_size;
    let hidden_shape_val = if dynamic_seq {
        Shape::from_dims(
            &[
                Dim::Static(batch),
                Dim::Dynamic(sym::SEQ),
                Dim::Static(n_embd),
            ],
            f,
        )
    } else {
        hidden_shape(batch, seq, n_embd)
    };
    let ids_shape = if dynamic_seq {
        Shape::from_dims(&[Dim::Static(batch), Dim::Dynamic(sym::SEQ)], f)
    } else {
        Shape::new(&[batch, seq], f)
    };

    let cfg_c = cfg.clone();
    let weights_c = weights.clone();
    let recur_sink: Arc<Mutex<Vec<HirNodeId>>> = Arc::new(Mutex::new(Vec::new()));
    let mtp_sink: Arc<Mutex<Option<HirNodeId>>> = Arc::new(Mutex::new(None));
    let packed_sink: Arc<Mutex<PackedParams>> = Arc::new(Mutex::new(PackedParams::new()));

    let mut flow = ModelFlow::new("qwen35_prefill_cache");

    if !prefill_from_hidden || enable_mtp {
        flow = flow.input("input_ids", ids_shape.clone());
    }
    // Host-gathered embeddings: feed `inputs_embeds` instead of uploading the
    // full `[vocab, hidden]` F32 token_embd table as a resident device param
    // (Bonsai-27B: [248320,5120] = 4.7 GiB). input_ids stays declared — it's
    // still used for positions/masking downstream.
    let host_embed =
        host_embed_enabled_for_bytes(weights_c.token_embd.len() * 4) && !prefill_from_hidden;
    if host_embed {
        flow = flow.input("inputs_embeds", hidden_shape_val.clone());
    }
    if prefill_from_hidden {
        flow = flow.input("prefill_hidden", hidden_shape_val.clone());
    }
    if need_last_idx {
        flow = flow.input("last_token_idx", Shape::new(&[batch], f));
    }
    if runtime_mrope {
        let rope_shape = if dynamic_seq {
            Shape::from_dims(&[Dim::Dynamic(sym::SEQ), Dim::Static(head_half)], f)
        } else {
            Shape::new(&[seq, head_half], f)
        };
        flow = flow
            .input("rope_cos", rope_shape.clone())
            .input("rope_sin", rope_shape);
    }

    let weights_embed = weights_c.clone();
    let cfg_embed = cfg_c.clone();
    let hidden_out = hidden_shape_val.clone();
    flow = flow.plugin_named("qwen35.prefill_cache.embed", move |emit, _| {
        let prefill_h = prefill_from_hidden
            .then(|| emit.flow_input("prefill_hidden").map(|v| v.hir_id()))
            .transpose()?;
        let ids_opt = if prefill_from_hidden {
            enable_mtp
                .then(|| emit.flow_input("input_ids").map(|v| v.hir_id()))
                .transpose()?
        } else {
            let ids = emit.flow_input("input_ids")?.hir_id();
            Some(ids)
        };
        let hir = emit
            .module
            .as_hir_mut()
            .expect("qwen35 prefill-cache flow requires HIR stage");
        let mut gb = HirMut::new(hir);
        let h = if prefill_from_hidden {
            // Hidden states are host-built and fed as `prefill_hidden`. Do **not**
            // register `token_embd.weight` into the compiled graph — that table is
            // unused on this path and costs hundreds of MiB–1 GiB on CUDA.
            if weights_embed.token_embd.is_empty() {
                return Err(anyhow::anyhow!(
                    "qwen35: prefill_from_hidden requires token_embd"
                ));
            }
            let _ = weights_embed.lm_vocab_size(&cfg_embed);
            prefill_h.expect("prefill_hidden")
        } else if host_embed {
            // Embeddings gathered on the host, fed as `inputs_embeds` — the
            // token_embd table never becomes a resident device param.
            emit.flow_input("inputs_embeds")?.hir_id()
        } else {
            let ids = ids_opt.expect("input_ids");
            let n_vocab = weights_embed.lm_vocab_size(&cfg_embed);
            let embed_w = super::builder::register_param(
                &mut gb,
                emit.params,
                "token_embd.weight",
                weights_embed.token_embd.to_vec(),
                Shape::new(&[n_vocab, n_embd], f),
            );
            gb.gather_(embed_w, ids, 0)
        };
        if let Some(ids) = ids_opt {
            emit.set_named("qwen35.prefill_cache.input_ids", ids);
        }
        Ok(Some(emit.wrap(h, hidden_out.clone())))
    });

    if !runtime_mrope {
        let cfg_rope = cfg_c.clone();
        flow = flow.plugin_named("qwen35.rope", move |emit, input| {
            let (cos_data, sin_data) = rope::build_mrope_tables(&cfg_rope, max_pos, head_half);
            let cos_shape = Shape::new(&[max_pos, head_half], f);
            let sin_shape = cos_shape.clone();
            let cos_id = emit.synth_param(ROPE_COS, cos_data, cos_shape);
            let sin_id = emit.synth_param(ROPE_SIN, sin_data, sin_shape);
            emit.set_named(ROPE_COS, cos_id);
            emit.set_named(ROPE_SIN, sin_id);
            Ok(input)
        });
    } else {
        flow = flow.plugin_named("qwen35.rope.bind", move |emit, input| {
            emit.set_named(ROPE_COS, emit.flow_input("rope_cos")?.hir_id());
            emit.set_named(ROPE_SIN, emit.flow_input("rope_sin")?.hir_id());
            Ok(input)
        });
    }

    flow = flow.plugin_named("qwen35.snapshot", |emit, input| {
        let h = input.ok_or_else(|| anyhow::anyhow!("snapshot requires hidden"))?;
        emit.set_named(H_PRE_NORM, h.hir_id());
        Ok(Some(h))
    });

    let trunk_count = weights_c.trunk_layers.len();
    let weights_layers = weights_c.clone();
    let cfg_layers = cfg_c.clone();
    let recur = recur_sink.clone();
    let packed_arc = packed_sink.clone();
    flow = flow.repeat_layers(trunk_count, move |il| {
        let cfg = cfg_layers.clone();
        let weights = weights_layers.clone();
        let recur = recur.clone();
        let packed_arc = packed_arc.clone();
        // Borrow layer via Arc — avoid cloning the full weight bundle per layer.
        let layer_il = il;
        let layer_out = hidden_shape_val.clone();
        plugin_named(format!("qwen35.prefill_cache.l{il}"), move |emit, input| {
            let hidden =
                input.ok_or_else(|| anyhow::anyhow!("prefill-cache layer requires hidden"))?;
            let cos = if runtime_mrope {
                emit.flow_input("rope_cos")?.hir_id()
            } else {
                emit.named(ROPE_COS)?
            };
            let sin = if runtime_mrope {
                emit.flow_input("rope_sin")?.hir_id()
            } else {
                emit.named(ROPE_SIN)?
            };
            let h_in = hidden.hir_id();
            let hir = emit
                .module
                .as_hir_mut()
                .expect("qwen35 prefill-cache flow requires HIR stage");
            let mut gb = HirMut::new(hir);
            let mut layer_recur = Vec::new();
            let mut packed = packed_arc.lock().expect("packed sink");
            let h = emit_qwen35_prefill_cache_trunk_layer(
                &mut gb,
                emit.params,
                &mut packed,
                &cfg,
                layer_il,
                &weights.trunk_layers[layer_il],
                Qwen35BsLayout::new(batch, seq, dynamic_seq),
                h_in,
                cos,
                sin,
                &mut layer_recur,
            )?;
            recur.lock().expect("recur sink").extend(layer_recur);
            Ok(Some(emit.wrap(h, layer_out.clone())))
        })
    });

    if need_last_idx {
        flow = flow.gather_last_token_dynamic(batch);
    }

    let weights_tail = weights_c.clone();
    let cfg_tail = cfg_c.clone();
    let mtp_out = mtp_sink.clone();
    let packed_tail = packed_sink.clone();
    flow = flow.plugin_named("qwen35.prefill_cache.tail", move |emit, input| {
        let hidden = input.ok_or_else(|| anyhow::anyhow!("prefill-cache tail requires hidden"))?;
        let cos = if runtime_mrope {
            emit.flow_input("rope_cos")?.hir_id()
        } else {
            emit.named(ROPE_COS)?
        };
        let sin = if runtime_mrope {
            emit.flow_input("rope_sin")?.hir_id()
        } else {
            emit.named(ROPE_SIN)?
        };
        let input_ids = if enable_mtp {
            emit.named("qwen35.prefill_cache.input_ids")?
        } else {
            emit.named(H_PRE_NORM)?
        };
        let last_idx = if need_last_idx {
            Some(emit.flow_input("last_token_idx")?.hir_id())
        } else {
            None
        };
        let h_pre = emit.named(H_PRE_NORM)?;
        let h_for_lm = hidden.hir_id();
        let hir = emit
            .module
            .as_hir_mut()
            .expect("qwen35 prefill-cache flow requires HIR stage");
        let mut gb = HirMut::new(hir);
        let mut packed = packed_tail.lock().expect("packed sink");
        let (logits, mtp, normed) = emit_qwen35_prefill_tail(
            &mut gb,
            emit.params,
            &mut packed,
            &cfg_tail,
            &weights_tail,
            batch,
            seq,
            h_for_lm,
            h_pre,
            input_ids,
            cos,
            sin,
            with_lm_head,
            enable_mtp,
            export_normed_hidden,
            fast_mtp,
            last_idx,
        )?;
        if let Some(mtp_id) = mtp {
            *mtp_out.lock().expect("mtp sink") = Some(mtp_id);
        }
        let n_vocab = weights_tail.lm_vocab_size(&cfg_tail);
        let primary = normed.or(logits).unwrap_or(hidden.hir_id());
        let out_shape = if export_normed_hidden {
            Shape::new(&[batch, 1, n_embd], f)
        } else if with_lm_head {
            Shape::new(&[batch, 1, n_vocab], f)
        } else {
            hidden.shape.clone()
        };
        Ok(Some(emit.wrap(primary, out_shape)))
    });

    if export_normed_hidden {
        flow = flow.output("hidden");
    } else if with_lm_head {
        flow = flow.output("logits");
    } else {
        flow = flow.output("hidden");
    }

    let mut built = flow.build(&mut InlineQwen35Weights {
        weights: &weights_c,
        cfg: &cfg_c,
    })?;
    let layer_side = recur_sink.lock().expect("recur sink").clone();
    let mtp = mtp_sink.lock().expect("mtp sink").take();
    built = finish_hir_side_outputs(built, layer_side, mtp);
    let packed = packed_sink.lock().expect("packed sink").clone();
    built
        .into_parts()
        .map(|(hir, params)| (hir, params, packed))
}

pub fn build_qwen35_prefill_cache_flow(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    opts: &Qwen35PrefillCacheOpts,
) -> Result<(
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    build_qwen35_prefill_cache_model_flow(cfg, weights.into(), opts)
}

pub fn build_qwen35_prefill_cache_built(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    opts: &Qwen35PrefillCacheOpts,
) -> Result<(BuiltModel, PackedParams)> {
    let (hir, params, packed) = build_qwen35_prefill_cache_flow(cfg, weights, opts)?;
    let mut built = rlx_core::flow_util::built_from_hir(hir, params)?;
    if let Some(profile) = opts.profile.clone() {
        built.profile = profile;
    }
    Ok((built, packed))
}

/// Options for trunk layer-export prefill (diagnostic bisect graphs).
#[derive(Debug, Clone)]
pub struct Qwen35TrunkExportOpts {
    pub batch: usize,
    pub seq: usize,
    pub with_lm_head: bool,
    pub last_logits_only: bool,
    pub enable_mtp_head: bool,
    pub fast_mtp: bool,
    pub export_normed_hidden: bool,
    pub profile: Option<CompileProfile>,
}

impl Qwen35TrunkExportOpts {
    pub fn probe(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            with_lm_head: false,
            last_logits_only: true,
            enable_mtp_head: false,
            fast_mtp: false,
            export_normed_hidden: false,
            profile: None,
        }
    }
}

/// Runtime-MRoPE prefill via [`Qwen35Flow::runtime_mrope`].
pub fn build_qwen35_runtime_mrope_prefill_flow(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
) -> Result<(
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    let mut flow = Qwen35Flow::prefill(cfg, weights, batch, seq).runtime_mrope();
    if fast_mtp {
        flow = flow.fast_mtp();
    }
    if enable_mtp_head {
        flow = flow.mtp_head();
    }
    if with_lm_head {
        flow = flow.lm_head();
    }
    if last_logits_only {
        flow = if with_lm_head {
            flow.last_token_logits()
        } else {
            flow.last_token_index()
        };
    }
    let (built, packed) = flow.build_with_weights(weights)?;
    let (hir, params) = built.into_parts()?;
    Ok((hir, params, packed))
}

/// Trunk layer-export assembly via [`ModelFlow`] (per-layer last-token hidden taps).
pub fn build_qwen35_trunk_export_model_flow(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    opts: &Qwen35TrunkExportOpts,
) -> Result<(
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    anyhow::ensure!(
        opts.last_logits_only,
        "export_trunk_layer_hiddens requires last_logits_only=true"
    );

    let batch = opts.batch;
    let seq = opts.seq;
    let with_lm_head = opts.with_lm_head;
    let enable_mtp = opts.enable_mtp_head;
    let export_normed_hidden = opts.export_normed_hidden;
    let fast_mtp = opts.fast_mtp;
    let n_embd = cfg.hidden_size;
    let f = DType::F32;
    let hidden = hidden_shape(batch, seq, n_embd);
    let tap_shape = Shape::new(&[batch, 1, n_embd], f);

    let cfg_c = cfg.clone();
    let weights_c = weights.clone();
    let tap_sink: Arc<Mutex<Vec<HirNodeId>>> = Arc::new(Mutex::new(Vec::new()));
    let tail_sink: Arc<Mutex<Vec<HirNodeId>>> = Arc::new(Mutex::new(Vec::new()));
    let packed_sink: Arc<Mutex<PackedParams>> = Arc::new(Mutex::new(PackedParams::new()));

    let head_half = cfg.key_length / 2;
    let max_pos = cfg.max_position_embeddings;
    let trunk_count = weights_c.trunk_layers.len();

    let flow_base = ModelFlow::new("qwen35_trunk_export")
        .input("input_ids", Shape::new(&[batch, seq], f))
        .input("last_token_idx", Shape::new(&[batch], f));
    // Large-vocab models (Bonsai-27B: token_embd [248320,5120] F32 = 4.7 GiB)
    // otherwise keep the whole embedding table resident on the device just to
    // gather the prompt's rows. Host-embed (auto ≥1 GiB / env override) gathers
    // those rows host-side and feeds them as `inputs_embeds`.
    let mut flow = if host_embed_enabled_for_bytes(weights_c.token_embd.len() * 4) {
        flow_base.embed_host("token_embd.weight", n_embd)
    } else {
        flow_base.embed("token_embd.weight")
    };

    let cfg_rope = cfg_c.clone();
    flow = flow
        .plugin_named("qwen35.rope", move |emit, input| {
            let (cos_data, sin_data) = rope::build_mrope_tables(&cfg_rope, max_pos, head_half);
            let cos_shape = Shape::new(&[max_pos, head_half], f);
            let sin_shape = cos_shape.clone();
            let cos_id = emit.synth_param(ROPE_COS, cos_data, cos_shape);
            let sin_id = emit.synth_param(ROPE_SIN, sin_data, sin_shape);
            emit.set_named(ROPE_COS, cos_id);
            emit.set_named(ROPE_SIN, sin_id);
            Ok(input)
        })
        .plugin_named("qwen35.trunk_export.embed_tap", {
            let tap = tap_sink.clone();
            move |emit, input| {
                let hidden = input.ok_or_else(|| anyhow::anyhow!("embed tap requires hidden"))?;
                let h = hidden.hir_id();
                let last_idx = emit.flow_input("last_token_idx")?.hir_id();
                let hir = emit
                    .module
                    .as_hir_mut()
                    .expect("trunk export flow requires HIR stage");
                let mut gb = HirMut::new(hir);
                let tap_id = emit_qwen35_gather_last_token(&mut gb, h, batch, last_idx);
                tap.lock().expect("tap sink").push(tap_id);
                Ok(Some(hidden))
            }
        });

    let weights_layers = weights_c.clone();
    let cfg_layers = cfg_c.clone();
    let tap_layers = tap_sink.clone();
    let packed_arc = packed_sink.clone();
    flow = flow.repeat_layers(trunk_count, move |il| {
        trunk_export_layer_plugin(
            il,
            &cfg_layers,
            &weights_layers,
            batch,
            seq,
            tap_layers.clone(),
            packed_arc.clone(),
            hidden.clone(),
        )
    });

    flow = flow.plugin_named("qwen35.snapshot", |emit, input| {
        let h = input.ok_or_else(|| anyhow::anyhow!("snapshot requires hidden"))?;
        emit.set_named(H_PRE_NORM, h.hir_id());
        Ok(Some(h))
    });

    flow = flow.gather_last_token_dynamic(batch);

    let weights_tail = weights_c.clone();
    let cfg_tail = cfg_c.clone();
    let tail_extra = tail_sink.clone();
    let packed_tail = packed_sink.clone();
    flow = flow.plugin_named("qwen35.trunk_export.tail", move |emit, input| {
        let hidden = input.ok_or_else(|| anyhow::anyhow!("trunk export tail requires hidden"))?;
        let cos = emit.named(ROPE_COS)?;
        let sin = emit.named(ROPE_SIN)?;
        let input_ids = emit.flow_input("input_ids")?.hir_id();
        let last_idx = emit.flow_input("last_token_idx")?.hir_id();
        let h_pre = emit.named(H_PRE_NORM)?;
        let h_for_lm = hidden.hir_id();
        let hir = emit
            .module
            .as_hir_mut()
            .expect("trunk export flow requires HIR stage");
        let mut gb = HirMut::new(hir);
        let mut packed = packed_tail.lock().expect("packed sink");
        let (logits, mtp, normed) = emit_qwen35_prefill_tail(
            &mut gb,
            emit.params,
            &mut packed,
            &cfg_tail,
            &weights_tail,
            batch,
            seq,
            h_for_lm,
            h_pre,
            input_ids,
            cos,
            sin,
            with_lm_head,
            enable_mtp,
            export_normed_hidden,
            fast_mtp,
            Some(last_idx),
        )?;
        let mut tail = tail_extra.lock().expect("tail sink");
        if let Some(n) = normed {
            tail.push(n);
        }
        if let Some(l) = logits {
            tail.push(l);
        }
        if let Some(m) = mtp {
            tail.push(m);
        }
        Ok(Some(hidden))
    });

    let tap_out = tap_sink.clone();
    flow = flow
        .plugin_named("qwen35.trunk_export.primary", move |emit, _input| {
            let primary = tap_out
                .lock()
                .expect("tap sink")
                .first()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("trunk export missing embed tap"))?;
            Ok(Some(emit.wrap(primary, tap_shape.clone())))
        })
        .output("trunk_hidden");

    let mut built = flow.build(&mut InlineQwen35Weights {
        weights: &weights_c,
        cfg: &cfg_c,
    })?;
    let taps = tap_sink.lock().expect("tap sink").clone();
    let mut extra: Vec<HirNodeId> = taps.get(1..).unwrap_or(&[]).to_vec();
    extra.extend(tail_sink.lock().expect("tail sink").clone());
    if !extra.is_empty() {
        built = built.with_extra_hir_outputs(extra);
    }
    let packed = packed_sink.lock().expect("packed sink").clone();
    built
        .into_parts()
        .map(|(hir, params)| (hir, params, packed))
}

pub fn build_qwen35_trunk_export_flow(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    opts: &Qwen35TrunkExportOpts,
) -> Result<(
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    build_qwen35_trunk_export_model_flow(cfg, weights, opts)
}

pub fn build_qwen35_trunk_export_built(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    opts: &Qwen35TrunkExportOpts,
) -> Result<(BuiltModel, PackedParams)> {
    let (hir, params, packed) = build_qwen35_trunk_export_flow(cfg, weights, opts)?;
    let mut built = rlx_core::flow_util::built_from_hir(hir, params)?;
    if let Some(profile) = opts.profile.clone() {
        built.profile = profile;
    }
    Ok((built, packed))
}

/// Single trunk-layer probe (external hidden in → one block out).
#[derive(Clone)]
pub struct Qwen35LayerProbeFlow<'a> {
    cfg: &'a Qwen35Config,
    weights: Qwen35Weights,
    layer: usize,
    batch: usize,
    seq: usize,
    export_post_attn: bool,
}

impl<'a> Qwen35LayerProbeFlow<'a> {
    pub fn new(
        cfg: &'a Qwen35Config,
        weights: Qwen35Weights,
        layer: usize,
        batch: usize,
        seq: usize,
    ) -> Self {
        Self {
            cfg,
            weights,
            layer,
            batch,
            seq,
            export_post_attn: false,
        }
    }

    pub fn export_post_attn(mut self) -> Self {
        self.export_post_attn = true;
        self
    }

    pub fn build(self) -> Result<(BuiltModel, PackedParams)> {
        let (hir, params, packed) = build_qwen35_layer_probe_model_flow(
            self.cfg,
            self.weights,
            self.layer,
            self.batch,
            self.seq,
            self.export_post_attn,
        )?;
        Ok((rlx_core::flow_util::built_from_hir(hir, params)?, packed))
    }
}

/// Native layer-probe assembly: `trunk_h` → one trunk block → hidden (+ optional post-attn).
pub fn build_qwen35_layer_probe_model_flow(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    il: usize,
    batch: usize,
    seq: usize,
    export_post_attn: bool,
) -> Result<(
    HirModule,
    std::collections::HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    anyhow::ensure!(
        il < weights.trunk_layers.len(),
        "layer probe: il={il} out of range ({} trunk layers)",
        weights.trunk_layers.len()
    );

    let n_embd = cfg.hidden_size;
    let head_half = cfg.key_length / 2;
    let max_pos = cfg.max_position_embeddings;
    let f = DType::F32;
    let hidden = hidden_shape(batch, seq, n_embd);

    let cfg_c = cfg.clone();
    let weights_c = weights.clone();
    let layer = weights_c.trunk_layers[il].clone();
    let post_sink: Arc<Mutex<Option<HirNodeId>>> = Arc::new(Mutex::new(None));
    let packed_sink: Arc<Mutex<PackedParams>> = Arc::new(Mutex::new(PackedParams::new()));

    let mut flow = ModelFlow::new("qwen35_layer_probe").input("trunk_h", hidden.clone());

    let cfg_rope = cfg_c.clone();
    flow = flow.plugin_named("qwen35.rope", move |emit, input| {
        let (cos_data, sin_data) = rope::build_mrope_tables(&cfg_rope, max_pos, head_half);
        let cos_shape = Shape::new(&[max_pos, head_half], f);
        let sin_shape = cos_shape.clone();
        let cos_id = emit.synth_param(ROPE_COS, cos_data, cos_shape);
        let sin_id = emit.synth_param(ROPE_SIN, sin_data, sin_shape);
        emit.set_named(ROPE_COS, cos_id);
        emit.set_named(ROPE_SIN, sin_id);
        Ok(input)
    });

    let post_out = post_sink.clone();
    let packed_arc = packed_sink.clone();
    flow = flow.plugin_named("qwen35.layer_probe", move |emit, input| {
        let hidden = input.ok_or_else(|| anyhow::anyhow!("layer probe requires hidden"))?;
        let cos = emit.named(ROPE_COS)?;
        let sin = emit.named(ROPE_SIN)?;
        let h_in = hidden.hir_id();
        let hir = emit
            .module
            .as_hir_mut()
            .expect("layer probe flow requires HIR stage");
        let mut gb = HirMut::new(hir);
        let mut packed = packed_arc.lock().expect("packed sink");
        let mut post_slot = HirNodeId(0);
        let post_ref = if export_post_attn {
            Some(&mut post_slot)
        } else {
            None
        };
        let h = emit_qwen35_layer_probe_layer(
            &mut gb,
            emit.params,
            &mut packed,
            &cfg_c,
            il,
            &layer,
            Qwen35BsLayout::new(batch, seq, false),
            h_in,
            cos,
            sin,
            post_ref,
        )?;
        if export_post_attn {
            *post_out.lock().expect("post sink") = Some(post_slot);
        }
        Ok(Some(emit.wrap(h, hidden.shape.clone())))
    });

    flow = flow.output("hidden");

    let mut built = flow.build(&mut InlineQwen35Weights {
        weights: &weights_c,
        cfg,
    })?;
    if let Some(post) = *post_sink.lock().expect("post sink") {
        built = built.with_extra_hir_outputs(vec![post]);
    }
    let packed = packed_sink.lock().expect("packed sink").clone();
    built
        .into_parts()
        .map(|(hir, params)| (hir, params, packed))
}

fn hidden_shape(batch: usize, seq: usize, hidden: usize) -> Shape {
    Shape::new(&[batch, seq, hidden], DType::F32)
}

fn gdn_layer_plugin(
    layer_idx: usize,
    cfg: &Qwen35Config,
    lin: Qwen35LinearLayer,
    bs: Qwen35BsLayout,
    out_shape: Shape,
) -> FlowStage {
    gdn_layer_plugin_with_packed(layer_idx, cfg, lin, bs, out_shape, None)
}

/// `packed_arc` variant — populates the shared sink so K-quant linear-
/// layer weights (`ssm_*` / `attn_*` packed matmuls) reach the runner's
/// `upload_packed_opt` step. The plain `gdn_layer_plugin` keeps the
/// historical drop-the-local behaviour for callers that don't carry a
/// packed-aware loader (small in-memory F32 tests).
fn gdn_layer_plugin_with_packed(
    layer_idx: usize,
    cfg: &Qwen35Config,
    lin: Qwen35LinearLayer,
    bs: Qwen35BsLayout,
    out_shape: Shape,
    packed_arc: Option<Arc<Mutex<PackedParams>>>,
) -> FlowStage {
    let cfg = cfg.clone();
    plugin_named(format!("qwen35.gdn{layer_idx}"), move |emit, input| {
        let hidden = input.ok_or_else(|| anyhow::anyhow!("GDN layer requires hidden input"))?;
        let mut local = PackedParams::new();
        let (hir, params) = emit.hir_and_params();
        let mut gb = HirMut::new(hir);
        let out = emit_qwen35_gdn_prefill_layer(
            &mut gb,
            params,
            &mut local,
            &cfg,
            layer_idx,
            &lin,
            bs,
            hidden.hir_id(),
        )?;
        if let Some(arc) = packed_arc.as_ref() {
            let mut guard = arc.lock().expect("packed sink");
            for (k, v) in local.drain() {
                guard.insert(k, v);
            }
        }
        Ok(Some(emit.wrap(out, out_shape.clone())))
    })
}

fn trunk_export_layer_plugin(
    layer_idx: usize,
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    batch: usize,
    seq: usize,
    tap_sink: Arc<Mutex<Vec<HirNodeId>>>,
    packed_arc: Arc<Mutex<PackedParams>>,
    out_shape: Shape,
) -> FlowStage {
    let cfg = cfg.clone();
    let weights = weights.clone();
    let tap = tap_sink;
    plugin_named(
        format!("qwen35.trunk_export.l{layer_idx}"),
        move |emit, input| {
            let hidden =
                input.ok_or_else(|| anyhow::anyhow!("trunk export layer requires hidden"))?;
            let last_idx = emit.flow_input("last_token_idx")?.hir_id();
            let cos = emit.named(ROPE_COS)?;
            let sin = emit.named(ROPE_SIN)?;
            let h_in = hidden.hir_id();
            let hir = emit
                .module
                .as_hir_mut()
                .expect("trunk export flow requires HIR stage");
            let mut gb = HirMut::new(hir);
            let mut packed = packed_arc.lock().expect("packed sink");
            let bs = Qwen35BsLayout::new(batch, seq, false);
            let h = match weights.trunk_layers.get(layer_idx) {
                Some(Qwen35TrunkLayer::Linear(lin)) => emit_qwen35_gdn_prefill_layer(
                    &mut gb,
                    emit.params,
                    &mut packed,
                    &cfg,
                    layer_idx,
                    lin,
                    bs,
                    h_in,
                )?,
                Some(Qwen35TrunkLayer::FullAttn(fa)) => emit_qwen35_full_attn_prefill_layer(
                    &mut gb,
                    emit.params,
                    &mut packed,
                    &cfg,
                    layer_idx,
                    fa,
                    bs,
                    h_in,
                    cos,
                    sin,
                )?,
                None => h_in,
            };
            let tap_id = emit_qwen35_gather_last_token(&mut gb, h, batch, last_idx);
            tap.lock().expect("tap sink").push(tap_id);
            Ok(Some(emit.wrap(h, out_shape.clone())))
        },
    )
}

#[allow(dead_code)]
fn full_attn_layer_plugin(
    layer_idx: usize,
    cfg: &Qwen35Config,
    fa: Qwen35FullAttnLayer,
    bs: Qwen35BsLayout,
    out_shape: Shape,
) -> FlowStage {
    full_attn_layer_plugin_with_packed(layer_idx, cfg, fa, bs, out_shape, None)
}

/// `packed_arc` variant — see [`gdn_layer_plugin_with_packed`].
fn full_attn_layer_plugin_with_packed(
    layer_idx: usize,
    cfg: &Qwen35Config,
    fa: Qwen35FullAttnLayer,
    bs: Qwen35BsLayout,
    out_shape: Shape,
    packed_arc: Option<Arc<Mutex<PackedParams>>>,
) -> FlowStage {
    let cfg = cfg.clone();
    plugin_named(format!("qwen35.fa{layer_idx}"), move |emit, input| {
        let hidden = input.ok_or_else(|| anyhow::anyhow!("full-attn layer requires hidden"))?;
        let cos = emit.named(ROPE_COS)?;
        let sin = emit.named(ROPE_SIN)?;
        let mut local = PackedParams::new();
        let (hir, params) = emit.hir_and_params();
        let mut gb = HirMut::new(hir);
        let out = emit_qwen35_full_attn_prefill_layer(
            &mut gb,
            params,
            &mut local,
            &cfg,
            layer_idx,
            &fa,
            bs,
            hidden.hir_id(),
            cos,
            sin,
        )?;
        if let Some(arc) = packed_arc.as_ref() {
            let mut guard = arc.lock().expect("packed sink");
            for (k, v) in local.drain() {
                guard.insert(k, v);
            }
        }
        Ok(Some(emit.wrap(out, out_shape.clone())))
    })
}

struct InlineQwen35Weights<'a> {
    weights: &'a Qwen35Weights,
    cfg: &'a Qwen35Config,
}

impl rlx_flow::WeightSource for InlineQwen35Weights<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        if key == "token_embd.weight" {
            let h = self.cfg.hidden_size;
            let v = self.weights.lm_vocab_size(self.cfg);
            return Ok((self.weights.token_embd.to_vec(), vec![v, h]));
        }
        if transpose {
            bail!("inline qwen35 weights: transpose not supported for `{key}`");
        }
        bail!("inline qwen35 weights: missing `{key}`")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        MatWeight, Qwen35FullAttnLayer, Qwen35LayerFfn, Qwen35MtpLayer, Qwen35TrunkLayer,
        build_qwen35_hir_sized_ext,
    };

    fn mat(data: Vec<f32>) -> MatWeight {
        MatWeight::F32(data)
    }

    fn tiny_cfg() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 4,
            nextn_predict_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            key_length: 4,
            value_length: 4,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_dim_count: 4,
            rope_dim_sections: vec![],
            mrope_interleaved: false,
            rms_norm_offset: false,
            full_attention_interval: 3,
            ssm_conv_kernel: 4,
            ssm_group_count: 2,
            ssm_inner_size: 8,
            ssm_state_size: 4,
            ssm_time_step_rank: 2,
            tie_word_embeddings: true,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    fn ramp(n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
    }

    fn linear_layer(cfg: &Qwen35Config) -> Qwen35LinearLayer {
        let n_embd = cfg.hidden_size;
        let n_state = cfg.ssm_state_size;
        let n_k_heads = cfg.ssm_group_count;
        let n_v_heads = cfg.ssm_time_step_rank;
        let key_dim = n_state * n_k_heads;
        let value_dim = n_state * n_v_heads;
        let conv_channels = key_dim * 2 + value_dim;
        let n_ff = cfg.intermediate_size;
        let k_conv = cfg.ssm_conv_kernel;
        Qwen35LinearLayer {
            attn_norm: vec![1.0f32; n_embd],
            attn_post_norm: vec![1.0f32; n_embd],
            attn_qkv: mat(ramp(n_embd * conv_channels, 0.01)),
            attn_gate: mat(ramp(n_embd * value_dim, 0.01)),
            ssm_conv1d: ramp(k_conv * conv_channels, 0.02),
            ssm_dt_bias: ramp(n_v_heads, 0.05),
            ssm_a: vec![-1.0f32; n_v_heads],
            ssm_beta: mat(ramp(n_embd * n_v_heads, 0.01)),
            ssm_alpha: mat(ramp(n_embd * n_v_heads, 0.01)),
            ssm_norm: vec![1.0f32; n_state],
            ssm_out: mat(ramp(value_dim * n_embd, 0.01)),
            ffn: Qwen35LayerFfn::Dense {
                gate: mat(ramp(n_embd * n_ff, 0.01)),
                down: mat(ramp(n_ff * n_embd, 0.01)),
                up: mat(ramp(n_embd * n_ff, 0.01)),
            },
        }
    }

    fn full_attn_layer(cfg: &Qwen35Config) -> Qwen35FullAttnLayer {
        let n_embd = cfg.hidden_size;
        let n_head = cfg.num_attention_heads;
        let n_kv_head = cfg.num_key_value_heads;
        let head_dim = cfg.key_length;
        let q_gate_cols = n_head * head_dim * 2;
        let kv_cols = n_kv_head * head_dim;
        let n_ff = cfg.intermediate_size;
        Qwen35FullAttnLayer {
            attn_norm: vec![1.0f32; n_embd],
            attn_post_norm: vec![1.0f32; n_embd],
            attn_q_gate: mat(ramp(n_embd * q_gate_cols, 0.01)),
            attn_k: mat(ramp(n_embd * kv_cols, 0.01)),
            attn_v: mat(ramp(n_embd * kv_cols, 0.01)),
            attn_output: mat(ramp(n_head * head_dim * n_embd, 0.01)),
            attn_q_norm: vec![1.0f32; head_dim],
            attn_k_norm: vec![1.0f32; head_dim],
            ffn: Qwen35LayerFfn::Dense {
                gate: mat(ramp(n_embd * n_ff, 0.01)),
                down: mat(ramp(n_ff * n_embd, 0.01)),
                up: mat(ramp(n_embd * n_ff, 0.01)),
            },
        }
    }

    fn synth_weights(cfg: &Qwen35Config) -> Qwen35Weights {
        let n_embd = cfg.hidden_size;
        let n_vocab = cfg.vocab_size;
        let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
        let interval = cfg.full_attention_interval.max(1);

        let mut trunk = Vec::new();
        for il in 0..n_main {
            let is_full = ((il + 1) % interval) == 0;
            trunk.push(if is_full {
                Qwen35TrunkLayer::FullAttn(full_attn_layer(cfg))
            } else {
                Qwen35TrunkLayer::Linear(linear_layer(cfg))
            });
        }
        let mtp = Qwen35MtpLayer {
            base: full_attn_layer(cfg),
            eh_proj: mat(ramp(2 * n_embd * n_embd, 0.01)),
            enorm: vec![1.0f32; n_embd],
            hnorm: vec![1.0f32; n_embd],
            embed_tokens: None,
            shared_head_head: None,
            shared_head_norm: None,
        };

        Qwen35Weights {
            token_embd: std::sync::Arc::from(ramp(n_vocab * n_embd, 0.001)),
            output_norm: vec![1.0f32; n_embd],
            output: None,
            token_embd_lm: None,
            trunk_layers: trunk,
            mtp_layers: vec![mtp],
        }
    }

    #[test]
    fn one_gdn_layer_flow_builds() {
        let cfg = tiny_cfg();
        let empty = Qwen35Weights {
            token_embd: std::sync::Arc::from([]),
            output_norm: vec![],
            output: None,
            token_embd_lm: None,
            trunk_layers: vec![],
            mtp_layers: vec![],
        };
        let built = Qwen35Flow::one_gdn_layer(&cfg, linear_layer(&cfg), 0, 1, 4)
            .build_with_weights(&empty)
            .unwrap();
        let hir = built.0.into_hir().unwrap();
        assert!(
            hir.len() > 8,
            "GDN layer should expand into a non-trivial subgraph (got {} nodes)",
            hir.len()
        );
    }

    #[test]
    fn prefill_flow_matches_builder_graph_size() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let batch = 1;
        let seq = 4;

        let (hir_ref, _, _) = build_qwen35_hir_sized_ext(
            &cfg,
            weights.clone(),
            batch,
            seq,
            true,
            true,
            true,
            false,
            None,
            false,
            false,
            false,
            false,
        )
        .unwrap();

        let built = Qwen35Flow::prefill(&cfg, &weights, batch, seq)
            .last_token_logits()
            .lm_head()
            .mtp_head()
            .build_with_weights(&weights)
            .unwrap();
        let hir_flow = built.0.into_hir().unwrap();

        let node_diff = hir_flow.len().abs_diff(hir_ref.len());
        assert!(
            node_diff <= 1,
            "flow prefill should match builder node count within scaffolding (flow={}, builder={}, diff={})",
            hir_flow.len(),
            hir_ref.len(),
            node_diff
        );
        assert_eq!(hir_flow.outputs.len(), hir_ref.outputs.len());
    }

    #[test]
    fn decode_model_flow_matches_hir_node_count() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let past_seq = 8;
        let opts = Qwen35DecodeOpts::step(1, past_seq);
        let (hir_flow, _, _) =
            build_qwen35_decode_model_flow(&cfg, std::sync::Arc::new(weights.clone()), &opts)
                .unwrap();
        let (hir_ref, _, _) = crate::builder::build_qwen35_decode_hir_assembled(
            &cfg,
            std::sync::Arc::new(weights),
            1,
            true,
            true,
            false,
            Some(past_seq),
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            hir_flow.len(),
            hir_ref.len(),
            "decode ModelFlow should match delegated hir node count"
        );
        assert_eq!(hir_flow.outputs.len(), hir_ref.outputs.len());
    }

    #[test]
    fn decode_flow_places_mtp_logits_before_recurrent_side_outputs() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let mut opts = Qwen35DecodeOpts::step(1, 4);
        opts.enable_mtp_head = true;
        let (hir, _, _) = build_qwen35_decode_flow(&cfg, weights, &opts).unwrap();
        let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
        let interval = cfg.full_attention_interval.max(1);
        let n_full = (0..n_main).filter(|il| ((il + 1) % interval) == 0).count();
        let n_linear = n_main - n_full;
        let expected = 1 + 1 + 2 * n_linear + 2 * n_full;
        assert_eq!(hir.outputs.len(), expected);
        assert_eq!(
            hir.outputs.len(),
            8,
            "tiny: logits + mtp + 2*(conv,ssm) + (k,v)"
        );
    }

    #[test]
    fn decode_flow_builds_with_recurrent_side_outputs() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let past_seq = 8;
        let opts = Qwen35DecodeOpts::step(1, past_seq);

        let (hir, _, _) = build_qwen35_decode_flow(&cfg, weights, &opts).unwrap();
        assert!(
            hir.len() > 10,
            "decode HIR should be non-trivial (got {} nodes)",
            hir.len()
        );

        let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
        let interval = cfg.full_attention_interval.max(1);
        let n_full = (0..n_main).filter(|il| ((il + 1) % interval) == 0).count();
        let n_linear = n_main - n_full;
        // logits + (conv, ssm) per linear layer + (k, v) per full-attn layer
        let expected_outputs = 1 + 2 * n_linear + 2 * n_full;
        assert_eq!(
            hir.outputs.len(),
            expected_outputs,
            "decode should export logits plus recurrent/KV side tensors"
        );
    }

    #[test]
    fn runtime_mrope_prefill_flow_matches_hir_node_count() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let (hir_flow, _, _) =
            build_qwen35_runtime_mrope_prefill_flow(&cfg, &weights, 1, 4, true, true, false, false)
                .unwrap();
        let (hir_ref, _, _) = crate::builder::build_qwen35_runtime_mrope_prefill_hir_assembled(
            &cfg, weights, 1, 4, true, true, false, false,
        )
        .unwrap();
        assert_eq!(
            hir_flow.len(),
            hir_ref.len(),
            "runtime MRoPE ModelFlow should match delegated hir node count"
        );
        assert_eq!(hir_flow.outputs.len(), hir_ref.outputs.len());
    }

    #[test]
    fn trunk_export_model_flow_matches_hir_node_count() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let opts = Qwen35TrunkExportOpts::probe(1, 4);
        let (hir_flow, _, _) =
            build_qwen35_trunk_export_model_flow(&cfg, weights.clone(), &opts).unwrap();
        let (hir_ref, _, _) = crate::builder::build_qwen35_trunk_export_hir_assembled(
            &cfg, weights, 1, 4, false, true, false, false, false,
        )
        .unwrap();
        assert_eq!(
            hir_flow.len(),
            hir_ref.len(),
            "trunk export ModelFlow should match delegated hir node count"
        );
        assert_eq!(hir_flow.outputs.len(), hir_ref.outputs.len());
    }

    #[test]
    fn prefill_cache_model_flow_matches_hir_node_count() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let opts = Qwen35PrefillCacheOpts::static_cache(1, 4);
        let (hir_flow, _, _) = build_qwen35_prefill_cache_model_flow(
            &cfg,
            std::sync::Arc::new(weights.clone()),
            &opts,
        )
        .unwrap();
        let (hir_ref, _, _) = crate::builder::build_qwen35_prefill_cache_hir_assembled(
            &cfg,
            std::sync::Arc::new(weights),
            1,
            4,
            true,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            hir_flow.len(),
            hir_ref.len(),
            "prefill-cache ModelFlow should match delegated hir node count"
        );
        assert_eq!(hir_flow.outputs.len(), hir_ref.outputs.len());
    }

    #[test]
    fn prefill_cache_flow_exports_recurrent_state() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let opts = Qwen35PrefillCacheOpts::static_cache(1, 4);
        let (hir, _, _) = build_qwen35_prefill_cache_flow(&cfg, weights, &opts).unwrap();
        assert!(hir.len() > 10, "prefill-cache HIR should be non-trivial");

        let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
        let interval = cfg.full_attention_interval.max(1);
        let n_full = (0..n_main).filter(|il| ((il + 1) % interval) == 0).count();
        let n_linear = n_main - n_full;
        // logits + (conv, ssm) per linear + (k, v) per full-attn
        let expected_outputs = 1 + 2 * n_linear + 2 * n_full;
        assert_eq!(hir.outputs.len(), expected_outputs);
    }

    #[test]
    fn layer_probe_flow_builds_one_block() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let (built, _) = Qwen35LayerProbeFlow::new(&cfg, weights, 0, 1, 4)
            .build()
            .unwrap();
        let hir = built.into_hir().unwrap();
        assert_eq!(hir.outputs.len(), 1);
        assert!(hir.len() > 5);
    }

    #[test]
    fn layer_probe_model_flow_matches_hir_node_count() {
        let cfg = tiny_cfg();
        let weights = synth_weights(&cfg);
        let (hir_flow, _, _) =
            build_qwen35_layer_probe_model_flow(&cfg, weights.clone(), 0, 1, 4, false).unwrap();
        let (hir_ref, _, _) =
            crate::builder::build_qwen35_layer_probe_hir_assembled(&cfg, &weights, 0, 1, 4, false)
                .unwrap();
        assert_eq!(hir_flow.len(), hir_ref.len());
        assert_eq!(hir_flow.outputs.len(), hir_ref.outputs.len());
    }
}

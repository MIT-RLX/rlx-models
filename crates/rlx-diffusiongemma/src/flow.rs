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

//! The two graphs DiffusionGemma runs.
//!
//! **Encoder** — an ordinary causal Gemma 4 MoE prefill over the prompt. Its
//! only unusual job is to *tap* each layer's post-RoPE K and post-`v_norm` V as
//! side outputs, which become the read-only cache the denoiser attends to.
//!
//! **Decoder (denoiser)** — runs once per denoising step over a fixed-size
//! canvas. It embeds the canvas tokens, folds in the previous step's
//! self-conditioning signal, attends bidirectionally over
//! `[encoder K/V ; canvas K/V]`, and emits soft-capped logits plus the soft
//! embeddings that become the *next* step's self-conditioning signal.
//!
//! Emitting the soft embeddings in-graph matters: they are
//! `softmax(logits) @ embed_tokens`, a `[canvas, vocab] × [vocab, hidden]`
//! product. Computing it on the host would mean shipping the full
//! `[256, 262144]` logits block back every step.
//!
//! Both stacks read the *same* weights (`model.decoder.*`) — HF ties the text
//! encoder to the decoder — except for the per-layer `layer_scalar`, which the
//! encoder takes from `model.encoder.language_model.layers.{i}.layer_scalar`.
//!
//! [`DecoderOutputs`] selects what the denoiser returns: the full logits (exact
//! and simple, but 268 MB per step at production size) or an in-graph reduction
//! to `entropy` / `argmax` / `sampled`.

use anyhow::{Result, anyhow};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow, WeightSource};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, MaskKind, Op, ReduceOp};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::attention::AttnDims;
use crate::config::{DiffusionGemmaConfig, TextConfig};
use crate::layer::{LayerDims, emit_decoder_layer, emit_encoder_layer, emit_self_conditioning};
use crate::moe::MoeDims;

/// Token-embedding tensor (shared by both stacks and the tied LM head).
pub const EMBED_KEY: &str = "model.decoder.embed_tokens.weight";
/// Final RMS norm of both stacks.
pub const FINAL_NORM_KEY: &str = "model.decoder.norm";
/// Layer prefix both stacks project from.
pub const LAYER_PREFIX: &str = "model.decoder.layers";
/// Per-layer scalar for the encoder stack (the one weight that is *not* tied).
pub const ENCODER_SCALAR_PREFIX: &str = "model.encoder.language_model.layers";
/// Self-conditioning block.
pub const SELF_COND_PREFIX: &str = "model.decoder.self_conditioning";

/// Graph input carrying the previous step's self-conditioning signal.
pub const SC_SIGNAL_INPUT: &str = "sc_signal";
/// Graph input carrying the canvas token ids.
pub const CANVAS_INPUT: &str = "canvas_ids";
/// Graph input carrying pre-built prompt embeddings (multimodal encoder path).
pub const INPUTS_EMBEDS_INPUT: &str = "inputs_embeds";
/// Graph input carrying this step's sampling temperature (see
/// [`crate::config::DiffusionGenerationConfig::temperature`]). The reference
/// applies it as a logits processor *after* the soft cap, and the
/// self-conditioning signal is built from the scaled logits — so it has to be
/// inside the graph, not applied by the caller afterwards.
pub const TEMPERATURE_INPUT: &str = "temperature";
/// Side output: soft embeddings to feed back as [`SC_SIGNAL_INPUT`].
pub const SOFT_EMBED_OUTPUT: &str = "soft_embeds";
/// Side output (reduced decoder only): per-position entropy, `[1, canvas]`.
pub const ENTROPY_OUTPUT: &str = "entropy";
/// Side output (reduced decoder only): per-position argmax, `[1, canvas]`.
pub const ARGMAX_OUTPUT: &str = "argmax";
/// Side output (reduced decoder only): per-position categorical draw,
/// `[1, canvas]`.
pub const SAMPLED_OUTPUT: &str = "sampled";

/// What the denoiser hands back per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderOutputs {
    /// Emit the full `[1, canvas, vocab]` logits and let the host reduce them.
    /// Exact and simple, but at production sizes that block is 268 MB *per
    /// denoising step*.
    Logits,
    /// Reduce in-graph to `entropy` / `argmax` / `sampled`, all `[1, canvas]`.
    /// The logits never leave the device; the readback drops to a few KB.
    ///
    /// Sampling uses the Gumbel-max trick (`argmax(logits + g)`, `g =
    /// -ln(-ln u)`), which draws from the same categorical distribution as
    /// inverse-CDF sampling but consumes a different RNG stream — so it will
    /// not reproduce [`crate::sampler::row_sample`] draw for draw.
    Reduced { seed: u64 },
}

/// Non-destructive [`WeightSource`] over a [`WeightMap`].
///
/// The encoder and the denoiser read the *same* tensors, and the denoiser graph
/// is rebuilt whenever the prompt length changes, so the default destructive
/// `take` would empty the map on the first build.
pub(crate) struct SharedWeights<'a>(pub(crate) &'a WeightMap);

impl WeightSource for SharedWeights<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self
            .0
            .get(key)
            .ok_or_else(|| anyhow!("DiffusionGemma checkpoint is missing `{key}`"))?;
        if !transpose {
            return Ok((data.to_vec(), shape.to_vec()));
        }
        anyhow::ensure!(
            shape.len() == 2,
            "transpose requested for non-2D tensor `{key}` (shape {shape:?})"
        );
        let (r, c) = (shape[0], shape[1]);
        let mut out = vec![0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = data[i * c + j];
            }
        }
        Ok((out, vec![c, r]))
    }

    fn has(&self, key: &str) -> bool {
        self.0.has(key)
    }
}

/// Encoder K tap name for layer `i`.
pub fn enc_k_name(layer: usize) -> String {
    format!("enc_k.{layer}")
}
/// Encoder V tap name for layer `i`.
pub fn enc_v_name(layer: usize) -> String {
    format!("enc_v.{layer}")
}

fn layer_dims(cfg: &TextConfig, layer: usize, seq: usize, scalar_key: String) -> LayerDims {
    LayerDims {
        attn: AttnDims {
            hidden: cfg.hidden_size,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.layer_kv_heads(layer),
            head_dim: cfg.layer_head_dim(layer),
            k_eq_v: cfg.layer_k_eq_v(layer),
            eps: cfg.rms_norm_eps,
            seq,
        },
        moe: MoeDims {
            hidden: cfg.hidden_size,
            moe_inter: cfg.moe_intermediate_size,
            num_experts: cfg.num_experts,
            top_k: cfg.top_k_experts,
            rows: seq,
            eps: cfg.rms_norm_eps,
            root_scale: cfg.router_root_scale(),
            experts_pretransposed: true,
        },
        intermediate: cfg.intermediate_size,
        hidden: cfg.hidden_size,
        eps: cfg.rms_norm_eps,
        seq,
        layer_scalar_key: scalar_key,
    }
}

/// `embed_tokens(ids) · sqrt(hidden)`.
fn emit_embed(
    emit: &mut Emit<'_>,
    ids_input: &str,
    seq: usize,
    cfg: &TextConfig,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let table = emit.load_param(EMBED_KEY, false)?; // [vocab, hidden]
    let ids = emit.flow_input(ids_input)?.hir_id();
    let scale = emit.synth_param("embed_scale", vec![cfg.embed_scale()], Shape::new(&[1], f));
    let mut gb = HirMut::new(emit.hir());
    let ids2 = gb.reshape_(ids, vec![1, seq as i64]);
    let e = gb.gather_(table, ids2, 0); // [1, seq, hidden]
    Ok(gb.mul(e, scale))
}

fn emit_rms(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    dim: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let gamma = emit.load_param(&format!("{key}.weight"), false)?;
    let beta = emit.synth_param(
        &format!("{key}.beta"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, gamma, beta, eps))
}

fn rope_inputs(cfg: &TextConfig, seq: usize) -> Vec<(String, Shape)> {
    let f = DType::F32;
    let mut out = Vec::new();
    let mut seen_sliding = false;
    let mut seen_full = false;
    for l in 0..cfg.num_hidden_layers {
        let full = cfg.is_full(l);
        if (full && seen_full) || (!full && seen_sliding) {
            continue;
        }
        if full {
            seen_full = true;
        } else {
            seen_sliding = true;
        }
        let half = cfg.layer_head_dim(l) / 2;
        let (c, s) = cfg.rope_input_names(l);
        out.push((c.to_string(), Shape::new(&[seq, half], f)));
        out.push((s.to_string(), Shape::new(&[seq, half], f)));
    }
    out
}

/// Build the prompt encoder for a fixed `seq`.
///
/// Inputs: `input_ids [1, seq]` plus the per-layer-type RoPE tables
/// (`rope_cos_sliding` / `rope_sin_sliding` / `rope_cos_full` / `rope_sin_full`,
/// see [`TextConfig::rope_tables`]).
///
/// Outputs: `hidden [1, seq, hidden]`, then `enc_k.{i}` / `enc_v.{i}` for every
/// layer, each `[1, seq, layer_kv_dim]`.
pub fn build_encoder_flow(
    cfg: &DiffusionGemmaConfig,
    weights: &WeightMap,
    seq: usize,
) -> Result<BuiltModel> {
    build_encoder(cfg, weights, seq, false)
}

/// Like [`build_encoder_flow`], but the prompt arrives as pre-built embeddings
/// on [`INPUTS_EMBEDS_INPUT`] instead of token ids.
///
/// This is the multimodal entry point: vision soft tokens replace the token
/// embeddings at image positions. The caller supplies embeddings that are
/// *already* scaled — see [`crate::vision::merge_multimodal_embeds`], which
/// applies `sqrt(hidden)` to text rows and leaves projected soft tokens
/// untouched, exactly as HF's `masked_scatter` does.
pub fn build_encoder_flow_embeds(
    cfg: &DiffusionGemmaConfig,
    weights: &WeightMap,
    seq: usize,
) -> Result<BuiltModel> {
    build_encoder(cfg, weights, seq, true)
}

fn build_encoder(
    cfg: &DiffusionGemmaConfig,
    weights: &WeightMap,
    seq: usize,
    from_embeds: bool,
) -> Result<BuiltModel> {
    cfg.validate()?;
    let t = &cfg.text_config;
    let f = DType::F32;
    let hidden = t.hidden_size;

    let mut flow =
        ModelFlow::new("diffusiongemma_encoder").with_profile(CompileProfile::llama32_prefill());
    flow = if from_embeds {
        flow.input(INPUTS_EMBEDS_INPUT, Shape::new(&[1, seq, hidden], f))
    } else {
        flow.input("input_ids", Shape::new(&[1, seq], f))
    };
    for (name, shape) in rope_inputs(t, seq) {
        flow = flow.input(name, shape);
    }

    let hs = Shape::new(&[1, seq, hidden], f);
    {
        let cfg_t = t.clone();
        let hs = hs.clone();
        flow = flow.plugin_named("embed", move |emit, _prev| {
            let e = if from_embeds {
                emit.flow_input(INPUTS_EMBEDS_INPUT)?.hir_id()
            } else {
                emit_embed(emit, "input_ids", seq, &cfg_t)?
            };
            Ok(Some(emit.wrap(e, hs.clone())))
        });
    }

    for i in 0..t.num_hidden_layers {
        let d = layer_dims(
            t,
            i,
            seq,
            format!("{ENCODER_SCALAR_PREFIX}.{i}.layer_scalar"),
        );
        let prefix = format!("{LAYER_PREFIX}.{i}");
        let (cos_name, sin_name) = t.rope_input_names(i);
        // Sliding layers are windowed-causal; full layers plain causal.
        //
        // Off-by-one: rlx's `SlidingWindow(w)` admits `ki ∈ [qi - w, qi]`, i.e.
        // `w + 1` positions, while HF's overlay is `kv_idx > q_idx - w`, i.e.
        // `w` positions including the query. So the window argument is
        // `sliding_window - 1`.
        let mask = if t.is_full(i) {
            MaskKind::Causal
        } else {
            MaskKind::SlidingWindow(t.sliding_window.saturating_sub(1))
        };
        let hs = hs.clone();
        flow = flow.plugin_named(format!("layer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("encoder layer{i} needs a hidden input"))?
                .hir_id();
            let cos = emit.flow_input(cos_name)?.hir_id();
            let sin = emit.flow_input(sin_name)?.hir_id();
            let (out, tap) = emit_encoder_layer(emit, &prefix, x, &d, cos, sin, mask)?;
            emit.state.side_outputs.push((enc_k_name(i), tap.k));
            emit.state.side_outputs.push((enc_v_name(i), tap.v));
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    let eps = t.rms_norm_eps;
    {
        let hs = hs.clone();
        flow = flow.plugin_named("final_norm", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("final_norm needs a hidden input"))?
                .hir_id();
            let out = emit_rms(emit, FINAL_NORM_KEY, x, hidden, eps)?;
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    flow.output("hidden")
        .build_with(&mut SharedWeights(weights), None)
}

/// How much of the encoder cache each layer type sees.
///
/// Full-attention layers read the whole prompt; sliding layers only the last
/// `sliding_window` positions, which is exactly what HF's sliding cache layer
/// retains. The denoiser applies no mask of its own — locality lives in the
/// cache, not in the mask.
#[derive(Debug, Clone, Copy)]
pub struct EncoderCacheLens {
    pub sliding: usize,
    pub full: usize,
}

impl EncoderCacheLens {
    pub fn for_prompt(cfg: &TextConfig, prompt_len: usize) -> Self {
        Self {
            sliding: prompt_len.min(cfg.sliding_window),
            full: prompt_len,
        }
    }

    pub fn for_layer(&self, cfg: &TextConfig, layer: usize) -> usize {
        if cfg.is_full(layer) {
            self.full
        } else {
            self.sliding
        }
    }
}

/// Build the denoiser for a fixed canvas length and encoder cache size.
///
/// Inputs: `canvas_ids [1, canvas]`, `sc_signal [1, canvas, hidden]`, the RoPE
/// tables for the canvas positions, and `enc_k.{i}` / `enc_v.{i}`
/// `[1, cache_len_i, layer_kv_dim]`.
///
/// Outputs: `logits [1, canvas, vocab]` (soft-capped) and `soft_embeds
/// [1, canvas, hidden]` — feed the latter back as `sc_signal` next step.
pub fn build_decoder_flow(
    cfg: &DiffusionGemmaConfig,
    weights: &WeightMap,
    canvas: usize,
    cache: EncoderCacheLens,
) -> Result<BuiltModel> {
    build_decoder(cfg, weights, canvas, cache, DecoderOutputs::Logits)
}

/// Like [`build_decoder_flow`], but chooses what the denoiser returns.
pub fn build_decoder_flow_with(
    cfg: &DiffusionGemmaConfig,
    weights: &WeightMap,
    canvas: usize,
    cache: EncoderCacheLens,
    outputs: DecoderOutputs,
) -> Result<BuiltModel> {
    build_decoder(cfg, weights, canvas, cache, outputs)
}

fn build_decoder(
    cfg: &DiffusionGemmaConfig,
    weights: &WeightMap,
    canvas: usize,
    cache: EncoderCacheLens,
    outputs: DecoderOutputs,
) -> Result<BuiltModel> {
    cfg.validate()?;
    let t = &cfg.text_config;
    let f = DType::F32;
    let hidden = t.hidden_size;
    let vocab = t.vocab_size;

    let mut flow = ModelFlow::new("diffusiongemma_decoder")
        .with_profile(CompileProfile::llama32_prefill())
        .input(CANVAS_INPUT, Shape::new(&[1, canvas], f))
        .input(SC_SIGNAL_INPUT, Shape::new(&[1, canvas, hidden], f))
        .input(TEMPERATURE_INPUT, Shape::new(&[1], f));
    for (name, shape) in rope_inputs(t, canvas) {
        flow = flow.input(name, shape);
    }
    for i in 0..t.num_hidden_layers {
        let kv_dim = t.layer_kv_heads(i) * t.layer_head_dim(i);
        let len = cache.for_layer(t, i);
        flow = flow
            .input(enc_k_name(i), Shape::new(&[1, len, kv_dim], f))
            .input(enc_v_name(i), Shape::new(&[1, len, kv_dim], f));
    }

    let hs = Shape::new(&[1, canvas, hidden], f);
    {
        let cfg_t = t.clone();
        let hs = hs.clone();
        let eps = t.rms_norm_eps;
        flow = flow.plugin_named("embed_self_cond", move |emit, _prev| {
            let e = emit_embed(emit, CANVAS_INPUT, canvas, &cfg_t)?;
            let sc = emit.flow_input(SC_SIGNAL_INPUT)?.hir_id();
            let out = emit_self_conditioning(emit, SELF_COND_PREFIX, e, sc, hidden, eps)?;
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    for i in 0..t.num_hidden_layers {
        let d = layer_dims(t, i, canvas, format!("{LAYER_PREFIX}.{i}.layer_scalar"));
        let prefix = format!("{LAYER_PREFIX}.{i}");
        let (cos_name, sin_name) = t.rope_input_names(i);
        let enc_len = cache.for_layer(t, i);
        let (kn, vn) = (enc_k_name(i), enc_v_name(i));
        let hs = hs.clone();
        flow = flow.plugin_named(format!("layer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("decoder layer{i} needs a hidden input"))?
                .hir_id();
            let cos = emit.flow_input(cos_name)?.hir_id();
            let sin = emit.flow_input(sin_name)?.hir_id();
            let enc_k = emit.flow_input(&kn)?.hir_id();
            let enc_v = emit.flow_input(&vn)?.hir_id();
            let out = emit_decoder_layer(emit, &prefix, x, &d, cos, sin, enc_k, enc_v, enc_len)?;
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    let eps = t.rms_norm_eps;
    let softcap = t.final_logit_softcapping;
    let embed_scale = t.embed_scale();
    {
        let logits_shape = Shape::new(&[1, canvas, vocab], f);
        flow = flow.plugin_named("lm_head", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("lm_head needs a hidden input"))?
                .hir_id();
            let x = emit_rms(emit, FINAL_NORM_KEY, x, hidden, eps)?;

            // Tied LM head: logits = hidden @ embed_tokensᵀ.
            //
            // Loaded under its own name, NOT via `load_param(EMBED_KEY, true)`:
            // `Emit::load_param` caches nodes by (key, transpose) but writes the
            // params map under `key` alone, so loading one tensor both ways
            // creates two nodes sharing a name and the second load silently
            // clobbers the first one's data.
            let (t_data, t_shape) = emit.weights.take(EMBED_KEY, true)?;
            let table_t = emit.synth_param("lm_head.tied_weight", t_data, Shape::new(&t_shape, f)); // [hidden, vocab]
            let cap = emit.synth_param("logit_softcap", vec![softcap], Shape::new(&[1], f));
            let inv_cap = emit.synth_param(
                "logit_softcap_inv",
                vec![1.0 / softcap],
                Shape::new(&[1], f),
            );
            let temp = emit.flow_input(TEMPERATURE_INPUT)?.hir_id();
            let logits = {
                let mut gb = HirMut::new(emit.hir());
                let raw = gb.mm(x, table_t);
                // Gemma logit soft-cap: tanh(l / c) · c.
                let scaled = gb.mul(raw, inv_cap);
                let capped = gb.tanh(scaled);
                let capped = gb.mul(capped, cap);
                // Linear temperature schedule, applied after the cap — this is
                // the `processed_logits` the sampler and the self-conditioning
                // signal both consume.
                gb.div(capped, temp)
            };

            // Self-conditioning signal for the next denoising step:
            // softmax(logits) @ embed_tokens · sqrt(hidden). Kept in-graph so
            // the [canvas, vocab] block never crosses the device boundary.
            let table = emit.load_param(EMBED_KEY, false)?; // [vocab, hidden]
            let escale =
                emit.synth_param("soft_embed_scale", vec![embed_scale], Shape::new(&[1], f));
            let soft = {
                let mut gb = HirMut::new(emit.hir());
                let probs = gb.sm(logits, -1);
                let se = gb.mm(probs, table);
                gb.mul(se, escale)
            };
            emit.state
                .side_outputs
                .push((SOFT_EMBED_OUTPUT.to_string(), soft));

            if let DecoderOutputs::Reduced { seed } = outputs {
                emit_logit_reduction(emit, logits, canvas, vocab, seed)?;
                // The reduction outputs are what the caller reads; the logits
                // themselves stay on-device as an intermediate.
                let entropy = emit.named(ENTROPY_OUTPUT)?;
                return Ok(Some(emit.wrap(entropy, Shape::new(&[1, canvas], f))));
            }
            Ok(Some(emit.wrap(logits, logits_shape.clone())))
        });
    }

    let primary = match outputs {
        DecoderOutputs::Logits => "logits",
        DecoderOutputs::Reduced { .. } => ENTROPY_OUTPUT,
    };
    flow.output(primary)
        .build_with(&mut SharedWeights(weights), None)
}

/// Reduce `[1, canvas, vocab]` logits to the three per-position quantities the
/// sampler actually needs, so the logits never cross the device boundary.
///
/// * `entropy` — `-Σ p·log p`, computed through a log-sum-exp so it stays finite
///   for peaked distributions (a naive `log(softmax(x))` underflows to `-inf`
///   and yields `0 · -inf = NaN`).
/// * `argmax` — the greedy draft.
/// * `sampled` — a categorical draw via Gumbel-max: `argmax(logits + g)` with
///   `g = -ln(-ln u)`, `u ~ U(0,1)` clamped away from the endpoints.
fn emit_logit_reduction(
    emit: &mut Emit<'_>,
    logits: HirNodeId,
    canvas: usize,
    vocab: usize,
    seed: u64,
) -> Result<()> {
    let f = DType::F32;
    let row = Shape::new(&[1, canvas, vocab], f);
    let red = Shape::new(&[1, canvas, 1], f);
    let flat = Shape::new(&[1, canvas], f);

    let mut gb = HirMut::new(emit.hir());
    // Stable log-softmax: z = x - max, logp = z - log(Σ exp z).
    let max = gb.add_node(
        Op::Reduce {
            op: ReduceOp::Max,
            axes: vec![2],
            keep_dim: true,
        },
        vec![logits],
        red.clone(),
    );
    let z = gb.sub(logits, max);
    let e = gb.exp(z);
    let sum = gb.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![2],
            keep_dim: true,
        },
        vec![e],
        red.clone(),
    );
    let lse = gb.add_node(Op::Activation(Activation::Log), vec![sum], red.clone());
    let logp = gb.sub(z, lse);
    let p = gb.exp(logp);
    let plogp = gb.mul(p, logp);
    let neg_h = gb.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![2],
            keep_dim: false,
        },
        vec![plogp],
        flat.clone(),
    );
    let entropy = gb.add_node(Op::Activation(Activation::Neg), vec![neg_h], flat.clone());

    let argmax = gb.add_node(
        Op::ArgMax {
            axis: 2,
            keep_dim: false,
        },
        vec![logits],
        flat.clone(),
    );

    // Gumbel-max. `u` is clamped off 0 and 1 so the double log stays finite.
    let u = gb.add_node(
        Op::RngUniform {
            low: 1e-7,
            high: 1.0 - 1e-7,
            key: seed,
            op_seed: None,
        },
        vec![logits],
        row.clone(),
    );
    let lu = gb.add_node(Op::Activation(Activation::Log), vec![u], row.clone());
    let nlu = gb.add_node(Op::Activation(Activation::Neg), vec![lu], row.clone());
    let llu = gb.add_node(Op::Activation(Activation::Log), vec![nlu], row.clone());
    let gumbel = gb.add_node(Op::Activation(Activation::Neg), vec![llu], row.clone());
    let perturbed = gb.add(logits, gumbel);
    let sampled = gb.add_node(
        Op::ArgMax {
            axis: 2,
            keep_dim: false,
        },
        vec![perturbed],
        flat,
    );

    emit.set_named(ENTROPY_OUTPUT, entropy);
    emit.state
        .side_outputs
        .push((ARGMAX_OUTPUT.to_string(), argmax));
    emit.state
        .side_outputs
        .push((SAMPLED_OUTPUT.to_string(), sampled));
    Ok(())
}

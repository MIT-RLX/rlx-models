// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! KV-cached pipeline-parallel Qwen3 stage (decode path).
//!
//! Unlike [`crate::pipeline::Qwen3PipelineStage`] (which recomputes the
//! whole prefill graph each token), this stage seeds a per-block KV cache
//! from the prompt, then runs a single-token decode graph each step —
//! O(layers) per token instead of O(seq·layers).
//!
//! Each rank owns a contiguous layer range but builds its block graph with
//! **local** layer indices `0..block_len`, so `bind_decode_inputs` and the
//! `past_k_{i}` cache slots line up. Weight *names* stay global: each layer
//! rebases its own prefix to `model.layers.{start + local}`, which is what
//! lets a rank bind the tensor map it was handed instead of a renumbered
//! clone of it. The cross-block hand-off is
//! the same as the prefill pipeline: First embeds tokens, Middle/Last adopt
//! an `inputs_embeds` hidden tensor, Last runs the final norm + LM head.
//!
//! Being a rank at all — the KV cache, the decode step, keeping only this
//! block's weights, the bucketed compile cache — is
//! [`rlx_core::pipeline_block::FamilyPipelineStage`], shared with every
//! other family. What is left here is [`Qwen3Family`]: two graphs, a few
//! shapes, and how this family is told its position.
//!
//! Drives the same [`PipelineCoordinator`](rlx_distributed::PipelineCoordinator):
//! on the first `forward_step` the cache is empty so every block seeds from
//! the prompt; thereafter each block decodes one token.

use crate::config::Qwen3Config;
use crate::flow::{qwen3_lm_head_stage, rope_tables, validate_cfg};
use anyhow::Result;
use rlx_core::flow_bridge::{WeightLoaderSource, compile_options_from_profile};
use rlx_core::pipeline_block::{FamilyPipelineStage, PipelineFamily, Tensors};
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::blocks::{
    Qwen3DecodeLayerSpec, Qwen3DecodeLayerStage, Qwen3DecoderSpec, Qwen3DecoderStage,
    RopeTablesStage,
};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, SideOutputs};
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_ir::{DType, Shape};
use rlx_runtime::{CompileOptions, Device};
use std::ops::Range;
use std::sync::Arc;

fn rope_slice(cfg: &Qwen3Config, pos: usize) -> (Vec<f32>, Vec<f32>) {
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

/// One prefill layer whose **weights** come from the global layer index
/// `weight_base + local`, while its position in the block graph stays
/// `local`.
///
/// A rank holds a slice of the model, but the checkpoint names every layer
/// by its global index. Rebasing the weight prefix here is what lets a rank
/// bind the tensor map it was handed; the alternative — renaming every
/// per-layer tensor to a local index — clones the whole block's weights per
/// rank, and makes the layer order invisible to anything downstream.
fn seed_layer(
    local: usize,
    weight_base: usize,
    spec: Qwen3DecoderSpec,
    kv_sink: &SideOutputs,
) -> FlowStage {
    let mut stage = Qwen3DecoderStage::layer_with_kv(local, spec, kv_sink.inner());
    stage.layer_prefix = format!("model.layers.{}", weight_base + local);
    FlowStage::Named {
        name: format!("layer{local}"),
        inner: Arc::new(FlowStage::Qwen3Decoder(stage)),
    }
}

/// As [`seed_layer`], for the single-token decode graph.
///
/// `layer_idx` stays `local` deliberately: it indexes this block's
/// `past_k_{i}` / `past_v_{i}` cache slots, which are numbered from zero
/// per rank, whereas `layer_prefix` names weights in the checkpoint.
fn decode_layer(
    local: usize,
    weight_base: usize,
    spec: Qwen3DecodeLayerSpec,
    kv_out: &SideOutputs,
) -> FlowStage {
    let mut stage = Qwen3DecodeLayerStage::layer(local, spec, kv_out.inner());
    stage.layer_prefix = format!("model.layers.{}", weight_base + local);
    FlowStage::Named {
        name: format!("layer{local}"),
        inner: Arc::new(FlowStage::Qwen3DecodeLayer(stage)),
    }
}

/// Prefill graph for one block over `block_len` local layers, exporting
/// per-layer K/V (to seed the cache). First/Single embed `input_ids`;
/// others adopt the `inputs_embeds` hidden tensor. Logits blocks gather
/// the last token + final-norm + LM head; others output raw hidden.
fn build_block_seed(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    block_len: usize,
    weight_base: usize,
    embed_input: bool,
    produce_logits: bool,
) -> Result<BuiltModel> {
    validate_cfg(cfg)?;
    let f = DType::F32;
    let h = cfg.hidden_size;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let (cos, sin) = rope_tables(cfg);
    let spec = Qwen3DecoderSpec {
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: dh,
        eps,
        hidden_shape: hidden_shape.clone(),
        batch,
        seq,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
        mask: crate::builder::attn_mask_kind(cfg),
        moe: None,
    };
    let sink = SideOutputs::new();

    let mut flow = ModelFlow::new("qwen3_block_seed").with_profile(CompileProfile::qwen3_prefill());
    flow = if embed_input {
        flow.input("input_ids", Shape::new(&[batch, seq], DType::I32))
    } else {
        flow.input("inputs_embeds", hidden_shape.clone())
    };
    flow = flow
        .rope_tables(RopeTablesStage::param(
            cfg.max_position_embeddings,
            dh / 2,
            cos,
            sin,
        ))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);
    if embed_input {
        flow = flow.token_embed();
    }
    flow = flow.repeat_layers(block_len, {
        let spec = spec.clone();
        let sink = sink.clone();
        move |i| seed_layer(i, weight_base, spec.clone(), &sink)
    });

    let built = if produce_logits {
        flow.gather_last_token_at(batch, seq)
            .final_norm(eps)
            .raw_stage(qwen3_lm_head_stage(cfg))
            .output("logits")
            .build(&mut WeightLoaderSource(weights))?
    } else {
        flow.output("hidden")
            .build(&mut WeightLoaderSource(weights))?
    };
    Ok(built.with_extra_hir_outputs(sink.drain()))
}

/// Single-token decode graph for one block over `block_len` local layers,
/// reading/writing per-layer K/V for `past_seq` cached positions.
fn build_block_decode(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    block_len: usize,
    weight_base: usize,
    embed_input: bool,
    produce_logits: bool,
    use_custom_mask: bool,
) -> Result<BuiltModel> {
    validate_cfg(cfg)?;
    let f = DType::F32;
    let h = cfg.hidden_size;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let half = dh / 2;
    let kv_dim = cfg.kv_proj_dim();
    let hidden_shape = Shape::new(&[batch, 1, h], f);
    let past_kv_shape = Shape::new(&[batch, past_seq, kv_dim], f);
    let dspec = Qwen3DecodeLayerSpec {
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: dh,
        kv_group_size: cfg.kv_group_size(),
        eps,
        use_custom_mask,
        hidden_shape: hidden_shape.clone(),
        batch,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
        moe: None,
    };
    let kv_out = SideOutputs::new();

    let mut flow =
        ModelFlow::new("qwen3_block_decode").with_profile(CompileProfile::qwen3_decode());
    flow = if embed_input {
        flow.input("input_ids", Shape::new(&[batch, 1], DType::I32))
    } else {
        flow.input("inputs_embeds", hidden_shape.clone())
    };
    flow = flow
        .input("rope_cos", Shape::new(&[1, half], f))
        .input("rope_sin", Shape::new(&[1, half], f));
    if use_custom_mask {
        flow = flow.input("mask", Shape::new(&[batch, past_seq + 1], f));
    }
    for i in 0..block_len {
        flow = flow
            .input(format!("past_k_{i}"), past_kv_shape.clone())
            .input(format!("past_v_{i}"), past_kv_shape.clone());
    }
    flow = flow
        .bind_decode_inputs(block_len, use_custom_mask, true)
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);
    if embed_input {
        flow = flow.token_embed();
    }
    flow = flow.repeat_layers(block_len, {
        let spec = dspec.clone();
        let sink = kv_out.clone();
        move |i| decode_layer(i, weight_base, spec.clone(), &sink)
    });

    let built = if produce_logits {
        flow.final_norm(eps)
            .raw_stage(qwen3_lm_head_stage(cfg))
            .output("logits")
            .build(&mut WeightLoaderSource(weights))?
    } else {
        flow.output("hidden_states")
            .build(&mut WeightLoaderSource(weights))?
    };
    Ok(built.with_extra_hir_outputs(kv_out.drain()))
}

/// Qwen3's answer to [`PipelineFamily`]: the two graphs and the handful of
/// shapes that make this family this family.
///
/// Everything about *being* a rank — seeding a KV cache from the prompt,
/// decoding a token per call, holding only this block's weights, the
/// bucketed compile cache — lives once in [`FamilyPipelineStage`].
pub struct Qwen3Family {
    cfg: Qwen3Config,
}

impl Qwen3Family {
    pub fn new(cfg: Qwen3Config) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }
}

impl PipelineFamily for Qwen3Family {
    fn seed_graph(
        &self,
        weights: &mut dyn WeightLoader,
        batch: usize,
        seq: usize,
        layers: Range<usize>,
        embed: bool,
        logits: bool,
    ) -> Result<BuiltModel> {
        build_block_seed(
            &self.cfg,
            weights,
            batch,
            seq,
            layers.len(),
            layers.start,
            embed,
            logits,
        )
    }

    fn decode_graph(
        &self,
        weights: &mut dyn WeightLoader,
        batch: usize,
        past: usize,
        layers: Range<usize>,
        embed: bool,
        logits: bool,
        custom_mask: bool,
    ) -> Result<BuiltModel> {
        build_block_decode(
            &self.cfg,
            weights,
            batch,
            past,
            layers.len(),
            layers.start,
            embed,
            logits,
            custom_mask,
        )
    }

    fn num_hidden_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }

    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    fn kv_proj_dim(&self) -> usize {
        self.cfg.kv_proj_dim()
    }

    fn tie_word_embeddings(&self) -> bool {
        self.cfg.tie_word_embeddings
    }

    /// Qwen3 is told its position as rope tables sliced on the host — unlike
    /// llama, which takes the position itself and derives them in the graph.
    fn decode_inputs(&self, past: usize) -> Vec<(&'static str, Vec<f32>)> {
        let (cos, sin) = rope_slice(&self.cfg, past);
        vec![("rope_cos", cos), ("rope_sin", sin)]
    }

    fn compile_options(&self, device: Device, decode: bool) -> CompileOptions {
        let profile = if decode {
            CompileProfile::qwen3_decode()
        } else {
            CompileProfile::qwen3_prefill()
        };
        compile_options_from_profile(&profile, device, KernelDispatchConfig::default())
    }
}

/// One rank's block of a Qwen3 model.
pub type Qwen3PipelineDecodeStage = FamilyPipelineStage<Qwen3Family>;

/// This rank's block, with the layers split evenly.
pub fn stage(
    cfg: Qwen3Config,
    device: Device,
    rank: u32,
    world: u32,
    all_weights: Tensors,
) -> Qwen3PipelineDecodeStage {
    FamilyPipelineStage::new(Qwen3Family::new(cfg), device, rank, world, all_weights)
}

/// This rank's block, with its layers given rather than split evenly.
///
/// An even split is only right when the machines are alike: give a 16 GB
/// laptop and a 64 GB desktop seven layers each and the laptop decides how
/// fast the whole pipeline runs, or fails to hold its block at all.
///
/// **Every rank must be given the same overall split.** The ranges decide
/// which weights a rank loads and how many layers its graph has; two ranks
/// disagreeing runs some layers twice and others never, with no shape error
/// to catch it.
pub fn stage_with_layers(
    cfg: Qwen3Config,
    device: Device,
    rank: u32,
    world: u32,
    all_weights: Tensors,
    layers: Range<usize>,
) -> Qwen3PipelineDecodeStage {
    FamilyPipelineStage::with_layer_range(
        Qwen3Family::new(cfg),
        device,
        rank,
        world,
        all_weights,
        layers,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use rlx_distributed::{NetTransport, PipelineCoordinator, ProcessGroup};
    use std::net::{Ipv4Addr, SocketAddr, TcpListener};
    use std::sync::Arc;
    use std::thread;

    fn cfg() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 48,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 32,
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

    fn fill(name: &str, n: usize) -> Vec<f32> {
        let seed = name
            .bytes()
            .fold(2166136261u32, |a, b| (a ^ b as u32).wrapping_mul(16777619));
        (0..n)
            .map(|i| {
                let x = seed.wrapping_add((i as u32).wrapping_mul(2654435761));
                ((x % 2000) as f32 / 1000.0 - 1.0) * 0.05
            })
            .collect()
    }

    fn synth(c: &Qwen3Config) -> Tensors {
        fn put(t: &mut Tensors, key: String, shape: Vec<usize>) {
            let n: usize = shape.iter().product();
            let d = fill(&key, n);
            t.insert(key, (d, shape));
        }
        let h = c.hidden_size;
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let im = c.intermediate_size;
        let dh = c.head_dim;
        let mut t = Tensors::new();
        put(
            &mut t,
            "model.embed_tokens.weight".into(),
            vec![c.vocab_size, h],
        );
        for l in 0..c.num_hidden_layers {
            let lp = format!("model.layers.{l}");
            put(&mut t, format!("{lp}.input_layernorm.weight"), vec![h]);
            put(
                &mut t,
                format!("{lp}.post_attention_layernorm.weight"),
                vec![h],
            );
            put(&mut t, format!("{lp}.self_attn.q_proj.weight"), vec![q, h]);
            put(&mut t, format!("{lp}.self_attn.k_proj.weight"), vec![kv, h]);
            put(&mut t, format!("{lp}.self_attn.v_proj.weight"), vec![kv, h]);
            put(&mut t, format!("{lp}.self_attn.o_proj.weight"), vec![h, q]);
            put(&mut t, format!("{lp}.self_attn.q_norm.weight"), vec![dh]);
            put(&mut t, format!("{lp}.self_attn.k_norm.weight"), vec![dh]);
            put(&mut t, format!("{lp}.mlp.gate_proj.weight"), vec![im, h]);
            put(&mut t, format!("{lp}.mlp.up_proj.weight"), vec![im, h]);
            put(&mut t, format!("{lp}.mlp.down_proj.weight"), vec![h, im]);
        }
        put(&mut t, "model.norm.weight".into(), vec![h]);
        put(&mut t, "lm_head.weight".into(), vec![c.vocab_size, h]);
        t
    }

    fn argmax(v: &[f32]) -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap()
    }

    /// Single-node greedy reference via the KV-cached generator (the exact
    /// decode semantics the pipeline must reproduce).
    fn reference_cached(c: &Qwen3Config, w: &Tensors, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut wm = WeightMap::from_tensors(w.clone());
        let mut g =
            crate::generator::Qwen3Generator::from_loader(c.clone(), &mut wm, Device::Cpu).unwrap();
        g.prefill(prompt);
        let opts = crate::sampling::SampleOpts::greedy();
        g.generate_cached(n, opts).unwrap()
    }

    /// Spin up `world` loopback ranks and return what each one generated.
    /// Every rank should return the same ids, because the sampled token is
    /// broadcast.
    fn pipeline_outputs(
        c: &Qwen3Config,
        w: &Tensors,
        prompt: &[u32],
        n: usize,
        cached: bool,
        world: u32,
    ) -> Vec<Vec<u32>> {
        pipeline_outputs_with(c, w, prompt, n, cached, world, 64)
    }

    /// As above, but with an explicit decode-cache ceiling — the ladder's
    /// top bucket. A server sizes it from the model's context window, which
    /// is far larger than any test prompt, so the bucket a short sequence
    /// lands in is not the same one a `max_past` of 64 gives.
    #[allow(clippy::too_many_arguments)]
    fn pipeline_outputs_with(
        c: &Qwen3Config,
        w: &Tensors,
        prompt: &[u32],
        n: usize,
        cached: bool,
        world: u32,
        max_past: usize,
    ) -> Vec<Vec<u32>> {
        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let c = Arc::new(c.clone());
        let w = Arc::new(w.clone());
        let prompt = Arc::new(prompt.to_vec());

        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let (addrs, c, w, prompt) = (addrs.clone(), c.clone(), w.clone(), prompt.clone());
                thread::spawn(move || {
                    let t =
                        NetTransport::from_listener(rank as u32, world, listener, addrs, 1 << 20)
                            .unwrap();
                    let stage = stage((*c).clone(), Device::Cpu, rank as u32, world, (*w).clone());
                    let mut stage = if cached {
                        stage.with_decode_cache(max_past)
                    } else {
                        stage
                    };
                    let coord = PipelineCoordinator::new(ProcessGroup::new(Arc::new(t)));
                    let mut tokens = (*prompt).clone();
                    coord
                        .generate(&mut stage, &mut tokens, n, argmax, |_| false)
                        .unwrap()
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    }

    #[test]
    fn decode_pipeline_matches_cached_generator() {
        let c = cfg();
        let w = synth(&c);
        let prompt = vec![1u32, 5, 3, 2, 7];
        let n = 6usize;
        let reference = reference_cached(&c, &w, &prompt, n);

        // Both the recompile-each-step path and the bucketed-compile-cache
        // path must reproduce the reference.
        for cached in [false, true] {
            for world in [2u32, 3] {
                for (rank, produced) in pipeline_outputs(&c, &w, &prompt, n, cached, world)
                    .into_iter()
                    .enumerate()
                {
                    assert_eq!(
                        produced, reference,
                        "cached={cached} world={world} rank={rank}"
                    );
                }
            }
        }
    }

    /// Shaped like Qwen3-0.6B: 28 layers (so a world of 2 splits 14/14),
    /// tied embeddings, and `num_heads * head_dim != hidden_size`. Narrow
    /// enough to run in a test.
    fn cfg_realistic() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 28,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 32,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: true,
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

    #[test]
    fn a_deep_pipeline_matches_the_generator_at_a_server_sized_cache() {
        // The combination a server actually runs: many layers, tied
        // embeddings, and a decode cache sized from the context window
        // rather than from the prompt. Sharded output diverged from
        // single-machine output on the *first decode step* under exactly
        // this shape while the small-config test passed.
        let c = cfg_realistic();
        let mut w = synth(&c);
        w.remove("lm_head.weight");
        let prompt = vec![3u32, 17, 42, 8, 1, 99];
        let n = 6usize;
        let reference = reference_cached(&c, &w, &prompt, n);

        for world in [2u32, 3] {
            for (rank, produced) in
                pipeline_outputs_with(&c, &w, &prompt, n, true, world, c.max_position_embeddings)
                    .into_iter()
                    .enumerate()
            {
                assert_eq!(produced, reference, "world={world} rank={rank}");
            }
        }
    }

    #[test]
    fn a_tied_lm_head_works_on_the_last_block() {
        // Tied embeddings mean there is no `lm_head.weight`: the embedding
        // table *is* the head. The block that holds the head is the one that
        // does not embed, so nothing in its graph reads that table — and the
        // tied head has to read it itself. It did not, and every real tied
        // model (which is most small ones) failed its first prefill with
        // "missing model.embed_tokens.weight for tied lm_head".
        let mut c = cfg();
        c.tie_word_embeddings = true;
        let mut w = synth(&c);
        w.remove("lm_head.weight");

        let prompt = vec![1u32, 5, 3, 2, 7];
        let n = 4usize;
        let reference = reference_cached(&c, &w, &prompt, n);

        for world in [2u32, 3] {
            for (rank, produced) in pipeline_outputs(&c, &w, &prompt, n, true, world)
                .into_iter()
                .enumerate()
            {
                assert_eq!(produced, reference, "world={world} rank={rank}");
            }
        }
    }
}

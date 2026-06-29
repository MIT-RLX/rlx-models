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
//! **local** layer indices `0..block_len` (so `bind_decode_inputs` and the
//! `past_k_{i}` cache slots line up), and its weights are remapped from
//! global `model.layers.{g}.*` to local `model.layers.{g-start}.*`. The
//! cross-block hand-off is the same as the prefill pipeline: First embeds
//! tokens, Middle/Last adopt an `inputs_embeds` hidden tensor, Last runs
//! the final norm + LM head. The stage is stateful — its `cache` is `None`
//! until the first call seeds it.
//!
//! Drives the same [`PipelineCoordinator`](rlx_distributed::PipelineCoordinator):
//! on the first `forward_step` the cache is empty so every block seeds from
//! the prompt; thereafter each block decodes one token.

use crate::config::Qwen3Config;
use crate::flow::{qwen3_lm_head_stage, rope_tables, validate_cfg};
use anyhow::{Result, bail};
use rlx_core::autoregressive::{
    KvCacheState, kv_from_prefill_outputs, run_bucketed_kv_decode, split_decode_logits_kv,
};
use rlx_core::flow_bridge::{WeightLoaderSource, compile_options_from_profile};
use rlx_core::flow_util::graph_from_built;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use rlx_distributed::{
    BlockInput, BlockOutput, BlockRole, BlockRunner, block_role, pipeline_layer_range,
};
use rlx_flow::blocks::{
    Qwen3DecodeLayerSpec, Qwen3DecoderSpec, RopeTablesStage, qwen3_decode_layer_fused,
    qwen3_prefill_layer_fused_kv,
};
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow, SideOutputs};
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_ir::{DType, Shape};
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput};
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;

type Tensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

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
        move |i| qwen3_prefill_layer_fused_kv(i, spec.clone(), sink.inner())
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
        move |i| qwen3_decode_layer_fused(i, spec.clone(), sink.inner())
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

/// Stateful KV-cached pipeline stage for one rank.
pub struct Qwen3PipelineDecodeStage {
    cfg: Qwen3Config,
    device: Device,
    role: BlockRole,
    block_len: usize,
    embed_input: bool,
    produce_logits: bool,
    batch: usize,
    /// Weights for this block, remapped to local layer indices.
    weights: Tensors,
    /// `None` until the first call seeds it from the prompt.
    cache: Option<KvCacheState>,
    /// Optional bucketed compile cache. When set, decode steps reuse a
    /// compiled graph per power-of-two `past_seq` bucket (custom mask +
    /// padded K/V) instead of recompiling each step — O(log N) compiles
    /// over a generation instead of O(N).
    decode_cache: Option<BucketedCompileCache>,
}

impl Qwen3PipelineDecodeStage {
    pub fn new(
        cfg: Qwen3Config,
        device: Device,
        rank: u32,
        world: u32,
        all_weights: Tensors,
    ) -> Self {
        let range = pipeline_layer_range(cfg.num_hidden_layers, rank, world);
        let role = block_role(rank, world);
        let (embed_input, produce_logits) = match role {
            BlockRole::Single => (true, true),
            BlockRole::First => (true, false),
            BlockRole::Middle => (false, false),
            BlockRole::Last => (false, true),
        };
        let start = range.start;
        let block_len = range.len();

        // Filter to this block + remap per-layer names to local indices.
        let mut weights = Tensors::new();
        for (k, v) in all_weights {
            if let Some(rest) = k.strip_prefix("model.layers.") {
                let mut it = rest.splitn(2, '.');
                let idx: usize = it.next().unwrap_or("").parse().unwrap_or(usize::MAX);
                let suffix = it.next().unwrap_or("");
                if range.contains(&idx) {
                    weights.insert(format!("model.layers.{}.{suffix}", idx - start), v);
                }
            } else if k == "model.embed_tokens.weight" {
                if embed_input || (produce_logits && cfg.tie_word_embeddings) {
                    weights.insert(k, v);
                }
            } else if k == "model.norm.weight" {
                if produce_logits {
                    weights.insert(k, v);
                }
            } else if k == "lm_head.weight" && produce_logits && !cfg.tie_word_embeddings {
                weights.insert(k, v);
            }
        }

        Self {
            cfg,
            device,
            role,
            block_len,
            embed_input,
            produce_logits,
            batch: 1,
            weights,
            cache: None,
            decode_cache: None,
        }
    }

    /// Enable the bucketed decode compile cache (compile once per
    /// power-of-two `past_seq` bucket up to `max_past`). Strongly
    /// recommended for steady-state generation.
    pub fn with_decode_cache(mut self, max_past: usize) -> Self {
        self.decode_cache = Some(BucketedCompileCache::power_of_two_ladder(
            self.device,
            1,
            max_past.max(1) as u64,
        ));
        self
    }

    pub fn role(&self) -> BlockRole {
        self.role
    }

    fn opts(&self, decode: bool) -> CompileOptions {
        let profile = if decode {
            CompileProfile::qwen3_decode()
        } else {
            CompileProfile::qwen3_prefill()
        };
        compile_options_from_profile(&profile, self.device, KernelDispatchConfig::default())
    }

    /// First call: prefill the whole input, seeding the cache.
    fn seed(&mut self, input: BlockInput<'_>) -> Result<Vec<f32>> {
        let seq = match &input {
            BlockInput::Tokens(t) => t.len(),
            BlockInput::Hidden(hh) => hh.len() / (self.batch * self.cfg.hidden_size),
        };
        let mut wm = WeightMap::from_tensors(self.weights.clone());
        let built = build_block_seed(
            &self.cfg,
            &mut wm,
            self.batch,
            seq,
            self.block_len,
            self.embed_input,
            self.produce_logits,
        )?;
        let (graph, params) = graph_from_built(built)?;
        let mut compiled = Session::new(self.device).compile_with(graph, &self.opts(false));
        for (n, d) in &params {
            compiled.set_param(n, d);
        }
        let outputs = match input {
            BlockInput::Tokens(t) => {
                let ids: Vec<f32> = t.iter().map(|&x| x as f32).collect();
                compiled.run(&[("input_ids", ids.as_slice())])
            }
            BlockInput::Hidden(hh) => compiled.run(&[("inputs_embeds", hh)]),
        };
        let (main, kv) = kv_from_prefill_outputs(
            outputs,
            self.batch,
            seq,
            self.cfg.kv_proj_dim(),
            self.block_len,
        )?;
        self.cache = Some(kv);
        Ok(main)
    }

    /// Steady-state decode, recompiling the graph each step (no cache).
    fn decode_recompile(&mut self, input: BlockInput<'_>) -> Result<Vec<f32>> {
        let past = self.cache.as_ref().unwrap().past_len;
        let (cos, sin) = rope_slice(&self.cfg, past);

        let mut wm = WeightMap::from_tensors(self.weights.clone());
        let built = build_block_decode(
            &self.cfg,
            &mut wm,
            self.batch,
            past,
            self.block_len,
            self.embed_input,
            self.produce_logits,
            false,
        )?;
        let (graph, params) = graph_from_built(built)?;
        let mut compiled = Session::new(self.device).compile_with(graph, &self.opts(true));
        for (n, d) in &params {
            compiled.set_param(n, d);
        }

        // Build the input list. Owned buffers live until after `run`.
        let (main_name, main_buf): (&str, Vec<f32>) = match input {
            BlockInput::Tokens(t) => {
                if t.len() <= past {
                    bail!("decode: token history len {} <= past {past}", t.len());
                }
                ("input_ids", vec![t[past] as f32])
            }
            BlockInput::Hidden(hh) => {
                let per = self.batch * self.cfg.hidden_size;
                if hh.len() != per {
                    bail!(
                        "decode: expected one hidden position ({per} f32), got {}",
                        hh.len()
                    );
                }
                ("inputs_embeds", hh.to_vec())
            }
        };
        let key_strs: Vec<String> = (0..self.block_len)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let cache = self.cache.as_ref().unwrap();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(3 + 2 * self.block_len);
        inputs.push((main_name, main_buf.as_slice()));
        inputs.push(("rope_cos", cos.as_slice()));
        inputs.push(("rope_sin", sin.as_slice()));
        for i in 0..self.block_len {
            inputs.push((key_strs[2 * i].as_str(), cache.layers_k[i].as_slice()));
            inputs.push((key_strs[2 * i + 1].as_str(), cache.layers_v[i].as_slice()));
        }
        let outputs = compiled.run(&inputs);
        let (main, layers_k, layers_v) = split_decode_logits_kv(outputs, self.block_len)?;

        let c = self.cache.as_mut().unwrap();
        c.past_len = past + 1;
        c.layers_k = layers_k;
        c.layers_v = layers_v;
        Ok(main)
    }

    /// Steady-state decode reusing a per-bucket compiled graph (custom
    /// mask + padded K/V), via the shared bucketed driver. Same result as
    /// [`Self::decode_recompile`] but O(log N) compiles per generation.
    fn decode_bucketed(&mut self, input: BlockInput<'_>) -> Result<Vec<f32>> {
        let past = self.cache.as_ref().unwrap().past_len;
        let (cos, sin) = rope_slice(&self.cfg, past);
        let kv = self.cache.as_ref().unwrap().clone();
        let opts = self.opts(true);
        let kv_dim = self.cfg.kv_proj_dim();
        let block_len = self.block_len;
        let embed = self.embed_input;
        let logits = self.produce_logits;

        // The new token (First/Single) or its hidden (Middle/Last).
        let (main_name, main_buf): (&str, Vec<f32>) = match input {
            BlockInput::Tokens(t) => {
                if t.len() <= past {
                    bail!("decode: token history len {} <= past {past}", t.len());
                }
                ("input_ids", vec![t[past] as f32])
            }
            BlockInput::Hidden(hh) => {
                let per = self.batch * self.cfg.hidden_size;
                if hh.len() != per {
                    bail!(
                        "decode: expected one hidden position ({per} f32), got {}",
                        hh.len()
                    );
                }
                ("inputs_embeds", hh.to_vec())
            }
        };

        // Bucket upper bound for this `past`, and the matching mask.
        let cache_dec = self.decode_cache.as_mut().unwrap();
        let upper = cache_dec
            .bucket_for(past as u64)
            .and_then(|idx| cache_dec.buckets().nth(idx).map(|r| (r.end - 1) as usize))
            .unwrap_or(past);
        let mask = bucket_decode_mask(past, upper);

        let fixed = [
            CacheRunInput {
                name: main_name,
                data: &main_buf,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_cos",
                data: &cos,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_sin",
                data: &sin,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: &mask,
                row_inner: None,
            },
        ];

        let cfg = self.cfg.clone();
        let weights = self.weights.clone();
        let (main, layers_k, layers_v) = run_bucketed_kv_decode(
            cache_dec,
            past,
            &kv,
            kv_dim,
            block_len,
            &fixed,
            |upper| {
                let mut wm = WeightMap::from_tensors(weights.clone());
                let built = build_block_decode(
                    &cfg,
                    &mut wm,
                    1,
                    upper as usize,
                    block_len,
                    embed,
                    logits,
                    true,
                )
                .expect("block decode graph");
                graph_from_built(built).expect("block decode graph parts")
            },
            &opts,
        )?;

        let c = self.cache.as_mut().unwrap();
        c.past_len = past + 1;
        c.layers_k = layers_k;
        c.layers_v = layers_v;
        Ok(main)
    }
}

impl BlockRunner for Qwen3PipelineDecodeStage {
    fn role(&self) -> BlockRole {
        self.role
    }

    fn run(&mut self, input: BlockInput<'_>) -> Result<BlockOutput> {
        let main = if self.cache.is_none() {
            self.seed(input)?
        } else if self.decode_cache.is_some() {
            self.decode_bucketed(input)?
        } else {
            self.decode_recompile(input)?
        };
        Ok(if self.produce_logits {
            BlockOutput::Logits(main)
        } else {
            BlockOutput::Hidden(main)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                let listeners: Vec<TcpListener> = (0..world)
                    .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
                    .collect();
                let addrs: Vec<SocketAddr> =
                    listeners.iter().map(|l| l.local_addr().unwrap()).collect();
                let c = Arc::new(c.clone());
                let w = Arc::new(w.clone());
                let prompt = Arc::new(prompt.clone());

                let handles: Vec<_> = listeners
                    .into_iter()
                    .enumerate()
                    .map(|(rank, listener)| {
                        let (addrs, c, w, prompt) =
                            (addrs.clone(), c.clone(), w.clone(), prompt.clone());
                        thread::spawn(move || {
                            let t = NetTransport::from_listener(
                                rank as u32,
                                world,
                                listener,
                                addrs,
                                1 << 20,
                            )
                            .unwrap();
                            let stage = Qwen3PipelineDecodeStage::new(
                                (*c).clone(),
                                Device::Cpu,
                                rank as u32,
                                world,
                                (*w).clone(),
                            );
                            let mut stage = if cached {
                                stage.with_decode_cache(64)
                            } else {
                                stage
                            };
                            let coord = PipelineCoordinator::new(ProcessGroup::new(Arc::new(t)));
                            let mut tokens = (*prompt).clone();
                            coord
                                .generate(&mut stage, &mut tokens, n, |l| argmax(l), |_| false)
                                .unwrap()
                        })
                    })
                    .collect();

                for (rank, h) in handles.into_iter().enumerate() {
                    let produced = h.join().unwrap();
                    assert_eq!(
                        produced, reference,
                        "cached={cached} world={world} rank={rank}"
                    );
                }
            }
        }
    }
}

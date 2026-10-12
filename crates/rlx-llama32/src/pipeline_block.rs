// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Llama 3.2 as one rank of a model split across machines.
//!
//! Everything about *being* a rank — seeding a KV cache from the prompt,
//! decoding a token per call, keeping only this block's weights, the
//! bucketed compile cache — lives once in
//! [`rlx_core::pipeline_block::FamilyPipelineStage`]. What is here is only
//! what makes this family this family: two graphs and a handful of shapes.
//!
//! Mistral and Ministral come along for free: they are this runner with a
//! different config (see `rlx_mistral`, which re-exports it).
//!
//! The graphs come from [`Llama32Flow`], which already knew how to build
//! every shape a block needs — a hidden-state input instead of an embedding,
//! raw hidden out instead of logits, K/V exported for the cache — and gained
//! `layer_range` so it can be pointed at a slice of the model.

use crate::config::Llama32Config;
use crate::flow::Llama32Flow;
use anyhow::Result;
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::pipeline_block::{FamilyPipelineStage, PipelineFamily, Tensors};
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::{BuiltModel, CompileProfile};
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_runtime::{CompileOptions, Device};
use std::ops::Range;

/// Llama 3.2's answer to [`PipelineFamily`].
pub struct Llama32Family {
    cfg: Llama32Config,
}

impl Llama32Family {
    pub fn new(cfg: Llama32Config) -> Self {
        // No rope tables to resolve here: the decode graph is told the
        // position and works them out itself, scaling included.
        Self { cfg }
    }

    pub fn config(&self) -> &Llama32Config {
        &self.cfg
    }

    /// Shared between the two graph builders: a block adopts a hidden state
    /// unless it embeds, and outputs hidden unless it owns the head.
    fn shape<'a>(flow: Llama32Flow<'a>, embed: bool, logits: bool) -> Llama32Flow<'a> {
        let flow = if embed { flow } else { flow.inputs_embeds() };
        if logits {
            flow.lm_head().last_token_logits()
        } else {
            // The model's final norm goes with the head, so a block that
            // does not own the head must not apply it: the next block would
            // normalise an already-normalised hidden state.
            flow.hidden_only().without_final_norm()
        }
    }
}

impl PipelineFamily for Llama32Family {
    fn seed_graph(
        &self,
        weights: &mut dyn WeightLoader,
        batch: usize,
        seq: usize,
        layers: Range<usize>,
        embed: bool,
        logits: bool,
    ) -> Result<BuiltModel> {
        Self::shape(
            Llama32Flow::for_prefill(&self.cfg, batch, seq)
                .layer_range(layers)
                .export_kv(),
            embed,
            logits,
        )
        .build(weights)
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
        let flow = Llama32Flow::for_decode(&self.cfg, batch, past).layer_range(layers);
        let flow = if custom_mask {
            flow.custom_mask()
        } else {
            flow
        };
        Self::shape(flow, embed, logits).build(weights)
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

    fn decode_inputs(&self, past: usize) -> Vec<(&'static str, Vec<f32>)> {
        // Llama's decode graph takes the position and derives the rope in
        // the graph (`DecodeRopeParamsStage`), rather than taking cos/sin
        // sliced on the host as qwen3's does.
        vec![("position", vec![past as f32])]
    }

    fn compile_options(&self, device: Device, decode: bool) -> CompileOptions {
        let profile = if decode {
            CompileProfile::llama32_decode()
        } else {
            CompileProfile::llama32_prefill()
        };
        compile_options_from_profile(&profile, device, KernelDispatchConfig::default())
    }
}

/// One rank's block of a Llama 3.2 (or Mistral) model.
pub type Llama32PipelineStage = FamilyPipelineStage<Llama32Family>;

/// This rank's block, with the layers split evenly.
pub fn stage(
    cfg: Llama32Config,
    device: Device,
    rank: u32,
    world: u32,
    all_weights: Tensors,
) -> Llama32PipelineStage {
    FamilyPipelineStage::new(Llama32Family::new(cfg), device, rank, world, all_weights)
}

/// This rank's block, with its layers given rather than split evenly.
pub fn stage_with_layers(
    cfg: Llama32Config,
    device: Device,
    rank: u32,
    world: u32,
    all_weights: Tensors,
    layers: Range<usize>,
) -> Llama32PipelineStage {
    FamilyPipelineStage::with_layer_range(
        Llama32Family::new(cfg),
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
    use std::collections::HashMap;
    use std::net::{Ipv4Addr, SocketAddr, TcpListener};
    use std::sync::Arc;
    use std::thread;

    fn cfg() -> Llama32Config {
        Llama32Config {
            vocab_size: 48,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            head_dim: Some(8),
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

    /// Deterministic pseudo-weights, so the reference and the pipeline are
    /// comparing the same model rather than the same random seed.
    fn fill(name: &str, n: usize) -> Vec<f32> {
        let seed = name
            .bytes()
            .fold(7u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
        (0..n)
            .map(|i| {
                let x = seed
                    .wrapping_add(i as u64)
                    .wrapping_mul(6364136223846793005);
                (((x >> 33) % 2000) as f32 - 1000.0) / 20000.0
            })
            .collect()
    }

    fn synth(c: &Llama32Config) -> Tensors {
        let h = c.hidden_size;
        let dh = c.head_dim();
        let q = c.num_attention_heads * dh;
        let kv = c.num_key_value_heads * dh;
        let im = c.intermediate_size;
        let mut t: Tensors = HashMap::new();
        let mut put = |k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            let d = fill(&k, n);
            t.insert(k, (d, shape));
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
        t
    }

    fn argmax(v: &[f32]) -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap()
    }

    /// The single-machine answer: llama32's own KV-cached generator.
    fn reference(c: &Llama32Config, w: &Tensors, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut wm = WeightMap::from_tensors(w.clone());
        let mut g =
            crate::generator::Llama32Generator::from_loader(c.clone(), &mut wm, Device::Cpu)
                .unwrap();
        g.prefill(prompt);
        g.generate_cached(n, rlx_qwen3::sampling::SampleOpts::greedy())
            .unwrap()
    }

    /// Spin up `world` ranks on loopback and return what each generated.
    fn sharded(
        c: &Llama32Config,
        w: &Tensors,
        prompt: &[u32],
        n: usize,
        world: u32,
        cached: bool,
        split: Option<Vec<usize>>,
    ) -> Vec<Vec<u32>> {
        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let (c, w, prompt) = (
            Arc::new(c.clone()),
            Arc::new(w.clone()),
            Arc::new(prompt.to_vec()),
        );
        let split = Arc::new(split);

        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let (addrs, c, w, prompt, split) = (
                    addrs.clone(),
                    c.clone(),
                    w.clone(),
                    prompt.clone(),
                    split.clone(),
                );
                thread::spawn(move || {
                    let t =
                        NetTransport::from_listener(rank as u32, world, listener, addrs, 1 << 20)
                            .unwrap();
                    // Counts in rank order; rank 0 holds the last block, so
                    // the ranges stack upward from rank `world-1`.
                    let stage = match split.as_ref() {
                        Some(counts) => {
                            let start: usize = counts[rank + 1..].iter().sum();
                            stage_with_layers(
                                (*c).clone(),
                                Device::Cpu,
                                rank as u32,
                                world,
                                (*w).clone(),
                                start..start + counts[rank],
                            )
                        }
                        None => stage((*c).clone(), Device::Cpu, rank as u32, world, (*w).clone()),
                    };
                    let mut stage = if cached {
                        stage.with_decode_cache(64)
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

    /// Build the graphs for every role, with no network involved.
    ///
    /// First because a pipeline test cannot tell a graph that fails to build
    /// from one that hangs: a rank that panics leaves the others waiting on
    /// a hidden state that will never arrive, and the whole test deadlocks
    /// with nothing to read.
    #[test]
    fn every_role_builds_its_graphs() {
        let c = cfg();
        let w = synth(&c);
        let family = Llama32Family::new(c.clone());
        // rank 1 of 2 is First (embeds, hidden out); rank 0 is Last (adopts
        // hidden, owns the head). A middle block does neither.
        for (name, layers, embed, logits) in [
            ("first", 0..3, true, false),
            ("last", 3..6, false, true),
            ("middle", 2..4, false, false),
            ("single", 0..6, true, true),
        ] {
            let mut wm =
                WeightMap::from_tensors(family.block_weights(w.clone(), &layers, embed, logits));
            family
                .seed_graph(&mut wm, 1, 4, layers.clone(), embed, logits)
                .unwrap_or_else(|e| panic!("{name} seed graph: {e:#}"));

            let mut wm =
                WeightMap::from_tensors(family.block_weights(w.clone(), &layers, embed, logits));
            family
                .decode_graph(&mut wm, 1, 4, layers.clone(), embed, logits, true)
                .unwrap_or_else(|e| panic!("{name} decode graph: {e:#}"));
        }
    }

    #[test]
    fn a_split_llama_answers_the_same_as_a_whole_one() {
        // The claim the whole thing rests on: splitting the layers across
        // machines must not change the answer.
        let c = cfg();
        let w = synth(&c);
        let prompt = vec![1u32, 5, 3, 2, 7];
        let n = 6usize;
        let want = reference(&c, &w, &prompt, n);

        for cached in [false, true] {
            for world in [2u32, 3] {
                for (rank, got) in sharded(&c, &w, &prompt, n, world, cached, None)
                    .into_iter()
                    .enumerate()
                {
                    assert_eq!(got, want, "cached={cached} world={world} rank={rank}");
                }
            }
        }
    }

    #[test]
    fn an_uneven_split_answers_the_same_too() {
        // Machines are not alike, so the split will not be even. Rank 0
        // takes four of six layers here, rank 1 the other two.
        let c = cfg();
        let w = synth(&c);
        let prompt = vec![2u32, 9, 4];
        let n = 5usize;
        let want = reference(&c, &w, &prompt, n);

        for counts in [vec![4usize, 2], vec![1, 5], vec![3, 2, 1]] {
            let world = counts.len() as u32;
            for (rank, got) in sharded(&c, &w, &prompt, n, world, true, Some(counts.clone()))
                .into_iter()
                .enumerate()
            {
                assert_eq!(got, want, "split={counts:?} rank={rank}");
            }
        }
    }

    #[test]
    fn a_tied_head_works_on_the_last_block() {
        // Tied embeddings mean there is no `lm_head.weight`: the table is
        // the head. The block holding the head is the one that does not
        // embed, so it has to take the table anyway.
        let mut c = cfg();
        c.tie_word_embeddings = true;
        let mut w = synth(&c);
        w.remove("lm_head.weight");

        let prompt = vec![3u32, 1, 4];
        let n = 4usize;
        let want = reference(&c, &w, &prompt, n);
        for world in [2u32, 3] {
            for (rank, got) in sharded(&c, &w, &prompt, n, world, true, None)
                .into_iter()
                .enumerate()
            {
                assert_eq!(got, want, "world={world} rank={rank}");
            }
        }
    }
}

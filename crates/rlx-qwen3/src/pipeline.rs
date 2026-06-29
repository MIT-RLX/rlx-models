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

//! Pipeline-parallel Qwen3 stage (L3).
//!
//! Builds a prefill graph for a *contiguous range of layers* and wires it
//! to [`rlx_distributed`]'s pipeline coordinator:
//!
//!   - **First** block (highest rank): `input_ids` → token-embed →
//!     layers `[start, end)` → `hidden` output.
//!   - **Middle** block: `hidden_states` input → layers → `hidden` output.
//!   - **Last** block (rank 0): `hidden_states` input → layers →
//!     gather-last-token → final norm → LM head → `logits` output.
//!   - **Single** (world == 1): the whole model.
//!
//! The block graph reuses the exact same per-layer stage
//! ([`qwen3_prefill_layer_fused`]) the single-node prefill graph uses, so a
//! block of all layers is identical to the monolithic model. The only new
//! piece is a tiny plugin stage that adopts a declared `hidden_states`
//! input as the active tensor (instead of embedding token ids).
//!
//! NOTE: the partition / weight-filter logic here is unit-tested, but the
//! assembled block *graph's numerics* require validation on a machine with
//! real Qwen3 weights (run a 1-rank `Single` block and compare logits to
//! the monolithic [`crate::builder::build_qwen3_graph_sized_last_logits`]).

use crate::config::Qwen3Config;
use crate::flow::{qwen3_lm_head_stage, rope_tables, validate_cfg};
use anyhow::{Result, anyhow, bail};
use rlx_core::flow_bridge::{WeightLoaderSource, compile_options_from_profile};
use rlx_core::flow_util::graph_from_built;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use rlx_distributed::{
    BlockInput, BlockOutput, BlockRole, BlockRunner, block_role, pipeline_layer_range,
};
use rlx_flow::blocks::{Qwen3DecoderSpec, RopeTablesStage, qwen3_prefill_layer_fused};
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow, plugin_named};
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::ops::Range;

/// What a pipeline block must build: which layers, whether it embeds the
/// input tokens, and whether it produces logits.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    pub layers: Range<usize>,
    /// First/Single: take `input_ids` and embed. Else: take `hidden_states`.
    pub embed_input: bool,
    /// Last/Single: gather last token, final-norm, LM head → logits.
    pub produce_logits: bool,
}

impl BlockSpec {
    pub fn for_role(role: BlockRole, layers: Range<usize>) -> Self {
        let (embed_input, produce_logits) = match role {
            BlockRole::Single => (true, true),
            BlockRole::First => (true, false),
            BlockRole::Middle => (false, false),
            BlockRole::Last => (false, true),
        };
        Self {
            layers,
            embed_input,
            produce_logits,
        }
    }
}

/// Does weight `name` belong to a block with `spec`?
///
/// Per-layer weights (`model.layers.{i}.*`) belong iff `i` is in range.
/// Embedding belongs to the embedding block, and also to a logits block
/// when embeddings are tied (the LM head transposes the embed matrix).
/// `model.norm.weight` and `lm_head.weight` belong to the logits block.
pub fn block_weight_filter(name: &str, cfg: &Qwen3Config, spec: &BlockSpec) -> bool {
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let idx_str = rest.split('.').next().unwrap_or("");
        return match idx_str.parse::<usize>() {
            Ok(i) => spec.layers.contains(&i),
            Err(_) => false,
        };
    }
    match name {
        "model.embed_tokens.weight" => {
            spec.embed_input || (spec.produce_logits && cfg.tie_word_embeddings)
        }
        "model.norm.weight" => spec.produce_logits,
        "lm_head.weight" => spec.produce_logits && !cfg.tie_word_embeddings,
        _ => false,
    }
}

/// Build the [`BuiltModel`] for one pipeline block.
pub fn build_qwen3_block_built(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    spec: &BlockSpec,
) -> Result<BuiltModel> {
    validate_cfg(cfg)?;

    let f = DType::F32;
    let h = cfg.hidden_size;
    let dh = cfg.head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let (cos_data, sin_data) = rope_tables(cfg);

    let decoder_spec = Qwen3DecoderSpec {
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

    let mut flow = ModelFlow::new("qwen3_block").with_profile(CompileProfile::qwen3_prefill());

    // Declare the block's input.
    flow = if spec.embed_input {
        flow.input("input_ids", Shape::new(&[batch, seq], DType::I32))
    } else {
        flow.input("hidden_states", hidden_shape.clone())
    };

    // Shared params the layers reference by name.
    flow = flow
        .rope_tables(RopeTablesStage::param(
            cfg.max_position_embeddings,
            dh / 2,
            cos_data,
            sin_data,
        ))
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", dh);

    // Establish the active hidden tensor: embed token ids, or adopt the
    // `hidden_states` input as-is.
    flow = if spec.embed_input {
        flow.token_embed()
    } else {
        flow.raw_stage(plugin_named("adopt_hidden_states", |emit, _active| {
            Ok(Some(emit.flow_input("hidden_states")?))
        }))
    };

    // This rank's layers, by their global index.
    for global_idx in spec.layers.clone() {
        flow = flow.raw_stage(qwen3_prefill_layer_fused(global_idx, decoder_spec.clone()));
    }

    let built = if spec.produce_logits {
        flow.gather_last_token_at(batch, seq)
            .final_norm(eps)
            .raw_stage(qwen3_lm_head_stage(cfg))
            .output("logits")
            .build(&mut WeightLoaderSource(weights))?
    } else {
        flow.output("hidden")
            .build(&mut WeightLoaderSource(weights))?
    };
    Ok(built)
}

/// Build the block as a compile-ready `(Graph, params)`.
pub fn build_qwen3_block_graph(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    spec: &BlockSpec,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    graph_from_built(build_qwen3_block_built(cfg, weights, batch, seq, spec)?)
}

/// One rank's Qwen3 layer block — implements [`BlockRunner`] for
/// pipeline-parallel inference.
pub struct Qwen3PipelineStage {
    cfg: Qwen3Config,
    device: Device,
    role: BlockRole,
    spec: BlockSpec,
    batch: usize,
    /// Weights for this block only (filtered by [`block_weight_filter`]).
    weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Qwen3PipelineStage {
    /// Build this rank's stage by filtering `all_weights` down to the
    /// block it owns. In production each rank loads only its shard from
    /// disk; passing the full map and filtering is the simple path.
    pub fn new(
        cfg: Qwen3Config,
        device: Device,
        rank: u32,
        world: u32,
        mut all_weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Self {
        let layers = pipeline_layer_range(cfg.num_hidden_layers, rank, world);
        let role = block_role(rank, world);
        let spec = BlockSpec::for_role(role, layers);
        all_weights.retain(|k, _| block_weight_filter(k, &cfg, &spec));
        Self {
            cfg,
            device,
            role,
            spec,
            batch: 1,
            weights: all_weights,
        }
    }

    pub fn role(&self) -> BlockRole {
        self.role
    }
    pub fn layers(&self) -> Range<usize> {
        self.spec.layers.clone()
    }
    /// Number of weight tensors held after filtering (cheap shard check).
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    fn run_block(&self, seq: usize, input: BlockInput<'_>) -> Result<Vec<f32>> {
        let mut wm = WeightMap::from_tensors(self.weights.clone());
        let (graph, params) =
            build_qwen3_block_graph(&self.cfg, &mut wm, self.batch, seq, &self.spec)?;
        let opts = compile_options_from_profile(
            &CompileProfile::qwen3_prefill(),
            self.device,
            KernelDispatchConfig::default(),
        );
        let mut compiled = Session::new(self.device).compile_with(graph, &opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        let outputs = match input {
            BlockInput::Tokens(ids) => {
                let ids_f32: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
                compiled.run(&[("input_ids", ids_f32.as_slice())])
            }
            BlockInput::Hidden(hidden) => compiled.run(&[("hidden_states", hidden)]),
        };
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("qwen3 block produced no output"))
    }
}

impl BlockRunner for Qwen3PipelineStage {
    fn role(&self) -> BlockRole {
        self.role
    }

    fn run(&mut self, input: BlockInput<'_>) -> Result<BlockOutput> {
        let seq = match &input {
            BlockInput::Tokens(ids) => ids.len(),
            BlockInput::Hidden(hidden) => {
                let per_pos = self.batch * self.cfg.hidden_size;
                if per_pos == 0 || !hidden.len().is_multiple_of(per_pos) {
                    bail!(
                        "hidden length {} not a multiple of batch*hidden {}",
                        hidden.len(),
                        per_pos
                    );
                }
                hidden.len() / per_pos
            }
        };
        let out = self.run_block(seq, input)?;
        Ok(if self.spec.produce_logits {
            BlockOutput::Logits(out)
        } else {
            BlockOutput::Hidden(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(layers: usize, tied: bool) -> Qwen3Config {
        Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: tied,
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
    fn spec_for_roles() {
        let s = BlockSpec::for_role(BlockRole::First, 0..4);
        assert!(s.embed_input && !s.produce_logits);
        let s = BlockSpec::for_role(BlockRole::Last, 12..16);
        assert!(!s.embed_input && s.produce_logits);
        let s = BlockSpec::for_role(BlockRole::Middle, 4..8);
        assert!(!s.embed_input && !s.produce_logits);
        let s = BlockSpec::for_role(BlockRole::Single, 0..16);
        assert!(s.embed_input && s.produce_logits);
    }

    #[test]
    fn first_block_takes_embed_and_its_layers() {
        let c = cfg(16, false);
        let spec = BlockSpec::for_role(BlockRole::First, 0..4);
        assert!(block_weight_filter("model.embed_tokens.weight", &c, &spec));
        assert!(block_weight_filter(
            "model.layers.0.self_attn.q_proj.weight",
            &c,
            &spec
        ));
        assert!(block_weight_filter(
            "model.layers.3.mlp.gate_proj.weight",
            &c,
            &spec
        ));
        assert!(!block_weight_filter(
            "model.layers.4.mlp.gate_proj.weight",
            &c,
            &spec
        ));
        assert!(!block_weight_filter("model.norm.weight", &c, &spec));
        assert!(!block_weight_filter("lm_head.weight", &c, &spec));
    }

    #[test]
    fn last_block_takes_norm_head_and_its_layers() {
        let c = cfg(16, false);
        let spec = BlockSpec::for_role(BlockRole::Last, 12..16);
        assert!(block_weight_filter(
            "model.layers.15.self_attn.o_proj.weight",
            &c,
            &spec
        ));
        assert!(!block_weight_filter(
            "model.layers.11.self_attn.o_proj.weight",
            &c,
            &spec
        ));
        assert!(block_weight_filter("model.norm.weight", &c, &spec));
        assert!(block_weight_filter("lm_head.weight", &c, &spec));
        // Untied: the last block does NOT need the embedding matrix.
        assert!(!block_weight_filter("model.embed_tokens.weight", &c, &spec));
    }

    #[test]
    fn tied_embeddings_pull_embed_into_last_block() {
        let c = cfg(16, true);
        let spec = BlockSpec::for_role(BlockRole::Last, 12..16);
        // Tied: the LM head transposes embed_tokens, so the last block needs it.
        assert!(block_weight_filter("model.embed_tokens.weight", &c, &spec));
        // And there is no separate lm_head.weight to load.
        assert!(!block_weight_filter("lm_head.weight", &c, &spec));
    }

    #[test]
    fn middle_block_only_its_layers() {
        let c = cfg(16, false);
        let spec = BlockSpec::for_role(BlockRole::Middle, 4..8);
        assert!(block_weight_filter(
            "model.layers.4.input_layernorm.weight",
            &c,
            &spec
        ));
        assert!(block_weight_filter(
            "model.layers.7.mlp.down_proj.weight",
            &c,
            &spec
        ));
        assert!(!block_weight_filter(
            "model.layers.8.mlp.down_proj.weight",
            &c,
            &spec
        ));
        assert!(!block_weight_filter("model.embed_tokens.weight", &c, &spec));
        assert!(!block_weight_filter("model.norm.weight", &c, &spec));
    }

    // ---- numerical equivalence: pipeline split == monolithic --------

    fn fill_det(name: &str, n: usize) -> Vec<f32> {
        // FNV-ish seed per tensor, then a cheap per-element hash → small
        // deterministic non-zero values (no rng dependency).
        let seed = name
            .bytes()
            .fold(2166136261u32, |a, b| (a ^ b as u32).wrapping_mul(16777619));
        (0..n)
            .map(|i| {
                let x = seed.wrapping_add((i as u32).wrapping_mul(2654435761));
                ((x % 2000) as f32 / 1000.0 - 1.0) * 0.1 // in [-0.1, 0.1)
            })
            .collect()
    }

    fn synthetic_weights_all(c: &Qwen3Config) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        fn put(t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, key: String, shape: Vec<usize>) {
            let n: usize = shape.iter().product();
            let data = fill_det(&key, n);
            t.insert(key, (data, shape));
        }
        let h = c.hidden_size;
        let q_dim = c.num_attention_heads * c.head_dim;
        let kv_dim = c.num_key_value_heads * c.head_dim;
        let int_dim = c.intermediate_size;
        let dh = c.head_dim;
        let mut t = HashMap::new();
        put(
            &mut t,
            "model.embed_tokens.weight".into(),
            vec![c.vocab_size, h],
        );
        for i in 0..c.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            put(&mut t, format!("{lp}.input_layernorm.weight"), vec![h]);
            put(
                &mut t,
                format!("{lp}.post_attention_layernorm.weight"),
                vec![h],
            );
            put(
                &mut t,
                format!("{lp}.self_attn.q_proj.weight"),
                vec![q_dim, h],
            );
            put(
                &mut t,
                format!("{lp}.self_attn.k_proj.weight"),
                vec![kv_dim, h],
            );
            put(
                &mut t,
                format!("{lp}.self_attn.v_proj.weight"),
                vec![kv_dim, h],
            );
            put(
                &mut t,
                format!("{lp}.self_attn.o_proj.weight"),
                vec![h, q_dim],
            );
            put(&mut t, format!("{lp}.self_attn.q_norm.weight"), vec![dh]);
            put(&mut t, format!("{lp}.self_attn.k_norm.weight"), vec![dh]);
            put(
                &mut t,
                format!("{lp}.mlp.gate_proj.weight"),
                vec![int_dim, h],
            );
            put(&mut t, format!("{lp}.mlp.up_proj.weight"), vec![int_dim, h]);
            put(
                &mut t,
                format!("{lp}.mlp.down_proj.weight"),
                vec![h, int_dim],
            );
        }
        put(&mut t, "model.norm.weight".into(), vec![h]);
        put(&mut t, "lm_head.weight".into(), vec![c.vocab_size, h]);
        t
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Run the pipeline blocks for `world` ranks in-process — First
    /// (highest rank) through Last (rank 0) — feeding hidden states
    /// directly (no transport; the relay itself is tested in
    /// rlx-distributed). Returns the last block's logits.
    fn run_pipeline_inproc(
        c: &Qwen3Config,
        weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        prompt: &[u32],
        world: u32,
    ) -> Vec<f32> {
        let mut hidden: Option<Vec<f32>> = None;
        let mut logits: Option<Vec<f32>> = None;
        for rank in (0..world).rev() {
            let mut stage =
                Qwen3PipelineStage::new(c.clone(), Device::Cpu, rank, world, weights.clone());
            let input = match block_role(rank, world) {
                BlockRole::First | BlockRole::Single => BlockInput::Tokens(prompt),
                _ => BlockInput::Hidden(hidden.as_ref().expect("prior block hidden")),
            };
            match stage.run(input).unwrap() {
                BlockOutput::Hidden(hh) => hidden = Some(hh),
                BlockOutput::Logits(ll) => logits = Some(ll),
            }
        }
        logits.expect("last block produced logits")
    }

    /// Build the monolithic last-token logits for `prompt` (greedy
    /// reference).
    fn monolithic_logits(
        c: &Qwen3Config,
        weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        prompt: &[u32],
    ) -> Vec<f32> {
        let opts = compile_options_from_profile(
            &CompileProfile::qwen3_prefill(),
            Device::Cpu,
            KernelDispatchConfig::default(),
        );
        let mut wm = WeightMap::from_tensors(weights.clone());
        let (graph, params) =
            crate::builder::build_qwen3_graph_sized_last_logits(c, &mut wm, 1, prompt.len(), false)
                .unwrap();
        let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
        for (n, d) in &params {
            compiled.set_param(n, d);
        }
        let ids: Vec<f32> = prompt.iter().map(|&t| t as f32).collect();
        compiled
            .run(&[("input_ids", ids.as_slice())])
            .into_iter()
            .next()
            .unwrap()
    }

    /// End-to-end: real `Qwen3PipelineStage`s driven by the real
    /// `PipelineCoordinator` over a loopback TCP mesh (one thread per
    /// rank). Every rank must return the same broadcast token, equal to
    /// the monolithic greedy choice. Combines L0 (TCP) + L3 (graph) + L4
    /// (relay).
    #[test]
    fn pipeline_through_coordinator_over_tcp() {
        use rlx_distributed::{NetTransport, PipelineCoordinator, ProcessGroup};
        use std::net::{Ipv4Addr, SocketAddr, TcpListener};
        use std::sync::Arc;
        use std::thread;

        let c = cfg(4, false);
        let weights = synthetic_weights_all(&c);
        let prompt = vec![1u32, 5, 3, 2];
        let ref_tok = argmax(&monolithic_logits(&c, &weights, &prompt));

        for world in [2u32, 3, 4] {
            let listeners: Vec<TcpListener> = (0..world)
                .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
                .collect();
            let addrs: Vec<SocketAddr> =
                listeners.iter().map(|l| l.local_addr().unwrap()).collect();

            let c = Arc::new(c.clone());
            let weights = Arc::new(weights.clone());
            let prompt = Arc::new(prompt.clone());

            let handles: Vec<_> = listeners
                .into_iter()
                .enumerate()
                .map(|(rank, listener)| {
                    let (addrs, c, weights, prompt) =
                        (addrs.clone(), c.clone(), weights.clone(), prompt.clone());
                    thread::spawn(move || {
                        let t = NetTransport::from_listener(
                            rank as u32,
                            world,
                            listener,
                            addrs,
                            1 << 20,
                        )
                        .unwrap();
                        let mut stage = Qwen3PipelineStage::new(
                            (*c).clone(),
                            Device::Cpu,
                            rank as u32,
                            world,
                            (*weights).clone(),
                        );
                        let coord = PipelineCoordinator::new(ProcessGroup::new(Arc::new(t)));
                        let tok = coord
                            .forward_step(&mut stage, &prompt, |l| argmax(l) as u32)
                            .unwrap();
                        coord.barrier().unwrap();
                        tok
                    })
                })
                .collect();

            for (rank, h) in handles.into_iter().enumerate() {
                let tok = h.join().unwrap();
                assert_eq!(tok as usize, ref_tok, "world={world} rank={rank}");
            }
        }
    }

    /// Multi-token greedy generation through the real TCP pipeline must
    /// match the monolithic recompute-greedy sequence, token for token.
    #[test]
    fn pipeline_generate_matches_monolithic_greedy_sequence() {
        use rlx_distributed::{NetTransport, PipelineCoordinator, ProcessGroup};
        use std::net::{Ipv4Addr, SocketAddr, TcpListener};
        use std::sync::Arc;
        use std::thread;

        let c = cfg(4, false);
        let weights = synthetic_weights_all(&c);
        let prompt = vec![1u32, 5, 3, 2];
        let n_gen = 5usize;

        // Reference: greedy via the monolithic graph, recompute per step.
        let reference: Vec<u32> = {
            let mut toks = prompt.clone();
            let mut out = Vec::new();
            for _ in 0..n_gen {
                let tok = argmax(&monolithic_logits(&c, &weights, &toks)) as u32;
                toks.push(tok);
                out.push(tok);
            }
            out
        };

        let world = 3u32;
        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();

        let c = Arc::new(c);
        let weights = Arc::new(weights);
        let prompt = Arc::new(prompt);

        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let (addrs, c, weights, prompt) =
                    (addrs.clone(), c.clone(), weights.clone(), prompt.clone());
                thread::spawn(move || {
                    let t =
                        NetTransport::from_listener(rank as u32, world, listener, addrs, 1 << 20)
                            .unwrap();
                    let mut stage = Qwen3PipelineStage::new(
                        (*c).clone(),
                        Device::Cpu,
                        rank as u32,
                        world,
                        (*weights).clone(),
                    );
                    let coord = PipelineCoordinator::new(ProcessGroup::new(Arc::new(t)));
                    let mut tokens = (*prompt).clone();
                    coord
                        .generate(
                            &mut stage,
                            &mut tokens,
                            n_gen,
                            |l| argmax(l) as u32,
                            |_| false,
                        )
                        .unwrap()
                })
            })
            .collect();

        for (rank, h) in handles.into_iter().enumerate() {
            let produced = h.join().unwrap();
            assert_eq!(produced, reference, "rank {rank}");
        }
    }

    #[test]
    fn pipeline_split_matches_monolithic_logits() {
        let c = cfg(4, false); // 4 layers, untied
        let weights = synthetic_weights_all(&c);
        let prompt = [1u32, 5, 3, 2];
        let seq = prompt.len();
        let device = Device::Cpu;
        let opts = compile_options_from_profile(
            &CompileProfile::qwen3_prefill(),
            device,
            KernelDispatchConfig::default(),
        );

        // Reference: monolithic last-token logits.
        let mut wm = WeightMap::from_tensors(weights.clone());
        let (graph, params) =
            crate::builder::build_qwen3_graph_sized_last_logits(&c, &mut wm, 1, seq, false)
                .unwrap();
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        for (n, d) in &params {
            compiled.set_param(n, d);
        }
        let ids: Vec<f32> = prompt.iter().map(|&t| t as f32).collect();
        let reference = compiled
            .run(&[("input_ids", ids.as_slice())])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(reference.len(), c.vocab_size);
        let ref_arg = argmax(&reference);

        // world == 1 checks Single == monolithic; world 2/4 exercise the
        // hidden-state handoff across block boundaries.
        for world in [1u32, 2, 4] {
            let pipe = run_pipeline_inproc(&c, &weights, &prompt, world);
            assert_eq!(pipe.len(), c.vocab_size, "world={world}");
            let max_diff = reference
                .iter()
                .zip(&pipe)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                argmax(&pipe),
                ref_arg,
                "world={world}: argmax differs (max logit diff {max_diff})"
            );
            assert!(
                max_diff < 1e-2,
                "world={world}: max logit diff {max_diff} exceeds tolerance"
            );
        }
    }
}

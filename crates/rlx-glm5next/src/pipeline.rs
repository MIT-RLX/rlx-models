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

//! Pipeline-parallel `glm5next` stage.
//!
//! Each rank owns a contiguous slice of the 45 decoder layers and implements
//! [`BlockRunner`], so [`rlx_distributed::PipelineCoordinator`] can drive a
//! forward pass across machines. The graph a rank builds is
//! [`build_glm5next_block_with_source`] over its own [`BlockSpec`]; a one-rank
//! world builds the whole model, which is what keeps the split honest.
//!
//! ## What crosses the wire
//!
//! The mHC stream state `[1, seq, hc_mult, hidden]` — **not** a hidden state.
//! The four residual streams run the model's full depth and are averaged only
//! by `hc_head` at the very end, so a cut has to carry all four. See
//! [`BlockSpec`] for why collapsing at the boundary is not equivalent, and note
//! the consequence for planning: a `glm5next` pipeline moves `hc_mult`x (here
//! 4x) the bytes per token of a plain transformer the same width.
//!
//! ## Rank order
//!
//! [`rlx_distributed`] assigns blocks in REVERSE: rank 0 is the `Last` block
//! (it holds the LM head) and rank `world-1` is the `First` (it embeds). Use
//! [`block_role`] / [`pipeline_layer_range`] rather than deriving it.

use crate::config::Glm5NextConfig;
use crate::flow::{BlockSpec, STREAM_INPUT, block_weight_filter, build_glm5next_block_with_source};
use anyhow::{Result, anyhow, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_distributed::{
    BlockInput, BlockOutput, BlockRole, BlockRunner, block_role, pipeline_layer_range,
};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::ops::Range;

/// Compiled graphs kept per stage by default: one for decode's `seq = 1` and
/// one for the prompt length in flight.
pub const DEFAULT_GRAPH_CACHE: usize = 2;

/// The [`BlockSpec`] for one rank of a `world`-rank pipeline.
pub fn block_spec_for(cfg: &Glm5NextConfig, rank: u32, world: u32) -> BlockSpec {
    let role = block_role(rank, world);
    let layers = pipeline_layer_range(cfg.num_hidden_layers, rank, world);
    BlockSpec {
        layers,
        embed_input: matches!(role, BlockRole::First | BlockRole::Single),
        produce_logits: matches!(role, BlockRole::Last | BlockRole::Single),
    }
}

/// One rank's `glm5next` layer block.
pub struct Glm5NextPipelineStage {
    cfg: Glm5NextConfig,
    device: Device,
    role: BlockRole,
    spec: BlockSpec,
    /// Only this block's weights, per [`block_weight_filter`].
    weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
    /// Compiled graph per sequence length, bounded by `cache_cap`.
    ///
    /// Not an optimization so much as the difference between a usable stage and
    /// an unusable one: a graph is built for a FIXED `seq`, and rebuilding it
    /// per call meant re-emitting 45 layers, re-running the compiler, and
    /// re-uploading every weight in the shard on every forward pass. Decode
    /// calls this once per token at `seq = 1`, so all of that happened for each
    /// token generated.
    ///
    /// Keyed by `seq` because that is the only thing that varies; the weights
    /// and the layer range are fixed at construction.
    ///
    /// **Bounded, because each entry owns an uploaded copy of the shard.** A
    /// `CompiledGraph` holds its params in its own arena, so caching one per
    /// distinct prompt length is one full copy of this rank's weights per
    /// length — gigabytes each on a real shard, on a node the planner sized to
    /// hold exactly one. Serving varied prompts would climb until the node died,
    /// which is a worse failure than the recompile the cache exists to avoid.
    compiled: HashMap<usize, CompiledGraph>,
    /// Sequence lengths in least-recently-used order.
    lru: Vec<usize>,
    /// How many graphs to keep. Decode reuses one `seq` forever and a prompt
    /// contributes one more, so the useful default is small.
    cache_cap: usize,
    /// How many graphs have actually been built — the observable that makes
    /// the cache testable without timing anything.
    builds: usize,
}

impl Glm5NextPipelineStage {
    /// Build this rank's stage, keeping only the weights it owns.
    ///
    /// Passing the whole map and filtering is the simple path; a real
    /// deployment loads only its shard, and [`block_weight_filter`] is the same
    /// predicate either way.
    pub fn new(
        cfg: Glm5NextConfig,
        device: Device,
        rank: u32,
        world: u32,
        mut all_weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<Self> {
        if world == 0 {
            bail!("glm5next pipeline: world must be >= 1");
        }
        let role = block_role(rank, world);
        let spec = block_spec_for(&cfg, rank, world);
        all_weights.retain(|k, _| block_weight_filter(k, &cfg, &spec));
        Ok(Self {
            cfg,
            device,
            role,
            spec,
            weights: all_weights,
            compiled: HashMap::new(),
            lru: Vec::new(),
            cache_cap: DEFAULT_GRAPH_CACHE,
            builds: 0,
        })
    }

    pub fn role(&self) -> BlockRole {
        self.role
    }
    pub fn layers(&self) -> Range<usize> {
        self.spec.layers.clone()
    }
    pub fn spec(&self) -> &BlockSpec {
        &self.spec
    }
    /// Weight tensors retained after filtering — a cheap shard check.
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    /// Graphs built so far. Exceeds [`Self::graphs_cached`] once eviction has
    /// started, which is the signal that `cache_cap` is too small for the
    /// traffic.
    pub fn graphs_built(&self) -> usize {
        self.builds
    }

    /// Graphs currently held, each owning a copy of this rank's weights.
    pub fn graphs_cached(&self) -> usize {
        self.compiled.len()
    }

    /// Keep at most `n` compiled graphs.
    ///
    /// Raise it for a server fielding a few recurring prompt lengths; every
    /// extra slot costs another copy of the shard's weights, so this is a
    /// memory decision, not a speed one. `n == 0` is treated as 1: a stage has
    /// to hold the graph it is running.
    pub fn with_graph_cache(mut self, n: usize) -> Self {
        self.cache_cap = n.max(1);
        self.evict_to_cap();
        self
    }

    fn evict_to_cap(&mut self) {
        while self.compiled.len() > self.cache_cap {
            let Some(oldest) = (!self.lru.is_empty()).then(|| self.lru.remove(0)) else {
                break;
            };
            self.compiled.remove(&oldest);
        }
    }

    /// Elements of stream state per position: `hc_mult * hidden`.
    fn per_position(&self) -> usize {
        self.cfg.hc_mult * self.cfg.hidden_size
    }

    /// The compiled graph for this `seq`, building it on first use.
    fn graph_for(&mut self, seq: usize) -> Result<&mut CompiledGraph> {
        if !self.compiled.contains_key(&seq) {
            // Cloning the shard is the expensive part, so it happens here — on
            // a miss — rather than on every forward pass.
            let mut wm = WeightMap::from_tensors(self.weights.clone());
            let built = build_glm5next_block_with_source(
                &self.cfg,
                &mut WeightMapSource(&mut wm),
                seq,
                &self.spec,
            )?;
            let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
            let mut compiled = Session::new(self.device).compile(graph);
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            self.compiled.insert(seq, compiled);
            self.builds += 1;
        }
        // Touch, then bound. Touching first means the length being run is never
        // the one evicted, even at `cache_cap == 1`.
        self.lru.retain(|&s| s != seq);
        self.lru.push(seq);
        self.evict_to_cap();
        Ok(self
            .compiled
            .get_mut(&seq)
            .expect("just inserted or already resident"))
    }

    fn run_block(&mut self, seq: usize, input: BlockInput<'_>) -> Result<Vec<f32>> {
        let compiled = self.graph_for(seq)?;
        let outputs = match input {
            BlockInput::Tokens(ids) => {
                let ids_f32: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
                compiled.run(&[("input_ids", ids_f32.as_slice())])
            }
            BlockInput::Hidden(streams) => compiled.run(&[(STREAM_INPUT, streams)]),
        };
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("glm5next block produced no output"))
    }
}

impl BlockRunner for Glm5NextPipelineStage {
    fn role(&self) -> BlockRole {
        self.role
    }

    fn run(&mut self, input: BlockInput<'_>) -> Result<BlockOutput> {
        let seq = match &input {
            BlockInput::Tokens(ids) => ids.len(),
            BlockInput::Hidden(streams) => {
                let per = self.per_position();
                if per == 0 || !streams.len().is_multiple_of(per) {
                    bail!(
                        "glm5next block: {} incoming elements is not a multiple of \
                         hc_mult*hidden = {per}; a cut carries all {} residual \
                         streams, not a collapsed hidden state",
                        streams.len(),
                        self.cfg.hc_mult
                    );
                }
                streams.len() / per
            }
        };
        if seq == 0 {
            bail!("glm5next block: empty input");
        }
        let out = self.run_block(seq, input)?;
        Ok(if self.spec.produce_logits {
            BlockOutput::Logits(out)
        } else {
            BlockOutput::Hidden(out)
        })
    }
}

// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! One rank's block of a model split across machines, for any family.
//!
//! The bookkeeping of being a pipeline rank has nothing to do with which
//! model it is. A rank holds a contiguous range of layers, seeds a KV cache
//! from the prompt on its first call, decodes one token per call after that,
//! and hands its hidden state to the next rank. Only two things are
//! family-specific: the graph for a block of layers, and a handful of shapes.
//!
//! So that is the seam. [`PipelineFamily`] is what a family provides;
//! [`FamilyPipelineStage`] is everything else, written once.
//!
//! ```text
//!   rank 2 (First)        rank 1 (Middle)        rank 0 (Last)
//!   embed + layers 0..9 ─▶ layers 9..18      ─▶ layers 18..28 + LM head
//! ```
//!
//! Ranks count backwards: rank 0 holds the *last* block, because it is the
//! one with the logits and therefore the only one that can answer.

use crate::autoregressive::{KvCacheState, kv_from_prefill_outputs, run_bucketed_kv_decode};
use crate::flow_util::graph_from_built;
use crate::weight_loader::WeightLoader;
use crate::weight_map::WeightMap;
use anyhow::{Result, bail};
use rlx_distributed::{
    BlockInput, BlockOutput, BlockRole, BlockRunner, block_role, pipeline_layer_range,
};
use rlx_flow::BuiltModel;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput};
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;
use std::ops::Range;

/// Weights as the stages take them: name → (values, shape).
pub type Tensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// What a model family must provide for its layers to be split across
/// machines.
///
/// Both graph methods are given the **global** layer range, because that is
/// what the checkpoint's weight names are indexed by. A family whose graph
/// builder can be pointed at a slice of the model uses the range directly;
/// one whose builder only knows how to count from zero remaps its weights in
/// [`Self::block_weights`] and ignores `layers.start`.
pub trait PipelineFamily: Send {
    /// Prefill graph over `layers`, exporting per-layer K/V so the cache can
    /// be seeded.
    ///
    /// `embed` means this block turns token ids into hidden states (the
    /// first block); otherwise it adopts a hidden state from the rank above.
    /// `logits` means it finishes with the final norm and the LM head (the
    /// last block); otherwise it outputs raw hidden.
    fn seed_graph(
        &self,
        weights: &mut dyn WeightLoader,
        batch: usize,
        seq: usize,
        layers: Range<usize>,
        embed: bool,
        logits: bool,
    ) -> Result<BuiltModel>;

    /// Single-token decode graph over the same layers, reading and writing
    /// the per-layer K/V for `past` cached positions.
    #[allow(clippy::too_many_arguments)]
    fn decode_graph(
        &self,
        weights: &mut dyn WeightLoader,
        batch: usize,
        past: usize,
        layers: Range<usize>,
        embed: bool,
        logits: bool,
        custom_mask: bool,
    ) -> Result<BuiltModel>;

    fn num_hidden_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    /// Width of one layer's K (and V) projection — how the exported cache is
    /// sliced back apart.
    fn kv_proj_dim(&self) -> usize;
    fn tie_word_embeddings(&self) -> bool;

    /// The per-step inputs this family's decode graph needs beyond the new
    /// token and the attention mask — in practice, however it is told the
    /// position.
    ///
    /// Families genuinely differ here, and the difference is silent: qwen3's
    /// decode graph takes `rope_cos`/`rope_sin` sliced on the host, while
    /// llama's takes a single `position` and derives them in the graph.
    /// Feeding one convention to the other does not fail — the unknown names
    /// are ignored and the missing one defaults — so every decode step runs
    /// at position zero and the model answers plausible nonsense from the
    /// second token on. Prefill is unaffected, because there the rope tables
    /// are a parameter rather than an input, which is what makes this look
    /// like a decode bug rather than a wiring one.
    fn decode_inputs(&self, past: usize) -> Vec<(&'static str, Vec<f32>)>;

    fn compile_options(&self, device: Device, decode: bool) -> CompileOptions;

    /// Reduce the whole model's weights to the ones this block needs.
    ///
    /// The default keeps the layers in `layers` under the names they already
    /// have, plus the embedding table and head where this block needs them.
    /// A family whose graph builder counts layers from zero overrides this
    /// to renumber them.
    ///
    /// It defers to [`crate::block_weights::block_needs`], which is also
    /// what [`crate::block_weights::load`] consults to decide what to read
    /// off disk in the first place. Keeping one definition is the point: a
    /// block that loads a weight its graph does not want merely wastes
    /// memory, but one that skips a weight its graph *does* want fails at
    /// graph-building on that rank alone, with every shape agreeing.
    fn block_weights(
        &self,
        all: Tensors,
        layers: &Range<usize>,
        embed: bool,
        logits: bool,
    ) -> Tensors {
        crate::block_weights::filter(all, layers, embed, logits, self.tie_word_embeddings())
    }
}

/// One rank's block, for any family that implements [`PipelineFamily`].
///
/// Stateful by design: whether it holds a KV cache is how it tells seeding
/// from decoding, which also makes it single-sequence. A server reuses it
/// across requests and calls [`Self::reset`] to start a new one.
pub struct FamilyPipelineStage<F: PipelineFamily> {
    family: F,
    device: Device,
    role: BlockRole,
    /// Global layer range this rank owns.
    layers: Range<usize>,
    embed_input: bool,
    produce_logits: bool,
    batch: usize,
    weights: Tensors,
    cache: Option<KvCacheState>,
    decode_cache: Option<BucketedCompileCache>,
}

impl<F: PipelineFamily> FamilyPipelineStage<F> {
    /// This rank's block, with the layers split evenly.
    pub fn new(family: F, device: Device, rank: u32, world: u32, all_weights: Tensors) -> Self {
        let layers = pipeline_layer_range(family.num_hidden_layers(), rank, world);
        Self::with_layer_range(family, device, rank, world, all_weights, layers)
    }

    /// This rank's block, with its layers given rather than split evenly.
    ///
    /// An even split is only right when the machines are alike. Every rank
    /// must be given the same overall division — two ranks disagreeing runs
    /// some layers twice and others never, with no shape error to catch it.
    pub fn with_layer_range(
        family: F,
        device: Device,
        rank: u32,
        world: u32,
        all_weights: Tensors,
        layers: Range<usize>,
    ) -> Self {
        let role = block_role(rank, world);
        let (embed_input, produce_logits) = crate::block_weights::block_ends(rank, world);
        let weights = family.block_weights(all_weights, &layers, embed_input, produce_logits);
        Self {
            family,
            device,
            role,
            layers,
            embed_input,
            produce_logits,
            batch: 1,
            weights,
            cache: None,
            decode_cache: None,
        }
    }

    /// Compile one decode graph per power-of-two `past` bucket, up to
    /// `max_past`. Strongly recommended: without it every token recompiles.
    pub fn with_decode_cache(mut self, max_past: usize) -> Self {
        self.decode_cache = Some(BucketedCompileCache::power_of_two_ladder(
            self.device,
            1,
            max_past.max(1) as u64,
        ));
        self
    }

    /// As [`Self::with_decode_cache`], but `1` compiles a single graph
    /// spanning the whole context instead of a ladder.
    pub fn with_decode_buckets(self, buckets: usize, max_past: usize) -> Self {
        let max = max_past.max(1) as u64;
        if buckets <= 1 {
            let mut this = self;
            this.decode_cache = Some(crate::flow_util::single_bucket_decode_cache(
                this.device,
                max as usize,
            ));
            return this;
        }
        self.with_decode_cache(max_past)
    }

    pub fn role(&self) -> BlockRole {
        self.role
    }

    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Forget the KV cache, so the next call seeds a new sequence.
    pub fn reset(&mut self) {
        self.cache = None;
    }

    /// First call: prefill the input and seed the cache.
    fn seed(&mut self, input: BlockInput<'_>) -> Result<Vec<f32>> {
        let seq = match &input {
            BlockInput::Tokens(t) => t.len(),
            BlockInput::Hidden(hh) => hh.len() / (self.batch * self.family.hidden_size()),
        };
        if seq == 0 {
            bail!("a prefill needs at least one position");
        }
        let mut wm = WeightMap::from_tensors(self.weights.clone());
        let built = self.family.seed_graph(
            &mut wm,
            self.batch,
            seq,
            self.layers.clone(),
            self.embed_input,
            self.produce_logits,
        )?;
        let (graph, params) = graph_from_built(built)?;
        let mut compiled = Session::new(self.device)
            .compile_with(graph, &self.family.compile_options(self.device, false));
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
            self.family.kv_proj_dim(),
            self.layers.len(),
        )?;
        self.cache = Some(kv);
        Ok(main)
    }

    /// The new token (first block) or its hidden state (the others).
    fn decode_input(&self, input: BlockInput<'_>, past: usize) -> Result<(&'static str, Vec<f32>)> {
        match input {
            BlockInput::Tokens(t) => {
                if t.len() <= past {
                    bail!("decode: token history len {} <= past {past}", t.len());
                }
                Ok(("input_ids", vec![t[past] as f32]))
            }
            BlockInput::Hidden(hh) => {
                let per = self.batch * self.family.hidden_size();
                if hh.len() != per {
                    bail!(
                        "decode: expected one hidden position ({per} f32), got {}",
                        hh.len()
                    );
                }
                Ok(("inputs_embeds", hh.to_vec()))
            }
        }
    }

    /// Steady-state decode, reusing a compiled graph per bucket.
    fn decode_bucketed(&mut self, input: BlockInput<'_>) -> Result<Vec<f32>> {
        let past = self.cache.as_ref().expect("seeded").past_len;
        let extra = self.family.decode_inputs(past);
        let kv = self.cache.as_ref().expect("seeded").clone();
        let opts = self.family.compile_options(self.device, true);
        let kv_dim = self.family.kv_proj_dim();
        let block_len = self.layers.len();
        let (main_name, main_buf) = self.decode_input(input, past)?;

        let cache_dec = self.decode_cache.as_mut().expect("bucketed");
        let upper = cache_dec
            .bucket_for(past as u64)
            .and_then(|idx| cache_dec.buckets().nth(idx).map(|r| (r.end - 1) as usize))
            .unwrap_or(past);
        let mask = bucket_decode_mask(past, upper);

        let mut fixed = vec![
            CacheRunInput {
                name: main_name,
                data: &main_buf,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: &mask,
                row_inner: None,
            },
        ];
        // Whatever this family needs in order to know the position.
        for (name, data) in &extra {
            fixed.push(CacheRunInput {
                name,
                data,
                row_inner: None,
            });
        }

        let weights = self.weights.clone();
        let layers = self.layers.clone();
        let family = &self.family;
        let (batch, embed, logits) = (self.batch, self.embed_input, self.produce_logits);
        let (main, layers_k, layers_v) = run_bucketed_kv_decode(
            cache_dec,
            past,
            &kv,
            kv_dim,
            block_len,
            &fixed,
            |upper| {
                let mut wm = WeightMap::from_tensors(weights.clone());
                let built = family
                    .decode_graph(
                        &mut wm,
                        batch,
                        upper as usize,
                        layers.clone(),
                        embed,
                        logits,
                        true,
                    )
                    .expect("block decode graph");
                graph_from_built(built).expect("block decode graph parts")
            },
            &opts,
        )?;

        let c = self.cache.as_mut().expect("seeded");
        c.past_len = past + 1;
        c.layers_k = layers_k;
        c.layers_v = layers_v;
        Ok(main)
    }

    /// Steady-state decode, recompiling every step. Correct but far slower;
    /// for isolating the compile cache when something looks wrong.
    fn decode_recompile(&mut self, input: BlockInput<'_>) -> Result<Vec<f32>> {
        let past = self.cache.as_ref().expect("seeded").past_len;
        let extra = self.family.decode_inputs(past);
        let kv = self.cache.as_ref().expect("seeded").clone();
        let (main_name, main_buf) = self.decode_input(input, past)?;

        let mut wm = WeightMap::from_tensors(self.weights.clone());
        let built = self.family.decode_graph(
            &mut wm,
            self.batch,
            past,
            self.layers.clone(),
            self.embed_input,
            self.produce_logits,
            false,
        )?;
        let (graph, params) = graph_from_built(built)?;
        let mut compiled = Session::new(self.device)
            .compile_with(graph, &self.family.compile_options(self.device, true));
        for (n, d) in &params {
            compiled.set_param(n, d);
        }

        let mut inputs: Vec<(&str, &[f32])> = vec![(main_name, main_buf.as_slice())];
        for (name, data) in &extra {
            inputs.push((name, data.as_slice()));
        }
        for i in 0..self.layers.len() {
            inputs.push((
                Box::leak(format!("past_k_{i}").into_boxed_str()),
                &kv.layers_k[i],
            ));
            inputs.push((
                Box::leak(format!("past_v_{i}").into_boxed_str()),
                &kv.layers_v[i],
            ));
        }
        let outputs = compiled.run(&inputs);
        let (main, next) = kv_from_prefill_outputs(
            outputs,
            self.batch,
            past + 1,
            self.family.kv_proj_dim(),
            self.layers.len(),
        )?;
        self.cache = Some(next);
        Ok(main)
    }
}

impl<F: PipelineFamily> BlockRunner for FamilyPipelineStage<F> {
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

    /// A family that builds nothing — enough to check the bookkeeping that
    /// does not depend on a model: which weights a block keeps, and which
    /// role each rank plays.
    struct Stub {
        layers: usize,
        tied: bool,
    }

    impl PipelineFamily for Stub {
        fn seed_graph(
            &self,
            _w: &mut dyn WeightLoader,
            _b: usize,
            _s: usize,
            _l: Range<usize>,
            _e: bool,
            _g: bool,
        ) -> Result<BuiltModel> {
            bail!("the stub builds no graphs")
        }
        fn decode_graph(
            &self,
            _w: &mut dyn WeightLoader,
            _b: usize,
            _p: usize,
            _l: Range<usize>,
            _e: bool,
            _g: bool,
            _m: bool,
        ) -> Result<BuiltModel> {
            bail!("the stub builds no graphs")
        }
        fn num_hidden_layers(&self) -> usize {
            self.layers
        }
        fn hidden_size(&self) -> usize {
            8
        }
        fn kv_proj_dim(&self) -> usize {
            4
        }
        fn tie_word_embeddings(&self) -> bool {
            self.tied
        }
        fn decode_inputs(&self, past: usize) -> Vec<(&'static str, Vec<f32>)> {
            vec![("position", vec![past as f32])]
        }
        fn compile_options(&self, _d: Device, _decode: bool) -> CompileOptions {
            CompileOptions::default()
        }
    }

    fn all_weights(layers: usize) -> Tensors {
        let mut t = Tensors::new();
        let mut put = |k: String| {
            t.insert(k, (vec![0.0], vec![1]));
        };
        put("model.embed_tokens.weight".into());
        put("model.norm.weight".into());
        put("lm_head.weight".into());
        for l in 0..layers {
            put(format!("model.layers.{l}.self_attn.q_proj.weight"));
            put(format!("model.layers.{l}.mlp.up_proj.weight"));
        }
        // Something no block should ever take.
        put("visual.patch_embed.proj.weight".into());
        t
    }

    fn kept(rank: u32, world: u32, layers: usize, tied: bool) -> Vec<String> {
        let stage = FamilyPipelineStage::new(
            Stub { layers, tied },
            Device::Cpu,
            rank,
            world,
            all_weights(layers),
        );
        let mut names: Vec<String> = stage.weights.keys().cloned().collect();
        names.sort();
        names
    }

    #[test]
    fn a_block_keeps_its_own_layers_under_their_own_names() {
        // The default filter does not renumber: the range is global because
        // that is what the checkpoint's names are indexed by.
        let names = kept(1, 2, 4, false);
        assert!(names.iter().any(|n| n.starts_with("model.layers.0.")));
        assert!(names.iter().any(|n| n.starts_with("model.layers.1.")));
        assert!(
            !names.iter().any(|n| n.starts_with("model.layers.2.")),
            "{names:?}"
        );
    }

    #[test]
    fn rank_zero_holds_the_last_layers_and_the_head() {
        let names = kept(0, 2, 4, false);
        assert!(names.iter().any(|n| n.starts_with("model.layers.2.")));
        assert!(names.iter().any(|n| n.starts_with("model.layers.3.")));
        assert!(names.contains(&"model.norm.weight".to_string()));
        assert!(names.contains(&"lm_head.weight".to_string()));
        assert!(
            !names.contains(&"model.embed_tokens.weight".to_string()),
            "an untied last block does not embed: {names:?}"
        );
    }

    #[test]
    fn a_tied_head_makes_the_last_block_keep_the_embedding_table() {
        // Tied weights mean `lm_head.weight` does not exist — the table is
        // the head, and a last block without it has nothing to project with.
        let names = kept(0, 2, 4, true);
        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(
            !names.contains(&"lm_head.weight".to_string()),
            "a tied model should not also take a separate head: {names:?}"
        );
    }

    #[test]
    fn the_first_block_embeds_and_takes_no_head() {
        let names = kept(1, 2, 4, false);
        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(!names.contains(&"model.norm.weight".to_string()));
        assert!(!names.contains(&"lm_head.weight".to_string()));
    }

    #[test]
    fn nothing_else_in_the_checkpoint_is_taken() {
        for rank in 0..2 {
            let names = kept(rank, 2, 4, false);
            assert!(
                !names.iter().any(|n| n.starts_with("visual.")),
                "rank {rank} took a vision tower: {names:?}"
            );
        }
    }

    #[test]
    fn one_machine_is_a_single_block_that_does_everything() {
        let stage = FamilyPipelineStage::new(
            Stub {
                layers: 4,
                tied: false,
            },
            Device::Cpu,
            0,
            1,
            all_weights(4),
        );
        assert_eq!(stage.role(), BlockRole::Single);
        assert_eq!(stage.layers(), 0..4);
        assert!(stage.embed_input && stage.produce_logits);
    }

    #[test]
    fn an_explicit_range_overrides_the_even_split() {
        let stage = FamilyPipelineStage::with_layer_range(
            Stub {
                layers: 28,
                tied: false,
            },
            Device::Cpu,
            0,
            2,
            all_weights(28),
            20..28,
        );
        assert_eq!(stage.layers(), 20..28);
        let names: Vec<String> = stage.weights.keys().cloned().collect();
        assert!(names.iter().any(|n| n.starts_with("model.layers.20.")));
        assert!(!names.iter().any(|n| n.starts_with("model.layers.19.")));
    }
}

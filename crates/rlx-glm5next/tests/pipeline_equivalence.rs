// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! A `glm5next` pipeline split must compute what the whole model computes.
//!
//! The claim under test is the one that makes pipeline parallelism usable at
//! all: cutting the layer stack across ranks changes *where* arithmetic happens,
//! never *what* it is. So an N-rank relay has to reproduce the single-graph
//! logits, and the interesting failures are all at the seams — which tensor
//! crosses a cut, which rank owns the embedding and which the head, and whether
//! the ranks are numbered the way the coordinator thinks.
//!
//! Running the relay for real needs a transport; the arithmetic can be checked
//! without one by driving the same [`BlockRunner`]s by hand in rank order, which
//! is what `relay` does.

mod common;

use common::{HC, HIDDEN, LAYERS, VOCAB, cfg, dev, tensor_map, weights};
use rlx_core::flow_util::compile_built;
use rlx_distributed::{BlockInput, BlockOutput, BlockRole, BlockRunner, block_role};
use rlx_glm5next::{
    BlockSpec, Glm5NextConfig, Glm5NextPipelineStage, block_spec_for, block_weight_filter,
    build_glm5next_text_flow,
};
use std::collections::HashMap;

/// Fewer than four layers would make the 4-rank case degenerate (empty stages),
/// so the equivalence check below would stop testing what it claims to.
const _: () = assert!(LAYERS >= 4);

/// Logits from the monolithic graph — the reference every split is compared to.
fn whole_model(c: &Glm5NextConfig, ids: &[f32]) -> Vec<f32> {
    let mut w = weights(c);
    let built = build_glm5next_text_flow(c, &mut w, ids.len(), true).expect("build whole model");
    let mut compiled = compile_built(built, dev()).expect("compile");
    compiled
        .run(&[("input_ids", ids)])
        .into_iter()
        .next()
        .expect("logits")
}

/// Every tensor of the tiny checkpoint, as the stages want them.
fn all_tensors(c: &Glm5NextConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    tensor_map(c)
}

/// Drive a `world`-rank pipeline by hand, in the coordinator's order.
///
/// `rlx_distributed` numbers blocks in REVERSE — rank `world-1` embeds, rank 0
/// runs the head — so the relay walks ranks downward.
fn relay(c: &Glm5NextConfig, world: u32, ids: &[u32]) -> Vec<f32> {
    let tensors = all_tensors(c);
    let mut stages: Vec<Glm5NextPipelineStage> = (0..world)
        .map(|r| {
            Glm5NextPipelineStage::new(c.clone(), dev(), r, world, tensors.clone())
                .expect("build stage")
        })
        .collect();

    let mut carry: Option<Vec<f32>> = None;
    let mut logits = None;
    for rank in (0..world).rev() {
        let stage = &mut stages[rank as usize];
        let input = match &carry {
            None => BlockInput::Tokens(ids),
            Some(h) => BlockInput::Hidden(h),
        };
        match stage.run(input).expect("run stage") {
            BlockOutput::Hidden(h) => carry = Some(h),
            BlockOutput::Logits(l) => logits = Some(l),
        }
    }
    logits.expect("the Last rank must produce logits")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Splitting the stack across 2, 3 and 4 ranks must not move the logits.
///
/// Not an approximation: each rank runs the same per-layer emitters on the same
/// weights in the same order, so the only thing a split can change is float
/// association across a graph boundary. The tolerance is for that, not for a
/// difference in the maths.
#[test]
fn a_split_pipeline_reproduces_the_whole_model() {
    let c = cfg(64);
    let ids: Vec<u32> = (0..6).map(|i| (i * 3 % VOCAB) as u32).collect();
    let ids_f: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
    let want = whole_model(&c, &ids_f);
    assert!(
        want.iter().all(|v| v.is_finite()),
        "reference logits are not finite"
    );

    for world in [1u32, 2, 3, 4] {
        let got = relay(&c, world, &ids);
        let d = max_abs_diff(&want, &got);
        assert!(
            d <= 2e-4,
            "{world}-rank pipeline diverges from the whole model by {d:.3e}"
        );
    }
}

/// A stage must compile its graph once, not once per token.
///
/// A block graph is built for a FIXED `seq`, so the obvious implementation
/// rebuilds it on every call — which for decode means re-emitting 45 layers,
/// re-running the compiler and re-uploading the whole shard for each token
/// generated. Nothing about the output reveals it: the answers are identical,
/// the stage is just unusable. Counting builds is the way to catch it, since a
/// timing assertion on a shared machine is not something to rely on.
#[test]
fn a_stage_compiles_once_per_sequence_length() {
    let c = cfg(64);
    let tensors = all_tensors(&c);
    let mut stage = Glm5NextPipelineStage::new(c.clone(), dev(), 1, 2, tensors).expect("stage");
    assert_eq!(
        stage.graphs_built(),
        0,
        "construction should compile nothing"
    );

    let ids: Vec<u32> = (0..4).map(|i| (i % VOCAB) as u32).collect();
    for _ in 0..3 {
        stage.run(BlockInput::Tokens(&ids)).expect("run");
    }
    assert_eq!(
        stage.graphs_built(),
        1,
        "the stage rebuilt its graph per call; decode would recompile 45 layers \
         and re-upload the shard for every token"
    );

    // A different length is a genuinely different graph and must build once.
    let longer: Vec<u32> = (0..6).map(|i| (i % VOCAB) as u32).collect();
    stage.run(BlockInput::Tokens(&longer)).expect("run");
    stage.run(BlockInput::Tokens(&longer)).expect("run");
    assert_eq!(stage.graphs_built(), 2);
}

/// The graph cache must be bounded.
///
/// Each entry owns an uploaded copy of this rank's weights, so one graph per
/// distinct prompt length is one full shard copy per length — gigabytes each in
/// production, on a node the planner sized to hold exactly one. An unbounded
/// cache turns a recompile (slow) into an OOM (dead), which is the worse
/// failure and the easier one to ship by accident.
#[test]
fn the_graph_cache_is_bounded() {
    let c = cfg(64);
    let tensors = all_tensors(&c);
    let mut stage = Glm5NextPipelineStage::new(c.clone(), dev(), 0, 1, tensors).expect("stage");

    for len in 1..=6usize {
        let ids: Vec<u32> = (0..len).map(|i| (i % VOCAB) as u32).collect();
        stage.run(BlockInput::Tokens(&ids)).expect("run");
        assert!(
            stage.graphs_cached() <= rlx_glm5next::pipeline::DEFAULT_GRAPH_CACHE,
            "after {len} distinct lengths the stage holds {} graphs, each a copy \
             of the shard",
            stage.graphs_cached()
        );
    }
    assert_eq!(
        stage.graphs_built(),
        6,
        "every distinct length needs a build"
    );
}

/// Even at a cache of one, the length being run is never the one evicted —
/// otherwise `graph_for` would return a graph it had just dropped.
#[test]
fn a_single_slot_cache_still_serves_the_running_length() {
    let c = cfg(64);
    let tensors = all_tensors(&c);
    let mut stage = Glm5NextPipelineStage::new(c.clone(), dev(), 0, 1, tensors)
        .expect("stage")
        .with_graph_cache(1);

    let short: Vec<u32> = vec![1, 2];
    let long: Vec<u32> = vec![1, 2, 3, 4];
    for _ in 0..2 {
        stage.run(BlockInput::Tokens(&short)).expect("short");
        stage.run(BlockInput::Tokens(&long)).expect("long");
    }
    assert_eq!(stage.graphs_cached(), 1);
    // Thrashing, which is what a cap of one buys: each alternation rebuilds.
    assert_eq!(stage.graphs_built(), 4);
}

/// Eviction must not change any answer — a stale binding would show up as a
/// rebuilt graph disagreeing with the one it replaced.
#[test]
fn eviction_does_not_change_the_answer() {
    let c = cfg(64);
    let tensors = all_tensors(&c);
    let mut stage = Glm5NextPipelineStage::new(c.clone(), dev(), 0, 1, tensors)
        .expect("stage")
        .with_graph_cache(1);

    let a: Vec<u32> = (0..3).map(|i| (i % VOCAB) as u32).collect();
    let b: Vec<u32> = (0..5).map(|i| ((i * 3) % VOCAB) as u32).collect();

    let BlockOutput::Logits(first) = stage.run(BlockInput::Tokens(&a)).unwrap() else {
        panic!("logits");
    };
    // Force `a`'s graph out, then bring it back.
    stage.run(BlockInput::Tokens(&b)).unwrap();
    let BlockOutput::Logits(again) = stage.run(BlockInput::Tokens(&a)).unwrap() else {
        panic!("logits");
    };
    assert_eq!(
        first, again,
        "a rebuilt graph disagrees with the evicted one"
    );
}

/// Caching the graph must not change what it computes — a stale input binding
/// or a param set only on the first build would show up here.
#[test]
fn a_cached_graph_returns_the_same_answer_every_time() {
    let c = cfg(64);
    let tensors = all_tensors(&c);
    let mut stage = Glm5NextPipelineStage::new(c.clone(), dev(), 0, 1, tensors).expect("stage");
    assert_eq!(stage.role(), BlockRole::Single);

    let a: Vec<u32> = (0..5).map(|i| (i % VOCAB) as u32).collect();
    let b: Vec<u32> = (0..5).map(|i| ((i * 7 + 1) % VOCAB) as u32).collect();

    let BlockOutput::Logits(first_a) = stage.run(BlockInput::Tokens(&a)).unwrap() else {
        panic!("Single must produce logits");
    };
    let BlockOutput::Logits(first_b) = stage.run(BlockInput::Tokens(&b)).unwrap() else {
        panic!("logits");
    };
    let BlockOutput::Logits(again_a) = stage.run(BlockInput::Tokens(&a)).unwrap() else {
        panic!("logits");
    };
    assert_eq!(first_a, again_a, "the reused graph gave a different answer");
    assert_ne!(
        first_a, first_b,
        "two different token sequences gave identical logits — the cached graph \
         is not reading its input"
    );
}

/// A cut carries the mHC stream state, all `hc_mult` of them.
///
/// This is the one structural way `glm5next` differs from an ordinary pipeline.
/// The four residual streams run the model's whole depth and are averaged only
/// at `hc_head`; a stage that handed on a collapsed `[1, seq, hidden]` would be
/// broadcasting one vector into four identical streams at the next rank and
/// silently discarding everything mHC had mixed. Sizes are the cheap way to
/// catch a regression to that.
#[test]
fn a_cut_carries_every_residual_stream() {
    let c = cfg(64);
    let seq = 5usize;
    let ids: Vec<u32> = (0..seq).map(|i| (i % VOCAB) as u32).collect();
    let tensors = all_tensors(&c);

    // rank 1 of 2 is `First`: it embeds and hands on the streams.
    let mut first =
        Glm5NextPipelineStage::new(c.clone(), dev(), 1, 2, tensors.clone()).expect("stage");
    assert_eq!(first.role(), BlockRole::First);
    let out = first.run(BlockInput::Tokens(&ids)).expect("run");
    let BlockOutput::Hidden(h) = out else {
        panic!("the First block must hand on hidden state, not logits");
    };
    assert_eq!(
        h.len(),
        seq * HC * HIDDEN,
        "a cut must carry all {HC} streams ({} elements), not a collapsed \
         hidden state ({} elements)",
        seq * HC * HIDDEN,
        seq * HIDDEN
    );
}

/// The stream state is what makes the pipeline cost what it costs.
///
/// Worth an assertion rather than a comment: a planner sizing inter-stage
/// transfer from `hidden_size` would under-count `glm5next` by `hc_mult`.
#[test]
fn transfer_width_is_hc_mult_times_hidden() {
    let c = cfg(64);
    assert_eq!(c.hc_mult, HC);
    assert_eq!(c.hc_mult * c.hidden_size, HC * HIDDEN);
    assert!(
        c.hc_mult > 1,
        "a model with one stream would make this test vacuous"
    );
}

/// Every rank must claim a disjoint slice, and together they must claim all of
/// it — a layer run twice or not at all is silently wrong output.
#[test]
fn the_ranks_partition_the_layer_stack() {
    let c = cfg(64);
    for world in 1u32..=5 {
        let mut seen = vec![0usize; c.num_hidden_layers];
        for r in 0..world {
            for l in block_spec_for(&c, r, world).layers {
                seen[l] += 1;
            }
        }
        assert!(
            seen.iter().all(|&n| n == 1),
            "world {world} does not partition {} layers: {seen:?}",
            c.num_hidden_layers
        );
    }
}

/// Exactly one rank embeds and exactly one runs the head — and, because the
/// coordinator numbers blocks in reverse, they are the ranks it expects.
#[test]
fn the_ends_of_the_model_are_owned_once_each() {
    let c = cfg(64);
    for world in 1u32..=4 {
        let specs: Vec<BlockSpec> = (0..world).map(|r| block_spec_for(&c, r, world)).collect();
        assert_eq!(
            specs.iter().filter(|s| s.embed_input).count(),
            1,
            "world {world}: embedding is not owned exactly once"
        );
        assert_eq!(
            specs.iter().filter(|s| s.produce_logits).count(),
            1,
            "world {world}: the LM head is not owned exactly once"
        );
        // The coordinator sends tokens to `First` and reads logits from `Last`.
        let first = (0..world).find(|&r| specs[r as usize].embed_input).unwrap();
        let last = (0..world)
            .find(|&r| specs[r as usize].produce_logits)
            .unwrap();
        assert_eq!(
            block_role(first, world),
            if world == 1 {
                BlockRole::Single
            } else {
                BlockRole::First
            }
        );
        assert_eq!(
            block_role(last, world),
            if world == 1 {
                BlockRole::Single
            } else {
                BlockRole::Last
            }
        );
    }
}

/// A rank must hold only its own weights — that is the entire point of sharding
/// a 93 GB checkpoint across machines. A filter that leaked would still compute
/// the right answer here and cost every node the whole model in production, so
/// the check is on what is NOT kept.
#[test]
fn a_rank_keeps_only_the_weights_it_runs() {
    let c = cfg(64);
    let tensors = all_tensors(&c);
    let world = 4u32;

    let mut total = 0usize;
    for r in 0..world {
        let stage =
            Glm5NextPipelineStage::new(c.clone(), dev(), r, world, tensors.clone()).expect("stage");
        let spec = block_spec_for(&c, r, world);
        total += stage.weight_count();

        for name in tensors.keys() {
            let kept = block_weight_filter(name, &c, &spec);
            if let Some(rest) = name.strip_prefix("blk.") {
                let i: usize = rest.split('.').next().unwrap().parse().unwrap();
                assert_eq!(
                    kept,
                    spec.layers.contains(&i),
                    "rank {r} (layers {:?}) got `{name}` wrong",
                    spec.layers
                );
            }
        }
        assert!(
            stage.weight_count() < tensors.len(),
            "rank {r} of {world} kept every tensor — the shard filter is not filtering"
        );
    }
    // Each tensor lands on exactly one rank: the per-layer ones by range, the
    // embedding and head by role. (`tie_word_embeddings` is false here, so the
    // embedding is not shared between two ranks.)
    assert!(!c.tie_word_embeddings);
    assert_eq!(
        total,
        tensors.len(),
        "the shards do not partition the checkpoint"
    );
}

/// A block cannot silently accept a stream state of the wrong width.
#[test]
fn a_mis_sized_stream_state_is_rejected() {
    let c = cfg(64);
    let tensors = all_tensors(&c);
    let mut middle =
        Glm5NextPipelineStage::new(c.clone(), dev(), 1, 3, tensors).expect("build middle stage");
    assert_eq!(middle.role(), BlockRole::Middle);

    // A collapsed hidden state: the plausible mistake, and not a multiple of
    // hc_mult * hidden.
    let bad = vec![0.0f32; 5 * HIDDEN + 1];
    let err = match middle.run(BlockInput::Hidden(&bad)) {
        Ok(_) => panic!("a mis-sized stream state must be rejected, not run"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("hc_mult"), "unhelpful error: {err}");
}

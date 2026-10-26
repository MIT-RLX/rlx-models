// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! A `qwen3` pipeline split must compute what the whole model computes.
//!
//! `Qwen3PipelineDecodeStage` had no automated coverage at all — five examples
//! drive it, every one of them needing real weights. That is a bad place to be
//! for a distributed stage, because the failures live at the seams (which
//! tensor crosses a cut, which rank owns the embedding and which the head, how
//! the KV cache is sliced) and every one of them produces plausible,
//! well-scaled, wrong logits.
//!
//! Running the relay for real needs a transport; the arithmetic does not. The
//! `relay` helper drives the same [`BlockRunner`]s by hand in rank order,
//! which is what the coordinator does between sends.
//!
//! ## This test depends on a property it does not check
//!
//! Weights come from [`rlx_qwen3::synth`], and the comparison is only
//! meaningful because they are non-zero **and different per layer**. Verified
//! by sabotage: giving every layer identical weights leaves both tests here
//! passing, because an interchangeable layer stack cannot reveal a layer-order
//! mistake. What catches that is `synth`'s own
//! `each_layer_has_its_own_weights`, which is therefore load-bearing for this
//! file rather than a formality.
//!
//! Also verified by sabotage: relaying ranks low-to-high instead of
//! high-to-low (rank 0 owns the head) diverges by 1.0 relative, so the rank
//! ordering really is under test. Two more, against the decode step: stuffing
//! the rope slice at position zero every step, and pointing every layer of a
//! block at the block's first layer, each fail only
//! `a_decode_step_matches_prefilling_the_longer_prompt` — and only at `TOL`,
//! see the note there.

use rlx_core::weight_map::WeightMap;
use rlx_distributed::{BlockInput, BlockOutput, BlockRunner};
use rlx_qwen3::pipeline_decode::{self, Qwen3PipelineDecodeStage};
use rlx_qwen3::synth::{Tensors, synth_tensors, tiny_config};
use rlx_qwen3::{Qwen3Config, generator::Qwen3Generator};
use rlx_runtime::Device;

const LAYERS: usize = 4;
const PROMPT: [u32; 5] = [3, 1, 4, 1, 5];

/// How far a split may differ from the whole model before it is a bug.
///
/// Measured rather than guessed, because the obvious choice is far too
/// loose to catch anything: on the CPU every prefill width here is
/// **bit-exact** against the monolithic graph, and the bucketed decode path
/// differs by 4.7e-7. A decode step whose position is wrong — rope sliced at
/// zero every step, the classic way to break this — moves the logits by only
/// 1.5e-4 relative at this size, so a 1e-3-ish tolerance copied from a
/// parity test passes a model that re-attends position zero forever. This
/// sits between the two, with room for a device whose kernels reduce in
/// another order.
const TOL: f32 = 1e-5;

fn dev() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn cfg() -> Qwen3Config {
    tiny_config(LAYERS)
}

fn tensors(c: &Qwen3Config) -> Tensors {
    synth_tensors(c)
}

/// Logits from the single whole-model graph — the reference.
fn whole_model(c: &Qwen3Config, ids: &[u32]) -> Vec<f32> {
    let mut wm = WeightMap::from_tensors(tensors(c));
    let mut g = Qwen3Generator::from_loader(c.clone(), &mut wm, dev())
        .expect("build the monolithic generator");
    // `sequence_logits` returns every position; the stage returns the last
    // one, which is what a decode step needs.
    let all = g.sequence_logits(ids).expect("monolithic logits");
    let v = c.vocab_size;
    assert_eq!(all.len(), ids.len() * v, "one logit row per position");
    all[(ids.len() - 1) * v..].to_vec()
}

/// One pass down the ranks: the first one reads tokens, each of the others
/// the hidden state from the rank above, and rank 0 returns logits.
///
/// Rank 0 owns the head, so ranks run from the highest down — the same order
/// the coordinator uses, and getting it backwards is one of the mistakes this
/// is here to catch.
fn one_pass(stages: &mut [Qwen3PipelineDecodeStage], hist: &[u32]) -> Vec<f32> {
    let mut carry: Option<Vec<f32>> = None;
    let mut logits = None;
    for rank in (0..stages.len()).rev() {
        let input = match &carry {
            None => BlockInput::Tokens(hist),
            Some(h) => BlockInput::Hidden(h),
        };
        match stages[rank].run(input).expect("run stage") {
            BlockOutput::Hidden(h) => carry = Some(h),
            BlockOutput::Logits(l) => logits = Some(l),
        }
    }
    logits.expect("the head rank must produce logits")
}

/// Logits from a `world`-way split: prefill `ids`, then one decode step per
/// token in `extra`, returning the logits after the last of them.
///
/// `bucketed` picks which decode path runs — the per-bucket compiled graph
/// (what a generation actually uses) or a fresh compile per step. They are
/// separate code paths over separate graphs (custom mask and padded K/V
/// versus an exact-extent causal one), so a test that drives only one of
/// them is testing half of decode.
fn relay_steps(
    c: &Qwen3Config,
    world: u32,
    ids: &[u32],
    extra: &[u32],
    bucketed: bool,
) -> Vec<f32> {
    let t = tensors(c);
    let mut stages: Vec<Qwen3PipelineDecodeStage> = (0..world)
        .map(|r| {
            let s = pipeline_decode::stage(c.clone(), dev(), r, world, t.clone());
            if bucketed { s.with_decode_cache(64) } else { s }
        })
        .collect();

    let mut hist = ids.to_vec();
    let mut out = one_pass(&mut stages, &hist);
    for &tok in extra {
        hist.push(tok);
        out = one_pass(&mut stages, &hist);
    }
    out
}

/// Logits from a `world`-way split over the prompt alone.
fn relay(c: &Qwen3Config, world: u32, ids: &[u32]) -> Vec<f32> {
    relay_steps(c, world, ids, &[], false)
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit width");
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = a.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
    max_abs / scale
}

/// A split pipeline reproduces the whole model, at every width.
///
/// `LAYERS = 4` so that `world = 2` and `world = 4` both give every rank at
/// least one layer — a width that leaves a rank empty would stop testing the
/// thing it claims to.
#[test]
fn a_split_pipeline_reproduces_the_whole_model() {
    let c = cfg();
    let want = whole_model(&c, &PROMPT);
    assert_eq!(want.len(), c.vocab_size, "last-position logits");
    assert!(want.iter().all(|v| v.is_finite()));
    // Non-constant output, or the comparison below is vacuous.
    let spread = want.iter().cloned().fold(f32::MIN, f32::max)
        - want.iter().cloned().fold(f32::MAX, f32::min);
    assert!(spread > 1e-4, "logits are nearly constant: spread {spread}");

    for world in [1u32, 2, 4] {
        let got = relay(&c, world, &PROMPT);
        let d = max_rel_diff(&want, &got);
        assert!(
            d < TOL,
            "world {world}: a {world}-way split differs from the whole model by \
             {d:.3e} relative"
        );
    }
}

/// Each width agrees with every other, independent of the monolithic path.
///
/// Checked separately because it isolates the *split* from the stage: if the
/// stage itself disagrees with the whole model, the test above fails for every
/// width at once, while a seam bug shows up here as one width differing.
#[test]
fn every_split_width_agrees_with_world_one() {
    let c = cfg();
    let one = relay(&c, 1, &PROMPT);
    for world in [2u32, 4] {
        let d = max_rel_diff(&one, &relay(&c, world, &PROMPT));
        assert!(d < TOL, "world {world} vs world 1: {d:.3e} relative");
    }
}

/// A decode step off a split prefill matches prefilling the longer prompt.
///
/// This is the part of the stage a generation spends all its time in, and
/// the part the prefill tests above cannot see: the decode graph is a
/// different graph, reading the K/V the prefill exported and taking its
/// position from host-sliced rope tables. A cut in the wrong place, a cache
/// slot off by one, or a position that stays at zero all leave the prefill
/// tests passing.
#[test]
fn a_decode_step_matches_prefilling_the_longer_prompt() {
    let c = cfg();
    const NEXT: [u32; 2] = [9, 2];
    let mut hist = PROMPT.to_vec();
    hist.extend_from_slice(&NEXT);
    let want = whole_model(&c, &hist);

    for world in [1u32, 2, 4] {
        for bucketed in [false, true] {
            let got = relay_steps(&c, world, &PROMPT, &NEXT, bucketed);
            let d = max_rel_diff(&want, &got);
            assert!(
                d < TOL,
                "world {world} (bucketed {bucketed}): two decode steps differ from \
                 the whole model by {d:.3e} relative"
            );
        }
    }
}

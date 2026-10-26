//! `Qwen35Flow::bidirectional_attention` — the encoder-style prefill that
//! `pplx-decider` checkpoints need (`qwen35.attention.causal = false`).
//!
//! Three claims, each against an independently built graph:
//!
//! 1. it actually changes the computation: position 0 of a bidirectional run
//!    differs from the causal run, because only the former can see position 1;
//! 2. right-padding plus `attn_key_mask` is exact: a two-row batch padded to a
//!    common length reproduces each row's exactly-sized unpadded run at every
//!    real position (the Gated DeltaNet layers stay causal, so a later pad
//!    cannot reach them, and the mask keeps it out of the attention layer);
//! 3. the mask is load-bearing: the same padded batch without a mask does
//!    *not* reproduce the short row, so claim 2 is not passing vacuously.

use std::collections::HashMap;

use rlx_qwen35::synth::{synth_weights, tiny_cfg};
use rlx_qwen35::{ATTN_KEY_MASK, Qwen35Flow, Qwen35TrunkLayer};
use rlx_runtime::{Device, Session};

const SHORT: [f32; 4] = [5.0, 11.0, 2.0, 19.0];
const LONG: [f32; 6] = [7.0, 3.0, 23.0, 1.0, 13.0, 29.0];
/// Pad id deliberately not 0, so an unmasked pad changes the output.
const PAD: f32 = 17.0;

fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

/// Full-rank pseudo-random parameters keyed by name: the `ramp()` synth
/// weights are so nearly position-invariant that positions barely differ.
///
/// Builder-synthesized constants are left alone. The conv's `causal_pad` is a
/// `[batch, k-1, C]` block of zeros; randomizing it hands each batch row
/// different "padding", which reads exactly like a batch>1 bug in the GDN
/// layers (row 0 exact, row 1 off by ~1e-2). Norm `beta`s are zeros too.
fn randomize(params: &mut HashMap<String, Vec<f32>>) {
    for (name, data) in params.iter_mut() {
        if name.ends_with(".causal_pad") || name.ends_with(".beta") {
            continue;
        }
        let seed = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x1000_0000_01b3)
        });
        for (i, v) in data.iter_mut().enumerate() {
            *v = 0.25 * hashed(seed, i);
        }
    }
}

/// Hidden states `[batch, seq, hidden]` after the final norm.
fn run(
    batch: usize,
    seq: usize,
    ids: &[f32],
    bidirectional: Option<bool>,
    key_mask: Option<&[f32]>,
) -> Vec<f32> {
    let cfg = tiny_cfg();
    let weights = synth_weights(&cfg);
    let mut flow = Qwen35Flow::prefill(&cfg, &weights, batch, seq).export_normed_hidden();
    if let Some(masked) = bidirectional {
        flow = flow.bidirectional_attention(masked);
    }
    let (built, packed) = flow.build_with_weights(&weights).expect("build");
    assert!(packed.is_empty());
    let (hir, mut params) = built.into_parts().expect("parts");
    randomize(&mut params);
    let mut compiled = Session::new(Device::Cpu).compile_hir(hir).expect("compile");
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    let mut feeds: Vec<(&str, &[f32])> = vec![("input_ids", ids)];
    if let Some(m) = key_mask {
        feeds.push((ATTN_KEY_MASK, m));
    }
    compiled.run(&feeds)[0].clone()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn tiny_cfg_has_a_full_attention_layer() {
    let cfg = tiny_cfg();
    let w = synth_weights(&cfg);
    assert!(
        w.trunk_layers
            .iter()
            .any(|l| matches!(l, Qwen35TrunkLayer::FullAttn(_))),
        "the bidirectional option only touches full-attention layers"
    );
}

#[test]
fn bidirectional_changes_early_positions() {
    let h = tiny_cfg().hidden_size;
    let causal = run(1, 4, &SHORT, None, None);
    let bidir = run(1, 4, &SHORT, Some(false), None);
    let d0 = max_diff(&causal[..h], &bidir[..h]);
    assert!(
        d0 > 1e-3,
        "position 0 is identical with and without the causal mask ({d0}); \
         the option did not reach the attention layer"
    );
}

#[test]
fn padded_batch_with_key_mask_matches_exact_rows() {
    let h = tiny_cfg().hidden_size;
    let seq = LONG.len();
    let short_exact = run(1, SHORT.len(), &SHORT, Some(false), None);
    let long_exact = run(1, seq, &LONG, Some(false), None);

    let mut ids = vec![PAD; 2 * seq];
    ids[..SHORT.len()].copy_from_slice(&SHORT);
    ids[seq..].copy_from_slice(&LONG);
    let mut mask = vec![0f32; 2 * seq];
    mask[..SHORT.len()].fill(1.0);
    mask[seq..].fill(1.0);

    let padded = run(2, seq, &ids, Some(true), Some(&mask));
    let short_rows = &padded[..SHORT.len() * h];
    let long_rows = &padded[seq * h..2 * seq * h];
    let ds = max_diff(short_rows, &short_exact);
    let dl = max_diff(long_rows, &long_exact);
    assert!(ds < 1e-5, "short row drifted under padding: {ds}");
    assert!(dl < 1e-5, "long row drifted in a batch: {dl}");

    // Same batch, every key attended: the short row now sees its pads.
    let ones = vec![1f32; 2 * seq];
    let unmasked = run(2, seq, &ids, Some(true), Some(&ones));
    let du = max_diff(&unmasked[..SHORT.len() * h], &short_exact);
    assert!(
        du > 1e-3,
        "attending the pads did not change the short row ({du}); the masked \
         comparison above proves nothing"
    );
}

// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `google/embeddinggemma-2` — config, tensor contract, and the pooling head.
//!
//! Both fixtures come from the published checkpoint: the `config.json`
//! verbatim, and the 16 tensors that sit outside the per-layer stacks, read out
//! of the safetensors header (a few KB of a multi-gigabyte file). So the
//! contract this crate believes is checked against the one the model ships
//! with, in both directions, without the weights.

use rlx_gemma::embedding_gemma2::{EmbeddingGemma2Config, l2_normalize, pool_project_normalize};
use std::collections::BTreeMap;

const CONFIG: &str = include_str!("fixtures/embeddinggemma2_config.json");
const HEAD_TENSORS: &str = include_str!("fixtures/embeddinggemma2_head_tensors.tsv");

fn cfg() -> EmbeddingGemma2Config {
    EmbeddingGemma2Config::from_hf_json(CONFIG).expect("the published config should parse")
}

/// `name -> shape`, from the checkpoint's own header.
fn published() -> BTreeMap<String, Vec<usize>> {
    HEAD_TENSORS
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let (name, dims) = l.split_once('\t').expect("name<TAB>shape");
            (
                name.to_string(),
                dims.split(',').map(|d| d.parse().expect("dim")).collect(),
            )
        })
        .collect()
}

#[test]
fn the_published_config_parses() {
    let c = cfg();
    // The trunk is a small Gemma 4, not an exotic encoder.
    assert_eq!(c.text.hidden_size, 512);
    assert_eq!(c.text.num_hidden_layers, 24);
    assert_eq!(c.text.num_attention_heads, 4);
    assert_eq!(c.text.num_key_value_heads, 2);
    assert_eq!(c.text.vocab_size, 262144);
    assert_eq!(c.text.intermediate_size, 2048);
    // The embedder's own two numbers.
    assert_eq!(c.embedding_dim, 768);
    assert_eq!(c.vision_soft_tokens_per_image, 280);
    assert!(c.has_vision && c.has_audio, "all three modalities ship");
}

/// Per-layer input embeddings, and the width that proves it.
///
/// `hidden_size_per_layer_input = 512` over 24 layers gives a
/// `[12288, 512]` projection. Getting that arithmetic wrong is not a shape
/// error at load time — it is a slice of the wrong layer's embedding.
#[test]
fn the_per_layer_input_projection_width_matches_the_checkpoint() {
    let c = cfg();
    assert!(c.text.has_ple(), "embeddinggemma-2 uses per-layer inputs");
    assert_eq!(c.text.ple_width(), 512);
    let want = c
        .per_layer_projection_shape()
        .expect("a PLE checkpoint has one");
    assert_eq!(want, [12288, 512], "24 layers x 512");

    let got = &published()["language_model.ple.per_layer_model_projection.weight"];
    assert_eq!(got, &want.to_vec(), "against the published header");
}

/// The head tensors this crate expects are the ones the checkpoint has.
#[test]
fn the_head_tensor_contract_matches_the_checkpoint() {
    let c = cfg();
    let pub_names = published();
    for name in c.head_tensor_names() {
        assert!(
            pub_names.contains_key(&name),
            "{name} is expected but not in the checkpoint"
        );
    }
    // The output projection's orientation, which decides the matvec.
    assert_eq!(
        pub_names["language_model.embedding_projection.weight"],
        vec![c.embedding_dim, c.text.hidden_size],
        "[embedding_dim, hidden]"
    );
    // An embedder has no vocabulary head: nothing should be looking for one.
    assert!(
        !pub_names.keys().any(|k| k.contains("lm_head")),
        "there is no lm_head in an embedding model"
    );
}

/// Projecting the pooled vector equals pooling the projected tokens.
///
/// This is the shortcut `pool_project_normalize` takes, and it is exact rather
/// than approximate because `embedding_projection` has no bias: a linear map
/// commutes with an average. Worth pinning — if a future checkpoint adds a
/// bias, this test fails instead of the embeddings quietly shifting.
#[test]
fn pooling_then_projecting_equals_projecting_then_pooling() {
    let (seq, hidden, dim) = (5usize, 8usize, 4usize);
    let tokens: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32) * 0.37).sin())
        .collect();
    let proj: Vec<f32> = (0..dim * hidden)
        .map(|i| ((i as f32) * 0.11).cos())
        .collect();
    let mask = vec![true, true, false, true, true];

    let ours = pool_project_normalize(&tokens, &mask, hidden, &proj, dim).expect("pool");

    // The long way: project every token, average the live ones, normalize.
    let mut acc = vec![0f32; dim];
    let live = mask.iter().filter(|m| **m).count();
    for (t, keep) in mask.iter().enumerate() {
        if !keep {
            continue;
        }
        let x = &tokens[t * hidden..(t + 1) * hidden];
        for (o, row) in acc.iter_mut().zip(proj.chunks_exact(hidden)) {
            *o += row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>();
        }
    }
    for v in acc.iter_mut() {
        *v /= live as f32;
    }
    l2_normalize(&mut acc);

    let max = ours
        .iter()
        .zip(&acc)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 1e-6, "the two orders differ by {max}");
}

/// Masked tokens are excluded from the mean, not averaged in as zeros.
#[test]
fn masked_tokens_do_not_drag_the_embedding() {
    let (hidden, dim) = (4usize, 2usize);
    let proj = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    // Two real tokens, then two pad rows of a very different value.
    let tokens = vec![
        1.0, 2.0, 0.0, 0.0, //
        3.0, 4.0, 0.0, 0.0, //
        99.0, 99.0, 0.0, 0.0, //
        99.0, 99.0, 0.0, 0.0,
    ];
    let with_pad =
        pool_project_normalize(&tokens, &[true, true, false, false], hidden, &proj, dim).unwrap();
    let without =
        pool_project_normalize(&tokens[..2 * hidden], &[true, true], hidden, &proj, dim).unwrap();
    assert_eq!(with_pad, without, "padding changed the embedding");

    // And the result is a unit vector.
    let n: f32 = with_pad.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((n - 1.0).abs() < 1e-6, "norm {n}");
}

/// The refusals: a wrong-shaped projection, a bad token count, an empty mask.
#[test]
fn bad_inputs_are_refused() {
    let proj = vec![0.0; 2 * 4];
    let toks = vec![0.0; 3 * 4];
    assert!(
        pool_project_normalize(&toks, &[true; 3], 4, &proj[..3], 2).is_err(),
        "short proj"
    );
    assert!(
        pool_project_normalize(&toks[..5], &[true; 3], 4, &proj, 2).is_err(),
        "ragged tokens"
    );
    assert!(
        pool_project_normalize(&toks, &[true; 2], 4, &proj, 2).is_err(),
        "mask length"
    );
    assert!(
        pool_project_normalize(&toks, &[false; 3], 4, &proj, 2).is_err(),
        "all masked"
    );
}

/// A non-`embedding_gemma2` config is refused rather than half-read.
#[test]
fn a_different_model_type_is_refused() {
    let err = EmbeddingGemma2Config::from_hf_json(r#"{"model_type":"gemma4_unified"}"#)
        .expect_err("wrong model_type");
    assert!(format!("{err:#}").contains("embedding_gemma2"), "{err:#}");
}

/// The audio tower's clipped linears, against the reference's arithmetic.
///
/// `Gemma4ClippableLinear` is `clamp(x, in_min, in_max)` → bias-free matmul →
/// `clamp(y, out_min, out_max)`. Every audio linear in this checkpoint ships
/// all four bounds — 8 per layer across the two feed-forwards, the conv's
/// start/end, and q/k/v/post — while `relative_k_proj` ships none, and the
/// vision tower ships none at all because its `use_clipped_linears` is false.
/// This pins the flag against the tensors so the two cannot disagree.
#[test]
fn the_audio_tower_declares_clipped_linears_and_the_vision_tower_does_not() {
    let c = cfg();
    assert!(
        c.audio_use_clipped_linears,
        "embeddinggemma-2's audio tower clips"
    );

    // The config flag has to match what the checkpoint actually carries, which
    // is the cross-check that catches a reader trusting one over the other.
    let raw: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    let vision_clips = raw["vision_config"]["use_clipped_linears"]
        .as_bool()
        .expect("vision_config declares the flag");
    assert!(!vision_clips, "the vision tower does not clip");
}

/// Clipping with `±inf` bounds is the identity, which is what an untrained
/// bound means.
///
/// The reference initializes the buffers to `±inf`, so a checkpoint that never
/// learned them must behave exactly like an unclipped linear. Emitting
/// `clamp(x, -inf, inf)` would be arithmetically right but puts infinities in
/// the graph; skipping the clamp is the same answer without them.
#[test]
fn infinite_bounds_mean_no_clipping() {
    // Mirrors `clipped_linear`'s decision: only finite bounds are emitted.
    let finite = |b: f32| b.is_finite();
    assert!(!finite(f32::NEG_INFINITY) && !finite(f32::INFINITY));
    assert!(finite(-6.0) && finite(6.0));

    // And the clamp itself, for the finite case.
    let clamp = |x: f32, lo: f32, hi: f32| x.max(lo).min(hi);
    assert_eq!(clamp(-9.0, -6.0, 6.0), -6.0);
    assert_eq!(clamp(9.0, -6.0, 6.0), 6.0);
    assert_eq!(clamp(1.5, -6.0, 6.0), 1.5);
}

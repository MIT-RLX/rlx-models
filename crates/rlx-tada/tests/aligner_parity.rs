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

//! Parity vs `transformers.Wav2Vec2ForCTC`.
//!
//! The aligner decides which frame each text token owns, and a one-frame error
//! there shifts a token's whole acoustic latent — so the CTC logits are pinned
//! against a real HF forward rather than checked for plausibility.
//!
//! Two details in this stack are easy to get wrong and invisible afterwards:
//! the positional convolution is weight-normed along the **kernel** axis
//! (`dim=2`) rather than the output-channel axis every other weight-normed conv
//! in TADA uses, and its even kernel width makes the output one step too long,
//! which upstream trims from the end (`Wav2Vec2SamePadLayer`). Both are covered
//! here.

mod common;
use common::device;
use rlx_tada::aligner::Aligner;
use rlx_tada::config::AlignerConfig;
use rlx_tada::weights::TensorStore;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    hidden: usize,
    layers: usize,
    heads: usize,
    ffn: usize,
    vocab: usize,
    conv_dim: usize,
    conv_layers: Vec<Vec<usize>>,
    pos_k: usize,
    pos_groups: usize,
    x: Vec<f32>,
    frames: usize,
    logits: Vec<f32>,
}

#[test]
fn matches_the_upstream_wav2vec2_ctc() {
    let f: Fixture = serde_json::from_str(include_str!("fixtures/aligner_reference.json"))
        .expect("parse aligner_reference.json");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/aligner_reference.safetensors");
    let store = std::sync::Arc::new(TensorStore::open(&path).expect("open fixture weights"));

    let cfg = AlignerConfig {
        hidden_size: f.hidden,
        num_hidden_layers: f.layers,
        num_attention_heads: f.heads,
        intermediate_size: f.ffn,
        vocab_size: f.vocab,
        layer_norm_eps: 1e-5,
        conv_layers: f.conv_layers.iter().map(|c| (c[0], c[1], c[2])).collect(),
        conv_dim: f.conv_dim,
        num_conv_pos_embeddings: f.pos_k,
        num_conv_pos_embedding_groups: f.pos_groups,
        feat_extract_norm_groups: f.conv_dim,
    };
    let aligner = Aligner::load(store, cfg).expect("load aligner");
    assert_eq!(aligner.vocab_size(), f.vocab);
    assert_eq!(
        aligner.frames_for(f.x.len()),
        f.frames,
        "conv front end emits a different frame count than torch"
    );

    let (got, frames) = aligner.logits(device(), &f.x).expect("run aligner");
    assert_eq!(frames, f.frames);
    assert_eq!(got.len(), f.logits.len());

    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in got.iter().zip(&f.logits).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst < 2e-3,
        "worst deviation {worst} at index {at} (got {}, want {})",
        got[at],
        f.logits[at]
    );

    // What the alignment actually consumes is the argmax per frame, so pin
    // that too: it is the quantity a small numeric drift could still flip.
    let argmax = |v: &[f32]| -> Vec<usize> {
        v.chunks_exact(f.vocab)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect()
    };
    assert_eq!(
        argmax(&got),
        argmax(&f.logits),
        "per-frame argmax disagrees with torch"
    );
}

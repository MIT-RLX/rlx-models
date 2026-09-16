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

//! Parity vs `tada.modules.encoder.WavEncoder` and
//! `tada.modules.decoder.DACDecoder`.
//!
//! These stacks are DAC's, so the graphs come from `rlx-dac`. What is *not*
//! borrowed — and is what this test exercises — is the mapping from TADA's
//! checkpoint into `rlx-dac`'s layer structs: the weight-norm fusion (TADA
//! ships the new `parametrizations.weight.original0/1` form, `rlx-dac`'s own
//! loader reads the legacy `weight_g`/`weight_v`), the per-block dilation
//! schedule, and the stride-derived padding. Every one of those is a silent
//! failure if wrong: the output stays finite and merely sounds incorrect.

use ndarray::Array2;
use rlx_dac::graph::CodecGraph;
mod common;
use common::device;
use rlx_tada::weights::{TensorStore, load_wav_decoder, load_wav_encoder};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    enc_strides: Vec<usize>,
    dec_strides: Vec<usize>,
    x: Vec<f32>,
    ye_shape: Vec<usize>,
    ye: Vec<f32>,
    z: Vec<f32>,
    yd_shape: Vec<usize>,
    yd: Vec<f32>,
}

fn worst_deviation(got: &[f32], want: &[f32]) -> (f32, usize) {
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

fn fixture() -> (Fixture, TensorStore) {
    let f: Fixture = serde_json::from_str(include_str!("fixtures/codec_conv_reference.json"))
        .expect("parse codec_conv_reference.json");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/codec_conv_reference.safetensors");
    (f, TensorStore::open(&path).expect("open fixture weights"))
}

#[test]
fn wav_encoder_matches_upstream() {
    let (f, store) = fixture();
    let enc = load_wav_encoder(&store, "wav_encoder", &f.enc_strides).expect("load wav encoder");
    let mut g = CodecGraph::encoder(device(), &enc, f.x.len()).expect("compile");
    let (c, t) = g.out_dims();
    assert_eq!(
        (c, t),
        (f.ye_shape[1], f.ye_shape[2]),
        "output geometry disagrees with torch"
    );
    let out: Array2<f32> = g.run(&f.x).expect("run");
    let got: Vec<f32> = out.iter().copied().collect();
    let (worst, at) = worst_deviation(&got, &f.ye);
    assert!(
        worst < 1e-4,
        "encoder deviates by {worst} at {at} (got {}, want {})",
        got[at],
        f.ye[at]
    );
}

#[test]
fn wav_decoder_matches_upstream() {
    let (f, store) = fixture();
    let dec = load_wav_decoder(&store, "wav_decoder", &f.dec_strides).expect("load wav decoder");
    let in_c = 6;
    let in_t = f.z.len() / in_c;
    let mut g = CodecGraph::decoder(device(), &dec, in_c, in_t).expect("compile");
    let (c, t) = g.out_dims();
    assert_eq!(
        (c, t),
        (f.yd_shape[1], f.yd_shape[2]),
        "output geometry disagrees with torch"
    );
    let out: Array2<f32> = g.run(&f.z).expect("run");
    let got: Vec<f32> = out.iter().copied().collect();
    let (worst, at) = worst_deviation(&got, &f.yd);
    assert!(
        worst < 1e-4,
        "decoder deviates by {worst} at {at} (got {}, want {})",
        got[at],
        f.yd[at]
    );
}

// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported VibeVoice decoder package, opened by rlx alone, decodes what
//! `VibeDecoderGraph` does from the real weights (`tests/fixtures`), at two
//! numbers of frames.
#![cfg(feature = "export")]

use rlx_core::export::ExportOptions;
use rlx_runtime::model::Tensor;
use rlx_vibevoice::export::export_rlxp;
use rlx_vibevoice::model::DIM_IN;
use rlx_vibevoice::{VibeDecoderGraph, VibeWeights};
use serde_json::json;

#[test]
fn an_exported_vibevoice_decodes_what_the_runner_does() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let weights = dir.join("vv_dec.safetensors");
    if !weights.is_file() {
        eprintln!("skip: no {}", weights.display());
        return;
    }
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("vv_ref.json")).unwrap()).unwrap();
    let latent: Vec<f32> = v["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let full = v["T"].as_u64().unwrap() as usize;
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("vibevoice.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&weights, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let w = VibeWeights::from_safetensors(&std::fs::read(&weights).unwrap()).unwrap();
    for t in [full, full - 2] {
        let cut: Vec<f32> = latent
            .chunks_exact(full)
            .flat_map(|row| row[..t].iter().copied())
            .collect();
        let input = Tensor::f32("latent", vec![DIM_IN, t], &cut);
        let got = m.run(&[input], &json!({})).unwrap()[0].to_f32().unwrap();
        let mut g = VibeDecoderGraph::compile_for(rlx_runtime::Device::Cpu, &w, t).unwrap();
        let want = g.run(&cut).unwrap();
        assert_eq!(got.len(), want.len(), "{t} frames");
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        eprintln!("{t} frames: {} samples, off by {worst}", got.len());
        assert!(worst < 1e-4, "{t} frames: off by {worst}");
    }
}

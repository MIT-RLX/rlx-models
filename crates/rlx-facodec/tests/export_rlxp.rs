// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported FACodec decoder package, opened by rlx alone, decodes what
//! `FacodecDecoderGraph` does for the speaker's timbre from the real
//! weights (`tests/fixtures`), at two numbers of frames and two speakers.
#![cfg(feature = "export")]

use rlx_core::export::ExportOptions;
use rlx_facodec::export::export_rlxp;
use rlx_facodec::model::IN_CH;
use rlx_facodec::{FacodecDecoderGraph, FacodecWeights};
use rlx_runtime::model::Tensor;
use serde_json::json;

#[test]
fn an_exported_facodec_decodes_what_the_runner_does() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let weights = dir.join("facodec_dec.safetensors");
    if !weights.is_file() {
        eprintln!("skip: no {}", weights.display());
        return;
    }
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("facodec_ref.json")).unwrap()).unwrap();
    let floats = |k: &str| -> Vec<f32> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let (emb, spk) = (floats("emb"), floats("spk"));
    let full = v["T"].as_u64().unwrap() as usize;
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("facodec.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&weights, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let w = FacodecWeights::from_safetensors(&std::fs::read(&weights).unwrap()).unwrap();
    // The reference speaker, and another.
    let other: Vec<f32> = spk.iter().rev().map(|x| x * 0.7).collect();
    for (t, speaker) in [(full, &spk), (full - 2, &other)] {
        let cut: Vec<f32> = emb
            .chunks_exact(full)
            .flat_map(|row| row[..t].iter().copied())
            .collect();
        let inputs = [
            Tensor::f32("emb", vec![IN_CH, t], &cut),
            Tensor::f32("spk", vec![1, IN_CH], speaker),
        ];
        let got = m.run(&inputs, &json!({})).unwrap()[0].to_f32().unwrap();
        let (gamma, beta) = w.timbre_affine(speaker);
        let mut g =
            FacodecDecoderGraph::compile_for(rlx_runtime::Device::Cpu, &w, &gamma, &beta, t)
                .unwrap();
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

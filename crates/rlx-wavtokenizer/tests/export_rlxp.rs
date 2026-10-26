// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported WavTokenizer decoder package, opened by rlx alone, decodes
//! what the runner's graph and `head_to_wav` do from the real weights
//! (`tests/fixtures`), at two numbers of frames.
#![cfg(feature = "export")]

use rlx_core::export::ExportOptions;
use rlx_runtime::model::Tensor;
use rlx_wavtokenizer::export::export_rlxp;
use rlx_wavtokenizer::{WavtokDecoderGraph, WavtokWeights, head_to_wav, model};
use serde_json::json;

#[test]
fn an_exported_wavtokenizer_decodes_what_the_runner_does() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let weights = dir.join("wavtokenizer.safetensors");
    if !weights.is_file() {
        eprintln!("skip: no {}", weights.display());
        return;
    }
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("wavtokenizer_ref.json")).unwrap()).unwrap();
    let feats: Vec<f32> = v["feats"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let full = feats.len() / model::INPUT_CH;
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("wavtokenizer.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&weights, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let w = WavtokWeights::from_safetensors(&std::fs::read(&weights).unwrap()).unwrap();
    for t in [full, full - 3] {
        // Each channel's first `t` frames.
        let cut: Vec<f32> = feats
            .chunks_exact(full)
            .flat_map(|row| row[..t].iter().copied())
            .collect();
        let input = Tensor::f32("feats", vec![1, model::INPUT_CH, t, 1], &cut);
        let got = m.run(&[input], &json!({})).unwrap()[0].to_f32().unwrap();
        let mut g = WavtokDecoderGraph::compile_for(rlx_runtime::Device::Cpu, &w, t).unwrap();
        let want = head_to_wav(&g.run(&cut).unwrap(), t, &w.head.window);
        assert_eq!(got.len(), want.len(), "{t} frames");
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let peak = want.iter().fold(0f32, |a, x| a.max(x.abs()));
        eprintln!(
            "{t} frames: {} samples, off by {worst} (peak {peak})",
            got.len()
        );
        assert!(worst < 1e-4 * peak.max(1.0), "{t} frames: off by {worst}");
    }
}

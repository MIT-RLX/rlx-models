// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported NanoCodec decoder package, opened by rlx alone, decodes what
//! `NanoDecoderGraph::decode_codes` does from the real weights
//! (`tests/fixtures`), at two numbers of frames.
#![cfg(feature = "export")]

use rlx_core::export::ExportOptions;
use rlx_nanocodec::export::export_rlxp;
use rlx_nanocodec::{NanoDecoderGraph, NanoWeights};
use rlx_runtime::model::Tensor;
use serde_json::json;

#[test]
fn an_exported_nanocodec_decodes_what_the_runner_does() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let weights = dir.join("nano_dec.safetensors");
    if !weights.is_file() {
        eprintln!("skip: no {}", weights.display());
        return;
    }
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("nano_ref.json")).unwrap()).unwrap();
    let codes: Vec<Vec<i64>> = v["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| {
            g.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect()
        })
        .collect();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("nanocodec.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&weights, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let w = NanoWeights::from_safetensors(&std::fs::read(&weights).unwrap()).unwrap();
    let full = codes[0].len();
    for t in [full, full - 5] {
        let cut: Vec<Vec<i64>> = codes.iter().map(|g| g[..t].to_vec()).collect();
        let bytes: Vec<u8> = cut
            .iter()
            .flatten()
            .flat_map(|&c| (c as i32).to_le_bytes())
            .collect();
        let input = Tensor::new("codes", rlx_ir::DType::I32, vec![cut.len(), t], bytes);
        let got = m.run(&[input], &json!({})).unwrap()[0].to_f32().unwrap();
        let mut runner = NanoDecoderGraph::compile_for(rlx_runtime::Device::Cpu, &w, t).unwrap();
        let want = runner.decode_codes(&cut).unwrap();
        assert_eq!(got.len(), want.len(), "{t} frames");
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        eprintln!("{t} frames: {} samples, off by {worst}", got.len());
        assert!(worst < 1e-4, "{t} frames: off by {worst}");
    }
}

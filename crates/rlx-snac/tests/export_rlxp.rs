// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported SNAC decoder package, opened by rlx alone, decodes what
//! `SnacDecoder::decode(codes, None)` does, from the real 24 kHz weights
//! (`tests/fixtures`), at two numbers of codes.
#![cfg(feature = "export")]

use rlx_core::HierarchicalCodes;
use rlx_core::export::ExportOptions;
use rlx_runtime::model::Tensor;
use rlx_snac::SnacDecoder;
use rlx_snac::export::export_rlxp;
use serde_json::json;

#[test]
fn an_exported_snac_decodes_what_the_codec_does() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let weights = dir.join("snac24_decoder.safetensors");
    if !weights.is_file() {
        eprintln!("skip: no {}", weights.display());
        return;
    }
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("snac24_ref.json")).unwrap()).unwrap();
    let levels: Vec<Vec<u32>> = v["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| {
            l.as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_u64().unwrap() as u32)
                .collect()
        })
        .collect();

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("snac.rlxp");
    let report = export_rlxp(&weights, &out, &ExportOptions::default()).unwrap();
    eprintln!("{report}");
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let codec = SnacDecoder::from_safetensors_path(&weights, rlx_runtime::Device::Cpu).unwrap();

    // All the reference's codes (6 coarse frames), then the first 4.
    for frames in [6, 4] {
        let codes: Vec<Vec<u32>> = levels
            .iter()
            .enumerate()
            .map(|(i, l)| l[..frames << i].to_vec())
            .collect();
        let inputs: Vec<Tensor> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let v: Vec<u8> = c.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect();
                Tensor::new(format!("codes_{i}"), rlx_ir::DType::I32, vec![c.len()], v)
            })
            .collect();
        let got = m.run(&inputs, &json!({})).unwrap()[0].to_f32().unwrap();
        let want = codec.decode(&HierarchicalCodes::new(codes), None).unwrap();
        assert_eq!(got.len(), want.len(), "{frames} frames");
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let peak = want.iter().fold(0f32, |a, x| a.max(x.abs()));
        eprintln!(
            "{frames} frames: {} samples, off by {worst} (peak {peak})",
            got.len()
        );
        assert!(
            worst < 1e-3 * peak.max(1e-3),
            "{frames} frames: off by {worst}"
        );
    }
}

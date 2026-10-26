// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported EnCodec decoder package, opened by rlx alone, decodes what
//! `EncodecCodec::decode` does from the real 24 kHz weights
//! (`tests/fixtures`): all the reference's codes, then fewer frames and
//! fewer codebooks (a lower bandwidth).
#![cfg(feature = "export")]

use rlx_core::export::ExportOptions;
use rlx_encodec::EncodecCodec;
use rlx_encodec::export::export_rlxp;
use rlx_runtime::model::Tensor;
use serde_json::json;

#[test]
fn an_exported_encodec_decodes_what_the_codec_does() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let weights = dir.join("encodec24.safetensors");
    if !weights.is_file() {
        eprintln!("skip: no {}", weights.display());
        return;
    }
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("encodec24_ref.json")).unwrap()).unwrap();
    let codes: Vec<Vec<u32>> = v["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_i64().unwrap() as u32)
                .collect()
        })
        .collect();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("encodec.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&weights, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let codec = EncodecCodec::from_safetensors_path(&weights, rlx_runtime::Device::Cpu).unwrap();
    let (n, t) = (codes.len(), codes[0].len());
    for (books, frames) in [(n, t), (n.min(4), t - 3)] {
        let cut: Vec<Vec<u32>> = codes[..books]
            .iter()
            .map(|r| r[..frames].to_vec())
            .collect();
        let bytes: Vec<u8> = cut
            .iter()
            .flatten()
            .flat_map(|&c| (c as i32).to_le_bytes())
            .collect();
        let input = Tensor::new("codes", rlx_ir::DType::I32, vec![books, frames], bytes);
        let got = m.run(&[input], &json!({})).unwrap()[0].to_f32().unwrap();
        let want = codec.decode(&cut).unwrap();
        assert_eq!(got.len(), want.len(), "{books}×{frames}");
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let peak = want.iter().fold(0f32, |a, x| a.max(x.abs()));
        eprintln!("{books} codebooks × {frames} frames: off by {worst} (peak {peak})");
        assert!(
            worst < 1e-4 * peak.max(1.0),
            "{books}×{frames}: off by {worst}"
        );
    }
}

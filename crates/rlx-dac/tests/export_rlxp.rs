// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! An exported DAC decoder package, opened by rlx alone, decodes what
//! `DacCodec::decode_codes` does from real weights (`RLX_DAC_DIR`, or the
//! workspace's `weights/tts/parler-dac`, the 44.1 kHz model): every
//! codebook, then fewer frames and fewer codebooks.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_dac::export::export_rlxp;
use rlx_dac::{DacCodec, DacCodes};
use rlx_runtime::model::Tensor;
use serde_json::json;

fn model_dir() -> Option<PathBuf> {
    let dir = rlx_ir::env::var_result("RLX_DAC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/parler-dac")
        });
    dir.join("model.safetensors").is_file().then_some(dir)
}

#[test]
fn an_exported_dac_decodes_what_the_codec_does() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: no DAC weights (set RLX_DAC_DIR)");
        return;
    };
    let codec = DacCodec::open(&dir).unwrap();
    let (n, t, bins) = (
        codec.config().n_codebooks,
        10,
        codec.config().codebook_size as u64,
    );
    // Codes from a fixed linear congruential sequence.
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let codes: Vec<Vec<u32>> = (0..n)
        .map(|_| {
            (0..t)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) % bins) as u32
                })
                .collect()
        })
        .collect();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("dac.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
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
        let want = codec
            .decode_codes(&DacCodes::from_quantizer_layout(cut))
            .unwrap();
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

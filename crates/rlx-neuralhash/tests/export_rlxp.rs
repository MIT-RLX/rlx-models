// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! NeuralHash exported as a package hashes what `NeuralHasher` does: the
//! shipping model (macOS's Vision.framework copy; skipped where it is
//! absent), images of several sizes and aspects, the package run by rlx
//! alone.

#![cfg(feature = "export")]

use std::path::Path;

use rlx_neuralhash::{HASH_BITS, NeuralHash, NeuralHasher};
use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use serde_json::json;

const RESOURCES: &str = "/System/Library/Frameworks/Vision.framework/Versions/A/Resources";

fn image(h: usize, w: usize) -> Vec<u8> {
    // A smooth gradient with some texture: real-image-like, so the hash's
    // scores are not all near zero.
    (0..h * w * 3)
        .map(|i| {
            let (p, c) = (i / 3, i % 3);
            let (y, x) = (p / w, p % w);
            let v = (x * 255 / w.max(1)) as i64 * (c as i64 + 1) / 3
                + (y * 200 / h.max(1)) as i64
                + ((x * 7 + y * 13 + c * 31) % 23) as i64;
            (v % 256) as u8
        })
        .collect()
}

#[test]
fn the_exported_neuralhash_hashes_what_the_runner_does() {
    let res = Path::new(RESOURCES);
    if !res.join(rlx_neuralhash::export::SEED_FILE).is_file() {
        eprintln!("skipped: no NeuralHash model in {RESOURCES}");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("neuralhash.rlxp");
    let report = rlx_neuralhash::export::export_rlxp(
        res,
        &out,
        &rlx_neuralhash::export::ExportOptions::default(),
    )
    .unwrap();
    eprintln!("{report}");

    let net = [
        "NeuralHashv3b_fp16-current.espresso.net",
        "NeuralHashv3b-current.espresso.net",
    ]
    .iter()
    .map(|n| res.join(n))
    .find(|p| p.is_file())
    .unwrap();
    let mut hasher = NeuralHasher::open_netfile(
        &net,
        res.join(rlx_neuralhash::export::SEED_FILE),
        Device::Cpu,
    )
    .unwrap();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (h, w) in [(360, 360), (480, 640), (123, 77), (900, 300)] {
        let rgb = image(h, w);
        let want_embedding = hasher
            .model_mut()
            .embed(&rlx_neuralhash::preprocess::from_rgb8(&rgb, w, h).unwrap())
            .unwrap();
        let want = hasher.hash_rgb8(&rgb, w, h).unwrap();
        let input = Tensor::new("image", rlx_ir::DType::U8, vec![h, w, 3], rgb);
        let outs = m.run(&[input], &json!({})).unwrap();
        let embedding = outs[0].to_f32().unwrap();
        let bits = outs[2].to_f32().unwrap();
        assert_eq!(bits.len(), HASH_BITS);
        let off = embedding
            .iter()
            .zip(&want_embedding)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let scale = want_embedding.iter().fold(0f32, |a, x| a.max(x.abs()));
        let bools: Vec<bool> = bits.iter().map(|&b| b > 0.5).collect();
        let got = NeuralHash::from_bits(&bools).unwrap();
        eprintln!(
            "{h}x{w}: {} vs {}, embedding off by {off} of {scale}",
            got.to_hex(),
            want.to_hex()
        );
        // The resize graph's coefficients are f32, Pillow's f64: a few
        // pixels in ten thousand land a level apart (none at 360 × 360).
        assert!(
            off <= 1e-4 * scale,
            "{h}x{w}: embedding off by {off} of {scale}"
        );
        assert_eq!(got, want, "{h}x{w}");
    }
}

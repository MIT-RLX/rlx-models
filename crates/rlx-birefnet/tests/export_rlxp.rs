// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! BiRefNet exported as a package gives the alpha `BiRefNet::alpha` does:
//! the real `BiRefNet_lite` (`weights/image/birefnet/BiRefNet_lite`, or
//! `RLX_BIREFNET_DIR`; skipped where absent), the network at 512, images of
//! two sizes.

#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_birefnet::BiRefNet;
use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use serde_json::json;

fn dir() -> Option<PathBuf> {
    let d = rlx_ir::env::var("RLX_BIREFNET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../weights/image/birefnet/BiRefNet_lite")
        });
    d.join("model.safetensors").is_file().then_some(d)
}

fn image(w: usize, h: usize) -> Vec<u8> {
    // A bright disc on a darker textured ground: something to segment.
    (0..w * h * 3)
        .map(|i| {
            let (p, c) = (i / 3, i % 3);
            let (y, x) = ((p / w) as f32 / h as f32, (p % w) as f32 / w as f32);
            let inside = (x - 0.5).powi(2) + (y - 0.5).powi(2) < 0.08;
            let base = if inside { 210 } else { 40 };
            (base + ((p * 7 + c * 31) % 29)) as u8
        })
        .collect()
}

#[test]
fn an_exported_birefnet_gives_the_runners_alpha() {
    let Some(dir) = dir() else {
        eprintln!("skipped: no BiRefNet checkpoint");
        return;
    };
    let side = 512;
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("birefnet.rlxp");
    let report = rlx_birefnet::export::export_at(&dir, side, &out, &Default::default()).unwrap();
    eprintln!("{report}");
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = BiRefNet::open(&dir, side, Device::Cpu).unwrap();
    for (w, h) in [(300, 200), (512, 512)] {
        let px = image(w, h);
        let img = image::RgbImage::from_raw(w as u32, h as u32, px.clone()).unwrap();
        let want = runner.alpha(&img).unwrap();
        let input = Tensor::new("image", rlx_ir::DType::U8, vec![h, w, 3], px);
        let outs = m.run(&[input], &json!({})).unwrap();
        assert_eq!(outs[0].shape, vec![h, w]);
        let got = outs[0].to_f32().unwrap();
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let mean = got
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
            / got.len() as f32;
        eprintln!("{w}x{h}: alpha off by {worst} (mean {mean})");
        assert!(worst < 1e-3, "{w}x{h}: alpha off by {worst}");
    }
}

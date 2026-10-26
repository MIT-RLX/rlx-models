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

#![cfg(feature = "export")]

//! An exported SigLIP 2 package, its preprocessing in a package of its
//! own, against `Siglip2Runner`: the preprocessing alone against
//! `siglip_normalize_nchw`, and the image embedding against
//! `encode_image`, at images shrunk, enlarged and kept at 224. The model
//! comes from `RLX_SIGLIP2_MODEL` (default `weights/siglip2-base-224`);
//! the test skips without it. `RLX_TEST_DEVICE` picks the device.

use image::{DynamicImage, RgbImage};
use rlx_runtime::model::Tensor;
use rlx_siglip2::export::{ExportOptions, export_rlxp};
use rlx_siglip2::{Siglip2Runner, siglip_normalize_nchw};
use serde_json::json;
use std::path::PathBuf;

/// A test pattern: gradients with a checkerboard, so both resize passes
/// see edges.
fn pattern(w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        let check = if (x / 9 + y / 7) % 2 == 0 { 60 } else { 0 };
        image::Rgb([
            ((x * 255 / w.max(1)) as u8).saturating_add(check),
            ((y * 255 / h.max(1)) as u8).saturating_sub(check / 2),
            ((x * 3 + y * 5) % 256) as u8,
        ])
    })
}

fn image_tensor(img: &RgbImage) -> Tensor {
    Tensor::new(
        "image",
        rlx_ir::DType::U8,
        vec![img.height() as usize, img.width() as usize, 3],
        img.as_raw().clone(),
    )
}

#[test]
fn an_exported_siglip2_matches_the_runner() {
    let dir = rlx_ir::env::var_os("RLX_SIGLIP2_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("weights/siglip2-base-224"));
    if !dir.join("model.safetensors").exists() {
        eprintln!("skip: no SigLIP 2 checkpoint at {}", dir.display());
        return;
    }
    let device = rlx_ir::env::var("RLX_TEST_DEVICE").unwrap_or_else(|| "cpu".into());
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("siglip2.rlxp");
    let pre = tmp.path().join("siglip-preprocess.rlxp");
    let opts = ExportOptions {
        frontend_package: Some(pre.clone()),
        ..ExportOptions::default()
    };
    let report = export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!(
        "{report:?}, {} MB + preprocessing {} KB",
        std::fs::metadata(&out).unwrap().len() / 1_000_000,
        std::fs::metadata(&pre).unwrap().len() / 1_000
    );

    let options = json!({"device": device});
    let mut front =
        rlx_runtime::model::open_source(pre.to_str().unwrap(), options.clone()).unwrap();
    let mut model = rlx_runtime::model::open_source(out.to_str().unwrap(), options).unwrap();
    assert_eq!(model.info().task, "image-feature-extraction");
    let mut runner = Siglip2Runner::builder().model_dir(&dir).build().unwrap();

    for (w, h) in [(300, 200), (100, 80), (224, 224), (517, 389)] {
        let img = pattern(w, h);
        // The preprocessing alone. The graph resamples in Pillow's fixed
        // point and the runner in floats; each stores 8 bits a pass, so a
        // value near a rounding edge lands a step (1/127.5) apart. Against
        // Pillow itself the graph is off at 1, 8, 54 and 0 of these values,
        // the runner at 227, 64, 113 and 0.
        let want = siglip_normalize_nchw(img.as_raw(), h as usize, w as usize, 224);
        let got = front.run(&[image_tensor(&img)], &json!({})).unwrap()[0]
            .to_f32()
            .unwrap();
        assert_eq!(got.len(), want.len());
        let off = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| (*a - *b).abs() > 1e-5)
            .count();
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1.0 / 127.5 + 1e-5, "{w}x{h}: off by {worst}");
        assert!(
            off * 500 <= got.len(),
            "{w}x{h}: {off} of {} values off",
            got.len()
        );

        // The whole package against the runner.
        let want = runner
            .encode_image(&DynamicImage::ImageRgb8(img.clone()))
            .unwrap();
        let got = model.run(&[image_tensor(&img)], &json!({})).unwrap()[0]
            .to_f32()
            .unwrap();
        assert_eq!(got.len(), want.len());
        let dot: f32 = got.iter().zip(&want).map(|(a, b)| a * b).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (norm(&got) * norm(&want));
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "{w}x{h}: preprocessing {off} values off; embedding cosine {cosine}, max diff {diff}"
        );
        assert!(cosine > 0.9999, "{w}x{h}: cosine {cosine}");
    }
}

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

//! An exported BioCLIP-2 package, opened by rlx alone, embeds images and
//! texts as `BioClip2Runner` does, from the real weights
//! (`RLX_BIOCLIP2_MODEL`, or the workspace's `weights/bioclip-2`): images
//! of three shapes, its preprocessing in the package, and two texts.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_bioclip2::BioClip2Runner;
use rlx_bioclip2::export::export_rlxp;
use rlx_core::export::ExportOptions;
use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use serde_json::json;

fn model_dir() -> Option<PathBuf> {
    let dir = rlx_ir::env::var_os("RLX_BIOCLIP2_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/bioclip-2")
        });
    dir.join("open_clip_model.safetensors")
        .is_file()
        .then_some(dir)
}

fn normalized(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn image(h: usize, w: usize) -> Vec<u8> {
    (0..h * w * 3)
        .map(|i| {
            let (y, x, c) = (i / (3 * w), (i / 3) % w, i % 3);
            ((x * 7 + y * 3 + c * 50 + (x * y) % 97) % 256) as u8
        })
        .collect()
}

#[test]
fn an_exported_bioclip2_embeds_what_the_runner_does() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: no BioCLIP-2 weights (set RLX_BIOCLIP2_MODEL)");
        return;
    };
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("bioclip2.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let source = out.to_str().unwrap();
    // One model, two entries: `run` an image goes to `image`, `embed` to
    // `text`.
    let mut m = rlx_runtime::model::open_source(source, json!({})).unwrap();
    let mut runner = BioClip2Runner::builder()
        .model_dir(&dir)
        .device(Device::Cpu)
        .build()
        .unwrap();

    for (h, w) in [(224, 224), (300, 451), (517, 389)] {
        let rgb = image(h, w);
        let input = Tensor::new("image", rlx_ir::DType::U8, vec![h, w, 3], rgb.clone());
        let got = m.run(&[input], &json!({})).unwrap()[0].to_f32().unwrap();
        let img = image::RgbImage::from_raw(w as u32, h as u32, rgb).unwrap();
        let want = runner
            .encode_image(&image::DynamicImage::ImageRgb8(img))
            .unwrap();
        assert_eq!(got.len(), want.len());
        let cos: f32 = normalized(&got)
            .iter()
            .zip(normalized(&want))
            .map(|(a, b)| a * b)
            .sum();
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let peak = want.iter().fold(0f32, |a, x| a.max(x.abs()));
        eprintln!("image {h}x{w}: off by {worst} (peak {peak}), cosine {cos}");
        // The preprocessing rounds a few pixels a level apart from the
        // runner's (both are Pillow's bicubic in floats).
        assert!(cos > 0.9999, "image {h}x{w}: cosine {cos}");
        assert!(worst < 1e-3 * peak, "image {h}x{w}: off by {worst}");
    }

    let texts = [
        "a photo of Danaus plexippus, the monarch butterfly.",
        "a photo of a red-tailed hawk perched on a branch",
    ];
    let got = m.embed(&texts, &json!({})).unwrap();
    for (t, g) in texts.iter().zip(&got) {
        let want = normalized(&runner.encode_text(t).unwrap());
        let worst = g
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        eprintln!("text {t:?}: off by {worst}");
        assert!(worst < 1e-4, "text {t:?}: off by {worst}");
    }
}

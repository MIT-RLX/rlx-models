// SPDX-License-Identifier: GPL-3.0-only
//! Detection/line-grouping/recognition breakdown for one image — to localize
//! where a wide line fragments. `OCR_MODEL_DIR=<dir> OCR_TEST_IMAGE=<png>
//! [OCR_DEVICE=metal] cargo test -p rlx-ocr --release --features metal
//! --test ocr_diag -- --ignored --nocapture`
#![cfg(feature = "rlx")]

use image::GenericImageView;
use rlx_ocr::{ImageSource, OcrEngine};
use rlx_runtime::Device;

#[test]
#[ignore]
fn diag_lines() {
    let dir = std::env::var("OCR_MODEL_DIR").expect("OCR_MODEL_DIR");
    let img_path = std::env::var("OCR_TEST_IMAGE").expect("OCR_TEST_IMAGE");
    let device = match std::env::var("OCR_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .as_str()
    {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" => Device::Gpu,
        _ => Device::Cpu,
    };
    let engine = OcrEngine::from_model_dir_on_device(&dir, device).unwrap();
    let img = image::open(&img_path).unwrap();
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    let input = engine
        .prepare_input(ImageSource::from_bytes(&rgb, (w, h)).unwrap())
        .unwrap();

    let words = engine.detect_words(&input).unwrap();
    eprintln!("[diag] device={device:?} image {w}x{h}");
    eprintln!("[diag] detected {} word boxes", words.len());
    let lines = engine.find_text_lines(&input, &words);
    eprintln!("[diag] grouped into {} lines", lines.len());
    for (i, line) in lines.iter().enumerate() {
        let centers: Vec<(i32, i32)> = line
            .iter()
            .map(|r| {
                let c = r.center();
                (c.x as i32, c.y as i32)
            })
            .collect();
        eprintln!(
            "[diag]   line {i}: {} words, centers={centers:?}",
            line.len()
        );
    }
    let texts = engine.recognize_text(&input, &lines).unwrap();
    for (i, t) in texts.iter().enumerate() {
        eprintln!("[diag]   rec {i}: {:?}", t.as_ref().map(|x| x.text()));
    }
}

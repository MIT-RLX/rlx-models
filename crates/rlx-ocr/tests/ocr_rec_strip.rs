// SPDX-License-Identifier: GPL-3.0-only
//! Dump the recognition input strip(s) for one image as PNGs, to see whether a
//! dropped character is actually present (and legible) in the model's input.
//! `OCR_MODEL_DIR=<dir> OCR_TEST_IMAGE=<png> cargo test -p rlx-ocr --release
//! --features metal --test ocr_rec_strip -- --ignored --nocapture`
#![cfg(feature = "rlx")]

use image::{GenericImageView, GrayImage, Luma};
use rlx_ocr::{ImageSource, OcrEngine};
use rlx_runtime::Device;
use rten_tensor::prelude::*;

#[test]
#[ignore]
fn dump_rec_strip() {
    let dir = std::env::var("OCR_MODEL_DIR").expect("OCR_MODEL_DIR");
    let img_path = std::env::var("OCR_TEST_IMAGE").expect("OCR_TEST_IMAGE");
    let engine = OcrEngine::from_model_dir_on_device(&dir, Device::Metal).unwrap();
    let dynimg = image::open(&img_path).unwrap();
    let (w, h) = dynimg.dimensions();
    let rgb = dynimg.to_rgb8().into_raw();
    let input = engine
        .prepare_input(ImageSource::from_bytes(&rgb, (w, h)).unwrap())
        .unwrap();
    let words = engine.detect_words(&input).unwrap();
    let lines = engine.find_text_lines(&input, &words);
    eprintln!("[strip] {} lines", lines.len());

    for (i, line) in lines.iter().enumerate() {
        let strip = engine.prepare_recognition_input(&input, line).unwrap();
        let [sh, sw] = strip.shape();
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in strip.iter() {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        let rng = (mx - mn).max(1e-6);
        let mut img = GrayImage::new(sw as u32, sh as u32);
        for y in 0..sh {
            for x in 0..sw {
                let v = strip[[y, x]];
                let p = (((v - mn) / rng) * 255.0).clamp(0.0, 255.0) as u8;
                img.put_pixel(x as u32, y as u32, Luma([p]));
            }
        }
        let out = format!("/tmp/rec_strip_{i}.png");
        img.save(&out).unwrap();
        eprintln!("[strip] line {i}: {sh}x{sw} -> {out} (min {mn:.3} max {mx:.3})");
    }
}

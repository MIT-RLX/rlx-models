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
//! Per-stage latency profiler for the native OCR pipeline (warm cache).
//!
//! `OCR_MODEL_DIR=<dir> OCR_TEST_IMAGE=<png> [OCR_DEVICE=cpu|metal|mlx|gpu] \
//!    cargo test -p rlx-ocr --release --features metal --test ocr_profile -- --ignored --nocapture`
#![cfg(feature = "rlx")]

use image::GenericImageView;
use rlx_ocr::{ImageSource, OcrEngine};
use rlx_runtime::Device;
use std::time::Instant;

fn device_from_env() -> Device {
    match std::env::var("OCR_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .as_str()
    {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" => Device::Gpu,
        _ => Device::Cpu,
    }
}

fn stage(label: &str, mut f: impl FnMut()) {
    let t = Instant::now();
    f();
    eprintln!(
        "[profile] {label:<34}: {:>9.1} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );
}

#[test]
#[ignore]
fn profile_native_ocr() {
    let dir = std::env::var("OCR_MODEL_DIR").expect("set OCR_MODEL_DIR");
    let img_path = std::env::var("OCR_TEST_IMAGE").expect("set OCR_TEST_IMAGE");
    let device = device_from_env();
    eprintln!("[profile] device = {device:?}");
    let engine = OcrEngine::from_model_dir_on_device(&dir, device).expect("load engine");
    let img = image::open(&img_path).expect("open image");
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    let input = engine
        .prepare_input(ImageSource::from_bytes(&rgb, (w, h)).unwrap())
        .unwrap();

    // Warm the compiled-graph caches (detection + every recognition width bucket).
    stage("warmup get_text (cold compile+run)", || {
        engine.get_text(&input).unwrap();
    });

    let words = engine.detect_words(&input).unwrap();
    let lines = engine.find_text_lines(&input, &words);
    eprintln!(
        "[profile] image {w}x{h} | words={} | lines={}",
        words.len(),
        lines.len()
    );

    stage("detect_text_pixels (U-Net graph)", || {
        engine.detect_text_pixels(&input).unwrap();
    });
    stage("detect_words (graph + postprocess)", || {
        engine.detect_words(&input).unwrap();
    });
    stage("recognize_text (all lines)", || {
        engine.recognize_text(&input, &lines).unwrap();
    });
    stage("get_text (end-to-end)", || {
        engine.get_text(&input).unwrap();
    });
}

// SPDX-License-Identifier: GPL-3.0-only
//! Detection-mask divergence CPU vs GPU — quantifies the f32 difference that
//! fragments wide lines on Metal/wgpu. `OCR_MODEL_DIR=<dir> OCR_TEST_IMAGE=<png>
//! cargo test -p rlx-ocr --release --features metal --test ocr_mask_diff
//! -- --ignored --nocapture`
#![cfg(feature = "rlx")]

use image::GenericImageView;
use rlx_ocr::{ImageSource, OcrEngine};
use rlx_runtime::Device;
use rten_tensor::prelude::*;

fn mask(dir: &str, device: Device, rgb: &[u8], w: u32, h: u32) -> Vec<f32> {
    let engine = OcrEngine::from_model_dir_on_device(dir, device).unwrap();
    let input = engine
        .prepare_input(ImageSource::from_bytes(rgb, (w, h)).unwrap())
        .unwrap();
    engine
        .detect_text_pixels(&input)
        .unwrap()
        .iter()
        .copied()
        .collect()
}

#[test]
#[ignore]
fn mask_cpu_vs_gpu() {
    let dir = std::env::var("OCR_MODEL_DIR").expect("OCR_MODEL_DIR");
    let img_path = std::env::var("OCR_TEST_IMAGE").expect("OCR_TEST_IMAGE");
    let device = match std::env::var("OCR_GPU")
        .unwrap_or_else(|_| "metal".into())
        .as_str()
    {
        "mlx" => Device::Mlx,
        "gpu" => Device::Gpu,
        _ => Device::Metal,
    };
    let img = image::open(&img_path).unwrap();
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();

    let cpu = mask(&dir, Device::Cpu, &rgb, w, h);

    // Warm-arena test: run detection TWICE on ONE engine. If run-2 differs from
    // run-1, an op reads an arena region it didn't write this run (read-before-
    // write); zero-init only neutralizes run-1's stale data, not run-2's.
    let eng = OcrEngine::from_model_dir_on_device(&dir, device).unwrap();
    let inp = eng
        .prepare_input(ImageSource::from_bytes(&rgb, (w, h)).unwrap())
        .unwrap();
    let m1: Vec<f32> = eng
        .detect_text_pixels(&inp)
        .unwrap()
        .iter()
        .copied()
        .collect();
    let m2: Vec<f32> = eng
        .detect_text_pixels(&inp)
        .unwrap()
        .iter()
        .copied()
        .collect();
    let r12 = m1
        .iter()
        .zip(&m2)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let d1 = cpu
        .iter()
        .zip(&m1)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let d2 = cpu
        .iter()
        .zip(&m2)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[mask] WARM-ARENA: run1-vs-run2={r12:.5}  run1-vs-cpu={d1:.5}  run2-vs-cpu={d2:.5}");

    let gpu = mask(&dir, device, &rgb, w, h);
    assert_eq!(cpu.len(), gpu.len());

    let thr = 0.2f32; // typical text_threshold
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut flips = 0usize; // pixels that cross the threshold differently
    let (mut cpu_on, mut gpu_on) = (0usize, 0usize);
    for (&c, &g) in cpu.iter().zip(&gpu) {
        let d = (c - g).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        if (c >= thr) != (g >= thr) {
            flips += 1;
        }
        cpu_on += (c >= thr) as usize;
        gpu_on += (g >= thr) as usize;
    }
    eprintln!("[mask] {device:?} vs Cpu | n={} ", cpu.len());
    eprintln!("[mask] max abs diff   = {max_abs:.5}");
    eprintln!("[mask] mean abs diff  = {:.6}", sum_abs / cpu.len() as f64);
    eprintln!(
        "[mask] threshold flips= {flips} ({:.3}% of pixels)",
        flips as f64 / cpu.len() as f64 * 100.0
    );
    eprintln!("[mask] pixels >= {thr}: cpu={cpu_on} gpu={gpu_on}");
}

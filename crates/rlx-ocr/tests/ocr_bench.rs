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
//! Real-image OCR benchmark across backends: correctness, speed/latency,
//! precision/accuracy (CER/WER vs ground truth). One backend may fail without
//! aborting the others, so this doubles as a per-backend health check.
//!
//! `OCR_MODEL_DIR=<dir> OCR_BENCH_MANIFEST=<tsv> OCR_DEVICES=cpu,metal,mlx,gpu \
//!    [OCR_BENCH_LIMIT=4] cargo test -p rlx-ocr --release \
//!    --features metal,mlx,gpu --test ocr_bench -- --ignored --nocapture`
//! Peak RSS: wrap the built binary in `/usr/bin/time -l`.
#![cfg(feature = "rlx")]

use image::GenericImageView;
use rlx_ocr::{ImageSource, OcrEngine};
use rlx_runtime::Device;
use std::time::Instant;

fn parse_device(s: &str) -> Option<Device> {
    Some(match s.trim() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        _ => return None,
    })
}

fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ai) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, bj) in b.iter().enumerate() {
            let cost = if ai == bj { 0 } else { 1 };
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn cer(r: &str, h: &str) -> f64 {
    let (rc, hc): (Vec<char>, Vec<char>) = (r.chars().collect(), h.chars().collect());
    if rc.is_empty() {
        return if hc.is_empty() { 0.0 } else { 1.0 };
    }
    edit_distance(&rc, &hc) as f64 / rc.len() as f64
}
fn wer(r: &str, h: &str) -> f64 {
    let (rw, hw): (Vec<&str>, Vec<&str>) = (
        r.split_whitespace().collect(),
        h.split_whitespace().collect(),
    );
    if rw.is_empty() {
        return if hw.is_empty() { 0.0 } else { 1.0 };
    }
    edit_distance(&rw, &hw) as f64 / rw.len() as f64
}
fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() as f64 * p) as usize).min(s.len().saturating_sub(1))]
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

type Img = (Vec<u8>, u32, u32, String);

fn run_device(dir: &str, device: Device, inputs: &[Img]) {
    eprintln!("\n========== device = {device:?} ==========");
    let engine = match OcrEngine::from_model_dir_on_device(dir, device) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  ✗ FAILED to construct engine on {device:?}: {e:#}");
            return;
        }
    };
    let (mut lat, mut cers, mut wers, mut exact, mut errors) =
        (vec![], vec![], vec![], 0usize, 0usize);
    for (rgb, w, h, truth) in inputs {
        let prep = || {
            engine
                .prepare_input(ImageSource::from_bytes(rgb, (*w, *h)).unwrap())
                .unwrap()
        };
        // warm this image's width bucket, then time + check.
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let input = prep();
            let _ = engine.get_text(&input)?;
            let input = prep();
            let t = Instant::now();
            let out = engine.get_text(&input)?;
            Ok::<_, anyhow::Error>((out, t.elapsed().as_secs_f64() * 1000.0))
        }));
        match res {
            Ok(Ok((out, ms))) => {
                let got = out.replace('\n', " ").trim().to_string();
                let (c, wr) = (cer(truth, &got), wer(truth, &got));
                if got == *truth {
                    exact += 1;
                }
                lat.push(ms);
                cers.push(c);
                wers.push(wr);
                let ok = if got == *truth { "✓" } else { "✗" };
                eprintln!("  {ok} {ms:7.1}ms CER={c:.3} | {truth:?} -> {got:?}");
            }
            Ok(Err(e)) => {
                errors += 1;
                eprintln!("  ✗ error: {e:#} | {truth:?}");
            }
            Err(_) => {
                errors += 1;
                eprintln!("  ✗ PANIC during get_text | {truth:?}");
            }
        }
    }
    let n = inputs.len();
    eprintln!("  ── {device:?}: exact {exact}/{n}, errors {errors}");
    if !lat.is_empty() {
        eprintln!(
            "     char-acc {:.2}%  word-acc {:.2}%  lat mean/p50/p95 {:.1}/{:.1}/{:.1} ms  {:.1} img/s",
            (1.0 - mean(&cers)) * 100.0,
            (1.0 - mean(&wers)) * 100.0,
            mean(&lat),
            pct(&lat, 0.5),
            pct(&lat, 0.95),
            1000.0 / mean(&lat).max(0.001),
        );
    }
}

#[test]
#[ignore]
fn bench_real_images() {
    let dir = std::env::var("OCR_MODEL_DIR").expect("set OCR_MODEL_DIR");
    let manifest = std::env::var("OCR_BENCH_MANIFEST").expect("set OCR_BENCH_MANIFEST");
    let limit: usize = std::env::var("OCR_BENCH_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let devices: Vec<Device> = std::env::var("OCR_DEVICES")
        .unwrap_or_else(|_| "cpu".into())
        .split(',')
        .filter_map(parse_device)
        .collect();

    let inputs: Vec<Img> = std::fs::read_to_string(&manifest)
        .expect("read manifest")
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .take(limit)
        .map(|(path, truth)| {
            let img = image::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
            let (w, h) = img.dimensions();
            (img.to_rgb8().into_raw(), w, h, truth.to_string())
        })
        .collect();
    eprintln!(
        "OCR benchmark: {} images, devices {devices:?}",
        inputs.len()
    );
    for device in devices {
        run_device(&dir, device, &inputs);
    }
}

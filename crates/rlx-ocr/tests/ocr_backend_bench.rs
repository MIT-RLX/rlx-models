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

//! Wall-clock OCR benchmarks: native RLX (`safetensors` + compiled graphs).
//!
//! ```bash
//! OCR_MODEL_DIR=~/.cache/ocrs OCR_TEST_IMAGE=/path/to/page.png \
//!   cargo test -p rlx-ocr --test ocr_backend_bench --features "rlx,convert-rten" --release -- --nocapture
//!
//! OCR_PARITY_DOWNLOAD=1 cargo test -p rlx-ocr --test ocr_backend_bench --features "rlx,convert-rten" --release ocr_bench_report -- --nocapture
//! ```

use anyhow::{Context, Result};
use image::RgbImage;
use image::imageops::FilterType;
use rlx_cli::parse_standard_device;
use rlx_ocr::config::DetectionParams;
use rlx_ocr::{
    ImageSource, OcrEngine, OcrEngineParams, RlxTextDetector, input_image, resolve_model_dir,
};
use rlx_runtime::Device;
use rten_tensor::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

#[path = "assets.rs"]
mod assets;
#[path = "env.rs"]
mod bench_env;

const WARMUP: usize = 2;
const ITERS: usize = 5;
const DEFAULT_BATCH_SIZES: &[usize] = &[1, 2, 4];
const DEFAULT_IMAGE_SIZES: &[(u32, u32)] = &[
    (400, 300),
    (800, 600),
    (1200, 900),
    (2000, 2000),
    (2320, 776),
    (3200, 1200),
];

fn bench_batch_sizes() -> Vec<usize> {
    if let Some(s) = bench_env::env_var("OCR_BENCH_BATCH", "OCRS_BENCH_BATCH") {
        let v: Vec<usize> = s
            .split(',')
            .filter_map(|p| p.trim().parse().ok())
            .filter(|&b| b >= 1)
            .collect();
        if !v.is_empty() {
            return v;
        }
    }
    DEFAULT_BATCH_SIZES.to_vec()
}

fn parse_size_token(token: &str) -> Option<(u32, u32)> {
    let (w, h) = token.split_once(['x', 'X'])?;
    let width: u32 = w.trim().parse().ok()?;
    let height: u32 = h.trim().parse().ok()?;
    (width > 0 && height > 0).then_some((width, height))
}

fn bench_image_sizes() -> Vec<(u32, u32)> {
    if let Some(s) = bench_env::env_var("OCR_BENCH_SIZES", "OCRS_BENCH_SIZES") {
        let v: Vec<(u32, u32)> = s
            .split(',')
            .filter_map(|p| parse_size_token(p.trim()))
            .collect();
        if !v.is_empty() {
            return v;
        }
    }
    DEFAULT_IMAGE_SIZES.to_vec()
}

fn effective_detection_hw(
    img_h: usize,
    img_w: usize,
    model_h: usize,
    model_w: usize,
) -> (usize, usize) {
    let pad_h = img_h.max(model_h);
    let pad_w = img_w.max(model_w);
    if pad_h == model_h && pad_w == model_w {
        (pad_h, pad_w)
    } else {
        (model_h, model_w)
    }
}

fn bench_device() -> Result<Device> {
    if let Some(s) = bench_env::env_var("OCR_DEVICE", "OCRS_DEVICE") {
        return parse_standard_device("ocr", &s);
    }
    Ok(Device::Cpu)
}

fn model_dir() -> Option<PathBuf> {
    if let Some(dir) = bench_env::env_var("OCR_MODEL_DIR", "OCRS_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }
    if bench_env::env_is_1("OCR_PARITY_DOWNLOAD", "OCRS_PARITY_DOWNLOAD") {
        let dir = assets::default_model_dir();
        if assets::ensure_safetensors_exports(&dir).is_ok() {
            return Some(dir);
        }
    }
    None
}

fn test_image_path() -> Option<PathBuf> {
    if let Some(p) = bench_env::env_var("OCR_TEST_IMAGE", "OCRS_TEST_IMAGE") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    if bench_env::env_is_1("OCR_PARITY_DOWNLOAD", "OCRS_PARITY_DOWNLOAD") {
        let path = assets::default_test_image();
        if assets::ensure_test_image(&path).is_ok() {
            return Some(path);
        }
    }
    None
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_ms<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    median(samples)
}

struct EngineBundle {
    engine: OcrEngine,
    model_h: usize,
    model_w: usize,
}

struct BenchAssets {
    input: rlx_ocr::OcrInput,
    tensor_h: usize,
    tensor_w: usize,
    width: u32,
    height: u32,
}

fn load_engine_bundle() -> Result<EngineBundle> {
    let dir = model_dir().context("model_dir")?;
    let device = bench_device().context("bench_device")?;
    assets::ensure_safetensors_exports(&dir)?;
    let (det, rec) = resolve_model_dir(&dir).context("resolve_model_dir")?;
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(det.clone()),
        recognition_model: Some(rec.clone()),
        device,
        ..Default::default()
    })
    .context("OcrEngine::new")?;
    let detector = RlxTextDetector::from_path(&det, DetectionParams::default(), device)
        .context("RlxTextDetector")?;
    let (model_h, model_w) = detector
        .fixed_input_hw()
        .context("detection model must have fixed H×W input")?;
    Ok(EngineBundle {
        engine,
        model_h,
        model_w,
    })
}

fn prepare_assets(
    bundle: &EngineBundle,
    rgb: &[u8],
    width: u32,
    height: u32,
) -> Result<BenchAssets> {
    let source = ImageSource::from_bytes(rgb, (width, height)).context("ImageSource")?;
    let input = bundle
        .engine
        .prepare_input(source)
        .context("prepare_input")?;
    let view = input_image(&input);
    let img_h = view.size(1);
    let img_w = view.size(2);
    let (tensor_h, tensor_w) = effective_detection_hw(img_h, img_w, bundle.model_h, bundle.model_w);
    Ok(BenchAssets {
        input,
        tensor_h,
        tensor_w,
        width,
        height,
    })
}

fn load_base_image() -> Result<RgbImage> {
    let image_path = test_image_path().context("test_image_path")?;
    Ok(image::open(&image_path)
        .with_context(|| format!("open {:?}", image_path))?
        .into_rgb8())
}

fn resize_page(base: &RgbImage, width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let out = image::imageops::resize(base, width, height, FilterType::Triangle);
    let (w, h) = out.dimensions();
    (out.into_raw(), w, h)
}

fn load_assets() -> Result<BenchAssets> {
    let bundle = load_engine_bundle()?;
    let base = load_base_image()?;
    let (w, h) = base.dimensions();
    let rgb = base.into_raw();
    prepare_assets(&bundle, &rgb, w, h)
}

fn bench_detection_ms(engine: &OcrEngine, assets: &BenchAssets) -> f64 {
    bench_ms(|| {
        let _ = engine.detect_text_pixels(&assets.input).unwrap();
    })
}

fn bench_get_text_ms(engine: &OcrEngine, assets: &BenchAssets) -> f64 {
    bench_ms(|| {
        let _ = engine.get_text(&assets.input).unwrap();
    })
}

fn bench_get_text_sequential_pages(engine: &OcrEngine, assets: &BenchAssets, pages: usize) -> f64 {
    bench_ms(|| {
        for _ in 0..pages {
            let _ = engine.get_text(&assets.input).unwrap();
        }
    })
}

fn report_size_case(bundle: &EngineBundle, assets: &BenchAssets) -> Result<()> {
    eprintln!(
        "size {}x{} → detection tensor {}x{}",
        assets.width, assets.height, assets.tensor_h, assets.tensor_w
    );
    let det_ms = bench_detection_ms(&bundle.engine, assets);
    let full_ms = bench_get_text_ms(&bundle.engine, assets);
    eprintln!("  rlx detection={det_ms:.1}ms get_text={full_ms:.1}ms");
    Ok(())
}

#[test]
fn ocr_bench_report() -> Result<()> {
    let bundle = match load_engine_bundle() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip ocr_bench_report: {e:#}");
            return Ok(());
        }
    };
    let assets = load_assets()?;
    let batches = bench_batch_sizes();

    let device = bench_device().unwrap_or(Device::Cpu);
    eprintln!(
        "rlx-ocr bench (native) device={device:?} model canvas {}x{}",
        bundle.model_h, bundle.model_w
    );
    eprintln!("--- RLX sequential multi-page get_text ---");
    for &pages in &batches {
        if pages == 1 {
            continue;
        }
        let total_ms = bench_get_text_sequential_pages(&bundle.engine, &assets, pages);
        eprintln!(
            "  pages={pages} total_median={total_ms:.1}ms ({:.1}ms/page)",
            total_ms / pages as f64
        );
    }

    report_size_case(&bundle, &assets)
}

#[test]
fn ocr_bench_image_sizes() -> Result<()> {
    let bundle = match load_engine_bundle() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip ocr_bench_image_sizes: {e:#}");
            return Ok(());
        }
    };
    let base = match load_base_image() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip ocr_bench_image_sizes: {e:#}");
            return Ok(());
        }
    };
    let sizes = bench_image_sizes();
    eprintln!(
        "ocr image-size sweep ({} targets, model {}x{})",
        sizes.len(),
        bundle.model_h,
        bundle.model_w
    );

    for &(target_w, target_h) in &sizes {
        eprintln!("---");
        let (rgb, w, h) = resize_page(&base, target_w, target_h);
        let assets = prepare_assets(&bundle, &rgb, w, h)?;
        report_size_case(&bundle, &assets)?;
    }
    Ok(())
}

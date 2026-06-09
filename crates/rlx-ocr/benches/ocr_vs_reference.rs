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

//! Criterion bench: native `rlx-ocr` vs upstream [`ocrs`](https://crates.io/crates/ocrs) 0.12.x (RTen baseline).
//!
//! ```bash
//! OCR_PARITY_DOWNLOAD=1 cargo bench -p rlx-ocr --features parity-ocrs,convert-rten --bench ocr_vs_reference --release
//! ```

#![cfg(feature = "parity-ocrs")]

use anyhow::{Context, Result};
use criterion::{Criterion, criterion_group, criterion_main};
use ocrs::{ImageSource as RefImageSource, OcrEngine as RefEngine, OcrEngineParams as RefParams};
use rlx_ocr::{ImageSource, OcrEngine, resolve_model_dir};
use rten::Model;
use std::hint::black_box;
use std::path::{Path, PathBuf};

#[path = "../tests/assets.rs"]
mod assets;
#[path = "../tests/env.rs"]
mod bench_env;

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

fn rten_paths(dir: &Path) -> Result<(PathBuf, PathBuf)> {
    assets::ensure_rten_checkpoints(dir)?;
    Ok((
        dir.join(rlx_ocr::HF_DETECTION_RTEN),
        dir.join(rlx_ocr::HF_RECOGNITION_RTEN),
    ))
}

struct BenchSetup {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    reference: RefEngine,
    rlx: OcrEngine,
    ref_input: ocrs::OcrInput,
    rlx_input: rlx_ocr::OcrInput,
}

fn build_setup() -> Option<BenchSetup> {
    let dir = model_dir()?;
    let image_path = test_image_path()?;
    assets::ensure_safetensors_exports(&dir).ok()?;
    let (det_st, rec_st) = resolve_model_dir(&dir).ok()?;
    let (det_rten, rec_rten) = rten_paths(&dir).ok()?;
    let det_model = Model::load_file(&det_rten).ok()?;
    let rec_model = Model::load_file(&rec_rten).ok()?;
    let reference = RefEngine::new(RefParams {
        detection_model: Some(det_model),
        recognition_model: Some(rec_model),
        ..Default::default()
    })
    .ok()?;
    let rlx = OcrEngine::from_paths(&det_st, &rec_st).ok()?;
    let img = image::open(&image_path).ok()?.into_rgb8();
    let (width, height) = img.dimensions();
    let rgb = img.into_raw();
    let ref_src = RefImageSource::from_bytes(&rgb, (width, height)).ok()?;
    let rlx_src = ImageSource::from_bytes(&rgb, (width, height)).ok()?;
    let ref_input = reference.prepare_input(ref_src).ok()?;
    let rlx_input = rlx.prepare_input(rlx_src).ok()?;
    Some(BenchSetup {
        rgb,
        width,
        height,
        reference,
        rlx,
        ref_input,
        rlx_input,
    })
}

fn bench(c: &mut Criterion) {
    let Some(setup) = build_setup() else {
        eprintln!(
            "skipping ocr_vs_reference — set OCR_MODEL_DIR + OCR_TEST_IMAGE or OCR_PARITY_DOWNLOAD=1"
        );
        return;
    };

    eprintln!(
        "ocr bench image {}x{} ({} bytes rgb)",
        setup.width,
        setup.height,
        setup.rgb.len()
    );

    let mut group = c.benchmark_group("ocr_why_rust");
    group.sample_size(20);

    group.bench_function("reference_preprocess", |b| {
        let (w, h) = (setup.width, setup.height);
        let rgb = &setup.rgb;
        b.iter(|| {
            let src = RefImageSource::from_bytes(black_box(rgb), black_box((w, h))).unwrap();
            let input = setup.reference.prepare_input(src).unwrap();
            black_box(input);
        });
    });

    group.bench_function("rlx_preprocess", |b| {
        let (w, h) = (setup.width, setup.height);
        let rgb = &setup.rgb;
        b.iter(|| {
            let src = ImageSource::from_bytes(black_box(rgb), black_box((w, h))).unwrap();
            let input = setup.rlx.prepare_input(src).unwrap();
            black_box(input);
        });
    });

    group.bench_function("reference_detection_mask", |b| {
        b.iter(|| {
            let mask = setup
                .reference
                .detect_text_pixels(black_box(&setup.ref_input))
                .unwrap();
            black_box(mask);
        });
    });

    group.bench_function("rlx_detection_mask", |b| {
        b.iter(|| {
            let mask = setup
                .rlx
                .detect_text_pixels(black_box(&setup.rlx_input))
                .unwrap();
            black_box(mask);
        });
    });

    group.bench_function("reference_get_text", |b| {
        b.iter(|| {
            let text = setup
                .reference
                .get_text(black_box(&setup.ref_input))
                .unwrap();
            black_box(text);
        });
    });

    group.bench_function("rlx_get_text", |b| {
        b.iter(|| {
            let text = setup.rlx.get_text(black_box(&setup.rlx_input)).unwrap();
            black_box(text);
        });
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

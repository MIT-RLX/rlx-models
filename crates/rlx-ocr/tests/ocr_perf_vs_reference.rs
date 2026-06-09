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

//! End-to-end latency gate: native `rlx-ocr` must beat upstream `ocrs` on the same machine.
//!
//! Off by default. Enable with `OCR_PERF_GATE=1` and model paths (or download):
//!
//! ```bash
//! OCR_PERF_GATE=1 OCR_PARITY_DOWNLOAD=1 \
//!   cargo test -p rlx-ocr --test ocr_perf_vs_reference --features parity-ocrs,convert-rten --release -- --nocapture
//! ```

#![cfg(feature = "parity-ocrs")]

use anyhow::{Context, Result};
use image::GenericImageView;
use ocrs::{
    DecodeMethod as RefDecodeMethod, ImageSource as RefImageSource, OcrEngine as RefEngine,
    OcrEngineParams as RefParams,
};
use rlx_ocr::{ImageSource, OcrEngine, resolve_model_dir};
use rten::Model;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[path = "assets.rs"]
mod assets;
#[path = "env.rs"]
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

/// `rlx` median must be ≤ this fraction of `ocrs` median (0.97 → at least 3% faster on CPU).
const MAX_RLX_VS_REF_RATIO_DEFAULT: f64 = 0.97;
const MAX_RLX_VS_REF_RATIO_STRICT: f64 = 0.95;

fn max_rlx_vs_ref_ratio() -> f64 {
    if std::env::var("OCR_PERF_STRICT").ok().as_deref() == Some("1") {
        MAX_RLX_VS_REF_RATIO_STRICT
    } else {
        MAX_RLX_VS_REF_RATIO_DEFAULT
    }
}
const WARMUP_RUNS: usize = 2;
const TIMED_RUNS: usize = 5;

fn median_ms(samples: &[Duration]) -> f64 {
    let mut ms: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ms[ms.len() / 2]
}

fn time_get_text(engine: &OcrEngine, bytes: &[u8], w: u32, h: u32) -> Result<Duration> {
    let t0 = Instant::now();
    let src = ImageSource::from_bytes(bytes, (w, h))?;
    let input = engine.prepare_input(src)?;
    let _ = engine.get_text(&input)?;
    Ok(t0.elapsed())
}

fn time_ref_get_text(engine: &RefEngine, bytes: &[u8], w: u32, h: u32) -> Result<Duration> {
    let t0 = Instant::now();
    let src = RefImageSource::from_bytes(bytes, (w, h))?;
    let input = engine.prepare_input(src)?;
    let _ = engine.get_text(&input)?;
    Ok(t0.elapsed())
}

#[test]
fn rlx_ocr_faster_than_upstream_on_get_text() -> Result<()> {
    if std::env::var("OCR_PERF_GATE").ok().as_deref() != Some("1") {
        eprintln!("skip rlx_ocr_faster_than_upstream: set OCR_PERF_GATE=1");
        return Ok(());
    }
    let Some(dir) = model_dir() else {
        eprintln!("skip: set OCR_MODEL_DIR or OCR_PARITY_DOWNLOAD=1");
        return Ok(());
    };
    let Some(image_path) = test_image_path() else {
        eprintln!("skip: set OCR_TEST_IMAGE or OCR_PARITY_DOWNLOAD=1");
        return Ok(());
    };

    assets::ensure_safetensors_exports(&dir)?;
    assets::ensure_rten_checkpoints(&dir)?;
    let (det_st, rec_st) = resolve_model_dir(&dir)?;
    let det_rten = dir.join(rlx_ocr::HF_DETECTION_RTEN);
    let rec_rten = dir.join(rlx_ocr::HF_RECOGNITION_RTEN);
    anyhow::ensure!(det_rten.is_file(), "missing {:?}", det_rten);
    anyhow::ensure!(rec_rten.is_file(), "missing {:?}", rec_rten);

    let det_model = Model::load_file(&det_rten).with_context(|| format!("load {det_rten:?}"))?;
    let rec_model = Model::load_file(&rec_rten).with_context(|| format!("load {rec_rten:?}"))?;
    let rlx = OcrEngine::from_paths(&det_st, &rec_st)?;
    let reference = RefEngine::new(RefParams {
        detection_model: Some(det_model),
        recognition_model: Some(rec_model),
        decode_method: RefDecodeMethod::Greedy,
        ..Default::default()
    })?;

    let img = image::open(&image_path).with_context(|| format!("open {:?}", image_path))?;
    let (w, h) = img.dimensions();
    let bytes = img.to_rgb8().into_raw();

    for _ in 0..WARMUP_RUNS {
        let _ = time_get_text(&rlx, &bytes, w, h)?;
        let _ = time_ref_get_text(&reference, &bytes, w, h)?;
    }

    let mut rlx_samples = Vec::with_capacity(TIMED_RUNS);
    let mut ref_samples = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        rlx_samples.push(time_get_text(&rlx, &bytes, w, h)?);
        ref_samples.push(time_ref_get_text(&reference, &bytes, w, h)?);
    }

    let rlx_med = median_ms(&rlx_samples);
    let ref_med = median_ms(&ref_samples);
    let max_ratio = max_rlx_vs_ref_ratio();
    let ratio = rlx_med / ref_med.max(1e-9);

    eprintln!(
        "ocr perf gate ({image_path:?}): rlx get_text median {rlx_med:.1} ms, ocrs {ref_med:.1} ms, ratio {ratio:.3} (need ≤ {max_ratio})",
    );
    assert!(
        ratio <= max_ratio,
        "rlx-ocr get_text {rlx_med:.1} ms is not faster than ocrs {ref_med:.1} ms (ratio {ratio:.3}, need ≤ {max_ratio}). \
         For ≥5% set OCR_PERF_STRICT=1."
    );
    Ok(())
}

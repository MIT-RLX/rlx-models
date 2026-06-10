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

//! Shared helpers for OCR backend quick-check tests (env-gated real weights).

use anyhow::{Context, Result};
use rlx_ocr::{
    DetectionParams, ImageSource, OcrEngine, RlxTextDetector, RlxTextRecognizer, resolve_model_dir,
};
use rlx_runtime::{Device, is_available};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use std::path::PathBuf;

#[path = "assets.rs"]
mod assets;
#[path = "env.rs"]
mod bench_env;

const GREY_H: usize = 64;
const GREY_W: usize = 96;
const REC_W: usize = 200;

pub fn model_dir() -> Option<PathBuf> {
    if let Some(dir) = bench_env::env_var("OCR_MODEL_DIR", "OCRS_MODEL_DIR") {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }
    if bench_env::env_is_1("OCR_PARITY_DOWNLOAD", "OCRS_PARITY_DOWNLOAD") {
        let dir = assets::default_model_dir();
        if assets::ensure_safetensors_exports(&dir).is_ok() {
            return Some(dir);
        }
    }
    None
}

fn load_weight_paths() -> Result<(PathBuf, PathBuf)> {
    let dir =
        model_dir().context("set OCR_MODEL_DIR to a directory with ocr-*-full.safetensors")?;
    let (det, rec) = resolve_model_dir(&dir)?;
    Ok((det, rec))
}

fn is_skip_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}").to_lowercase();
    msg.contains("unsupported")
        || msg.contains("not supported")
        || msg.contains("not lowerable")
        || msg.contains("not available")
        || msg.contains("no backend")
        || msg.contains("doesn't claim support")
        || msg.contains("convtranspose2d")
}

fn is_skip_panic(payload: &(dyn std::any::Any + Send)) -> bool {
    let msg = if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        return false;
    };
    let lower = msg.to_lowercase();
    lower.contains("doesn't claim support")
        || lower.contains("not yet lowered")
        || lower.contains("convtranspose2d")
}

pub fn run_detection_forward(device: Device) -> Result<()> {
    let (det, _) = load_weight_paths()?;
    let detector = RlxTextDetector::from_path(&det, DetectionParams::default(), device)?;
    let grey: Vec<f32> = vec![0.82; GREY_H * GREY_W];
    let image = NdTensor::from_data([1, GREY_H, GREY_W], grey);
    let mask = detector.detect_text_pixels(image.view())?;
    assert_eq!(mask.shape(), [GREY_H, GREY_W]);
    assert!(mask.iter().all(|v| v.is_finite()));
    Ok(())
}

pub fn run_recognition_logits(device: Device) -> Result<()> {
    let (_, rec) = load_weight_paths()?;
    let recognizer = RlxTextRecognizer::from_path(&rec, device)?;
    let input = NdTensor::from_data([1, 1, GREY_H, REC_W], vec![0.82f32; GREY_H * REC_W]);
    let logits = recognizer.run_batch_logits(input)?;
    assert_eq!(logits.ndim(), 3);
    assert!(logits.size(0) >= 1);
    assert_eq!(logits.size(2), rlx_ocr::model::NUM_CLASSES);
    assert!(logits.iter().all(|v| v.is_finite()));
    Ok(())
}

pub fn run_engine_get_text(device: Device) -> Result<()> {
    let (det, rec) = load_weight_paths()?;
    let engine = OcrEngine::from_paths_on_device(det, rec, device)?;
    let grey: Vec<u8> = vec![200; GREY_H * GREY_W];
    let source = ImageSource::from_bytes(&grey, (GREY_W as u32, GREY_H as u32))?;
    let input = engine.prepare_input(source)?;
    let text = engine.get_text(&input)?;
    assert!(text.len() < 10_000);
    Ok(())
}

fn run_or_skip<F>(device: Device, label: &str, f: F)
where
    F: FnOnce() -> Result<()>,
{
    if !is_available(device) {
        eprintln!("skip ocr {label} on {device:?}: backend not available in this build");
        return;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) if is_skip_error(&e) => {
            eprintln!("skip ocr {label} on {device:?}: {e:#}");
        }
        Ok(Err(e)) => panic!("ocr {label} on {device:?} failed: {e:#}"),
        Err(payload) => {
            if is_skip_panic(payload.as_ref()) {
                eprintln!("skip ocr {label} on {device:?}: backend missing op(s)");
            } else {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

pub fn run_detection_forward_if_available(device: Device) {
    run_or_skip(device, "detection", || run_detection_forward(device));
}

pub fn run_recognition_logits_if_available(device: Device) {
    run_or_skip(device, "recognition", || run_recognition_logits(device));
}

pub fn run_engine_get_text_if_available(device: Device) {
    run_or_skip(device, "pipeline", || run_engine_get_text(device));
}

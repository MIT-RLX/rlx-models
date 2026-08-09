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

//! Parity vs the upstream [`ocrs`](https://crates.io/crates/ocrs) crate (0.12.x).
//!
//! Requires feature `parity-ocrs` (not in default features).
//!
//! ```sh
//! # Unit-level (no weights):
//! cargo test -p rlx-ocr --test ocrs_parity --features parity-ocrs --release
//!
//! # Full pipeline (downloads ~tens of MB once):
//! OCRS_PARITY_DOWNLOAD=1 cargo test -p rlx-ocr --test ocrs_parity --features parity-ocrs --release ocr_pipeline_matches_reference -- --nocapture
//!
//! # Or point at a local model dir + image:
//! OCRS_MODEL_DIR=~/.cache/ocrs OCRS_TEST_IMAGE=page.png \
//!   cargo test -p rlx-ocr --test ocrs_parity --features parity-ocrs --release ocr_pipeline_matches_reference
//! ```

#![cfg(feature = "parity-ocrs")]

#[path = "assets.rs"]
mod assets;
#[path = "env.rs"]
mod bench_env;

use anyhow::{Context, Result, bail};
use ocrs::OcrEngineParams as RefParams;
use ocrs::{
    DecodeMethod as RefDecodeMethod, ImageSource as RefImageSource, OcrEngine as RefEngine,
};
use rlx_ocr::{
    BLACK_VALUE, DEFAULT_ALPHABET, DecodeMethod, DetectionParams, DimOrder, HF_DETECTION_RTEN,
    HF_RECOGNITION_RTEN, ImageSource, OcrEngine, input_image, prepare_image, resolve_model_dir,
};
use rlx_ocr::{RlxTextDetector, RlxTextRecognizer, inference::RtenTextRecognizer};
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use std::path::{Path, PathBuf};

const REFERENCE_ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~€ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

const MASK_MAX_ABS: f32 = 1e-5;
const PREPROCESS_MAX_ABS: f32 = 1e-5;
/// Recognition log-probs (RTen vs native RLX; typical max abs diff ~4e-5).
const RECOGNITION_LOGITS_MAX_ABS: f32 = 5e-5;
/// Minimum cosine similarity between aligned detection masks (1.0 = identical).
const MASK_MIN_COSINE: f32 = 1.0 - 1e-6;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    dot / (na * nb)
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= PREPROCESS_MAX_ABS,
        "expected {expected}, got {actual}"
    );
}

fn tensors_max_abs_diff_3(a: NdTensorView<f32, 3>, b: NdTensorView<f32, 3>) -> f32 {
    assert_eq!(a.shape(), b.shape());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn tensors_max_abs_diff(a: NdTensorView<f32, 2>, b: NdTensorView<f32, 2>) -> f32 {
    assert_eq!(a.shape(), b.shape());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn default_alphabet_matches_reference() {
    assert_eq!(DEFAULT_ALPHABET, REFERENCE_ALPHABET);
    assert_eq!(BLACK_VALUE, -0.5);
}

/// Greyscale preprocessing — same cases as upstream `ocrs` `preprocess` tests.
#[test]
fn preprocess_matches_reference_cases() {
    const ITU: [f32; 3] = [0.299, 0.587, 0.114];

    fn expected_grey(r: f32, g: f32, b: f32) -> f32 {
        BLACK_VALUE + r * ITU[0] + g * ITU[1] + b * ITU[2]
    }

    let grey_hwc = prepare_image(ImageSource::from_bytes(&[0, 128, 255, 64], (2, 2)).unwrap());
    assert_eq!(grey_hwc.shape(), [1, 2, 2]);
    assert_close(grey_hwc[[0, 0, 0]], BLACK_VALUE);
    assert_close(grey_hwc[[0, 0, 1]], BLACK_VALUE + 128.0 / 255.0);
    assert_close(grey_hwc[[0, 1, 0]], BLACK_VALUE + 1.0);
    assert_close(grey_hwc[[0, 1, 1]], BLACK_VALUE + 64.0 / 255.0);

    let rgb = prepare_image(ImageSource::from_bytes(&[100, 150, 200], (1, 1)).unwrap());
    assert_close(
        rgb[[0, 0, 0]],
        expected_grey(100.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0),
    );
}

#[test]
fn ctc_greedy_matches_reference_logic() {
    let rows = [
        vec![0.0, 0.9, 0.1, 0.0],
        vec![0.0, 0.9, 0.1, 0.0],
        vec![0.0, 0.1, 0.8, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
    ];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let tensor = NdTensor::from_data([4, 4], flat);
    let hyp = rlx_ocr::ctc::decode(tensor.view(), DecodeMethod::Greedy);
    let steps: Vec<_> = hyp.steps().iter().map(|s| (s.label, s.pos)).collect();
    #[cfg(feature = "rten-inference")]
    {
        let ref_hyp = rten::ctc::CtcDecoder::new().decode_greedy(tensor.view());
        let ref_steps: Vec<_> = ref_hyp.steps().iter().map(|s| (s.label, s.pos)).collect();
        assert_eq!(steps, ref_steps);
    }
    assert_eq!(steps, vec![(1, 0), (2, 2)]);
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

fn build_engines(dir: &Path) -> Result<(RefEngine, OcrEngine)> {
    let (det_st, rec_st) = resolve_model_dir(dir)?;
    let det_rten = dir.join(HF_DETECTION_RTEN);
    let rec_rten = dir.join(HF_RECOGNITION_RTEN);
    let det_model = Model::load_file(&det_rten).with_context(|| format!("load {det_rten:?}"))?;
    let rec_model = Model::load_file(&rec_rten).with_context(|| format!("load {rec_rten:?}"))?;
    let reference = RefEngine::new(RefParams {
        detection_model: Some(det_model),
        recognition_model: Some(rec_model),
        decode_method: RefDecodeMethod::Greedy,
        ..Default::default()
    })?;
    let rlx = OcrEngine::from_paths(&det_st, &rec_st)?;
    Ok((reference, rlx))
}

/// Recognition logits on a fixed NCHW batch.
#[test]
fn recognition_logits_match_reference() -> Result<()> {
    let Some(dir) = model_dir() else {
        eprintln!("skip recognition_logits_match_reference");
        return Ok(());
    };
    let rec_rten = dir.join(HF_RECOGNITION_RTEN);
    let ref_rec = RtenTextRecognizer::from_path(&rec_rten)?;
    let rlx_rec = RlxTextRecognizer::from_model_dir(&dir, rlx_runtime::Device::Cpu)?;

    let w = 200usize;
    let h = 64usize;
    let input = NdTensor::from_data([1, 1, h, w], vec![0.82f32; h * w]);
    let ref_out = ref_rec.run(input.clone())?;
    let rlx_out = rlx_rec.run_batch_logits(input.clone())?;
    assert_eq!(ref_out.shape(), rlx_out.shape());
    let err = tensors_max_abs_diff_3(ref_out.view(), rlx_out.view());
    eprintln!("recognition logits max abs diff {err}");
    assert!(
        err <= RECOGNITION_LOGITS_MAX_ABS,
        "recognition logits max abs diff {err}"
    );
    Ok(())
}

/// Same greyscale tensor through upstream `TextDetector` vs [`RlxTextDetector`].
#[test]
fn detection_model_matches_reference() -> Result<()> {
    let Some(dir) = model_dir() else {
        eprintln!(
            "skip detection_model_matches_reference: set OCRS_MODEL_DIR or OCRS_PARITY_DOWNLOAD=1"
        );
        return Ok(());
    };
    let (det_st, _) = resolve_model_dir(&dir)?;
    let det_rten = dir.join(HF_DETECTION_RTEN);
    let det_model = Model::load_file(&det_rten)?;
    let reference = RefEngine::new(RefParams {
        detection_model: Some(det_model),
        recognition_model: None,
        ..Default::default()
    })?;
    let rlx_det = RlxTextDetector::from_path(
        &det_st,
        DetectionParams::default(),
        rlx_runtime::Device::Cpu,
    )?;

    let h = 64usize;
    let w = 96usize;
    let grey = NdTensor::from_data([1, h, w], vec![0.82f32; h * w]);
    let prep = prepare_image(ImageSource::from_tensor(grey.view(), DimOrder::Chw)?);
    let ref_in = reference.prepare_input(RefImageSource::from_tensor(
        grey.view(),
        ocrs::DimOrder::Chw,
    )?)?;
    let ref_mask = reference.detect_text_pixels(&ref_in)?;
    let rlx_mask = rlx_det.detect_text_pixels(prep.view())?;
    let err = tensors_max_abs_diff(ref_mask.view(), rlx_mask.view());
    assert!(
        err <= MASK_MAX_ABS,
        "detector-only max abs diff {err} (preprocess bypassed with shared CHW tensor)"
    );
    Ok(())
}

#[test]
fn ocr_pipeline_matches_reference() -> Result<()> {
    let Some(dir) = model_dir() else {
        eprintln!(
            "skip ocr_pipeline_matches_reference: set OCRS_MODEL_DIR or OCRS_PARITY_DOWNLOAD=1"
        );
        return Ok(());
    };
    let Some(image_path) = test_image_path() else {
        eprintln!(
            "skip ocr_pipeline_matches_reference: set OCRS_TEST_IMAGE or OCRS_PARITY_DOWNLOAD=1"
        );
        return Ok(());
    };

    let (reference, rlx) = build_engines(&dir)?;
    let img = image::open(&image_path)
        .with_context(|| format!("open image {image_path:?}"))?
        .into_rgb8();
    let (w, h) = img.dimensions();
    let bytes = img.as_raw();

    let ref_in = reference.prepare_input(RefImageSource::from_bytes(bytes, (w, h))?)?;
    let rlx_src = ImageSource::from_bytes(bytes, (w, h))?;
    let rlx_prep_direct = prepare_image(rlx_src);
    let rlx_in = rlx.prepare_input(ImageSource::from_bytes(bytes, (w, h))?)?;
    let rlx_prep = input_image(&rlx_in);
    let prep_err = tensors_max_abs_diff_3(rlx_prep_direct.view(), rlx_prep.view());
    assert!(
        prep_err <= PREPROCESS_MAX_ABS,
        "prepare_input vs prepare_image max abs diff {prep_err}"
    );

    let ref_mask = reference.detect_text_pixels(&ref_in)?;
    let rlx_mask = rlx.detect_text_pixels(&rlx_in)?;
    assert_eq!(ref_mask.shape(), rlx_mask.shape(), "mask shape mismatch");
    let mask_err = tensors_max_abs_diff(ref_mask.view(), rlx_mask.view());
    assert!(
        mask_err <= MASK_MAX_ABS,
        "detection mask max abs diff {mask_err} > {MASK_MAX_ABS}"
    );
    let ref_flat: Vec<f32> = ref_mask.iter().copied().collect();
    let rlx_flat: Vec<f32> = rlx_mask.iter().copied().collect();
    let cos = cosine_similarity(&ref_flat, &rlx_flat);
    assert!(
        cos >= MASK_MIN_COSINE,
        "detection mask cosine similarity {cos} < {MASK_MIN_COSINE}"
    );

    let ref_words = reference.detect_words(&ref_in)?;
    let rlx_words = rlx.detect_words(&rlx_in)?;
    assert_eq!(
        ref_words.len(),
        rlx_words.len(),
        "word count: reference {} vs rlx {}",
        ref_words.len(),
        rlx_words.len()
    );

    let ref_lines = reference.find_text_lines(&ref_in, &ref_words);
    let rlx_lines = rlx.find_text_lines(&rlx_in, &rlx_words);
    assert_eq!(ref_lines.len(), rlx_lines.len());

    let ref_rec = reference.recognize_text(&ref_in, &ref_lines)?;
    let rlx_rec = rlx.recognize_text(&rlx_in, &rlx_lines)?;
    assert_eq!(ref_rec.len(), rlx_rec.len());
    for (i, (a, b)) in ref_rec.iter().zip(rlx_rec.iter()).enumerate() {
        match (a, b) {
            (Some(ra), Some(rb)) => {
                let ref_s = ra.to_string();
                let rlx_s = rb.text();
                if ref_s != rlx_s {
                    eprintln!("line {i} text mismatch:\n  ref: {ref_s:?}\n  rlx: {rlx_s:?}");
                }
                assert_eq!(ref_s, rlx_s, "line {i} text mismatch");
            }
            (None, None) => {}
            (Some(ra), None) => {
                eprintln!("line {i}: rlx None, ref Some({:?})", ra.to_string());
                bail!("line {i}: reference/rlx recognition mismatch");
            }
            (None, Some(rb)) => {
                eprintln!("line {i}: ref None, rlx Some({:?})", rb.text());
                bail!("line {i}: reference/rlx recognition mismatch");
            }
        }
    }

    if let (Some(ref_line), Some(rlx_line)) = (ref_lines.first(), rlx_lines.first()) {
        if !ref_line.is_empty() && !rlx_line.is_empty() {
            let ref_prep = reference.prepare_recognition_input(&ref_in, ref_line)?;
            let rlx_prep = rlx.prepare_recognition_input(&rlx_in, rlx_line)?;
            assert_eq!(ref_prep.shape(), rlx_prep.shape());
            let prep_err = tensors_max_abs_diff(ref_prep.view(), rlx_prep.view());
            assert!(
                prep_err <= PREPROCESS_MAX_ABS,
                "prepare_recognition_input max abs diff {prep_err}"
            );
        }
    }

    let ref_text = reference.get_text(&ref_in)?;
    let rlx_text = rlx.get_text(&rlx_in)?;
    if ref_text != rlx_text {
        eprintln!(
            "full-page text mismatch\n--- reference ---\n{ref_text}\n--- rlx ---\n{rlx_text}"
        );
        for (i, (a, b)) in ref_text.lines().zip(rlx_text.lines()).enumerate() {
            if a != b {
                eprintln!("line {i} differs:\n  ref: {a:?}\n  rlx: {b:?}");
            }
        }
    }
    assert_eq!(
        ref_text, rlx_text,
        "full-page text mismatch (see stderr for diff)"
    );

    let _ = rlx_prep;
    Ok(())
}

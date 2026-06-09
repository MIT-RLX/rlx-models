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

//! Quick check with real ocrs safetensors (set `OCR_MODEL_DIR`).

use anyhow::Result;
use rlx_ocr::{ImageSource, OcrEngine, OcrEngineParams, input_image, resolve_model_dir};
use rten_tensor::prelude::*;

#[path = "env.rs"]
mod bench_env;

fn model_dir() -> Option<std::path::PathBuf> {
    bench_env::env_var("OCR_MODEL_DIR", "OCRS_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .filter(|p| p.is_dir())
}

#[test]
fn ocr_detection_and_engine_with_real_weights() -> Result<()> {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set OCR_MODEL_DIR to a directory with ocrs safetensors");
        return Ok(());
    };
    let (det, _rec) = match resolve_model_dir(&dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skip: {e}");
            return Ok(());
        }
    };

    let cfg = rlx_ocr::model::DetectionGraphConfig {
        batch: 1,
        height: 64,
        width: 96,
    };
    let mut wm = rlx_ocr::load_safetensors_weights(&det)?;
    rlx_ocr::model::build_detection_graph(&mut wm, cfg)?;

    // Forward pass: enable when runtime NCHW path is verified on this host.
    if std::env::var("OCR_RUN_FORWARD").ok().as_deref() == Some("1") {
        let detector = rlx_ocr::RlxTextDetector::from_safetensors_sized(
            &det,
            Default::default(),
            cfg,
            rlx_runtime::Device::Cpu,
        )?;
        let grey = vec![128u8; 96 * 64];
        let src = ImageSource::from_bytes(&grey, (96, 64))?;
        let input = OcrEngine::new(OcrEngineParams::default())?.prepare_input(src)?;
        let view = input_image(&input);
        let mask_a = detector.detect_text_pixels(view)?;
        let mask_b = detector.detect_text_pixels(view)?;
        assert_eq!(mask_a.shape(), mask_b.shape());
    }
    Ok(())
}

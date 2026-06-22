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

//! Real-weights parity probe (ignored by default). Compares this crate's
//! detections against an HF `transformers` reference produced by
//! `reference_backend.py`.
//!
//! Setup:
//!   pip install torch transformers pillow numpy
//!   python crates/rlx-grounding-dino/reference_backend.py \
//!       --image cat.jpg --text "a cat. a remote control." --out /tmp/gdino_ref
//!
//! Run:
//!   RLX_GDINO_DIR=<hf snapshot dir>  \
//!   RLX_GDINO_IMAGE=cat.jpg          \
//!   RLX_GDINO_REF=/tmp/gdino_ref     \
//!   cargo test -p rlx-grounding-dino --test e2e_real -- --ignored --nocapture

use rlx_grounding_dino::GroundingDino;
use rlx_grounding_dino::config::GroundingDinoConfig;
use rlx_grounding_dino::tokenizer::text_tokens_from_ids;
use std::path::Path;

fn iou(a: &[f32; 4], b: &[f64]) -> f32 {
    let x0 = a[0].max(b[0] as f32);
    let y0 = a[1].max(b[1] as f32);
    let x1 = a[2].min(b[2] as f32);
    let y1 = a[3].min(b[3] as f32);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = ((b[2] - b[0]) * (b[3] - b[1])).max(0.0) as f32;
    if inter <= 0.0 {
        return 0.0;
    }
    inter / (area_a + area_b - inter)
}

#[test]
#[ignore = "needs real checkpoint + Python reference dump (set RLX_GDINO_*)"]
fn matches_hf_reference() {
    let dir = std::env::var("RLX_GDINO_DIR").expect("set RLX_GDINO_DIR to the HF snapshot dir");
    let image = std::env::var("RLX_GDINO_IMAGE").expect("set RLX_GDINO_IMAGE");
    let refdir = std::env::var("RLX_GDINO_REF").expect("set RLX_GDINO_REF to reference output dir");

    let dir = Path::new(&dir);
    let cfg = GroundingDinoConfig::from_file(&dir.join("config.json")).unwrap();
    let model = GroundingDino::from_checkpoint(&dir.join("model.safetensors"), cfg).unwrap();

    // Use the exact token ids from the reference to remove tokenizer variance.
    let inputs: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(Path::new(&refdir).join("inputs.json")).unwrap(),
    )
    .unwrap();
    let ids: Vec<u32> = inputs["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let tokens = text_tokens_from_ids(ids);

    let img = image::open(&image).unwrap().to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb = img.into_raw();

    let dets = model.detect(&rgb, h, w, &tokens, 0.3, 0.25);

    let reference: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(Path::new(&refdir).join("detections.json")).unwrap(),
    )
    .unwrap();
    let ref_boxes = reference["boxes"].as_array().unwrap();
    let ref_scores = reference["scores"].as_array().unwrap();
    eprintln!("rust dets={} ref dets={}", dets.len(), ref_boxes.len());

    // Each reference detection should have a close match among ours.
    let mut matched = 0;
    for (rb, rs) in ref_boxes.iter().zip(ref_scores.iter()) {
        let rbox: Vec<f64> = rb
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let rscore = rs.as_f64().unwrap() as f32;
        let best = dets
            .iter()
            .map(|d| iou(&d.bbox, &rbox))
            .fold(0.0f32, f32::max);
        eprintln!("ref score={rscore:.3} best IoU={best:.3}");
        if best > 0.5 {
            matched += 1;
        }
    }
    assert!(
        matched as f32 >= 0.8 * ref_boxes.len() as f32,
        "only {matched}/{} reference detections matched (IoU>0.5)",
        ref_boxes.len()
    );
}

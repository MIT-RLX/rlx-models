// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A SAM v1 package segments what `Sam::forward` does: a photo, one point,
//! two points, a box, a point with a mask prompt, one mask and three — on
//! the ViT-B checkpoint where it is on disk (`weights/vision/sam-v1`, or
//! `RLX_SAM_WEIGHTS`).

#![cfg(feature = "export")]

use std::path::{Path, PathBuf};

use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use rlx_sam::export::ExportOptions;
use rlx_sam::{Sam, SamConfig};
use serde_json::json;

fn weights() -> Option<PathBuf> {
    let p = rlx_ir::env::var("RLX_SAM_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../weights/vision/sam-v1/sam_vit_b_meta.safetensors")
        });
    p.exists().then_some(p)
}

struct Prompt {
    name: &'static str,
    points: Vec<f32>,
    labels: Vec<f32>,
    boxes: Option<Vec<f32>>,
    mask: bool,
    multimask: bool,
}

#[test]
fn a_package_segments_what_sam_does() {
    let Some(path) = weights() else {
        eprintln!("no SAM v1 checkpoint; skipped");
        return;
    };
    let photo =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../rlx-locateanything/fixtures/sample.jpg");
    let img = image::open(&photo).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let rgb = img.into_raw();

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("sam.rlxp");
    let report = rlx_sam::export::export_rlxp(&path, &out, &ExportOptions::default()).unwrap();
    eprintln!("{report}");

    let mut sam =
        Sam::from_safetensors_on(path.to_str().unwrap(), SamConfig::vit_b(), Device::Cpu).unwrap();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    // A low-resolution mask prompt: SAM's own first answer, as its users feed it back.
    let (first, _) = sam
        .forward(
            &rgb,
            h,
            w,
            Some((&[400.0, 300.0], &[1.0])),
            None,
            None,
            false,
        )
        .unwrap();
    let prior = first.mask_logits.clone();

    let prompts = [
        Prompt {
            name: "one point",
            points: vec![400.0, 300.0],
            labels: vec![1.0],
            boxes: None,
            mask: false,
            multimask: true,
        },
        Prompt {
            name: "two points, one mask",
            points: vec![400.0, 300.0, 620.0, 180.0],
            labels: vec![1.0, 0.0],
            boxes: None,
            mask: false,
            multimask: false,
        },
        Prompt {
            name: "a box",
            points: vec![],
            labels: vec![],
            boxes: Some(vec![200.0, 150.0, 700.0, 520.0]),
            mask: false,
            multimask: true,
        },
        Prompt {
            name: "a point and a mask",
            points: vec![400.0, 300.0],
            labels: vec![1.0],
            boxes: None,
            mask: true,
            multimask: false,
        },
    ];
    for p in &prompts {
        let points = (!p.points.is_empty()).then_some((p.points.as_slice(), p.labels.as_slice()));
        let (want, (rh, rw)) = sam
            .forward(
                &rgb,
                h,
                w,
                points,
                p.boxes.as_deref(),
                p.mask.then_some(prior.as_slice()),
                p.multimask,
            )
            .unwrap();
        let mut inputs = vec![Tensor::new(
            "image",
            rlx_ir::DType::U8,
            vec![h, w, 3],
            rgb.clone(),
        )];
        if !p.points.is_empty() {
            inputs.push(Tensor::f32("points", vec![p.labels.len(), 2], &p.points));
            inputs.push(Tensor::f32("labels", vec![p.labels.len()], &p.labels));
        }
        if let Some(b) = &p.boxes {
            inputs.push(Tensor::f32("boxes", vec![b.len() / 4, 4], b));
        }
        if p.mask {
            inputs.push(Tensor::f32("mask", vec![256, 256], &prior));
        }
        let outs = m
            .run(&inputs, &json!({"multimask": p.multimask}))
            .unwrap_or_else(|e| panic!("{}: {e:#}", p.name));
        let get = |n: &str| outs.iter().find(|t| t.name == n).unwrap().to_f32().unwrap();
        let (masks, iou) = (get("masks"), get("iou"));
        assert_eq!(masks.len(), want.mask_logits.len(), "{}", p.name);
        let size = &outs.iter().find(|t| t.name == "size").unwrap().data;
        let size: Vec<i32> = size
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(size, vec![rh as i32, rw as i32], "{}", p.name);
        let worst = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        let scale = want.mask_logits.iter().fold(0f32, |m, v| m.max(v.abs()));
        let flipped = masks
            .iter()
            .zip(&want.mask_logits)
            .filter(|(a, b)| (**a > 0.0) != (**b > 0.0))
            .count();
        let (dm, di) = (
            worst(&masks, &want.mask_logits),
            worst(&iou, &want.iou_pred),
        );
        eprintln!(
            "{}: masks within {dm:.3e} (logits up to {scale:.1}), {flipped} of {} pixels on the \
             other side of 0; iou within {di:.3e}",
            p.name,
            masks.len()
        );
        assert!(dm <= 1e-3 * scale.max(1.0), "{}: masks off by {dm}", p.name);
        assert!(di <= 1e-3, "{}: iou off by {di}", p.name);
    }
}

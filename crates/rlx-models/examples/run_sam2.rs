// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Segment Anything v2 — end-to-end via `SamRunner::predict_image`.
// Hiera backbone with FPN neck + memory attention. Loads weights,
// runs the encoder + prompt encoder + mask decoder against a
// single foreground click.
//
// Usage:
//   cargo run --release -p rlx-models --features metal \
//       --example run_sam2 -- /path/to/sam2_hiera_b.safetensors
//
//   RLX_SAM2_VARIANT = tiny | small | base_plus | large (default tiny)
//
// Equivalent CLI call:
//   rlx-sam2 (or `rlx-run sam2`) --weights sam2_hiera_b.safetensors --device metal \
//                --point 512,512

use anyhow::{Result, anyhow};
use rlx_models::run::{SamArch, SamPredictionAny, SamRunner};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let weights = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: run_sam2 <weights.safetensors>"))?;

    let runner = SamRunner::builder(SamArch::Sam2)
        .weights(&weights)
        .device(Device::Metal)
        .build()?;
    eprintln!("{}", runner.summary());

    let (h_in, w_in) = (1024usize, 1024usize);
    let mut rgb = vec![0u8; h_in * w_in * 3];
    for y in 0..h_in {
        for x in 0..w_in {
            let base = (y * w_in + x) * 3;
            rgb[base] = (x * 255 / w_in) as u8;
            rgb[base + 1] = (y * 255 / h_in) as u8;
            rgb[base + 2] = ((x + y) * 127 / (h_in + w_in)) as u8;
        }
    }

    let points_xy = [w_in as f32 / 2.0, h_in as f32 / 2.0];
    let points_lbl = [1.0f32];

    let t0 = std::time::Instant::now();
    let pred = runner.predict_image(
        &rgb,
        h_in,
        w_in,
        Some((&points_xy, &points_lbl)),
        /*boxes*/ None,
        /*text_tokens*/ &[],
    )?;
    match pred {
        SamPredictionAny::Sam2(p) => {
            eprintln!(
                "[sam2] forward in {:?} — {} masks at {}×{}, object_score_logits={:?}",
                t0.elapsed(),
                p.num_masks,
                p.h_out,
                p.w_out,
                &p.object_score_logits[..p.object_score_logits.len().min(3)]
            );
            eprintln!(
                "[sam2] iou predictions: {:?}",
                &p.iou_pred[..p.iou_pred.len().min(p.num_masks)]
            );
        }
        _ => unreachable!("SamArch::Sam2 dispatches to SamPredictionAny::Sam2"),
    }
    Ok(())
}

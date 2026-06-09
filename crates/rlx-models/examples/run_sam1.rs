// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Segment Anything v1 — end-to-end through the unified
// `SamRunner::predict_image` API. Loads weights, encodes a
// (synthetic) image, runs the prompt encoder + mask decoder
// against a single foreground click.
//
// Usage:
//   cargo run --release -p rlx-models --features metal \
//       --example run_sam1 -- /path/to/sam_vit_b.safetensors
//
//   RLX_SAM_VARIANT=vit_b | vit_l | vit_h    # default vit_b
//
// Equivalent CLI call:
//   rlx-sam1 (or `rlx-run sam1`) --weights sam_vit_b.safetensors --device metal \
//                --point 512,512

use anyhow::{Result, anyhow};
use rlx_models::run::{SamArch, SamPredictionAny, SamRunner};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let weights = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: run_sam1 <weights.safetensors>"))?;

    let runner = SamRunner::builder(SamArch::Sam1)
        .weights(&weights)
        .device(Device::Metal)
        .build()?;
    eprintln!("{}", runner.summary());

    // Synthetic 1024×1024 RGB gradient — swap for
    // image::open(path)?.to_rgb8().as_raw() to use a real picture.
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

    // Single foreground click at the image center.
    let points_xy = [w_in as f32 / 2.0, h_in as f32 / 2.0];
    let points_lbl = [1.0f32]; // 1 = foreground, 0 = background

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
        SamPredictionAny::Sam1(p) => {
            eprintln!(
                "[sam1] forward in {:?} — {} masks, mask_side={}, iou={:?}",
                t0.elapsed(),
                p.num_masks,
                p.mask_side,
                &p.iou_pred[..p.iou_pred.len().min(p.num_masks)]
            );
        }
        _ => unreachable!("SamArch::Sam1 dispatches to SamPredictionAny::Sam1"),
    }
    Ok(())
}

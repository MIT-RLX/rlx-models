// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Segment Anything v3 — end-to-end via `SamRunner::predict_image`
// with a pre-tokenized text prompt (SAM 3's detector is text-
// conditioned; pass real tokenizer output for non-trivial queries).
//
// Usage:
//   cargo run --release -p rlx-models --features metal \
//       --example run_sam3 -- /path/to/sam3.safetensors
//
// Equivalent CLI call:
//   rlx-sam3 (or `rlx-run sam3`) --weights sam3.safetensors --device metal \
//                --point 512,512 --text-tokens 0,1,2,3,4,5,6,7

use anyhow::{Result, anyhow};
use rlx_models::run::{SamArch, SamPredictionAny, SamRunner};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let weights = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: run_sam3 <weights.safetensors>"))?;

    let runner = SamRunner::builder(SamArch::Sam3)
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

    // Stand-in token ids — replace with real tokenizer output
    // (e.g. `tokenizers::Tokenizer::from_pretrained("…")`) and the
    // SAM 3 text-encoder vocab.
    let text_tokens: Vec<u32> = (0..32u32).collect();
    let points_xy = [w_in as f32 / 2.0, h_in as f32 / 2.0];
    let points_lbl = [1.0f32];

    let t0 = std::time::Instant::now();
    let pred = runner.predict_image(
        &rgb,
        h_in,
        w_in,
        Some((&points_xy, &points_lbl)),
        /*boxes*/ None,
        &text_tokens,
    )?;
    match pred {
        SamPredictionAny::Sam3(p) => {
            eprintln!(
                "[sam3] forward in {:?} — instances={}, mask_shape={:?}, boxes_shape={:?}, out=({},{})",
                t0.elapsed(),
                p.num_instances,
                p.mask_shape,
                p.boxes_shape,
                p.h_out,
                p.w_out
            );
            eprintln!(
                "[sam3] top scores (first 5): {:?}",
                &p.scores[..p.scores.len().min(5)]
            );
        }
        _ => unreachable!("SamArch::Sam3 dispatches to SamPredictionAny::Sam3"),
    }
    Ok(())
}

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

//
//! Per-stage SAM3 image-pipeline benchmark on the native CPU backend.
//! Run alongside `tests/sam3_parity_helpers/bench.py` for a head-to-head.
//!
//!   RLX_SAM3_WEIGHTS=/path/sam3.safetensors \
//!     cargo run --release --example sam3_bench

use anyhow::{Context, Result};
use rlx_models::sam3::{SAM3_IMG_SIZE, Sam3, Sam3Config};
use std::env;
use std::time::Instant;

fn synthesize_image() -> Vec<u8> {
    let n = SAM3_IMG_SIZE * SAM3_IMG_SIZE * 3;
    let mut v = vec![0u8; n];
    for y in 0..SAM3_IMG_SIZE {
        for x in 0..SAM3_IMG_SIZE {
            for c in 0..3 {
                let fx = x as f32 / SAM3_IMG_SIZE as f32;
                let fy = y as f32 / SAM3_IMG_SIZE as f32;
                let phase = (c as f32) * 0.7;
                let s = (std::f32::consts::TAU * fx + phase).sin()
                    * (std::f32::consts::PI * fy + phase).cos();
                let val = ((s + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                v[(y * SAM3_IMG_SIZE + x) * 3 + c] = val;
            }
        }
    }
    v
}

fn fmt(vs: &[f32]) -> String {
    let avg = vs.iter().sum::<f32>() / vs.len() as f32;
    let min = vs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    format!("avg={avg:8.1}ms  min={min:8.1}ms  max={max:8.1}ms")
}

fn main() -> Result<()> {
    let weights = rlx_ir::env::var("RLX_SAM3_WEIGHTS")
        .context("set RLX_SAM3_WEIGHTS to a converted .safetensors checkpoint")?;
    let warmup: usize = env::var("BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let iters: usize = env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    eprintln!("loading SAM3 from {weights}");
    let t_build = Instant::now();
    let cfg = Sam3Config::base();
    let mut model = Sam3::from_safetensors(&weights, cfg)?;
    eprintln!("build+load: {:.1}s", t_build.elapsed().as_secs_f32());

    let image = synthesize_image();
    // BPE tokenizer is not ported — use the reference dumper to produce
    // tokens if available, otherwise fall back to all-PAD (shape parity).
    let tokens_path = rlx_ir::env::var("RLX_SAM3_TOKENS_BIN");
    let tokens: Vec<u32> = if let Some(p) = tokens_path {
        let bytes = std::fs::read(&p).with_context(|| format!("reading {p}"))?;
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
            .collect()
    } else {
        vec![0u32; 32]
    };

    let mut totals = Vec::with_capacity(iters);
    let mut vis = Vec::with_capacity(iters);
    let mut text = Vec::with_capacity(iters);
    let mut enc = Vec::with_capacity(iters);
    let mut pred = Vec::with_capacity(iters);

    let mut step = |totals: &mut Vec<f32>,
                    vis: &mut Vec<f32>,
                    text: &mut Vec<f32>,
                    enc: &mut Vec<f32>,
                    pred: &mut Vec<f32>|
     -> Result<()> {
        let t0 = Instant::now();
        let t = Instant::now();
        let encoded = model.encode_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE)?;
        vis.push(t.elapsed().as_secs_f32() * 1000.0);
        let _ = encoded;

        let t = Instant::now();
        let text_out = model.encode_text_tokens(&tokens, 1, tokens.len())?;
        text.push(t.elapsed().as_secs_f32() * 1000.0);

        // Isolated detector encoder bench using already-computed neck + text.
        let neck = model.predict_neck(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE)?;
        let lvl = &neck[2]; // scale=1.0 level
        let t = Instant::now();
        let _ = model.run_encoder(
            &lvl.features,
            &lvl.pos,
            &text_out.text_memory_resized,
            &text_out.attention_mask,
            1,
            lvl.h,
            lvl.w,
            tokens.len(),
        )?;
        enc.push(t.elapsed().as_secs_f32() * 1000.0);

        let t = Instant::now();
        let _ = model.predict_image_text(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE, &tokens)?;
        pred.push(t.elapsed().as_secs_f32() * 1000.0);

        totals.push(t0.elapsed().as_secs_f32() * 1000.0);
        Ok(())
    };

    eprintln!("warmup × {warmup}");
    for _ in 0..warmup {
        step(&mut totals, &mut vis, &mut text, &mut enc, &mut pred)?;
    }
    totals.clear();
    vis.clear();
    text.clear();
    enc.clear();
    pred.clear();

    eprintln!("measured × {iters}");
    for _ in 0..iters {
        step(&mut totals, &mut vis, &mut text, &mut enc, &mut pred)?;
    }

    let backend = if rlx_ir::env::flag("RLX_SAM3_ENCODER_HOST") {
        "host"
    } else {
        "IR"
    };
    println!("# rust bench (release, BLAS=Accelerate, encoder={backend})");
    println!("  rust encode_image       : {}", fmt(&vis));
    println!("  rust encode_text_tokens : {}", fmt(&text));
    println!("  rust run_encoder        : {}", fmt(&enc));
    println!("  rust predict_image_text : {}", fmt(&pred));
    println!("  rust full bench loop    : {}", fmt(&totals));
    Ok(())
}

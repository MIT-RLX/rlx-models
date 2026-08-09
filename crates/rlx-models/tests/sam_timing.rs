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

//! Single-shot wall-clock timings for SAM v1 + DINOv2 vs candle, CPU
//! and Metal. Cheap alternative to the criterion benches when you
//! just want a quick speed snapshot. Each forward runs 3 times with
//! one warmup; median is printed.
//!
//! ```bash
//! RLX_SAM_WEIGHTS=/tmp/rlx_sam/sam_vit_b_01ec64.safetensors \
//! RLX_DINOV2_WEIGHTS=/tmp/rlx_dino/dinov2_vits14.safetensors \
//!   cargo test -p rlx-models --features parity-candle --release sam_timing -- --nocapture --test-threads 1
//! ```

#![cfg(feature = "parity-candle")]

mod compile_support;

use anyhow::Result;
use candle_core::{DType as CDType, Device as CDevice, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::dinov2 as candle_dinov2;
use candle_transformers::models::segment_anything::image_encoder::ImageEncoderViT;
use rlx_models::WeightMap;
use rlx_models::dinov2::{DinoV2Config, assemble_hidden, build_dinov2_graph_sized};
use rlx_models::sam::{
    SAM_IMG_SIZE, SAM_PROMPT_EMBED_DIM, Sam, SamConfig, SamEncoderConfig, assemble_patch_tokens,
    build_sam_encoder_graph,
};
use rlx_runtime::Device;
use std::time::Instant;

const ITERS: usize = 3;
const WARMUP: usize = 1;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_it<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1000.0); // ms
    }
    median(samples)
}

#[test]
// 6.28 / 3.14 are deliberate 2-dp values, not TAU/PI: they define the
// synthetic parity input, and changing them would change every dumped
// reference blob compared against here.
#[allow(clippy::approx_constant)]
fn timing_sam_vit_b() -> Result<()> {
    let Some(weights) = rlx_ir::env::var("RLX_SAM_WEIGHTS") else {
        eprintln!("skipping (set RLX_SAM_WEIGHTS)");
        return Ok(());
    };

    // Synthetic deterministic image.
    let mut image = vec![0f32; 3 * SAM_IMG_SIZE * SAM_IMG_SIZE];
    for c in 0..3 {
        let phase = (c as f32) * 0.7;
        for y in 0..SAM_IMG_SIZE {
            for x in 0..SAM_IMG_SIZE {
                image[c * SAM_IMG_SIZE * SAM_IMG_SIZE + y * SAM_IMG_SIZE + x] =
                    (6.28 * x as f32 / SAM_IMG_SIZE as f32 + phase).sin()
                        * (3.14 * y as f32 / SAM_IMG_SIZE as f32 + phase).cos();
            }
        }
    }

    // ── candle CPU ──
    let cdev = CDevice::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], CDType::F32, &cdev)? };
    let candle_enc = ImageEncoderViT::new(
        SAM_IMG_SIZE,
        16,
        3,
        768,
        12,
        12,
        SAM_PROMPT_EMBED_DIM,
        true,
        true,
        true,
        14,
        &[2, 5, 8, 11],
        vb.pp("image_encoder"),
    )?;
    let img_t = Tensor::from_slice(&image, (1, 3, SAM_IMG_SIZE, SAM_IMG_SIZE), &cdev)?;
    let candle_ms = time_it(|| {
        let _ = candle_enc.forward(&img_t).unwrap();
    });

    // ── rlx CPU encoder (graph + neck) ──
    let cfg = SamEncoderConfig::vit_b();
    let mut wm_cpu = WeightMap::from_file(&weights)?;
    let (graph, params, pre) = build_sam_encoder_graph(&cfg, &mut wm_cpu)?;
    let mut compiled_cpu = compile_support::compile_sam(Device::Cpu, graph, params);
    let rlx_cpu_ms = time_it(|| {
        let hidden = assemble_patch_tokens(&pre, &image).unwrap();
        // Neck is inside the compiled graph — the run IS the full encoder.
        let _body = compiled_cpu
            .run(&[("hidden", hidden.as_slice())])
            .into_iter()
            .next()
            .unwrap();
    });

    // ── rlx Metal encoder (gated) ──
    #[cfg(feature = "metal")]
    let rlx_metal_ms = {
        let cfg2 = SamEncoderConfig::vit_b();
        let mut wm_m = WeightMap::from_file(&weights)?;
        let (graph2, params2, pre2) = build_sam_encoder_graph(&cfg2, &mut wm_m)?;
        let mut compiled_m = compile_support::compile_sam(Device::Metal, graph2, params2);
        Some(time_it(|| {
            let hidden = assemble_patch_tokens(&pre2, &image).unwrap();
            // Neck is inside the compiled graph — the run IS the full encoder.
            let _body = compiled_m
                .run(&[("hidden", hidden.as_slice())])
                .into_iter()
                .next()
                .unwrap();
        }))
    };
    #[cfg(not(feature = "metal"))]
    let rlx_metal_ms: Option<f64> = None;

    // ── rlx end-to-end (encode + decode with point prompt) ──
    let mut sam_e2e = Sam::from_safetensors(&weights, SamConfig::vit_b())?;
    let e2e_ms = time_it(|| {
        let emb = sam_e2e.encode_image(&image);
        let _ = sam_e2e
            .predict_masks(
                &emb,
                Some((&[512.0f32, 512.0], &[1.0f32])),
                None,
                None,
                true,
            )
            .unwrap();
    });

    eprintln!(
        "\n=== SAM ViT-B @ 1024×1024 (median of {} iters, 1 warmup) ===",
        ITERS
    );
    eprintln!("  candle CPU encoder:         {:>8.1} ms", candle_ms);
    eprintln!(
        "  rlx    CPU encoder:         {:>8.1} ms  ({:.2}× faster than candle)",
        rlx_cpu_ms,
        candle_ms / rlx_cpu_ms
    );
    if let Some(m) = rlx_metal_ms {
        eprintln!(
            "  rlx    Metal encoder:       {:>8.1} ms  ({:.2}× faster than candle)",
            m,
            candle_ms / m
        );
    }
    eprintln!("  rlx    CPU full (enc+dec):  {:>8.1} ms", e2e_ms);

    Ok(())
}

#[test]
// 6.28 / 3.14 are deliberate 2-dp values, not TAU/PI: they define the
// synthetic parity input, and changing them would change every dumped
// reference blob compared against here.
#[allow(clippy::approx_constant)]
fn timing_dinov2_vit_small() -> Result<()> {
    let Some(weights) = rlx_ir::env::var("RLX_DINOV2_WEIGHTS") else {
        eprintln!("skipping (set RLX_DINOV2_WEIGHTS)");
        return Ok(());
    };

    const IMG: usize = 518;
    let mut image = vec![0f32; 3 * IMG * IMG];
    for c in 0..3 {
        let phase = (c as f32) * 0.7;
        for y in 0..IMG {
            for x in 0..IMG {
                image[c * IMG * IMG + y * IMG + x] = (6.28 * x as f32 / IMG as f32 + phase).sin()
                    * (3.14 * y as f32 / IMG as f32 + phase).cos();
            }
        }
    }

    // candle
    let cdev = CDevice::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], CDType::F32, &cdev)? };
    let model = candle_dinov2::vit_small(vb)?;
    let img_t = Tensor::from_slice(&image, (1, 3, IMG, IMG), &cdev)?;
    let candle_ms = time_it(|| {
        let _ = model.forward(&img_t).unwrap();
    });

    // rlx CPU
    let cfg = DinoV2Config::vit_small(IMG);
    let mut wm = WeightMap::from_file(&weights)?;
    let (graph, params, pre) = build_dinov2_graph_sized(&cfg, &mut wm, 1)?;
    let mut compiled = compile_support::compile_encoder(Device::Cpu, graph, params);
    let rlx_cpu_ms = time_it(|| {
        let hidden = assemble_hidden(&pre, &image, 1, 14, IMG).unwrap();
        let _ = compiled.run(&[("hidden", hidden.as_slice())]);
    });

    eprintln!(
        "\n=== DINOv2 ViT-S @ 518×518 (median of {} iters, 1 warmup) ===",
        ITERS
    );
    eprintln!("  candle CPU forward:  {:>8.1} ms", candle_ms);
    eprintln!(
        "  rlx    CPU forward:  {:>8.1} ms  ({:.2}× faster than candle)",
        rlx_cpu_ms,
        candle_ms / rlx_cpu_ms
    );

    Ok(())
}

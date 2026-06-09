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

//! Criterion bench: rlx-models SAM v1 vs candle SAM (CPU).
//!
//! ## Running
//!
//! ```bash
//! RLX_SAM_WEIGHTS=/path/to/sam_vit_b_01ec64.safetensors \
//!   cargo bench -p rlx-models --features parity-candle --bench sam_vs_candle
//! ```
//!
//! Three benchmarks per side (encoder, full forward at single-point
//! prompt, image_embeddings cached + decoder only):
//!   - `encoder_forward`   — image → 256·64·64 embeddings
//!   - `full_forward`      — image → masks + IoU (one foreground point)
//!   - `decoder_only`      — cached embeddings → masks + IoU

#![cfg(feature = "parity-candle")]

use candle_core::{DType as CDType, Device as CDevice, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::image_encoder::ImageEncoderViT;
use candle_transformers::models::segment_anything::mask_decoder::MaskDecoder;
use candle_transformers::models::segment_anything::prompt_encoder::PromptEncoder;
use criterion::{Criterion, criterion_group, criterion_main};
use rlx_models::sam::{
    SAM_EMBED_HW, SAM_IMG_SIZE, SAM_MASK_IN_CHANS, SAM_PROMPT_EMBED_DIM, Sam, SamConfig,
};
use std::hint::black_box;

fn synthesize_image() -> Vec<f32> {
    let n = 3 * SAM_IMG_SIZE * SAM_IMG_SIZE;
    let mut v = vec![0f32; n];
    for c in 0..3 {
        let phase = (c as f32) * 0.7;
        for y in 0..SAM_IMG_SIZE {
            for x in 0..SAM_IMG_SIZE {
                let fx = x as f32 / SAM_IMG_SIZE as f32;
                let fy = y as f32 / SAM_IMG_SIZE as f32;
                v[c * SAM_IMG_SIZE * SAM_IMG_SIZE + y * SAM_IMG_SIZE + x] =
                    (6.28 * fx + phase).sin() * (3.14 * fy + phase).cos();
            }
        }
    }
    v
}

fn bench(c: &mut Criterion) {
    let Some(weights) = rlx_ir::env::var("RLX_SAM_WEIGHTS") else {
        eprintln!("skipping sam_vs_candle bench — set RLX_SAM_WEIGHTS");
        return;
    };
    let image = synthesize_image();

    // ── candle setup ──
    let cdev = CDevice::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&weights], CDType::F32, &cdev).expect("candle: load")
    };
    let candle_encoder = ImageEncoderViT::new(
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
    )
    .expect("candle: build encoder");
    let candle_prompt = PromptEncoder::new(
        SAM_PROMPT_EMBED_DIM,
        (SAM_EMBED_HW, SAM_EMBED_HW),
        (SAM_IMG_SIZE, SAM_IMG_SIZE),
        SAM_MASK_IN_CHANS,
        vb.pp("prompt_encoder"),
    )
    .expect("candle: build prompt encoder");
    let candle_decoder = MaskDecoder::new(
        SAM_PROMPT_EMBED_DIM,
        3,
        3,
        SAM_PROMPT_EMBED_DIM,
        vb.pp("mask_decoder"),
    )
    .expect("candle: build decoder");
    let candle_img = Tensor::from_slice(&image, (1, 3, SAM_IMG_SIZE, SAM_IMG_SIZE), &cdev)
        .expect("candle: tensor");
    let candle_pts = Tensor::from_slice(&[512.0f32, 512.0], (1, 1, 2), &cdev).unwrap();
    let candle_labels = Tensor::from_slice(&[1.0f32], (1, 1), &cdev).unwrap();

    // Pre-encode once for the decoder-only bench.
    let candle_embed = candle_encoder.forward(&candle_img).unwrap();

    // ── rlx setup ──
    let mut rlx_sam = Sam::from_safetensors(&weights, SamConfig::vit_b()).expect("rlx: build sam");
    let pt_coords = vec![512.0f32, 512.0];
    let pt_labels = vec![1.0f32];
    let rlx_embed = rlx_sam.encode_image(&image);

    let mut group = c.benchmark_group("sam_vit_b_1024");
    group.sample_size(10);

    group.bench_function("candle_encoder_forward", |b| {
        b.iter(|| {
            let out = candle_encoder.forward(black_box(&candle_img)).unwrap();
            black_box(out);
        });
    });
    group.bench_function("rlx_encoder_forward", |b| {
        b.iter(|| {
            let out = rlx_sam.encode_image(black_box(&image));
            black_box(out);
        });
    });

    group.bench_function("candle_decoder_only", |b| {
        b.iter(|| {
            let (sparse, dense) = candle_prompt
                .forward(Some((&candle_pts, &candle_labels)), None, None)
                .unwrap();
            let image_pe = candle_prompt.get_dense_pe().unwrap();
            let (masks, iou) = candle_decoder
                .forward(black_box(&candle_embed), &image_pe, &sparse, &dense, true)
                .unwrap();
            black_box((masks, iou));
        });
    });
    group.bench_function("rlx_decoder_only", |b| {
        b.iter(|| {
            let pred = rlx_sam
                .predict_masks(
                    black_box(&rlx_embed),
                    Some((&pt_coords, &pt_labels)),
                    None,
                    None,
                    true,
                )
                .unwrap();
            black_box(pred);
        });
    });

    group.bench_function("candle_full_forward", |b| {
        b.iter(|| {
            let emb = candle_encoder.forward(black_box(&candle_img)).unwrap();
            let (sparse, dense) = candle_prompt
                .forward(Some((&candle_pts, &candle_labels)), None, None)
                .unwrap();
            let image_pe = candle_prompt.get_dense_pe().unwrap();
            let (masks, iou) = candle_decoder
                .forward(&emb, &image_pe, &sparse, &dense, true)
                .unwrap();
            black_box((masks, iou));
        });
    });
    group.bench_function("rlx_full_forward", |b| {
        b.iter(|| {
            let emb = rlx_sam.encode_image(black_box(&image));
            let pred = rlx_sam
                .predict_masks(&emb, Some((&pt_coords, &pt_labels)), None, None, true)
                .unwrap();
            black_box(pred);
        });
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

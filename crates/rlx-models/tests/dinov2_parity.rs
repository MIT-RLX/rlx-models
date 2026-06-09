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

//! DINOv2 numerical parity test: rlx-models vs candle-transformers.
//!
//! ## Running
//!
//! Provide a path to `dinov2_vits14.safetensors` (the same checkpoint
//! candle's `dinov2` example uses, from `lmz/candle-dino-v2`). Either
//! download it manually or via the HF Hub CLI:
//!
//! ```bash
//! huggingface-cli download lmz/candle-dino-v2 dinov2_vits14.safetensors
//! ```
//!
//! Then:
//!
//! ```bash
//! RLX_DINOV2_WEIGHTS=/path/to/dinov2_vits14.safetensors \
//!   cargo test -p rlx-models --features parity-candle --release dinov2_parity_vs_candle
//! ```
//!
//! Without `RLX_DINOV2_WEIGHTS` set, the test is skipped (printing a
//! note) so CI without the checkpoint stays green.
//!
//! ## What "parity" means
//!
//! Both implementations are fed the same `[1, 3, 518, 518]` input
//! tensor (so candle's `interpolate_pos_encoding` short-circuits) and
//! both run on CPU. We compare the post-`head` logits row by row and
//! assert `max_abs_diff <= 5e-3`. Tolerance accounts for differing
//! reduction orders in softmax-attention and matmul.

#![cfg(feature = "parity-candle")]

mod compile_support;

use anyhow::Result;
use candle_core::{DType as CDType, Device as CDevice, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::dinov2 as candle_dinov2;
use rlx_models::WeightMap;
use rlx_models::dinov2::{DinoV2Config, assemble_hidden, build_dinov2_graph_sized};
use rlx_runtime::{Device, Session};

const IMG_SIZE: usize = 518;
const PATCH_SIZE: usize = 14;
const PARITY_TOL: f32 = 5e-3;

fn weights_path() -> Option<String> {
    rlx_ir::env::var("RLX_DINOV2_WEIGHTS")
}

/// Deterministic [1, 3, 518, 518] f32 tensor in ImageNet-normalized
/// range. We synthesize a smooth sine pattern instead of loading a real
/// image — this keeps the test self-contained but still exercises every
/// channel across the full spatial domain.
fn synthesize_image() -> Vec<f32> {
    let n = 3 * IMG_SIZE * IMG_SIZE;
    let mut v = vec![0f32; n];
    for c in 0..3 {
        let phase = (c as f32) * 0.7;
        for y in 0..IMG_SIZE {
            for x in 0..IMG_SIZE {
                let fx = x as f32 / IMG_SIZE as f32;
                let fy = y as f32 / IMG_SIZE as f32;
                // Range roughly [-1, 1] — plausible post-ImageNet-norm.
                let val = (6.28 * fx + phase).sin() * (3.14 * fy + phase).cos();
                v[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = val;
            }
        }
    }
    v
}

fn run_candle(weights: &str, image: &[f32]) -> Result<Vec<f32>> {
    let device = CDevice::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], CDType::F32, &device)? };
    let model = candle_dinov2::vit_small(vb)?;
    let img_t = Tensor::from_slice(image, (1, 3, IMG_SIZE, IMG_SIZE), &device)?;
    let logits = model.forward(&img_t)?;
    let logits = logits.i(0)?.to_vec1::<f32>()?;
    Ok(logits)
}

fn run_rlx(weights: &str, image: &[f32]) -> Result<Vec<f32>> {
    let cfg = DinoV2Config::vit_small(IMG_SIZE);
    let mut wm = WeightMap::from_file(weights)?;
    let (graph, params, pre) = build_dinov2_graph_sized(&cfg, &mut wm, /*batch*/ 1)?;

    let mut compiled =
        rlx_models::flow_util::compile_graph_encoder_with_params(Device::Cpu, graph, params)?;

    let hidden = assemble_hidden(&pre, image, 1, PATCH_SIZE, IMG_SIZE)?;
    let outputs = compiled.run(&[("hidden", hidden.as_slice())]);
    let logits = outputs.into_iter().next().unwrap_or_default();
    Ok(logits)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "logit length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut max = 0f32;
    let mut idx = 0;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max {
            max = d;
            idx = i;
        }
    }
    (max, idx)
}

/// Load an image from disk and convert to a `[3, 518, 518]` NCHW f32
/// tensor using the *exact* same pipeline as candle's
/// `imagenet::load_image518` (`resize_to_fill` with `FilterType::Triangle`
/// + ImageNet mean/std). This guarantees both implementations receive
/// bit-identical input tensors, so any parity gap is downstream of I/O.
fn load_image_518(path: &str) -> Result<Vec<f32>> {
    use image::imageops::FilterType;
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    let img = image::ImageReader::open(path)?
        .decode()?
        .resize_to_fill(IMG_SIZE as u32, IMG_SIZE as u32, FilterType::Triangle)
        .to_rgb8();
    let raw = img.into_raw(); // HWC u8

    let mut nchw = vec![0f32; 3 * IMG_SIZE * IMG_SIZE];
    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            for c in 0..3 {
                let v = raw[(y * IMG_SIZE + x) * 3 + c] as f32 / 255.0;
                nchw[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = (v - MEAN[c]) / STD[c];
            }
        }
    }
    Ok(nchw)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / s).collect()
}

fn top_k(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    idx.sort_by(|a, b| b.1.total_cmp(&a.1));
    idx.into_iter().take(k).collect()
}

fn print_top5(label: &str, logits: &[f32]) {
    let probs = softmax(logits);
    eprintln!("[{label}] top-5:");
    for (rank, (idx, p)) in top_k(&probs, 5).into_iter().enumerate() {
        eprintln!("  {:>2}. class {:>4}  {:>6.2}%", rank + 1, idx, 100.0 * p);
    }
}

#[test]
fn dinov2_parity_with_image() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping (set RLX_DINOV2_WEIGHTS)");
        return Ok(());
    };
    // Default to candle's bike.jpg if no override.
    let image_path = rlx_ir::env::var("RLX_DINOV2_IMAGE").unwrap_or_else(|| {
        "/Users/Shared/candle/candle-examples/examples/yolo-v8/assets/bike.jpg".into()
    });
    if !std::path::Path::new(&image_path).exists() {
        eprintln!("skipping — image not found at {image_path}");
        return Ok(());
    }

    let image = load_image_518(&image_path)?;
    eprintln!(
        "[dinov2 parity / image] using {} ({} bytes NCHW tensor)",
        image_path,
        image.len() * 4
    );

    let candle_logits = run_candle(&weights, &image)?;
    let rlx_logits = run_rlx(&weights, &image)?;

    print_top5("candle", &candle_logits);
    print_top5("rlx", &rlx_logits);

    let (diff, idx) = max_abs_diff(&rlx_logits, &candle_logits);
    eprintln!(
        "[dinov2 parity / image] max |Δlogit| = {diff:.4e} at index {idx} (rlx={}, candle={})",
        rlx_logits[idx], candle_logits[idx]
    );

    // Same tolerance as the synthetic test; if the GELU fix held, this
    // should also land at the f32 noise floor.
    assert!(
        diff <= PARITY_TOL,
        "rlx vs candle on real image: max |Δ| = {diff:.4e} > {PARITY_TOL:.4e}"
    );

    // Strong functional check: both implementations must agree on the
    // ImageNet top-1 prediction.
    let r_top1 = top_k(&softmax(&rlx_logits), 1)[0].0;
    let c_top1 = top_k(&softmax(&candle_logits), 1)[0].0;
    assert_eq!(
        r_top1, c_top1,
        "top-1 disagrees: rlx={r_top1}, candle={c_top1}"
    );

    Ok(())
}

#[test]
fn dinov2_parity_vs_candle() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!(
            "skipping dinov2 parity test — set RLX_DINOV2_WEIGHTS=/path/to/dinov2_vits14.safetensors"
        );
        return Ok(());
    };

    let image = synthesize_image();
    let candle_logits = run_candle(&weights, &image)?;
    let rlx_logits = run_rlx(&weights, &image)?;

    let (diff, idx) = max_abs_diff(&rlx_logits, &candle_logits);
    eprintln!(
        "[dinov2 parity] {} logits compared; max |Δ| = {diff:.4e} at index {idx} (rlx={}, candle={})",
        rlx_logits.len(),
        rlx_logits[idx],
        candle_logits[idx],
    );

    assert!(
        diff <= PARITY_TOL,
        "rlx vs candle DINOv2 logits diverge: max |Δ| = {diff:.4e} > {PARITY_TOL:.4e}"
    );
    Ok(())
}

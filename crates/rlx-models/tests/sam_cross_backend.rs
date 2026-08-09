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

//! Cross-backend comparison: run SAM ViT-B encoder on every available
//! backend and report pairwise cosine similarity + L2 distance on the
//! 256×64×64 image embeddings and the post-decoder 3×256×256 mask
//! logits.
//!
//! Each pair gets:
//!   - cosine sim          (1.0 = bit-identical direction)
//!   - max |Δ|             (worst-case pointwise mismatch)
//!   - mean |Δ|            (average mismatch)
//!   - relative L2         (‖a-b‖₂ / ‖a‖₂)
//!   - binary mask agreement %

#![cfg(feature = "parity-candle")]

mod compile_support;

use anyhow::Result;
use rlx_models::sam::{Device, SAM_IMG_SIZE, Sam, SamConfig};

const POINTS: &[f32] = &[512.0, 512.0];
const LABELS: &[f32] = &[1.0];

// 6.28 / 3.14 are deliberate 2-dp values, not TAU/PI: they define the
// synthetic parity input, and changing them would change every dumped
// reference blob compared against here.
#[allow(clippy::approx_constant)]
fn synth_image() -> Vec<f32> {
    let n = 3 * SAM_IMG_SIZE * SAM_IMG_SIZE;
    let mut v = vec![0f32; n];
    for c in 0..3 {
        let phase = (c as f32) * 0.7;
        for y in 0..SAM_IMG_SIZE {
            for x in 0..SAM_IMG_SIZE {
                v[c * SAM_IMG_SIZE * SAM_IMG_SIZE + y * SAM_IMG_SIZE + x] =
                    (6.28 * x as f32 / SAM_IMG_SIZE as f32 + phase).sin()
                        * (3.14 * y as f32 / SAM_IMG_SIZE as f32 + phase).cos();
            }
        }
    }
    v
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-12)) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).sum();
    (s / a.len() as f64) as f32
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let diff: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let ref_norm: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    ((diff.sqrt()) / (ref_norm + 1e-12)) as f32
}

fn binary_agreement(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let agree = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x > 0.0) == (**y > 0.0))
        .count();
    agree as f64 / n as f64
}

fn run_backend(
    label: &str,
    device: Device,
    weights: &str,
    image: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    eprintln!("[{label}] running on {device:?}...");
    let mut sam = Sam::from_safetensors_on(weights, SamConfig::vit_b(), device)?;
    let emb = sam.encode_image(image);
    let pred = sam.predict_masks(&emb, Some((POINTS, LABELS)), None, None, true)?;
    eprintln!(
        "[{label}] embed_len={} mask_len={} num_masks={}",
        emb.len(),
        pred.mask_logits.len(),
        pred.num_masks
    );
    Ok((emb, pred.mask_logits))
}

fn report_pair(name: &str, a: (&[f32], &[f32]), b: (&[f32], &[f32])) {
    let (a_emb, a_msk) = a;
    let (b_emb, b_msk) = b;
    let emb_cos = cosine(a_emb, b_emb);
    let emb_max = max_abs(a_emb, b_emb);
    let emb_mean = mean_abs(a_emb, b_emb);
    let emb_l2 = rel_l2(a_emb, b_emb);
    let mask_cos = cosine(a_msk, b_msk);
    let mask_max = max_abs(a_msk, b_msk);
    let mask_mean = mean_abs(a_msk, b_msk);
    let mask_l2 = rel_l2(a_msk, b_msk);
    let mask_agree = binary_agreement(a_msk, b_msk);
    eprintln!();
    eprintln!("=== {name} ===");
    eprintln!(
        "  image_emb  cos={:.9}  max={:.4e}  mean={:.4e}  rel_l2={:.4e}",
        emb_cos, emb_max, emb_mean, emb_l2
    );
    eprintln!(
        "  mask_lgts  cos={:.9}  max={:.4e}  mean={:.4e}  rel_l2={:.4e}",
        mask_cos, mask_max, mask_mean, mask_l2
    );
    eprintln!("  binary_mask_agreement: {:.6}%", 100.0 * mask_agree);
}

#[test]
fn sam_cross_backend_distance() -> Result<()> {
    let Some(weights) = rlx_ir::env::var("RLX_SAM_WEIGHTS") else {
        eprintln!("skipping (set RLX_SAM_WEIGHTS)");
        return Ok(());
    };
    let image = synth_image();

    // Run each backend then drop the Sam *before* creating the next.
    // Some backends share global state (Metal kernels cache, MLX device
    // singleton) that can leak across instances if they coexist.
    let (cpu_emb, cpu_msk) = { run_backend("cpu", Device::Cpu, &weights, &image)? };

    #[cfg(feature = "metal")]
    let (metal_emb, metal_msk) = { run_backend("metal", Device::Metal, &weights, &image)? };
    #[cfg(not(feature = "metal"))]
    let (metal_emb, metal_msk): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());

    #[cfg(feature = "mlx")]
    let (mlx_emb, mlx_msk) = { run_backend("mlx", Device::Mlx, &weights, &image)? };
    #[cfg(not(feature = "mlx"))]
    let (mlx_emb, mlx_msk): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());

    #[cfg(feature = "metal")]
    report_pair(
        "CPU vs METAL",
        (&cpu_emb, &cpu_msk),
        (&metal_emb, &metal_msk),
    );
    #[cfg(feature = "mlx")]
    report_pair("CPU vs MLX", (&cpu_emb, &cpu_msk), (&mlx_emb, &mlx_msk));
    #[cfg(all(feature = "metal", feature = "mlx"))]
    report_pair(
        "METAL vs MLX",
        (&metal_emb, &metal_msk),
        (&mlx_emb, &mlx_msk),
    );

    Ok(())
}

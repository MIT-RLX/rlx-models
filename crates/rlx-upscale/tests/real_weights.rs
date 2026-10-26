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

//! Real-checkpoint tests.
//!
//! These need weights, which this repository does not ship. Point
//! `RLX_UPSCALE_MODELS` at a directory of `.pth` files and they run over every
//! one; without it they skip.
//!
//! ```text
//! RLX_UPSCALE_MODELS=~/models cargo test -p rlx-upscale --test real_weights -- --nocapture
//! ```
//!
//! # What "correct" means without a PyTorch oracle
//!
//! Running the reference is not possible here, so the check is a property the
//! *task* guarantees rather than a golden output: **downscaling a ×N result by
//! N must line up with the input**. A transposed kernel, a mis-ordered pixel
//! shuffle or a botched reparameterization all move energy across the image,
//! and that survives downscaling.
//!
//! The metric is **correlation**, not PSNR, and the distinction is the whole
//! point. Two kinds of model are in scope:
//!
//! * *Efficient SR* models trained on clean bicubic degradation reproduce the
//!   input almost exactly — SPANV2 round-trips at 42 dB.
//! * *Restoration* models are trained to *change* the image: remove compression
//!   artifacts, re-saturate, denoise. A 2× anime SPAN round-trips at 12 dB
//!   while producing a visibly better picture than the one that scores 42.
//!
//! A ×1 model has nothing to downscale and is compared to the input directly.
//!
//! PSNR cannot tell the second kind from a broken port; correlation can, since
//! it is invariant to the per-channel gain and offset a restoration model
//! applies but is destroyed by any spatial scrambling. PSNR is still printed,
//! because a sudden change in it is worth looking at.
//!
//! Measured on release checkpoints, for calibration:
//!
//! | model | r | PSNR |
//! |---|---|---|
//! | SPANV2 ×4 (`team22_spanv2_c2`, NTIRE 2026 winner) | 0.9998 | 50.6 dB |
//! | RealPLKSR ×2 dysample+layernorm (`2xPublic_…_real`) | 0.9963 | 38.5 dB |
//! | Compact ×4 (`realesr-general-x4v3`) | 0.9977 | 38.3 dB |
//! | SPAN ×2 anime restoration (`2xHFA2kSPAN`) | 0.9496 | 13.8 dB |

use rlx_runtime::Device;
use rlx_upscale::{UpscaleOptions, Upscaler};

/// A photograph-like synthetic image: smooth low-frequency content, a ramp,
/// mild texture and one hard edge.
///
/// Deliberately *not* a checkerboard. A hard 8px checkerboard through a ×2
/// restoration model is so far outside its training distribution that the
/// result says more about the test than the port.
fn test_image(w: usize, h: usize) -> Vec<u8> {
    let mut px = vec![0u8; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let edge = if fx > 0.6 && fy > 0.4 { 0.18 } else { 0.0 };
            let v = [
                0.5 + 0.35 * (fx * 6.0).sin() * (fy * 4.0).cos(),
                0.45 + 0.3 * fx + 0.15 * (fy * 9.0).sin(),
                0.5 + 0.3 * ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt(),
            ];
            for (c, val) in v.iter().enumerate() {
                px[i + c] = ((val + edge).clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    px
}

/// Box-downscale by an integer factor.
fn downscale(px: &[u8], w: usize, h: usize, f: usize) -> Vec<u8> {
    let (ow, oh) = (w / f, h / f);
    let mut out = vec![0u8; 3 * ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            for c in 0..3 {
                let mut acc = 0u32;
                for dy in 0..f {
                    for dx in 0..f {
                        acc += px[(((y * f + dy) * w) + x * f + dx) * 3 + c] as u32;
                    }
                }
                out[(y * ow + x) * 3 + c] = (acc / (f * f) as u32) as u8;
            }
        }
    }
    out
}

/// Pearson correlation between two images, per channel, averaged.
///
/// Invariant to the per-channel `a·x + b` a restoration model applies, and
/// destroyed by any spatial rearrangement.
fn correlation(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() / 3;
    let mut total = 0.0;
    for c in 0..3 {
        let xs: Vec<f64> = (0..n).map(|i| a[i * 3 + c] as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| b[i * 3 + c] as f64).collect();
        let mx = xs.iter().sum::<f64>() / n as f64;
        let my = ys.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let (mut dx, mut dy) = (0.0, 0.0);
        for i in 0..n {
            let (u, v) = (xs[i] - mx, ys[i] - my);
            num += u * v;
            dx += u * u;
            dy += v * v;
        }
        // A constant channel correlates with nothing; treat it as agreement
        // only if the other side is constant too.
        total += if dx <= f64::EPSILON || dy <= f64::EPSILON {
            f64::from(dx <= f64::EPSILON && dy <= f64::EPSILON)
        } else {
            num / (dx * dy).sqrt()
        };
    }
    total / 3.0
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

#[test]
fn real_checkpoints_round_trip_through_a_downscale() {
    let Some(dir) = rlx_ir::env::var_os("RLX_UPSCALE_MODELS") else {
        eprintln!("RLX_UPSCALE_MODELS not set; skipping real-weight tests");
        return;
    };
    let mut found = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("reading RLX_UPSCALE_MODELS")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("pth" | "pt" | "safetensors")
            )
        })
        .collect();
    entries.sort();

    // Tier 2 on CPU is minutes per model; `RLX_UPSCALE_DEVICE=metal` (or any
    // other backend compiled in) makes the suite practical to run routinely.
    let device = match rlx_ir::env::var_result("RLX_UPSCALE_DEVICE") {
        Ok(name) => rlx_upscale::parse_upscale_device(&name)
            .unwrap_or_else(|e| panic!("RLX_UPSCALE_DEVICE={name}: {e:#}")),
        Err(_) => Device::Cpu,
    };
    eprintln!("running on {device:?}");

    for path in entries {
        let mut up = match Upscaler::open(&path, device, UpscaleOptions::default()) {
            Ok(u) => u,
            Err(e) => panic!("{}: {e:#}", path.display()),
        };
        let cfg = up.config().clone();
        let scale = cfg.scale;

        // A whole number of windows so the transformers are happy. The
        // transformer families get a smaller canvas: a 12-group DRCT-L over a
        // 192² tile is minutes per image on CPU, and the check does not get any
        // sharper for the extra pixels.
        let m = cfg.size_multiple().max(8);
        let (w, h) = if cfg.arch.is_transformer() {
            ((128 / m).max(2) * m, (96 / m).max(2) * m)
        } else {
            (m * 12, m * 8)
        };
        let src = test_image(w, h);
        let (out, ow, oh) = up
            .upscale_rgb8(&src, w, h)
            .unwrap_or_else(|e| panic!("{}: {e:#}", cfg.summary()));
        assert_eq!((ow, oh), (w * scale, h * scale));

        // A file-backed `Upscaler` hands its tensors to the graph and then lets
        // them go — holding a second copy for the life of the object costs
        // 231 MB on DRCT-L for a recompile that usually never comes.
        assert!(
            !up.holds_weights(),
            "{}: still holding a copy of the weights after compiling",
            cfg.summary()
        );

        // A ×1 restoration model has nothing to downscale — it is already the
        // input's size, so compare directly. Skipping it, as this used to,
        // meant the denoising path had no real-weight coverage at all.
        let back = if scale == 1 {
            out.clone()
        } else {
            downscale(&out, ow, oh, scale)
        };
        let r = correlation(&back, &src);
        let db = psnr(&back, &src);
        eprintln!(
            "{:<44} {w}×{h} ×{scale}  round-trip r={r:.4}  ({db:.1} dB)",
            cfg.summary()
        );
        // Collect every model before asserting: one bad checkpoint should not
        // hide the results for the rest.
        if r < 0.93 {
            failures.push(format!(
                "{} ({}): r={r:.4} ({db:.1} dB)",
                cfg.summary(),
                path.file_name().unwrap_or_default().to_string_lossy()
            ));
        }
        found += 1;
    }
    assert!(found > 0, "no usable checkpoints in {dir:?}");
    assert!(
        failures.is_empty(),
        "downscaling these results does not line up with the input. A low \
         correlation is spatial, not tonal, so this is a structurally wrong \
         port rather than a model that merely renders differently:\n  {}",
        failures.join("\n  ")
    );
}

// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// DINOv2 (ViT-S/B/L @ patch 14) — end-to-end through the
// `DinoV2Runner` builder. Resizes + normalizes the image to
// `img_size × img_size` ImageNet-normalized NCHW, assembles the
// CLS + register + patch tokens host-side, runs the compiled graph
// on the chosen device, returns either classifier logits (when the
// checkpoint includes a head) or post-LN feature tokens.
//
// Usage:
//   cargo run --release -p rlx-models --features metal \
//       --example run_dinov2 -- /path/to/dinov2_vitb14.safetensors
//
//   RLX_DINOV2_VARIANT  = small | base | large   (default base)
//   RLX_DINOV2_IMG_SIZE = 518 (default), must be a multiple of 14
//
// Equivalent CLI call:
//   rlx-dinov2 (or `rlx-run dinov2`) --weights dinov2_vitb14.safetensors --device metal \
//                  --variant base --img-size 518

use anyhow::{Result, anyhow};
use rlx_models::run::{DinoV2Output, DinoV2Runner, DinoV2Variant};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let weights = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: run_dinov2 <weights.safetensors>"))?;

    let variant = match rlx_ir::env::var("RLX_DINOV2_VARIANT")
        .unwrap_or_else(|| "base".into())
        .as_str()
    {
        "small" => DinoV2Variant::Small,
        "base" => DinoV2Variant::Base,
        "large" => DinoV2Variant::Large,
        other => anyhow::bail!("RLX_DINOV2_VARIANT must be small|base|large, got {other}"),
    };
    let img_size: usize = rlx_ir::env::var("RLX_DINOV2_IMG_SIZE")
        .and_then(|s| s.parse().ok())
        .unwrap_or(518);

    let mut runner = DinoV2Runner::builder()
        .weights(&weights)
        .device(Device::Metal)
        .variant(variant)
        .img_size(img_size)
        .batch(1)
        .build()?;
    let cfg = runner.config();
    eprintln!(
        "[dinov2] compiled — variant={variant:?} hidden={} layers={} heads={} num_classes={} img={}",
        cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_classes, img_size,
    );

    // Synthetic image — replace with `image::open(p)?.to_rgb8().as_raw()`
    // for a real picture (any HW; the runner resizes internally).
    let (h_in, w_in) = (img_size, img_size);
    let mut rgb = vec![0u8; h_in * w_in * 3];
    for y in 0..h_in {
        for x in 0..w_in {
            let base = (y * w_in + x) * 3;
            rgb[base] = (x * 255 / w_in) as u8;
            rgb[base + 1] = (y * 255 / h_in) as u8;
            rgb[base + 2] = ((x + y) * 127 / (h_in + w_in)) as u8;
        }
    }

    let t0 = std::time::Instant::now();
    let out = runner.predict_image(&rgb, h_in, w_in)?;
    let dt = t0.elapsed();
    match out {
        DinoV2Output::Logits {
            per_batch,
            num_classes,
        } => {
            eprintln!(
                "[dinov2] logits in {dt:?} — classes={num_classes}, batches={}",
                per_batch.len()
            );
            for (b, logits) in per_batch.iter().enumerate() {
                let (top1, top1_val) = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap();
                eprintln!("  batch[{b}] top1={top1} logit={top1_val:.3}");
            }
        }
        DinoV2Output::Tokens {
            per_batch,
            seq,
            hidden,
        } => {
            eprintln!(
                "[dinov2] tokens in {dt:?} — seq={seq} hidden={hidden}, batches={}",
                per_batch.len()
            );
            for (b, toks) in per_batch.iter().enumerate() {
                let cls = &toks[..hidden];
                let cls_norm: f32 = cls.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("  batch[{b}] ||cls||₂ = {cls_norm:.3}");
            }
        }
    }
    Ok(())
}

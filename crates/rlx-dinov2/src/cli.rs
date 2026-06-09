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

// RLX CLI
use crate::{DinoV2Output, DinoV2Runner, DinoV2Variant};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;
pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut variant_str = "base".to_string();
    let mut img_size = 518usize;
    let mut batch = 1usize;
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                weights = Some(req(args, &mut i)?.into());
            }
            "--device" => device = req(args, &mut i)?,
            "--variant" => variant_str = req(args, &mut i)?,
            "--img-size" => {
                img_size = req(args, &mut i)?.parse().context("--img-size: usize")?;
            }
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch: usize")?,
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("rlx-dinov2 — see README for flags");
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_standard_device("dinov2", &device)?;
    let variant = match variant_str.as_str() {
        "small" | "vit-s" => DinoV2Variant::Small,
        "base" | "vit-b" => DinoV2Variant::Base,
        "large" | "vit-l" => DinoV2Variant::Large,
        other => bail!("--variant: expected small|base|large (got {other})"),
    };

    eprintln!(
        "[rlx-dinov2] dinov2: weights={weights:?} device={device:?} variant={variant:?} img_size={img_size} batch={batch}"
    );
    let mut runner = DinoV2Runner::builder()
        .weights(&weights)
        .device(device)
        .variant(variant)
        .img_size(img_size)
        .batch(batch)
        .build()?;
    eprintln!(
        "[rlx-dinov2] compiled — hidden={} layers={} num_classes={}",
        runner.config().hidden_size,
        runner.config().num_hidden_layers,
        runner.config().num_classes
    );

    if dry {
        eprintln!("[rlx-dinov2] --dry set; skipping forward pass");
        return Ok(());
    }

    // Synthetic image — replace with image::open(...).to_rgb8().as_raw().
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
                "[rlx-dinov2] dinov2 logits in {dt:?} — batch={} classes={}",
                per_batch.len(),
                num_classes
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
                "[rlx-dinov2] dinov2 tokens in {dt:?} — batch={} seq={seq} hidden={hidden}",
                per_batch.len()
            );
            // CLS token (index 0) summary: ||cls||₂
            for (b, toks) in per_batch.iter().enumerate() {
                let cls = &toks[..hidden];
                let norm: f32 = cls.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("  batch[{b}] ||cls||₂ = {norm:.3}");
            }
        }
    }
    Ok(())
}

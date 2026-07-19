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
use crate::Siglip2Runner;
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

/// Entry point for the `rlx-siglip2` binary: parse flags, run zero-shot (or
/// print the image embedding norm when `--labels` is omitted).
pub fn run(args: &[String]) -> Result<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut image: Option<PathBuf> = None;
    let mut labels: Vec<String> = Vec::new();
    let mut raw_prompts = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" | "--weights" => model_dir = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--image" => image = Some(req(args, &mut i)?.into()),
            "--labels" => {
                labels = req(args, &mut i)?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            // Use the labels verbatim as prompts (skip the caption template).
            "--raw-prompts" => raw_prompts = true,
            "--help" | "-h" => {
                eprintln!(
                    "rlx-siglip2 --model-dir <dir> --image <path> --labels \"a,b,c\" \
                     [--raw-prompts] [--device cpu|metal|mlx|cuda|rocm|gpu|vulkan]"
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let model_dir = model_dir.ok_or_else(|| anyhow!("--model-dir is required"))?;
    let device = parse_standard_device("siglip2", &device)?;

    eprintln!("[rlx-siglip2] model_dir={model_dir:?} device={device:?}");
    let mut runner = Siglip2Runner::builder()
        .model_dir(&model_dir)
        .device(device)
        .build()?;
    let cfg = *runner.config();
    eprintln!(
        "[rlx-siglip2] compiled — embed_dim={} vision(L{} w{} h{}) text(L{} w{} h{})",
        cfg.embed_dim,
        cfg.vision.layers,
        cfg.vision.width,
        cfg.vision.heads,
        cfg.text.layers,
        cfg.text.width,
        cfg.text.heads
    );

    let image = image.ok_or_else(|| anyhow!("--image is required"))?;
    let img = image::open(&image).with_context(|| format!("opening image {image:?}"))?;

    if labels.is_empty() {
        let feat = runner.encode_image(&img)?;
        let norm: f32 = feat.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!(
            "[rlx-siglip2] image embedding dim={} ||x||₂={norm:.4}",
            feat.len()
        );
        return Ok(());
    }

    // SigLIP's canonical zero-shot caption template (unless --raw-prompts).
    let prompts: Vec<String> = if raw_prompts {
        labels.clone()
    } else {
        labels
            .iter()
            .map(|l| format!("This is a photo of {l}."))
            .collect()
    };
    let t0 = std::time::Instant::now();
    let logits = runner.zeroshot(&[img], &prompts)?;
    let dt = t0.elapsed();
    // SigLIP: independent per-pair sigmoid, not a softmax over labels.
    let row = &logits[0];
    eprintln!("[rlx-siglip2] zero-shot in {dt:?} (sigmoid match probability):");
    let mut ranked: Vec<usize> = (0..labels.len()).collect();
    ranked.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
    for &idx in &ranked {
        let p = sigmoid(row[idx]);
        println!("  {:>7.3}%  {}", p * 100.0, labels[idx]);
    }
    Ok(())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

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

// RLX CLI — DINOv3 ViT encoder.
use crate::{DinoV3Config, DinoV3Runner};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

/// Parse args and run the `rlx-dinov3` command: load a checkpoint, compile on
/// the chosen device, embed an image (or a synthetic tile), and optionally dump
/// the embedding / token grid. See `--help` for flags.
pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut config_path: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut dump: Option<PathBuf> = None;
    let mut dump_tokens: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut img_size = 224usize;
    let mut batch = 1usize;
    let mut variant = "vitb16".to_string();
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--config" => config_path = Some(req(args, &mut i)?.into()),
            "--image" => image = Some(req(args, &mut i)?.into()),
            "--dump" => dump = Some(req(args, &mut i)?.into()),
            "--dump-tokens" => dump_tokens = Some(req(args, &mut i)?.into()),
            "--variant" => variant = req(args, &mut i)?,
            "--device" => device = req(args, &mut i)?,
            "--img-size" => {
                img_size = req(args, &mut i)?.parse().context("--img-size: usize")?;
            }
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch: usize")?,
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-dinov3 — Meta DINOv3 ViT encoder (2D-axial RoPE, registers)\n\
                     flags: --weights <model.safetensors> [--config <config.json>] \
                     [--variant vits16|vitb16|vitl16] [--image <path>] [--dump <emb.bin>] \
                     [--dump-tokens <tok.bin>] [--device cpu|metal|mlx|wgpu|...] \
                     [--img-size 224] [--batch N] [--dry]\n\
                     note: the pretrained weights are gated on HuggingFace; convert the \
                     safetensors and pass its config.json via --config for non-default variants."
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_standard_device("dinov3", &device)?;

    let cfg = match &config_path {
        Some(p) => {
            let mut c = DinoV3Config::from_file(p)
                .with_context(|| format!("reading dinov3 config {p:?}"))?;
            c.image_size = img_size;
            c
        }
        None => match variant.as_str() {
            "vits16" => DinoV3Config::vit_s16(img_size),
            "vitb16" => DinoV3Config::vit_b16(img_size),
            "vitl16" => DinoV3Config::vit_l16(img_size),
            other => bail!("unknown --variant {other} (use vits16|vitb16|vitl16 or --config)"),
        },
    };

    eprintln!(
        "[rlx-dinov3] weights={weights:?} device={device:?} img_size={img_size} batch={batch}"
    );
    let mut runner = DinoV3Runner::builder()
        .weights(&weights)
        .device(device)
        .img_size(img_size)
        .batch(batch)
        .config(cfg)
        .build()?;
    eprintln!(
        "[rlx-dinov3] compiled — hidden={} layers={} heads={} reg={} gated_mlp={} seq={}",
        runner.config().hidden_size,
        runner.config().num_hidden_layers,
        runner.config().num_attention_heads,
        runner.config().num_register_tokens,
        runner.config().use_gated_mlp,
        runner.config().seq_len(),
    );

    if dry {
        eprintln!("[rlx-dinov3] --dry set; skipping forward pass");
        return Ok(());
    }

    let (rgb, h_in, w_in) = match &image {
        Some(path) => {
            let img = image::open(path)
                .with_context(|| format!("opening image {path:?}"))?
                .to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            (img.into_raw(), h, w)
        }
        None => {
            let (h, w) = (img_size, img_size);
            let mut rgb = vec![0u8; h * w * 3];
            for y in 0..h {
                for x in 0..w {
                    let base = (y * w + x) * 3;
                    rgb[base] = (x * 255 / w) as u8;
                    rgb[base + 1] = (y * 255 / h) as u8;
                    rgb[base + 2] = ((x + y) * 127 / (h + w)) as u8;
                }
            }
            (rgb, h, w)
        }
    };

    let t0 = std::time::Instant::now();
    let out = runner.predict_image(&rgb, h_in, w_in)?;
    let dt = t0.elapsed();
    eprintln!(
        "[rlx-dinov3] forward in {dt:?} — batch={} seq={} hidden={}",
        out.embeddings.len(),
        out.seq,
        out.hidden
    );
    for (b, emb) in out.embeddings.iter().enumerate() {
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let head: Vec<String> = emb.iter().take(6).map(|v| format!("{v:.4}")).collect();
        eprintln!(
            "  batch[{b}] ||emb||₂={norm:.4} emb[..6]=[{}]",
            head.join(", ")
        );
    }

    if let Some(path) = &dump {
        let bytes: Vec<u8> = out.embeddings[0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(path, &bytes).with_context(|| format!("writing embedding to {path:?}"))?;
        eprintln!(
            "[rlx-dinov3] wrote {}-d embedding to {path:?}",
            out.embeddings[0].len()
        );
    }
    if let Some(path) = &dump_tokens {
        let bytes: Vec<u8> = out.tokens[0].iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, &bytes).with_context(|| format!("writing tokens to {path:?}"))?;
        eprintln!(
            "[rlx-dinov3] wrote {}x{} tokens to {path:?}",
            out.seq, out.hidden
        );
    }
    Ok(())
}

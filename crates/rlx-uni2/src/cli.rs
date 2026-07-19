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

//! Command-line entry point for the `rlx-uni2` binary — load a UNI2-h
//! checkpoint, embed an image (or a deterministic synthetic tile), and print
//! the pooled `[CLS]` embedding. See [`run`] for the flag list.

use crate::Uni2Runner;
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

const HELP: &str = "\
rlx-uni2 — MahmoodLab/UNI2-h ViT-H/14 pathology encoder

Usage: rlx-uni2 --weights <model.safetensors> [options]

Options:
  --weights <path>   Model weights (.safetensors). Convert the gated
                     pytorch_model.bin to safetensors first — see the README.
  --image <path>     Input image (any size; resized + ImageNet-normalized to
                     224). Omit to run a deterministic synthetic tile.
  --device <dev>     cpu | metal | mlx | cuda | rocm | gpu (wgpu) | vulkan
                     [default: cpu]
  --img-size <n>     Square input size; must be a multiple of 14 [default: 224]
  --batch <n>        Batch size [default: 1]
  --dump <path>      Write the [1536] CLS embedding as little-endian f32
  --dump-tokens <p>  Write the full [seq x 1536] token grid as little-endian f32
  --layers <n>       (debug) Truncate to the first n transformer blocks
  --dry              Compile only; skip the forward pass
  -h, --help         Show this help";

/// Parse `args` and run one UNI2-h forward pass.
///
/// `--weights` is required. With no `--image`, a deterministic synthetic
/// gradient tile is used so the binary is runnable without an input file.
/// Progress and the resulting embedding summary are printed to stderr.
pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut dump: Option<PathBuf> = None;
    let mut dump_tokens: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut img_size = 224usize;
    let mut batch = 1usize;
    let mut layers: Option<usize> = None;
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--image" => image = Some(req(args, &mut i)?.into()),
            "--dump" => dump = Some(req(args, &mut i)?.into()),
            "--dump-tokens" => dump_tokens = Some(req(args, &mut i)?.into()),
            "--layers" => {
                layers = Some(req(args, &mut i)?.parse().context("--layers: usize")?);
            }
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
                eprintln!("{HELP}");
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_standard_device("uni2", &device)?;

    eprintln!("[rlx-uni2] weights={weights:?} device={device:?} img_size={img_size} batch={batch}");
    let mut builder = Uni2Runner::builder()
        .weights(&weights)
        .device(device)
        .img_size(img_size)
        .batch(batch);
    if let Some(n) = layers {
        // Debug/bring-up: run a truncated stack (uses blocks.0..n of the checkpoint).
        let mut cfg = crate::Uni2Config::uni2_h(img_size);
        cfg.num_hidden_layers = n;
        builder = builder.config(cfg);
        eprintln!("[rlx-uni2] DEBUG: truncated to {n} layers");
    }
    let mut runner = builder.build()?;
    eprintln!(
        "[rlx-uni2] compiled — hidden={} layers={} heads={} reg_tokens={} seq={}",
        runner.config().hidden_size,
        runner.config().num_hidden_layers,
        runner.config().num_attention_heads,
        runner.config().num_register_tokens,
        runner.config().seq_len(),
    );

    if dry {
        eprintln!("[rlx-uni2] --dry set; skipping forward pass");
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
            // Deterministic synthetic gradient tile.
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
        "[rlx-uni2] forward in {dt:?} — batch={} seq={} hidden={}",
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

    // Optional: dump the batch-0 embedding as little-endian f32 (parity checks).
    if let Some(path) = &dump {
        let bytes: Vec<u8> = out.embeddings[0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(path, &bytes).with_context(|| format!("writing embedding to {path:?}"))?;
        eprintln!(
            "[rlx-uni2] wrote {}-d embedding to {path:?}",
            out.embeddings[0].len()
        );
    }
    // Optional: dump the full [seq · hidden] token sequence (per-row parity).
    if let Some(path) = &dump_tokens {
        let bytes: Vec<u8> = out.tokens[0].iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, &bytes).with_context(|| format!("writing tokens to {path:?}"))?;
        eprintln!(
            "[rlx-uni2] wrote {}x{} tokens to {path:?}",
            out.seq, out.hidden
        );
    }
    Ok(())
}

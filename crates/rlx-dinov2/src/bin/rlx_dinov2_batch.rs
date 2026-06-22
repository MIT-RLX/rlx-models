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
//! Batch DINOv2 feature extractor.
//!
//! Reads every `view_NN.png` (or `*.png`) in a directory, runs the
//! ViT encoder once per image, and writes a `view_NN.safetensors` to an
//! output directory. Each output holds a single tensor named
//! `"features"` with shape `[h_patches, w_patches, hidden]` — exactly
//! what `rlx-splat-anim cluster-by-features` expects.
//!
//! Usage:
//! ```bash
//! cargo run --release --features metal --bin rlx-dinov2-batch -- \
//!     --weights /Users/Shared/rlx-models/weights/dinov2/dinov2_vitl14.meta.safetensors \
//!     --variant large --img-size 518 \
//!     --views /Users/Shared/ant/anatomy/views \
//!     --out   /Users/Shared/ant/anatomy/features
//! ```

use anyhow::{Context, Result, anyhow, bail};
use rlx_dinov2::{DinoV2Output, DinoV2Runner, DinoV2Variant};
use rlx_runtime::Device;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::BTreeMap;
use std::path::PathBuf;

struct Args {
    weights: PathBuf,
    variant: DinoV2Variant,
    img_size: usize,
    device: Device,
    views: PathBuf,
    out: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut weights: Option<PathBuf> = None;
    let mut variant = DinoV2Variant::Large;
    let mut img_size: usize = 518;
    let mut device_name = "metal".to_string();
    let mut views: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--weights" => {
                weights = Some(argv[i + 1].clone().into());
                i += 2;
            }
            "--variant" => {
                variant = match argv[i + 1].as_str() {
                    "small" => DinoV2Variant::Small,
                    "base" => DinoV2Variant::Base,
                    "large" => DinoV2Variant::Large,
                    o => bail!("--variant must be small|base|large, got {o}"),
                };
                i += 2;
            }
            "--img-size" => {
                img_size = argv[i + 1].parse()?;
                i += 2;
            }
            "--device" => {
                device_name = argv[i + 1].clone();
                i += 2;
            }
            "--views" => {
                views = Some(argv[i + 1].clone().into());
                i += 2;
            }
            "--out" => {
                out = Some(argv[i + 1].clone().into());
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-dinov2-batch — flags:\n\
                     \t--weights PATH           DINOv2 safetensors (Meta naming)\n\
                     \t--variant small|base|large   (default large)\n\
                     \t--img-size N             multiple of 14 (default 518)\n\
                     \t--device cpu|metal|mlx|cuda|rocm|gpu (default metal)\n\
                     \t--views DIR              input PNG directory\n\
                     \t--out   DIR              output safetensors directory"
                );
                std::process::exit(0);
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "gpu" => Device::Gpu,
        o => bail!("--device {o} not supported"),
    };
    Ok(Args {
        weights: weights.ok_or_else(|| anyhow!("--weights is required"))?,
        variant,
        img_size,
        device,
        views: views.ok_or_else(|| anyhow!("--views is required"))?,
        out: out.ok_or_else(|| anyhow!("--out is required"))?,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out).with_context(|| format!("create {}", args.out.display()))?;

    // Build an encoder-only config explicitly: the ImageNet classifier
    // head isn't shipped with the HF DINOv2 backbone safetensors, and
    // we only want patch tokens anyway.
    use rlx_dinov2::DinoV2Config;
    let mut cfg = match args.variant {
        DinoV2Variant::Small => DinoV2Config::vit_small(args.img_size),
        DinoV2Variant::Base => DinoV2Config::vit_base(args.img_size),
        DinoV2Variant::Large => DinoV2Config::vit_large(args.img_size),
    };
    cfg.num_classes = 0;
    let mut runner = DinoV2Runner::builder()
        .weights(args.weights.to_str().unwrap())
        .device(args.device)
        .config(cfg)
        .batch(1)
        .build()?;
    let cfg = runner.config();
    let patches_per_side = args.img_size / 14;
    let n_patches = patches_per_side * patches_per_side;
    let skip_tokens = 1 + cfg.num_register_tokens; // [cls, registers..., patches...]
    eprintln!(
        "[batch] variant={:?} hidden={} img={} patches/side={} skip_tokens={}",
        args.variant, cfg.hidden_size, args.img_size, patches_per_side, skip_tokens
    );

    // Discover *.png in `views/` sorted by name → `view_NN.safetensors`.
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&args.views)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .filter(|p| {
            // Skip the contact-sheet composite if present.
            p.file_stem().and_then(|s| s.to_str()) != Some("composite")
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no PNGs in {}", args.views.display());
    }
    eprintln!("[batch] processing {} views", paths.len());

    for png in &paths {
        let img = image::open(png)
            .with_context(|| format!("open {}", png.display()))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let rgb = img.as_raw();

        let t = std::time::Instant::now();
        let out = runner.predict_image(rgb, h as usize, w as usize)?;
        let DinoV2Output::Tokens {
            per_batch,
            seq,
            hidden,
        } = out
        else {
            bail!("expected token output for {}", png.display());
        };
        let elapsed = t.elapsed();

        // Extract just the patch tokens — skip [cls + registers].
        let tokens = &per_batch[0];
        if tokens.len() != seq * hidden {
            bail!(
                "token tensor length {} != seq*hidden = {}*{}",
                tokens.len(),
                seq,
                hidden
            );
        }
        if seq < skip_tokens + n_patches {
            bail!("seq={seq} too short for skip={skip_tokens} + patches={n_patches}");
        }
        let patch_bytes_len = n_patches * hidden * 4;
        let mut patch_bytes = Vec::with_capacity(patch_bytes_len);
        for v in &tokens[skip_tokens * hidden..(skip_tokens + n_patches) * hidden] {
            patch_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut tensors = BTreeMap::new();
        let view = TensorView::new(
            Dtype::F32,
            vec![patches_per_side, patches_per_side, hidden],
            &patch_bytes,
        )?;
        tensors.insert("features".to_string(), view);
        let blob = safetensors::serialize(&tensors, None)?;

        let stem = png.file_stem().and_then(|s| s.to_str()).unwrap_or("view");
        let out_path = args.out.join(format!("{stem}.safetensors"));
        std::fs::write(&out_path, &blob)?;
        eprintln!(
            "  {} → {} ({elapsed:?})",
            png.file_name().unwrap().to_string_lossy(),
            out_path.file_name().unwrap().to_string_lossy()
        );
    }
    Ok(())
}

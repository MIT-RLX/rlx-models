// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Batch BioCLIP-2 dense-patch feature extractor — mirrors
// `rlx-dinov2-batch`, but uses BioCLIP-2's bio-aligned vision tower so
// downstream `cluster-by-features` clustering can be compared
// like-for-like with DINOv2.
//
// Usage:
//     rlx-bioclip2-batch \
//         --model-dir weights/bioclip-2 \
//         --device metal \
//         --views <dir of PNGs> \
//         --out <dir for view_NN.safetensors>
//
// Each output safetensors holds a `"features"` tensor of shape
// `[h_patches, w_patches, hidden]` (e.g. 16×16×1024 for ViT-L/14 at
// 224 px), consumable as-is by `rlx-splat-anim cluster-by-features`.

use anyhow::{Context, Result, anyhow, bail};
use rlx_bioclip2::BioClip2Runner;
use rlx_runtime::Device;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::BTreeMap;
use std::path::PathBuf;

struct Args {
    model_dir: PathBuf,
    views: PathBuf,
    out: PathBuf,
    device: Device,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out).with_context(|| format!("create {}", args.out.display()))?;

    let mut runner = BioClip2Runner::builder()
        .model_dir(&args.model_dir)
        .device(args.device)
        .patch_features(true)
        .build()?;

    let cfg = runner.config();
    let image_size = cfg.vision.image_size;
    let patch_size = cfg.vision.patch_size;
    let patches_per_side = image_size / patch_size;
    let n_patches = patches_per_side * patches_per_side;
    let width = cfg.vision.width;
    eprintln!(
        "[bioclip2-batch] img={} patch={} patches/side={} hidden={} device={:?}",
        image_size, patch_size, patches_per_side, width, args.device
    );

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&args.views)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .filter(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .is_none_or(|s| s != "composite")
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no PNGs in {}", args.views.display());
    }
    eprintln!("[bioclip2-batch] processing {} views", paths.len());

    for png in &paths {
        let img = image::open(png).with_context(|| format!("open {}", png.display()))?;
        let t = std::time::Instant::now();
        let raw = runner.encode_image(&img)?;
        let elapsed = t.elapsed();
        if raw.len() != n_patches * width {
            bail!(
                "patch features shape mismatch for {}: got {} floats, expected {}×{}={}",
                png.display(),
                raw.len(),
                n_patches,
                width,
                n_patches * width
            );
        }
        let mut bytes = Vec::with_capacity(raw.len() * 4);
        for v in &raw {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut tensors = BTreeMap::new();
        let view = TensorView::new(
            Dtype::F32,
            vec![patches_per_side, patches_per_side, width],
            &bytes,
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

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut model_dir: Option<PathBuf> = None;
    let mut views: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut device_str = "cpu".to_string();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model-dir" => {
                model_dir = Some(argv[i + 1].clone().into());
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
            "--device" => {
                device_str = argv[i + 1].clone();
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-bioclip2-batch --model-dir <dir> --views <dir> --out <dir> [--device cpu|metal|mlx|cuda|rocm|gpu|vulkan]"
                );
                std::process::exit(0);
            }
            o => bail!("unknown flag: {o}"),
        }
    }
    let device = match device_str.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "gpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        o => bail!("unknown device {o}"),
    };
    Ok(Args {
        model_dir: model_dir.ok_or_else(|| anyhow!("--model-dir required"))?,
        views: views.ok_or_else(|| anyhow!("--views required"))?,
        out: out.ok_or_else(|| anyhow!("--out required"))?,
        device,
    })
}

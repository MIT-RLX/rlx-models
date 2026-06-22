// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Batch SAM v1 mask generator for the rlx-splat-anim pipeline.
//
// Reads `prompts.json` produced by `rlx-splat-anim project-prompts`
// (per-view, per-bone pixel coordinates derived from the rig.json
// pivots) and writes one mask PNG per (view, bone) under
// `<out>/view_<NN>/<bone>.png`. Those masks are what
// `rlx-splat-anim backproject` consumes.
//
// Usage:
//     rlx-sam-batch
//         --weights weights/sam_v1/sam_vit_b_meta.safetensors
//         --views out/dinov2_test/ant_views
//         --prompts out/dinov2_test/ant_prompts.json
//         --out out/sam_masks/ant
//         [--device cpu|metal]    (default cpu — Metal still hits a
//                                   rel_pos reshape bug on this Mac)
//
// `prompts.json` schema:
// [
//   { "view": "view_00.png",
//     "width": 518, "height": 518,
//     "points": [{ "bone": "antenna_L_scape", "x": 256.3, "y": 198.1 }] },
//   ...
// ]

use anyhow::{Context, Result, anyhow, bail};
use image::{ImageBuffer, Luma};
use rlx_core::validate_sam_device;
use rlx_runtime::Device;
use rlx_sam::config::SamConfig;
use rlx_sam::sam::Sam;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct ViewPrompts {
    view: String,
    width: u32,
    height: u32,
    points: Vec<BonePrompt>,
}

#[derive(Deserialize)]
struct BonePrompt {
    bone: String,
    x: f32,
    y: f32,
}

struct Args {
    weights: PathBuf,
    views_dir: PathBuf,
    prompts: PathBuf,
    out: PathBuf,
    device: Device,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out).with_context(|| format!("creating {}", args.out.display()))?;

    eprintln!(
        "[sam-batch] weights={:?} device={:?} prompts={:?}",
        args.weights, args.device, args.prompts
    );
    let weights_str = args
        .weights
        .to_str()
        .ok_or_else(|| anyhow!("non-utf8 weights path"))?;

    // SAM v1 ViT-B is the only published Meta release; the InsectSAM
    // fine-tune we also downloaded is the same architecture, so this
    // config is correct for both.
    let cfg = SamConfig::vit_b();
    let mut sam = Sam::from_safetensors_on(weights_str, cfg, args.device)?;
    let sam_input_side = sam.input_image_size();

    let prompts_raw = fs::read_to_string(&args.prompts)
        .with_context(|| format!("reading {}", args.prompts.display()))?;
    let prompts: Vec<ViewPrompts> =
        serde_json::from_str(&prompts_raw).context("parsing prompts.json")?;
    eprintln!(
        "[sam-batch] {} views, {} (view, bone) prompt pairs",
        prompts.len(),
        prompts.iter().map(|v| v.points.len()).sum::<usize>(),
    );

    for view in &prompts {
        let png_path = args.views_dir.join(&view.view);
        let img = image::open(&png_path)
            .with_context(|| format!("opening {}", png_path.display()))?
            .to_rgb8();
        if img.width() != view.width || img.height() != view.height {
            bail!(
                "{} dims ({}×{}) don't match prompts.json ({}×{})",
                png_path.display(),
                img.width(),
                img.height(),
                view.width,
                view.height
            );
        }
        let view_stem = Path::new(&view.view)
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("view filename has no stem"))?
            .to_string();
        let view_out = args.out.join(&view_stem);
        fs::create_dir_all(&view_out)
            .with_context(|| format!("creating {}", view_out.display()))?;

        eprintln!(
            "[sam-batch] {} → {} bone prompts",
            view.view,
            view.points.len()
        );

        // Encode image once; reuse for every bone prompt in this view.
        let t_enc = std::time::Instant::now();
        let (image_nchw, _) = rlx_sam::preprocess::preprocess_image(
            img.as_raw(),
            view.height as usize,
            view.width as usize,
        );
        let image_embeddings = sam.encode_image(&image_nchw);
        eprintln!(
            "  encode {:.2}s; {} bone prompts",
            t_enc.elapsed().as_secs_f32(),
            view.points.len()
        );

        for bp in &view.points {
            // The SAM preprocess resizes the longest side to 1024 and
            // letterboxes the short side, then divides by the resize
            // ratio. We mirror that here so the point lives in the
            // same coordinate frame as the encoded image.
            let scale = sam_input_side as f32 / view.width.max(view.height) as f32;
            let x_sam = bp.x * scale;
            let y_sam = bp.y * scale;

            let pred = sam.predict_masks(
                &image_embeddings,
                Some((&[x_sam, y_sam], &[1.0])),
                None,
                None,
                true,
            )?;
            let best_idx = pred
                .iou_pred
                .iter()
                .enumerate()
                .take(pred.num_masks)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mask_side = pred.mask_side;
            let mask_logits = &pred.mask_logits
                [best_idx * mask_side * mask_side..(best_idx + 1) * mask_side * mask_side];
            // One-off diagnostic: print logit stats for the first prompt
            // of each view so we can sanity-check the threshold. SAM
            // logits typically range ~[-30, +30]; a healthy mask has
            // positive logits inside the segmented region.
            if std::env::var("RLX_SAM_DEBUG_LOGITS").ok().as_deref() == Some("1") {
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                let mut sum = 0.0_f64;
                let mut pos = 0usize;
                for &v in mask_logits {
                    if v < lo {
                        lo = v;
                    }
                    if v > hi {
                        hi = v;
                    }
                    sum += v as f64;
                    if v > 0.0 {
                        pos += 1;
                    }
                }
                let mean = sum / mask_logits.len() as f64;
                eprintln!(
                    "    {} @ ({:.0},{:.0})  mask={}×{}  best_idx={} iou={:?}  logits min={:.2} max={:.2} mean={:.3} pos={}/{}",
                    bp.bone,
                    bp.x,
                    bp.y,
                    mask_side,
                    mask_side,
                    best_idx,
                    &pred.iou_pred[..pred.num_masks],
                    lo,
                    hi,
                    mean,
                    pos,
                    mask_logits.len()
                );
            }

            // Upscale mask_side² logits to view.width × view.height
            // via nearest-neighbour. Threshold > 0 → binary.
            let mut mask_img: ImageBuffer<Luma<u8>, Vec<u8>> =
                ImageBuffer::new(view.width, view.height);
            // The mask is in the SAM 1024×1024 letterboxed space; the
            // actual content sits in the top-left rectangle of size
            // (scale * width, scale * height). We sample the mask at
            // each output pixel via the same scale.
            let mask_scale = mask_side as f32 / sam_input_side as f32;
            for y in 0..view.height {
                for x in 0..view.width {
                    let mx = (x as f32 * scale * mask_scale) as usize;
                    let my = (y as f32 * scale * mask_scale) as usize;
                    let mx = mx.min(mask_side - 1);
                    let my = my.min(mask_side - 1);
                    let logit = mask_logits[my * mask_side + mx];
                    let v = if logit > 0.0 { 255 } else { 0 };
                    mask_img.put_pixel(x, y, Luma([v]));
                }
            }
            let out_path = view_out.join(format!("{}.png", bp.bone));
            mask_img
                .save(&out_path)
                .with_context(|| format!("writing {}", out_path.display()))?;
        }
    }
    eprintln!("[sam-batch] done — masks under {}", args.out.display());
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut argv = std::env::args().skip(1);
    let mut weights: Option<PathBuf> = None;
    let mut views_dir: Option<PathBuf> = None;
    let mut prompts: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut device_str = "cpu".to_string();
    while let Some(a) = argv.next() {
        match a.as_str() {
            "--weights" => weights = Some(argv.next().ok_or_else(|| anyhow!("--weights"))?.into()),
            "--views" => views_dir = Some(argv.next().ok_or_else(|| anyhow!("--views"))?.into()),
            "--prompts" => prompts = Some(argv.next().ok_or_else(|| anyhow!("--prompts"))?.into()),
            "--out" => out = Some(argv.next().ok_or_else(|| anyhow!("--out"))?.into()),
            "--device" => device_str = argv.next().ok_or_else(|| anyhow!("--device"))?,
            "--help" | "-h" => {
                eprintln!(
                    "rlx-sam-batch — SAM v1 mask generator\n\
                     \t--weights PATH         SAM safetensors (Meta naming)\n\
                     \t--views DIR            view PNGs from rlx-splat-anim render-views\n\
                     \t--prompts PATH         prompts.json from rlx-splat-anim project-prompts\n\
                     \t--out DIR              output directory for masks/view_NN/<bone>.png\n\
                     \t--device cpu|metal|... (default cpu; Metal currently broken)"
                );
                std::process::exit(0);
            }
            o => bail!("unknown flag: {o}"),
        }
    }
    let device = parse_sam_device("sam-batch", &device_str)?;
    Ok(Args {
        weights: weights.ok_or_else(|| anyhow!("--weights required"))?,
        views_dir: views_dir.ok_or_else(|| anyhow!("--views required"))?,
        prompts: prompts.ok_or_else(|| anyhow!("--prompts required"))?,
        out: out.ok_or_else(|| anyhow!("--out required"))?,
        device,
    })
}

fn parse_sam_device(name: &str, s: &str) -> Result<Device> {
    let dev = match s {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "gpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        "tpu" => Device::Tpu,
        other => bail!("{name}: unknown device `{other}`"),
    };
    validate_sam_device(name, dev)?;
    Ok(dev)
}

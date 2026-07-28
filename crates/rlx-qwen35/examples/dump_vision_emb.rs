//! Dump Qwen3.5 / Fara vision embeddings for parity checks.
//!
//! ```bash
//! cargo run -p rlx-qwen35 --example dump_vision_emb --release --features "apple-silicon qwen35-vlm" -- \
//!   --model-dir .cache/fara/4b --image /tmp/fara-tiny.png --force-size 128x96 \
//!   --out /tmp/fara_rlx_vision.npy
//! ```

use anyhow::{Context, Result};
use rlx_core::SafetensorsMmapLoader;
use rlx_qwen35::{MmProjConfig, MmProjWeights, Qwen35VisionEncoder};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut model_dir = PathBuf::from(".cache/fara/4b");
    let mut image = PathBuf::from("/tmp/fara-tiny.png");
    let mut out = PathBuf::from("/tmp/fara_rlx_vision.npy");
    let mut force_w: Option<usize> = None;
    let mut force_h: Option<usize> = None;
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model-dir" => model_dir = PathBuf::from(it.next().context("--model-dir")?),
            "--image" => image = PathBuf::from(it.next().context("--image")?),
            "--out" => out = PathBuf::from(it.next().context("--out")?),
            "--force-size" => {
                let s = it.next().context("--force-size WxH")?;
                let (w, h) = s.split_once('x').context("--force-size WxH")?;
                force_w = Some(w.parse()?);
                force_h = Some(h.parse()?);
            }
            other => anyhow::bail!("unknown arg {other}"),
        }
    }

    let cfg_path = model_dir.join("config.json");
    let mut cfg = MmProjConfig::from_hf_config_json(&cfg_path)?;
    // Match the tiny Fara smoke path unless the caller forces a size.
    if force_w.is_none() {
        if let Ok(v) = env::var("RLX_QWEN35_IMAGE_MIN_PIXELS") {
            if let Ok(n) = v.parse() {
                cfg.image_min_pixels = n;
            }
        }
        if let Ok(v) = env::var("RLX_QWEN35_IMAGE_MAX_PIXELS") {
            if let Ok(n) = v.parse() {
                cfg.image_max_pixels = n;
            }
        }
    }
    let mut loader = SafetensorsMmapLoader::open(&model_dir)?;
    let weights = MmProjWeights::from_hf_visual(&cfg, &mut loader)?;

    let img = image::open(&image).with_context(|| format!("open {}", image.display()))?;
    let (src_w, src_h) = (img.width() as usize, img.height() as usize);
    let (tw, th) = match (force_w, force_h) {
        (Some(w), Some(h)) => (w, h),
        _ => {
            // Same helper as the encoder path.
            let align = cfg.patch_size * cfg.n_merge;
            let mut h_bar = align.max(((src_h as f32 / align as f32).round() as usize) * align);
            let mut w_bar = align.max(((src_w as f32 / align as f32).round() as usize) * align);
            if h_bar * w_bar > cfg.image_max_pixels {
                let beta = ((src_h * src_w) as f32 / cfg.image_max_pixels as f32).sqrt();
                h_bar = align.max(((src_h as f32 / beta / align as f32).floor() as usize) * align);
                w_bar = align.max(((src_w as f32 / beta / align as f32).floor() as usize) * align);
            } else if h_bar * w_bar < cfg.image_min_pixels {
                let beta = (cfg.image_min_pixels as f32 / (src_h * src_w) as f32).sqrt();
                h_bar = ((src_h as f32 * beta / align as f32).ceil() as usize) * align;
                w_bar = ((src_w as f32 * beta / align as f32).ceil() as usize) * align;
            }
            (w_bar, h_bar)
        }
    };
    let resized = if (src_w, src_h) != (tw, th) {
        img.resize_exact(
            tw as u32,
            th as u32,
            image::imageops::FilterType::CatmullRom,
        )
    } else {
        img
    };
    let rgb = resized.to_rgb8().into_raw();
    eprintln!(
        "[dump_vision] src={src_w}x{src_h} resized={tw}x{th} n_embd={} layers={}",
        cfg.n_embd, cfg.n_layer
    );

    let mut enc = Qwen35VisionEncoder::from_parts(cfg, weights, tw, th)?;
    let out_emb = enc.encode_rgb(&rgb, tw, th)?;
    let dim = out_emb.embeddings.len() / out_emb.n_tokens.max(1);
    let mean = out_emb.embeddings.iter().sum::<f32>() / out_emb.embeddings.len().max(1) as f32;
    let absmax = out_emb
        .embeddings
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    let l2_0 = out_emb.embeddings[..dim]
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    eprintln!(
        "[dump_vision] tokens={} grid={}x{} dim={dim} mean={mean:.6} absmax={absmax:.4} l2_0={l2_0:.4}",
        out_emb.n_tokens, out_emb.grid_x, out_emb.grid_y
    );

    let rows = out_emb.n_tokens as u32;
    let cols = dim as u32;
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let mut header_bytes = header.into_bytes();
    while (10 + header_bytes.len() + 1) % 16 != 0 {
        header_bytes.push(b' ');
    }
    header_bytes.push(b'\n');
    let mut file = std::fs::File::create(&out)?;
    use std::io::Write;
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&(header_bytes.len() as u16).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    for &x in &out_emb.embeddings {
        file.write_all(&x.to_le_bytes())?;
    }
    eprintln!("[dump_vision] wrote {}", out.display());
    Ok(())
}

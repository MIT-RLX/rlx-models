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

//! `rlx-gemma --multimodal` CLI — text + image + audio inference.
//!
//! ```bash
//! rlx-gemma --multimodal \
//!     --weights model.safetensors \
//!     --config config.json \
//!     --image photo.jpg \
//!     --prompt "describe <image>" \
//!     --device metal
//! ```

use crate::multimodal::{GemmaMultimodalConfig, load_wav_mono_16khz};
use crate::multimodal_runner::{GemmaMultimodalRunner, MultimodalWeights, ProjectorLayout};
use crate::{GemmaConfig, GemmaConfigSource, GemmaRunner, encode_prompt_auto};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::parse_gemma_device;
use rlx_qwen3::SampleOpts;
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut prompt: Option<String> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut audio: Vec<PathBuf> = Vec::new();
    let mut max_tokens = 32usize;
    let mut max_seq = 4096usize;
    let mut max_soft_tokens = 280usize;
    let mut max_side_patches = 32usize;
    let mut videos: Vec<PathBuf> = Vec::new();
    let mut temperature = 0f32;
    let mut top_p = 1f32;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "--weights" => {
                i += 1;
                weights = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--weights needs a value"))?
                        .into(),
                );
            }
            "--config" => {
                i += 1;
                config = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--config needs a value"))?
                        .into(),
                );
            }
            "--device" => {
                i += 1;
                device = args
                    .get(i)
                    .cloned()
                    .ok_or_else(|| anyhow!("--device needs a value"))?;
            }
            "--prompt" => {
                i += 1;
                prompt = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| anyhow!("--prompt needs a value"))?,
                );
            }
            "--tokenizer" => {
                i += 1;
                tokenizer = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--tokenizer needs a value"))?
                        .into(),
                );
            }
            "--image" => {
                i += 1;
                images.push(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--image needs a value"))?
                        .into(),
                );
            }
            "--audio" => {
                i += 1;
                audio.push(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--audio needs a value"))?
                        .into(),
                );
            }
            "--video" => {
                i += 1;
                videos.push(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--video needs a value"))?
                        .into(),
                );
            }
            "--max-soft-tokens" => {
                i += 1;
                max_soft_tokens = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-soft-tokens needs a value"))?
                    .parse()
                    .context("--max-soft-tokens")?;
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-tokens needs a value"))?
                    .parse()
                    .context("--max-tokens")?;
            }
            "--max-seq" => {
                i += 1;
                max_seq = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-seq needs a value"))?
                    .parse()
                    .context("--max-seq")?;
            }
            "--max-side-patches" => {
                i += 1;
                max_side_patches = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-side-patches needs a value"))?
                    .parse()
                    .context("--max-side-patches")?;
            }
            "--temperature" => {
                i += 1;
                temperature = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--temperature needs a value"))?
                    .parse()
                    .context("--temperature")?;
            }
            "--top-p" => {
                i += 1;
                top_p = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--top-p needs a value"))?
                    .parse()
                    .context("--top-p")?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
        i += 1;
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let config_path = config.ok_or_else(|| anyhow!("--config is required (HF config.json)"))?;
    let prompt = prompt.ok_or_else(|| anyhow!("--prompt is required"))?;
    let device = parse_gemma_device(&device)?;

    let cfg = GemmaConfig::from_file(&config_path)
        .with_context(|| format!("loading LM config {config_path:?}"))?;
    let mm_cfg = GemmaMultimodalConfig::from_file(&config_path)
        .with_context(|| format!("loading multimodal config {config_path:?}"))?;

    if !mm_cfg.has_vision() && !mm_cfg.has_audio() {
        bail!(
            "config.json at {config_path:?} has no vision_config / audio_config — \
             use `rlx-gemma` (text-only) for this model"
        );
    }

    eprintln!(
        "[rlx-gemma mm] weights={weights:?} device={device:?} images={} audio={}",
        images.len(),
        audio.len()
    );

    let lm_hidden = cfg.hidden_size;
    let mut mm = GemmaMultimodalRunner::new(
        mm_cfg.clone(),
        lm_hidden,
        device,
        Some(max_soft_tokens),
        None,
    )?;

    let mm_weights = load_projector_weights(&weights)?;

    if mm_weights.layout() == ProjectorLayout::Unified {
        eprintln!("[rlx-gemma mm] unified Gemma 4 projector layout");
    }

    let mut image_soft_counts: Vec<usize> = Vec::new();
    let mut image_soft = Vec::new();
    for path in &images {
        let count = mm.image_soft_token_count(path).unwrap_or(max_soft_tokens);
        image_soft_counts.push(count);
        let projected = mm.project_image_file(path, &mm_weights, max_side_patches)?;
        image_soft.extend(projected);
    }

    let mut video_soft = Vec::new();
    let mut video_soft_counts: Vec<usize> = Vec::new();
    for path in &videos {
        // One frame per --video path for now; HF uses timestamp-prefixed multi-frame prompts.
        video_soft_counts.push(70);
        let projected = mm.project_video_frame(path, &mm_weights)?;
        video_soft.extend(projected);
    }

    let mut audio_soft = Vec::new();
    let mut audio_sample_counts: Vec<usize> = Vec::new();
    for path in &audio {
        let samples = load_wav_mono_16khz(path)?;
        audio_sample_counts.push(samples.len());
        let projected = mm.project_audio_file(path, &mm_weights)?;
        audio_soft.extend(projected);
    }

    let encode =
        |s: &str| -> Result<Vec<u32>> { encode_prompt_auto(&weights, tokenizer.as_deref(), s) };
    let token_ids = mm.tokenize_prompt(
        &prompt,
        &image_soft_counts,
        &audio_sample_counts,
        &video_soft_counts,
        encode,
    )?;

    let sample = SampleOpts {
        temperature,
        top_p,
        ..SampleOpts::greedy()
    };
    let mut runner = GemmaRunner::builder()
        .weights(weights.clone())
        .device(device)
        .max_seq(max_seq.max(token_ids.len() + max_tokens))
        .stream(true)
        .sample(sample)
        .config(GemmaConfigSource::JsonFile(config_path.clone()))
        .build()?;

    eprintln!(
        "[rlx-gemma mm] tokenized: {} ids ({} image rows, {} audio rows)",
        token_ids.len(),
        image_soft.len() / lm_hidden.max(1),
        audio_soft.len() / lm_hidden.max(1),
    );

    let t0 = std::time::Instant::now();
    let mut printed = 0usize;
    runner.generate_multimodal(
        &mm_cfg,
        &token_ids,
        &image_soft,
        &audio_soft,
        &video_soft,
        max_tokens,
        |tok| {
            print!("{tok} ");
            std::io::stdout().flush().ok();
            printed += 1;
        },
    )?;
    let dt = t0.elapsed();
    println!();
    eprintln!(
        "[rlx-gemma mm] generated {printed} tokens in {:.2?} ({:.1} tok/s) — \
         prefill_seq={} ({} vision + {} audio rows)",
        dt,
        printed as f64 / dt.as_secs_f64(),
        token_ids.len(),
        image_soft.len() / lm_hidden.max(1),
        audio_soft.len() / lm_hidden.max(1),
    );
    Ok(())
}

fn load_projector_weights(weights: &Path) -> Result<MultimodalWeights> {
    if let Some(p) = find_mmproj_gguf(weights.parent()) {
        eprintln!("[rlx-gemma mm] loading projector weights from {p:?}");
        return MultimodalWeights::from_mmproj_gguf(p);
    }
    eprintln!("[rlx-gemma mm] loading projector weights from {weights:?}");
    MultimodalWeights::from_safetensors(weights)
}

fn find_mmproj_gguf(dir: Option<&Path>) -> Option<PathBuf> {
    let dir = dir?;
    let direct = dir.join("mmproj.gguf");
    if direct.exists() {
        return Some(direct);
    }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("mmproj") && name.ends_with(".gguf") {
            return Some(entry.path());
        }
    }
    None
}

fn print_help() {
    eprintln!(
        "rlx-gemma --multimodal — Gemma 4 vision + audio LM\n\
         \n\
         REQUIRED\n\
           --weights <path>         safetensors model file (LM + unified projectors)\n\
           --config <path>          HF config.json (with vision_config/audio_config)\n\
           --prompt <text>          prompt with <image>, <|image|>, <audio>, <|video|> markers\n\
         \n\
         INPUTS\n\
           --image <path>           repeatable; matched in order to image markers\n\
           --audio <path>           repeatable; matched in order to audio markers\n\
           --video <path>           repeatable; one frame per path (70 soft tokens each)\n\
         \n\
         RUNTIME\n\
           --device cpu|metal|mlx|wgpu|cuda|rocm|vulkan   default cpu\n\
           --tokenizer <path>       optional; auto-discovered next to weights\n\
           --max-tokens N           default 32\n\
           --max-seq N              default 4096\n\
           --max-soft-tokens N      unified vision budget (70|140|280|560|1120); default 280\n\
           --max-side-patches N     legacy layout only; default 32\n\
           --temperature F          default 0 (greedy)\n\
           --top-p F                default 1\n\
         \n\
         OUTPUT\n\
           Generated text to stdout; logs to stderr.\n\
         "
    );
}

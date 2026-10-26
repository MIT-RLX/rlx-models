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

//! `rlx-minicpm-v` command line: one chat turn about one or more images.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use rlx_qwen35::SampleOpts;

use crate::{MiniCpmVRunner, ResamplerRunner, RgbImage, VideoOpts, Visual};

const USAGE: &str = "rlx-minicpm-v (--model-dir <hf-or-gguf-dir> [--prefer Q4_K_M] | \
    --weights <lm.gguf> --mmproj <mmproj.gguf>) [--image <path>]... \
    [--video <path> [--max-frames N] [--video-slices N] [--video-fps N] [--packing N]]... \
    --prompt \"…\" \
    [--max-tokens N] [--temperature T] [--thinking] [--bf16-vision] [--packed] \
    [--tokenizer tokenizer.json] [--max-seq N] [--device cpu|metal|mlx|cuda|rocm|gpu|vulkan]";

pub fn run(args: &[String]) -> Result<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut lm_gguf: Option<PathBuf> = None;
    let mut mmproj: Option<PathBuf> = None;
    let mut prefer: Option<String> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut packed = false;
    let mut images: Vec<PathBuf> = Vec::new();
    // Pictures and clips in the order given: `true` marks a clip.
    let mut media: Vec<(bool, PathBuf)> = Vec::new();
    let mut video_opts = VideoOpts::default();
    let mut pack = crate::video::PackOpts::default();
    let mut prompt: Option<String> = None;
    let mut device = "cpu".to_string();
    let mut max_tokens = 256usize;
    let mut temperature = 0.0f32;
    let mut thinking = false;
    let mut bf16_vision = false;
    let mut max_seq: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => model_dir = Some(req(args, &mut i)?.into()),
            // A directory works too; a `.gguf` file is the LM half of a pair.
            "--weights" => {
                let p: PathBuf = req(args, &mut i)?.into();
                if p.is_file() {
                    lm_gguf = Some(p);
                } else {
                    model_dir = Some(p);
                }
            }
            "--mmproj" => mmproj = Some(req(args, &mut i)?.into()),
            "--prefer" => prefer = Some(req(args, &mut i)?),
            "--tokenizer" => tokenizer = Some(req(args, &mut i)?.into()),
            "--packed" => {
                packed = true;
                i += 1;
            }
            "--image" => {
                let p: PathBuf = req(args, &mut i)?.into();
                images.push(p.clone());
                media.push((false, p));
            }
            "--video" => media.push((true, req(args, &mut i)?.into())),
            "--max-frames" => video_opts.max_num_frames = req(args, &mut i)?.parse()?,
            "--video-slices" => video_opts.max_slice_nums = req(args, &mut i)?.parse()?,
            // 4.5 packed video (the model card's `encode_video`).
            "--video-fps" => pack.fps = req(args, &mut i)?.parse()?,
            "--packing" => pack.force_packing = Some(req(args, &mut i)?.parse()?),
            "--prompt" => prompt = Some(req(args, &mut i)?),
            "--device" => device = req(args, &mut i)?,
            "--max-tokens" => max_tokens = req(args, &mut i)?.parse()?,
            "--temperature" => temperature = req(args, &mut i)?.parse()?,
            "--max-seq" => max_seq = Some(req(args, &mut i)?.parse()?),
            "--thinking" => {
                thinking = true;
                i += 1;
            }
            "--bf16-vision" => {
                bf16_vision = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown flag: {other}\n{USAGE}"),
        }
    }
    let prompt = prompt.ok_or_else(|| anyhow!("--prompt is required\n{USAGE}"))?;
    let device = parse_standard_device("minicpm-v", &device)?;
    let opts = if temperature > 0.0 {
        SampleOpts {
            temperature,
            ..SampleOpts::greedy()
        }
    } else {
        SampleOpts::greedy()
    };

    // The resampler generation (2.x–4.5) has its own runner.
    if let Some(d) = model_dir
        .as_ref()
        .filter(|d| crate::resampler_gen::is_model_dir(d))
    {
        let clips: Vec<&PathBuf> = media.iter().filter(|(v, _)| *v).map(|(_, p)| p).collect();
        if clips.len() > 1 || (!clips.is_empty() && !images.is_empty()) {
            bail!("the resampler generation takes images, or one --video (4.5)");
        }
        let mut b = ResamplerRunner::builder()
            .model_dir(d)
            .device(device)
            .bf16_vision(bf16_vision);
        if let Some(n) = max_seq {
            b = b.max_seq(n);
        }
        let mut runner = b.build()?;
        if let Some(clip) = clips.first() {
            let video = runner.load_packed_video(clip, &pack)?;
            return stream(|on_text| {
                runner.chat_video(&video, &prompt, max_tokens, opts, thinking, on_text)
            });
        }
        let imgs = images
            .iter()
            .map(RgbImage::open)
            .collect::<Result<Vec<_>>>()?;
        return stream(|on_text| runner.chat(&imgs, &prompt, max_tokens, opts, thinking, on_text));
    }

    let mut b = MiniCpmVRunner::builder()
        .device(device)
        .bf16_vision(bf16_vision);
    match (lm_gguf, mmproj, model_dir) {
        (Some(lm), Some(mp), _) => b = b.gguf(lm, mp),
        (Some(_), None, _) => bail!("--weights <file.gguf> needs --mmproj\n{USAGE}"),
        (None, _, Some(d)) => b = b.model_dir(d),
        (None, _, None) => bail!("--model-dir or --weights is required\n{USAGE}"),
    }
    if let Some(q) = prefer {
        b = b.prefer(q);
    }
    if let Some(t) = tokenizer {
        b = b.tokenizer(t);
    }
    if packed {
        b = b.packed(true);
    }
    if let Some(n) = max_seq {
        b = b.max_seq(n);
    }
    let mut runner = b.build()?;
    let visuals = media
        .iter()
        .map(|(video, p)| {
            if *video {
                runner.load_video(p, video_opts)
            } else {
                runner.slice(&RgbImage::open(p)?).map(Visual::Image)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    stream(|on_text| runner.chat_media(&visuals, &prompt, max_tokens, opts, thinking, on_text))
}

/// Run `chat`, printing text as it arrives, and report the time.
fn stream(chat: impl FnOnce(&mut dyn FnMut(&str)) -> Result<String>) -> Result<()> {
    let t = web_time::Instant::now();
    let mut stdout = std::io::stdout();
    let text = chat(&mut |piece: &str| {
        let _ = stdout.write_all(piece.as_bytes());
        let _ = stdout.flush();
    })?;
    println!();
    eprintln!(
        "[rlx-minicpm-v] {} chars in {:.1}s",
        text.chars().count(),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

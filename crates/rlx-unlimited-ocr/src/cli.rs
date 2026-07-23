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

//! `rlx-unlimited-ocr` CLI — single image, multi-image, or PDF OCR.

use crate::device::resolve_device;
use crate::fixtures::probe_image_path;
use crate::hub::{default_model_dir, resolve_weights_path};
use crate::infer::{InferenceOptions, UnlimitedOcrSession};
use crate::lm_precision::LmWeightPrecision;
use crate::preprocess::ImageMode;
use crate::runner::UnlimitedOcrRunner;
use anyhow::{Context, Result, bail};
use rlx_cli::req;
use std::path::PathBuf;

struct CliArgs {
    weights: Option<PathBuf>,
    image: Option<PathBuf>,
    images: Vec<PathBuf>,
    pdf: Option<PathBuf>,
    device: Option<String>,
    max_tokens: usize,
    mode: Option<ImageMode>,
    prompt: Option<String>,
    lm_precision: LmWeightPrecision,
    dry: bool,
    list_keys: bool,
    download_only: bool,
}

pub fn run(args: &[String]) -> Result<()> {
    let Some(cli) = parse_args(args)? else {
        return Ok(());
    };
    run_parsed(cli)
}

/// `Ok(None)` after `--help`.
fn parse_args(args: &[String]) -> Result<Option<CliArgs>> {
    let mut weights = None;
    let mut image = None;
    let mut images = Vec::new();
    let mut pdf = None;
    let mut device = None;
    let mut max_tokens = 4096;
    let mut mode = None;
    let mut prompt = None;
    let mut lm_precision = LmWeightPrecision::Auto;
    let mut dry = false;
    let mut list_keys = false;
    let mut download_only = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "--model-dir" => weights = Some(req(args, &mut i)?.into()),
            "--image" => image = Some(req(args, &mut i)?.into()),
            "--images" => {
                let s = req(args, &mut i)?;
                images = s.split(',').map(|p| PathBuf::from(p.trim())).collect();
            }
            "--pdf" => pdf = Some(req(args, &mut i)?.into()),
            "--device" => device = Some(req(args, &mut i)?),
            "--max-tokens" => {
                max_tokens = req(args, &mut i)?.parse().context("--max-tokens")?;
            }
            "--mode" => {
                let s = req(args, &mut i)?;
                mode = Some(ImageMode::parse(&s).with_context(|| {
                    format!("--mode: unknown mode {s:?} (expected base|gundam|multi)")
                })?);
            }
            "--prompt" => prompt = Some(req(args, &mut i)?),
            "--lm-precision" => {
                let s = req(args, &mut i)?;
                lm_precision = LmWeightPrecision::parse(&s).with_context(|| {
                    format!("--lm-precision: unknown {s:?} (expected f32|f16|bf16|q8_0|q4_0|auto)")
                })?;
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--list-keys" => {
                list_keys = true;
                i += 1;
            }
            "--download" => {
                download_only = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(None);
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    Ok(Some(CliArgs {
        weights,
        image,
        images,
        pdf,
        device,
        max_tokens,
        mode,
        prompt,
        lm_precision,
        dry,
        list_keys,
        download_only,
    }))
}

fn run_parsed(cli: CliArgs) -> Result<()> {
    if cli.download_only {
        #[cfg(feature = "hf-download")]
        {
            let dir = crate::download::fetch_default()?;
            eprintln!("\nDone. Snapshot:\n  {}", dir.display());
            return Ok(());
        }
        #[cfg(not(feature = "hf-download"))]
        {
            anyhow::bail!(
                "rebuild with --features hf-download, or run:\n  \
                 huggingface-cli download baidu/Unlimited-OCR"
            );
        }
    }

    let model_dir = match &cli.weights {
        Some(w) => resolve_weights_path(w)?,
        None => default_model_dir()?,
    };
    if cli.weights.is_none() {
        eprintln!("[rlx-unlimited-ocr] weights {}", model_dir.display());
    }

    let device = resolve_device(cli.device.as_deref())?;

    if cli.dry || cli.list_keys {
        let runner = UnlimitedOcrRunner::open(&model_dir, device)?;
        if cli.list_keys {
            let mut keys: Vec<_> = runner.store().keys().iter().cloned().collect();
            keys.sort();
            for k in keys {
                println!("{k}");
            }
        }
        if cli.dry {
            eprintln!(
                "[rlx-unlimited-ocr] dry ok — tensors={} hidden={} vocab={} device={device:?}",
                runner.store().keys().len(),
                runner.config().hidden_size,
                runner.config().vocab_size,
            );
        }
        return Ok(());
    }

    let mut options = InferenceOptions::for_ocr()
        .device(device)
        .weight_precision(cli.lm_precision);
    options = options.max_new_tokens(cli.max_tokens);
    if let Some(mode) = cli.mode {
        options = options.mode(mode);
    }
    if let Some(prompt) = cli.prompt {
        options = options.prompt(prompt);
    }
    let mut session = UnlimitedOcrSession::open(&model_dir, options)?;

    let result = if let Some(pdf) = &cli.pdf {
        session.run_pdf(pdf)
    } else if !cli.images.is_empty() {
        session.run_multi(&cli.images)
    } else {
        let image_path = cli.image.unwrap_or_else(probe_image_path);
        if !image_path.is_file() {
            bail!("image not found: {}", image_path.display());
        }
        session.run_single(&image_path)
    }?;

    for (i, page) in result.pages.iter().enumerate() {
        println!("--- page {i} ---\n{page}");
    }
    eprintln!(
        "[rlx-unlimited-ocr] done — {} prompt + {} new tokens",
        result.prompt_len, result.new_tokens
    );
    Ok(())
}

fn print_help() {
    eprintln!(
        "rlx-unlimited-ocr — baidu/Unlimited-OCR (SAM + CLIP DeepEncoder + MoE LM)\n\
         \n\
         Weights (optional — HF Hub cache by default):\n\
           [--model-dir PATH]        Dir, `hf`, or Hub id `baidu/Unlimited-OCR`\n\
         \n\
         Input (choose one; default: bundled fixtures/sample.jpg):\n\
           [--image PATH]            Single page/document image\n\
           [--images PATH,PATH,…]    Multiple images for one document\n\
           [--pdf PATH]              PDF — rasterized to pages, then OCR'd\n\
         \n\
         Device & decode:\n\
           [--device auto|cpu|metal|cuda|…]  default: auto (RLX_DEVICE)\n\
           [--lm-precision f32|f16|bf16|q8_0|q4_0|auto]  default: auto\n\
           [--max-tokens N]          default: 4096\n\
           [--mode base|gundam|multi]  default: gundam (auto for --images/--pdf)\n\
           [--prompt TEXT]           default: \"<image>document parsing.\"\n\
         \n\
         Other:\n\
           [--download]              Fetch weights into Hugging Face cache\n\
           [--dry] [--list-keys]\n\
         \n\
         Env: RLX_UNLIMITED_OCR_DIR, RLX_UNLIMITED_OCR_IMAGE, RLX_DEVICE,\n\
              RLX_UNLIMITED_OCR_ASSUME_RAM_BYTES (force Auto RAM budget)\n\
         \n\
         Quick start:\n\
           just fetch-unlimited-ocr\n\
           just unlimited-ocr -- --image page.png"
    );
}

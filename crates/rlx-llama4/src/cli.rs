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

//! CLI for the Llama-4 text model: `--weights <dir> --prompt <text>`.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_cli::parse_device;
use tokenizers::Tokenizer;

use crate::runner::Llama4Runner;
use crate::vl_runner::Llama4VlRunner;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<String> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut device = "cpu".to_string();
    let mut max_tokens = 32usize;
    let mut image: Option<String> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" | "--data" | "--model-dir" => {
                weights = Some(it.next().context("--weights <dir>")?.clone())
            }
            "--prompt" => prompt = it.next().context("--prompt <text>")?.clone(),
            "--device" => device = it.next().context("--device <name>")?.clone(),
            "--max-tokens" => max_tokens = it.next().context("--max-tokens <n>")?.parse()?,
            "--image" => image = Some(it.next().context("--image <path>")?.clone()),
            other => anyhow::bail!("unknown arg {other}"),
        }
    }
    let weights = weights.context("--weights <checkpoint dir> is required")?;
    let dev = parse_device(&device)?;
    let dir = Path::new(&weights);
    let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;
    let enc = tok
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("encoding prompt: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();

    let mut on_token = |t: u32| {
        if let Ok(s) = tok.decode(&[t], true) {
            print!("{s}");
            let _ = std::io::stdout().flush();
        }
        true
    };

    match image {
        Some(path) => {
            let img = image::open(&path)
                .with_context(|| format!("opening image {path}"))?
                .to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let mut runner = Llama4VlRunner::from_checkpoint(dir, dev)?;
            eprintln!(
                "llama4-vl: {} prompt tokens, image {w}x{h}, device {device}",
                ids.len()
            );
            runner.generate_multimodal(
                &ids,
                &img.into_raw(),
                w,
                h,
                max_tokens,
                None,
                &mut on_token,
            )?;
        }
        None => {
            let mut runner = Llama4Runner::from_checkpoint(dir, dev)?;
            eprintln!(
                "llama4: {} prompt tokens, device {device}, max_tokens {max_tokens}",
                ids.len()
            );
            runner.generate(&ids, max_tokens, None, &mut on_token)?;
        }
    }
    println!();
    Ok(())
}

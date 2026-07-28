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

//! CLI for Llama-3.2-Vision: `--weights <dir> --image <path> --prompt <text>`.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_cli::parse_device;
use tokenizers::Tokenizer;

use crate::runner::MllamaRunner;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<String> = None;
    let mut image: Option<String> = None;
    let mut prompt = "Describe this image in detail.".to_string();
    let mut device = "cpu".to_string();
    let mut max_tokens = 32usize;
    let mut raw = false;
    let mut dump_vision: Option<String> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" | "--data" | "--model-dir" => {
                weights = Some(it.next().context("--weights <dir>")?.clone())
            }
            "--image" => image = Some(it.next().context("--image <path>")?.clone()),
            "--prompt" => prompt = it.next().context("--prompt <text>")?.clone(),
            "--device" => device = it.next().context("--device <name>")?.clone(),
            "--max-tokens" => max_tokens = it.next().context("--max-tokens <n>")?.parse()?,
            "--raw" => raw = true,
            "--dump-vision" => {
                dump_vision = Some(it.next().context("--dump-vision <path>")?.clone())
            }
            other => anyhow::bail!("unknown arg {other}"),
        }
    }

    let weights = weights.context("--weights <checkpoint dir> is required")?;
    let image = image.context("--image <path> is required")?;
    let dev = parse_device(&device)?;
    let dir = Path::new(&weights);

    let mut runner = MllamaRunner::from_checkpoint(dir, dev)?;

    // Vision-only parity mode: dump cross_states as raw f32 + shape sidecar.
    if let Some(out) = dump_vision {
        let img = image::open(&image)
            .with_context(|| format!("opening image {image}"))?
            .to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let (cs, nt, np, hidden) = runner.vision_features(&img.into_raw(), w, h)?;
        let mut bytes = Vec::with_capacity(cs.len() * 4);
        for v in &cs {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(format!("{out}.f32"), &bytes)?;
        std::fs::write(
            format!("{out}.json"),
            format!("{{\"shape\": [{nt}, {np}, {hidden}]}}"),
        )?;
        eprintln!("wrote {out}.f32 [{nt}, {np}, {hidden}] and {out}.json");
        return Ok(());
    }
    let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;

    // Llama-3.2 chat prompt with the image token (unless --raw).
    let text = if raw {
        prompt
    } else {
        format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
             <|image|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    };
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("encoding prompt: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();

    let img = image::open(&image)
        .with_context(|| format!("opening image {image}"))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb = img.into_raw();

    eprintln!(
        "mllama: {} prompt tokens, image {w}x{h}, device {device}, max_tokens {max_tokens}",
        ids.len()
    );

    runner.generate_multimodal_ids(&ids, &rgb, w, h, max_tokens, &mut |t| {
        if let Ok(s) = tok.decode(&[t], true) {
            print!("{s}");
            let _ = std::io::stdout().flush();
        }
        true
    })?;
    println!();
    Ok(())
}

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

//! `rlx-lfm` CLI — LFM2 / LFM2.5 GGUF text generation.

use crate::lfm2_gguf::{Lfm2GgufRunner, resolve_gguf};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

const HELP: &str = "\
rlx-lfm — LiquidAI LFM2 / LFM2.5 GGUF text generation (hybrid ShortConv + attention)

USAGE:
  rlx-lfm --weights <FILE.gguf|DIR> --prompt \"...\" [options]
  rlx-lfm --weights <FILE.gguf|DIR> --prompt-ids 1,2,3 [options]

OPTIONS:
  --weights <PATH>     .gguf file (or a directory containing one)
  --prompt <TEXT>      text prompt (requires the `tokenizer` feature)
  --prompt-ids <a,b>   raw token ids instead of --prompt
  --device <NAME>      cpu | metal | mlx | cuda | rocm | gpu | vulkan (default cpu)
  --max-tokens <N>     new tokens to generate (default 32)
  --raw                decode without skipping special tokens
  -h, --help           show this help
";

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut prompt: Option<String> = None;
    let mut prompt_ids: Option<Vec<u32>> = None;
    let mut max_tokens = 32usize;
    let mut raw = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "--model" => weights = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--prompt" => prompt = Some(req(args, &mut i)?),
            "--prompt-ids" => {
                prompt_ids = Some(
                    req(args, &mut i)?
                        .split(',')
                        .map(|s| s.trim().parse::<u32>())
                        .collect::<Result<_, _>>()
                        .context("--prompt-ids: comma-separated u32 list")?,
                );
            }
            "--max-tokens" => {
                max_tokens = req(args, &mut i)?.parse().context("--max-tokens: usize")?;
            }
            "--raw" => {
                raw = true;
                i += 1;
            }
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(());
            }
            other => bail!("unknown flag: {other} (try --help)"),
        }
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let gguf = resolve_gguf(&weights)?;
    let dev = parse_standard_device("lfm", &device)?;

    eprintln!("[rlx-lfm] loading {} on {dev:?}", gguf.display());
    let runner = Lfm2GgufRunner::open(&gguf, dev)?;
    let c = runner.config();
    eprintln!(
        "[rlx-lfm] LFM2.5: hidden={} layers={} heads={}/{} head_dim={} inter={} conv_k={} attn_layers={:?} vocab={}",
        c.hidden_size,
        c.num_hidden_layers,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.head_dim,
        c.intermediate_size,
        c.conv_kernel,
        c.full_attn_layers,
        c.vocab_size,
    );

    let ids = resolve_prompt_ids(&gguf, prompt.as_deref(), prompt_ids)?;
    if ids.is_empty() {
        bail!("empty prompt");
    }

    let t0 = std::time::Instant::now();
    let generated = generate(&runner, &gguf, &ids, max_tokens, raw)?;
    let dt = t0.elapsed();
    eprintln!(
        "\n[rlx-lfm] {} tokens in {:.2?} ({:.1} tok/s) on {dev:?}",
        generated.len(),
        dt,
        generated.len() as f64 / dt.as_secs_f64().max(1e-9),
    );
    Ok(())
}

#[cfg(feature = "tokenizer")]
fn resolve_prompt_ids(
    gguf: &std::path::Path,
    prompt: Option<&str>,
    prompt_ids: Option<Vec<u32>>,
) -> Result<Vec<u32>> {
    if let Some(ids) = prompt_ids {
        return Ok(ids);
    }
    let text = prompt.ok_or_else(|| anyhow!("provide --prompt or --prompt-ids"))?;
    rlx_qwen35::encode_prompt_from_gguf(gguf, text)
}

#[cfg(not(feature = "tokenizer"))]
fn resolve_prompt_ids(
    _gguf: &std::path::Path,
    prompt: Option<&str>,
    prompt_ids: Option<Vec<u32>>,
) -> Result<Vec<u32>> {
    if let Some(ids) = prompt_ids {
        return Ok(ids);
    }
    if prompt.is_some() {
        bail!(
            "--prompt requires the `tokenizer` feature; rebuild with --features tokenizer, or use --prompt-ids"
        );
    }
    bail!("provide --prompt-ids (or build with --features tokenizer for --prompt)")
}

#[cfg(feature = "tokenizer")]
fn generate(
    runner: &Lfm2GgufRunner,
    gguf: &std::path::Path,
    ids: &[u32],
    max_tokens: usize,
    raw: bool,
) -> Result<Vec<u32>> {
    use std::io::Write;
    let skip_special = !raw;
    let mut generated: Vec<u32> = Vec::new();
    let mut last_len = 0usize;
    runner.generate(ids, max_tokens, |tok| {
        generated.push(tok);
        if let Ok(text) = rlx_qwen35::decode_ids_from_gguf(gguf, &generated, skip_special) {
            if text.len() >= last_len {
                print!("{}", &text[last_len..]);
            } else {
                print!("\r{text}");
            }
            last_len = text.len();
            std::io::stdout().flush().ok();
        }
        true
    })?;
    println!();
    if let Ok(full) = rlx_qwen35::decode_ids_from_gguf(gguf, &generated, skip_special) {
        eprintln!("[rlx-lfm] continuation:\n{full}");
    }
    Ok(generated)
}

#[cfg(not(feature = "tokenizer"))]
fn generate(
    runner: &Lfm2GgufRunner,
    _gguf: &std::path::Path,
    ids: &[u32],
    max_tokens: usize,
    _raw: bool,
) -> Result<Vec<u32>> {
    let generated = runner.generate(ids, max_tokens, |_| true)?;
    let rendered: Vec<String> = generated.iter().map(|t| t.to_string()).collect();
    println!("{}", rendered.join(","));
    Ok(generated)
}

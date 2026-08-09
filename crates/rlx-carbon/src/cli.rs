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

//! `rlx-carbon` CLI — DNA generation with the Carbon models.

use crate::{CarbonRunner, SampleOpts};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_llama32_device, req};
#[cfg(feature = "tokenizer")]
use std::io::Write;
use std::path::PathBuf;

const HELP: &str = "\
rlx-carbon — Carbon DNA language models (Llama backbone + hybrid DNA tokenizer)

USAGE:
  rlx-carbon --model <DIR> --prompt <SEQ> [options]
  rlx-carbon --model <DIR> --prompt-ids 151669,152105,... [options]

OPTIONS:
  --model, --weights <PATH>  Carbon model dir (or model.safetensors/.gguf inside)
  --prompt <TEXT|DNA>        Prompt. A bare ACGTN sequence is auto-wrapped as a
                             <dna>…</dna> region; text with <dna> tags is honored.
  --prompt-ids <a,b,...>     Raw token ids instead of --prompt.
  --dna / --no-dna           Force / disable treating the whole prompt as DNA.
  --device <cpu|metal|mlx|cuda|...>   Execution device (default cpu).
  --max-tokens <N>           New tokens to generate (default 64).
  --max-seq <N>              Compile / KV-cache sequence cap (default 512).
  --temperature <F>          Sampling temperature (default 0 = greedy).
  --top-p <F>                Nucleus sampling top-p (default 1).
  --packed                   Packed GGUF matmul (K-quant weights only).
  --raw                      Decode without skipping special tokens.
  -h, --help                 Show this help.
";

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut prompt: Option<String> = None;
    let mut prompt_ids: Option<Vec<u32>> = None;
    let mut max_tokens = 64usize;
    let mut max_seq = 512usize;
    let mut temperature = 0f32;
    let mut top_p = 1f32;
    let mut packed: Option<bool> = None;
    let mut force_dna = false;
    let mut no_dna = false;
    let mut raw = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "--weights" => weights = Some(req(args, &mut i)?.into()),
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
            "--max-seq" => max_seq = req(args, &mut i)?.parse().context("--max-seq: usize")?,
            "--temperature" => {
                temperature = req(args, &mut i)?.parse().context("--temperature: f32")?;
            }
            "--top-p" => top_p = req(args, &mut i)?.parse().context("--top-p: f32")?,
            "--dna" => {
                force_dna = true;
                i += 1;
            }
            "--no-dna" => {
                no_dna = true;
                i += 1;
            }
            "--packed" => {
                packed = Some(true);
                i += 1;
            }
            "--no-packed" => {
                packed = Some(false);
                i += 1;
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

    let weights = weights.ok_or_else(|| anyhow!("--model (or --weights) is required"))?;
    let device = parse_llama32_device(&device)?;
    let sample = SampleOpts {
        temperature,
        top_p,
        ..SampleOpts::greedy()
    };

    let mut b = CarbonRunner::builder()
        .weights(weights.clone())
        .device(device)
        .max_seq(max_seq)
        .sample(sample);
    if let Some(p) = packed {
        b = b.packed_weights(p);
    }

    eprintln!("[rlx-carbon] loading weights={weights:?} device={device:?} max_seq={max_seq}");
    let mut runner = b.build()?;
    eprintln!(
        "[rlx-carbon] ready — vocab={} hidden={} layers={} kv_heads={}",
        runner.config().vocab_size,
        runner.config().hidden_size,
        runner.config().num_hidden_layers,
        runner.config().num_key_value_heads,
    );

    // Resolve prompt ids.
    let ids = if let Some(ids) = prompt_ids {
        ids
    } else if let Some(text) = prompt.as_deref() {
        encode_prompt(&runner, text, force_dna, no_dna)?
    } else {
        bail!("provide --prompt <SEQ> or --prompt-ids <a,b,...>");
    };
    if ids.is_empty() {
        bail!("empty prompt after tokenization");
    }

    generate_and_report(&mut runner, &ids, max_tokens, raw)
}

#[cfg(feature = "tokenizer")]
fn encode_prompt(
    runner: &CarbonRunner,
    text: &str,
    force_dna: bool,
    no_dna: bool,
) -> Result<Vec<u32>> {
    let has_tag = text.contains("<dna>");
    // Auto-detect a bare nucleotide sequence so `--prompt ATCG…` just works.
    let looks_dna = !has_tag
        && !text.trim().is_empty()
        && text.chars().all(|c| {
            c.is_whitespace() || matches!(c.to_ascii_uppercase(), 'A' | 'C' | 'G' | 'T' | 'N')
        });
    let treat_dna = !no_dna && (force_dna || looks_dna);

    if has_tag {
        // Honor the user's own tags verbatim (open or closed).
        return runner.tokenizer().encode_opt(text, Some(false));
    }
    if treat_dna {
        // OPEN region (`<dna>…` with no closing tag) so the model *continues*
        // the sequence. A closed `</dna>` marks the sequence complete and the
        // model immediately emits end-of-sequence.
        eprintln!("[rlx-carbon] treating prompt as an open DNA region (<dna>…) for continuation");
        return runner
            .tokenizer()
            .encode_opt(&format!("<dna>{text}"), Some(false));
    }
    // Plain text → base BPE.
    runner.tokenizer().encode_opt(text, Some(false))
}

#[cfg(not(feature = "tokenizer"))]
fn encode_prompt(_r: &CarbonRunner, _t: &str, _f: bool, _n: bool) -> Result<Vec<u32>> {
    bail!(
        "--prompt requires the `tokenizer` feature; rebuild with --features tokenizer, or use --prompt-ids"
    )
}

#[cfg(feature = "tokenizer")]
fn generate_and_report(
    runner: &mut CarbonRunner,
    ids: &[u32],
    max_tokens: usize,
    raw: bool,
) -> Result<()> {
    let skip_special = !raw;
    let t0 = std::time::Instant::now();
    let generated = runner.generate_ids_streaming(ids, max_tokens, skip_special, |piece| {
        print!("{piece}");
        std::io::stdout().flush().ok();
    })?;
    let dt = t0.elapsed();
    println!();
    eprintln!(
        "[rlx-carbon] generated {} tokens in {:.2?} ({:.1} tok/s)",
        generated.len(),
        dt,
        generated.len() as f64 / dt.as_secs_f64().max(1e-9)
    );
    let full = runner.tokenizer().decode(&generated, skip_special)?;
    println!("[rlx-carbon] continuation:\n{full}");
    Ok(())
}

#[cfg(not(feature = "tokenizer"))]
fn generate_and_report(
    runner: &mut CarbonRunner,
    ids: &[u32],
    max_tokens: usize,
    _raw: bool,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    runner.generate_ids(ids, max_tokens, |tok| generated.push(tok))?;
    let dt = t0.elapsed();
    eprintln!(
        "[rlx-carbon] generated {} tokens in {:.2?} (ids only; build with --features tokenizer to detokenize)",
        generated.len(),
        dt
    );
    let rendered: Vec<String> = generated.iter().map(|t| t.to_string()).collect();
    println!("{}", rendered.join(","));
    Ok(())
}

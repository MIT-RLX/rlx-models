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

//! `rlx-minicpm-sala` command line: one chat turn.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use rlx_qwen3::sampling::SampleOpts;

use crate::{SalaRunner, chat_prompt};

const USAGE: &str = "rlx-minicpm-sala --model-dir <dir> --prompt \"…\" [--raw] [--max-tokens N] \
    [--temperature T] [--chunk N] [--max-seq N] [--f32-weights] \
    [--device cpu|metal|mlx|cuda|rocm|gpu|vulkan]";

pub fn run(args: &[String]) -> Result<()> {
    let mut dir: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut device = "cpu".to_string();
    let mut max_tokens = 256usize;
    let mut temperature = 0.0f32;
    let mut chunk: Option<usize> = None;
    let mut max_seq: Option<usize> = None;
    let mut raw = false;
    let mut f32_weights = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" | "--weights" => dir = Some(req(args, &mut i)?.into()),
            "--prompt" => prompt = Some(req(args, &mut i)?),
            "--device" => device = req(args, &mut i)?,
            "--max-tokens" => max_tokens = req(args, &mut i)?.parse()?,
            "--temperature" => temperature = req(args, &mut i)?.parse()?,
            "--chunk" => chunk = Some(req(args, &mut i)?.parse()?),
            "--max-seq" => max_seq = Some(req(args, &mut i)?.parse()?),
            // Feed the prompt as is, without the chat template.
            "--raw" => {
                raw = true;
                i += 1;
            }
            "--f32-weights" => {
                f32_weights = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown flag: {other}\n{USAGE}"),
        }
    }
    let dir = dir.ok_or_else(|| anyhow!("--model-dir is required\n{USAGE}"))?;
    let prompt = prompt.ok_or_else(|| anyhow!("--prompt is required\n{USAGE}"))?;
    let device = parse_standard_device("minicpm-sala", &device)?;
    let mut b = SalaRunner::builder()
        .model_dir(&dir)
        .device(device)
        .f32_weights(f32_weights);
    if let Some(n) = chunk {
        b = b.chunk(n);
    }
    if let Some(n) = max_seq {
        b = b.max_seq(n);
    }
    let mut runner = b.build()?;
    let tok_path = runner
        .tokenizer()
        .ok_or_else(|| anyhow!("{dir:?} has no tokenizer.json"))?
        .to_path_buf();
    let tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow!("load {tok_path:?}: {e}"))?;
    let text = if raw { prompt } else { chat_prompt(&prompt) };
    let ids = tok
        .encode(text.as_str(), false)
        .map_err(|e| anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec();
    let opts = if temperature > 0.0 {
        SampleOpts {
            temperature,
            ..SampleOpts::greedy()
        }
    } else {
        SampleOpts::greedy()
    };
    let t = web_time::Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut shown = 0usize;
    let mut stdout = std::io::stdout();
    let new = runner.generate(&ids, max_tokens, opts, |id| {
        out.push(id);
        if let Ok(s) = tok.decode(&out, true)
            && !s.ends_with('\u{FFFD}')
            && s.len() > shown
        {
            let _ = stdout.write_all(&s.as_bytes()[shown..]);
            let _ = stdout.flush();
            shown = s.len();
        }
        true
    })?;
    println!();
    eprintln!(
        "[rlx-minicpm-sala] {} prompt + {} new tokens in {:.1}s",
        ids.len(),
        new.len(),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

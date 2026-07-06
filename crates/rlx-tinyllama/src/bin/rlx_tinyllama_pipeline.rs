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

//! Transformers-style one-liner CLI for TinyLlama.
//!
//! ```sh
//! rlx-tinyllama-pipeline --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
//!   --prompt "What is the capital of France?"
//! # raw completion instead of chat:
//! rlx-tinyllama-pipeline --model <id|path> --completion --prompt "Once upon a time"
//! ```

use std::io::Write;
use std::process::ExitCode;

use rlx_runtime::Device;
use rlx_tinyllama::pipeline::{ChatMessage, GenerationConfig, TextGeneration};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rlx-tinyllama-pipeline: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let mut model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string();
    let mut prompt: Option<String> = None;
    let mut system: Option<String> = None;
    let mut device = Device::Cpu;
    let mut max_new_tokens = 256usize;
    let mut temperature = 0f32;
    let mut top_p = 1f32;
    let mut completion = false;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                model = next(&args, &mut i)?;
            }
            "--prompt" | "-p" => {
                prompt = Some(next(&args, &mut i)?);
            }
            "--system" => {
                system = Some(next(&args, &mut i)?);
            }
            "--device" | "-d" => {
                device = parse_device(&next(&args, &mut i)?)?;
            }
            "--max-new-tokens" | "-n" => {
                max_new_tokens = next(&args, &mut i)?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--max-new-tokens expects an integer"))?;
            }
            "--temperature" | "-t" => {
                temperature = next(&args, &mut i)?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--temperature expects a float"))?;
            }
            "--top-p" => {
                top_p = next(&args, &mut i)?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--top-p expects a float"))?;
            }
            "--completion" => {
                completion = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown flag: {other} (try --help)"),
        }
    }

    let prompt = prompt.ok_or_else(|| anyhow::anyhow!("--prompt is required (try --help)"))?;

    eprintln!("[rlx-tinyllama-pipeline] loading {model} on {device:?} …");
    let mut pipe = TextGeneration::from_pretrained_on(&model, device)?;

    let cfg = GenerationConfig {
        max_new_tokens,
        temperature,
        top_p,
        ..GenerationConfig::default()
    };

    let mut out = std::io::stdout();
    let emit = |piece: &str, out: &mut std::io::Stdout| {
        let _ = out.write_all(piece.as_bytes());
        let _ = out.flush();
    };

    if completion {
        print!("{prompt}");
        emit("", &mut out);
        pipe.generate_stream(&prompt, &cfg, |piece| emit(piece, &mut out))?;
    } else {
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(ChatMessage::system(sys));
        }
        messages.push(ChatMessage::user(prompt));
        pipe.chat_stream(&messages, &cfg, |piece| emit(piece, &mut out))?;
    }
    println!();
    Ok(())
}

fn next(args: &[String], i: &mut usize) -> anyhow::Result<String> {
    let v = args
        .get(*i + 1)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{} needs a value", args[*i]))?;
    *i += 2;
    Ok(v)
}

fn parse_device(s: &str) -> anyhow::Result<Device> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "vulkan" => Device::Vulkan,
        "rocm" => Device::Rocm,
        other => anyhow::bail!("unknown device {other:?} (cpu|metal|mlx|cuda|vulkan|rocm)"),
    })
}

fn print_help() {
    eprintln!(
        "rlx-tinyllama-pipeline — transformers-style TinyLlama inference\n\
         \n\
         USAGE:\n\
           rlx-tinyllama-pipeline --prompt \"…\" [OPTIONS]\n\
         \n\
         OPTIONS:\n\
           -m, --model <id|path>       HF repo id or local dir/.safetensors/.gguf\n\
                                       [default: TinyLlama/TinyLlama-1.1B-Chat-v1.0]\n\
           -p, --prompt <text>         Prompt / user message (required)\n\
               --system <text>         Optional system message (chat mode)\n\
               --completion            Raw completion instead of chat template\n\
           -d, --device <dev>          cpu|metal|mlx|cuda|vulkan|rocm [default: cpu]\n\
           -n, --max-new-tokens <n>    [default: 256]\n\
           -t, --temperature <f>       0 = greedy [default: 0]\n\
               --top-p <f>             nucleus cutoff [default: 1]\n"
    );
}

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

//! Carbon DNA generation on any rlx backend.
//!
//! Carbon is a stock Llama graph, so it runs on every backend `rlx-llama32`
//! supports. Pick the device with `--device` (build with the matching backend
//! feature); greedy output is identical across backends.
//!
//! ```sh
//! # CPU (always available)
//! cargo run -p rlx-carbon --example dna_generate --features tokenizer --release -- \
//!   --model /path/to/Carbon-500M --device cpu \
//!   --prompt ATGGCGACCTTTAGCGATCTGGGCAAAGAACTGCGTACC --max-tokens 48
//!
//! # Apple GPU/ANE (Metal, MLX, wgpu, CoreML)
//! cargo run -p rlx-carbon --example dna_generate --features tokenizer,apple-silicon --release -- \
//!   --model /path/to/Carbon-500M --device metal   --prompt ATGGCGACCTTTAGCGATCTG
//!
//! # NVIDIA / AMD / portable Vulkan
//! cargo run -p rlx-carbon --example dna_generate --features tokenizer,cuda   --release -- --device cuda   --model DIR
//! cargo run -p rlx-carbon --example dna_generate --features tokenizer,rocm   --release -- --device rocm   --model DIR
//! cargo run -p rlx-carbon --example dna_generate --features tokenizer,vulkan --release -- --device vulkan --model DIR
//! ```
//!
//! The model directory (or `RLX_CARBON_MODEL`) must hold `config.json`,
//! `model.safetensors`, `tokenizer.json`, and `dna_config.json`.

use anyhow::{Result, anyhow};
use rlx_carbon::CarbonRunner;
use rlx_cli::parse_llama32_device;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model: Option<String> = std::env::var("RLX_CARBON_MODEL").ok();
    let mut device = "cpu".to_string();
    let mut prompt = "ATGGCGACCTTTAGCGATCTGGGCAAAGAACTGCGTACCGATCTGGCAGAT".to_string();
    let mut max_tokens = 48usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "--weights" => {
                i += 1;
                model = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--model needs a value"))?
                        .clone(),
                );
            }
            "--device" => {
                i += 1;
                device = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--device needs a value"))?
                    .clone();
            }
            "--prompt" => {
                i += 1;
                prompt = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--prompt needs a value"))?
                    .clone();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-tokens needs a value"))?
                    .parse()?;
            }
            other => return Err(anyhow!("unknown arg {other}")),
        }
        i += 1;
    }

    let model = model.ok_or_else(|| {
        anyhow!("pass --model <dir> or set RLX_CARBON_MODEL to a Carbon model directory")
    })?;
    let dev = parse_llama32_device(&device)?;

    let t_load = std::time::Instant::now();
    let mut carbon = CarbonRunner::from_pretrained(&model, dev)?;
    println!(
        "[dna_generate] loaded Carbon on {dev:?} in {:.1?} — vocab={} hidden={} layers={}",
        t_load.elapsed(),
        carbon.config().vocab_size,
        carbon.config().hidden_size,
        carbon.config().num_hidden_layers,
    );

    // `Some(true)` opens a `<dna>…` region so the model *continues* the sequence.
    let t_gen = std::time::Instant::now();
    let out = carbon.complete(&prompt, max_tokens, Some(true))?;
    let dt = t_gen.elapsed();

    println!("[dna_generate] prompt      : {prompt}");
    println!("[dna_generate] continuation: {}", out.text);
    println!(
        "[dna_generate] {} tokens (~{} bp) in {:.2?} ({:.1} tok/s) on {dev:?}",
        out.generated.len(),
        out.text.len(),
        dt,
        out.generated.len() as f64 / dt.as_secs_f64().max(1e-9),
    );
    Ok(())
}

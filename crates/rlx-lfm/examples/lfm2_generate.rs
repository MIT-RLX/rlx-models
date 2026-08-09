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

//! LFM2 / LFM2.5 GGUF text generation on any rlx backend.
//!
//! LFM2.5 is a hybrid ShortConv + GQA-attention decoder; the dense-F32 graph
//! runs natively on every standard backend. Pick the device with `--device`
//! (build with the matching backend feature).
//!
//! ```sh
//! # CPU
//! cargo run -p rlx-lfm --example lfm2_generate --features tokenizer --release -- \
//!   --weights /path/LFM2.5-2.6B-Q4_K_M.gguf --device cpu \
//!   --prompt "The capital of France is" --max-tokens 24
//!
//! # Apple GPU (metal | mlx | gpu)
//! cargo run -p rlx-lfm --example lfm2_generate --features tokenizer,apple-silicon --release -- \
//!   --weights DIR_OR_FILE.gguf --device metal --prompt "Q: 2+2? A:"
//!
//! # NVIDIA / AMD / Vulkan
//! cargo run -p rlx-lfm --example lfm2_generate --features tokenizer,cuda   --release -- --device cuda   --weights FILE.gguf
//! cargo run -p rlx-lfm --example lfm2_generate --features tokenizer,vulkan --release -- --device vulkan --weights FILE.gguf
//! ```

use anyhow::{Result, anyhow};
use rlx_cli::parse_standard_device;
use rlx_lfm::{Lfm2GgufRunner, resolve_gguf};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut weights: Option<String> = std::env::var("RLX_LFM_WEIGHTS").ok();
    let mut device = "cpu".to_string();
    let mut prompt = "The capital of France is".to_string();
    let mut max_tokens = 24usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "--model" => {
                i += 1;
                weights = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--weights needs a value"))?
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

    let weights =
        weights.ok_or_else(|| anyhow!("pass --weights <FILE.gguf|DIR> or set RLX_LFM_WEIGHTS"))?;
    let gguf = resolve_gguf(&PathBuf::from(&weights))?;
    let dev = parse_standard_device("lfm", &device)?;

    let t_load = std::time::Instant::now();
    let runner = Lfm2GgufRunner::open(&gguf, dev)?;
    let c = runner.config();
    println!(
        "[lfm2_generate] loaded on {dev:?} in {:.1?} — hidden={} layers={} heads={}/{} attn_layers={:?} vocab={}",
        t_load.elapsed(),
        c.hidden_size,
        c.num_hidden_layers,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.full_attn_layers,
        c.vocab_size,
    );

    let ids = rlx_qwen35::encode_prompt_from_gguf(&gguf, &prompt)?;
    let t_gen = std::time::Instant::now();
    let generated = runner.generate(&ids, max_tokens, |_| true)?;
    let dt = t_gen.elapsed();
    let text = rlx_qwen35::decode_ids_from_gguf(&gguf, &generated, true)?;

    println!("[lfm2_generate] prompt      : {prompt}");
    println!("[lfm2_generate] continuation: {text}");
    println!(
        "[lfm2_generate] {} tokens in {:.2?} ({:.1} tok/s) on {dev:?}",
        generated.len(),
        dt,
        generated.len() as f64 / dt.as_secs_f64().max(1e-9),
    );
    Ok(())
}

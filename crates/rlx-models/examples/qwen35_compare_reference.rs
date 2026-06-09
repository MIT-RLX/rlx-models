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

//! llama.cpp parity reference — emits `REF_LOGIT idx=… token=… value=…` lines.
//!
//! ```text
//! cargo run --release -p rlx-models --features parity-llama --example qwen35_compare_reference -- \
//!     /path/to/model.gguf --prompt-ids 1,2,3 --top-k 16
//! ```

#[cfg(not(feature = "parity-llama"))]
fn main() {
    eprintln!("qwen35_compare_reference requires --features parity-llama");
    std::process::exit(1);
}

#[cfg(feature = "parity-llama")]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, Result, bail};
    use rlx_models::qwen35::llama_reference;
    let mut args = std::env::args().skip(1);
    let weights = args.next().context(
        "usage: qwen35_compare_reference <weights.gguf> [--prompt-ids 1,2,3] [--top-k N]",
    )?;

    let mut prompt_ids: Vec<u32> = vec![1, 2, 3];
    let mut top_k = 16usize;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--prompt-ids" => {
                let raw = args.next().context("--prompt-ids")?;
                prompt_ids = raw
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<std::result::Result<_, _>>()
                    .context("--prompt-ids")?;
            }
            "--top-k" => {
                top_k = args.next().context("--top-k")?.parse()?;
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    eprintln!("# Loading {weights} via llama-cpp-2…");
    let pairs = llama_reference::top_k_logits(weights.as_ref(), &prompt_ids, top_k)?;
    eprintln!("# REF logits: top-{top_k} from last prompt token");
    for (rank, (tok, val)) in pairs.iter().enumerate() {
        println!("REF_LOGIT idx={rank} token={tok} value={val:.6}");
    }
    Ok(())
}

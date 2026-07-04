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

//! Transformers-style TinyLlama inference in a handful of lines.
//!
//! ```sh
//! cargo run -p rlx-tinyllama --example pipeline --features pipeline --release
//! # or point at a local checkpoint:
//! MODEL=/tmp/rlx-weights/TinyLlama-1.1B-Chat-v1.0 \
//!   cargo run -p rlx-tinyllama --example pipeline --features pipeline --release
//! ```
//!
//! Compare with the `transformers` equivalent:
//!
//! ```python
//! from transformers import pipeline
//! pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
//! print(pipe("Once upon a time", max_new_tokens=64)[0]["generated_text"])
//! ```

use std::io::Write;

use rlx_tinyllama::pipeline::{ChatMessage, GenerationConfig, TextGeneration};

fn main() -> anyhow::Result<()> {
    let model = std::env::var("MODEL")
        .unwrap_or_else(|_| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    println!("loading {model} …");
    let mut pipe = TextGeneration::from_pretrained(&model)?;

    // ── raw completion ────────────────────────────────────────────
    let cfg = GenerationConfig::default().with_max_new_tokens(64);
    let out = pipe.generate("Once upon a time", &cfg)?;
    println!("\n[completion]\nOnce upon a time{out}");

    // ── chat, streamed token-by-token ─────────────────────────────
    println!("\n[chat] (streaming)");
    let messages = [ChatMessage::user("Name three primary colors, comma separated.")];
    print!("assistant: ");
    std::io::stdout().flush().ok();
    pipe.chat_stream(&messages, &cfg, |piece| {
        print!("{piece}");
        std::io::stdout().flush().ok();
    })?;
    println!();

    Ok(())
}

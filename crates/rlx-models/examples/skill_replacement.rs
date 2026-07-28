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

//! End-to-end migration example (PLAN.md M8).
//!
//! Demonstrates the call shape `skill` adopts after dropping
//! `llama-cpp-4`. Everything goes through `rlx_models::run::*` —
//! no per-family imports required:
//!
//! ```text
//! GGUF path
//!   ├─ auto_chat_template(path)      → ChatTemplate
//!   ├─ template.render(messages, …)  → rendered prompt string
//!   ├─ auto_tokenize(path, prompt)   → Vec<u32>  prompt ids
//!   ├─ auto_runner(path)             → Box<dyn LmRunner>
//!   └─ runner.generate(ids, N, &cb)  → Vec<u32>  generated ids
//! ```
//!
//! Run with:
//!
//! ```sh
//! cargo run --example skill_replacement -- \
//!     --weights /path/to/Qwen3.5-4B-Q4_K_M.gguf \
//!     --prompt "What is 2 + 2?" \
//!     --n-new 32
//! ```

use anyhow::{Context, Result, bail};
use rlx_models::run::{
    ChatMessage, auto_chat_template, auto_detokenize, auto_runner, auto_tokenize,
};
use std::path::PathBuf;

#[derive(Debug, Default)]
struct Args {
    weights: Option<PathBuf>,
    prompt: Option<String>,
    system: Option<String>,
    n_new: usize,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        n_new: 32,
        ..Default::default()
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" => args.weights = it.next().map(PathBuf::from),
            "--prompt" => args.prompt = it.next(),
            "--system" => args.system = it.next(),
            "--n-new" => args.n_new = it.next().and_then(|s| s.parse().ok()).unwrap_or(32),
            other => bail!("unknown arg: {other}"),
        }
    }
    if args.weights.is_none() {
        bail!("--weights <path/to.gguf> required");
    }
    if args.prompt.is_none() {
        bail!("--prompt <text> required");
    }
    Ok(args)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let weights = args.weights.as_ref().unwrap();
    let prompt = args.prompt.as_ref().unwrap();

    eprintln!("# 1) sniff arch + load chat template from {weights:?}");
    let template =
        auto_chat_template(weights).with_context(|| format!("auto_chat_template({weights:?})"))?;
    eprintln!(
        "#    bos = {:?}, eos = {:?}",
        template.bos_token(),
        template.eos_token()
    );

    eprintln!("# 2) render chat template");
    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = &args.system {
        messages.push(ChatMessage::system(sys.clone()));
    }
    messages.push(ChatMessage::user(prompt.clone()));
    let rendered = template.render(&messages, true)?;
    eprintln!("#    rendered prompt ({} chars):", rendered.len());
    for line in rendered.lines().take(8) {
        eprintln!("#      {line}");
    }
    if rendered.lines().count() > 8 {
        eprintln!("#      …");
    }

    eprintln!("# 3) tokenize");
    let prompt_ids = auto_tokenize(weights, &rendered, None)
        .with_context(|| format!("auto_tokenize({weights:?})"))?;
    eprintln!("#    {} prompt tokens", prompt_ids.len());

    eprintln!("# 4) build runner");
    let mut runner = auto_runner(weights).with_context(|| format!("auto_runner({weights:?})"))?;
    eprintln!("#    family = {}", runner.family());

    eprintln!("# 5) generate {} tokens", args.n_new);
    let mut produced: Vec<u32> = Vec::new();
    let generated = runner.generate(&prompt_ids, args.n_new, &mut |tok: u32| -> bool {
        produced.push(tok);
        // Streaming hook — in `skill`, this would push the token
        // to the UI / SSE channel. The example just prints the id.
        print!("{tok} ");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        // Return true to keep generating; return false here to
        // stop early on an EOS id (Qwen35Runner honors this;
        // Qwen3 / Gemma / Llama32 currently ignore it).
        true
    })?;
    println!();
    eprintln!("# generated {} ids: {:?}", generated.len(), generated);

    eprintln!("# 6) detokenize → text");
    let text = auto_detokenize(weights, &generated, None, true)
        .with_context(|| format!("auto_detokenize({weights:?})"))?;
    println!("\n=== decoded output ===\n{text}\n======================");

    Ok(())
}

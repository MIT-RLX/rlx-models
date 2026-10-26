// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Transcode `lm/model-ternary.safetensors` to the `Q2_0` GGUF explicitly
//! (the runtime does this on first load).
//!
//! ```text
//! cargo run -p rlx-kittentts2 --release --example transcode -- \
//!     weights/tts/kitten-tts-2 /tmp/model-q2_0.gguf
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use rlx_kittentts2::Kitten2Config;
use rlx_qwen3::Qwen3Config;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = PathBuf::from(args.first().context("repo dir")?);
    let out = PathBuf::from(args.get(1).context("output .gguf")?);
    let cfg = Kitten2Config::from_dir(&repo)?;
    let lm = Qwen3Config::from_file(&repo.join(&cfg.lm_dir).join("config.json"))?;
    let ternary = repo.join(cfg.lm_packed.as_deref().context("no lm_packed")?);
    let r = rlx_kittentts2::ternary::transcode(&ternary, &lm, &out)?;
    println!("{r:?} → {}", out.display());
    Ok(())
}

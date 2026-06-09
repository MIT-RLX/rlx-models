// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Download MiniCPM5-1B GGUF quants from Hugging Face.
//!
//! ```bash
//! just fetch-minicpm5-gguf Q4_K_M
//! just fetch-minicpm5-gguf all
//! # or: cargo run -p rlx-models --example minicpm5_gguf_download --features hf-download --release -- Q4_K_M
//!
//! just minicpm5 -- --weights …/MiniCPM5-1B-Q4_K_M.gguf --packed --device cpu --prompt-ids 1,42
//! ```

use rlx_minicpm5::{MINICPM5_GGUF_FILES, fetch_minicpm5_gguf};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let cache = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| rlx_minicpm5::default_hf_cache_dir());

    let dest = std::env::var("RLX_MINICPM5_GGUF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rlx-weights/MiniCPM5-1B-GGUF"));

    let arg = std::env::args().nth(1);
    let labels: Vec<&str> = match arg.as_deref() {
        Some("all") => MINICPM5_GGUF_FILES.iter().map(|(l, _)| *l).collect(),
        Some(label) => vec![label],
        None => vec!["Q4_K_M"],
    };

    std::fs::create_dir_all(&dest)?;
    for label in labels {
        let path = fetch_minicpm5_gguf(&cache, &dest, label)?;
        println!("{label}: {}", path.display());
    }
    println!("\nexport RLX_MINICPM5_GGUF_DIR={}", dest.display());
    Ok(())
}

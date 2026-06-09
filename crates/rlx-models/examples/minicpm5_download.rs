// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Download [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) via Hugging Face Hub.
//!
//! ```bash
//! just fetch-minicpm5
//! # or: cargo run -p rlx-models --example minicpm5_download --features hf-download --release
//!
//! just minicpm5 -- --weights /tmp/rlx-weights/MiniCPM5-1B/model-00000-of-00001.safetensors …
//! just example-minicpm5
//! ```

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let cache = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| rlx_minicpm5::default_hf_cache_dir());

    let dest = std::env::var("MINICPM5_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rlx-weights/MiniCPM5-1B"));

    let dir = rlx_minicpm5::fetch_minicpm5_1b(&cache, &dest)?;
    let weights = dir.join("model-00000-of-00001.safetensors");
    let weights = if weights.is_file() {
        weights
    } else {
        dir.join("model.safetensors")
    };

    println!("MiniCPM5-1B ready under:\n  {}", dir.display());
    println!("\nexport RLX_MINICPM5_WEIGHTS={}", weights.display());
    println!(
        "export RLX_MINICPM5_CONFIG={}",
        dir.join("config.json").display()
    );
    println!("export MINICPM5_MODEL_DIR={}", dir.display());
    Ok(())
}

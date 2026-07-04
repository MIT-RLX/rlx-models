// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Download [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) via Hugging Face Hub.
//!
//! ```bash
//! just fetch-tinyllama
//! # or: cargo run -p rlx-models --example tinyllama_download --features hf-download,tinyllama --release
//!
//! just tinyllama -- --weights /tmp/rlx-weights/TinyLlama-1.1B-Chat-v1.0/model-00001-of-00003.safetensors …
//! ```

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let cache = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| rlx_tinyllama::default_hf_cache_dir());

    let dest = std::env::var("TINYLLAMA_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rlx-weights/TinyLlama-1.1B-Chat-v1.0"));

    let dir = rlx_tinyllama::fetch_tinyllama_1_1b(&cache, &dest)?;
    let weights = dir.join("model-00001-of-00003.safetensors");
    let weights = if weights.is_file() {
        weights
    } else {
        dir.join("model.safetensors")
    };

    println!("TinyLlama-1.1B-Chat ready under:\n  {}", dir.display());
    println!("\nexport RLX_TINYLLAMA_WEIGHTS={}", weights.display());
    println!(
        "export RLX_TINYLLAMA_CONFIG={}",
        dir.join("config.json").display()
    );
    println!("export TINYLLAMA_MODEL_DIR={}", dir.display());
    Ok(())
}

// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//!
//! Download Nanbeige4.2-3B from Hugging Face.
//!
//! ```sh
//! just fetch-nanbeige
//! # or: cargo run -p rlx-nanbeige --example fetch_nanbeige --features hf-download --release
//! ```

use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let dest = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("NANBEIGE_MODEL_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/tmp/rlx-weights/Nanbeige4.2-3B"));
    let cache = std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
        .unwrap_or_else(|_| rlx_nanbeige::default_hf_cache_dir());
    println!("fetching {} → {dest:?}", rlx_nanbeige::HF_MODEL_ID_3B);
    let dir = rlx_nanbeige::fetch_nanbeige42_3b(&cache, &dest)?;
    println!("ok: {dir:?}");
    println!(
        "weights: {}/model-00001-of-00002.safetensors",
        dir.display()
    );
    Ok(())
}

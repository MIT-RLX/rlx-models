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

//! Download `inclusionAI/LLaDA2.0-mini` (smallest public LLaDA2 MoE) for e2e tests.
//!
//! ```bash
//! cargo run -p rlx-models --example llada2_download --features hf-download
//! export LLADA2_MODEL_DIR=~/.cache/huggingface/hub/models--inclusionAI--LLaDA2.0-mini/snapshots/<hash>
//! cargo test -p rlx-models --test llada2_e2e_parity -- --nocapture
//! ```

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let cache = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                .join(".cache/huggingface")
        });
    let dir = rlx_models::llada2::download::download_llada2_mini(&cache)?;
    println!("LLaDA2 weights ready under:\n  {}", dir.display());
    println!("\nexport LLADA2_MODEL_DIR={}", dir.display());
    Ok(())
}

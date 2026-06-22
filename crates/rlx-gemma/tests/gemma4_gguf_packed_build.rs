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

//! Quick check: Gemma 4 12B Q4 GGUF packed graph builds (no Metal compile).
//!
//! ```bash
//! RLX_GEMMA4_FIXTURE=/path/to/gemma4-12B-it \
//!   cargo test -p rlx-gemma --test gemma4_gguf_packed_build -- --nocapture
//! ```

mod gemma4_bench_common;

use anyhow::Result;
use gemma4_bench_common::{resolve_gemma4_config, resolve_gemma4_gguf};
use rlx_core::weight_loader::GgufLoader;
use rlx_gemma::build_gemma_decode_graph_sized_packed;
use rlx_gemma::build_gemma_graph_sized_packed;
use std::collections::HashMap;
use std::path::PathBuf;

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os("RLX_GEMMA4_FIXTURE").map(PathBuf::from)
}

#[test]
fn gemma4_q4_packed_graph_builds_without_rope_panic() -> Result<()> {
    let Some(dir) = fixture_dir() else {
        eprintln!("[gemma4 gguf build] RLX_GEMMA4_FIXTURE unset — skip");
        return Ok(());
    };
    let Some(gguf) = resolve_gemma4_gguf(&dir) else {
        eprintln!("[gemma4 gguf build] no .gguf in fixture — skip");
        return Ok(());
    };
    let cfg = resolve_gemma4_config(&dir, &gguf)?;
    let mut loader = GgufLoader::from_file(gguf.to_str().unwrap())?;
    let mut packed = HashMap::new();
    let (graph, _params) =
        build_gemma_graph_sized_packed(&cfg, &mut loader, 1, 128, true, true, false, &mut packed)?;
    eprintln!(
        "[gemma4 gguf build] ok: {} nodes, {} packed tensors",
        graph.nodes().len(),
        packed.len()
    );
    Ok(())
}

#[test]
fn gemma4_q4_packed_decode_graph_builds() -> Result<()> {
    let Some(dir) = fixture_dir() else {
        eprintln!("[gemma4 gguf decode build] RLX_GEMMA4_FIXTURE unset — skip");
        return Ok(());
    };
    let Some(gguf) = resolve_gemma4_gguf(&dir) else {
        eprintln!("[gemma4 gguf decode build] no .gguf in fixture — skip");
        return Ok(());
    };
    let cfg = resolve_gemma4_config(&dir, &gguf)?;
    let mut loader = GgufLoader::from_file(gguf.to_str().unwrap())?;
    let mut packed = HashMap::new();
    let (graph, _params) =
        build_gemma_decode_graph_sized_packed(&cfg, &mut loader, 1, 128, true, &mut packed)?;
    eprintln!(
        "[gemma4 gguf decode build] ok: {} nodes, {} packed tensors",
        graph.nodes().len(),
        packed.len()
    );
    Ok(())
}

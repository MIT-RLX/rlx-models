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

// Env-gated: qwen35 weight load vocab width vs cfg metadata.

mod compile_support;

use rlx_gguf::GgufFile;
use rlx_models::qwen35::{Qwen35Config, Qwen35Weights};
use rlx_models::weight_loader::GgufLoader;
use std::path::PathBuf;

#[test]
fn qwen35_loaded_lm_vocab_matches_embedding_table() {
    let path = match std::env::var("QWEN35_GGUF_PATH") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => PathBuf::from("/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf"),
    };
    if !path.is_file() {
        eprintln!("skip: missing {}", path.display());
        return;
    }

    let raw = GgufFile::from_path(&path).expect("open");
    let cfg = Qwen35Config::from_gguf(&raw).expect("cfg");
    let mut loader = GgufLoader::from_file(path.to_str().unwrap()).expect("loader");
    let w = Qwen35Weights::from_loader(&mut loader, &cfg).expect("weights");

    eprintln!(
        "cfg.vocab_size={} token_embd.len={} lm_vocab_size={} hidden={}",
        cfg.vocab_size,
        w.token_embd.len(),
        w.lm_vocab_size(&cfg),
        cfg.hidden_size
    );
    assert_eq!(
        w.lm_vocab_size(&cfg),
        248_320,
        "expected full Qwen3.5 0.8B LM vocab from embedding table"
    );
}

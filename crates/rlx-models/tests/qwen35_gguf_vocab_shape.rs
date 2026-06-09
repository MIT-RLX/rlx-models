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

// Env-gated: print qwen35 GGUF vocab / embedding shape.
//
//   QWEN35_GGUF_PATH=/path/to/model.gguf cargo test -p rlx-models qwen35_gguf_vocab_shape --release -- --nocapture

mod compile_support;

use rlx_gguf::GgufFile;
use rlx_models::Qwen35Config;
use std::path::PathBuf;

#[test]
fn qwen35_gguf_vocab_shape() {
    let path = match std::env::var("QWEN35_GGUF_PATH") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => PathBuf::from("/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf"),
    };
    if !path.is_file() {
        eprintln!("skip qwen35_gguf_vocab_shape: missing {}", path.display());
        return;
    }

    let raw = GgufFile::from_path(&path).expect("open gguf");
    let cfg = Qwen35Config::from_gguf(&raw).expect("parse cfg");
    let emb = raw.tensors.get("token_embd.weight").expect("token_embd");
    let out = raw.tensors.get("output.weight");
    eprintln!(
        "qwen35 GGUF {}: cfg.vocab_size={} token_embd.shape={:?} dtype={:?} output={:?}",
        path.display(),
        cfg.vocab_size,
        emb.shape,
        emb.dtype,
        out.map(|t| t.shape.as_slice())
    );
}

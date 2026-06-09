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

// Env-gated: K-quant manual dequant vs GgufLoader::dequant_f32 for one trunk weight.
//
//   QWEN35_GGUF_PATH=/path/to/model.gguf cargo test -p rlx-models qwen35_kquant_dequant --release -- --nocapture

mod compile_support;

use rlx_gguf::{GgmlType, dequant_q4_k, dequant_q5_k, dequant_q6_k};
use rlx_models::weight_loader::GgufLoader;
use std::path::PathBuf;

const DEFAULT_NON_MTP_Q4: &str = "/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf";

fn gguf_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("QWEN35_GGUF_PATH") {
        if !p.is_empty() {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    let path = PathBuf::from(DEFAULT_NON_MTP_Q4);
    path.is_file().then_some(path)
}

fn manual_k_dequant(dtype: GgmlType, bytes: &[u8], n: usize) -> Option<Vec<f32>> {
    match dtype {
        GgmlType::Q4K => dequant_q4_k(bytes, n).ok(),
        GgmlType::Q5K => dequant_q5_k(bytes, n).ok(),
        GgmlType::Q6K => dequant_q6_k(bytes, n).ok(),
        _ => None,
    }
}

#[test]
fn qwen35_kquant_dequant_matches_gguf_loader_for_attn_qkv() {
    let path = match gguf_path() {
        Some(p) => p,
        None => {
            eprintln!("skip qwen35_kquant_dequant: set QWEN35_GGUF_PATH");
            return;
        }
    };

    let key = "blk.0.attn_qkv.weight";
    let loader = GgufLoader::from_file(path.to_str().unwrap()).expect("gguf");
    let t = loader
        .file()
        .get(key)
        .unwrap_or_else(|| panic!("missing tensor {key}"));
    let n = t.n_elements();
    let (ref_flat, _shape) = loader.file().dequant_f32(key).expect("dequant_f32");
    let bytes = loader.file().tensor_bytes(t).expect("tensor bytes");
    let manual = manual_k_dequant(t.dtype, bytes, n)
        .unwrap_or_else(|| panic!("no manual dequant for {:?}", t.dtype));

    assert_eq!(ref_flat.len(), manual.len());
    let mut max_abs = 0f32;
    for (a, b) in ref_flat.iter().zip(&manual) {
        max_abs = max_abs.max((a - b).abs());
    }
    eprintln!(
        "qwen35 {:?} dequant {key}: n={n} max_abs={max_abs:.6e}",
        t.dtype
    );
    assert!(
        max_abs <= 1e-6,
        "{:?} manual dequant diverged from GgufLoader max_abs {max_abs:.6e}",
        t.dtype
    );
}

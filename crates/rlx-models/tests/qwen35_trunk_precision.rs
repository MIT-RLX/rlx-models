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

// Env-gated: compare packed vs F32 trunk hidden against llama.cpp.
//
//   QWEN35_RUN_LLAMA_PARITY=1 QWEN35_GGUF_PATH=/path/to/model.gguf \
//     cargo test -p rlx-models --test qwen35_trunk_precision --features parity-llama --release -- --nocapture

#![allow(dead_code)]

mod compile_support;

use rlx_ir::DType;
use rlx_models::qwen35::{
    Qwen35Config, Qwen35Weights, build_qwen35_graph_sized_ext, last_token_indices, pack_input_ids,
};
use rlx_models::weight_loader::GgufLoader;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

const DEFAULT_NON_MTP_Q4: &str = "/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf";

fn gguf_path() -> Option<PathBuf> {
    if std::env::var("QWEN35_RUN_LLAMA_PARITY").ok().as_deref() != Some("1") {
        return None;
    }
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / na.sqrt() / nb.sqrt()) as f32
}

fn run_hidden(path: &Path, packed: bool) -> Vec<f32> {
    let prompt_ids = vec![1u32, 2, 3];
    let max_seq = prompt_ids.len();
    let mut loader = GgufLoader::from_file(path.to_str().unwrap()).expect("gguf");
    loader.include_mtp(true);
    let cfg = Qwen35Config::from_gguf(loader.file()).expect("cfg");
    let weights = if packed {
        Qwen35Weights::from_loader_packed(&mut loader, &cfg).expect("packed")
    } else {
        Qwen35Weights::from_loader(&mut loader, &cfg).expect("f32")
    };
    let (graph, params, packed_params) = build_qwen35_graph_sized_ext(
        &cfg, weights, 1, max_seq, true, true, false, false, None, false, false, true, false,
    )
    .expect("graph");
    let mut compiled = compile_support::compile_qwen35_prefill(Device::Cpu, graph, params.clone());

    if packed {
        let mut loader2 = GgufLoader::from_file(path.to_str().unwrap()).expect("gguf2");
        loader2.include_mtp(true);
        for (param_name, (loader_key, _scheme, _shape)) in &packed_params {
            let bytes = loader2
                .tensor_bytes_borrowed(loader_key)
                .unwrap_or_else(|| panic!("missing {loader_key}"));
            compiled.set_param_typed(param_name, bytes, DType::U8);
        }
    }
    let padded = pack_input_ids(&[prompt_ids], max_seq).expect("pack");
    let last_idx = last_token_indices(&[max_seq]);
    let outs = compiled.run(&[
        ("input_ids", padded.as_slice()),
        ("last_token_idx", last_idx.as_slice()),
    ]);
    outs[0].to_vec()
}

#[test]
#[cfg(feature = "parity-llama")]
fn qwen35_f32_trunk_hidden_is_closer_to_llama_than_packed() {
    let path = match gguf_path() {
        Some(p) => p,
        None => {
            eprintln!("skip qwen35_trunk_precision: set QWEN35_RUN_LLAMA_PARITY=1");
            return;
        }
    };
    let prompt_ids = vec![1u32, 2, 3];
    let ref_hidden =
        rlx_models::qwen35::llama_reference::last_token_hidden(&path, &prompt_ids).expect("ref");

    let packed_h = run_hidden(&path, true);
    let f32_h = run_hidden(&path, false);
    let cos_packed = cosine_similarity(&packed_h, &ref_hidden);
    let cos_f32 = cosine_similarity(&f32_h, &ref_hidden);
    eprintln!("qwen35 trunk hidden vs llama:");
    eprintln!("  packed Q matmul: {cos_packed:.6}");
    eprintln!("  F32 dequant matmul: {cos_f32:.6}");
    assert!(
        cos_f32 >= cos_packed - 1e-6,
        "expected F32 trunk >= packed (f32={cos_f32:.6}, packed={cos_packed:.6})"
    );
}

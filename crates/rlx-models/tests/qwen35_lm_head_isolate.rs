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

// Env-gated: isolate LM-head vs trunk mismatch for Tier A.1 parity.
//
//   QWEN35_RUN_LLAMA_PARITY=1 QWEN35_GGUF_PATH=/path/to/model.gguf \
//     cargo test -p rlx-models --test qwen35_lm_head_isolate --features parity-llama --release -- --nocapture

#![allow(dead_code, clippy::collapsible_if, clippy::question_mark)]

mod compile_support;

use rlx_cpu::blas::sgemm_bt;
use rlx_models::weight_loader::GgufLoader;
use rlx_models::{Qwen35Config, Qwen35Weights};
use std::path::{Path, PathBuf};

const DEFAULT_NON_MTP_Q4: &str = "/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf";

fn gguf_path() -> Option<PathBuf> {
    if std::env::var("QWEN35_RUN_LLAMA_PARITY").ok().as_deref() != Some("1") {
        if std::env::var("QWEN35_GGUF_PATH")
            .ok()
            .filter(|s| !s.is_empty())
            .is_none()
        {
            return None;
        }
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

/// Tied LM head: logits[v] = dot(hidden, token_embd[v]).
fn tied_logits_sgemm(
    hidden: &[f32],
    token_embd: &[f32],
    n_vocab: usize,
    n_embd: usize,
) -> Vec<f32> {
    let mut logits = vec![0f32; n_vocab];
    // hidden [1, k], embed rows [n_vocab, k] stored row-major → B[n,k], C = A @ B^T.
    sgemm_bt(hidden, token_embd, &mut logits, 1, n_embd, n_vocab, 1.0);
    logits
}

fn top_k_from_logits(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter()
        .take(k)
        .map(|i| (i as u32, logits[i]))
        .collect()
}

fn load_token_embd(path: &Path) -> (Qwen35Config, Vec<f32>, bool) {
    let mut loader = GgufLoader::from_file(path.to_str().unwrap()).expect("gguf");
    loader.include_mtp(true);
    let cfg = Qwen35Config::from_gguf(loader.file()).expect("cfg");
    let weights = Qwen35Weights::from_loader(&mut loader, &cfg).expect("f32 weights");
    let has_output = weights.output.is_some();
    (cfg, weights.token_embd, has_output)
}

#[cfg(feature = "parity-llama")]
fn rlx_hidden(path: &Path, prompt_ids: &[u32]) -> Vec<f32> {
    use rlx_ir::DType;
    use rlx_models::qwen35::{build_qwen35_graph_sized_ext, last_token_indices, pack_input_ids};
    use rlx_runtime::{Device, Session};

    let max_seq = prompt_ids.len();
    let mut loader = GgufLoader::from_file(path.to_str().unwrap()).expect("gguf");
    loader.include_mtp(true);
    let cfg = Qwen35Config::from_gguf(loader.file()).expect("cfg");
    let weights = Qwen35Weights::from_loader_packed(&mut loader, &cfg).expect("packed weights");
    let (graph, params, packed) = build_qwen35_graph_sized_ext(
        &cfg, weights, 1, max_seq, true, true, false, false, None, false, false, true, false,
    )
    .expect("graph");
    let mut compiled = compile_support::compile_qwen35_prefill(Device::Cpu, graph, params.clone());

    for (param_name, (loader_key, _scheme, _shape)) in &packed {
        let bytes = loader
            .tensor_bytes_borrowed(loader_key)
            .unwrap_or_else(|| panic!("missing {loader_key}"));
        compiled.set_param_typed(param_name, bytes, DType::U8);
    }
    let padded = pack_input_ids(&[prompt_ids.to_vec()], max_seq).expect("pack");
    let last_idx = last_token_indices(&[prompt_ids.len()]);
    let outs = compiled.run(&[
        ("input_ids", padded.as_slice()),
        ("last_token_idx", last_idx.as_slice()),
    ]);
    outs[0].to_vec()
}

#[test]
#[cfg(feature = "parity-llama")]
fn qwen35_lm_head_isolates_trunk_vs_head_mismatch() {
    let path = match gguf_path() {
        Some(p) => p,
        None => {
            eprintln!("skip qwen35_lm_head_isolate: set QWEN35_RUN_LLAMA_PARITY=1");
            return;
        }
    };

    let prompt_ids = vec![1u32, 2, 3];
    let (cfg, token_embd, has_output) = load_token_embd(&path);
    assert!(
        !has_output,
        "0.8B checkpoint should use tied embeddings (no output.weight)"
    );
    let n_embd = cfg.hidden_size;
    let n_vocab = token_embd.len() / n_embd;

    let ref_hidden =
        rlx_models::qwen35::llama_reference::last_token_hidden(&path, &prompt_ids).expect("ref h");
    let rlx_hidden = rlx_hidden(&path, &prompt_ids);
    let ref_logits = rlx_models::qwen35::llama_reference::last_token_logits(&path, &prompt_ids)
        .expect("ref logits");

    let mut runner = Qwen35RunnerBuilder::default()
        .weights(&path)
        .max_seq(prompt_ids.len())
        .packed_weights(true)
        .last_logits_only(true)
        .build()
        .expect("runner");
    let rlx_logits = runner
        .predict_logits(&prompt_ids)
        .expect("rlx logits")
        .logits;

    let n = ref_logits.len().min(rlx_logits.len()).min(n_vocab);
    let ref_from_ref_h = tied_logits_sgemm(&ref_hidden, &token_embd, n, n_embd);
    let ref_from_rlx_h = tied_logits_sgemm(&rlx_hidden, &token_embd, n, n_embd);
    let rlx_from_ref_h = tied_logits_sgemm(&ref_hidden, &token_embd, n, n_embd);

    let cos_ref_h = cosine_similarity(&ref_hidden, &rlx_hidden);
    let cos_ref_logits_self = cosine_similarity(&ref_logits[..n], &ref_from_ref_h[..n]);
    let cos_rlx_logits_self = cosine_similarity(&rlx_logits[..n], &ref_from_rlx_h[..n]);
    let cos_ref_logits_from_rlx_h = cosine_similarity(&ref_logits[..n], &ref_from_rlx_h[..n]);
    let cos_rlx_logits_from_ref_h = cosine_similarity(&rlx_logits[..n], &rlx_from_ref_h[..n]);
    let cos_end_to_end = cosine_similarity(&ref_logits[..n], &rlx_logits[..n]);

    eprintln!("qwen35 lm-head isolate (n_vocab={n}, tied):");
    eprintln!("  hidden cosine (RLX vs llama):           {cos_ref_h:.6}");
    eprintln!("  llama logits vs sgemm(llama h, embed):  {cos_ref_logits_self:.6}");
    eprintln!("  RLX logits vs sgemm(RLX h, embed):     {cos_rlx_logits_self:.6}");
    eprintln!("  llama logits vs sgemm(RLX h, embed):   {cos_ref_logits_from_rlx_h:.6}");
    eprintln!("  RLX logits vs sgemm(llama h, embed):    {cos_rlx_logits_from_ref_h:.6}");
    eprintln!("  end-to-end logits cosine:                {cos_end_to_end:.6}");

    let ref_top = top_k_from_logits(&ref_logits, 16);
    let cross_top = top_k_from_logits(&rlx_from_ref_h, 16);
    eprintln!("  top-3 llama: {:?}", &ref_top[..3]);
    eprintln!("  top-3 sgemm(llama h): {:?}", &cross_top[..3]);

    assert!(
        cos_ref_logits_self >= 0.9995,
        "llama logits vs sgemm(llama h, F32 embed) ({cos_ref_logits_self:.6}); \
         llama likely uses Q6K tied matmul at runtime"
    );
    assert!(
        cos_rlx_logits_self >= 0.9999,
        "RLX logits != tied matmul on RLX hidden ({cos_rlx_logits_self:.6})"
    );
    assert!(
        (cos_rlx_logits_from_ref_h - cos_end_to_end).abs() <= 1e-4,
        "cross-hidden lm_head: rlx(sgemm(llama h))={cos_rlx_logits_from_ref_h:.6} \
         vs end-to-end={cos_end_to_end:.6}"
    );

    // If cross hidden→logits matches llama, trunk hidden mismatch explains the gap.
    let hidden_explains = cos_ref_logits_from_rlx_h;
    eprintln!("  => hidden-only mismatch to llama logits: {hidden_explains:.6}");
    assert!(
        hidden_explains >= cos_end_to_end - 1e-4,
        "unexpected: cross-hidden logits worse than end-to-end"
    );
}

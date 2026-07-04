//! Env-gated Gemma 3 270M logits vs llama.cpp (`parity-llama` feature).
//!
//! ```sh
//! RLX_GEMMA3_GGUF=/tmp/rlx-weights/gemma-3-270m.gguf \
//! RLX_GEMMA3_RUN_LLAMA_PARITY=1 \
//! cargo test -p rlx-gemma --features "apple-silicon parity-llama" \
//!   --test gemma3_270m_logits --release -- --test-threads=1 --nocapture
//! ```

use rlx_gemma::GemmaRunner;
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::PathBuf;

fn weights() -> Option<PathBuf> {
    std::env::var("RLX_GEMMA3_GGUF").ok().map(PathBuf::from)
}

/// HF `unsloth/gemma-3-270m-it` chat ids for
/// "What is two plus two? Answer in one short sentence."
const HF_CHAT_IDS: &[u32] = &[
    2, 105, 2364, 107, 3689, 563, 1156, 2915, 1156, 236881, 25685, 528, 886, 2822, 13315, 236761,
    106, 107, 105, 4368, 107,
];

fn greedy_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<u32> {
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked.truncate(k);
    ranked.into_iter().map(|(i, _)| i).collect()
}

#[cfg(feature = "parity-llama")]
fn logits_parity_stats(rlx: &[f32], llama: &[f32]) -> (f32, f32) {
    let n = rlx.len().min(llama.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let a = rlx[i];
        let b = llama[i];
        dot += (a as f64) * (b as f64);
        na += (a as f64) * (a as f64);
        nb += (b as f64) * (b as f64);
    }
    let cos = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        0.0
    };

    let mut focus = std::collections::HashSet::new();
    for t in top_k_indices(llama, 100) {
        focus.insert(t as usize);
    }
    for t in top_k_indices(rlx, 100) {
        focus.insert(t as usize);
    }
    let mut max_abs = 0f32;
    for i in focus {
        if i < n {
            max_abs = max_abs.max((rlx[i] - llama[i]).abs());
        }
    }
    (max_abs, cos)
}

#[cfg(feature = "parity-llama")]
#[test]
fn gemma3_270m_top1_matches_llama_on_hf_chat_prompt() {
    if std::env::var("RLX_GEMMA3_RUN_LLAMA_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_GEMMA3_RUN_LLAMA_PARITY=1");
        return;
    }
    let Some(weights) = weights() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };

    let mut runner = GemmaRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(Device::Cpu)
        .max_seq(512)
        .build()
        .expect("build");

    let rlx_logits = runner.predict_logits(HF_CHAT_IDS).expect("predict_logits");
    let llama_logits =
        rlx_gemma::llama_reference::last_token_logits(&weights, HF_CHAT_IDS).expect("llama logits");

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, val)| (i as u32, *val))
    };
    let (rlx_top, rlx_val) = argmax(&rlx_logits).unwrap();
    let (llama_top, llama_val) = argmax(&llama_logits).unwrap();
    eprintln!("rlx top1={rlx_top} ({rlx_val:.4}) llama top1={llama_top} ({llama_val:.4})");
    assert_eq!(
        rlx_top, llama_top,
        "Gemma 3 270M packed prefill top-1 must match llama.cpp on HF chat prompt"
    );
}

/// Incremental decode: prefill prompt, greedy step 0 from cache, step 1 from bucketed decode.
#[test]
fn gemma3_270m_incremental_decode_matches_one_shot_prefill() {
    let Some(weights) = weights() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };

    let mut runner = GemmaRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(Device::Cpu)
        .max_seq(512)
        .build()
        .expect("build");

    let prefill_logits = runner.predict_logits(HF_CHAT_IDS).expect("prefill");
    let step0 = greedy_argmax(&prefill_logits);
    assert_eq!(step0, 11634, "greedy step 0");

    let mut extended = HF_CHAT_IDS.to_vec();
    extended.push(step0);
    let one_shot_logits = runner
        .predict_logits(&extended)
        .expect("one-shot prefill on prompt+step0");
    let one_shot_step1 = greedy_argmax(&one_shot_logits);

    let mut runner2 = GemmaRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(Device::Cpu)
        .max_seq(512)
        .sample(SampleOpts::greedy())
        .build()
        .expect("build2");
    let incremental = runner2
        .generate(HF_CHAT_IDS, 2, |_| {})
        .expect("generate 2");
    assert_eq!(incremental[0], step0);
    assert_eq!(incremental[1], one_shot_step1);
}

#[cfg(feature = "parity-llama")]
#[test]
fn gemma3_270m_prefill_top32_and_logit_stats_match_llama() {
    if std::env::var("RLX_GEMMA3_RUN_LLAMA_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_GEMMA3_RUN_LLAMA_PARITY=1");
        return;
    }
    let Some(weights) = weights() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };

    let mut runner = GemmaRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(Device::Cpu)
        .max_seq(512)
        .build()
        .expect("build");

    let rlx = runner.predict_logits(HF_CHAT_IDS).expect("rlx logits");
    let llama =
        rlx_gemma::llama_reference::last_token_logits(&weights, HF_CHAT_IDS).expect("llama logits");
    assert_eq!(rlx.len(), llama.len());

    let rlx_top32 = top_k_indices(&rlx, 32);
    let llama_top32 = top_k_indices(&llama, 32);
    eprintln!("rlx top32   = {rlx_top32:?}");
    eprintln!("llama top32 = {llama_top32:?}");

    let rlx_top2 = top_k_indices(&rlx, 2);
    let llama_top2 = top_k_indices(&llama, 2);
    assert_eq!(
        rlx_top2, llama_top2,
        "top-2 greedy ranking must match llama.cpp"
    );

    let rlx_set: std::collections::HashSet<_> = rlx_top32.iter().copied().collect();
    let overlap = llama_top32
        .iter()
        .filter(|t| rlx_set.contains(t))
        .count();
    assert!(
        overlap >= 28,
        "top-32 token sets should largely agree (overlap={overlap}/32)"
    );

    let (max_abs, cos) = logits_parity_stats(&rlx, &llama);
    eprintln!("logit parity: top-100 max_abs={max_abs:.4} full-vocab cosine={cos:.6}");
    assert!(
        max_abs < 2.0,
        "top-100 logit max_abs vs llama ({max_abs:.4})"
    );
    assert!(cos > 0.99, "full-vocab logit cosine vs llama ({cos:.6})");
}

#[cfg(feature = "parity-llama")]
#[test]
fn gemma3_270m_hidden_cosine_matches_llama() {
    if std::env::var("RLX_GEMMA3_RUN_LLAMA_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_GEMMA3_RUN_LLAMA_PARITY=1");
        return;
    }
    let Some(weights) = weights() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };

    let mut runner = GemmaRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(Device::Cpu)
        .max_seq(512)
        .build()
        .expect("build");

    let rlx_h = runner
        .predict_last_hidden(HF_CHAT_IDS)
        .expect("rlx hidden");
    let llama_h =
        rlx_gemma::llama_reference::last_token_hidden(&weights, HF_CHAT_IDS).expect("llama hidden");
    assert_eq!(rlx_h.len(), llama_h.len());

    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut l2 = 0f64;
    for (a, b) in rlx_h.iter().zip(llama_h.iter()) {
        let x = *a as f64;
        let y = *b as f64;
        l2 += (x - y).powi(2);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let cos = (dot / (na.sqrt() * nb.sqrt())) as f32;
    eprintln!("hidden parity: L2={:.3e} cosine={cos:.6}", l2.sqrt());
    assert!(cos > 0.995, "last-token hidden cosine vs llama ({cos:.6})");
}

#[cfg(feature = "parity-llama")]
#[test]
fn gemma3_270m_greedy_decode_matches_llama_short_and_long() {
    if std::env::var("RLX_GEMMA3_RUN_LLAMA_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_GEMMA3_RUN_LLAMA_PARITY=1");
        return;
    }
    let Some(weights) = weights() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };

    let greedy_steps = 16usize;
    let mut runner = GemmaRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(Device::Cpu)
        .max_seq(512)
        .sample(SampleOpts::greedy())
        .build()
        .expect("build");
    let rlx = runner
        .generate(HF_CHAT_IDS, greedy_steps, |_| {})
        .expect("rlx greedy");

    let llama = rlx_gemma::llama_reference::greedy_generation_ids(
        &weights,
        HF_CHAT_IDS,
        greedy_steps as u32,
        512,
    )
    .expect("llama greedy");

    eprintln!("rlx greedy  = {rlx:?}");
    eprintln!("llama greedy = {llama:?}");
    assert_eq!(
        rlx, llama,
        "packed greedy decode must match llama.cpp for {greedy_steps} steps (EOG-aware)"
    );
}

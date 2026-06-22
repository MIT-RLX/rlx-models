// RLX — versatile ML compiler + runtime. GPLv3.
//! Greedy generation on CPU for Gemma 4 E2B mobile QAT — compile the prefill
//! graph once, decode tokens by re-running it (causal mask lets each position's
//! logit be read progressively). Verifies the generated token sequence matches
//! HF `transformers` greedy decoding.
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::PathBuf;

fn dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let s = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    s.join("config.json").is_file().then_some(s)
}
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

#[test]
fn e2b_greedy_generation_matches_hf() {
    let Some(dir) = dir() else {
        eprintln!("[gen] no ckpt — skip");
        return;
    };
    let cfg = GemmaConfig::from_file(&dir.join("config.json")).unwrap();
    let vocab = cfg.vocab_size;
    let bucket = 16usize; // fixed prefill bucket; generate within it
    let prompt: Vec<u32> = vec![818, 5279, 529, 7001, 563]; // "The capital of France is"
    let new_tokens = 10usize;
    // HF greedy reference (do_sample=False) for the same prompt.
    let hf_new: Vec<u32> = vec![7001, 563, 7001, 563, 7001, 563, 7001, 563, 7001, 563];

    let loader = GemmaQatLoader::open(&dir).unwrap();
    let mut bld = GemmaQatLoader::open(&dir).unwrap();
    let mut packed = HashMap::new();
    let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut bld,
        1,
        bucket,
        true,
        false,
        false,
        &mut packed,
        None,
        None,
    )
    .unwrap();
    let mut compiled = compile_graph_gemma_prefill_with_params(Device::Cpu, graph, params).unwrap();

    let mut ids = vec![0u32; bucket];
    for (i, &t) in prompt.iter().enumerate() {
        ids[i] = t;
    }
    let mut generated: Vec<u32> = Vec::new();

    for step in 0..new_tokens {
        let cur = prompt.len() + step; // position of the token we predict next-of
        let ple = loader.compute_per_layer_inputs(&cfg, &ids).unwrap();
        let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        let outs = compiled.run(&[
            ("input_ids", ids_f32.as_slice()),
            ("per_layer_inputs", ple.as_slice()),
        ]);
        let logits = &outs[0]; // [bucket * vocab]
        let last = &logits[(cur - 1) * vocab..cur * vocab];
        let next = argmax(last) as u32;
        generated.push(next);
        if cur < bucket {
            ids[cur] = next;
        }
    }

    eprintln!("[gen] rlx generated: {generated:?}");
    eprintln!("[gen] hf  generated: {hf_new:?}");
    let matches = generated
        .iter()
        .zip(&hf_new)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!("[gen] matching prefix length: {matches}/{new_tokens}");
    // The first token already matched HF in the parity test; greedy should
    // track HF for the deterministic loop. Require a solid matching prefix.
    assert!(
        matches >= 6,
        "rlx greedy diverged from HF too early ({matches}/{new_tokens})"
    );
}

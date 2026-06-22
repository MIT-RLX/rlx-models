// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// GPLv3 — see repository LICENSE.

//! End-to-end CPU parity for the Gemma 4 E2B mobile QAT text model.
//!
//! Loads the real checkpoint through `GemmaQatLoader`, precomputes the
//! Per-Layer-Embedding inputs, builds + compiles the prefill graph on CPU,
//! runs it for a fixed prompt and compares the last-token logits to the HF
//! `transformers` reference in `fixtures/gemma4_e2b/last_logits.bin`.
//!
//! Skips if the checkpoint or fixtures are absent. Regenerate fixtures with
//! `.venv-gemma-ref/bin/python scripts/gemma4_e2b_dump.py --reference --hidden-states`.

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;

fn fixture_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("RLX_GEMMA4_E2B_DIR") {
        let p = PathBuf::from(d);
        return p.join("config.json").is_file().then_some(p);
    }
    let home = std::env::var_os("HOME")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/\
         models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("config.json").is_file().then_some(snap)
}

fn read_f32_bin(p: &std::path::Path) -> Option<Vec<f32>> {
    let raw = std::fs::read(p).ok()?;
    Some(
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-12)) as f32
}

#[test]
fn gemma4_e2b_cpu_logits_match_hf() {
    let Some(dir) = fixture_dir() else {
        eprintln!("[e2b cpu parity] checkpoint not found — skipping");
        return;
    };
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fx = manifest.join("../../fixtures/gemma4_e2b");
    // Our F32 forward skips activation SRQ, so compare against the no-SRQ HF
    // reference (apples-to-apples). The with-SRQ reference predicts the same
    // next token but its final-softcapped logit distribution differs.
    let fx_nosrq = manifest.join("../../fixtures/gemma4_e2b_nosrq");
    let Some(hf_last) = read_f32_bin(&fx_nosrq.join("last_logits.bin"))
        .or_else(|| read_f32_bin(&fx.join("last_logits.bin")))
    else {
        eprintln!("[e2b cpu parity] last_logits.bin absent — skipping");
        return;
    };

    let cfg = GemmaConfig::from_file(&dir.join("config.json")).expect("config");
    let ids: Vec<u32> = vec![818, 5279, 529, 7001, 563]; // "The capital of France is"
    let seq = ids.len();

    let loader_ple = GemmaQatLoader::open(&dir).expect("open loader (ple)");
    let ple = loader_ple
        .compute_per_layer_inputs(&cfg, &ids)
        .expect("per_layer_inputs");

    let mut loader = GemmaQatLoader::open(&dir).expect("open loader");
    let mut packed = HashMap::new();
    let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut loader,
        1,
        seq,
        true,
        false,
        false,
        &mut packed,
        None,
        None,
    )
    .expect("build graph");

    let mut compiled =
        compile_graph_gemma_prefill_with_params(Device::Cpu, graph, params).expect("compile cpu");

    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let outs = compiled.run(&[
        ("input_ids", ids_f32.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
    ]);
    let logits = &outs[0];
    let vocab = cfg.vocab_size;
    assert_eq!(logits.len(), seq * vocab, "logits shape");
    let last = &logits[(seq - 1) * vocab..seq * vocab];

    assert!(last.iter().all(|v| v.is_finite()), "non-finite logits");
    let my_tok = argmax(last);
    let hf_tok = argmax(&hf_last);
    let cos = cosine(last, &hf_last);
    let maxd = last
        .iter()
        .zip(&hf_last)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[e2b cpu parity] argmax rlx={my_tok} hf={hf_tok} | cosine={cos:.5} | maxdiff={maxd:.4}"
    );
    eprintln!(
        "[e2b cpu parity] rlx top logit={:.3} hf top logit={:.3}",
        last[my_tok], hf_last[hf_tok]
    );

    // Activation SRQ is not yet applied in the F32 path, so exact match isn't
    // expected; the next-token prediction and overall direction must agree.
    assert_eq!(my_tok, hf_tok, "next-token argmax differs from HF");
    assert!(cos > 0.95, "logit cosine {cos} too low vs HF");
}

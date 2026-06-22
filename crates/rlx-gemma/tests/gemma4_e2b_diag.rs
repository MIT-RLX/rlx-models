// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// GPLv3 — see repository LICENSE.
//! Layer-wise bisection of the E2B CPU forward vs HF hidden states.

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;

fn fixture_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("config.json").is_file().then_some(snap)
}
fn rd(p: &std::path::Path) -> Option<Vec<f32>> {
    let raw = std::fs::read(p).ok()?;
    Some(
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}
fn cmp(tag: &str, a: &[f32], b: &[f32]) {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb + 1e-12);
    let maxd = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[diag] {tag:24} cos={cos:.5} maxdiff={maxd:.4} | rlx[:4]={:?} hf[:4]={:?}",
        &a[..4.min(a.len())],
        &b[..4.min(b.len())]
    );
}

#[test]
fn e2b_layerwise_bisect() {
    let Some(dir) = fixture_dir() else {
        eprintln!("[diag] no ckpt");
        return;
    };
    let fx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_e2b");
    let cfg = GemmaConfig::from_file(&dir.join("config.json")).unwrap();
    let ids: Vec<u32> = vec![818, 5279, 529, 7001, 563];
    let seq = ids.len();
    let h = cfg.hidden_size;

    let lp = GemmaQatLoader::open(&dir).unwrap();
    let ple = lp.compute_per_layer_inputs(&cfg, &ids).unwrap();

    unsafe { std::env::set_var("RLX_TAP_L0", "1") };
    let mut loader = GemmaQatLoader::open(&dir).unwrap();
    let mut packed = HashMap::new();
    let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut loader,
        1,
        seq,
        /*lm_head*/ false,
        false,
        false,
        &mut packed,
        None,
        None,
    )
    .unwrap();
    let mut compiled = compile_graph_gemma_prefill_with_params(Device::Cpu, graph, params).unwrap();
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let outs = compiled.run(&[
        ("input_ids", ids_f32.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
    ]);
    eprintln!("[diag] {} outputs", outs.len());

    // outs[0] = post-model.norm hidden; outs[1]=tap1 embed; outs[16]=tap11 layer0-final.
    let tok0 = |t: &[f32]| t[0..h].to_vec(); // first token row

    let g = |name: &str| rd(&fx.join(name));
    let qd = cfg.num_attention_heads * cfg.layer_head_dim(0); // 2048
    let kvd = cfg.layer_num_kv_heads(0) * cfg.layer_head_dim(0); // 256
    let t = |o: &[f32], dim: usize| o[0..dim].to_vec(); // token 0
    if let Some(x) = g("hidden_0.bin") {
        cmp("embed(tap1)", &tok0(&outs[1]), &tok0(&x));
    }
    if let Some(x) = g("hf_input_ln.bin") {
        cmp("input_ln(tap2)", &tok0(&outs[2]), &tok0(&x));
    }
    if let Some(x) = g("hf_q.bin") {
        cmp("Q postnorm(tap3)", &t(&outs[8], qd), &t(&x, qd));
    }
    if let Some(x) = g("hf_k.bin") {
        cmp("K postnorm(tap4)", &t(&outs[9], kvd), &t(&x, kvd));
    }
    if let Some(x) = g("hf_v.bin") {
        cmp("V postnorm(tap5)", &t(&outs[10], kvd), &t(&x, kvd));
    }
    if let Some(x) = g("hf_attn_preo.bin") {
        cmp("attn preo(tap8)", &t(&outs[13], qd), &t(&x, qd));
    }
    if let Some(x) = g("hf_attn_postnorm.bin") {
        cmp("attn_postnorm(tap9)", &tok0(&outs[14]), &tok0(&x));
    }
    if let Some(x) = g("hidden_after_l0.bin") {
        cmp("layer0-final(tap11)", &tok0(&outs[16]), &tok0(&x));
    }
}

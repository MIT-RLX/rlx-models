// RLX — versatile ML compiler + runtime. GPLv3.
//! Tap layer 34 internals vs HF (no-SRQ) to find the last-layer bug.
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
fn rd(p: &std::path::Path) -> Option<Vec<f32>> {
    let r = std::fs::read(p).ok()?;
    Some(
        r.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}
fn cmp(tag: &str, a: &[f32], b: &[f32]) {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let md = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "[l34] {tag:18} cos={:.4} maxdiff={md:.4} rlx[:4]={:?} hf[:4]={:?}",
        d / (na * nb + 1e-12),
        &a[..4.min(a.len())],
        &b[..4.min(b.len())]
    );
}

#[test]
fn e2b_layer34_taps() {
    let Some(d) = dir() else {
        eprintln!("[l34] no ckpt");
        return;
    };
    let fx =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_e2b_nosrq");
    let cfg = GemmaConfig::from_file(&d.join("config.json")).unwrap();
    let ids: Vec<u32> = vec![818, 5279, 529, 7001, 563];
    let lp = GemmaQatLoader::open(&d).unwrap();
    let ple = lp.compute_per_layer_inputs(&cfg, &ids).unwrap();
    unsafe {
        std::env::set_var("RLX_TAP_L0", "1");
        std::env::set_var("RLX_TAP_LAYER", "34");
    }
    let mut loader = GemmaQatLoader::open(&d).unwrap();
    let mut packed = HashMap::new();
    let (g, p) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut loader,
        1,
        ids.len(),
        false,
        false,
        false,
        &mut packed,
        None,
        None,
    )
    .unwrap();
    let mut c = compile_graph_gemma_prefill_with_params(Device::Cpu, g, p).unwrap();
    let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let outs = c.run(&[
        ("input_ids", idf.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
    ]);
    eprintln!("[l34] {} outputs", outs.len());
    let qd = cfg.num_attention_heads * cfg.layer_head_dim(34); // 4096
    let h = cfg.hidden_size;
    let t = |o: &[f32], dim: usize| o[0..dim].to_vec();
    let add = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
    let _scale = |a: &[f32], s: f32| -> Vec<f32> { a.iter().map(|x| x * s).collect() };
    let hin = rd(&fx.join("hidden_34.bin")).map(|v| t(&v, h)); // layer-34 input
    let paln = rd(&fx.join("l34_paln.bin")).map(|v| t(&v, h));
    let pffn = rd(&fx.join("l34_pffn.bin")).map(|v| t(&v, h));
    let plec = rd(&fx.join("l34_ple.bin")).map(|v| t(&v, h));
    if let Some(x) = rd(&fx.join("l34_inln.bin")) {
        cmp("input_ln(tap2)", &t(&outs[2], h), &t(&x, h));
    }
    if let Some(x) = rd(&fx.join("l34_q.bin")) {
        cmp("Q postnorm(tap3)", &t(&outs[8], qd), &t(&x, qd));
    }
    if let Some(x) = rd(&fx.join("l34_attn_preo.bin")) {
        cmp("attn preo(tap8)", &t(&outs[13], qd), &t(&x, qd));
    }
    if let (Some(p),) = (paln.clone(),) {
        cmp("attn postnorm(tap9)", &t(&outs[14], h), &p);
    }
    // res1 = input + attn_postnorm  (tap10 = outs[15])
    if let (Some(hi), Some(p)) = (hin.clone(), paln.clone()) {
        cmp("res1(tap10)", &t(&outs[15], h), &add(&hi, &p));
    }
    // res2 = res1 + post_ffn_norm  (tap12 = outs[16])
    if let (Some(hi), Some(p), Some(pf)) = (hin.clone(), paln.clone(), pffn) {
        let r2 = add(&add(&hi, &p), &pf);
        cmp("res2/FFN(tap12)", &t(&outs[16], h), &r2);
    }
    // post-PLE = res2 + ple_contrib (tap13 = outs[17])
    if let (Some(hi), Some(p), Some(pl)) = (hin.clone(), paln, plec) {
        // recompute res2 again (cheap)
        if let Some(pf) = rd(&fx.join("l34_pffn.bin")).map(|v| t(&v, h)) {
            let r2 = add(&add(&hi, &p), &pf);
            cmp("post-PLE(tap13)", &t(&outs[17], h), &add(&r2, &pl));
        }
    }
    // final = post-PLE * layer_scalar[34]=0.26171875 (tap11 = outs[18]); also vs hidden_35
    if let Some(x) = rd(&fx.join("hidden_35.bin")) {
        cmp("layer34 final(tap18)", &t(&outs[18], h), &t(&x, h));
    }
}

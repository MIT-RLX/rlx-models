// RLX — versatile ML compiler + runtime. GPLv3.
//! Post-final-norm per-token parity vs HF no-SRQ (hidden_states[35] = lm_head input).
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::PathBuf;

fn dir() -> Option<PathBuf> {
    let h = std::env::var_os("HOME")?;
    let b = std::path::Path::new(&h).join(
        ".cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let s = std::fs::read_dir(&b).ok()?.flatten().next()?.path();
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
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    d / (na * nb + 1e-12)
}

#[test]
fn e2b_postnorm_per_token() {
    let Some(d) = dir() else {
        eprintln!("no ckpt");
        return;
    };
    let fx =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_e2b_nosrq");
    let cfg = GemmaConfig::from_file(&d.join("config.json")).unwrap();
    let ids: Vec<u32> = vec![818, 5279, 529, 7001, 563];
    let h = cfg.hidden_size;
    let lp = GemmaQatLoader::open(&d).unwrap();
    let ple = lp.compute_per_layer_inputs(&cfg, &ids).unwrap();
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
    let pn = &outs[0]; // post-final-norm [seq, h]
    // HF no-SRQ hidden_states[35] = post-final-norm.
    let hf = rd(&fx.join("hidden_35.bin")).expect("hidden_35");
    for ti in 0..ids.len() {
        let r = &pn[ti * h..(ti + 1) * h];
        let f = &hf[ti * h..(ti + 1) * h];
        eprintln!("[pn] token {ti}  cos={:.5}", cos(r, f));
    }
}

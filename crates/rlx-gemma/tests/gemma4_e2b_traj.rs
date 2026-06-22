// RLX — versatile ML compiler + runtime. GPLv3.
//! Per-layer hidden-state trajectory vs HF (RLX_TAP_ALL), to find where the
//! E2B CPU forward starts diverging.
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
fn cos(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (d / (na * nb + 1e-12)) as f32
}

#[test]
fn e2b_trajectory() {
    let Some(d) = dir() else {
        eprintln!("[traj] no ckpt");
        return;
    };
    let sub = std::env::var("RLX_E2B_REF").unwrap_or_else(|_| "gemma4_e2b".into());
    let fx = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("fixtures")
        .join(&sub);
    eprintln!("[traj] reference dir: {sub}");
    let cfg = GemmaConfig::from_file(&d.join("config.json")).unwrap();
    let ids: Vec<u32> = vec![818, 5279, 529, 7001, 563];
    let h = cfg.hidden_size;
    let nl = cfg.num_hidden_layers;
    let lp = GemmaQatLoader::open(&d).unwrap();
    let ple = lp.compute_per_layer_inputs(&cfg, &ids).unwrap();
    unsafe { std::env::set_var("RLX_TAP_ALL", "1") };
    let mut loader = GemmaQatLoader::open(&d).unwrap();
    let mut packed = HashMap::new();
    let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
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
    let mut c = compile_graph_gemma_prefill_with_params(Device::Cpu, graph, params).unwrap();
    let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let outs = c.run(&[
        ("input_ids", idf.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
    ]);
    eprintln!("[traj] {} outputs (expect 1 + {nl})", outs.len());
    // outs[0]=post-norm; outs[1+i] = layer i output ↔ HF hidden_{i+1}
    for i in 0..nl {
        if let Some(hf) = rd(&fx.join(format!("hidden_{}.bin", i + 1))) {
            let r = &outs[1 + i][0..h];
            let full = cos(&outs[1 + i][0..ids.len() * h], &hf);
            let fa = cfg.is_full_attention_layer(i);
            let sh = cfg.is_kv_shared_layer(i);
            eprintln!(
                "[traj] layer {i:2} {}{}  cos(tok0)={:.4} cos(all)={:.4}",
                if fa { "FULL " } else { "slide" },
                if sh { " SHARED" } else { "       " },
                cos(r, &hf[0..h]),
                full
            );
        }
    }
}

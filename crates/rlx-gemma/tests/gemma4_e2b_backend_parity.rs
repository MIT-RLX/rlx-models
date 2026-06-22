// RLX — GPLv3. E2B QAT backend parity: resolved Metal→MLX matches CPU hidden states.

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_gemma::config::GemmaConfig;
use rlx_gemma::gemma_e2b::{compile_e2b_prefill, resolve_e2b_device};
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;

fn dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/\
         models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("config.json").is_file().then_some(snap)
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn hidden_last_token(device: Device, dir: &std::path::Path, cfg: &GemmaConfig) -> Option<Vec<f32>> {
    let ids: Vec<u32> = vec![818, 5279, 529, 7001, 563];
    let loader = GemmaQatLoader::open(dir).ok()?;
    let ple = loader.compute_per_layer_inputs(cfg, &ids).ok()?;
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let mut bld = GemmaQatLoader::open(dir).ok()?;
    let mut packed = HashMap::new();
    let (g, p) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        cfg,
        &mut bld,
        1,
        ids.len(),
        false,
        false,
        false,
        &mut packed,
        None,
        None,
    )
    .ok()?;
    let _exec = resolve_e2b_device(device);
    let mut c = compile_e2b_prefill(device, g, p).ok()?;
    let out = c
        .run(&[
            ("input_ids", ids_f32.as_slice()),
            ("per_layer_inputs", ple.as_slice()),
        ])
        .into_iter()
        .next()?;
    let h = cfg.hidden_size;
    let last = ids.len() - 1;
    Some(out[last * h..(last + 1) * h].to_vec())
}

#[test]
fn e2b_resolved_metal_matches_cpu_hidden() {
    let Some(d) = dir() else {
        eprintln!("[e2b backend parity] no checkpoint — skip");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("[e2b backend parity] Metal unavailable — skip");
        return;
    }
    let cfg = GemmaConfig::from_file(&d.join("config.json")).expect("config");
    let cpu = hidden_last_token(Device::Cpu, &d, &cfg).expect("cpu forward");
    let resolved = hidden_last_token(Device::Metal, &d, &cfg).expect("resolved forward");
    let exec = resolve_e2b_device(Device::Metal);
    let drift = l2(&cpu, &resolved);
    eprintln!("[e2b backend parity] exec={exec:?} hidden L2 vs CPU = {drift:.4e}");
    assert!(
        drift < 0.02,
        "resolved E2B device {exec:?} hidden drift {drift} too high vs CPU"
    );
}

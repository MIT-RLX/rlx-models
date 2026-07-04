// RLX — GPLv3. Vision-tower encoder numeric parity vs HF no-SRQ.
//!
//! Feeds the *same* synthetic patches + host-precomputed pos-embed that the HF
//! `Gemma4VisionModel.encoder` saw (fixtures/gemma4_vision/*), runs the rlx
//! `build_vision_encoder` graph on CPU, and compares the 16-layer encoder
//! `last_hidden_state` per-patch (cosine + max abs diff). The 2-D RoPE cos/sin
//! are recomputed on the host via `vision_rope_tables` (must match HF's
//! per-axis `cat(freqs,freqs)`).
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::gemma4_vision::{
    VisionConfig, build_vision_encoder, build_vision_features, vision_rope_tables,
};
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::{Device, is_available};
use std::path::{Path, PathBuf};

fn dir() -> Option<PathBuf> {
    let h = std::env::var_os("HOME")?;
    let b = Path::new(&h).join(
        ".cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let s = std::fs::read_dir(&b).ok()?.flatten().next()?.path();
    s.join("config.json").is_file().then_some(s)
}
fn rd(p: &Path) -> Option<Vec<f32>> {
    let r = std::fs::read(p).ok()?;
    Some(
        r.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}
fn rd_i32(p: &Path) -> Option<Vec<i32>> {
    let r = std::fs::read(p).ok()?;
    Some(
        r.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
fn vision_encoder_parity() {
    let Some(d) = dir() else {
        eprintln!("[vision parity] no ckpt — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_vision");
    let Some(pixels) = rd(&fx.join("pixels.bin")) else {
        eprintln!("[vision parity] no fixtures — skip");
        return;
    };
    let pos_embed = rd(&fx.join("pos_embed.bin")).expect("pos_embed");
    let hf = rd(&fx.join("encoder_out.bin")).expect("encoder_out");
    let posi = rd_i32(&fx.join("positions.bin")).expect("positions");

    let cfg = VisionConfig::default();
    let p = posi.len() / 2;
    let positions: Vec<(u32, u32)> = (0..p)
        .map(|i| (posi[2 * i] as u32, posi[2 * i + 1] as u32))
        .collect();
    let (rcos, rsin) = vision_rope_tables(&cfg, &positions);

    let mut loader = GemmaQatLoader::open(&d).unwrap();
    let (g, params) = build_vision_encoder(&cfg, &mut loader, 1, p).expect("build vision encoder");
    let mut c = compile_graph_gemma_prefill_with_params(Device::Cpu, g, params).unwrap();
    let outs = c.run(&[
        ("vision_pixels", pixels.as_slice()),
        ("vision_pos_embed", pos_embed.as_slice()),
        ("vision_rope_cos", rcos.as_slice()),
        ("vision_rope_sin", rsin.as_slice()),
    ]);
    let out = &outs[0]; // [P, 768]
    let h = cfg.hidden;
    let mut worst = 1.0f64;
    let mut maxdiff = 0.0f32;
    for pi in 0..p {
        let r = &out[pi * h..(pi + 1) * h];
        let f = &hf[pi * h..(pi + 1) * h];
        let cv = cos(r, f);
        if cv < worst {
            worst = cv;
        }
        for (a, b) in r.iter().zip(f) {
            maxdiff = maxdiff.max((a - b).abs());
        }
        if pi < 3 || cv < 0.999 {
            eprintln!("[vision parity] patch {pi}  cos={cv:.6}");
        }
    }
    eprintln!("[vision parity] worst cos over {p} patches = {worst:.6}, maxabs = {maxdiff:.4}");
    assert!(worst > 0.99, "vision encoder diverges: worst cos {worst}");
}

/// Localize the wgpu vision-encoder garbage: run N layers on CPU vs wgpu
/// (N from RLX_VIS_LAYERS, default 1) and report cos + wgpu magnitude. If N=1
/// already diverges, the bug is a layer-0 op (norm / 2-D RoPE / attention / mm).
#[test]
fn vision_wgpu_bisect() {
    let Some(d) = dir() else {
        eprintln!("[vis bisect] no ckpt — skip");
        return;
    };
    if !is_available(Device::Gpu) {
        eprintln!("[vis bisect] no wgpu — skip");
        return;
    }
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_vision");
    let Some(pixels) = rd(&fx.join("pixels.bin")) else {
        eprintln!("[vis bisect] no fixtures — skip");
        return;
    };
    let pos_embed = rd(&fx.join("pos_embed.bin")).expect("pos_embed");
    let posi = rd_i32(&fx.join("positions.bin")).expect("positions");
    let mut cfg = VisionConfig::default();
    cfg.layers = std::env::var("RLX_VIS_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let p = posi.len() / 2;
    let positions: Vec<(u32, u32)> = (0..p)
        .map(|i| (posi[2 * i] as u32, posi[2 * i + 1] as u32))
        .collect();
    let (rcos, rsin) = vision_rope_tables(&cfg, &positions);
    let run = |dev: Device| -> Vec<f32> {
        let mut l = GemmaQatLoader::open(&d).unwrap();
        let (g, params) = build_vision_encoder(&cfg, &mut l, 1, p).unwrap();
        let mut c = compile_graph_gemma_prefill_with_params(dev, g, params).unwrap();
        c.run(&[
            ("vision_pixels", pixels.as_slice()),
            ("vision_pos_embed", pos_embed.as_slice()),
            ("vision_rope_cos", rcos.as_slice()),
            ("vision_rope_sin", rsin.as_slice()),
        ])[0]
            .clone()
    };
    let oc = run(Device::Cpu);
    let og = run(Device::Gpu);
    let maxg = og.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let maxc = oc.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let n = oc.len().min(og.len());
    let gcos = cos(&oc[..n], &og[..n]); // global cos, robust to any tap width
    let tap = std::env::var("RLX_VIS_TAP").unwrap_or_else(|_| "full".into());
    eprintln!(
        "[vis bisect L={} tap={tap}] cpu-vs-wgpu cos = {gcos:.6}, cpu maxabs = {maxc:.4}, wgpu maxabs = {maxg:.4}, len {}/{}",
        cfg.layers, oc.len(), og.len()
    );
}

#[test]
fn vision_features_parity() {
    run_vision_features(Device::Cpu, "vision feats");
}

#[test]
fn vision_features_parity_metal() {
    if !is_available(Device::Metal) {
        eprintln!("[vision feats metal] no Metal — skip");
        return;
    }
    run_vision_features(Device::Metal, "vision feats metal");
}

#[test]
fn vision_features_parity_mlx() {
    if !is_available(Device::Mlx) {
        eprintln!("[vision feats mlx] no MLX — skip");
        return;
    }
    run_vision_features(Device::Mlx, "vision feats mlx");
}

#[test]
fn vision_features_parity_wgpu() {
    if !is_available(Device::Gpu) {
        eprintln!("[vision feats wgpu] no wgpu — skip");
        return;
    }
    run_vision_features(Device::Gpu, "vision feats wgpu");
}

#[test]
fn vision_features_parity_coreml() {
    if !is_available(Device::Ane) {
        eprintln!("[vision feats coreml] no CoreML/ANE — skip");
        return;
    }
    run_vision_features(Device::Ane, "vision feats coreml");
}

fn run_vision_features(dev: Device, tag: &str) {
    let Some(d) = dir() else {
        eprintln!("[{tag}] no ckpt — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_vision");
    let Some(pixels) = rd(&fx.join("feat_pixels.bin")) else {
        eprintln!("[{tag}] no fixtures — skip");
        return;
    };
    let pos_embed = rd(&fx.join("feat_pos_embed.bin")).expect("feat_pos_embed");
    let hf = rd(&fx.join("feat_out.bin")).expect("feat_out");
    let posi = rd_i32(&fx.join("feat_positions.bin")).expect("feat_positions");

    let cfg = VisionConfig::default();
    let p = posi.len() / 2;
    let positions: Vec<(u32, u32)> = (0..p)
        .map(|i| (posi[2 * i] as u32, posi[2 * i + 1] as u32))
        .collect();
    let (rcos, rsin) = vision_rope_tables(&cfg, &positions);

    let mut loader = GemmaQatLoader::open(&d).unwrap();
    let (g, params) =
        build_vision_features(&cfg, &mut loader, &positions, 3).expect("build vision features");
    let mut c = compile_graph_gemma_prefill_with_params(dev, g, params).unwrap();
    let outs = c.run(&[
        ("vision_pixels", pixels.as_slice()),
        ("vision_pos_embed", pos_embed.as_slice()),
        ("vision_rope_cos", rcos.as_slice()),
        ("vision_rope_sin", rsin.as_slice()),
    ]);
    let out = &outs[0]; // [L, 1536]
    let lm = cfg.lm_hidden;
    let l = hf.len() / lm;
    let mut worst = 1.0f64;
    let mut maxdiff = 0.0f32;
    for li in 0..l {
        let r = &out[li * lm..(li + 1) * lm];
        let f = &hf[li * lm..(li + 1) * lm];
        let cv = cos(r, f);
        if cv < worst {
            worst = cv;
        }
        for (a, b) in r.iter().zip(f) {
            maxdiff = maxdiff.max((a - b).abs());
        }
        eprintln!("[{tag}] soft token {li}  cos={cv:.6}");
    }
    eprintln!("[{tag}] {l} soft tokens, worst cos = {worst:.6}, maxabs = {maxdiff:.5}");
    assert!(
        worst > 0.99,
        "vision features diverge ({tag}): worst cos {worst}"
    );
}

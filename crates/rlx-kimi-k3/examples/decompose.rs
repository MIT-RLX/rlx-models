//! `decompose` — can we make the Kimi-K3 backbone matmuls cheaper by (1) LOW-RANK
//! decomposition, (2) SKIP/prune (sparsity), or (3) TERNARY ({-1,0,+1})? Measures,
//! per real weight: the **stable rank** `‖W‖²_F / σ₁²` (effective rank — a full-rank
//! matrix has stable rank ≈ a large fraction of min(K,N); a genuinely low-rank one
//! is tiny → low-rank r(K+N) < KN flops), density (fraction non-zero → prune
//! headroom), and the reconstruction error of ternary vs int4 vs int8 (can we go
//! below int4?). NOTE: MLA (q/kv_lora) + LatentMoE are ALREADY low-rank by design,
//! so this probes the remaining dense KDA projections.
//!
//!   cargo run -p rlx-kimi-k3 --features cluster --example decompose [-- model_dir]

use rayon::prelude::*;
use rlx_ir::Philox4x32;
use rlx_kimi_k3::common::{WeightQuant, fake_quant_weight};
use rlx_kimi_k3::config::KimiK3Config;
use rlx_kimi_k3::kda::{KdaDims, KdaWeights};
use rlx_kimi_k3::loader::CheckpointLoader;
use std::path::Path;

/// W[k,n] @ v[n] -> [k]
fn matvec(w: &[f32], k: usize, n: usize, v: &[f32]) -> Vec<f32> {
    (0..k)
        .into_par_iter()
        .map(|i| {
            let row = &w[i * n..i * n + n];
            row.iter().zip(v).map(|(a, b)| a * b).sum::<f32>()
        })
        .collect()
}
/// Wᵀ[n,k] @ u[k] -> [n]  (cache-friendly row accumulate)
fn matvec_t(w: &[f32], k: usize, n: usize, u: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; n];
    for i in 0..k {
        let ui = u[i];
        let row = &w[i * n..i * n + n];
        for (o, a) in out.iter_mut().zip(row) {
            *o += a * ui;
        }
    }
    out
}
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Top singular value via power iteration on WᵀW.
fn sigma1(w: &[f32], k: usize, n: usize) -> f32 {
    let mut rng = Philox4x32::new(0x5A);
    let mut v = vec![0f32; n];
    rng.fill_normal(&mut v);
    let mut s = norm(&v);
    v.iter_mut().for_each(|x| *x /= s);
    for _ in 0..40 {
        let u = matvec(w, k, n, &v);
        let mut vt = matvec_t(w, k, n, &u);
        s = norm(&vt);
        vt.iter_mut().for_each(|x| *x /= s.max(1e-30));
        v = vt;
    }
    let u = matvec(w, k, n, &v);
    norm(&u) // σ₁ = ‖W v‖ for the converged top right-singular vector v (‖v‖=1)
}

/// Absmean ternary (BitNet b1.58): scale=mean|W|, round to {-1,0,+1}·scale.
fn ternary_recon_err(w: &[f32]) -> f32 {
    let s = w.iter().map(|x| x.abs()).sum::<f32>() / w.len() as f32;
    let s = if s > 0.0 { s } else { 1.0 };
    let (mut sd, mut sb) = (0f64, 0f64);
    for &x in w {
        let t = (x / s).round().clamp(-1.0, 1.0) * s;
        let e = (x - t) as f64;
        sd += e * e;
        sb += (x as f64) * (x as f64);
    }
    (sd / sb.max(1e-30)).sqrt() as f32
}

fn recon_err(w: &[f32], k: usize, n: usize, q: WeightQuant) -> f32 {
    let d = fake_quant_weight(w, k, n, q);
    let (mut sd, mut sb) = (0f64, 0f64);
    for (a, b) in w.iter().zip(&d) {
        let e = (*a - *b) as f64;
        sd += e * e;
        sb += (*a as f64) * (*a as f64);
    }
    (sd / sb.max(1e-30)).sqrt() as f32
}

fn main() -> Result<(), String> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Volumes/FOUR/kimi".into());
    if !Path::new(&model_dir).join("config.json").exists() {
        eprintln!("skip: {model_dir}/config.json not found");
        return Ok(());
    }
    let kc =
        KimiK3Config::load(Path::new(&model_dir).join("config.json")).map_err(|e| e.to_string())?;
    let hidden = kc.text_config.hidden_size;
    let d = KdaDims {
        hidden,
        num_heads: 96,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq: 1,
    };
    let mut ck = CheckpointLoader::open(&model_dir).map_err(|e| e.to_string())?;
    let w: KdaWeights = ck
        .load_kda("language_model.model.layers.0", d)
        .map_err(|e| e.to_string())?;
    let proj = 96 * 128;

    // (name, weight, K, N)
    let mats: Vec<(&str, &[f32], usize, usize)> = vec![
        ("q_proj", &w.q_proj, hidden, proj),
        ("v_proj", &w.v_proj, hidden, proj),
        ("g_proj", &w.g_proj, hidden, proj),
        ("f_a", &w.f_a, hidden, 128),
        ("f_b", &w.f_b, 128, proj),
        ("o_proj", &w.o_proj, proj, hidden),
    ];

    eprintln!("\nKimi-K3 backbone weight structure (real layer-0 KDA):");
    eprintln!(
        "{:<8} {:>12} {:>10} {:>8} {:>8}   {:>9} {:>7} {:>7}",
        "mat", "shape", "stable_rk", "%ofmin", "density", "ternary", "int4", "int8"
    );
    for (name, wt, k, n) in mats {
        let fro2: f64 = wt.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let s1 = sigma1(wt, k, n);
        let stable_rank = fro2 / (s1 as f64 * s1 as f64).max(1e-30);
        let minkn = k.min(n);
        let nz = wt.iter().filter(|&&x| x != 0.0).count();
        let density = nz as f64 / wt.len() as f64;
        let tern = ternary_recon_err(wt);
        let i4 = recon_err(wt, k, n, WeightQuant::Int4G64);
        let i8 = recon_err(wt, k, n, WeightQuant::Int8Ch);
        eprintln!(
            "{name:<8} {:>12} {stable_rank:>10.0} {:>7.0}% {density:>8.3}   {tern:>8.1}% {:>6.1}% {:>6.1}%",
            format!("{k}x{n}"),
            100.0 * stable_rank / minkn as f64,
            i4 * 100.0,
            i8 * 100.0,
        );
    }
    eprintln!(
        "\nlow-rank pays only if stable_rank ≪ min(K,N) (then r(K+N)<KN flops). \
         Ternary recon-err ≫ int4 (already token-breaking) ⇒ hopeless for the backbone.\n\
         Big matrices (MLA q/kv-lora, LatentMoE latent) are ALREADY low-rank by architecture."
    );
    Ok(())
}

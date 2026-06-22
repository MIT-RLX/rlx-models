// Verify that rlx's graph `reshape → rms_norm → reshape` on a 4D
// tensor with 1D gamma produces the SAME result as a host-side
// per-head RMS norm. If they differ, the rlx runtime has a 4D
// shape-handling bug — the suspected root cause for task #50's
// persistent all-NaN logits.
//
// Construct a Q-like tensor [B=1, S=1, nh=16, head_dim=256], use a
// uniform gamma matching Gemma 4's q_norm (1.023 everywhere), run
// it through a tiny rlx graph (reshape → rms_norm → reshape) using
// the CPU backend, and compare against a pure-Rust per-head RMS
// norm computed in this binary.

use anyhow::Result;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

const B: usize = 1;
const S: usize = 1;
const NH: usize = 16;
const HEAD_DIM: usize = 256;
const Q_DIM: usize = NH * HEAD_DIM; // 4096

fn rms_norm_per_head(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for head in 0..NH {
        let off = head * HEAD_DIM;
        let slice = &x[off..off + HEAD_DIM];
        let sumsq: f32 = slice.iter().map(|v| v * v).sum();
        let inv_rms = (sumsq / HEAD_DIM as f32 + eps).sqrt().recip();
        for i in 0..HEAD_DIM {
            out[off + i] = slice[i] * inv_rms * gamma[i];
        }
    }
    out
}

fn stats(label: &str, x: &[f32]) {
    let mut n_nan = 0;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut n_finite = 0usize;
    for &v in x {
        if v.is_nan() {
            n_nan += 1;
            continue;
        }
        n_finite += 1;
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v as f64;
        sumsq += (v as f64) * (v as f64);
    }
    let mean = if n_finite > 0 {
        sum / n_finite as f64
    } else {
        0.0
    };
    let rms = (sumsq / n_finite.max(1) as f64).sqrt();
    println!(
        "{label:35} nan={n_nan:>3} min={min:+.3e} max={max:+.3e} mean={mean:+.3e} rms={rms:.3e}"
    );
}

fn main() -> Result<()> {
    let eps = 1e-6f32;

    // Generate a deterministic Q-shaped input.
    let q_data: Vec<f32> = (0..(B * S * Q_DIM))
        .map(|i| ((i as f32) * 0.0001).sin() * 3.0)
        .collect();
    // Gemma q_norm.weight: uniform 1.023; gemma_rms shifts by +1.
    let q_norm_w: Vec<f32> = vec![1.023f32; HEAD_DIM];
    let gamma_actual: Vec<f32> = q_norm_w.iter().map(|w| 1.0 + w).collect();

    // ── HOST REFERENCE ────────────────────────────────────────────
    let expected = rms_norm_per_head(&q_data, &gamma_actual, eps);
    stats("host: reshape→per-head rms_norm", &expected);

    // ── RLX GRAPH ─────────────────────────────────────────────────
    let mut g = Graph::new("rmsnorm_4d_diag");

    let q_in = g.input("q", Shape::new(&[B, S, Q_DIM], DType::F32));

    // Build a const for gamma_actual = 1.023 + 1 = 2.023 everywhere.
    let mut params: std::collections::HashMap<String, Vec<f32>> = Default::default();
    let gamma_id = g.param("gamma", Shape::new(&[HEAD_DIM], DType::F32));
    params.insert("gamma".into(), gamma_actual.clone());
    let beta_id = g.param("beta", Shape::new(&[HEAD_DIM], DType::F32));
    params.insert("beta".into(), vec![0f32; HEAD_DIM]);

    // reshape Q [B, S, Q_DIM] → [B, S, NH, HEAD_DIM]
    let q_4d = g.reshape_(q_in, vec![B as i64, S as i64, NH as i64, HEAD_DIM as i64]);
    // rms_norm — last-axis with gamma[HEAD_DIM], beta[HEAD_DIM]
    let q_normed = g.rms_norm(q_4d, gamma_id, beta_id, eps);
    // reshape back
    let q_out = g.reshape_(q_normed, vec![B as i64, S as i64, Q_DIM as i64]);
    g.outputs = vec![q_out];

    let _ = CompileOptions::default();
    let dev = std::env::var("RLX_DIAG_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = match dev.as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        _ => Device::Cpu,
    };
    println!("(device: {device:?})");
    let session = Session::new(device);
    let mut compiled = session.compile(g);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    let outs = compiled.run(&[("q", &q_data)]);
    let actual = &outs[0];
    stats("rlx-cpu: graph output", actual);

    // ── DIFF ──────────────────────────────────────────────────────
    let mut max_abs_diff = 0f32;
    let mut max_rel_diff = 0f32;
    let mut nan_in_actual = 0usize;
    for (a, e) in actual.iter().zip(expected.iter()) {
        if a.is_nan() {
            nan_in_actual += 1;
            continue;
        }
        let abs_d = (a - e).abs();
        if abs_d > max_abs_diff {
            max_abs_diff = abs_d;
        }
        let rel_d = abs_d / e.abs().max(1e-6);
        if rel_d > max_rel_diff {
            max_rel_diff = rel_d;
        }
    }
    println!("\nmax abs diff = {max_abs_diff:.4e}");
    println!("max rel diff = {max_rel_diff:.4e}");
    println!("nan in actual = {nan_in_actual}");
    if max_abs_diff < 1e-3 && nan_in_actual == 0 {
        println!("MATCH ✓ — rlx-cpu graph reshape+rms_norm+reshape matches host scalar reference.");
        println!(
            "The bug is NOT in this op chain — must be in RoPE / SDPA / o_proj / residual / FFN."
        );
    } else {
        println!("MISMATCH ✗ — rlx-cpu graph misexecutes the reshape+rms_norm+reshape pattern.");
        println!(
            "This IS the task #50 root cause. Fix in rlx-ir / rlx-cpu rms_norm shape handling."
        );
    }
    Ok(())
}

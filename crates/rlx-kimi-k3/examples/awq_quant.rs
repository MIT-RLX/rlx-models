//! `awq_quant` — does ACTIVATION-AWARE quantization (AWQ) rescue 4-bit where every
//! weight-only codebook (int4/mxfp4/nf4/outlier-mix, all ~14 dB) failed? AWQ scales
//! each INPUT channel of a weight by `act_importance^alpha` before int4-quantizing
//! (folding the inverse into the preceding norm at inference), spending precision
//! on the channels the activations actually exercise. This is the tractable member
//! of the GPTQ family (O(K·N), no Hessian inverse — full GPTQ on this 7168×12288
//! layer is ~6e14 flops, an offline-GPU cost).
//!
//! Faithful test on the REAL layer-0 `q_proj`: calibration + measurement use REAL
//! token embeddings pushed through the layer's REAL `input_layernorm`, and error is
//! the OUTPUT error `‖Xt·W_q − Xt·W‖/‖Xt·W‖` on a HELD-OUT token set — exactly what
//! AWQ minimizes. Compares int8 / int4-RTN / int4-AWQ.
//!
//!   cargo run -p rlx-kimi-k3 --features cluster --example awq_quant [-- model_dir]

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_kimi_k3::common::{WeightQuant, fake_quant_weight};
use rlx_kimi_k3::config::KimiK3Config;
use rlx_kimi_k3::kda::KdaDims;
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::Path;

const EMB: &str = "language_model.model.embed_tokens.weight";

/// `x[m,k] @ w[k,n] -> [m,n]` via a compiled 1-op graph (fast CPU BLAS/AMX).
fn matmul(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut hir = HirModule::new("mm");
    let mut g = HirMut::new(&mut hir);
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let mut p = HashMap::new();
    p.insert("w".to_string(), w.to_vec());
    let wi = g.param("w", Shape::new(&[k, n], DType::F32));
    let out = g.mm(xi, wi);
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    c.run(&[("x", x)]).remove(0)
}

fn rel_l2(base: &[f32], q: &[f32]) -> (f32, f32) {
    let (mut sd, mut sb) = (0f64, 0f64);
    for (b, v) in base.iter().zip(q) {
        let e = (*b - *v) as f64;
        sd += e * e;
        sb += (*b as f64) * (*b as f64);
    }
    let rel = (sd / sb.max(1e-30)).sqrt() as f32;
    let snr = if sd > 0.0 {
        10.0 * (sb / sd).log10() as f32
    } else {
        f32::INFINITY
    };
    (rel, snr)
}

/// RMSNorm a `[m,k]` batch in place with per-channel `weight[k]` (eps 1e-5).
fn rmsnorm(x: &mut [f32], m: usize, k: usize, w: &[f32]) {
    for r in 0..m {
        let row = &mut x[r * k..r * k + k];
        let ms: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / k as f64;
        let inv = 1.0 / (ms + 1e-5).sqrt();
        for (v, &g) in row.iter_mut().zip(w) {
            *v = (*v as f64 * inv) as f32 * g;
        }
    }
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
    let tc = &kc.text_config;
    let (hidden, vocab) = (tc.hidden_size, tc.vocab_size);
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
    let w = ck
        .load_kda("language_model.model.layers.0", d)
        .map_err(|e| e.to_string())?;
    let ln = ck
        .tensor_f32("language_model.model.layers.0.input_layernorm.weight")
        .map_err(|e| e.to_string())?;
    let wq = &w.q_proj; // [K=hidden, N=proj]
    let (k, n) = (hidden, w.q_proj.len() / hidden);
    eprintln!("q_proj [{k}, {n}] (real layer-0), calibrating on REAL normed embeddings");

    // real calibration + held-out test activations = embed(tokens) -> input_layernorm.
    let calib: Vec<u32> = (0..512u32)
        .map(|i| (i.wrapping_mul(313).wrapping_add(7)) % vocab as u32)
        .collect();
    let test: Vec<u32> = (0..256u32)
        .map(|i| (i.wrapping_mul(911).wrapping_add(101)) % vocab as u32)
        .collect();
    let mut xc = ck
        .gather_embed(EMB, &calib, hidden)
        .map_err(|e| e.to_string())?;
    let mut xt = ck
        .gather_embed(EMB, &test, hidden)
        .map_err(|e| e.to_string())?;
    let (mc, mt) = (calib.len(), test.len());
    rmsnorm(&mut xc, mc, k, &ln);
    rmsnorm(&mut xt, mt, k, &ln);

    // reference outputs (full precision).
    let base_t = matmul(&xt, mt, k, wq, n);

    // baselines: per-channel int8 and int4-g64 RTN.
    let q8 = fake_quant_weight(wq, k, n, WeightQuant::Int8Ch);
    let q4 = fake_quant_weight(wq, k, n, WeightQuant::Int4G64);
    let (r8, s8) = rel_l2(&base_t, &matmul(&xt, mt, k, &q8, n));
    let (r4, s4) = rel_l2(&base_t, &matmul(&xt, mt, k, &q4, n));

    // per-input-channel activation importance from the calibration set.
    let mut imp = vec![0f32; k];
    for r in 0..mc {
        for j in 0..k {
            imp[j] += xc[r * k + j].abs();
        }
    }
    let logmean: f64 = imp
        .iter()
        .map(|&a| ((a / mc as f32).max(1e-12) as f64).ln())
        .sum::<f64>()
        / k as f64;
    let gmean = logmean.exp() as f32;

    // AWQ: W'[j,:] = W[j,:] * s_j ; X'[:,j] = X[:,j] / s_j ; quantize W' int4-g64.
    // grid-search alpha on the CALIB output error, then report on the TEST set.
    let base_c = matmul(&xc, mc, k, wq, n);
    let apply = |alpha: f32| -> (Vec<f32>, Vec<f32>) {
        let mut s = vec![0f32; k];
        for j in 0..k {
            let a = (imp[j] / mc as f32).max(1e-12) / gmean;
            s[j] = a.powf(alpha).clamp(0.05, 20.0);
        }
        let mut wp = wq.clone();
        for j in 0..k {
            for c in 0..n {
                wp[j * n + c] *= s[j];
            }
        }
        let qp = fake_quant_weight(&wp, k, n, WeightQuant::Int4G64);
        (s, qp)
    };
    let scaled_x = |x: &[f32], m: usize, s: &[f32]| -> Vec<f32> {
        let mut o = x.to_vec();
        for r in 0..m {
            for j in 0..k {
                o[r * k + j] /= s[j];
            }
        }
        o
    };

    eprintln!(
        "\n{:>6} {:>10} {:>8}   (int4-AWQ vs held-out test)",
        "alpha", "relL2", "SNRdB"
    );
    let alphas = [0.0f32, 0.25, 0.5, 0.75, 1.0];
    let (mut best_a, mut best_c) = (0f32, f32::INFINITY);
    let mut test_rows = Vec::new();
    for &a in &alphas {
        let (s, qp) = apply(a);
        let (rc, _) = rel_l2(&base_c, &matmul(&scaled_x(&xc, mc, &s), mc, k, &qp, n));
        let (rt, st) = rel_l2(&base_t, &matmul(&scaled_x(&xt, mt, &s), mt, k, &qp, n));
        eprintln!("{a:>6.2} {rt:>10.3e} {st:>8.2}   (calib relL2 {rc:.3e})");
        test_rows.push((a, rt, st));
        if rc < best_c {
            best_c = rc;
            best_a = a;
        }
    }
    let (_, best_rt, best_st) = test_rows
        .iter()
        .find(|(a, _, _)| *a == best_a)
        .copied()
        .unwrap();

    eprintln!(
        "\n== summary (held-out test output error) ==\n\
         int8 per-channel : relL2 {r8:.3e}  SNR {s8:.2} dB  (8 bits)\n\
         int4-g64 RTN     : relL2 {r4:.3e}  SNR {s4:.2} dB  (4 bits)\n\
         int4-AWQ (a={best_a:.2})  : relL2 {best_rt:.3e}  SNR {best_st:.2} dB  (4 bits, activation-aware)\n\
         → AWQ closes {:.0}% of the int4→int8 SNR gap.",
        100.0 * ((best_st - s4) / (s8 - s4)).clamp(0.0, 1.0)
    );
    Ok(())
}

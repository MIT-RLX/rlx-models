// RLX — versatile ML compiler + runtime. GPLv3.
//! Numerical isolation check for **Kimi Delta Attention (KDA)**
//! ([`build_kimi_kda`]) — Kimi-Linear's novel primitive: a fine-grained
//! (per-key-dim) gated delta-net linear attention that `Op::GatedDeltaNet`
//! (scalar per-head gate) does NOT cover. Kimi-Linear-48B-A3B can't be run
//! e2e on a 64GB box, so this proves the KDA block — short-conv + q/k
//! RMS-norm&scale + a/b/out gates + the delta-rule recurrence + o-norm×σ(gate)
//! — against an inline port of mlx-lm `KimiDeltaAttention` + `gated_delta_ops`,
//! on tiny synthetic dense weights (NO checkpoint needed).
//!
//!   cargo run --release -p rlx-models-core --example kimi_kda_probe

use anyhow::Result;
use rlx_ir::{DType, Graph, Shape};
use rlx_models_core::standard_decoder::build_kimi_kda;
use rlx_models_core::weight_loader::WeightLoader;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    (x - x.floor()) as f32 - 0.5
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(v: f32) -> f32 {
    v.max(0.0) + (1.0 + (-v.abs()).exp()).ln()
}

struct MemLoader {
    t: HashMap<String, (Vec<f32>, Vec<usize>)>,
}
impl WeightLoader for MemLoader {
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.t
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing {key}"))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (d, s) = self.take(key)?;
        assert_eq!(s.len(), 2, "{key}");
        let (r, c) = (s[0], s[1]);
        let mut o = vec![0f32; d.len()];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = d[i * c + j];
            }
        }
        Ok((o, vec![c, r]))
    }
    fn len(&self) -> usize {
        self.t.len()
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.t.keys().cloned().collect()
    }
}

fn matmul(x: &[f32], w: &[f32], rows: usize, inn: usize, out: usize) -> Vec<f32> {
    // y[r,o] = sum_i x[r,i]*w[o,i]   (w is [out,in])
    let mut y = vec![0f32; rows * out];
    for r in 0..rows {
        for o in 0..out {
            let mut s = 0f32;
            for i in 0..inn {
                s += x[r * inn + i] * w[o * inn + i];
            }
            y[r * out + o] = s;
        }
    }
    y
}
fn conv_silu(inp: &[f32], w: &[f32], seq: usize, ch: usize, k: usize) -> Vec<f32> {
    // depthwise causal, left-pad k-1; out[t,c]=silu(sum_j w[c,j]*padded[t+j,c])
    let mut o = vec![0f32; seq * ch];
    for c in 0..ch {
        for t in 0..seq {
            let mut acc = 0f32;
            for j in 0..k {
                let pos = t as isize + j as isize - (k as isize - 1);
                if pos >= 0 {
                    acc += w[c * k + j] * inp[(pos as usize) * ch + c];
                }
            }
            o[t * ch + c] = silu(acc);
        }
    }
    o
}

fn main() -> Result<()> {
    let (hv, dk, ck, seq, hidden) = (4usize, 8usize, 4usize, 5usize, 16usize);
    let p = hv * dk; // 32
    let eps_o = 1e-5f32;
    let scale = (dk as f32).powf(-0.5);
    let sa = "sa";

    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mk =
        |seed: f64, n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| s * rnd(seed, i)).collect() };
    let q_w = mk(1.0, p * hidden, 0.2);
    let k_w = mk(2.0, p * hidden, 0.2);
    let v_w = mk(3.0, p * hidden, 0.2);
    let qc = mk(4.0, p * ck, 0.5);
    let kc = mk(5.0, p * ck, 0.5);
    let vc = mk(6.0, p * ck, 0.5);
    let fa = mk(7.0, dk * hidden, 0.2);
    let fb = mk(8.0, p * dk, 0.2);
    let bp = mk(9.0, hv * hidden, 0.2);
    let ga = mk(10.0, dk * hidden, 0.2);
    let gbw = mk(11.0, p * dk, 0.2);
    let a_log = mk(12.0, hv, 0.5)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>(); // >0-ish
    let dt_bias = mk(13.0, p, 0.1);
    let o_norm = mk(14.0, dk, 0.1)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>();
    let o_w = mk(15.0, hidden * p, 0.2);
    t.insert(
        format!("{sa}.q_proj.weight"),
        (q_w.clone(), vec![p, hidden]),
    );
    t.insert(
        format!("{sa}.k_proj.weight"),
        (k_w.clone(), vec![p, hidden]),
    );
    t.insert(
        format!("{sa}.v_proj.weight"),
        (v_w.clone(), vec![p, hidden]),
    );
    t.insert(
        format!("{sa}.q_conv.conv.weight"),
        (qc.clone(), vec![p, ck, 1]),
    );
    t.insert(
        format!("{sa}.k_conv.conv.weight"),
        (kc.clone(), vec![p, ck, 1]),
    );
    t.insert(
        format!("{sa}.v_conv.conv.weight"),
        (vc.clone(), vec![p, ck, 1]),
    );
    t.insert(
        format!("{sa}.f_a_proj.weight"),
        (fa.clone(), vec![dk, hidden]),
    );
    t.insert(format!("{sa}.f_b_proj.weight"), (fb.clone(), vec![p, dk]));
    t.insert(
        format!("{sa}.b_proj.weight"),
        (bp.clone(), vec![hv, hidden]),
    );
    t.insert(
        format!("{sa}.g_a_proj.weight"),
        (ga.clone(), vec![dk, hidden]),
    );
    t.insert(format!("{sa}.g_b_proj.weight"), (gbw.clone(), vec![p, dk]));
    t.insert(format!("{sa}.A_log"), (a_log.clone(), vec![hv]));
    t.insert(format!("{sa}.dt_bias"), (dt_bias.clone(), vec![p]));
    t.insert(format!("{sa}.o_norm.weight"), (o_norm.clone(), vec![dk]));
    t.insert(
        format!("{sa}.o_proj.weight"),
        (o_w.clone(), vec![hidden, p]),
    );
    let mut loader = MemLoader { t };

    let x: Vec<f32> = (0..seq * hidden).map(|i| 0.5 * rnd(9.9, i)).collect();

    // ── graph ──
    let mut g = Graph::new("kimi_kda_probe");
    let xin = g.input("x", Shape::new(&[1, seq, hidden], DType::F32));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
        HashMap::new();
    let out = build_kimi_kda(
        &mut g,
        &mut params,
        &mut packed,
        &mut loader,
        sa,
        xin,
        seq,
        hidden,
        hv,
        dk,
        ck,
        eps_o,
    )?;
    g.set_outputs(vec![out]);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let got = compiled
        .run(&[("x", x.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    // ── inline reference (mlx KimiDeltaAttention + gated_delta_ops) ──
    let rmsnorm = |vals: &[f32], gain: Option<&[f32]>, n: usize, eps: f32| -> Vec<f32> {
        // per-row RMS over last n
        let rows = vals.len() / n;
        let mut o = vec![0f32; vals.len()];
        for r in 0..rows {
            let mut ms = 0f32;
            for i in 0..n {
                let x = vals[r * n + i];
                ms += x * x;
            }
            let inv = 1.0 / (ms / n as f32 + eps).sqrt();
            for i in 0..n {
                let gg = gain.map(|w| w[i]).unwrap_or(1.0);
                o[r * n + i] = vals[r * n + i] * inv * gg;
            }
        }
        o
    };
    let qraw = conv_silu(&matmul(&x, &q_w, seq, hidden, p), &qc, seq, p, ck);
    let kraw = conv_silu(&matmul(&x, &k_w, seq, hidden, p), &kc, seq, p, ck);
    let vraw = conv_silu(&matmul(&x, &v_w, seq, hidden, p), &vc, seq, p, ck);
    // per-head rmsnorm(eps 1e-6) over dk + scale
    let qn = rmsnorm(&qraw, None, dk, 1e-6);
    let kn = rmsnorm(&kraw, None, dk, 1e-6);
    let qh: Vec<f32> = qn.iter().map(|v| v * scale * scale).collect();
    let kh: Vec<f32> = kn.iter().map(|v| v * scale).collect();
    // gates
    let a_logits = matmul(&matmul(&x, &fa, seq, hidden, dk), &fb, seq, dk, p); // [seq,p]
    let b_logits = matmul(&x, &bp, seq, hidden, hv); // [seq,hv]
    let out_gate = matmul(&matmul(&x, &ga, seq, hidden, dk), &gbw, seq, dk, p); // [seq,p]
    let neg_exp_a: Vec<f32> = a_log.iter().map(|v| -(v.exp())).collect();
    // g[t,h,i] = exp(neg_exp_a[h] * softplus(a_logits[t,h*dk+i] + dt_bias[h*dk+i]))
    let mut gdecay = vec![0f32; seq * p];
    for t_ in 0..seq {
        for h in 0..hv {
            for i in 0..dk {
                let idx = h * dk + i;
                let sp = softplus(a_logits[t_ * p + idx] + dt_bias[idx]);
                gdecay[t_ * p + idx] = (neg_exp_a[h] * sp).exp();
            }
        }
    }
    // recurrence
    let mut y = vec![0f32; seq * p]; // [seq, hv, dk]
    let mut state = vec![0f32; hv * dk * dk]; // [hv, dv, dk]
    for t_ in 0..seq {
        for h in 0..hv {
            let beta = sigmoid(b_logits[t_ * hv + h]);
            // decay
            for dv in 0..dk {
                for i in 0..dk {
                    state[(h * dk + dv) * dk + i] *= gdecay[t_ * p + h * dk + i];
                }
            }
            // kv_mem[dv] = sum_i state[dv,i]*k[i]
            let mut kv = vec![0f32; dk];
            for dv in 0..dk {
                let mut s = 0f32;
                for i in 0..dk {
                    s += state[(h * dk + dv) * dk + i] * kh[t_ * p + h * dk + i];
                }
                kv[dv] = s;
            }
            // delta, state update, y
            for dv in 0..dk {
                let delta = (vraw[t_ * p + h * dk + dv] - kv[dv]) * beta;
                for i in 0..dk {
                    state[(h * dk + dv) * dk + i] += kh[t_ * p + h * dk + i] * delta;
                }
            }
            for dv in 0..dk {
                let mut s = 0f32;
                for i in 0..dk {
                    s += state[(h * dk + dv) * dk + i] * qh[t_ * p + h * dk + i];
                }
                y[t_ * p + h * dk + dv] = s;
            }
        }
    }
    // o_norm(per-head, gain) * sigmoid(out_gate) → o_proj
    let yn = rmsnorm(&y, Some(&o_norm), dk, eps_o);
    let gated: Vec<f32> = yn
        .iter()
        .zip(&out_gate)
        .map(|(a, b)| a * sigmoid(*b))
        .collect();
    let refout = matmul(&gated, &o_w, seq, p, hidden);

    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in got.iter().zip(&refout) {
        dot += *a as f64 * *b as f64;
        na += *a as f64 * *a as f64;
        nb += *b as f64 * *b as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    let maxerr = got
        .iter()
        .zip(&refout)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let finite = got.iter().all(|v| v.is_finite());
    println!("── Kimi Delta Attention (KDA): rlx graph vs inline gated_delta_ops reference ──");
    println!("elements = {}  finite = {finite}", got.len());
    println!("cosine   = {cos:.8}");
    println!("max|err| = {maxerr:.3e}");
    if finite && cos > 0.999999 && maxerr < 1e-3 {
        println!("✅ KDA block matches the reference");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "KDA mismatch: cos={cos:.8} maxerr={maxerr:.3e}"
        ))
    }
}

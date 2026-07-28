// RLX — versatile ML compiler + runtime. GPLv3.
//! Numerical isolation check for the **Nemotron-H Mamba-2 (SSD) mixer**
//! ([`build_nemotron_h_mamba2`]). Nemotron-H's in-scope mlx-community checkpoints
//! are 30B–120B MoE giants that can't be run-validated on a 64GB box, so this
//! proves the *novel* piece — the mixer's plumbing around the (already
//! parity-tested) `Op::Mamba2` scan: the `in_proj` split, depthwise causal conv
//! (+bias)+SiLU, `x|B|C` split, group→head repeat of B/C, `a=-exp(A_log)`,
//! `dt=clamp(softplus(dt+dt_bias))`, `+D·x` skip, and the gated group-RMSNorm —
//! against an inline reference on tiny synthetic weights (NO checkpoint needed).
//!
//!   cargo run --release -p rlx-models-core --example nemotron_h_mamba2_probe

use anyhow::Result;
use rlx_ir::{DType, Graph, Shape};
use rlx_models_core::standard_decoder::{NemotronHSpec, build_nemotron_h_mamba2};
use rlx_models_core::weight_loader::WeightLoader;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

/// Deterministic pseudo-random in [-0.5, 0.5].
fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    (x - x.floor()) as f32 - 0.5
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// In-memory dense loader: every tensor is plain F32 row-major, so `load_proj`
/// falls through to its dense path (no quant), isolating the SSD math.
struct MemLoader {
    t: HashMap<String, (Vec<f32>, Vec<usize>)>,
}
impl WeightLoader for MemLoader {
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.t
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (d, s) = self.take(key)?;
        assert_eq!(s.len(), 2, "transpose needs 2D: {key}");
        let (r, c) = (s[0], s[1]);
        let mut out = vec![0f32; d.len()];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = d[i * c + j];
            }
        }
        Ok((out, vec![c, r]))
    }
    fn len(&self) -> usize {
        self.t.len()
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.t.keys().cloned().collect()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn main() -> Result<()> {
    // Tiny synthetic Mamba-2 mixer config.
    let (hidden, nh, dh, st, ng, k, seq) =
        (16usize, 4usize, 4usize, 8usize, 2usize, 4usize, 5usize);
    let eps = 1e-5f32;
    let spec = NemotronHSpec {
        vocab_size: 32,
        hidden_size: hidden,
        hybrid_pattern: vec!['M'],
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        attention_bias: false,
        mamba_num_heads: nh,
        mamba_head_dim: dh,
        ssm_state_size: st,
        conv_kernel: k,
        n_groups: ng,
        use_conv_bias: true,
        time_step_limit: (0.0, 1e9),
        intermediate_size: 16,
        moe_intermediate_size: 16,
        moe_shared_expert_intermediate_size: 16,
        n_routed_experts: 4,
        n_shared_experts: 0,
        num_experts_per_tok: 2,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
        norm_topk_prob: true,
        rms_norm_eps: eps,
    };
    let inter = nh * dh; // 16
    let conv_dim = inter + 2 * ng * st; // 48
    let in_out = inter + conv_dim + nh; // 68
    let hpg = nh / ng; // 2
    let gsize = inter / ng; // 8
    let mp = "backbone.layers.0.mixer";

    // Weights (natural [out, in] / row-major).
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let w_in: Vec<f32> = (0..in_out * hidden).map(|i| 0.15 * rnd(1.0, i)).collect();
    let w_conv: Vec<f32> = (0..conv_dim * k).map(|i| 0.4 * rnd(2.0, i)).collect();
    let b_conv: Vec<f32> = (0..conv_dim).map(|i| 0.1 * rnd(3.0, i)).collect();
    let dt_bias: Vec<f32> = (0..nh).map(|i| 0.05 * rnd(4.0, i)).collect();
    let a_log: Vec<f32> = (0..nh).map(|i| ((i + 1) as f32).ln()).collect(); // mlx init
    let d_skip: Vec<f32> = (0..nh).map(|i| 1.0 + 0.1 * rnd(5.0, i)).collect();
    let norm_w: Vec<f32> = (0..inter).map(|i| 1.0 + 0.1 * rnd(6.0, i)).collect();
    let w_out: Vec<f32> = (0..hidden * inter).map(|i| 0.15 * rnd(7.0, i)).collect();
    t.insert(
        format!("{mp}.in_proj.weight"),
        (w_in.clone(), vec![in_out, hidden]),
    );
    t.insert(
        format!("{mp}.conv1d.weight"),
        (w_conv.clone(), vec![conv_dim, k, 1]),
    );
    t.insert(
        format!("{mp}.conv1d.bias"),
        (b_conv.clone(), vec![conv_dim]),
    );
    t.insert(format!("{mp}.dt_bias"), (dt_bias.clone(), vec![nh]));
    t.insert(format!("{mp}.A_log"), (a_log.clone(), vec![nh]));
    t.insert(format!("{mp}.D"), (d_skip.clone(), vec![nh]));
    t.insert(format!("{mp}.norm.weight"), (norm_w.clone(), vec![inter]));
    t.insert(
        format!("{mp}.out_proj.weight"),
        (w_out.clone(), vec![hidden, inter]),
    );
    let mut loader = MemLoader { t };

    // Input activation [1, seq, hidden].
    let x: Vec<f32> = (0..seq * hidden).map(|i| 0.5 * rnd(9.0, i)).collect();

    // ── Build + run the mixer graph on CPU ──
    let mut g = Graph::new("nemo_mamba2_probe");
    let xin = g.input("x", Shape::new(&[1, seq, hidden], DType::F32));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
        HashMap::new();
    let out = build_nemotron_h_mamba2(
        &mut g,
        &mut params,
        &mut packed,
        &mut loader,
        mp,
        xin,
        seq,
        hidden,
        &spec,
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
    for (n, (b, _, _)) in &packed {
        compiled.set_param_typed(n, b, DType::U8);
    }
    let got = compiled
        .run(&[("x", x.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    // ── Inline reference (mirrors build_nemotron_h_mamba2 / mlx-lm) ──
    // proj = x @ W_in^T  → split gate|conv_input|dt
    let mut proj = vec![0f32; seq * in_out];
    for t_ in 0..seq {
        for o in 0..in_out {
            let mut s = 0f32;
            for hh in 0..hidden {
                s += x[t_ * hidden + hh] * w_in[o * hidden + hh];
            }
            proj[t_ * in_out + o] = s;
        }
    }
    let gate = |t_: usize, i: usize| proj[t_ * in_out + i]; // [.., 0..inter]
    // depthwise causal conv (+bias) + SiLU over conv_input = proj[.., inter..inter+conv_dim]
    let mut conv = vec![0f32; seq * conv_dim];
    for c in 0..conv_dim {
        for t_ in 0..seq {
            let mut acc = b_conv[c];
            for j in 0..k {
                // padded index t_+j into [k-1 zeros, seq]; original pos = t_+j-(k-1)
                let pos = t_ as isize + j as isize - (k as isize - 1);
                if pos >= 0 {
                    acc += w_conv[c * k + j] * proj[(pos as usize) * in_out + inter + c];
                }
            }
            conv[t_ * conv_dim + c] = silu(acc);
        }
    }
    let xssm = |t_: usize, h: usize, pi: usize| conv[t_ * conv_dim + h * dh + pi]; // [.., 0..inter]
    let bval = |t_: usize, gi: usize, n: usize| conv[t_ * conv_dim + inter + gi * st + n];
    let cval = |t_: usize, gi: usize, n: usize| conv[t_ * conv_dim + inter + ng * st + gi * st + n];
    // dt = clamp(softplus(dt_raw + dt_bias)); a = -exp(A_log)
    let sp = |v: f32| v.max(0.0) + (1.0 + (-v.abs()).exp()).ln();
    let mut dt = vec![0f32; seq * nh];
    for t_ in 0..seq {
        for h in 0..nh {
            let raw = proj[t_ * in_out + inter + conv_dim + h] + dt_bias[h];
            dt[t_ * nh + h] = sp(raw).clamp(spec.time_step_limit.0, spec.time_step_limit.1);
        }
    }
    let a: Vec<f32> = a_log.iter().map(|&v| -(v.exp())).collect();
    // SSD recurrence per head + D skip → y [seq, inter]
    let mut y = vec![0f32; seq * inter];
    for h in 0..nh {
        let gi = h / hpg;
        let mut state = vec![0f32; dh * st];
        for t_ in 0..seq {
            let da = (dt[t_ * nh + h] * a[h]).exp();
            for pi in 0..dh {
                let dtx = dt[t_ * nh + h] * xssm(t_, h, pi);
                for n in 0..st {
                    state[pi * st + n] = da * state[pi * st + n] + dtx * bval(t_, gi, n);
                }
            }
            for pi in 0..dh {
                let mut acc = 0f32;
                for n in 0..st {
                    acc += state[pi * st + n] * cval(t_, gi, n);
                }
                y[t_ * inter + h * dh + pi] = acc + xssm(t_, h, pi) * d_skip[h];
            }
        }
    }
    // gated group-RMSNorm: silu(gate)·y → RMS per group (size gsize) → ×norm_w
    let mut refout = vec![0f32; seq * hidden];
    for t_ in 0..seq {
        let mut gated = vec![0f32; inter];
        for i in 0..inter {
            gated[i] = silu(gate(t_, i)) * y[t_ * inter + i];
        }
        // per-group RMS (gsize each)
        let mut normed = vec![0f32; inter];
        for grp in 0..(inter / gsize) {
            let mut ms = 0f32;
            for i in 0..gsize {
                let v = gated[grp * gsize + i];
                ms += v * v;
            }
            ms /= gsize as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for i in 0..gsize {
                normed[grp * gsize + i] = gated[grp * gsize + i] * inv;
            }
        }
        for i in 0..inter {
            normed[i] *= norm_w[i];
        }
        // out = normed @ W_out^T  (W_out [hidden, inter])
        for o in 0..hidden {
            let mut s = 0f32;
            for i in 0..inter {
                s += normed[i] * w_out[o * inter + i];
            }
            refout[t_ * hidden + o] = s;
        }
    }

    let cos = cosine(&got, &refout);
    let maxerr = got
        .iter()
        .zip(&refout)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let finite = got.iter().all(|v| v.is_finite());
    println!("── Nemotron-H Mamba-2 mixer: rlx graph vs inline reference ──");
    println!("elements = {}", got.len());
    println!("finite   = {finite}");
    println!("cosine   = {cos:.8}");
    println!("max|err| = {maxerr:.3e}");
    if finite && cos > 0.999999 && maxerr < 1e-3 {
        println!("✅ Mamba-2 mixer wiring matches the reference");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "mixer mismatch: cos={cos:.8} maxerr={maxerr:.3e}"
        ))
    }
}

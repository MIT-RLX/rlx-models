// RLX — versatile ML compiler + runtime. GPLv3.
//! Isolation check for the **absorbed-MLA** path ([`build_deepseek_mla`] with
//! `absorbed_mla=true`) — the novel piece that enables **glm_moe_dsa** (GLM-5 /
//! DeepSeek-V3.2), whose checkpoints store per-head `embed_q`/`unembed_out`
//! (MultiLinear) instead of a single `kv_b_proj`. GLM-5 is a 78L/256-expert
//! giant (unvalidatable e2e), so this proves the absorbed projections are
//! correct by an ALGEBRAIC-EQUIVALENCE test: absorbed and non-absorbed MLA are
//! identical, so from random `embed_q`/`unembed_out` we build the equivalent
//! `kv_b_proj` and assert `build_deepseek_mla(absorbed=true)` matches
//! `build_deepseek_mla(absorbed=false)` — reusing the already-validated
//! (DeepSeek-V2-Lite cos 0.999995) q-LoRA/RoPE/attention tail, so the ONLY thing
//! under test is the embed_q/unembed_out → k_nope/value reshaping.
//!
//!   cargo run --release -p rlx-models-core --example glm_dsa_mla_probe

use anyhow::Result;
use rlx_ir::{DType, Graph, Shape};
use rlx_models_core::standard_decoder::{DeepseekSpec, RopeScaling, build_deepseek_mla};
use rlx_models_core::weight_loader::WeightLoader;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    (x - x.floor()) as f32 - 0.5
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

fn spec(
    h: usize,
    nope: usize,
    rope: usize,
    vd: usize,
    kv_lora: usize,
    q_lora: usize,
    hidden: usize,
    absorbed: bool,
) -> DeepseekSpec {
    DeepseekSpec {
        vocab_size: 16,
        hidden_size: hidden,
        num_hidden_layers: 1,
        num_attention_heads: h,
        q_lora_rank: q_lora,
        absorbed_mla: absorbed,
        kv_lora_rank: kv_lora,
        qk_nope_head_dim: nope,
        qk_rope_head_dim: rope,
        v_head_dim: vd,
        intermediate_size: 0,
        moe_intermediate_size: 0,
        n_routed_experts: 0,
        num_experts_per_tok: 0,
        n_shared_experts: 0,
        first_k_dense_replace: 0,
        routed_scaling_factor: 1.0,
        norm_topk_prob: false,
        sigmoid_gate: true,
        sqrtsoftplus_gate: false,
        swiglu_limit: 0.0,
        rope_theta: 10000.0,
        rope_scaling: RopeScaling::None,
        attn_score_scale: None,
        rope_neox: false, // GLM-5 / DSV32 use traditional=True → GptJ
        rms_norm_eps: 1e-5,
    }
}

fn run(
    loader: &mut MemLoader,
    sp: &DeepseekSpec,
    x: &[f32],
    seq: usize,
    hidden: usize,
    rope: usize,
) -> Vec<f32> {
    let mut g = Graph::new("mla_probe");
    let xin = g.input("x", Shape::new(&[1, seq, hidden], DType::F32));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
        HashMap::new();
    // Shared rope tables [seq, rope/2].
    let half = rope / 2;
    let mut cd = vec![0f32; seq * half];
    let mut sd = vec![0f32; seq * half];
    for p in 0..seq {
        for i in 0..half {
            let fr = 10000f64.powf(-(2.0 * i as f64) / rope as f64);
            let (s, c) = (p as f64 * fr).sin_cos();
            cd[p * half + i] = c as f32;
            sd[p * half + i] = s as f32;
        }
    }
    let cos = g.param("cos", Shape::new(&[seq, half], DType::F32));
    params.insert("cos".into(), cd);
    let sin = g.param("sin", Shape::new(&[seq, half], DType::F32));
    params.insert("sin".into(), sd);
    let out = build_deepseek_mla(
        &mut g,
        &mut params,
        &mut packed,
        loader,
        "model.layers.0",
        xin,
        cos,
        sin,
        1,
        seq,
        sp,
    )
    .unwrap();
    g.set_outputs(vec![out]);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    compiled.run(&[("x", x)]).into_iter().next().unwrap()
}

fn main() -> Result<()> {
    let (h, nope, rope, vd, kv_lora, q_lora, hidden, seq) = (2usize, 6, 4, 6, 12, 8, 10, 4);
    let qk = nope + rope;
    let sa = "model.layers.0.self_attn";
    let mk =
        |seed: f64, n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| s * rnd(seed, i)).collect() };

    // Shared q-LoRA / kv_a / o weights.
    let q_a = mk(1.0, q_lora * hidden, 0.3);
    let q_a_ln = mk(2.0, q_lora, 0.1)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>();
    let q_b = mk(3.0, h * qk * q_lora, 0.3);
    let kv_a = mk(4.0, (kv_lora + rope) * hidden, 0.3);
    let kv_a_ln = mk(5.0, kv_lora, 0.1)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>();
    let o_w = mk(6.0, hidden * h * vd, 0.3);
    // Absorbed weights: embed_q [h, kv_lora, nope], unembed_out [h, vd, kv_lora].
    let embed_q = mk(7.0, h * kv_lora * nope, 0.3);
    let unembed_out = mk(8.0, h * vd * kv_lora, 0.3);
    // Equivalent kv_b_proj [h*(nope+vd), kv_lora]: rows [h][0:nope]=embed_q[h]^T, [nope:nope+vd]=unembed_out[h].
    let mut kv_b = vec![0f32; h * (nope + vd) * kv_lora];
    for hd in 0..h {
        for r in 0..nope {
            for c in 0..kv_lora {
                // embed_q[hd, c, r]  (embed_q[h] is [kv_lora, nope])
                kv_b[(hd * (nope + vd) + r) * kv_lora + c] = embed_q[(hd * kv_lora + c) * nope + r];
            }
        }
        for j in 0..vd {
            for c in 0..kv_lora {
                // unembed_out[hd, j, c]  (unembed_out[h] is [vd, kv_lora])
                kv_b[(hd * (nope + vd) + nope + j) * kv_lora + c] =
                    unembed_out[(hd * vd + j) * kv_lora + c];
            }
        }
    }

    let base = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>| {
        t.insert(
            format!("{sa}.q_a_proj.weight"),
            (q_a.clone(), vec![q_lora, hidden]),
        );
        t.insert(
            format!("{sa}.q_a_layernorm.weight"),
            (q_a_ln.clone(), vec![q_lora]),
        );
        t.insert(
            format!("{sa}.q_b_proj.weight"),
            (q_b.clone(), vec![h * qk, q_lora]),
        );
        t.insert(
            format!("{sa}.kv_a_proj_with_mqa.weight"),
            (kv_a.clone(), vec![kv_lora + rope, hidden]),
        );
        t.insert(
            format!("{sa}.kv_a_layernorm.weight"),
            (kv_a_ln.clone(), vec![kv_lora]),
        );
        t.insert(
            format!("{sa}.o_proj.weight"),
            (o_w.clone(), vec![hidden, h * vd]),
        );
    };
    let mut ta: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    base(&mut ta);
    ta.insert(
        format!("{sa}.embed_q.weight"),
        (embed_q.clone(), vec![h, kv_lora, nope]),
    );
    ta.insert(
        format!("{sa}.unembed_out.weight"),
        (unembed_out.clone(), vec![h, vd, kv_lora]),
    );
    let mut tb: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    base(&mut tb);
    tb.insert(
        format!("{sa}.kv_b_proj.weight"),
        (kv_b, vec![h * (nope + vd), kv_lora]),
    );

    let x: Vec<f32> = (0..seq * hidden).map(|i| 0.5 * rnd(9.9, i)).collect();
    let got_absorbed = run(
        &mut MemLoader { t: ta },
        &spec(h, nope, rope, vd, kv_lora, q_lora, hidden, true),
        &x,
        seq,
        hidden,
        rope,
    );
    let got_kvb = run(
        &mut MemLoader { t: tb },
        &spec(h, nope, rope, vd, kv_lora, q_lora, hidden, false),
        &x,
        seq,
        hidden,
        rope,
    );

    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in got_absorbed.iter().zip(&got_kvb) {
        dot += *a as f64 * *b as f64;
        na += *a as f64 * *a as f64;
        nb += *b as f64 * *b as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    let maxerr = got_absorbed
        .iter()
        .zip(&got_kvb)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let finite = got_absorbed.iter().all(|v| v.is_finite());
    println!("── absorbed-MLA (embed_q/unembed_out) vs validated kv_b_proj path ──");
    println!("elements = {}  finite = {finite}", got_absorbed.len());
    println!("cosine   = {cos:.8}");
    println!("max|err| = {maxerr:.3e}");
    if finite && cos > 0.999999 && maxerr < 1e-3 {
        println!("✅ absorbed MLA matches the kv_b_proj path (glm_moe_dsa attention correct)");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "absorbed-MLA mismatch: cos={cos:.8} maxerr={maxerr:.3e}"
        ))
    }
}

// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: the full KimiLinear text decoder (hybrid KDA/MLA layers +
//! Attention Residuals + dense/MoE FFNs + final norm + lm_head) on a tiny
//! synthetic config; finite logits. Runs on CPU (KDA uses the per-channel
//! GatedDeltaNet, which is CPU-only for now).

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::flow::{
    AttnWeights, FfnWeights, FlowConfig, FlowWeights, LayerWeights, build_kimi_text_flow,
};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights};
use rlx_kimi_k3::mla::{MlaDims, MlaWeights};
use rlx_kimi_k3::moe::{DenseMlpWeights, MoeDims, MoeWeights};
use rlx_runtime::Device;

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("coreml") | Some("ane") => Device::Ane,
        Some("cuda") => Device::Cuda,
        Some("vulkan") | Some("vk") => Device::Vulkan,
        _ => Device::Cpu,
    }
}
use std::collections::HashMap;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.15
        })
        .collect()
}

fn kda_w(d: KdaDims, sd: u64) -> KdaWeights {
    let (hidden, h, hd, proj, k) = (d.hidden, d.num_heads, d.head_dim, d.proj(), d.conv_kernel);
    KdaWeights {
        q_proj: fill(hidden * proj, sd + 1),
        k_proj: fill(hidden * proj, sd + 2),
        v_proj: fill(hidden * proj, sd + 3),
        q_conv: fill(proj * k, sd + 4),
        k_conv: fill(proj * k, sd + 5),
        v_conv: fill(proj * k, sd + 6),
        f_a: fill(hidden * hd, sd + 7),
        f_b: fill(hd * proj, sd + 8),
        dt_bias: fill(proj, sd + 9),
        a_log: fill(hd, sd + 10),
        b_proj: fill(hidden * h, sd + 11),
        g_proj: fill(hidden * proj, sd + 12),
        o_norm: vec![1.0; hd],
        o_proj: fill(proj * hidden, sd + 13),
    }
}

fn mla_w(d: MlaDims, sd: u64) -> MlaWeights {
    let (hidden, h, ql, kvl, nope, rope, vd, qk) = (
        d.hidden,
        d.num_heads,
        d.q_lora_rank,
        d.kv_lora_rank,
        d.qk_nope_head_dim,
        d.qk_rope_head_dim,
        d.v_head_dim,
        d.qk(),
    );
    MlaWeights {
        q_a_proj: fill(hidden * ql, sd + 1),
        q_a_layernorm: vec![1.0; ql],
        q_b_proj: fill(ql * h * qk, sd + 2),
        kv_a_proj_with_mqa: fill(hidden * (kvl + rope), sd + 3),
        kv_a_layernorm: vec![1.0; kvl],
        kv_b_proj: fill(kvl * h * (nope + vd), sd + 4),
        g_proj: fill(hidden * h * vd, sd + 5),
        o_proj: fill(h * vd * hidden, sd + 6),
    }
}

fn moe_w(d: MoeDims, sd: u64) -> MoeWeights {
    let (hidden, l, mi, e, si) = (
        d.hidden,
        d.latent,
        d.moe_inter,
        d.num_experts,
        d.num_shared * d.moe_inter,
    );
    MoeWeights {
        router: fill(hidden * e, sd + 1),
        e_score_bias: fill(e, sd + 2),
        down_latent: fill(hidden * l, sd + 3),
        up_latent: fill(l * hidden, sd + 4),
        routed_norm: vec![1.0; l],
        experts_gate_up: fill(e * l * 2 * mi, sd + 5),
        experts_down: fill(e * mi * l, sd + 6),
        shared_gate: fill(hidden * si, sd + 7),
        shared_up: fill(hidden * si, sd + 8),
        shared_down: fill(si * hidden, sd + 9),
    }
}

fn layer(hidden: usize, attn: AttnWeights, ffn: FfnWeights, sd: u64) -> LayerWeights {
    LayerWeights {
        input_ln: vec![1.0; hidden],
        post_ln: vec![1.0; hidden],
        sa_res_norm: vec![1.0; hidden],
        sa_res_proj: fill(hidden, sd + 1),
        mlp_res_norm: vec![1.0; hidden],
        mlp_res_proj: fill(hidden, sd + 2),
        attn,
        ffn,
    }
}

#[test]
fn kimi_text_flow_compiles_and_runs() {
    let (batch, seq, hidden, vocab) = (1usize, 3usize, 16usize, 20usize);
    let kda = KdaDims {
        hidden,
        num_heads: 2,
        head_dim: 8,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch,
        seq,
    };
    let mla = MlaDims {
        hidden,
        num_heads: 2,
        q_lora_rank: 8,
        kv_lora_rank: 6,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 2,
        v_head_dim: 4,
        eps: 1e-5,
        batch,
        seq,
    };
    let moe = MoeDims {
        hidden,
        latent: 12,
        moe_inter: 8,
        num_experts: 4,
        top_k: 2,
        num_shared: 1,
        routed_scaling: 1.0,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch,
        seq,
    };
    let dense_inter = 24usize;

    // 4 layers, block_size 2: L0 KDA+dense (first_k_dense_replace=1), L1 KDA+MoE,
    // L2 MLA+MoE, L3 KDA+MoE. Boundaries at layers 0 and 2.
    let dw = DenseMlpWeights {
        gate: fill(hidden * dense_inter, 900),
        up: fill(hidden * dense_inter, 901),
        down: fill(dense_inter * hidden, 902),
    };
    let layers = vec![
        layer(
            hidden,
            AttnWeights::Kda(Box::new(kda_w(kda, 100))),
            FfnWeights::Dense(Box::new(dw)),
            10,
        ),
        layer(
            hidden,
            AttnWeights::Kda(Box::new(kda_w(kda, 200))),
            FfnWeights::Moe(Box::new(moe_w(moe, 200))),
            20,
        ),
        layer(
            hidden,
            AttnWeights::Mla(Box::new(mla_w(mla, 300))),
            FfnWeights::Moe(Box::new(moe_w(moe, 300))),
            30,
        ),
        layer(
            hidden,
            AttnWeights::Kda(Box::new(kda_w(kda, 400))),
            FfnWeights::Moe(Box::new(moe_w(moe, 400))),
            40,
        ),
    ];
    let w = FlowWeights {
        layers,
        final_norm: vec![1.0; hidden],
        out_res_norm: vec![1.0; hidden],
        out_res_proj: fill(hidden, 800),
        lm_head: fill(hidden * vocab, 801),
    };
    let cfg = FlowConfig {
        hidden,
        vocab,
        attn_res_block_size: 2,
        eps: 1e-5,
        kda,
        mla,
        moe,
        dense_inter,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch,
        seq,
    };

    let mut hir = HirModule::new("kimi_text_flow");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[batch, seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let logits = build_kimi_text_flow(&mut g, &mut params, h_in, &w, &cfg).expect("build flow");
    g.set_outputs(vec![logits]);

    let built = built_from_hir(hir, params).expect("build model");
    let mut compiled = compile_built(built, dev()).expect("compile flow");

    let hin = fill(batch * seq * hidden, 7);
    let y = compiled
        .run(&[("h", hin.as_slice())])
        .into_iter()
        .next()
        .expect("flow output");
    assert_eq!(y.len(), batch * seq * vocab);
    assert!(y.iter().all(|v| v.is_finite()), "logits must be finite");
}

// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! REAL-weight forward through the actual Kimi-K3 checkpoint: embed a few tokens
//! from `embed_tokens`, run them through **layer 0** (real KDA attention + dense
//! MLP + Attention-Residuals) via `build_kimi_text_stage`, and check the hidden
//! output is finite + sanely scaled. Also dequantizes one real MXFP4 expert.
//! This exercises the whole real-weight path — `CheckpointLoader`, the BF16/F32
//! tensor layouts/transposes, and the stage builder — on genuine weights.
//! The 114 GB backbone loads densely; the routed experts are paged one at a time
//! (a full 896-expert layer can't be materialized). Skips if unmounted.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::config::KimiK3Config;
use rlx_kimi_k3::flow::{
    AttnWeights, FfnWeights, FlowConfig, FlowWeights, LayerWeights, build_kimi_text_stage,
};
use rlx_kimi_k3::kda::KdaDims;
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::mla::MlaDims;
use rlx_kimi_k3::moe::{DenseMlpWeights, MoeDims};
use rlx_kimi_k3::runner::{run_moe_paged, run_prefix_streaming};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::Path;

const CKPT: &str = "/Volumes/FOUR/kimi";

fn kda_dims(seq: usize) -> KdaDims {
    KdaDims {
        hidden: 7168,
        num_heads: 96,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq,
    }
}

#[test]
fn real_layer0_stage_forward_finite() {
    if !Path::new(CKPT).exists() {
        eprintln!("skip: {CKPT} not mounted");
        return;
    }
    let mut ck = CheckpointLoader::open(CKPT).expect("open checkpoint");
    let hidden = 7168usize;
    let dense_inter = 33792usize;
    let tokens: Vec<u32> = vec![1, 100, 5000];
    let seq = tokens.len();
    let lp = "language_model.model.layers.0";

    // ── embed the tokens (gather rows from the real BF16 table) ──
    let h_in_data = ck
        .gather_embed("language_model.model.embed_tokens.weight", &tokens, hidden)
        .expect("embed");

    // ── load layer 0: KDA attention + dense MLP + norms + AttnRes projections ──
    let kda = ck.load_kda(lp, kda_dims(seq)).expect("kda weights");
    let dense = ck
        .load_dense_mlp(lp, hidden, dense_inter)
        .expect("dense mlp");
    let mut ln = |n: &str| ck.tensor_f32(&format!("{lp}.{n}")).unwrap();
    let layer = LayerWeights {
        input_ln: ln("input_layernorm.weight"),
        post_ln: ln("post_attention_layernorm.weight"),
        sa_res_norm: ln("self_attention_res_norm.weight"),
        sa_res_proj: ln("self_attention_res_proj.weight"), // [1,hidden] → hidden
        mlp_res_norm: ln("mlp_res_norm.weight"),
        mlp_res_proj: ln("mlp_res_proj.weight"),
        attn: AttnWeights::Kda(Box::new(kda)),
        ffn: FfnWeights::Dense(Box::new(DenseMlpWeights {
            gate: dense.gate,
            up: dense.up,
            down: dense.down,
        })),
    };

    // dims for the (unused-here) MLA/MoE cores the config still needs
    let mla = MlaDims {
        hidden,
        num_heads: 96,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        eps: 1e-5,
        batch: 1,
        seq,
    };
    let moe = MoeDims {
        hidden,
        latent: 3584,
        moe_inter: 3072,
        num_experts: 896,
        top_k: 16,
        num_shared: 2,
        routed_scaling: 1.0,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq,
    };
    let cfg = FlowConfig {
        hidden,
        vocab: 163840,
        attn_res_block_size: 12,
        eps: 1e-5,
        kda: kda_dims(seq),
        mla,
        moe,
        dense_inter,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq,
    };
    // dummy head (unused: last = false → returns hidden, not logits)
    let w = FlowWeights {
        layers: vec![layer],
        final_norm: vec![1.0; hidden],
        out_res_norm: vec![1.0; hidden],
        out_res_proj: vec![0.0; hidden],
        lm_head: vec![0.0; hidden], // unused
    };

    // ── build the 1-layer stage (hidden-out) and run the REAL forward ──
    let mut hir = HirModule::new("real_stage");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[1, seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let (out, _snaps) = build_kimi_text_stage(
        &mut g,
        &mut params,
        h_in,
        Vec::new(),
        &w.layers,
        0,
        false,
        &w,
        &cfg,
    )
    .expect("build stage");
    g.set_outputs(vec![out]);
    let built = built_from_hir(hir, params).expect("build model");
    let mut compiled = compile_built(built, Device::Cpu).expect("compile");
    let y = compiled.run(&[("h", h_in_data.as_slice())]).remove(0);

    assert_eq!(y.len(), seq * hidden);
    let finite = y.iter().all(|v| v.is_finite());
    let maxabs = y.iter().fold(0f32, |m, &v| m.max(v.abs()));
    eprintln!("real layer-0 stage forward: [{seq},{hidden}] finite={finite} max|h|={maxabs:.4}");
    assert!(finite, "hidden must be finite");
    assert!(
        maxabs > 0.0 && maxabs < 1e4,
        "hidden sanely scaled (max {maxabs})"
    );
}

#[test]
fn real_expert_dequant_mxfp4_finite() {
    if !Path::new(CKPT).exists() {
        eprintln!("skip: {CKPT} not mounted");
        return;
    }
    let mut ck = CheckpointLoader::open(CKPT).expect("open");
    // Layer 1 is a MoE layer; dequant expert 0's gate_up + down (MXFP4).
    let (l, mi) = (3584usize, 3072usize);
    let (gate_up, down) = ck
        .load_expert("language_model.model.layers.1", 0, l, mi)
        .expect("expert 0");
    assert_eq!(gate_up.len(), l * 2 * mi);
    assert_eq!(down.len(), mi * l);
    let ok = |v: &[f32]| v.iter().all(|x| x.is_finite());
    let maxabs = gate_up
        .iter()
        .chain(&down)
        .fold(0f32, |m, &v| m.max(v.abs()));
    let nz = gate_up.iter().filter(|&&v| v != 0.0).count();
    eprintln!(
        "MXFP4 expert 0: finite={} max|w|={maxabs:.4} nonzero={nz}/{}",
        ok(&gate_up) && ok(&down),
        gate_up.len()
    );
    assert!(ok(&gate_up) && ok(&down), "dequant must be finite");
    assert!(
        maxabs > 0.0 && maxabs < 100.0,
        "MXFP4 values sane (max {maxabs})"
    );
    assert!(nz > gate_up.len() / 10, "not mostly zero");
}

#[test]
fn real_moe_layer_paged_forward_finite() {
    if !Path::new(CKPT).exists() {
        eprintln!("skip: {CKPT} not mounted");
        return;
    }
    let mut ck = CheckpointLoader::open(CKPT).expect("open");
    let hidden = 7168usize;
    // A plausible hidden state (one real embedding row); MoE layer 1, paged.
    let h_in = ck
        .gather_embed("language_model.model.embed_tokens.weight", &[5000], hidden)
        .expect("embed");
    let d = MoeDims {
        hidden,
        latent: 3584,
        moe_inter: 3072,
        num_experts: 896,
        top_k: 16,
        num_shared: 2,
        routed_scaling: 2.5,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq: 1,
    };
    let out = run_moe_paged(
        &mut ck,
        "language_model.model.layers.1",
        &h_in,
        d,
        Device::Cpu,
    )
    .expect("paged moe");
    assert_eq!(out.len(), hidden);
    let finite = out.iter().all(|v| v.is_finite());
    let maxabs = out.iter().fold(0f32, |m, &v| m.max(v.abs()));
    eprintln!("real MoE layer-1 PAGED forward: [{hidden}] finite={finite} max|out|={maxabs:.4}");
    assert!(finite, "paged MoE output must be finite");
    assert!(maxabs > 0.0 && maxabs < 1e4, "sanely scaled (max {maxabs})");
}

#[test]
fn real_streaming_prefix_finite() {
    if !Path::new(CKPT).exists() {
        eprintln!("skip: {CKPT} not mounted");
        return;
    }
    let mut ck = CheckpointLoader::open(CKPT).expect("open");
    let kc = KimiK3Config::load(format!("{CKPT}/config.json")).expect("config");
    let tc = &kc.text_config;
    let hidden = 7168usize;
    let seq = 1usize;
    let mla = MlaDims {
        hidden,
        num_heads: 96,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        eps: 1e-5,
        batch: 1,
        seq,
    };
    let moe = MoeDims {
        hidden,
        latent: 3584,
        moe_inter: 3072,
        num_experts: 896,
        top_k: 16,
        num_shared: 2,
        routed_scaling: 2.5,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq,
    };
    let cfg = FlowConfig {
        hidden,
        vocab: 163840,
        attn_res_block_size: tc.attn_res_block_size.unwrap_or(12),
        eps: 1e-5,
        kda: kda_dims(seq),
        mla,
        moe,
        dense_inter: 33792,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq,
    };
    // Stream the first 3 real layers (dense L0 + paged-MoE L1/L2).
    let tokens = vec![5000u32];
    let n_layers = 3;
    let (h, _snaps) = run_prefix_streaming(&mut ck, tc, &cfg, &tokens, n_layers, Device::Cpu)
        .expect("streaming forward");
    assert_eq!(h.len(), seq * hidden);
    let finite = h.iter().all(|v| v.is_finite());
    let maxabs = h.iter().fold(0f32, |m, &v| m.max(v.abs()));
    eprintln!(
        "REAL streaming forward through {n_layers} layers: [{seq},{hidden}] finite={finite} max|h|={maxabs:.4}"
    );
    assert!(finite, "streamed hidden must be finite");
    assert!(maxabs > 0.0 && maxabs < 1e5, "sanely scaled (max {maxabs})");
}

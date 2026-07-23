// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Compile + run a tiny MoE LM on each available RLX backend.
//!
//! ```bash
//! cargo test -p rlx-unlimited-ocr --test backend_quick_check --features apple-silicon
//! just features=apple-silicon test-unlimited-ocr-backends
//! ```

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::Device;
use rlx_unlimited_ocr::compile_support::{lm_runtime_guard_for_pack, metal_lm_compile_guard};
use rlx_unlimited_ocr::config::{
    ClipTowerConfig, ProjectorConfig, SamTowerConfig, UnlimitedOcrConfig, UnlimitedOcrVisionConfig,
};
use rlx_unlimited_ocr::expert_pack::{
    PackedLmWeights, expert_down_exps_key, expert_gate_exps_key, expert_up_exps_key,
    pack_experts_in_map,
};
use rlx_unlimited_ocr::lm_graph::{
    build_unlimited_ocr_decode_built, build_unlimited_ocr_prefill_built,
    build_unlimited_ocr_prefill_built_from_pack, compute_rope_slice,
};
use rlx_unlimited_ocr::lm_precision::ResolvedLmPrecision;
use rlx_unlimited_ocr::resolve_device;
use rlx_unlimited_ocr::weights::UnlimitedOcrWeightPrefix;
use std::collections::HashMap;
use std::sync::Arc;

fn tiny_cfg() -> UnlimitedOcrConfig {
    UnlimitedOcrConfig {
        model_type: "unlimited-ocr".into(),
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        n_routed_experts: 4,
        n_shared_experts: 2,
        num_experts_per_tok: 2,
        moe_intermediate_size: 32,
        intermediate_size: 64,
        first_k_dense_replace: 1,
        vocab_size: 128,
        max_position_embeddings: 256,
        sliding_window: 16,
        use_mla: false,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_act: "silu".into(),
        bos_token_id: 0,
        eos_token_id: 1,
        pad_token_id: 2,
        image_token_id: 3,
        v_head_dim: Some(16),
        vision_config: UnlimitedOcrVisionConfig {
            sam: SamTowerConfig::default(),
            clip: ClipTowerConfig::default(),
            image_size: 1024,
        },
        projector: ProjectorConfig {
            input_dim: 2048,
            n_embed: 64,
            projector_type: "linear".into(),
        },
        patch_size: 16,
        downsample_ratio: 4,
    }
}

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.017 + seed).sin()) * 0.02)
        .collect()
}

fn synthetic_lm_weights(cfg: &UnlimitedOcrConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let ff_dense = cfg.intermediate_size;
    let moe_ff = cfg.moe_intermediate_size;
    let n_e = cfg.n_routed_experts;
    let shared_ff = moe_ff * cfg.n_shared_experts;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    t.insert(
        UnlimitedOcrWeightPrefix::embed_tokens().into(),
        (fill(v * h, 0.1), vec![v, h]),
    );
    t.insert(
        UnlimitedOcrWeightPrefix::lm_norm().into(),
        (fill(h, 0.2), vec![h]),
    );
    t.insert(
        UnlimitedOcrWeightPrefix::lm_head().into(),
        (fill(v * h, 0.3), vec![v, h]),
    );

    for layer in 0..cfg.num_hidden_layers {
        t.insert(
            UnlimitedOcrWeightPrefix::lm_input_layernorm(layer),
            (fill(h, 1.0 + layer as f32), vec![h]),
        );
        t.insert(
            UnlimitedOcrWeightPrefix::lm_post_attention_layernorm(layer),
            (fill(h, 2.0 + layer as f32), vec![h]),
        );
        for (pi, proj) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
            t.insert(
                UnlimitedOcrWeightPrefix::lm_attn(layer, proj),
                (fill(h * h, 3.0 + pi as f32), vec![h, h]),
            );
        }
        if cfg.is_dense_layer(layer) {
            for (pi, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
                t.insert(
                    UnlimitedOcrWeightPrefix::lm_dense_mlp(layer, proj),
                    (fill(ff_dense * h, 4.0 + pi as f32), vec![ff_dense, h]),
                );
            }
            t.insert(
                UnlimitedOcrWeightPrefix::lm_dense_mlp(layer, "down_proj"),
                (fill(h * ff_dense, 4.5), vec![h, ff_dense]),
            );
        } else {
            t.insert(
                UnlimitedOcrWeightPrefix::lm_moe_gate(layer),
                (fill(n_e * h, 5.0), vec![n_e, h]),
            );
            for (pi, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
                t.insert(
                    UnlimitedOcrWeightPrefix::lm_moe_shared_expert(layer, proj),
                    (fill(shared_ff * h, 6.0 + pi as f32), vec![shared_ff, h]),
                );
            }
            t.insert(
                UnlimitedOcrWeightPrefix::lm_moe_shared_expert(layer, "down_proj"),
                (fill(h * shared_ff, 6.5), vec![h, shared_ff]),
            );
            for e in 0..n_e {
                for (pi, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
                    t.insert(
                        UnlimitedOcrWeightPrefix::lm_moe_expert(layer, e, proj),
                        (
                            fill(moe_ff * h, 7.0 + e as f32 + pi as f32 * 0.1),
                            vec![moe_ff, h],
                        ),
                    );
                }
                t.insert(
                    UnlimitedOcrWeightPrefix::lm_moe_expert(layer, e, "down_proj"),
                    (fill(h * moe_ff, 8.0 + e as f32), vec![h, moe_ff]),
                );
            }
        }
    }

    let mut map = WeightMap::from_tensors(t);
    for layer in 0..cfg.num_hidden_layers {
        if !cfg.is_dense_layer(layer) {
            pack_experts_in_map(&mut map, layer, n_e, h, moe_ff).expect("pack experts");
            assert!(map.has(&expert_gate_exps_key(layer)));
            assert!(map.has(&expert_up_exps_key(layer)));
            assert!(map.has(&expert_down_exps_key(layer)));
        }
    }
    map
}

fn run_tiny_moe_on_device(device: Device) {
    let cfg = tiny_cfg();
    cfg.validate().expect("tiny cfg");
    let seq = 4usize;
    let h = cfg.hidden_size;
    let embeds: Vec<f32> = fill(seq * h, 0.5);

    let mut wm = synthetic_lm_weights(&cfg);
    let built = build_unlimited_ocr_prefill_built(&cfg, &mut wm, 1, seq).expect("prefill build");
    let mut compiled =
        metal_lm_compile_guard(device, || compile_built(built, device)).expect("compile");
    let outs = metal_lm_compile_guard(device, || {
        compiled.run(&[("inputs_embeds", embeds.as_slice())])
    });
    assert!(!outs.is_empty(), "prefill outputs");
    assert_eq!(outs[0].len(), cfg.vocab_size);
    assert!(outs[0].iter().all(|v| v.is_finite()));

    // Decode one step with past from prefill KV side outputs.
    let n_layers = cfg.num_hidden_layers;
    assert_eq!(outs.len(), 1 + 2 * n_layers);
    let past_seq = seq;
    let mut wm2 = synthetic_lm_weights(&cfg);
    let built_d =
        build_unlimited_ocr_decode_built(&cfg, &mut wm2, 1, past_seq).expect("decode build");
    let mut compiled_d =
        metal_lm_compile_guard(device, || compile_built(built_d, device)).expect("decode compile");

    let step = fill(h, 0.9);
    let (cos, sin) = compute_rope_slice(&cfg, past_seq);
    let mut pairs: Vec<(&str, &[f32])> = vec![
        ("inputs_embeds", step.as_slice()),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
    ];
    let past_owned: Vec<(String, Vec<f32>)> = (0..n_layers)
        .flat_map(|i| {
            [
                (format!("past_k_{i}"), outs[1 + 2 * i].clone()),
                (format!("past_v_{i}"), outs[1 + 2 * i + 1].clone()),
            ]
        })
        .collect();
    for (n, d) in &past_owned {
        pairs.push((n.as_str(), d.as_slice()));
    }
    let dout = metal_lm_compile_guard(device, || compiled_d.run(&pairs));
    assert_eq!(dout[0].len(), cfg.vocab_size);
    assert!(dout[0].iter().all(|v| v.is_finite()));
}

fn run_tiny_moe_q8_packed_ir(device: Device) {
    run_tiny_moe_quant_packed_ir(device, ResolvedLmPrecision::Q8_0);
}

fn run_tiny_moe_q4_packed_ir(device: Device) {
    run_tiny_moe_quant_packed_ir(device, ResolvedLmPrecision::Q4_0);
}

fn run_tiny_moe_quant_packed_ir(device: Device, prec: ResolvedLmPrecision) {
    let cfg = tiny_cfg();
    cfg.validate().expect("tiny cfg");
    let seq = 4usize;
    let h = cfg.hidden_size;
    let embeds: Vec<f32> = fill(seq * h, 0.5);
    let label = prec.as_str();

    let mut wm = synthetic_lm_weights(&cfg);
    let pack =
        Arc::new(PackedLmWeights::from_weight_map(&mut wm, cfg.clone(), prec).expect("quant pack"));
    assert!(pack.keeps_quants_in_ir());
    // Soft Q4: lm_head stays F16 (no U8 IR blob); experts stay packed.
    if prec == ResolvedLmPrecision::Q4_0 {
        let head = UnlimitedOcrWeightPrefix::lm_head();
        assert!(
            pack.ir_mat_blob(&head, false)
                .expect("lm_head blob")
                .is_none(),
            "Q4 soft-pack should keep lm_head as F16 IR (not U8)"
        );
        assert!(
            pack.ir_mat_blob(&expert_gate_exps_key(1), false)
                .expect("gate_exps blob")
                .is_some(),
            "Q4 soft-pack should keep routed experts as U8 typed params"
        );
    }

    let built = build_unlimited_ocr_prefill_built_from_pack(&cfg, &pack, 1, seq)
        .unwrap_or_else(|e| panic!("{label} prefill build: {e:#}"));
    assert!(
        !built.typed_params.is_empty(),
        "{label} pack should attach U8 typed params"
    );
    assert!(
        !built.params.contains_key(&expert_gate_exps_key(1)),
        "gate_exps should be typed U8, not F32 params"
    );

    let mut compiled = lm_runtime_guard_for_pack(device, &pack, || compile_built(built, device))
        .unwrap_or_else(|e| panic!("{label} compile: {e:#}"));
    let outs = lm_runtime_guard_for_pack(device, &pack, || {
        compiled.run(&[("inputs_embeds", embeds.as_slice())])
    });
    assert_eq!(outs[0].len(), cfg.vocab_size);
    assert!(outs[0].iter().all(|v| v.is_finite()));

    let n_layers = cfg.num_hidden_layers;
    assert_eq!(outs.len(), 1 + 2 * n_layers);
    let past_seq = seq;
    let built_d = rlx_unlimited_ocr::lm_graph::build_unlimited_ocr_decode_built_from_pack(
        &cfg, &pack, 1, past_seq,
    )
    .unwrap_or_else(|e| panic!("{label} decode build: {e:#}"));
    let mut compiled_d =
        lm_runtime_guard_for_pack(device, &pack, || compile_built(built_d, device))
            .unwrap_or_else(|e| panic!("{label} decode compile: {e:#}"));
    let step = fill(h, 0.9);
    let (cos, sin) = compute_rope_slice(&cfg, past_seq);
    let past_owned: Vec<(String, Vec<f32>)> = (0..n_layers)
        .flat_map(|i| {
            [
                (format!("past_k_{i}"), outs[1 + 2 * i].clone()),
                (format!("past_v_{i}"), outs[1 + 2 * i + 1].clone()),
            ]
        })
        .collect();
    let dout = lm_runtime_guard_for_pack(device, &pack, || {
        let mut pairs: Vec<(&str, &[f32])> = vec![
            ("inputs_embeds", step.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ];
        for (n, d) in &past_owned {
            pairs.push((n.as_str(), d.as_slice()));
        }
        compiled_d.run(&pairs)
    });
    assert_eq!(dout[0].len(), cfg.vocab_size);
    assert!(dout[0].iter().all(|v| v.is_finite()));
}

fn run_q8_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip unlimited-ocr Q8 MoE {device:?}: backend not available");
        return;
    }
    run_tiny_moe_q8_packed_ir(device);
}

fn run_q4_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip unlimited-ocr Q4 MoE {device:?}: backend not available");
        return;
    }
    run_tiny_moe_q4_packed_ir(device);
}

#[test]
fn tiny_moe_q8_packed_ir_cpu() {
    run_q8_if_available(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn tiny_moe_q8_packed_ir_metal() {
    run_q8_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn tiny_moe_q8_packed_ir_mlx() {
    // MLX GroupedMatMul / DequantGrouped routing still mismatches runtime
    // shapes for this MoE pattern (broadcast on top-k weights). Skip until
    // rlx-mlx gather_mm + host dequant path align.
    if !rlx_runtime::is_available(Device::Mlx) {
        eprintln!("skip unlimited-ocr Q8 MoE Mlx: backend not available");
        return;
    }
    eprintln!(
        "skip unlimited-ocr Q8 MoE Mlx: GroupedMatMul/DequantGrouped runtime shape unsupported"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn tiny_moe_q8_packed_ir_wgpu() {
    run_q8_if_available(Device::Gpu);
}

#[cfg(feature = "cuda")]
#[test]
fn tiny_moe_q8_packed_ir_cuda() {
    run_q8_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn tiny_moe_q8_packed_ir_rocm() {
    run_q8_if_available(Device::Rocm);
}

#[cfg(feature = "vulkan")]
#[test]
fn tiny_moe_q8_packed_ir_vulkan() {
    run_q8_if_available(Device::Vulkan);
}

#[test]
fn tiny_moe_q4_packed_ir_cpu() {
    run_q4_if_available(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn tiny_moe_q4_packed_ir_metal() {
    run_q4_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn tiny_moe_q4_packed_ir_mlx() {
    if !rlx_runtime::is_available(Device::Mlx) {
        eprintln!("skip unlimited-ocr Q4 MoE Mlx: backend not available");
        return;
    }
    eprintln!(
        "skip unlimited-ocr Q4 MoE Mlx: GroupedMatMul/DequantGrouped runtime shape unsupported"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn tiny_moe_q4_packed_ir_wgpu() {
    run_q4_if_available(Device::Gpu);
}

#[cfg(feature = "cuda")]
#[test]
fn tiny_moe_q4_packed_ir_cuda() {
    run_q4_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn tiny_moe_q4_packed_ir_rocm() {
    run_q4_if_available(Device::Rocm);
}

#[cfg(feature = "vulkan")]
#[test]
fn tiny_moe_q4_packed_ir_vulkan() {
    run_q4_if_available(Device::Vulkan);
}

/// F32 vs Q8_0 logits should be strongly correlated (BT layout sanity).
#[test]
fn tiny_moe_q8_correlates_with_f32_cpu() {
    assert_quant_correlates_cpu(ResolvedLmPrecision::Q8_0, 0.95);
}

/// Soft Q4 (F16 head/attn + Q4 experts) should still track F32 on a tiny MoE.
#[test]
fn tiny_moe_q4_correlates_with_f32_cpu() {
    assert_quant_correlates_cpu(ResolvedLmPrecision::Q4_0, 0.90);
}

fn assert_quant_correlates_cpu(prec: ResolvedLmPrecision, min_corr: f64) {
    let cfg = tiny_cfg();
    let seq = 4usize;
    let h = cfg.hidden_size;
    let embeds = fill(seq * h, 0.5);

    let mut wm_f = synthetic_lm_weights(&cfg);
    let built_f = build_unlimited_ocr_prefill_built(&cfg, &mut wm_f, 1, seq).unwrap();
    let mut c_f = compile_built(built_f, Device::Cpu).unwrap();
    let f32_logits = c_f.run(&[("inputs_embeds", embeds.as_slice())])[0].clone();

    let mut wm_q = synthetic_lm_weights(&cfg);
    let pack = Arc::new(PackedLmWeights::from_weight_map(&mut wm_q, cfg.clone(), prec).unwrap());
    let built_q = build_unlimited_ocr_prefill_built_from_pack(&cfg, &pack, 1, seq).unwrap();
    let mut c_q =
        lm_runtime_guard_for_pack(Device::Cpu, &pack, || compile_built(built_q, Device::Cpu))
            .unwrap();
    let q_logits = lm_runtime_guard_for_pack(Device::Cpu, &pack, || {
        c_q.run(&[("inputs_embeds", embeds.as_slice())])
    })[0]
        .clone();

    let corr = pearson(&f32_logits, &q_logits);
    assert!(
        corr > min_corr,
        "{} vs F32 logit correlation {corr:.4} too low (expected > {min_corr})",
        prec.as_str()
    );
}

fn pearson(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let ma = a.iter().map(|x| *x as f64).sum::<f64>() / n;
    let mb = b.iter().map(|x| *x as f64).sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..a.len() {
        let xa = a[i] as f64 - ma;
        let xb = b[i] as f64 - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    num / (da.sqrt() * db.sqrt() + 1e-12)
}

fn run_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip unlimited-ocr tiny MoE {device:?}: backend not available");
        return;
    }
    run_tiny_moe_on_device(device);
}

#[test]
fn tiny_moe_cpu() {
    run_if_available(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn tiny_moe_metal() {
    run_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn tiny_moe_mlx() {
    // MLX GroupedMatMul currently lowers to a mismatched runtime rank/size for
    // this MoE pattern (see rlx-mlx); keep the test registered but skip run.
    if !rlx_runtime::is_available(Device::Mlx) {
        eprintln!("skip unlimited-ocr tiny MoE Mlx: backend not available");
        return;
    }
    eprintln!(
        "skip unlimited-ocr tiny MoE Mlx: GroupedMatMul runtime shape unsupported on MLX for this graph"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn tiny_moe_wgpu() {
    run_if_available(Device::Gpu);
}

#[cfg(feature = "cuda")]
#[test]
fn tiny_moe_cuda() {
    run_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn tiny_moe_rocm() {
    run_if_available(Device::Rocm);
}

#[cfg(feature = "vulkan")]
#[test]
fn tiny_moe_vulkan() {
    run_if_available(Device::Vulkan);
}

#[test]
fn resolve_cpu() {
    assert_eq!(resolve_device(Some("cpu")).unwrap(), Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resolve_metal() {
    if rlx_runtime::is_available(Device::Metal) {
        assert_eq!(resolve_device(Some("metal")).unwrap(), Device::Metal);
    }
}

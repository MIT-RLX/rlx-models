// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Cross-backend consistency: run the same graph on CPU and on
//! `RLX_TEST_DEVICE`, per attention branch, and compare.
//!
//! The split matters — a whole-model comparison tells you a backend disagrees
//! but not which block did it. `layer_group_size` selects the attention mix
//! (`> num_hidden_layers` makes every layer MLA via the ragged-tail rule,
//! `== num_hidden_layers` leaves only the last one MLA), and
//! `first_k_dense_replace` takes the MoE out of the picture.
//!
//! Known result (2026-08-10): **all seven tested backends agree to f32 noise**
//! (CPU, Metal, MLX, CoreML, wgpu, CUDA, ROCm). Getting there took three upstream
//! fixes, all of them `Op::FusedSwiGLU::gate_first` being dropped — MLX and
//! CoreML in their own lowerings, and wgpu/CUDA/ROCm via the shared
//! `rlx-unfuse::expand_swiglu`. These tests are what localised each one.
//!
//! CoreML runs fp16 on the ANE and lands at cosine 0.99999297 with identical
//! argmax; the parity test gives `Device::Ane` an fp16 bound.
//!
//! Skips when `RLX_TEST_DEVICE` is unset or `cpu`.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_ling::{LingConfig, build_ling_text_flow, prepare_checkpoint};
use rlx_runtime::Device;
use std::collections::HashMap;

fn alt_device() -> Option<Device> {
    let s = std::env::var("RLX_TEST_DEVICE").ok()?;
    let s = s.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("cpu") {
        return None;
    }
    Some(rlx_cli::parse_device(s).expect("bad RLX_TEST_DEVICE"))
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.1
        })
        .collect()
}

/// 4-layer tiny config; `layer_group_size` picks the attention mix.
fn config(layer_group_size: usize) -> LingConfig {
    LingConfig::from_json_str(&format!(
        r#"{{"vocab_size":32,"hidden_size":16,"intermediate_size":24,"num_hidden_layers":4,
            "num_attention_heads":2,"head_dim":8,"rms_norm_eps":1e-6,"rope_theta":600000.0,
            "num_experts":8,"num_experts_per_tok":2,"num_shared_experts":1,
            "moe_intermediate_size":8,"moe_shared_expert_intermediate_size":8,
            "n_group":2,"topk_group":1,"routed_scaling_factor":2.5,"first_k_dense_replace":1,
            "q_lora_rank":12,"kv_lora_rank":10,"qk_nope_head_dim":8,"qk_rope_head_dim":4,
            "v_head_dim":8,"rope_interleave":true,
            "gated_attention_proj_granularity_type":"head_wise",
            "layer_group_size":{layer_group_size},"short_conv_kernel_size":4,
            "no_kda_lora":true,"kda_safe_gate":true,"kda_lower_bound":-5.0,
            "tie_word_embeddings":false}}"#
    ))
    .expect("parse config")
}

fn weights(cfg: &LingConfig) -> WeightMap {
    use rlx_ling::config::AttnKind;
    let h = cfg.hidden_size;
    let hh = cfg.num_attention_heads;
    let proj = cfg.kda_proj_dim();
    let (hd, qk) = (cfg.head_dim, cfg.qk_head_dim());
    let ql = cfg.q_lora_rank.unwrap();
    let (kvl, rope, nope, vd) = (
        cfg.kv_lora_rank,
        cfg.qk_rope_head_dim,
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
    );
    let mi = cfg.moe_intermediate_size;
    let si = cfg.shared_intermediate_size();
    let e = cfg.num_experts;

    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    // Norm gammas centre on 1 and `A_log` spans the reference init's `log U(1,16)`.
    // Uniformly tiny weights would shrink every activation toward zero, where any
    // two backends agree trivially and the comparison proves nothing.
    let mut put_scaled = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
                          k: String,
                          shape: Vec<usize>,
                          offset: f32,
                          gain: f32| {
        let n: usize = shape.iter().product();
        seed += 3;
        let v = fill(n, seed).iter().map(|x| offset + gain * x).collect();
        t.insert(k, (v, shape));
    };
    macro_rules! put {
        ($t:expr, $k:expr, $s:expr $(,)?) => {
            put_scaled($t, $k, $s, 0.0, 1.0)
        };
    }
    macro_rules! put_norm {
        ($t:expr, $k:expr, $s:expr $(,)?) => {
            put_scaled($t, $k, $s, 1.0, 1.0)
        };
    }
    /// `A_log = log U(1,16)` ⇒ `exp(A_log) ∈ [1,16]`, which saturates the KDA gate.
    macro_rules! put_a_log {
        ($t:expr, $k:expr, $s:expr $(,)?) => {
            put_scaled($t, $k, $s, 2.834, 2.0)
        };
    }

    put!(&mut t, rlx_ling::EMBED_KEY.into(), vec![cfg.vocab_size, h]);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        put_norm!(&mut t, format!("{lp}.input_layernorm.weight"), vec![h]);
        put_norm!(
            &mut t,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![h]
        );
        let at = format!("{lp}.attention");
        match cfg.attn_kind(i) {
            AttnKind::Mla => {
                put!(&mut t, format!("{at}.q_a_proj.weight"), vec![ql, h]);
                put_norm!(&mut t, format!("{at}.q_a_layernorm.weight"), vec![ql]);
                put!(&mut t, format!("{at}.q_b_proj.weight"), vec![hh * qk, ql]);
                put!(
                    &mut t,
                    format!("{at}.kv_a_proj_with_mqa.weight"),
                    vec![kvl + rope, h],
                );
                put_norm!(&mut t, format!("{at}.kv_a_layernorm.weight"), vec![kvl]);
                put!(
                    &mut t,
                    format!("{at}.kv_b_proj.weight"),
                    vec![hh * (nope + vd), kvl],
                );
                put!(&mut t, format!("{at}.g_proj.weight"), vec![hh, h]);
                put!(&mut t, format!("{at}.dense.weight"), vec![h, hh * vd]);
            }
            AttnKind::Kda => {
                for p in ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj"] {
                    put!(&mut t, format!("{at}.{p}.weight"), vec![proj, h]);
                }
                for c in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                    put!(
                        &mut t,
                        format!("{at}.{c}.weight"),
                        vec![proj, 1, cfg.short_conv_kernel_size],
                    );
                }
                put!(&mut t, format!("{at}.b_proj.weight"), vec![hh, h]);
                put_a_log!(&mut t, format!("{at}.A_log"), vec![hh]);
                put!(&mut t, format!("{at}.dt_bias"), vec![proj]);
                put_norm!(&mut t, format!("{at}.o_norm.weight"), vec![hd]);
                put!(&mut t, format!("{at}.o_proj.weight"), vec![h, proj]);
            }
        }
        let mlp = format!("{lp}.mlp");
        if cfg.is_moe_layer(i) {
            put!(&mut t, format!("{mlp}.gate.weight"), vec![e, h]);
            put!(&mut t, format!("{mlp}.gate.expert_bias"), vec![e]);
            for ei in 0..e {
                let b = format!("{mlp}.experts.{ei}");
                put!(&mut t, format!("{b}.gate_proj.weight"), vec![mi, h]);
                put!(&mut t, format!("{b}.up_proj.weight"), vec![mi, h]);
                put!(&mut t, format!("{b}.down_proj.weight"), vec![h, mi]);
            }
            for (p, n) in [("gate_proj", si), ("up_proj", si)] {
                put!(
                    &mut t,
                    format!("{mlp}.shared_experts.{p}.weight"),
                    vec![n, h],
                );
            }
            put!(
                &mut t,
                format!("{mlp}.shared_experts.down_proj.weight"),
                vec![h, si],
            );
        } else {
            let di = cfg.intermediate_size;
            put!(&mut t, format!("{mlp}.gate_proj.weight"), vec![di, h]);
            put!(&mut t, format!("{mlp}.up_proj.weight"), vec![di, h]);
            put!(&mut t, format!("{mlp}.down_proj.weight"), vec![h, di]);
        }
    }
    put_norm!(&mut t, "model.norm.weight".into(), vec![h]);
    put!(&mut t, "lm_head.weight".into(), vec![cfg.vocab_size, h]);
    WeightMap::from_tensors(t)
}

fn run_on(cfg: &LingConfig, seq: usize, device: Device) -> Vec<f32> {
    let mut wm = weights(cfg);
    prepare_checkpoint(cfg, &mut wm).expect("prepare");
    let built = build_ling_text_flow(cfg, &mut wm, seq, true).expect("build");
    let mut compiled = compile_built(built, device).expect("compile");
    let (cos, sin) = cfg.rope_tables(seq);
    let ids: Vec<f32> = (0..seq)
        .map(|i| ((i * 7) % cfg.vocab_size) as f32)
        .collect();
    compiled
        .run(&[
            ("input_ids", ids.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("output")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn compare(label: &str, cfg: &LingConfig, device: Device) {
    let seq = 5;
    let want = run_on(cfg, seq, Device::Cpu);
    let got = run_on(cfg, seq, device);
    let cos = cosine(&got, &want);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("{label}: cosine {cos:.8}, max |Δ| {max_abs:.3e}");
    assert!(
        cos > 0.9999,
        "{label}: {device:?} disagrees with CPU (cosine {cos:.8}, max |Δ| {max_abs:.3e})"
    );
}

/// CPU vs device on the *parity fixture* weights, which use realistic scales
/// (`A_log = log U(1,16)`, unit-centred norms) instead of the uniformly tiny
/// synthetic ones. Saturated KDA decay and a router that actually discriminates
/// only show up here.
#[test]
fn parity_fixture_matches_cpu() {
    let Some(device) = alt_device() else {
        eprintln!("skipping: set RLX_TEST_DEVICE to a non-CPU backend");
        return;
    };
    let Ok(dir) = std::env::var("RLX_LING_PARITY_DIR") else {
        eprintln!("skipping: set RLX_LING_PARITY_DIR (see parity_reference.rs)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = LingConfig::from_file(dir.join("config.json")).expect("config");
    let ids: Vec<u32> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("input_ids.json")).expect("ids"))
            .expect("ids json");
    let ids_f: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let seq = ids.len();
    let (cos_t, sin_t) = cfg.rope_tables(seq);

    let go = |device: Device| {
        let mut wm = WeightMap::from_safetensors_dir(&dir).expect("fixture weights");
        prepare_checkpoint(&cfg, &mut wm).expect("prepare");
        let built = build_ling_text_flow(&cfg, &mut wm, seq, true).expect("build");
        let mut compiled = compile_built(built, device).expect("compile");
        compiled
            .run(&[
                ("input_ids", ids_f.as_slice()),
                ("rope_cos", cos_t.as_slice()),
                ("rope_sin", sin_t.as_slice()),
            ])
            .into_iter()
            .next()
            .expect("output")
    };
    let want = go(Device::Cpu);
    let got = go(device);
    let cos = cosine(&got, &want);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("parity-fixture: cosine {cos:.8}, max |Δ| {max_abs:.3e}");
    assert!(
        cos > 0.9999,
        "{device:?} disagrees with CPU on the fixture (cosine {cos:.8}, max |Δ| {max_abs:.3e})"
    );
}

/// Same stack with the MoE swapped for dense MLPs everywhere. Divergence that
/// survives this is in attention or the norms; divergence that disappears is in
/// the router / `GroupedMatMul` expert path.
#[test]
fn dense_only_matches_cpu() {
    let Some(device) = alt_device() else {
        eprintln!("skipping: set RLX_TEST_DEVICE to a non-CPU backend");
        return;
    };
    let mut cfg = config(4);
    cfg.first_k_dense_replace = cfg.num_hidden_layers;
    assert!((0..4).all(|i| !cfg.is_moe_layer(i)));
    compare("dense-only", &cfg, device);
}

/// **Regression test** for the `rlx-unfuse::expand_swiglu` `gate_first` bug
/// (wgpu / ROCm / CUDA, fixed 2026-08-10). Kept because it is the smallest graph
/// that exposed it, and because the shape of the bug — a fused-kernel flag lost
/// on one lowering path — is easy to reintroduce.
///
/// A single
/// decoder layer — `RMSNorm → MLA → +res → RMSNorm → MoE → +res` — with hidden
/// states as the output (no lm_head).
///
/// This is the smallest graph that fails. Both halves are individually fine on
/// those backends:
///
/// * the same single layer with a **dense MLP** instead of the MoE is exact
///   (cosine 1.00000000) — so attention, the norms and the residuals are fine;
/// * the MoE block on its own, even fed from an RMSNorm, is exact
///   (`moe_block_reference`) — so the router and the expert matmuls are fine.
///
/// It took attention *and* live routed experts in one graph, because the swiglu
/// fusion only fires in a second, post-unfuse fusion round that the
/// attention-free graph never reaches — so the buggy `expand_swiglu` was only
/// exercised in the combined case. Before the fix, wgpu and CUDA gave
/// byte-identical wrong results (cosine 0.99989804), which is what established it
/// as one bug rather than three.
#[test]
fn single_layer_moe_after_attention_matches_cpu() {
    let Some(device) = alt_device() else {
        eprintln!("skipping: set RLX_TEST_DEVICE to a non-CPU backend");
        return;
    };
    let seq = 5;
    let go = |cfg: &LingConfig, device: Device| {
        let mut wm = weights(cfg);
        prepare_checkpoint(cfg, &mut wm).expect("prepare");
        // `with_lm_head = false` → compare hidden states, so a vocab projection
        // can't mask or amplify the difference.
        let built = build_ling_text_flow(cfg, &mut wm, seq, false).expect("build");
        let mut compiled = compile_built(built, device).expect("compile");
        let (cos, sin) = cfg.rope_tables(seq);
        let ids: Vec<f32> = (0..seq)
            .map(|i| ((i * 7) % cfg.vocab_size) as f32)
            .collect();
        compiled
            .run(&[
                ("input_ids", ids.as_slice()),
                ("rope_cos", cos.as_slice()),
                ("rope_sin", sin.as_slice()),
            ])
            .into_iter()
            .next()
            .expect("output")
    };

    // Control: one layer, dense MLP. Expected exact everywhere.
    let mut dense = config(4);
    dense.num_hidden_layers = 1;
    dense.first_k_dense_replace = 1;
    let d_cos = cosine(&go(&dense, device), &go(&dense, Device::Cpu));
    eprintln!("single-layer dense:  cosine {d_cos:.8}");

    // The reproducer: same layer, MoE FFN.
    let mut moe = config(4);
    moe.num_hidden_layers = 1;
    moe.first_k_dense_replace = 0;
    let want = go(&moe, Device::Cpu);
    let got = go(&moe, device);
    let m_cos = cosine(&got, &want);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("single-layer moe:    cosine {m_cos:.8}, max |Δ| {max_abs:.3e}");

    assert!(
        d_cos > 0.9999_999,
        "the dense control itself disagrees on {device:?} (cosine {d_cos:.8}) — \
         the reproducer below is not isolating the MoE"
    );
    assert!(
        m_cos > 0.9999_999,
        "{device:?}: one decoder layer with routed experts disagrees with CPU \
         (cosine {m_cos:.8}, max |Δ| {max_abs:.3e}) while the same layer with a \
         dense MLP is exact — smallest known reproducer"
    );
}

/// Router scores forced far apart, so expert order is unambiguous and no
/// near-tie can flip. `scores_for_routing = σ(logits) + expert_bias` and σ ∈ (0,1),
/// so an `expert_bias` spaced by ≥1 fully determines the ranking.
///
/// This separates "the backend rounds differently" from "the backend is wrong":
/// if a run that cannot tie-flip agrees to f32 noise, the divergence elsewhere is
/// discrete top-k sensitivity, not a broken kernel.
#[test]
fn moe_separated_router_matches_cpu() {
    let Some(device) = alt_device() else {
        eprintln!("skipping: set RLX_TEST_DEVICE to a non-CPU backend");
        return;
    };
    let cfg = config(4);
    let seq = 5;
    let go = |device: Device| {
        let mut wm = weights(&cfg);
        for i in 0..cfg.num_hidden_layers {
            let key = format!("model.layers.{i}.mlp.gate.expert_bias");
            if wm.has(&key) {
                let bias: Vec<f32> = (0..cfg.num_experts).map(|e| e as f32 * 4.0).collect();
                wm.insert(key, bias, vec![cfg.num_experts]);
            }
        }
        prepare_checkpoint(&cfg, &mut wm).expect("prepare");
        let built = build_ling_text_flow(&cfg, &mut wm, seq, true).expect("build");
        let mut compiled = compile_built(built, device).expect("compile");
        let (cos, sin) = cfg.rope_tables(seq);
        let ids: Vec<f32> = (0..seq)
            .map(|i| ((i * 7) % cfg.vocab_size) as f32)
            .collect();
        compiled
            .run(&[
                ("input_ids", ids.as_slice()),
                ("rope_cos", cos.as_slice()),
                ("rope_sin", sin.as_slice()),
            ])
            .into_iter()
            .next()
            .expect("output")
    };
    let want = go(Device::Cpu);
    let got = go(device);
    let cos = cosine(&got, &want);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("moe-separated-router: cosine {cos:.8}, max |Δ| {max_abs:.3e}");
    assert!(
        cos > 0.9999_999,
        "{device:?} disagrees with CPU even with an unambiguous router \
         (cosine {cos:.8}, max |Δ| {max_abs:.3e}) — that is a kernel bug, not tie sensitivity"
    );
}

/// `routed_scaling_factor = 0` zeroes every routed expert's contribution, so the
/// MoE block reduces to its shared expert (plain matmuls). Isolates the
/// `GroupedMatMul` routed path from the rest of the block.
#[test]
fn moe_shared_expert_only_matches_cpu() {
    let Some(device) = alt_device() else {
        eprintln!("skipping: set RLX_TEST_DEVICE to a non-CPU backend");
        return;
    };
    let mut cfg = config(4);
    cfg.routed_scaling_factor = 0.0;
    compare("moe-shared-only", &cfg, device);
}

/// Every expert selected (`top_k == num_experts`, one group), so the routed set
/// is the full set regardless of score order. Divergence that survives is in the
/// expert `GroupedMatMul`; divergence that disappears is in which experts the
/// router picked.
#[test]
fn moe_all_experts_matches_cpu() {
    let Some(device) = alt_device() else {
        eprintln!("skipping: set RLX_TEST_DEVICE to a non-CPU backend");
        return;
    };
    let mut cfg = config(4);
    cfg.num_experts_per_tok = cfg.num_experts;
    cfg.n_group = 1;
    cfg.topk_group = 1;
    compare("moe-all-experts", &cfg, device);
}

#[test]
fn mla_only_matches_cpu() {
    let Some(device) = alt_device() else {
        eprintln!("skipping: set RLX_TEST_DEVICE to a non-CPU backend");
        return;
    };
    // layer_group_size > num_hidden_layers ⇒ every layer falls in the ragged tail.
    let cfg = config(99);
    assert!((0..4).all(|i| cfg.attn_kind(i) == rlx_ling::config::AttnKind::Mla));
    compare("mla-only", &cfg, device);
}

#[test]
fn kda_heavy_matches_cpu() {
    let Some(device) = alt_device() else {
        eprintln!("skipping: set RLX_TEST_DEVICE to a non-CPU backend");
        return;
    };
    // Only the last layer is MLA; layers 0..3 are KDA.
    let cfg = config(4);
    use rlx_ling::config::AttnKind;
    assert_eq!(cfg.attn_kind(0), AttnKind::Kda);
    assert_eq!(cfg.attn_kind(3), AttnKind::Mla);
    compare("kda-heavy", &cfg, device);
}

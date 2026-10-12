// RLX models — Qwen3.
// Copyright (C) 2026 Eugene Hauptmann.
// SPDX-License-Identifier: GPL-3.0-only

//! The decode graph must compute the same thing on Metal as on the CPU.
//!
//! Decode emits `logits` plus the appended K/V per layer, and the caller keeps
//! that K/V as its cache. A backend that returns correct logits but an empty
//! appended row is the worst case: generation continues, each step re-attends
//! only the prompt, and the model repeats itself a token behind. Comparing the
//! whole output list against the CPU catches that, where comparing generated
//! text only says "something is wrong somewhere".

#![cfg(target_os = "macos")]

use rlx_core::weight_map::WeightMap;
use rlx_qwen3::Qwen3Config;
use rlx_qwen3::builder::build_qwen3_decode_graph_sized_ext;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn tiny_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        qk_norm: true,
        sliding_window: None,
        max_window_layers: usize::MAX,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

/// Non-zero, deterministic weights. Zeros would make a dropped output
/// indistinguishable from a computed one.
/// Qwen3-0.6B's geometry with fewer layers and a small vocabulary. Kernel
/// selection on Metal keys off tensor widths — `head_dim` 128 and a 1024-wide
/// KV row take different paths than the toy sizes above — so the shapes that
/// matter are the real ones even when the layer count is not.
fn real_geometry_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 2048,
        hidden_size: 1024,
        intermediate_size: 3072,
        num_hidden_layers: 2,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 128,
        max_position_embeddings: 1024,
        ..tiny_cfg()
    }
}

/// 28 layers, which is what Qwen3-0.6B has — and therefore `1 + 2 * 28 = 57`
/// graph outputs. Dimensions stay small so the test is cheap; the point is the
/// output count and graph depth, not the arithmetic.
fn many_layers_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 256,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 28,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        max_position_embeddings: 1024,
        ..tiny_cfg()
    }
}

fn synthetic_weights(cfg: &Qwen3Config) -> WeightMap {
    let h = cfg.hidden_size;
    let (q_dim, kv_dim) = (cfg.q_proj_dim(), cfg.kv_proj_dim());
    let int_dim = cfg.intermediate_size;
    let dh = cfg.head_dim;
    let pat = |n: usize, salt: u32| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
                (x as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    };
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        (pat(cfg.vocab_size * h, 1), vec![cfg.vocab_size, h]),
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        let s = 10 + i as u32 * 97;
        for (name, n, shape) in [
            ("input_layernorm.weight", h, vec![h]),
            ("post_attention_layernorm.weight", h, vec![h]),
            ("self_attn.q_proj.weight", q_dim * h, vec![q_dim, h]),
            ("self_attn.k_proj.weight", kv_dim * h, vec![kv_dim, h]),
            ("self_attn.v_proj.weight", kv_dim * h, vec![kv_dim, h]),
            ("self_attn.o_proj.weight", h * q_dim, vec![h, q_dim]),
            ("self_attn.q_norm.weight", dh, vec![dh]),
            ("self_attn.k_norm.weight", dh, vec![dh]),
            ("mlp.gate_proj.weight", int_dim * h, vec![int_dim, h]),
            ("mlp.up_proj.weight", int_dim * h, vec![int_dim, h]),
            ("mlp.down_proj.weight", h * int_dim, vec![h, int_dim]),
        ] {
            t.insert(format!("{lp}.{name}"), (pat(n, s), shape));
        }
    }
    t.insert("model.norm.weight".into(), (pat(h, 7), vec![h]));
    t.insert(
        "lm_head.weight".into(),
        (pat(cfg.vocab_size * h, 3), vec![cfg.vocab_size, h]),
    );
    WeightMap::from_tensors(t)
}

/// `upper` is the bucket the graph is compiled for; `real` is how many of those
/// rows hold actual keys. `real < upper` is the bucketed case the decode loop
/// uses in practice — the rest is zero padding, masked out.
fn run_on(device: Device, cfg: &Qwen3Config, upper: usize, real: usize) -> Vec<Vec<f32>> {
    let mut wm = synthetic_weights(cfg);
    let (graph, params) =
        build_qwen3_decode_graph_sized_ext(cfg, &mut wm, 1, upper, true).expect("decode graph");
    let mut c = Session::new(device).compile(graph);
    for (name, data) in &params {
        c.set_param(name, data);
    }
    let kv_dim = cfg.kv_proj_dim();
    let half = cfg.head_dim / 2;
    // Deliberately distinct per row so a mis-indexed read is visible.
    let mut past_k = vec![0.0f32; upper * kv_dim];
    let mut past_v = vec![0.0f32; upper * kv_dim];
    for i in 0..real * kv_dim {
        past_k[i] = 0.01 * (i % 29) as f32 + 0.5;
        past_v[i] = 0.02 * (i % 31) as f32 - 0.3;
    }
    let cos: Vec<f32> = (0..half).map(|i| (0.1 * i as f32).cos()).collect();
    let sin: Vec<f32> = (0..half).map(|i| (0.1 * i as f32).sin()).collect();
    // Same rule as `rlx_runtime::attn_mask::bucket_decode_mask`: real past keys
    // and the new token are visible, the padding between them is not.
    let mask: Vec<f32> = (0..=upper)
        .map(|i| if i < real || i == upper { 1.0 } else { 0.0 })
        .collect();
    let ids = [3.0f32];

    let mut inputs: Vec<(&str, &[f32])> = vec![
        ("input_ids", &ids),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
        ("mask", &mask),
    ];
    let names: Vec<(String, String)> = (0..cfg.num_hidden_layers)
        .map(|l| (format!("past_k_{l}"), format!("past_v_{l}")))
        .collect();
    for (kn, vn) in &names {
        inputs.push((kn.as_str(), &past_k));
        inputs.push((vn.as_str(), &past_v));
    }
    c.run(&inputs)
}

#[test]
fn decode_outputs_match_cpu() {
    check(6, 6);
}

/// The bucketed case: the graph is compiled for a bucket wider than the context,
/// the surplus rows are zero and masked off, and the new token's K/V lands at
/// row `upper`. This is what a decode loop with a compile cache actually runs.
#[test]
fn bucketed_decode_outputs_match_cpu() {
    check(16, 4);
}

#[test]
fn bucketed_decode_matches_cpu_at_a_wide_bucket() {
    check(64, 5);
}

#[test]
fn real_geometry_bucketed_decode_matches_cpu() {
    check_cfg(real_geometry_cfg(), 64, 20);
}

/// The bucket a running model actually gets: orlx's ladder lands on 512 for a
/// short prompt, so ~500 of the padded rows are masked off. A row-copy kernel
/// that switches strategy above some width would only show up at this size.
#[test]
fn real_geometry_at_the_runners_bucket_matches_cpu() {
    check_cfg(real_geometry_cfg(), 512, 20);
}

/// 28 layers → 57 outputs, the shape the real model asks a backend for.
#[test]
fn twenty_eight_layer_decode_matches_cpu() {
    check_cfg(many_layers_cfg(), 512, 20);
}

/// Qwen3's real vocabulary, so the logits output is ~600 KB rather than 1 KB.
/// Output staging is sized from the outputs, so a large first output is the kind
/// of thing that can crowd out the ones after it.
/// Qwen3-0.6B ties its LM head to the embedding table, which changes the graph's
/// topology — the head reads a tensor that is also the embedding's source rather
/// than a weight of its own. Every case above supplies a separate `lm_head`.
#[test]
fn tied_embeddings_decode_matches_cpu() {
    let cfg = Qwen3Config {
        tie_word_embeddings: true,
        ..many_layers_cfg()
    };
    check_cfg(cfg, 512, 20);
}

#[test]
fn real_vocab_decode_matches_cpu() {
    let cfg = Qwen3Config {
        vocab_size: 151_936,
        ..many_layers_cfg()
    };
    check_cfg(cfg, 512, 20);
}

/// Metal is linked in only under the `metal` feature; without it
/// `Session::new(Device::Metal)` panics rather than erroring, so a plain
/// `cargo test` on a Mac would fail here for want of a backend rather than
/// for a parity break. Skip instead, the way the other backend-parity tests
/// in this workspace do.
fn have_metal() -> bool {
    if rlx_runtime::is_available(Device::Metal) {
        return true;
    }
    eprintln!("skip: Metal not available (build with --features metal)");
    false
}

fn check(upper: usize, real: usize) {
    check_cfg(tiny_cfg(), upper, real);
}

fn check_cfg(cfg: Qwen3Config, upper: usize, real: usize) {
    if !have_metal() {
        return;
    }
    let cpu = run_on(Device::Cpu, &cfg, upper, real);
    let metal = run_on(Device::Metal, &cfg, upper, real);

    assert_eq!(
        metal.len(),
        cpu.len(),
        "output count: metal {} vs cpu {}",
        metal.len(),
        cpu.len()
    );
    let kv_dim = cfg.kv_proj_dim();
    for (i, (m, c)) in metal.iter().zip(&cpu).enumerate() {
        assert_eq!(m.len(), c.len(), "output {i} length");
        let first_bad = m.iter().zip(c).position(|(a, b)| (a - b).abs() > 1e-3);
        if let Some(at) = first_bad {
            // Outputs 1.. are the K/V caches: naming the row makes "the
            // appended row is missing" legible without counting elements.
            let where_ = if i == 0 {
                format!("logits element {at}")
            } else {
                format!("row {} of {} (element {at})", at / kv_dim, m.len() / kv_dim)
            };
            panic!(
                "output {i} differs at {where_}\n  metal {:.5} vs cpu {:.5}\n  \
                 metal row sum {:.4}, cpu row sum {:.4}",
                m[at],
                c[at],
                m[(at / kv_dim) * kv_dim..][..kv_dim]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>(),
                c[(at / kv_dim) * kv_dim..][..kv_dim]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>(),
            );
        }
    }
}

/// The same decode step driven through `BucketedCompileCache` — the machinery a
/// real decode loop uses, including whatever the cache does to a compiled graph
/// to reuse it across steps. Everything above compiles a fresh session, so this
/// is the remaining difference between the tests and a running model.
#[test]
fn bucketed_compile_cache_decode_matches_cpu() {
    if !have_metal() {
        return;
    }
    let cfg = tiny_cfg();
    let (upper, real) = (16usize, 4usize);

    let one = |device: Device| -> Vec<Vec<f32>> {
        use rlx_core::autoregressive::run_bucketed_kv_decode;
        use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput};
        use rlx_runtime::kv_cache::LayerKvCache;

        let kv_dim = cfg.kv_proj_dim();
        let half = cfg.head_dim / 2;
        let layers = cfg.num_hidden_layers;

        // One bucket spanning the whole range. Bound rather than written
        // inline because `vec![0..n]` reads as `(0..n).collect()` at a glance,
        // which clippy flags for exactly that reason.
        let bucket = 0..(upper as u64 + 1);
        let mut cache = BucketedCompileCache::new(device, vec![bucket]);
        let kv = LayerKvCache {
            past_len: real,
            layers_k: (0..layers)
                .map(|_| {
                    (0..real * kv_dim)
                        .map(|i| 0.01 * (i % 29) as f32 + 0.5)
                        .collect()
                })
                .collect(),
            layers_v: (0..layers)
                .map(|_| {
                    (0..real * kv_dim)
                        .map(|i| 0.02 * (i % 31) as f32 - 0.3)
                        .collect()
                })
                .collect(),
            layers_kv_base: vec![0; layers],
        };
        let cos: Vec<f32> = (0..half).map(|i| (0.1 * i as f32).cos()).collect();
        let sin: Vec<f32> = (0..half).map(|i| (0.1 * i as f32).sin()).collect();
        let mask: Vec<f32> = (0..=upper)
            .map(|i| if i < real || i == upper { 1.0 } else { 0.0 })
            .collect();
        let ids = [3.0f32];
        let fixed = [
            CacheRunInput {
                name: "input_ids",
                data: &ids,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_cos",
                data: &cos,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_sin",
                data: &sin,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: &mask,
                row_inner: None,
            },
        ];
        let cfg2 = cfg.clone();
        let (logits, new_k, new_v) = run_bucketed_kv_decode(
            &mut cache,
            real,
            &kv,
            kv_dim,
            layers,
            &fixed,
            move |up| {
                let mut wm = synthetic_weights(&cfg2);
                build_qwen3_decode_graph_sized_ext(&cfg2, &mut wm, 1, up as usize, true)
                    .expect("decode graph")
            },
            // The real decode path compiles with the qwen3 decode profile, which
            // sets a device-specific fusion target. Defaults leave fusion off,
            // so using them here would test a configuration no model runs.
            &rlx_core::flow_bridge::compile_options_for_profile(
                &rlx_flow::CompileProfile::qwen3_decode(),
                device,
            ),
        )
        .expect("bucketed decode");
        let mut out = vec![logits];
        for l in 0..layers {
            out.push(new_k[l].clone());
            out.push(new_v[l].clone());
        }
        out
    };

    let cpu = one(Device::Cpu);
    let metal = one(Device::Metal);
    let kv_dim = cfg.kv_proj_dim();
    assert_eq!(metal.len(), cpu.len(), "output count");
    for (i, (m, c)) in metal.iter().zip(&cpu).enumerate() {
        assert_eq!(m.len(), c.len(), "output {i} length");
        if let Some(at) = m.iter().zip(c).position(|(a, b)| (a - b).abs() > 1e-3) {
            let row = if i == 0 { usize::MAX } else { at / kv_dim };
            panic!(
                "output {i} differs at element {at}{}\n  metal {:.5} vs cpu {:.5}\n  \
                 metal last row sum {:.4}, cpu last row sum {:.4}",
                if row == usize::MAX {
                    String::new()
                } else {
                    format!(" (row {row} of {})", m.len() / kv_dim)
                },
                m[at],
                c[at],
                m[m.len() - kv_dim..].iter().map(|v| v.abs()).sum::<f32>(),
                c[c.len() - kv_dim..].iter().map(|v| v.abs()).sum::<f32>(),
            );
        }
    }
}

/// Prefill then decode, in that order, in one process — the sequence a
/// generation actually performs.
///
/// Metal's command queue, MPS caches and kernel-tuning caches are
/// process-global, so a decode graph compiled after a prefill graph is not
/// necessarily the same as one compiled alone. Every test above compiles decode
/// first and in isolation, which is the one thing a real run never does.
#[test]
fn decode_after_prefill_in_one_process_matches_cpu() {
    use rlx_qwen3::builder::build_qwen3_graph_sized;

    if !have_metal() {
        return;
    }
    let cfg = real_geometry_cfg();
    let (upper, real) = (512usize, 20usize);

    let both = |device: Device| -> Vec<Vec<f32>> {
        // Prefill first, with KV outputs, exactly as seeding the cache does.
        let mut wm = synthetic_weights(&cfg);
        let (pg, pparams) =
            build_qwen3_graph_sized(&cfg, &mut wm, 1, real, true, true).expect("prefill graph");
        let mut pc = Session::new(device).compile(pg);
        for (name, data) in &pparams {
            pc.set_param(name, data);
        }
        let ids: Vec<f32> = (0..real).map(|i| (i % 29) as f32).collect();
        let _ = pc.run(&[("input_ids", &ids)]);
        // Then the decode step, in the same process, after prefill has run.
        run_on(device, &cfg, upper, real)
    };

    let cpu = both(Device::Cpu);
    let metal = both(Device::Metal);
    let kv_dim = cfg.kv_proj_dim();
    assert_eq!(metal.len(), cpu.len(), "output count");
    for (i, (m, c)) in metal.iter().zip(&cpu).enumerate() {
        assert_eq!(m.len(), c.len(), "output {i} length");
        if let Some(at) = m.iter().zip(c).position(|(a, b)| (a - b).abs() > 1e-3) {
            panic!(
                "output {i} differs at element {at} (row {} of {})\n  \
                 metal {:.5} vs cpu {:.5}\n  appended row sum: metal {:.4}, cpu {:.4}",
                at / kv_dim,
                m.len() / kv_dim,
                m[at],
                c[at],
                m[m.len() - kv_dim..].iter().map(|v| v.abs()).sum::<f32>(),
                c[c.len() - kv_dim..].iter().map(|v| v.abs()).sum::<f32>(),
            );
        }
    }
}

/// The KV output contract: exactly `past_seq + 1` rows, new token at row
/// `past_seq`.
///
/// Every consumer hard-codes that index — `feed_kv_row`, `read_output_row`,
/// `split_bucketed_decode_kv_per_layer` — and the attention mask is built
/// `past_seq + 1` wide. A graph that returns a row more puts the new token one
/// row further along and hands readers an untouched spare row instead, which
/// reads as zeros: generation continues, the cache never grows, and the model
/// repeats the previous token. Asserting the shape catches that at build time
/// rather than as strange output.
///
/// `RLX_QWEN3_INPLACE_KV` is set here because it is what used to add the extra
/// row, and `auto_metal_env` turns it on for Metal — so the regression would
/// reappear on exactly the configuration that shipped.
#[test]
fn kv_outputs_have_one_row_per_past_plus_the_new_token() {
    use rlx_ir::Graph;

    fn kv_rows(g: &Graph, idx: usize) -> usize {
        let dims = g.node(g.outputs[idx]).shape.dims();
        dims[dims.len() - 2].unwrap_static()
    }

    for inplace in ["0", "1"] {
        // SAFETY: tests in this file are single-threaded with respect to this
        // key; rlx reads it through its own registry during graph build.
        rlx_ir::env::set("RLX_QWEN3_INPLACE_KV", inplace);
        let cfg = tiny_cfg();
        let past = 12usize;
        let mut wm = synthetic_weights(&cfg);
        let (g, _) =
            build_qwen3_decode_graph_sized_ext(&cfg, &mut wm, 1, past, true).expect("decode graph");
        assert_eq!(
            g.outputs.len(),
            1 + 2 * cfg.num_hidden_layers,
            "logits plus K/V per layer"
        );
        for layer in 0..cfg.num_hidden_layers {
            for (label, idx) in [("K", 1 + 2 * layer), ("V", 2 + 2 * layer)] {
                assert_eq!(
                    kv_rows(&g, idx),
                    past + 1,
                    "layer {layer} {label} rows with RLX_QWEN3_INPLACE_KV={inplace}"
                );
            }
        }
        rlx_ir::env::unset("RLX_QWEN3_INPLACE_KV");
    }
}

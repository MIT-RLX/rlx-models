// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The loader contract against the *real* `google/diffusiongemma-26B-A4B-it`
//! checkpoint — every tensor name and shape, without downloading 51 GB.
//!
//! `fixtures/real_config.json` is the shipped config verbatim, and
//! `fixtures/real_checkpoint_shapes.json` is the name → shape map read out of
//! all eleven safetensors shard headers (1047 tensors, 25.82 B parameters —
//! matching the index's `total_parameters`).
//!
//! This is what catches a naming or geometry mistake now rather than after a
//! 51 GB download: that full-attention layers really do omit `v_proj`, that the
//! encoder's untied `layer_scalar` really lives under
//! `model.encoder.language_model.*`, that the expert banks are
//! `[E, 2·moe_inter, hidden]` before the pre-transpose, and so on.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use rlx_diffusiongemma::config::LayerType;
use rlx_diffusiongemma::vision::{VISION_PREFIX, VISION_PROJ_PREFIX};
use rlx_diffusiongemma::weights::{is_vision_key, required_keys};
use rlx_diffusiongemma::{DiffusionGemmaConfig, HF_MODEL_ID};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

fn real_config() -> DiffusionGemmaConfig {
    DiffusionGemmaConfig::from_file(fixtures().join("real_config.json"))
        .expect("the shipped config must parse")
}

fn real_shapes() -> HashMap<String, Vec<usize>> {
    let raw = std::fs::read_to_string(fixtures().join("real_checkpoint_shapes.json")).unwrap();
    serde_json::from_str(&raw).expect("shape fixture")
}

#[test]
fn the_shipped_config_parses_into_the_geometry_we_build() {
    let c = real_config();
    let t = &c.text_config;
    assert_eq!(c.model_type, "diffusion_gemma");
    assert_eq!(HF_MODEL_ID, "google/diffusiongemma-26B-A4B-it");
    assert_eq!(c.canvas_length, 256);
    assert_eq!(
        (t.num_hidden_layers, t.hidden_size, t.vocab_size),
        (30, 2816, 262_144)
    );
    assert_eq!((t.num_experts, t.top_k_experts), (128, 8));
    assert_eq!((t.moe_intermediate_size, t.intermediate_size), (704, 2112));
    assert_eq!(t.sliding_window, 1024);
    // 5:1 sliding:full, last layer full.
    let full: Vec<usize> = (0..t.num_hidden_layers).filter(|&l| t.is_full(l)).collect();
    assert_eq!(full, vec![5, 11, 17, 23, 29]);
    assert_eq!(t.layer_type(29), LayerType::Full);
    let v = c.vision_config.as_ref().expect("vision config");
    assert_eq!(
        (v.hidden_size, v.num_hidden_layers, v.head_dim),
        (1152, 27, 72)
    );
    assert_eq!(v.position_embedding_size, 10240);
    assert!(v.standardize && !v.use_clipped_linears);
}

#[test]
fn every_tensor_the_text_graphs_load_exists_in_the_checkpoint() {
    let c = real_config();
    let shapes = real_shapes();
    let have: HashSet<&str> = shapes.keys().map(|s| s.as_str()).collect();
    let missing: Vec<String> = required_keys(&c)
        .into_iter()
        .filter(|k| !have.contains(k.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "{} tensors the graphs load are absent from the checkpoint: {:?}",
        missing.len(),
        &missing[..missing.len().min(8)]
    );
}

/// The whole checkpoint is accounted for: text tensors the graphs load, plus
/// the vision tower and projector. Anything else would mean we are ignoring
/// weights that matter.
#[test]
fn no_checkpoint_tensor_is_unaccounted_for() {
    let c = real_config();
    let shapes = real_shapes();
    let required: HashSet<String> = required_keys(&c).into_iter().collect();
    let unaccounted: Vec<&str> = shapes
        .keys()
        .map(|s| s.as_str())
        .filter(|k| !required.contains(*k) && !is_vision_key(k))
        .collect();
    assert!(
        unaccounted.is_empty(),
        "{} checkpoint tensors are neither loaded nor recognised as vision: {:?}",
        unaccounted.len(),
        &unaccounted[..unaccounted.len().min(8)]
    );
    assert_eq!(shapes.len(), 1047, "fixture should cover the whole model");
}

#[test]
fn text_tensor_shapes_match_the_geometry_the_graphs_assume() {
    let c = real_config();
    let t = &c.text_config;
    let s = real_shapes();
    let get = |k: &str| -> &Vec<usize> { s.get(k).unwrap_or_else(|| panic!("missing {k}")) };
    let h = t.hidden_size;

    assert_eq!(get("model.decoder.embed_tokens.weight"), &[t.vocab_size, h]);
    assert_eq!(get("model.decoder.norm.weight"), &[h]);
    let sc = "model.decoder.self_conditioning";
    assert_eq!(
        get(&format!("{sc}.gate_proj.weight")),
        &[t.intermediate_size, h]
    );
    assert_eq!(
        get(&format!("{sc}.down_proj.weight")),
        &[h, t.intermediate_size]
    );
    assert_eq!(get(&format!("{sc}.pre_norm.weight")), &[h]);

    for l in 0..t.num_hidden_layers {
        let p = format!("model.decoder.layers.{l}");
        let dh = t.layer_head_dim(l);
        let kv = t.layer_kv_heads(l);
        let q_dim = t.num_attention_heads * dh;

        assert_eq!(
            get(&format!("{p}.self_attn.q_proj.weight")),
            &[q_dim, h],
            "L{l} q"
        );
        assert_eq!(
            get(&format!("{p}.self_attn.k_proj.weight")),
            &[kv * dh, h],
            "L{l} k"
        );
        assert_eq!(
            get(&format!("{p}.self_attn.o_proj.weight")),
            &[h, q_dim],
            "L{l} o"
        );
        // Per-head norms are head_dim wide, which differs between layer types.
        assert_eq!(
            get(&format!("{p}.self_attn.q_norm.weight")),
            &[dh],
            "L{l} q_norm"
        );
        assert_eq!(
            get(&format!("{p}.self_attn.k_norm.weight")),
            &[dh],
            "L{l} k_norm"
        );

        // Expert banks, pre-transpose.
        assert_eq!(
            get(&format!("{p}.experts.gate_up_proj")),
            &[t.num_experts, 2 * t.moe_intermediate_size, h],
            "L{l} gate_up"
        );
        assert_eq!(
            get(&format!("{p}.experts.down_proj")),
            &[t.num_experts, h, t.moe_intermediate_size],
            "L{l} down"
        );
        assert_eq!(get(&format!("{p}.router.proj.weight")), &[t.num_experts, h]);
        assert_eq!(get(&format!("{p}.router.scale")), &[h]);
        assert_eq!(
            get(&format!("{p}.router.per_expert_scale")),
            &[t.num_experts]
        );

        // Shared expert.
        assert_eq!(
            get(&format!("{p}.mlp.gate_proj.weight")),
            &[t.intermediate_size, h]
        );
        assert_eq!(
            get(&format!("{p}.mlp.down_proj.weight")),
            &[h, t.intermediate_size]
        );

        // Seven norms plus the two untied per-stack scalars.
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ] {
            assert_eq!(get(&format!("{p}.{n}.weight")), &[h], "L{l} {n}");
        }
        assert_eq!(get(&format!("{p}.layer_scalar")), &[1]);
        assert_eq!(
            get(&format!(
                "model.encoder.language_model.layers.{l}.layer_scalar"
            )),
            &[1],
            "L{l} encoder scalar"
        );
    }
}

/// The single most load-bearing structural claim: full-attention layers ship no
/// `v_proj` at all, and their geometry is 16×512 with 2 KV heads.
#[test]
fn only_sliding_layers_carry_a_v_proj() {
    let c = real_config();
    let t = &c.text_config;
    let s = real_shapes();
    let v_projs: Vec<usize> = (0..t.num_hidden_layers)
        .filter(|l| s.contains_key(&format!("model.decoder.layers.{l}.self_attn.v_proj.weight")))
        .collect();
    assert_eq!(v_projs.len(), 25, "25 of 30 layers have a v_proj");
    for l in 0..t.num_hidden_layers {
        let has_v = v_projs.contains(&l);
        assert_eq!(
            has_v,
            !t.layer_k_eq_v(l),
            "layer {l}: checkpoint has v_proj = {has_v}, but layer_k_eq_v says {}",
            t.layer_k_eq_v(l)
        );
        let dh = t.layer_head_dim(l);
        let kv = t.layer_kv_heads(l);
        if t.is_full(l) {
            assert_eq!((dh, kv), (512, 2), "layer {l} is a global layer");
        } else {
            assert_eq!((dh, kv), (256, 8), "layer {l} is a sliding layer");
            assert_eq!(
                s[&format!("model.decoder.layers.{l}.self_attn.v_proj.weight")],
                vec![kv * dh, t.hidden_size]
            );
        }
    }
}

#[test]
fn vision_tensor_shapes_match_the_tower_we_build() {
    let c = real_config();
    let v = c.vision_config.as_ref().unwrap();
    let s = real_shapes();
    let get = |k: &str| -> &Vec<usize> { s.get(k).unwrap_or_else(|| panic!("missing {k}")) };
    let h = v.hidden_size;

    assert_eq!(
        get(&format!("{VISION_PREFIX}.patch_embedder.input_proj.weight")),
        &[h, 3 * v.patch_size * v.patch_size]
    );
    assert_eq!(
        get(&format!(
            "{VISION_PREFIX}.patch_embedder.position_embedding_table"
        )),
        &[2, v.position_embedding_size, h]
    );
    assert_eq!(get(&format!("{VISION_PREFIX}.std_scale")), &[h]);
    assert_eq!(get(&format!("{VISION_PREFIX}.std_bias")), &[h]);
    assert_eq!(
        get(&format!("{VISION_PROJ_PREFIX}.embedding_projection.weight")),
        &[c.text_config.hidden_size, h]
    );

    for l in 0..v.num_hidden_layers {
        let p = format!("{VISION_PREFIX}.encoder.layers.{l}");
        // Vision linears nest under `.linear.weight` (ClippableLinear).
        for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            assert_eq!(
                get(&format!("{p}.self_attn.{n}.linear.weight")),
                &[h, h],
                "L{l} {n}"
            );
        }
        assert_eq!(get(&format!("{p}.self_attn.q_norm.weight")), &[v.head_dim]);
        assert_eq!(get(&format!("{p}.self_attn.k_norm.weight")), &[v.head_dim]);
        assert_eq!(
            get(&format!("{p}.mlp.gate_proj.linear.weight")),
            &[v.intermediate_size, h]
        );
        assert_eq!(
            get(&format!("{p}.mlp.down_proj.linear.weight")),
            &[h, v.intermediate_size]
        );
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            assert_eq!(get(&format!("{p}.{n}.weight")), &[h], "L{l} {n}");
        }
    }
    // Vision heads tile the hidden size exactly (no GQA in the tower).
    assert_eq!(v.num_attention_heads * v.head_dim, h);
}

/// Sanity on scale: the fixture should account for the published parameter
/// count, and the f32 expert footprint is what blocks a real-weight run.
#[test]
fn parameter_budget_matches_the_published_totals() {
    let c = real_config();
    let t = &c.text_config;
    let s = real_shapes();
    let total: usize = s.values().map(|sh| sh.iter().product::<usize>()).sum();

    // The index's `total_parameters` counts `nn.Parameter`s only. The tensors
    // that are `nn.Buffer`s — the two per-stack `layer_scalar`s per layer and
    // the vision tower's `std_scale` / `std_bias` — are stored but not counted,
    // which is exactly the 2364-element gap.
    let buffers: usize = s
        .iter()
        .filter(|(k, _)| {
            k.ends_with(".layer_scalar") || k.ends_with(".std_scale") || k.ends_with(".std_bias")
        })
        .map(|(_, sh)| sh.iter().product::<usize>())
        .sum();
    let v = c.vision_config.as_ref().unwrap();
    assert_eq!(buffers, 2 * t.num_hidden_layers + 2 * v.hidden_size);
    assert_eq!(
        total - buffers,
        25_823_778_864,
        "published total_parameters (buffers excluded)"
    );

    let experts: usize = s
        .iter()
        .filter(|(k, _)| k.contains(".experts."))
        .map(|(_, sh)| sh.iter().product::<usize>())
        .sum();
    // 30 layers × 128 experts × 3 × 704 × 2816.
    assert_eq!(
        experts,
        t.num_hidden_layers * t.num_experts * 3 * t.moe_intermediate_size * t.hidden_size
    );
    // ~91 GB as f32 — the reason a plain `WeightMap` cannot host real weights.
    let f32_gb = (experts * 4) as f64 / 1e9;
    assert!(
        (85.0..100.0).contains(&f32_gb),
        "routed experts are {f32_gb:.1} GB as f32"
    );
}

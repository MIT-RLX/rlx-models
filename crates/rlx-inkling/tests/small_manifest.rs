//! Inkling-Small against its published `config.json` and tensor manifest.
//!
//! Both fixtures come from the Hub verbatim (`scripts/fetch_small_manifest.py`,
//! safetensors headers only), so nothing here is a hand-written second copy of
//! the contract. No weights are needed.

use rlx_inkling::weights::{expected_text_hf_keys, hf_composite_conversions, rename_hf_key};
use rlx_inkling::{AttnLayerType, InklingConfig, InklingVariant, expected_hf_shapes};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

const CONFIG: &str = include_str!("fixtures/small/config.json");
const MANIFEST: &str = include_str!("fixtures/small/manifest.json");

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
}

fn manifest() -> BTreeMap<String, Entry> {
    serde_json::from_str(MANIFEST).expect("manifest fixture")
}

fn raw_text_config() -> serde_json::Value {
    let root: serde_json::Value = serde_json::from_str(CONFIG).expect("config fixture");
    root["text_config"].clone()
}

#[test]
fn config_parses_as_small() {
    let cfg = InklingConfig::from_json_str(CONFIG).expect("parse");
    let t = &cfg.text;
    assert_eq!(cfg.variant(), InklingVariant::Small);
    assert_eq!(t.num_hidden_layers, 42);
    assert_eq!(t.hidden_size, 4096);
    assert_eq!(
        (t.num_attention_heads, t.num_key_value_heads, t.head_dim),
        (32, 8, 128)
    );
    assert_eq!(
        (
            t.swa_num_attention_heads,
            t.swa_num_key_value_heads,
            t.swa_head_dim
        ),
        (32, 8, 128)
    );
    assert_eq!(t.dense_intermediate_size, 16_384);
    assert_eq!(t.moe_intermediate_size, 2048);
    assert_eq!((t.n_routed_experts, t.num_experts_per_tok), (256, 6));
    assert_eq!(t.n_shared_experts, 2);
    assert_eq!(t.logits_mup_width_multiplier, 16.0);
    assert_eq!(t.num_mtp_layers, 8);
    assert_eq!(cfg.audio.text_hidden_size, 4096);
    assert_eq!(cfg.vision.text_hidden_size, 4096);

    // The global layers are exactly those missing from `local_layer_ids`.
    let global: Vec<usize> = (0..42)
        .filter(|&i| t.layer_types[i] == AttnLayerType::Global)
        .collect();
    assert_eq!(global, [5, 11, 17, 23, 29, 35, 41]);
    let dense: Vec<usize> = (0..42).filter(|&i| t.is_dense_mlp(i)).collect();
    assert_eq!(dense, [0, 1]);
}

/// The builder hardcodes these; a checkpoint that changes one must fail here
/// instead of silently building the wrong graph.
#[test]
fn config_flags_match_what_the_builder_assumes() {
    let t = raw_text_config();
    assert_eq!(t["q_bias"], false);
    assert_eq!(t["o_bias"], false);
    assert_eq!(t["use_gate_bias"], true);
    assert_eq!(t["gate_activation"], "sigmoid");
    assert_eq!(t["norm_after_topk"], true);
    assert_eq!(t["use_global_scale"], true);
    assert_eq!(t["shared_expert_sink"], true);
    assert_eq!(t["use_sconv"], true);
    assert!(t["final_logit_softcapping"].is_null());
}

/// Every text tensor the shape table expects exists with that shape, and every
/// `model.llm.*` tensor in the checkpoint is expected.
#[test]
fn text_trunk_shapes_match_both_directions() {
    let cfg = InklingConfig::from_json_str(CONFIG).unwrap();
    let want = expected_hf_shapes(&cfg);
    let have = manifest();

    for (name, shape) in &want {
        let got = have
            .get(name)
            .unwrap_or_else(|| panic!("expected tensor missing from checkpoint: {name}"));
        assert_eq!(&got.shape, shape, "{name}");
    }
    let unexpected: Vec<&String> = have
        .keys()
        .filter(|k| k.starts_with("model.llm.") && !want.contains_key(*k))
        .collect();
    assert!(unexpected.is_empty(), "unread text tensors: {unexpected:?}");
}

#[test]
fn expected_key_list_matches_shape_table_and_checkpoint() {
    let cfg = InklingConfig::from_json_str(CONFIG).unwrap();
    let keys: BTreeSet<String> = expected_text_hf_keys(&cfg.text).into_iter().collect();
    let shapes: BTreeSet<String> = expected_hf_shapes(&cfg).into_keys().collect();
    assert_eq!(keys, shapes);
    let have = manifest();
    assert!(keys.iter().all(|k| have.contains_key(k)));
}

#[test]
fn dtypes_are_bf16_except_router_scalars() {
    for (name, e) in manifest() {
        let f32_ok = name.ends_with("mlp.gate.bias") || name.ends_with("mlp.gate.global_scale");
        let want = if f32_ok { "F32" } else { "BF16" };
        assert_eq!(e.dtype, want, "{name}");
    }
}

#[test]
fn mtp_head_matches_config() {
    let cfg = InklingConfig::from_json_str(CONFIG).unwrap();
    let have = manifest();
    let heads: BTreeSet<usize> = have
        .keys()
        .filter_map(|k| {
            k.strip_prefix("model.mtp.layers.")?
                .split('.')
                .next()?
                .parse()
                .ok()
        })
        .collect();
    assert_eq!(heads.len(), cfg.text.num_mtp_layers);

    let h = cfg.text.hidden_size;
    for i in 0..cfg.text.num_mtp_layers {
        let p = format!("model.mtp.layers.{i}");
        assert_eq!(have[&format!("{p}.input_proj.weight")].shape, [h, 2 * h]);
        // MTP blocks are dense (a single `w13_dn`) and follow `mtp_config.local_layer_ids`.
        let blk = format!("{p}.transformer_block");
        assert!(have.contains_key(&format!("{blk}.mlp.w13_dn.weight")));
        let sliding = cfg.text.mtp_local_layer_ids.contains(&i);
        let rel = have[&format!("{blk}.attn.rel_logits_proj.proj")].shape[1];
        let want = if sliding {
            cfg.text.sliding_window_size
        } else {
            cfg.text.rel_extent
        };
        assert_eq!(rel, want, "mtp {i} relative extent");
    }
}

/// The towers project into the trunk's hidden width, and the presence keys the
/// remote probe looks for exist. Their inner shapes are not derived from config.
#[test]
fn towers_project_into_hidden_width() {
    let cfg = InklingConfig::from_json_str(CONFIG).unwrap();
    let h = cfg.text.hidden_size;
    let have = manifest();
    for key in rlx_inkling::shapes::multimodal_presence_keys() {
        assert!(have.contains_key(*key), "{key}");
    }
    assert_eq!(have["model.visual.layers.linear_3.weight"].shape[0], h);
    assert_eq!(have["model.visual.final_norm.weight"].shape, [h]);
    assert_eq!(have["model.audio.encoder.weight"].shape[1], h);
    assert_eq!(have["model.audio.final_norm.weight"].shape, [h]);
}

/// The HF → eager conversion plan reads exactly the checkpoint's text tensors.
#[test]
fn conversion_plan_reads_every_text_tensor_once() {
    let cfg = InklingConfig::from_json_str(CONFIG).unwrap();
    let plan: Vec<String> = rlx_inkling::weights::hf_text_plan(&cfg.text)
        .iter()
        .map(|s| s.hf_key().to_string())
        .collect();
    let unique: BTreeSet<&String> = plan.iter().collect();
    assert_eq!(unique.len(), plan.len(), "a tensor is read twice");
    let text: BTreeSet<String> = manifest()
        .into_keys()
        .filter(|k| k.starts_with("model.llm."))
        .collect();
    assert_eq!(unique.into_iter().cloned().collect::<BTreeSet<_>>(), text);
}

/// Every tensor name is either renamed, converted, or deliberately left alone.
#[test]
fn every_text_tensor_has_a_canonical_mapping() {
    for name in manifest().keys().filter(|k| k.starts_with("model.llm.")) {
        let renamed = rename_hf_key(name);
        let composite = hf_composite_conversions(name);
        assert!(
            renamed.is_some() || composite.is_some(),
            "no mapping for {name}"
        );
    }
}

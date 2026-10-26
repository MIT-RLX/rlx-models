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

//! The published `config.json` and, with `RLX_FLINT_DIR` set to the
//! checkpoint, its tensor index — both directions, headers only.

use rlx_flint::FlintConfig;
use rlx_flint::synth::tensor_manifest;

const CONFIG: &str = include_str!("fixtures/config.json");

#[test]
fn the_published_config_parses_to_the_model_card() {
    let cfg = FlintConfig::from_json(CONFIG).unwrap();
    assert_eq!(cfg.num_hidden_layers, 49);
    assert_eq!(
        (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim
        ),
        (32, 4, 128)
    );
    assert_eq!(cfg.sliding_window, Some(513));
    assert_eq!(cfg.sliding_history(), Some(512));
    assert_eq!((cfg.num_experts, cfg.num_experts_per_tok), (256, 6));

    let specs: Vec<_> = (0..49).map(|i| cfg.layer(i)).collect();
    // Model card: 35 sliding + 14 full; 2 dense + 47 MoE.
    assert_eq!(specs.iter().filter(|s| s.sliding).count(), 35);
    assert_eq!(specs.iter().filter(|s| s.dense).count(), 2);
    // "RoPE on sliding-window layers; no positional encoding on full-attention
    // layers" — the two lists coincide exactly.
    for (i, s) in specs.iter().enumerate() {
        assert_eq!(s.rope, s.sliding, "layer {i}");
    }
    assert!(cfg.is_eos(150001) && cfg.is_eos(150003) && !cfg.is_eos(150004));
}

#[test]
fn the_generic_path_sniffs_a_flint_checkpoint_to_this_runner() {
    // `auto_runner` reads `model_type` from config.json and looks it up here;
    // without the entry a Flint directory is an unknown architecture.
    assert_eq!(
        rlx_core::model_registry::runner_for_hf_model_type("flint"),
        Some("flint")
    );
}

#[test]
fn configs_the_reference_does_not_implement_are_refused() {
    for (k, v) in [
        ("softmax_type", serde_json::json!("vanilla")),
        ("router_score_function", serde_json::json!("softmax")),
        ("decoder_sparse_step", serde_json::json!(2)),
    ] {
        let mut c: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
        c[k] = v;
        assert!(FlintConfig::from_value(&c).is_err(), "{k} was accepted");
    }
}

#[test]
fn the_real_checkpoint_matches_the_manifest_both_ways() {
    let Some(dir) = rlx_ir::env::var("RLX_FLINT_DIR") else {
        eprintln!("RLX_FLINT_DIR unset — skipping");
        return;
    };
    let cfg = FlintConfig::from_path(&dir).unwrap();
    let ckpt = rlx_flint::Checkpoint::open(&dir).unwrap();
    let want = tensor_manifest(&cfg);
    for (name, shape) in &want {
        let e = ckpt
            .entry(name)
            .unwrap_or_else(|_| panic!("the model reads {name}, the checkpoint lacks it"));
        assert_eq!(&e.shape, shape, "{name}");
    }
    let extra: Vec<&str> = ckpt.names().filter(|n| !want.contains_key(*n)).collect();
    assert!(
        extra.is_empty(),
        "never read: {:?}",
        &extra[..extra.len().min(5)]
    );
    assert_eq!(want.len(), 36_734);
}

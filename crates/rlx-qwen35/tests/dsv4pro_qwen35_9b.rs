// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` against this
//! crate's HF reader, with no weights present.
//!
//! The checkpoint is a fine-tune of `unsloth/Qwen3.5-9B` (`qwen3_5`,
//! `Qwen3_5ForConditionalGeneration`): DeepSeek-V4-Pro reasoning distilled into
//! the stock Qwen3.5-9B layout, then abliterated. Nothing about the
//! architecture is new, so the work is proving the reader describes the
//! *published* files. The fixtures are the real `config.json`, verbatim, and
//! every tensor's name, dtype and shape read from the five safetensors headers:
//!
//! ```sh
//! python3 scripts/hf_safetensors_subset.py \
//!     xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic \
//!     --manifest crates/rlx-qwen35/tests/fixtures/dsv4pro_qwen35_9b_manifest.json
//! ```
//!
//! Numerics on real weights: `just test-dsv4pro-qwen35-9b-parity` (a 4-layer
//! range-fetched subset vs transformers).

use rlx_qwen35::Qwen35Config;
use rlx_qwen35::hf::hf_decoder_layer_count;
use std::collections::BTreeMap;
use std::path::Path;

const CONFIG: &str = include_str!("fixtures/dsv4pro_qwen35_9b_config.json");
const MANIFEST: &str = include_str!("fixtures/dsv4pro_qwen35_9b_manifest.json");

const TEXT: &str = "model.language_model.";

fn config() -> Qwen35Config {
    let v: serde_json::Value = serde_json::from_str(CONFIG).expect("fixture is JSON");
    Qwen35Config::from_hf_config_value(&v, Path::new("config.json"))
        .expect("should parse as a Qwen3.5 config")
}

/// `{name: (dtype, shape)}` for all 775 published tensors.
fn manifest() -> BTreeMap<String, (String, Vec<usize>)> {
    let v: BTreeMap<String, (String, Vec<usize>)> =
        serde_json::from_str(MANIFEST).expect("manifest is JSON");
    assert_eq!(v.len(), 775, "tensor count in the published index");
    v
}

fn shape<'a>(m: &'a BTreeMap<String, (String, Vec<usize>)>, k: &str) -> &'a [usize] {
    &m.get(k)
        .unwrap_or_else(|| panic!("{k} missing from manifest"))
        .1
}

#[test]
fn hyper_parameters_are_read() {
    let c = config();
    assert_eq!(c.vocab_size, 248_320);
    assert_eq!(c.hidden_size, 4096);
    assert_eq!(c.intermediate_size, 12288);
    assert_eq!(c.num_attention_heads, 16);
    assert_eq!(c.num_key_value_heads, 4);
    assert_eq!(c.key_length, 256, "head_dim");
    assert_eq!(c.value_length, 256);
    assert_eq!(c.full_attention_interval, 4);
    // partial_rotary_factor 0.25 of 256.
    assert_eq!(c.rope_dim_count, 64);
    assert_eq!(c.rope_theta, 10_000_000.0);
    assert!(c.mrope_interleaved);
    assert_eq!(c.rope_dim_sections, vec![11, 11, 10]);
    // GDN: 16 key heads, 32 value heads, all 128 wide.
    assert_eq!(c.ssm_group_count, 16);
    assert_eq!(c.ssm_state_size, 128);
    assert_eq!(c.ssm_time_step_rank, 32);
    assert_eq!(c.ssm_inner_size, 32 * 128);
    assert_eq!(c.ssm_conv_kernel, 4);
    assert_eq!(c.rms_norm_eps, 1e-6);
}

/// The LM head is its own tensor here (0.8B ties it); reading the config's
/// default of `true` would silently project through the embedding.
#[test]
fn lm_head_is_untied() {
    assert!(!config().tie_word_embeddings);
    let m = manifest();
    assert_eq!(shape(&m, "lm_head.weight"), [248_320, 4096]);
    assert_eq!(
        shape(&m, &format!("{TEXT}embed_tokens.weight")),
        [248_320, 4096]
    );
}

/// The config says 32 layers with one MTP layer, which the reader turns into a
/// 31-layer trunk — right for a GGUF, where llama.cpp appends MTP as the last
/// `blk.N`, and one layer short here, where MTP is a separate `mtp.*` module.
/// The runner takes the depth from the tensor names instead; this is the check
/// that the names give 32.
#[test]
fn trunk_depth_comes_from_the_tensors() {
    assert_eq!(config().num_hidden_layers, 31, "config-only depth");
    let m = manifest();
    assert_eq!(
        hf_decoder_layer_count(m.keys().map(String::as_str)),
        Some(32)
    );
    assert!(m.contains_key("mtp.layers.0.self_attn.q_proj.weight"));
    assert!(!m.contains_key(&format!("{TEXT}layers.32.input_layernorm.weight")));
}

/// Every layer's kind and every projection's shape, against the config. The
/// schedule is `(i + 1) % 4 == 0` → full attention, the rest Gated DeltaNet.
#[test]
fn every_layer_matches_the_config() {
    let c = config();
    let m = manifest();
    let (h, ff) = (c.hidden_size, c.intermediate_size);
    let key_dim = c.ssm_group_count * c.ssm_state_size;
    let value_dim = c.ssm_inner_size;
    let n_v = c.ssm_time_step_rank;
    let (hd, nh, nkv) = (c.key_length, c.num_attention_heads, c.num_key_value_heads);
    let mut full = 0;
    for i in 0..32 {
        let p = format!("{TEXT}layers.{i}.");
        let s = |k: &str| shape(&m, &format!("{p}{k}")).to_vec();
        assert_eq!(s("mlp.gate_proj.weight"), [ff, h]);
        assert_eq!(s("mlp.up_proj.weight"), [ff, h]);
        assert_eq!(s("mlp.down_proj.weight"), [h, ff]);
        assert_eq!(s("input_layernorm.weight"), [h]);
        assert_eq!(s("post_attention_layernorm.weight"), [h]);
        let is_full = (i + 1) % c.full_attention_interval == 0;
        assert_eq!(
            m.contains_key(&format!("{p}self_attn.q_proj.weight")),
            is_full,
            "layer {i} kind"
        );
        if is_full {
            full += 1;
            // q_proj carries the output gate: 2 × heads × head_dim.
            assert_eq!(s("self_attn.q_proj.weight"), [2 * nh * hd, h]);
            assert_eq!(s("self_attn.k_proj.weight"), [nkv * hd, h]);
            assert_eq!(s("self_attn.v_proj.weight"), [nkv * hd, h]);
            assert_eq!(s("self_attn.o_proj.weight"), [h, nh * hd]);
            assert_eq!(s("self_attn.q_norm.weight"), [hd]);
            assert_eq!(s("self_attn.k_norm.weight"), [hd]);
        } else {
            assert_eq!(
                s("linear_attn.in_proj_qkv.weight"),
                [2 * key_dim + value_dim, h]
            );
            assert_eq!(
                s("linear_attn.conv1d.weight"),
                [2 * key_dim + value_dim, 1, c.ssm_conv_kernel]
            );
            assert_eq!(s("linear_attn.in_proj_z.weight"), [value_dim, h]);
            assert_eq!(s("linear_attn.in_proj_a.weight"), [n_v, h]);
            assert_eq!(s("linear_attn.in_proj_b.weight"), [n_v, h]);
            assert_eq!(s("linear_attn.A_log"), [n_v]);
            assert_eq!(s("linear_attn.dt_bias"), [n_v]);
            assert_eq!(s("linear_attn.norm.weight"), [c.ssm_state_size]);
            assert_eq!(s("linear_attn.out_proj.weight"), [h, value_dim]);
        }
    }
    assert_eq!(full, 8);
}

/// More value heads than key heads, by a whole factor: the case where HF's
/// grouped head order and the builder's tiled order differ, so the loader must
/// reorder (`HfQwen35Fixups::gdn_heads`). Qwen3.5-4B is the other model with
/// this geometry; 0.8B and 2B are the ones where it is a no-op.
#[test]
fn gdn_value_heads_need_the_reorder() {
    let c = config();
    assert!(c.ssm_time_step_rank > c.ssm_group_count);
    assert_eq!(c.ssm_time_step_rank % c.ssm_group_count, 0);
}

/// Nothing outside the four known modules — an unexpected prefix would be a
/// tensor no reader consumes.
#[test]
fn no_unknown_tensors() {
    let m = manifest();
    let known = [TEXT, "lm_head.", "model.visual.", "mtp."];
    for k in m.keys() {
        assert!(
            known.iter().any(|p| k.starts_with(p)),
            "unexpected tensor {k}"
        );
    }
    let text = m.keys().filter(|k| k.starts_with(TEXT)).count();
    // 32 × (5 shared) + 24 × 9 GDN + 8 × 6 attention + embed + norm.
    assert_eq!(text, 32 * 5 + 24 * 9 + 8 * 6 + 2);
}

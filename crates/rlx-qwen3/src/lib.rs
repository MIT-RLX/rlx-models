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

//! Qwen3 — Alibaba's dense causal-LM family (0.6B / 1.7B / 4B / 8B /
//! 14B / 32B). GQA + QK-norm + SwiGLU MLP + RoPE.
//!
//! Public entry points:
//!   - [`Qwen3Config`] — HF `config.json` deserialiser.
//!   - [`build_qwen3_graph_sized`] — emits the prefill IR graph.
//!
//! Weight keys mirror HF's `Qwen3ForCausalLM` layout so safetensors
//! checkpoints load with no remapping. GGUF support comes for free via
//! the `WeightLoader` adapter in `rlx_core::weight_loader`.
//!
//! Prefill and KV-cache decode run on every standard RLX backend when the
//! matching `rlx-runtime` feature is enabled (`cpu`, `metal`, `mlx`, `cuda`,
//! `rocm`, `gpu`, `vulkan`). Build with `all-backends` for a single binary
//! that accepts all `--device` values.

pub mod builder;
pub mod capabilities;
pub mod cli;
pub mod config;
pub mod embedder;
pub mod flow;
pub mod generator;
pub mod high_level_runner;
pub mod packed_decode;
pub mod pipeline;
pub mod pipeline_decode;
pub mod precision;
mod profile;
pub mod sampling;
pub mod spec;
pub mod tool_call;

pub use builder::{
    build_qwen3_decode_graph_sized, build_qwen3_graph_sized, build_qwen3_graph_sized_last_logits,
    build_qwen3_graph_sized_packed,
};
pub use capabilities::{STANDARD_DEVICE_NAMES, STANDARD_DEVICES, validate_device};
pub use config::Qwen3Config;
pub use flow::{
    Qwen3DecodeOpts, Qwen3Flow, Qwen3Mode, Qwen3PrefillOpts, build_qwen3_decode_built,
    build_qwen3_decode_embeds_built, build_qwen3_decode_flow, build_qwen3_decode_graph,
    build_qwen3_decode_hir, build_qwen3_prefill_built, build_qwen3_prefill_embeds_built,
    build_qwen3_prefill_flow, build_qwen3_prefill_graph,
};
#[cfg(feature = "mmap-kv")]
pub use generator::KvStoreConfig;
pub use generator::Qwen3Generator;
pub use high_level_runner::{Precision, Qwen3ConfigSource, Qwen3Runner, Qwen3RunnerBuilder};
pub use pipeline::{
    BlockSpec, Qwen3PipelineStage, block_weight_filter, build_qwen3_block_built,
    build_qwen3_block_graph,
};
pub use pipeline_decode::Qwen3PipelineDecodeStage;
pub use profile::{QWEN3_PROFILE_FILE, qwen3_profile_default, qwen3_profile_near_weights};
pub use sampling::{
    SampleOpts, apply_repetition_penalty, sample_token, sample_token_at, softmax_logits,
};
pub use spec::Qwen3Speculator;
pub use tool_call::{
    ToolCall, ToolResultCache, ToolSpec, ToolsSystemPromptCache, has_tool_call, parse_tool_calls,
    render_tool_response, render_tools_system,
};

/// One-import DX surface for downstream users — the runner + generation config,
/// sampling, tool-calling glue, and (under `mmap-kv`) the KV context store:
/// `use rlx_qwen3::prelude::*;`. A convenience grouping of already-public items;
/// it flips no defaults and gates the same features the crate does.
pub mod prelude {
    pub use crate::config::Qwen3Config;
    pub use crate::generator::Qwen3Generator;
    pub use crate::high_level_runner::{
        Precision, Qwen3ConfigSource, Qwen3Runner, Qwen3RunnerBuilder,
    };
    pub use crate::sampling::{SampleOpts, sample_token};
    pub use crate::tool_call::{
        ToolCall, ToolResultCache, ToolSpec, ToolsSystemPromptCache, has_tool_call,
        parse_tool_calls, render_tool_response, render_tools_system,
    };
    // Needed by every runner call site.
    #[cfg(feature = "mmap-kv")]
    pub use crate::generator::KvStoreConfig;
    pub use rlx_runtime::Device;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
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
            max_position_embeddings: 16,
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

    fn synthetic_weights(cfg: &Qwen3Config) -> WeightMap {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_proj_dim();
        let kv_dim = cfg.kv_proj_dim();
        let int_dim = cfg.intermediate_size;
        let dh = cfg.head_dim;

        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];

        t.insert(
            "model.embed_tokens.weight".into(),
            (z(cfg.vocab_size * h), vec![cfg.vocab_size, h]),
        );

        for i in 0..cfg.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            t.insert(format!("{lp}.input_layernorm.weight"), (z(h), vec![h]));
            t.insert(
                format!("{lp}.post_attention_layernorm.weight"),
                (z(h), vec![h]),
            );

            // HF stores linear weights as [out, in].
            t.insert(
                format!("{lp}.self_attn.q_proj.weight"),
                (z(q_dim * h), vec![q_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.k_proj.weight"),
                (z(kv_dim * h), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.v_proj.weight"),
                (z(kv_dim * h), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.o_proj.weight"),
                (z(h * q_dim), vec![h, q_dim]),
            );
            t.insert(format!("{lp}.self_attn.q_norm.weight"), (z(dh), vec![dh]));
            t.insert(format!("{lp}.self_attn.k_norm.weight"), (z(dh), vec![dh]));

            t.insert(
                format!("{lp}.mlp.gate_proj.weight"),
                (z(int_dim * h), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.up_proj.weight"),
                (z(int_dim * h), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.down_proj.weight"),
                (z(h * int_dim), vec![h, int_dim]),
            );
        }
        t.insert("model.norm.weight".into(), (z(h), vec![h]));

        WeightMap::from_tensors(t)
    }

    #[test]
    fn prefill_graph_builds_and_consumes_every_weight() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let (g, _params) = build_qwen3_graph_sized(&cfg, &mut wm, 1, 4, false, false).unwrap();
        assert_eq!(g.outputs.len(), 1);
        let out = g.outputs[0];
        let dims: Vec<usize> = g
            .shape(out)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        assert_eq!(dims, vec![1, 4, cfg.hidden_size]);
        assert_eq!(
            wm.len(),
            0,
            "unused weights: {:?}",
            wm.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn gqa_handles_kv_group_of_one() {
        let mut cfg = tiny_cfg();
        cfg.num_key_value_heads = cfg.num_attention_heads; // group=1 → no repeat path
        let mut wm = synthetic_weights(&cfg);
        let (g, _) = build_qwen3_graph_sized(&cfg, &mut wm, 1, 4, false, false).unwrap();
        let out = g.outputs[0];
        let dims: Vec<usize> = g
            .shape(out)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        assert_eq!(dims, vec![1, 4, cfg.hidden_size]);
    }

    #[test]
    fn rejects_indivisible_head_counts() {
        let mut cfg = tiny_cfg();
        cfg.num_key_value_heads = 3; // 4 % 3 != 0
        let mut wm = synthetic_weights(&cfg);
        let r = build_qwen3_graph_sized(&cfg, &mut wm, 1, 4, false, false);
        assert!(r.is_err());
    }

    fn synthetic_weights_with_lm_head(cfg: &Qwen3Config) -> WeightMap {
        let mut wm = synthetic_weights(cfg);
        let mut all: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
        for k in keys {
            let v = wm.take(&k).unwrap();
            all.insert(k, v);
        }
        all.insert(
            "lm_head.weight".into(),
            (
                vec![0f32; cfg.vocab_size * cfg.hidden_size],
                vec![cfg.vocab_size, cfg.hidden_size],
            ),
        );
        WeightMap::from_tensors(all)
    }

    #[test]
    fn lm_head_untied_produces_logits_shape() {
        let mut cfg = tiny_cfg();
        cfg.tie_word_embeddings = false;
        let mut wm = synthetic_weights_with_lm_head(&cfg);
        let (g, _) = build_qwen3_graph_sized(&cfg, &mut wm, 1, 4, true, false).unwrap();
        let out = g.outputs[0];
        let dims: Vec<usize> = g
            .shape(out)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        assert_eq!(dims, vec![1, 4, cfg.vocab_size]);
        assert_eq!(
            wm.len(),
            0,
            "unused weights: {:?}",
            wm.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn lm_head_tied_reuses_embed_weight() {
        let mut cfg = tiny_cfg();
        cfg.tie_word_embeddings = true;
        let mut wm = synthetic_weights(&cfg);
        // No lm_head.weight in the map — tied path must not request it.
        let (g, _) = build_qwen3_graph_sized(&cfg, &mut wm, 1, 4, true, false).unwrap();
        let out = g.outputs[0];
        let dims: Vec<usize> = g
            .shape(out)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        assert_eq!(dims, vec![1, 4, cfg.vocab_size]);
        assert_eq!(wm.len(), 0);
    }

    #[test]
    fn qwen3_rlx_toml_profile_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/qwen3.rlx.toml");
        let p = rlx_flow::CompileProfile::from_toml_path(&path).unwrap();
        assert!(p.passes.dce);
        assert!(!p.fusion.skip);
    }
}

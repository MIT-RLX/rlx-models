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

//! Qwen3.5 / Qwen3.6 — hybrid Gated DeltaNet + attention architecture.
//!
//! **Status:** dense `qwen35` and `qwen35moe` GGUF forward is wired end-to-end on CPU
//! (prefill, bucketed decode cache, optional MTP head, packed K-quants).
//! Every standard RLX backend accepts `--device` when the matching feature is
//! enabled (`cpu`, `metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`; build with
//! `all-backends`). Some GPU paths still run GDN or dequant matmul on the host
//! while the graph executes on the selected device. Numerical parity vs llama.cpp is env-gated
//! (`QWEN35_GGUF_PATH`, optional `parity-llama` feature).
//! See `PLAN.md` § Qwen3.5 for the remaining gap list (MoE, VLM parity hardening, …).
//!
//! # Architecture
//!
//! Qwen3 (and Qwen3.6 dense) is a pure transformer. Qwen3.5 replaces
//! most layers with a **gated DeltaNet** "linear attention" block and
//! inserts **standard full attention** every `full_attention_interval`
//! layers. An optional **MTP** (multi-token prediction) head at the top
//! uses full attention + shared LM head for speculative decoding.
//!
//! Per linear trunk layer (from `unsloth/Qwen3.5-0.8B-MTP-GGUF`):
//!
//! | tensor               | shape           | role |
//! |----------------------|-----------------|---|
//! | `attn_norm`          | `[1024]`        | RMS norm input |
//! | `attn_qkv`           | `[1024, 6144]`  | fused projection → `[gate, x, B, C, dt]` |
//! | `attn_gate`          | `[1024, 2048]`  | extra gating projection |
//! | `ssm_conv1d`         | `[4, 6144]`     | depthwise 1D conv (kernel=4) |
//! | `ssm_dt.bias`        | `[16]`          | per-rank delta-t bias |
//! | `ssm_a`              | `[16]`          | A diagonal log per rank |
//! | `ssm_alpha.weight`   | `[1024, 16]`    | α projection into dt_rank |
//! | `ssm_beta.weight`    | `[1024, 16]`    | β projection into dt_rank |
//! | `ssm_norm.weight`    | `[128]`         | norm over SSM state |
//! | `ssm_out.weight`     | `[2048, 1024]`  | output projection |
//! | `post_attention_norm`| `[1024]`        | post-SSM RMS norm |
//! | `ffn_{gate,up,down}` | standard SwiGLU | MLP |
//!
//! The MTP layer at index `block_count - nextn_predict_layers` switches
//! to standard attention (`attn_q/k/v/output` + `nextn.*` tensors).
//!
//! # Entry points
//!
//! - [`build_qwen35_graph_sized`] — one-shot prefill graph
//! - [`build_qwen35_prefill_cache_graph`] / [`build_qwen35_decode_graph`] —
//!   incremental generation with GDN + KV cache
//! - [`Qwen35Runner`] — high-level prefill / generate / spec-decode API
//! - [`validate_device`] — CPU, Metal, MLX, CUDA, ROCm, WGPU, Vulkan

mod builder;
mod cache;
mod capabilities;
mod chat;
pub mod cli;
mod config;
pub mod execution;
mod flow;
#[cfg(feature = "parity-llama")]
pub mod llama_reference;
mod lm_head;
mod moe_offload;
mod moe_store;
mod profile;
mod rope;
mod runner;
mod spec;
mod spec_runner;
mod tokenizer;
mod vision;
mod weights;

/// Synthetic configs/weights for tests and `qwen35_inference` bench.
pub mod synth;

pub use chat::{
    ChatMessage, ChatRole, encode_chat, encode_chat_auto, format_chatml, messages_from_prompt,
    parse_messages_json,
};
pub use tokenizer::{
    decode_ids, decode_ids_auto, decode_ids_from_gguf, encode_prompt, encode_prompt_auto,
    encode_prompt_from_gguf, resolve_tokenizer_path,
};

pub use builder::{
    PackedParams, Qwen35BsLayout, build_qwen35_decode_graph, build_qwen35_decode_graph_ext,
    build_qwen35_decode_hir_dynamic_ext, build_qwen35_decode_hir_ext, build_qwen35_graph_sized,
    build_qwen35_graph_sized_ext, build_qwen35_hir_sized_ext, build_qwen35_layer_probe_graph,
    build_qwen35_prefill_cache_graph, build_qwen35_prefill_cache_graph_ext,
    build_qwen35_prefill_cache_hir_dynamic_ext, build_qwen35_prefill_cache_hir_ext,
    build_qwen35_prefill_hidden_cache_hir_dynamic_ext, build_qwen35_prefix_graph,
    build_qwen35_trunk_export_graph, emit_qwen35_full_attn_prefill_layer,
    emit_qwen35_gdn_prefill_layer, emit_qwen35_prefill_tail,
};
pub use cache::{
    Qwen35DecodeCache, Qwen35LayerState, build_decode_attention_mask, decode_step_feeds,
    last_token_indices, pack_input_ids, pad_kv_to_bucket, recurrent_output_count,
    seed_cache_from_outputs, slice_kv_from_bucket, zero_prompt_padding_kv, zero_recurrent_inputs,
};
pub use capabilities::{STANDARD_DEVICE_NAMES, STANDARD_DEVICES, validate_device};
pub use config::{FAST_MTP_VOCAB, Qwen35Config, mtp_draft_vocab_size};
pub use execution::{
    Qwen35CompileCache, decode_config, get_or_specialize_component, get_or_specialize_hir,
    get_or_specialize_hir_with_options, hidden_prefill_config, prefill_config,
};
pub use flow::{
    Qwen35DecodeFlow, Qwen35DecodeOpts, Qwen35Flow, Qwen35LayerCtx, Qwen35LayerProbeFlow,
    Qwen35PrefillCacheFlow, Qwen35PrefillCacheOpts, Qwen35TrunkExportOpts,
    build_qwen35_decode_built, build_qwen35_decode_flow, build_qwen35_decode_model_flow,
    build_qwen35_layer_probe_model_flow, build_qwen35_prefill_built,
    build_qwen35_prefill_cache_built, build_qwen35_prefill_cache_flow,
    build_qwen35_prefill_cache_model_flow, build_qwen35_prefill_flow,
    build_qwen35_prefill_flow_built, build_qwen35_prefill_flow_ext,
    build_qwen35_runtime_mrope_prefill_flow, build_qwen35_trunk_export_built,
    build_qwen35_trunk_export_flow, build_qwen35_trunk_export_model_flow,
};
pub use moe_offload::{
    MoeOffloadState, build_expert_pool, build_moe_offload, count_moe_ffn_layers,
    decode_should_refresh, expert_param_bytes_f32, num_moe_ffn_layers,
};
pub use moe_store::{build_moe_expert_store, moe_host_bind_from_store, moe_layer_indices};
pub use profile::{QWEN35_PROFILE_FILE, qwen35_profile_default, qwen35_profile_near_weights};
pub use rlx_llada2::tide::{
    BlockDenoiseConfig, BlockDenoiseLoop, LLaDA2MoeConfig, PredictiveOffloadInfo,
    PredictiveOffloadParams, TideOffloadStats, aggregate_offload_stats,
    enable_predictive_expert_offload as tide_enable_offload, refresh_experts,
};
pub use rlx_qwen3::sampling::{SampleOpts, sample_token};
pub use rope::{
    mrope_prefill_feeds, mrope_row_for_sections, mrope_slice_at_pos, supports_multimodal_mrope,
    text_section_pos,
};
pub use runner::{
    Qwen35ConfigSource, Qwen35PrefillOutput, Qwen35PrefillSeed, Qwen35Runner, Qwen35RunnerBuilder,
};
pub use spec::{Qwen35MtpDraft, Qwen35TrunkTarget, speculative_decode_round};
pub use spec_runner::{Qwen35SpecRunner, Qwen35SpecRunnerBuilder};
pub use vision::{
    MEDIA_MARKER, MmProjConfig, MmProjWeights, MultimodalPrefill, MultimodalPrompt,
    Qwen35VisionEncoder, Qwen35VisionFlow, VISION_END, VISION_START, VisionEncodeOutput,
    build_multimodal_mrope_sections, build_qwen35_vision_built, build_qwen35_vision_graph,
    build_qwen35_vision_hir, build_vision_positions, image_chunk_n_pos, image_decoder_pos,
    load_vision_encoder, merge_text_and_vision_embd, preprocess_rgb,
};
#[cfg(feature = "qwen35-vlm")]
pub use vision::{encode_image_file, load_rgb_image};
pub use weights::{
    MatWeight, Qwen35FullAttnLayer, Qwen35LayerFfn, Qwen35LinearLayer, Qwen35MoeFfn,
    Qwen35MtpLayer, Qwen35TrunkLayer, Qwen35Weights,
};

/// Legacy redirect — qwen35 forward is implemented via
/// [`build_qwen35_graph_sized`]. Kept so older call sites get a
/// clear message instead of a missing-symbol error.
pub fn build_qwen35_graph_sized_stub(_cfg: &Qwen35Config) -> anyhow::Result<()> {
    Err(anyhow::anyhow!(
        "Qwen3.5 forward is implemented — use `build_qwen35_graph_sized` \
         or `Qwen35RunnerBuilder` instead of this stub."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory GGUF with `general.architecture =
    /// "qwen35"` + the keys [`Qwen35Config::from_gguf`] needs, then
    /// verify parsing succeeds and the stub builder returns a
    /// non-empty error.
    #[test]
    fn parses_qwen35_config_and_stub_errors_cleanly() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        let kv_count_off = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder KV count

        let write_string = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };

        let mut n_kv: u64 = 0;
        write_string(&mut buf, "general.architecture", "qwen35");
        n_kv += 1;
        write_u32(&mut buf, "qwen35.block_count", 25);
        n_kv += 1;
        write_u32(&mut buf, "qwen35.nextn_predict_layers", 1);
        n_kv += 1;
        write_u32(&mut buf, "qwen35.embedding_length", 1024);
        n_kv += 1;
        write_u32(&mut buf, "qwen35.feed_forward_length", 3584);
        n_kv += 1;
        write_u32(&mut buf, "qwen35.attention.head_count", 16);
        n_kv += 1;
        write_u32(&mut buf, "qwen35.attention.head_count_kv", 4);
        n_kv += 1;

        buf[kv_count_off..kv_count_off + 8].copy_from_slice(&n_kv.to_le_bytes());
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        let path = std::env::temp_dir().join("rlx_qwen35_config_test.gguf");
        std::fs::write(&path, &buf).unwrap();
        let raw = rlx_gguf::GgufFile::from_path(&path).unwrap();

        let cfg = Qwen35Config::from_gguf(&raw).unwrap();
        assert_eq!(cfg.num_hidden_layers, 25);
        assert_eq!(cfg.nextn_predict_layers, 1);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.intermediate_size, 3584);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.mtp_layer_start(), Some(24));

        let err = build_qwen35_graph_sized_stub(&cfg).unwrap_err().to_string();
        assert!(err.contains("build_qwen35_graph_sized"));
        assert!(err.contains("Qwen35RunnerBuilder"));

        std::fs::remove_file(&path).ok();
    }

    /// Same as [`parses_qwen35_config_and_stub_errors_cleanly`] but with
    /// `general.architecture = "qwen36"` and `qwen36.*` metadata keys.
    /// Confirms that the arch-prefix-aware lookup added in M1 reads the
    /// Qwen3.6 layout without falling back to Qwen3.5 defaults.
    #[test]
    fn parses_qwen36_config_via_arch_prefix() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let kv_count_off = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());

        let write_string = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };

        let mut n_kv: u64 = 0;
        write_string(&mut buf, "general.architecture", "qwen36");
        n_kv += 1;
        write_u32(&mut buf, "qwen36.block_count", 32);
        n_kv += 1;
        write_u32(&mut buf, "qwen36.embedding_length", 2048);
        n_kv += 1;
        write_u32(&mut buf, "qwen36.feed_forward_length", 5632);
        n_kv += 1;
        write_u32(&mut buf, "qwen36.attention.head_count", 32);
        n_kv += 1;
        write_u32(&mut buf, "qwen36.attention.head_count_kv", 8);
        n_kv += 1;

        buf[kv_count_off..kv_count_off + 8].copy_from_slice(&n_kv.to_le_bytes());
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        let path = std::env::temp_dir().join("rlx_qwen36_config_test.gguf");
        std::fs::write(&path, &buf).unwrap();
        let raw = rlx_gguf::GgufFile::from_path(&path).unwrap();

        let cfg = Qwen35Config::from_gguf(&raw).unwrap();
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 5632);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert!(!cfg.is_moe());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn qwen35_rlx_toml_profile_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/qwen35.rlx.toml");
        let p = rlx_flow::CompileProfile::from_toml_path(&path).unwrap();
        assert!(p.passes.dce);
        assert!(!p.fusion.skip);
    }
}

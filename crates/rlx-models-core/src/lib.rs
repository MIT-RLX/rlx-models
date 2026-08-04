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

//! Shared infrastructure for RLX model crates: HuggingFace config parsing,
//! safetensors / GGUF weight loading, tier-1 compile profile helpers, and
//! packed GGUF prefill guards ([`flow_bridge::packed_gguf_compile_guard`], etc.).

pub mod arch_registry;
/// Versatile model-asset loading, re-exported from the lean `rlx-assets` crate
/// so `rlx_core::asset_source::…` keeps working while lean crates can depend on
/// `rlx-assets` directly (no rlx compiler/runtime stack).
pub use rlx_assets as asset_source;
pub mod asr_bench;
pub mod asr_metrics;
pub mod audio;
pub mod audio_codec;
pub mod audio_ops_ir;
pub mod autoregressive;
pub mod codec_bench;
pub mod config;
pub mod dataprocessing;
pub mod device_capabilities;
pub mod distributed_bridge;
pub mod dsv4_opt;
pub mod dsv4_quant;
pub mod embedded_safetensors;
pub mod flow_bridge;
pub mod flow_util;
pub mod gguf_config;
pub mod gguf_resolve;
pub mod gguf_support;
pub mod gpu_kv;
pub mod host_kernels;
pub mod image_preprocess;
pub mod lm;
pub mod mlx_coverage;
pub mod mlx_npz_convert;
pub mod model_registry;
pub mod moe_weights;
pub mod prompt_cache;
pub mod resource;
pub mod safetensors_checkpoint;
pub mod standard_decoder;
pub mod vision_ops_ir;
pub mod weight_loader;
pub mod weight_map;
pub mod weight_registry;
pub mod weights;
pub mod weights_discover;

pub use asr_metrics::{
    EditCounts, WerAccumulator, batch_to_stream_factor, character_error_rate, edit_distance,
    normalize_words, rtfx, word_edit_counts, word_error_rate,
};
pub use audio::{resample_linear, resample_linear_interleaved};
pub use audio_codec::{
    AudioCodec, ChunkStreamer, CodecInfo, CompressStats, FileCodec, HierarchicalCodes, RvqCodes,
};
pub use device_capabilities::{
    LM_DEVICE_NAMES, LM_INFERENCE_DEVICE_PRIORITY, STANDARD_DEVICE_NAMES, STANDARD_DEVICES,
    device_memory_for_moe_offload, is_lm_device, is_standard_device, pick_lm_device,
    resolve_lm_device_str, validate_lm_device, validate_sam_device, validate_standard_device,
};

pub use gguf_config::{
    DINOV2_GGUF_ARCHES, EMBED_GGUF_ARCHES, EmbedGgufKind, FLUX_GGUF_ARCHES, GgufMemoryFootprint,
    SAM_GGUF_ARCHES, SAM2_GGUF_ARCHES, SAM3_GGUF_ARCHES, VJEPA2_GGUF_ARCHES, W2V_BERT_GGUF_ARCHES,
    embed_gguf_kind, gguf_memory_footprint, gguf_meta_u32, gguf_runner_hint, is_dinov2_gguf_arch,
    is_embed_gguf_arch, is_flux_gguf_arch, is_sam_gguf_arch, is_sam2_gguf_arch, is_sam3_gguf_arch,
    is_vjepa2_gguf_arch, is_w2v_bert_gguf_arch,
};
pub use gguf_resolve::{
    GgufTensorNameResolver, LlamaFamilyGgufResolver, PassThroughGgufResolver,
    PrefixStripGgufResolver, Qwen35NativeGgufResolver, register_gguf_tensor_resolver,
    resolve_gguf_tensor_name,
};
pub use gguf_support::{
    GgufModelFamily, ResolveWeightsOptions, assert_gguf_family, gguf_architecture_from_path,
    gguf_architecture_str, gguf_f32_bytes_estimate, gguf_family_for_arch,
    gguf_safetensors_only_hint, gguf_split_hint, gguf_split_siblings, gguf_validate_arch,
    laguna_allow_f32_expand, list_gguf_files_in_dir, load_gguf_file, resolve_weights_file,
    resolve_weights_file_with_options,
};

pub use asset_source::{AssetProvider, AssetSource, LocalDir, SourceSpec, load_materialized};
pub use autoregressive::{
    KvCacheState, compact_bucketed_kv_buffer, compile_cache_ensure_graph, infer_prefill_kv_seq,
    kv_from_prefill_outputs, kv_from_prefill_outputs_per_layer,
    packed_prefill_active_extent_enabled, past_kv_input_names, prefill_cache_key,
    run_bucketed_kv_decode, run_bucketed_kv_decode_graph_layers_scratch,
    run_bucketed_kv_decode_hir, run_bucketed_kv_decode_hir_scratch,
    run_bucketed_kv_decode_hir_uniform, run_bucketed_kv_decode_keyed,
    run_bucketed_kv_decode_keyed_batched, run_packed_prefill, split_bucketed_decode_kv,
    split_bucketed_decode_kv_per_layer, split_decode_logits_kv, split_decode_logits_kv_aux,
};
pub use config::{BertConfig, NomicBertConfig, NomicVisionConfig};
pub use distributed_bridge::{
    LoadKind, LoaderParamSource, ManifestParamSource, MapParamSource, StructureLoader,
    run_decoder_pipeline_local,
};
pub use embedded_safetensors::EmbeddedSafetensors;
pub use flow_bridge::{
    apply_compile_profile, compile_graph_encoder, compile_graph_gemma_decode,
    compile_graph_gemma_prefill, compile_graph_legacy, compile_graph_llama32_decode,
    compile_graph_llama32_prefill, compile_graph_qwen3_decode, compile_graph_qwen3_prefill,
    compile_graph_qwen35_decode, compile_graph_qwen35_prefill, compile_graph_sam,
    compile_graph_with_profile, compile_options_for_packed_gguf_prefill,
    compile_options_for_packed_gguf_prefill_with_profile, compile_options_for_profile,
    load_compile_profile, packed_gguf_compile_guard, packed_gguf_execution_device,
    profile_near_weights,
};
pub use flow_util::{
    WeightMapSource, bucket_cache_ensure_built, build_graph, built_from_graph, built_from_hir,
    built_from_hir_with_profile, compile_built, compile_built_cpu, compile_cache_ensure_built,
    compile_cache_ensure_built_with_options, compile_graph_encoder_with_params,
    compile_graph_gemma_decode_with_params, compile_graph_gemma_prefill_with_params,
    compile_graph_profile, compile_graph_qwen3_prefill_with_params,
    compile_graph_qwen35_decode_with_params, compile_graph_qwen35_prefill_with_params,
    compile_graph_sam_with_params, compile_graph_with_kv_export_params, graph_from_built,
    graph_from_hir,
};
pub use gguf_resolve::ensure_builtin_resolvers;
pub use gguf_support::DEFAULT_GGUF_PREFER_SUBSTR;
pub use gpu_kv::{
    GpuKvBinding, GpuKvCacheSet, cross_attn_gpu_handles_ready, device_supports_gpu_kv,
    install_cross_attn_gpu_handles, install_gpu_kv_handles, reinstall_gpu_kv_handles,
    run_bucketed_kv_decode_gpu, run_bucketed_kv_decode_gpu_hir, run_bucketed_kv_mtp_gpu,
    sync_gpu_kv_to_host,
};
pub use lm::{FlowBuildExt, into_compile_parts};
pub use mlx_coverage::{CoverageStatus, CoverageVia, ModelCoverage, classify_coverage};
pub use mlx_npz_convert::{MlxNpzConvertReport, convert_mlx_npz_to_hf, meta_name_to_hf};
pub use model_registry::{
    GgufModelRegistration, ensure_builtin_gguf_models, family_for_gguf_arch, hint_for_gguf_arch,
    lookup_gguf_arch, lookup_gguf_model_id, lookup_hf_model_type, register_gguf_model,
    registered_gguf_models, runner_for_gguf_arch, runner_for_hf_model_type,
};
pub use safetensors_checkpoint::{SafetensorsCheckpoint, SafetensorsMmapLoader};
pub use standard_decoder::{
    CachedExpertSource, DecoderSpec, DeepseekSpec, DeepseekV4Spec, ExpertSource, Glm4MoeSpec,
    GptOssSpec, HyV3Spec, KimiLinearSpec, Lfm2Spec, MinimaxSpec, ModelSupport, NemotronHSpec,
    PackedExpertSource, PagedGroupedMoe, PagedMlxExperts, RopeScaling, SharedExpertGpu,
    build_deepseek_mla, build_deepseek_prefill, build_deepseek_v4_prefill, build_deepseek_v4_stage,
    build_glm4moe_prefill, build_gpt_oss_prefill, build_hc_post, build_hc_pre, build_hc_sinkhorn,
    build_hy_v3_prefill, build_kimi_kda, build_kimi_linear_prefill, build_kv_compressor_overlap,
    build_kv_compressor_pool, build_lfm2_prefill, build_minimax_prefill, build_nemotron_h_prefill,
    build_standard_decoder_packed, build_v4_indexer_score, build_v4_o_lora,
    build_v4_sink_attention, build_v4_topk_gate, classify_config, deepseek_v4_generate_paged,
    deepseek_v4_generate_paged_layerwise, dense_swiglu_ffn, hash_route_experts, paged_moe_forward,
    paged_moe_forward_batched, paged_moe_io_profile_take, paged_moe_route,
};
pub use weight_loader::{
    ArcCacheLoader, ArcF32Tensor, GgufLoader, HfTranslatingLoader, WeightLoader,
    dequant_matmul_supported, ggml_type_to_quant_scheme, gguf_to_hf_name,
    gguf_to_hf_name_candidates, gguf_to_hf_name_for_arch, gguf_to_hf_qwen35_name, hf_to_gguf_name,
    hf_to_gguf_name_for_arch, is_mtp_weight, is_native_float_ggml, load_from_path,
};
pub use weight_map::{WeightDrainPolicy, WeightMap};
pub use weight_registry::{
    LoadWeightsOptions, LoadedWeights, RegisteredFormat, WeightFormatRegistration,
    format_for_extension, list_registered_formats, load_weight_map_resolved, load_weights_resolved,
    open_weight_loader, register_weight_format, registered_extensions_hint,
};
pub use weights::{
    GgufDirGuide, LoadOpts, ResolveOpts, default_resolve_opts, gguf_dir_guide, init,
    load_weight_map, open, open_map, open_map_with, open_with, pick, pick_default,
};
pub use weights_discover::{
    DiscoverOpts, DiscoveredFormat, DiscoveredWeight, WeightSourceKind, default_source_roots,
    looks_like_filesystem_path, resolve_weight_query, resolve_weight_query_in_roots,
    resolve_weights_path_or_query, scan_weights, scan_weights_in_roots,
};

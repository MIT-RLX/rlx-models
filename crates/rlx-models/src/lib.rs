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

//! RLX model loading — parse configs, load weights, build IR graphs.
//!
//! This crate is a thin facade over per-model workspace members (`rlx-qwen3`,
//! `rlx-sam`, …). Depend on a specific model crate directly when you only need
//! one family.

pub use rlx_core::{
    BertConfig, DiscoverOpts, DiscoveredFormat, DiscoveredWeight, EmbedGgufKind, FlowBuildExt,
    GgufDirGuide, GgufModelFamily, GgufModelRegistration, GgufTensorNameResolver,
    LlamaFamilyGgufResolver, LoadOpts, LoadWeightsOptions, LoadedWeights, NomicBertConfig,
    NomicVisionConfig, PassThroughGgufResolver, Qwen35NativeGgufResolver, RegisteredFormat,
    ResolveOpts, ResolveWeightsOptions, STANDARD_DEVICE_NAMES, WeightDrainPolicy,
    WeightFormatRegistration, WeightLoader, WeightMap, WeightMapSource, WeightSourceKind,
    arch_registry, assert_gguf_family, config, dataprocessing, default_source_roots,
    ensure_builtin_gguf_models, flow_bridge, flow_util, format_for_extension,
    gguf_architecture_str, gguf_dir_guide, gguf_f32_bytes_estimate, gguf_family_for_arch,
    gguf_resolve, gguf_runner_hint, gguf_support, into_compile_parts, is_standard_device,
    list_registered_formats, lm, load_from_path, load_weight_map_resolved, load_weights_resolved,
    looks_like_filesystem_path, model_registry, open as open_weights, open_map, open_map_with,
    open_with, register_gguf_model, register_gguf_tensor_resolver, register_weight_format,
    resolve_weight_query, resolve_weight_query_in_roots, resolve_weights_file,
    resolve_weights_file_with_options, resolve_weights_path_or_query, scan_weights,
    scan_weights_in_roots, validate_sam_device, validate_standard_device, vision_ops_ir,
    weight_loader, weight_map, weight_registry, weights, weights_discover,
};
pub use rlx_flow::{BuiltModel, CompileProfile};

#[cfg(feature = "bert")]
pub mod bert {
    pub use rlx_bert::bert::*;
}
#[cfg(feature = "bert")]
pub mod bert_flow {
    pub use rlx_bert::flow::*;
}
#[cfg(feature = "clinicalbert")]
pub mod clinicalbert {
    pub use rlx_clinicalbert::*;
}
#[cfg(feature = "nomic")]
pub mod nomic {
    pub use rlx_nomic::nomic::*;
}
#[cfg(feature = "nomic")]
pub mod nomic_flow {
    pub use rlx_nomic::flow::*;
}
#[cfg(feature = "vision")]
pub mod vision {
    pub use rlx_vision::vision::*;
}
#[cfg(feature = "vision")]
pub mod vision_flow {
    pub use rlx_vision::flow::*;
}
#[cfg(feature = "dinov2")]
pub mod dinov2 {
    pub use rlx_dinov2::*;
}
#[cfg(feature = "dinov3")]
pub mod dinov3 {
    pub use rlx_dinov3::*;
}
#[cfg(feature = "trellis2")]
pub mod trellis2 {
    pub use rlx_trellis2::*;
}
#[cfg(feature = "uni2")]
pub mod uni2 {
    pub use rlx_uni2::*;
}
#[cfg(feature = "hoct")]
pub mod hoct {
    pub use rlx_hoct::*;
}
#[cfg(feature = "bioclip2")]
pub mod bioclip2 {
    pub use rlx_bioclip2::*;
}
#[cfg(feature = "embed")]
pub mod embed {
    pub use rlx_embed::*;
}
#[cfg(feature = "flux2")]
pub mod flux2 {
    pub use rlx_flux2::*;
}
#[cfg(feature = "diamond")]
pub mod diamond {
    pub use rlx_diamond::*;
}
#[cfg(feature = "qwen3")]
pub mod qwen3 {
    pub use rlx_qwen3::*;
}
#[cfg(feature = "qwen35")]
pub mod qwen35 {
    pub use rlx_qwen35::*;
}
#[cfg(feature = "fara")]
pub mod fara {
    pub use rlx_fara::*;
}
#[cfg(feature = "qwen25-vl")]
pub mod qwen25_vl {
    pub use rlx_qwen25_vl::*;
}
#[cfg(feature = "qwen3-vl")]
pub mod qwen3_vl {
    pub use rlx_qwen3_vl::*;
}
#[cfg(feature = "lfm-vl")]
pub mod lfm_vl {
    pub use rlx_lfm_vl::*;
}
#[cfg(feature = "mistral-vl")]
pub mod mistral_vl {
    pub use rlx_mistral_vl::*;
}
#[cfg(feature = "llama4")]
pub mod llama4 {
    pub use rlx_llama4::*;
}
#[cfg(feature = "mllama")]
pub mod mllama {
    pub use rlx_mllama::*;
}
#[cfg(feature = "llama32")]
pub mod llama32 {
    pub use rlx_llama32::*;
}
#[cfg(feature = "gemma")]
pub mod gemma {
    pub use rlx_gemma::*;
}
#[cfg(feature = "llada2")]
pub mod llada2 {
    pub use rlx_llada2::llada2::*;
}
#[cfg(feature = "llada2")]
pub mod tide {
    pub use rlx_llada2::tide::*;
}
#[cfg(feature = "sam")]
pub mod sam {
    pub use rlx_sam::*;
}
#[cfg(feature = "sam2")]
pub mod sam2 {
    pub use rlx_sam2::*;
}
#[cfg(feature = "sam3")]
pub mod sam3 {
    pub use rlx_sam3::*;
}
#[cfg(feature = "vjepa2")]
pub mod vjepa2 {
    pub use rlx_vjepa2::*;
}
#[cfg(feature = "wav2vec2-bert")]
pub mod wav2vec2_bert {
    pub use rlx_wav2vec2_bert::*;
}
#[cfg(feature = "wav2vec2-asr")]
pub mod wav2vec2_asr {
    pub use rlx_wav2vec2_asr::*;
}
#[cfg(feature = "diarize")]
pub mod diarize {
    pub use rlx_diarize::*;
}
#[cfg(feature = "whisper")]
pub mod whisper {
    pub use rlx_whisper::*;
}
#[cfg(feature = "vad")]
pub mod vad {
    pub use rlx_vad::*;
}
#[cfg(feature = "wake")]
pub mod wake {
    pub use rlx_wake::*;
}
#[cfg(feature = "wakeword")]
pub mod wakeword {
    pub use rlx_wakeword::*;
}
#[cfg(feature = "openwakeword")]
pub mod openwakeword {
    pub use rlx_openwakeword::*;
}
#[cfg(feature = "nanowakeword")]
pub mod nanowakeword {
    pub use rlx_nanowakeword::*;
}
#[cfg(feature = "porcupine")]
pub mod porcupine {
    pub use rlx_porcupine::*;
}
#[cfg(feature = "voxrt")]
pub mod voxrt {
    pub use rlx_voxrt::*;
}
#[cfg(feature = "aec")]
pub mod aec {
    pub use rlx_aec::*;
}
#[cfg(feature = "voxtral")]
pub mod voxtral {
    pub use rlx_voxtral::*;
}
#[cfg(feature = "qwen3-asr")]
pub mod qwen3_asr {
    pub use rlx_qwen3_asr::*;
}
#[cfg(feature = "conformer-ctc")]
pub mod conformer_ctc {
    pub use rlx_conformer_ctc::*;
}
#[cfg(feature = "funasr")]
pub mod funasr {
    pub use rlx_funasr::*;
}
#[cfg(feature = "nemotron-asr")]
pub mod nemotron_asr {
    pub use rlx_nemotron_asr::*;
}
#[cfg(feature = "voxtral-tts")]
pub mod voxtral_tts {
    pub use rlx_voxtral_tts::*;
}

#[cfg(feature = "qwen3-tts")]
pub mod qwen3_tts {
    pub use rlx_qwen3_tts::*;
}
#[cfg(feature = "locateanything")]
pub mod locateanything {
    pub use rlx_locateanything::*;
}
#[cfg(feature = "unlimited-ocr")]
pub mod unlimited_ocr {
    pub use rlx_unlimited_ocr::*;
}
#[cfg(feature = "ocr")]
pub mod ocr {
    pub use rlx_ocr::*;
}
/// RLX streaming Conformer ASR (`rlx-asr`). Enable feature `streaming-asr`.
#[cfg(feature = "streaming-asr")]
pub mod streaming_asr {
    pub use rlx_asr::*;
}
/// PP-OCRv6 tiny/small OCR (native HIR + safetensors). Enable feature `ppocrv6`.
#[cfg(feature = "ppocrv6")]
pub mod ppocrv6 {
    pub use rlx_ppocrv6::*;
}
#[cfg(feature = "neutts")]
pub mod neutts {
    pub use rlx_neutts::*;
}
#[cfg(feature = "orpheus")]
pub mod orpheus {
    pub use rlx_orpheus::*;
}
#[cfg(feature = "kittentts")]
pub mod kittentts {
    pub use rlx_kittentts::*;
}
/// RLX FastSpeech2 + WaveRNN (`rlx-tts`). Enable feature `rlx-tts`.
/// Local pack: `weights/tts/rlx-tts/rlx-tts.gguf` (or `RLX_TTS_BUNDLE`).
#[cfg(feature = "rlx-tts")]
pub mod tts {
    pub use rlx_tts::*;
}
#[cfg(feature = "kokoro")]
pub mod kokoro {
    pub use rlx_kokoro::*;
}
#[cfg(feature = "supertonic")]
pub mod supertonic {
    pub use rlx_supertonic::*;
}
#[cfg(feature = "luxtts")]
pub mod luxtts {
    pub use rlx_luxtts::*;
}
#[cfg(feature = "zipvoice")]
pub mod zipvoice {
    pub use rlx_zipvoice::*;
}
#[cfg(feature = "moss-nano")]
pub mod moss_nano {
    pub use rlx_moss_nano::*;
}
#[cfg(feature = "maya1")]
pub mod maya1 {
    pub use rlx_maya1::*;
}
#[cfg(feature = "miratts")]
pub mod miratts {
    pub use rlx_miratts::*;
}
#[cfg(feature = "f5tts")]
pub mod f5tts {
    pub use rlx_f5tts::*;
}
#[cfg(feature = "piper")]
pub mod piper {
    pub use rlx_piper::*;
}
#[cfg(feature = "florence2")]
pub mod florence2 {
    pub use rlx_florence2::*;
}
#[cfg(feature = "mimi")]
pub mod mimi {
    pub use rlx_mimi::*;
}
#[cfg(feature = "tsac")]
pub mod tsac {
    pub use rlx_tsac::*;
}
#[cfg(feature = "moshi")]
pub mod moshi {
    pub use rlx_moshi::*;
}
#[cfg(feature = "neutts")]
pub use rlx_neutts::{
    BackboneModel, DEFAULT_N_CTX, GenerationConfig, NeuCodecDecoder, NeuCodecEncoder, NeuTTS,
    STOP_TOKEN, build_prompt, extract_ids,
};

#[deprecated(note = "use `rlx_models::ocr`")]
#[cfg(feature = "ocr")]
pub mod ocrs {
    pub use rlx_ocr::*;
}

// ── Stub families (PLAN.md M4 — no runner yet). Each exposes a
// `*Runner::builder().build()` that returns an error pointing at the
// milestone, so callers get a typed surface to wire against today.
#[cfg(feature = "mistral")]
pub mod mistral {
    pub use rlx_mistral::*;
}
#[cfg(feature = "bonsai")]
pub mod bonsai {
    pub use rlx_bonsai::*;
}
#[cfg(feature = "neutrino")]
pub mod neutrino {
    pub use rlx_neutrino::*;
}
#[cfg(feature = "minicpm5")]
pub mod minicpm5 {
    pub use rlx_minicpm5::*;
}
#[cfg(feature = "nanbeige")]
pub mod nanbeige {
    pub use rlx_nanbeige::*;
}
#[cfg(feature = "tinyllama")]
pub mod tinyllama {
    pub use rlx_tinyllama::*;
}
#[cfg(feature = "phi")]
pub mod phi {
    pub use rlx_phi::*;
}
#[cfg(feature = "omnicoder")]
pub mod omnicoder {
    pub use rlx_omnicoder::*;
}
#[cfg(feature = "granite")]
pub mod granite {
    pub use rlx_granite::*;
}
#[cfg(feature = "inkling")]
pub mod inkling {
    pub use rlx_inkling::*;
}
#[cfg(feature = "laguna")]
pub mod laguna {
    pub use rlx_laguna::*;
}
#[cfg(feature = "cohere")]
pub mod cohere {
    pub use rlx_cohere::*;
}
#[cfg(feature = "sam-ir")]
pub mod mask_hyper_matmul_ir {
    pub use rlx_sam_ir::mask_hyper_matmul_ir::*;
}
#[cfg(feature = "sam-ir")]
pub mod mask_prompt_ir {
    pub use rlx_sam_ir::mask_prompt_ir::*;
}
#[cfg(feature = "sam-ir")]
pub mod mlp_relu_ir {
    pub use rlx_sam_ir::mlp_relu_ir::*;
}
#[cfg(feature = "sam-ir")]
pub mod twoway_transformer_ir {
    pub use rlx_sam_ir::twoway_transformer_ir::*;
}

#[cfg(feature = "runner")]
pub mod run;
#[cfg(feature = "runner")]
mod sam_runner;

pub use rlx_core::flow_bridge::{
    apply_compile_profile, compile_graph_encoder, compile_graph_legacy,
    compile_graph_llama32_decode, compile_graph_llama32_prefill, compile_graph_qwen3_decode,
    compile_graph_qwen3_prefill, compile_graph_qwen35_decode, compile_graph_qwen35_prefill,
    compile_graph_sam, compile_graph_with_profile, load_compile_profile, profile_near_weights,
};
pub use rlx_core::flow_util::{
    build_graph, built_from_graph, built_from_hir, built_from_hir_with_profile, compile_built,
    compile_built_cpu, compile_graph_encoder_with_params, compile_graph_profile,
    compile_graph_qwen3_prefill_with_params, compile_graph_qwen35_decode_with_params,
    compile_graph_qwen35_prefill_with_params, compile_graph_sam_with_params, graph_from_built,
    graph_from_hir,
};

#[cfg(feature = "bert")]
pub use bert::build_bert_graph_sized;
#[cfg(feature = "bert")]
pub use bert_flow::{BertFlow, build_bert_built};
#[cfg(feature = "diarize")]
pub use diarize::{DiarizeConfig, DiarizeSession, SpeakerTurn as DiarizeSpeakerTurn};
#[cfg(feature = "dinov2")]
pub use dinov2::{
    DinoV2Built, DinoV2Config, DinoV2Flow, DinoV2PreprocessWeights, build_dinov2_built,
    build_dinov2_graph_sized,
};
#[cfg(feature = "embed")]
pub use embed::{
    Arch, BertTokenizer, EmbeddingModel, ImageEmbeddingModel, ModelArch, ModelInfo, Pooling,
    RlxBertModel, RlxEmbed, RlxNomicModel, RlxVisionModel, TokenizedBatch, assemble_vision_hidden,
    compile_model, detect_arch, embed_with_rlx, models_map,
};
#[cfg(feature = "flux2")]
pub use flux2::{
    DEFAULT_TEXT_ENCODER_LAYERS, Flux2CfgCombineFlow, Flux2CfgCombineGraph, Flux2Checkpoint,
    Flux2Config, Flux2Flow, Flux2ForwardBuilt, Flux2ForwardGraph, Flux2ForwardInput,
    Flux2GraphParams, Flux2PromptOutput, Flux2Session, Flux2SessionCache, Flux2SessionKey,
    Flux2TextEncoderBuilt, Flux2TextEncoderFlow, Flux2VaeConfig, Flux2VaeDecoderFlow,
    Flux2VaeEncoderFlow, Flux2VaeGraph, Flux2VaeWeights, Flux2Weights, build_flux2_cfg_combine_hir,
    build_flux2_forward_graph, build_flux2_forward_hir, build_flux2_minimal_graph,
    build_flux2_minimal_hir, build_flux2_text_encoder_hir, cfg_combine, compile_flux2_cfg_combine,
    compile_flux2_forward, compile_flux2_forward_via_flow, compile_flux2_minimal,
    compile_flux2_text_encoder_hir, download_flux2_repo, encode_flux2_prompt,
    encode_prompt_embeds_default_layers, encode_prompt_padded, extract_flux2_vae_weights,
    extract_flux2_weights, extract_text_encoder_weights, flux2_decode_packed_latents,
    flux2_prefers_compiled_hir, flux2_prefers_compiled_te, flux2_rgb_to_u8,
    flux2_transformer_forward, host_temb, load_and_apply_flux2_lora, load_flux2_vae_weights,
    load_flux2_weights, load_rgb_planar, load_text_encoder_weights, parse_lora_scale,
    prepare_latent_ids, prepare_text_ids, prepare_weight_map, resolve_text_encoder_dir,
    resolve_tokenizer_path, resolve_transformer_config, resolve_vae_dir, tiny_text_encoder_config,
};
#[cfg(feature = "gemma")]
pub use gemma::{
    GemmaArch, GemmaConfig, GemmaFlow, GemmaGenerator, build_gemma_decode_graph_sized,
    build_gemma_graph_sized, build_gemma_graph_sized_last_logits, build_gemma_graph_sized_packed,
    encode_prompt as gemma_encode_prompt, encode_prompt_auto as gemma_encode_prompt_auto,
    gemma_cfg_from_gguf, resolve_tokenizer_path as gemma_resolve_tokenizer_path,
};
#[cfg(feature = "llada2")]
pub use llada2::{
    LLaDA2MoeConfig, LLaDA2Runner, LLaDA2RunnerBuilder, LLaDA2Weights, build_llada2_forward_graph,
    default_memory_budget_bytes, validate_device as validate_llada2_device,
};
#[cfg(feature = "llama32")]
pub use llama32::{
    Llama32Config, Llama32Flow, Llama32Generator, build_llama32_decode_graph_sized,
    build_llama32_decode_graph_sized_ext, build_llama32_graph_sized,
    build_llama32_graph_sized_last_logits, build_llama32_graph_sized_packed,
    encode_prompt as llama32_encode_prompt, encode_prompt_auto as llama32_encode_prompt_auto,
    llama32_cfg_from_gguf, resolve_tokenizer_path as llama32_resolve_tokenizer_path,
};
#[cfg(feature = "nomic")]
pub use nomic::{build_nomic_diagnostic_graph, build_nomic_graph_sized};
#[cfg(feature = "nomic")]
pub use nomic_flow::{NomicFlow, build_nomic_built};
#[cfg(feature = "ocr")]
pub use ocr::{
    BLACK_VALUE, DEFAULT_ALPHABET, DecodeMethod, DetectionParams, DimOrder, HF_DETECTION_RTEN,
    HF_DETECTION_ST, HF_RECOGNITION_RTEN, HF_RECOGNITION_ST, ImageSource, OcrConfig, OcrEngine,
    OcrEngineParams, OcrInput, OcrOutput, OcrRunner, OcrRunnerBuilder, RotatedRect, TextChar,
    TextLine, TextWord, resolve_model_dir,
};
#[cfg(feature = "qwen3")]
pub use qwen3::{
    Qwen3Config, Qwen3Flow, Qwen3Generator, Qwen3PrefillOpts, Qwen3Speculator, SampleOpts,
    build_qwen3_graph_sized, build_qwen3_prefill_built, sample_token,
};
#[cfg(feature = "qwen3-tts")]
pub use qwen3_tts::{
    HF_MODEL_ID_06B_CUSTOM as QWEN3_TTS_HF_MODEL_ID, PRESET_SPEAKERS as QWEN3_TTS_SPEAKERS,
    Qwen3TtsBenchReport, Qwen3TtsConfig, Qwen3TtsRunner, Qwen3TtsWeightStore, TalkerEngine,
};
#[cfg(feature = "qwen35")]
pub use qwen35::{
    ChatMessage, ChatRole, MatWeight, Qwen35Config, Qwen35FullAttnLayer, Qwen35LayerFfn,
    Qwen35LinearLayer, Qwen35MoeFfn, Qwen35MtpLayer, Qwen35PrefillOutput, Qwen35Runner,
    Qwen35RunnerBuilder, Qwen35TrunkLayer, Qwen35Weights, build_qwen35_decode_graph,
    build_qwen35_decode_hir_dynamic_ext, build_qwen35_graph_sized, build_qwen35_graph_sized_ext,
    build_qwen35_graph_sized_stub, build_qwen35_prefill_cache_graph,
    build_qwen35_prefill_cache_graph_ext, build_qwen35_prefill_cache_hir_dynamic_ext,
    decode_step_feeds, encode_chat_auto, format_chatml, messages_from_prompt, mrope_prefill_feeds,
    mrope_row_for_sections, mrope_slice_at_pos, mtp_draft_vocab_size, pack_input_ids,
    parse_messages_json, recurrent_output_count, seed_cache_from_outputs,
    supports_multimodal_mrope, synth as qwen35_synth, text_section_pos, validate_device,
    zero_recurrent_inputs,
};
#[cfg(feature = "flux2")]
pub use rlx_flux2::{Flux2Output, Flux2Runner, Flux2RunnerBuilder};
#[cfg(feature = "runner")]
pub use run::{
    ConfigSource, DinoV2Output, DinoV2Runner, DinoV2RunnerBuilder, DinoV2Variant, Llama32Runner,
    Llama32RunnerBuilder, LmRunner, ModelRunner, Precision, Qwen3Runner, Qwen3RunnerBuilder,
    SamArch, SamPredictionAny, SamRunner, SamRunnerBuilder, Vjepa2Output, Vjepa2PoolOutput,
    Vjepa2PredictOutput, Vjepa2Runner, Vjepa2RunnerBuilder, Wav2Vec2BertRunner,
    Wav2Vec2BertRunnerBuilder, WeightFormat, debug_resolve_name, dispatch, dispatch_help,
    list_mtp_keys, open_gguf_loader, open_loader, open_loader_resolved, open_loader_with_format,
    register_runner, registered_runners, run_registered,
};
#[cfg(feature = "sam")]
pub use sam::{
    NeckWeights as SamNeckWeights, SamConfig, SamEncoderBuilt, SamEncoderConfig, SamEncoderFlow,
    SamPreprocessWeights, apply_neck_host as sam_apply_neck_host,
    assemble_patch_tokens as sam_assemble_patch_tokens, build_sam_encoder_built,
    build_sam_encoder_graph, preprocess_image as sam_preprocess_image,
};
#[cfg(feature = "sam2")]
pub use sam2::{
    FpnLevel as Sam2FpnLevel, FpnNeckWeights as Sam2FpnNeckWeights, Sam2, Sam2Config,
    Sam2DecoderConfig, Sam2FpnConfig, Sam2HieraConfig, Sam2ImageEncoderBuilt, Sam2ImageEncoderFlow,
    Sam2ImagePrediction, Sam2MaskDecoderOutput, Sam2MaskDecoderWeights, Sam2MemoryAttentionWeights,
    Sam2MemoryConfig, Sam2MemoryEncoderConfig, Sam2MemoryEncoderOutput, Sam2MemoryEncoderWeights,
    Sam2PreprocessWeights, Sam2PromptEncoderOutput, Sam2PromptEncoderWeights,
    Sam2TwoWayTransformerWeights, Sam2VideoState, apply_fpn_neck as sam2_apply_fpn_neck,
    apply_fpn_neck_host as sam2_apply_fpn_neck_host,
    assemble_patch_tokens as sam2_assemble_patch_tokens, build_sam2_image_encoder_built,
    build_sam2_image_encoder_graph, mask_decoder_forward as sam2_mask_decoder_forward,
    memory_attention_forward as sam2_memory_attention_forward,
    memory_encoder_forward as sam2_memory_encoder_forward,
    preprocess_image as sam2_preprocess_image,
    prompt_encoder_forward as sam2_prompt_encoder_forward,
    two_way_transformer_forward as sam2_two_way_transformer_forward,
};
#[cfg(feature = "sam3")]
pub use sam3::{
    Sam3, Sam3CompiledDecoder, Sam3Config, Sam3DetectorConfig, Sam3DetectorDecoderBuilt,
    Sam3DetectorDecoderFlow, Sam3DetectorEncoderFlow, Sam3EncodedImage, Sam3ImagePrediction,
    Sam3PreprocessWeights, Sam3TextConfig, Sam3TrackerConfig, Sam3VideoFramePrediction,
    Sam3VideoState, Sam3VitConfig, assemble_patch_tokens as sam3_assemble_patch_tokens,
    build_sam3_detector_decoder_built, build_sam3_detector_encoder_built,
    build_sam3_detector_encoder_graph, forward_decoder_ir_on,
    preprocess_image as sam3_preprocess_image,
};
#[cfg(feature = "llada2")]
pub use tide::{
    BlockDenoiseConfig, BlockDenoiseLoop, GenerateConfig, PredictiveOffloadInfo,
    PredictiveOffloadParams, TideOffloadStats, TideRunner, aggregate_offload_stats,
    refresh_experts,
};
#[cfg(feature = "uni2")]
pub use uni2::{Uni2Built, Uni2Config, Uni2Flow, Uni2Output, Uni2Runner, build_uni2_built};
#[cfg(feature = "vision")]
pub use vision::{VisionPreprocessWeights, build_vision_graph_sized};
#[cfg(feature = "vision")]
pub use vision_flow::{NomicVisionBuilt, NomicVisionFlow, build_nomic_vision_built};
#[cfg(feature = "vjepa2")]
pub use vjepa2::{
    Vjepa2Config, Vjepa2EncoderBuilt, Vjepa2EncoderFlow, Vjepa2EncoderOutput, Vjepa2EncoderWeights,
    Vjepa2Masks, Vjepa2ModelWeights, Vjepa2PatchEmbedWeights, Vjepa2PoolerFlow,
    Vjepa2PoolerWeights, Vjepa2PredictorFlow, Vjepa2PredictorWeights,
    build_vjepa2_encoder_graph_sized, conv3d_patch_embed, encode_video_native,
    extract_encoder_weights, extract_model_weights, extract_patch_embed_weights,
    extract_pooler_weights, extract_predictor_weights, normalize_video_hwc, pool_native,
    predict_native,
};
#[cfg(feature = "voxtral")]
pub use voxtral::{
    FAMILY as VOXTRAL_FAMILY, HF_MODEL_ID_MINI_3B, LanguageModelPrefixLoader,
    MelSpectrogram as VoxtralMel, VoxtralAudioConfig, VoxtralConfig, VoxtralRunner,
    VoxtralRunnerBuilder, VoxtralWeightPrefix, build_voxtral_decode_built,
    build_voxtral_encoder_built, build_voxtral_prefill_built, build_voxtral_projector_built,
    fuse_inputs_embeds, pcm_to_mel as voxtral_pcm_to_mel, transcription_prompt_ids,
};
#[cfg(feature = "voxtral-tts")]
pub use voxtral_tts::{
    CodecDecoder, HF_MODEL_ID as VOXTRAL_TTS_HF_MODEL_ID, PRESET_VOICES as VOXTRAL_TTS_VOICES,
    VoxtralTtsBenchReport, VoxtralTtsConfig, VoxtralTtsRunner, VoxtralTtsWeightStore,
};
#[cfg(feature = "wav2vec2-asr")]
pub use wav2vec2_asr::{AlignSession, AlignedWord, Wav2Vec2AsrConfig, align_model_for_language};
#[cfg(feature = "wav2vec2-bert")]
pub use wav2vec2_bert::{
    LogMelExtractor, LogMelFeatures, Wav2Vec2BertConfig, Wav2Vec2BertFlow,
    Wav2Vec2BertPreprocessConfig, build_wav2vec2_bert_built, build_wav2vec2_bert_graph_sized,
    load_wav_mono_f32,
};
#[cfg(feature = "whisper")]
pub use whisper::{
    MelSpectrogram, SubtitleFormat, TranscriptSegment, WhisperConfig, WhisperDecoderFlow,
    WhisperEncoderFlow, WhisperKvCache, WhisperPipeline, WhisperPipelineOpts, WhisperRunner,
    WhisperRunnerBuilder, WhisperTranscript, WhisperWeightPrefix, WordAlignMode, WordTiming,
    build_whisper_decode_step_built, build_whisper_decoder_built,
    build_whisper_decoder_graph_sized, build_whisper_decoder_prefill_built,
    build_whisper_encoder_built, build_whisper_encoder_graph_sized, default_mel_frames, pcm_to_mel,
    to_json_pretty, to_srt, to_tsv, to_vtt,
};

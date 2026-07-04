// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Orpheus TTS — Llama-3B speech LM (GGUF) + SNAC 24 kHz neural codec decoder.
//
// SNAC decode: host eager safetensors (default) or native RLX CoreML (`Device::Ane`,
// feature `coreml`). See [README](README.md) for weights, CLI, and bundled MP4 samples.

pub mod decoder;
pub mod download;
pub mod tokens;

#[cfg(feature = "llama")]
pub mod backbone;
#[cfg(feature = "llama")]
pub mod device;

pub mod runner;

#[cfg(feature = "llama")]
pub mod cli;

pub use decoder::{
    SAMPLE_RATE, SAMPLES_PER_FRAME, SnacBackend, SnacDecoder, SnacExec, SnacLoadOptions,
};
pub use download::{
    DEFAULT_ORPHEUS_DIR, DEFAULT_ORPHEUS_QUANT, DEFAULT_SNAC_DIR, HF_ORPHEUS_FT_GGUF_REPO,
    HF_SNAC_REPO, SNAC_DECODER_SAFETENSORS, default_hf_cache_dir, default_orpheus_dir,
    default_snac_dir, fetch_default, fetch_orpheus_gguf, fetch_snac_raw, orpheus_gguf_filename,
    print_snac_export_hint, resolve_orpheus_quant, snac_decoder_path,
};
pub use runner::{
    GenerationConfig, OrpheusTts, SynthesisResult, decode_orpheus_codes, normalize_pcm_peak,
};
pub use tokens::{
    CUSTOM_TOKEN_BASE, END_OF_SPEECH_ID, SNAC_TOKEN_OFFSET, START_OF_SPEECH_ID, STOP_TOKEN_ID,
    VOICES, VoiceCloneReference, accept_orpheus_stream_token, custom_token_id_to_code,
    frame_codes_to_token_ids, generated_ids_to_snac_codes, is_custom_token_id,
    orpheus_frame_codes_to_token_ids, pack_orpheus_codes,
};

#[cfg(feature = "llama")]
pub use backbone::{BackboneLoadOptions, BackboneModel, DEFAULT_COMPILE_SEQ_CAP, DEFAULT_N_CTX};

#[cfg(feature = "llama")]
pub use rlx_llama32::MetalGgufPrefillMode;

#[cfg(feature = "llama")]
pub use device::{
    OrpheusRuntimeDevice, default_snac_exec, lm_kv_decode_supported, parse_orpheus_device,
    preferred_synth_device, preferred_synth_device_lenient, resolve_orpheus_device,
    synth_device_for_tests,
};

#[cfg(feature = "llama")]
pub use tokens::{build_prompt, build_prompt_ids, build_voice_clone_prompt_ids};

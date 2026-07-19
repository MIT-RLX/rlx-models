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

//! **Qwen3-TTS** on RLX — talker (Qwen3-0.6B-shaped backbone) + code predictor + 12Hz codec.
//!
//! Full guide, audio samples, and charts: [`README.md`](../README.md).
//!
//! ```text
//! text + speaker → talker (group-0 codec AR) → code predictor (groups 1–15) → speech tokenizer decode → wav
//! ```
//!
//! **High-level API** — [`VoiceClone`] (Base checkpoint): `extract_reference` + `generate` /
//! [`VoiceClone::generate_stream`] with [`StreamMode::Batched`] or [`StreamMode::Progressive`].
//! Progressive partial-decode: Metal/MLX default to CPU speech
//! (`progressive_speech_decode_device`); set `RLX_QWEN3_TTS_PROGRESSIVE_SPEECH_GPU=1`
//! for on-device progressive once prefix parity holds.
//!
//! **Examples** — `bidirectional_voice_chat` (Whisper → Qwen3 LM → TTS), `jfk_voice_clone`,
//! `streaming_walkthrough`. Run `just voice-chat-demo` from the repo root for a bundled roundtrip.
//!
//! **Internals**
//! - Compiled talker via [`talker::TalkerEngine`] (`inputs_embeds` + external `codec_head`)
//! - Native code predictor + CustomVoice greedy synthesis ([`config::GenerationConfig::greedy_for_model_dir`])
//! - Native 12Hz speech tokenizer decode (GPU `pre_transformer` when available; conv/vocoder CPU)
//! - [`megakernel::Qwen3TtsMegakernel`] — warmed talker + CP per backend profile
//! - [`fused_e2e::E2EPipelinePlan`] — target: RLX-fused graphs end-to-end (talker + CP + speech)
//! - [`voice_clone`] — Base-model ICL / x-vector prompt builders
//! - [`speaker_encoder`] — native ECAPA-TDNN over Base `speaker_encoder.*` tensors
//!   (XVectorOnly clone is end-to-end Rust; ICL still needs the Mimi encoder)

pub mod bench;
pub mod cli;
pub mod code_predictor;
pub mod codec_frame;
pub mod codec_frame_fused;
pub mod compile_opts;
pub mod config;
pub mod cp_frame;
pub mod cp_greedy;
pub mod cp_megakernel;
pub mod fused_e2e;
pub mod fusion_bench;
pub mod gpu_pipeline;
pub mod hir_stitch;
pub mod kv_util;
pub mod load;
pub mod megakernel;
#[cfg(feature = "speculative-decode")]
pub mod megakernel_speculative;
pub mod mrope;
pub mod options;
pub mod progress;
pub mod prompt;
pub mod runner;
pub mod session;
pub mod speaker_encoder;
pub mod speech_tokenizer;
pub mod stream;
pub mod synth_opts;
pub mod synthesize;
pub mod talker;
pub mod text_embed;
pub mod tokens;
pub mod voice_clone;
pub mod voice_clone_api;
pub mod weights;

pub use bench::Qwen3TtsBenchReport;
pub use code_predictor::{CpBenchBackend, CpBenchReport, bench_cp_ab, bench_cp_predict_groups};
pub use codec_frame::{
    Qwen3TtsGraphProfiles, Qwen3TtsGraphRole, build_qwen3_tts_codec_frame_decode_built,
    build_qwen3_tts_decode_built, build_qwen3_tts_prefill_built,
};
pub use codec_frame_fused::{CodecFrameFusedEngine, build_qwen3_tts_codec_frame_built};
pub use config::{
    GenerationConfig, HF_MODEL_ID_06B_BASE, HF_MODEL_ID_06B_CUSTOM, HF_TOKENIZER_12HZ,
    Qwen3TtsConfig, TalkerConfig,
};
pub use cp_frame::{
    CP_DECODE_BACKBONE_STEPS, CP_DECODE_STEPS, CP_PREFILL_TWO,
    build_qwen3_tts_cp_decode_step_built, build_qwen3_tts_cp_prefill_two_built,
};
pub use fused_e2e::{
    CodecFrameScratch, E2EPipelinePlan, StageBackend, codec_frame_fused_step,
    codec_frame_step_dispatch,
};
pub use fusion_bench::{
    FusionBenchSummary, TalkerDecodeBenchReport, bench_cp_fused_vs_eager_one, bench_fusion_ab,
};
pub use load::Qwen3TtsWeightStore;
pub use megakernel::Qwen3TtsMegakernel;
pub use options::{Qwen3TtsOptions, Qwen3TtsRunnerBuilder};
pub use runner::{Qwen3TtsRunner, write_wav_mono};
pub use session::Qwen3TtsSession;
pub use stream::{PcmChunk, StreamConfig, StreamControl, StreamEvent, StreamMode, StreamStats};
#[cfg(feature = "tokio")]
pub use stream::{PcmChunkReceiver, generate_chunks_tokio};
#[cfg(feature = "async")]
pub use stream::{PcmChunkStream, generate_chunks_async};
pub use talker::TalkerEngine;
pub use tokens::PRESET_SPEAKERS;
pub use voice_clone::{VoiceCloneMode, VoiceClonePrompt};
pub use voice_clone_api::{SpeakerReference, VoiceClone};

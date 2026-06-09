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

//! **Voxtral-4B-TTS** on RLX — Ministral LM + acoustic flow matching + codec decode.
//!
//! Native Rust port of vLLM-Omni `VoxtralTTSAudioGeneration` (no Python at inference).

pub mod acoustic;
pub mod acoustic_compiled;
pub mod acoustic_engine;
pub mod acoustic_flow;
pub mod backbone;
pub mod bench;
pub mod cli;
pub mod codec;
pub mod config;
pub mod decode_shard_layer;
pub mod generation;
pub mod lm_flow;
pub mod load;
pub mod lora;
pub mod math;
pub mod options;
pub mod prompt_tokens;
pub mod rng;
pub mod runner;
pub mod speech_tokenizer;
pub mod tokens;
pub mod voice;
pub mod voice_clone;
pub mod voice_pt;
pub mod weights;

pub use backbone::{CompiledMinistralLm, MinistralLm, NativeTtsEngine};
pub use bench::VoxtralTtsBenchReport;
pub use codec::CodecDecoder;
pub use config::{HF_MODEL_ID, VoxtralTtsConfig};
pub use generation::GenerationConfig;
pub use load::VoxtralTtsWeightStore;
pub use lora::load_lora_bank;
pub use options::{VoxtralTtsOptions, VoxtralTtsRunnerBuilder};
pub use prompt_tokens::load_prompt_tokens;
pub use runner::{VoxtralTtsRunner, parse_codes_file, write_wav_mono};
pub use tokens::PRESET_VOICES;
pub use voice::VoiceEmbedding;
pub use voice_clone::{
    VoiceCloneSupport, clone_from_reference_audio, encode_reference_wav,
    encode_reference_wav_to_file, max_reference_seconds, voice_clone_support,
};

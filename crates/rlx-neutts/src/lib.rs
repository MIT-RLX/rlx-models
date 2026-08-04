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

//! NeuTTS — on-device voice-cloning TTS (GGUF backbone + NeuCodec decoder).
//!
//! ## Backends
//!
//! | Feature              | Stack                                      |
//! |----------------------|--------------------------------------------|
//! | `codec`              | Eager ndarray NeuCodec (default)           |
//! | `llama`              | [`rlx_llama32::Llama32Runner`] + GGUF vocab |
//! | `parity-llama-cpp`   | `llama-cpp-2` reference for parity tests   |
//! | `burn`               | Burn GPU/CPU NeuCodec                      |
//! | `rlx`                | RLX runtime (codec eager-parity)           |

pub mod decoder;
pub mod features;
pub mod tokens;
pub mod variant;

#[cfg(feature = "llama")]
pub mod backbone;

pub mod runner;

pub use decoder::{
    ENCODER_DEFAULT_INPUT_SAMPLES, ENCODER_SAMPLE_RATE, ENCODER_SAMPLES_PER_TOKEN, NeuCodecDecoder,
    NeuCodecEncoder, SAMPLE_RATE, SAMPLES_PER_TOKEN,
};
pub use features::{
    backbone_feature_enabled, burn_feature_enabled, codec_feature_enabled, cuda_feature_enabled,
    enabled_backend_labels, gpu_feature_enabled, llama_cpp_feature_enabled, llama_feature_enabled,
    metal_feature_enabled, mlx_feature_enabled, parity_llama_cpp_feature_enabled,
    rlx_feature_enabled, rocm_feature_enabled, vulkan_feature_enabled, wgpu_feature_enabled,
};
pub use runner::{GenerationConfig, NeuTTS};
pub use tokens::{NUM_SPEECH_TOKENS, STOP_TOKEN, build_prompt, extract_ids, ids_to_token_str};
pub use variant::NeuTtsVariant;

#[cfg(feature = "llama")]
pub use backbone::{BackboneModel, DEFAULT_N_CTX};

#[cfg(feature = "parity-llama-cpp")]
pub use backbone::llama_cpp::LlamaCppBackbone;

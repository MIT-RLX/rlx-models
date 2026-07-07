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

//! MOSS-TTS-Nano — OpenMOSS 0.1B hierarchical AR codec-LM TTS for RLX (Apache-2.0).
//!
//! A pure-autoregressive "audio-tokenizer + LLM" TTS exported to ONNX. A global
//! 12-layer transformer (prefill + KV-cached `decode_step`) drives a fused local
//! sampled-frame graph that emits 16 audio-codebook tokens per frame; a separate
//! MOSS-Audio-Tokenizer decodes the codes to 48 kHz stereo audio. Runs on
//! ONNX Runtime (CPU + CoreML/CUDA EPs). Voice cloning via 18 builtin voices.

pub mod config;
#[cfg(feature = "onnx")]
pub mod model;

pub use config::{BuiltinVoice, CodecInfo, Manifest, TtsConfig};
#[cfg(feature = "onnx")]
pub use model::{MossNano, SynthOpts};
pub use rlx_runtime::{Device, parse_device};

/// Weights repo (Apache-2.0). Also needs the codec repo
/// `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX` under `codec/`.
pub const DEFAULT_HF_REPO: &str = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX";
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/moss-nano";

/// Peak absolute amplitude (audibility check).
pub fn peak_amplitude(audio: &[f32]) -> f32 {
    audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

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

//! Convenience re-exports for `rlx-gemma` users.
//!
//! ```rust,ignore
//! use rlx_gemma::prelude::*;
//!
//! // Now you can reach for the most-used types without cherry-picking:
//! let cfg = GemmaConfig::from_file("/path/to/gemma-4-12b/config.json")?;
//! let mm = GemmaMultimodalConfig::from_file("/path/to/gemma-4-12b/config.json")?;
//! let runner = GemmaRunner::builder()
//!     .weights("/path/to/gemma-4-12b/model.safetensors")
//!     .device(Device::Metal)
//!     .build()?;
//! ```
//!
//! Imports the minimum public surface area to drive every Gemma 4
//! workflow this crate supports: text inference, vision/audio
//! preprocessing, projector compilation, embedding fusion.

// ── Config + architecture detection ──────────────────────────────
pub use crate::config::{GemmaArch, GemmaConfig};

// ── Text inference ──────────────────────────────────────────────
pub use crate::generator::GemmaGenerator;
pub use crate::runner::{GemmaConfigSource, GemmaRunner, GemmaRunnerBuilder};

// ── Multimodal config + preprocessing ────────────────────────────
pub use crate::multimodal::{
    AUDIO_MARKER, GemmaAudioConfig, GemmaMultimodalConfig, GemmaVisionConfig, IMAGE_MARKER,
    ImageNormalize, MediaSlot,
};
pub use crate::multimodal::{
    expand_media_placeholders, extract_image_patches, extract_image_patches_normalized,
    frame_audio_samples, fuse_multimodal_embeddings, load_image_patches,
    load_image_patches_normalized, load_wav_mono_16khz, parse_wav_16khz_mono, resample_linear,
    tokenize_with_media,
};

// ── Multimodal runtime ──────────────────────────────────────────
pub use crate::multimodal_runner::{GemmaMultimodalRunner, MultimodalWeights};

// ── Tokenizer (shared with rlx-qwen35) ──────────────────────────
pub use crate::prompt::{decode_token_auto, encode_chat_prompt_auto};
pub use rlx_qwen35::{decode_ids_auto, encode_prompt, encode_prompt_auto, resolve_tokenizer_path};

// ── Devices + sampling (so callers don't need to import upstream)
pub use rlx_qwen3::SampleOpts;
pub use rlx_runtime::Device;

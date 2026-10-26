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

//! [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) 4.6 / 4.7 for RLX.
//!
//! The text model is a Qwen3.5 hybrid trunk (Gated DeltaNet + gated full
//! attention, dense or MoE), which [`rlx_qwen35`] loads straight from the HF
//! directory. This crate adds what MiniCPM-V puts in front of it:
//!
//! - [`preprocess`] — thumbnail + slice grid (`MiniCPMV4_6ImageProcessor`);
//! - [`vision`] — the SigLIP-400M tower with the window-attention merger
//!   after layer `insert_layer_id`, and the 2×2 merger into LM space;
//! - [`positions`] — 4.7's canvas M-RoPE (`get_rope_index`), or plain 1-D
//!   positions for 4.6;
//! - [`prompt`] — the chat template and image placeholder expansion;
//! - [`MiniCpmVRunner`] — all of it end to end.
//!
//! MiniCPM-V 2.x–4.5 (`model_type = minicpmv`, a perceiver resampler over
//! SigLIP) are a different architecture and are rejected by the config.

pub mod cli;
pub mod config;
#[cfg(feature = "export")]
pub mod export;
pub mod feeds;
pub mod gguf;
pub mod positions;
pub mod preprocess;
pub mod prompt;
pub mod resampler_gen;
pub mod runner;
pub mod video;
pub mod vision;

pub use config::{
    DownsampleMode, MarkerIds, MiniCpmVConfig, MropeMode, Projector, SliceConfig, VisionConfig,
};
pub use gguf::GgufPair;
pub use preprocess::{Crop, SlicedImage, slice_image};
pub use prompt::{IMAGE_PAD, Media, VIDEO_PAD, chat_prompt, chat_prompt_media};
pub use resampler_gen::{
    Generation, PackedVideo, ResamplerConfig, ResamplerRunner, ResamplerRunnerBuilder,
};
pub use runner::{
    AssembledPrompt, MiniCpmVRunner, MiniCpmVRunnerBuilder, RgbImage, VideoOpts, Visual,
    load_vision_weights,
};

/// Family name used in the catalog and logs.
pub const FAMILY: &str = "MiniCPM-V";

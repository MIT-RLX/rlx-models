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

//! MiniMax-H3 (Hailuo 3.0) — omni-modal video + audio generation.
//!
//! H3 generates a video and its synchronized soundtrack **jointly**, from one
//! flow-matching transformer running full self-attention over a single packed
//! sequence that holds text, conditioning media, audio rows and video rows at
//! once. There is no cross-attention and no per-modality block weights: the
//! only modality-specific parts are the two input patch projections, a per-row
//! AdaLN tag, and the two output heads.
//!
//! ## Components
//!
//! | Component | Module | Notes |
//! |---|---|---|
//! | Joint video+audio DiT | [`transformer`] | 50 blocks, hidden 5376, 56×128 heads, ~33B with ~13B in the AdaLN branches |
//! | Packed layout | [`layout`] | the `(t, h, w)` rotary grid, modality tags and row indices every stage addresses |
//! | Flow scheduler | [`scheduler`] | rectified-flow Euler, data-ward velocity, `shift` 12 (video) / 3 (audio) |
//! | Video VAE | [`vae_video`] / [`vae_video_encoder`] | 36-layer ViT decoder + 3D causal CNN encoder, 16× spatial / 4× temporal |
//! | Audio VAE | [`vae_audio`] | DAC/BigVGAN-style, 32 kHz, 800-sample hop, stereo via a shared per-channel path |
//! | Text conditioning | [`text_encoder`] / [`qwen3vl`] | the contract, and the tapped stack: Qwen3-VL read at layer 50, not the last |
//!
//! ## Tasks
//!
//! - `t2va` — text to video+audio.
//! - `i2va` — one keyframe anchors the first (or last) latent frame.
//! - `fl2va` — first *and* last keyframes anchor both ends.
//! - `ref2va` — arbitrary image / video / audio references, packed as blocks on
//!   a shared rotary clock. Uses the separate `transformer_ref` weights.
//!
//! ## Status
//!
//! The DiT, the packed layout for all four tasks, the scheduler, the sampling
//! loop, **both VAEs in both directions** and the Qwen3-VL conditioning tap are
//! implemented and covered by CPU tests. The two VAEs are additionally verified
//! against the real checkpoint, encode and decode, including round trips.
//!
//! Not done: numerical parity for the text encoder (structurally tested only —
//! the ~60 GB encoder was not fetched), image-bearing prompts (no vision tower),
//! and end-to-end generation on the released DiT weights (28 GB per partition).
//! See the crate README for the full ledger.

use anyhow::Result;
use std::path::Path;

pub mod config;
pub mod layout;
pub mod rope;
pub mod scheduler;
pub mod transformer;
pub mod vae_audio;
pub mod vae_video;
pub mod vae_video_encoder;
pub mod weights;

pub mod cli;
pub mod pipeline;
pub mod qwen3vl;
pub mod text_encoder;

#[cfg(feature = "tokenizer")]
pub mod tokenizer;

pub use config::{
    H3AudioVaeConfig, H3Config, H3SchedulerConfig, H3TextEncoderConfig, H3TransformerConfig,
    H3VideoVaeConfig, MODALITY_NUM, Modality,
};
pub use layout::{
    AUDIO_CHANNELS, H3Geometry, H3Reference, KeyframeAnchor, PackedLayout, RowTimesteps,
    build_packed_sequence, build_ref2va_packed_sequence, build_row_timesteps,
    patchify_video_latents, unpatchify_video_latents,
};
pub use pipeline::{H3Pipeline, H3Request, H3Task};
pub use qwen3vl::{H3Qwen3VlEncoder, compile_text_encoder};
pub use rope::RopeTables;
pub use scheduler::H3Scheduler;
pub use transformer::{CompiledH3Dit, H3DitInputs};
pub use vae_video_encoder::{DiagonalGaussian, H3VideoEncoder, Volume};

/// Family name used in catalogs and CLI output.
pub const FAMILY: &str = "MiniMax-H3";

/// The `_class_name` the released `model_index.json` carries.
pub const PIPELINE_CLASS: &str = "MiniMaxH3ModularPipeline";

/// Quick check that `root` looks like a MiniMax-H3 checkpoint.
pub fn is_minimax_h3_checkpoint(root: &Path) -> bool {
    root.join("model_index.json").is_file()
        && (root.join("transformer").join("config.json").is_file()
            || root.join("transformer_ref").join("config.json").is_file())
}

/// Load every component config from a checkpoint root.
pub fn load_config(root: &Path) -> Result<H3Config> {
    H3Config::from_root(root)
}

pub fn cli_run(args: &[String]) -> Result<()> {
    cli::run(args)
}

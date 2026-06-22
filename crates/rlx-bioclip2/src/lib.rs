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

//! BioCLIP-2 — OpenCLIP ViT-L-14 biology foundation model
//! ([`imageomics/bioclip-2`](https://huggingface.co/imageomics/bioclip-2)).
//!
//! Two transformer towers projecting images and text into a shared 768-dim
//! space, plus zero-shot classification. Tensor keys match the published
//! `open_clip_model.safetensors`, so checkpoints load with no remapping.
//!
//! Public entry points:
//!   - [`BioClip2Config`] — model dimensions (`vit_l_14`, `from_open_clip_json`)
//!   - [`BioClip2Runner`] — load + compile + encode image/text + zero-shot
//!   - [`build_vision_flow`] / [`build_text_flow`] — the IR graph builders
//!   - [`assemble_vision_hidden`] / [`clip_normalize_nchw`] — host-side image plumbing

pub mod cli;
pub mod config;
pub mod flow;
pub mod preprocess;
pub mod runner;
pub mod text_embed;
pub mod tokenizer;

pub use config::{BioClip2Config, CLIP_MEAN, CLIP_STD, LN_EPS, TextCfg, VisionCfg};
pub use flow::{build_text_flow, build_vision_flow};
pub use preprocess::{VisionEmbedWeights, assemble_vision_hidden, clip_normalize_nchw};
pub use runner::{BioClip2Runner, BioClip2RunnerBuilder, ensure_model_dir};
pub use text_embed::{TextEmbedWeights, assemble_text_hidden};
pub use tokenizer::{ClipTokenizer, EOT_TOKEN, SOT_TOKEN, eot_index};

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

//! SigLIP 2 — Google's sigmoid-loss vision-language model
//! ([blog](https://huggingface.co/blog/siglip2)), ported to RLX.
//!
//! Two transformer towers projecting images and text into a shared space,
//! plus sigmoid zero-shot classification. Tensor keys match the published
//! HuggingFace `model.safetensors`, so checkpoints load with no remapping.
//!
//! Two architecture families are supported:
//!   - **Fixed-resolution** (`model_type = "siglip"`): Conv2d patch stem,
//!     e.g. `google/siglip2-base-patch16-224`.
//!   - **NaFlex** (`model_type = "siglip2"`): variable resolution / aspect
//!     ratio (see [`naflex`]).
//!
//! Both use a pre-LN transformer with `gelu_pytorch_tanh`, an attention-
//! pooling (MAP) vision head, and a bidirectional text tower pooled at its
//! last position.
//!
//! Public entry points:
//!   - [`Siglip2Config`] — model dimensions (`base_patch16_224`, `from_hf_config_json`)
//!   - [`Siglip2Runner`] — load + compile + encode image/text + zero-shot
//!   - [`build_vision_flow`] / [`build_text_flow`] — the IR graph builders

pub mod cli;
pub mod config;
pub mod flow;
pub mod naflex;
pub mod preprocess;
pub mod runner;
pub mod text_embed;
pub mod tokenizer;

pub use config::{LN_EPS, SIGLIP_MEAN, SIGLIP_STD, Siglip2Config, TextCfg, Variant, VisionCfg};
pub use flow::{build_text_flow, build_vision_flow};
pub use preprocess::{
    PoolingWeights, VisionEmbedWeights, assemble_vision_hidden, siglip_normalize_nchw,
};
pub use runner::{Siglip2Runner, Siglip2RunnerBuilder, ensure_model_dir};
pub use text_embed::{TextEmbedWeights, assemble_text_hidden};
pub use tokenizer::{EOS_TOKEN, PAD_TOKEN, SiglipTokenizer};

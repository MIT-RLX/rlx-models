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

//! FLUX.2 Qwen3 text encoder — prompt → `encoder_hidden_states` for the denoiser.

pub mod flow;
pub mod forward;
pub mod hir_builder;
pub mod prompt;
pub mod tokenizer;
pub mod weights;

pub use flow::{Flux2TextEncoderBuilt, Flux2TextEncoderFlow, build_flux2_text_encoder_built};

pub use forward::{Flux2PromptOutput, encode_prompt_embeds, encode_prompt_embeds_default_layers};
pub use hir_builder::{
    Flux2TextEncoderGraph, build_flux2_text_encoder_hir, compile_flux2_text_encoder_hir,
};
pub use prompt::{
    DEFAULT_TEXT_ENCODER_LAYERS, TINY_TEXT_ENCODER_LAYERS, encode_flux2_prompt, prepare_text_ids,
    resolve_text_encoder_dir, tiny_text_encoder_config, tokenize_flux2_prompt,
};
pub use tokenizer::{encode_prompt, encode_prompt_padded, resolve_tokenizer_path};
pub use weights::{
    Flux2TextEncoderWeights, extract_text_encoder_weights, load_text_encoder_weights,
    synthetic_text_encoder_weights,
};

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

//! FLUX.2 text position ids and prompt encoding helpers.

use super::forward::{Flux2PromptOutput, encode_prompt_embeds};
use super::tokenizer::encode_prompt_padded;
use super::weights::Flux2TextEncoderWeights;
use anyhow::Result;
use rlx_qwen3::Qwen3Config;
use std::path::Path;

/// Default hidden-state indices for FLUX.2 Klein (matches mflux).
pub const DEFAULT_TEXT_ENCODER_LAYERS: &[usize] = &[9, 18, 27];

/// Tiny config for basic tests (2 layers → use layers `[1, 2]`).
pub const TINY_TEXT_ENCODER_LAYERS: &[usize] = &[1, 2];

/// Build FLUX.2-style text position ids `[batch, seq, 4]` flattened as `[seq*4]`.
pub fn prepare_text_ids(batch: usize, seq: usize) -> Vec<f32> {
    let mut ids = vec![0.0f32; batch * seq * 4];
    for b in 0..batch {
        for t in 0..seq {
            let base = (b * seq + t) * 4;
            ids[base + 3] = t as f32;
        }
    }
    ids
}

/// `Qwen3Config` sized for [`super::weights::synthetic_text_encoder_weights`] tests.
pub fn tiny_text_encoder_config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: true,
        attention_bias: true,
        qk_norm: true,
        sliding_window: None,
        max_window_layers: usize::MAX,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

/// End-to-end: tokenize (optional) + text encoder → embeddings + text ids.
pub fn encode_flux2_prompt(
    te_weights: &Flux2TextEncoderWeights,
    te_cfg: &Qwen3Config,
    input_ids: &[u32],
    batch: usize,
    seq: usize,
    hidden_state_layers: &[usize],
) -> Result<(Flux2PromptOutput, Vec<f32>)> {
    let out = encode_prompt_embeds(
        te_weights,
        te_cfg,
        input_ids,
        batch,
        seq,
        hidden_state_layers,
    )?;
    let txt_ids = prepare_text_ids(batch, seq);
    Ok((out, txt_ids))
}

/// Resolve `text_encoder/` next to a transformer weights file or model root.
pub fn resolve_text_encoder_dir(model_path: &Path) -> Option<std::path::PathBuf> {
    crate::paths::find_component_dir(model_path, "text_encoder")
}

/// Load tokenizer + encode with right padding to `seq_len`.
pub fn tokenize_flux2_prompt(
    tokenizer_path: &Path,
    prompt: &str,
    seq_len: usize,
) -> Result<Vec<u32>> {
    encode_prompt_padded(tokenizer_path, prompt, seq_len)
}

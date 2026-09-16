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

//! NLLB / M2M100 safetensors weight-key helpers.
//!
//! Keys match HuggingFace `M2M100ForConditionalGeneration` checkpoints
//! (`facebook/nllb-200-*`), e.g. `model.shared.weight`,
//! `model.encoder.layers.0.self_attn.q_proj.weight`.

/// Language (M2M100) weight keys under the `model.` prefix.
pub mod lang {
    /// Prefer this over `model.encoder.embed_tokens.weight` (tied).
    pub const SHARED: &str = "model.shared.weight";
    /// Alternate tied embedding key some exports keep.
    pub const ENC_EMBED_TOKENS: &str = "model.encoder.embed_tokens.weight";
    /// Optional generation bias buffer (often zeros / absent).
    pub const FINAL_LOGITS_BIAS: &str = "final_logits_bias";

    pub fn enc_embed_positions() -> String {
        "model.encoder.embed_positions.weight".into()
    }
    pub fn dec_embed_positions() -> String {
        "model.decoder.embed_positions.weight".into()
    }
    pub fn enc_layernorm_embedding_w() -> String {
        "model.encoder.layernorm_embedding.weight".into()
    }
    pub fn enc_layernorm_embedding_b() -> String {
        "model.encoder.layernorm_embedding.bias".into()
    }
    pub fn dec_layernorm_embedding_w() -> String {
        "model.decoder.layernorm_embedding.weight".into()
    }
    pub fn dec_layernorm_embedding_b() -> String {
        "model.decoder.layernorm_embedding.bias".into()
    }

    /// Optional final encoder LayerNorm (present on many M2M100 exports).
    pub fn enc_final_layer_norm_w() -> String {
        "model.encoder.layer_norm.weight".into()
    }
    pub fn enc_final_layer_norm_b() -> String {
        "model.encoder.layer_norm.bias".into()
    }
    pub fn dec_final_layer_norm_w() -> String {
        "model.decoder.layer_norm.weight".into()
    }
    pub fn dec_final_layer_norm_b() -> String {
        "model.decoder.layer_norm.bias".into()
    }

    pub fn enc_layer(layer: usize, suffix: &str) -> String {
        format!("model.encoder.layers.{layer}.{suffix}")
    }
    pub fn dec_layer(layer: usize, suffix: &str) -> String {
        format!("model.decoder.layers.{layer}.{suffix}")
    }
}

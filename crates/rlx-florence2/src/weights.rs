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

//! Florence-2 safetensors weight-key helpers.
//!
//! Keys match the published `microsoft/Florence-2-large` checkpoint, e.g.
//! `vision_tower.blocks.2.5.spatial_block.window_attn.fn.qkv.weight`,
//! `language_model.model.decoder.layers.0.encoder_attn.k_proj.weight`.

/// Vision (DaViT) weight keys.
pub mod vision {
    /// `vision_tower.convs.{stage}.proj.{weight,bias}`.
    pub fn conv_proj_w(stage: usize) -> String {
        format!("vision_tower.convs.{stage}.proj.weight")
    }
    pub fn conv_proj_b(stage: usize) -> String {
        format!("vision_tower.convs.{stage}.proj.bias")
    }
    /// `vision_tower.convs.{stage}.norm.{weight,bias}` (LayerNorm).
    pub fn conv_norm_w(stage: usize) -> String {
        format!("vision_tower.convs.{stage}.norm.weight")
    }
    pub fn conv_norm_b(stage: usize) -> String {
        format!("vision_tower.convs.{stage}.norm.bias")
    }

    /// Block prefix: `vision_tower.blocks.{stage}.{depth}.{spatial|channel}_block`.
    pub fn block(stage: usize, depth: usize, which: &str) -> String {
        format!("vision_tower.blocks.{stage}.{depth}.{which}")
    }

    /// Depthwise conv `{block}.{conv1|conv2}.fn.dw.{weight,bias}`.
    pub fn dw_w(block: &str, conv: &str) -> String {
        format!("{block}.{conv}.fn.dw.weight")
    }
    pub fn dw_b(block: &str, conv: &str) -> String {
        format!("{block}.{conv}.fn.dw.bias")
    }

    /// Attention norm `{block}.{attn}.norm.{weight,bias}` where attn is
    /// `window_attn` or `channel_attn`.
    pub fn attn_norm_w(block: &str, attn: &str) -> String {
        format!("{block}.{attn}.norm.weight")
    }
    pub fn attn_norm_b(block: &str, attn: &str) -> String {
        format!("{block}.{attn}.norm.bias")
    }
    pub fn attn_qkv_w(block: &str, attn: &str) -> String {
        format!("{block}.{attn}.fn.qkv.weight")
    }
    pub fn attn_qkv_b(block: &str, attn: &str) -> String {
        format!("{block}.{attn}.fn.qkv.bias")
    }
    pub fn attn_proj_w(block: &str, attn: &str) -> String {
        format!("{block}.{attn}.fn.proj.weight")
    }
    pub fn attn_proj_b(block: &str, attn: &str) -> String {
        format!("{block}.{attn}.fn.proj.bias")
    }

    /// FFN `{block}.ffn.norm.*` and `{block}.ffn.fn.net.{fc1,fc2}.*`.
    pub fn ffn_norm_w(block: &str) -> String {
        format!("{block}.ffn.norm.weight")
    }
    pub fn ffn_norm_b(block: &str) -> String {
        format!("{block}.ffn.norm.bias")
    }
    pub fn ffn_fc1_w(block: &str) -> String {
        format!("{block}.ffn.fn.net.fc1.weight")
    }
    pub fn ffn_fc1_b(block: &str) -> String {
        format!("{block}.ffn.fn.net.fc1.bias")
    }
    pub fn ffn_fc2_w(block: &str) -> String {
        format!("{block}.ffn.fn.net.fc2.weight")
    }
    pub fn ffn_fc2_b(block: &str) -> String {
        format!("{block}.ffn.fn.net.fc2.bias")
    }

    pub const IMAGE_PROJECTION: &str = "image_projection";
    pub const IMAGE_PROJ_NORM_W: &str = "image_proj_norm.weight";
    pub const IMAGE_PROJ_NORM_B: &str = "image_proj_norm.bias";
    pub const POS_ROW: &str = "image_pos_embed.row_embeddings.weight";
    pub const POS_COL: &str = "image_pos_embed.column_embeddings.weight";
    pub const TEMPORAL: &str = "visual_temporal_embed.pos_idx_to_embed";
}

/// Language (BART) weight keys.
pub mod lang {
    pub const SHARED: &str = "language_model.model.shared.weight";
    pub const FINAL_LOGITS_BIAS: &str = "language_model.final_logits_bias";

    pub fn enc_embed_positions() -> String {
        "language_model.model.encoder.embed_positions.weight".into()
    }
    pub fn dec_embed_positions() -> String {
        "language_model.model.decoder.embed_positions.weight".into()
    }
    pub fn enc_layernorm_embedding_w() -> String {
        "language_model.model.encoder.layernorm_embedding.weight".into()
    }
    pub fn enc_layernorm_embedding_b() -> String {
        "language_model.model.encoder.layernorm_embedding.bias".into()
    }
    pub fn dec_layernorm_embedding_w() -> String {
        "language_model.model.decoder.layernorm_embedding.weight".into()
    }
    pub fn dec_layernorm_embedding_b() -> String {
        "language_model.model.decoder.layernorm_embedding.bias".into()
    }

    pub fn enc_layer(layer: usize, suffix: &str) -> String {
        format!("language_model.model.encoder.layers.{layer}.{suffix}")
    }
    pub fn dec_layer(layer: usize, suffix: &str) -> String {
        format!("language_model.model.decoder.layers.{layer}.{suffix}")
    }
}

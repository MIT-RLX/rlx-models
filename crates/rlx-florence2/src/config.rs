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

//! Florence-2 configuration (DaViT vision backbone + BART text model).
//!
//! Mirrors `configuration_florence2.py`. The `large` preset matches
//! [`microsoft/Florence-2-large`](https://huggingface.co/microsoft/Florence-2-large).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// LayerNorm epsilon — PyTorch `nn.LayerNorm` default.
pub const LN_EPS: f32 = 1e-5;

/// ImageNet-style normalization used by the CLIP image processor.
pub const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// DaViT vision backbone configuration (`vision_config`).
#[derive(Debug, Clone)]
pub struct Florence2VisionConfig {
    pub dim_embed: Vec<usize>,
    pub depths: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub num_groups: Vec<usize>,
    pub patch_size: Vec<usize>,
    pub patch_stride: Vec<usize>,
    pub patch_padding: Vec<usize>,
    pub patch_prenorm: Vec<bool>,
    pub window_size: usize,
    pub projection_dim: usize,
    pub mlp_ratio: f64,
    /// `image_pos_embed.max_pos_embeddings`.
    pub image_pos_embed_max: usize,
    /// `visual_temporal_embedding.max_temporal_embeddings`.
    pub visual_temporal_max: usize,
    /// Ordered list of `spatial_avg_pool` / `temporal_avg_pool` / `last_frame`.
    pub image_feature_source: Vec<String>,
}

impl Florence2VisionConfig {
    pub fn num_stages(&self) -> usize {
        self.dim_embed.len()
    }
    /// Output channel dim of the backbone (`dim_embed[-1]`).
    pub fn dim_out(&self) -> usize {
        *self.dim_embed.last().unwrap()
    }
}

/// BART text model configuration (`text_config`).
#[derive(Debug, Clone)]
pub struct Florence2TextConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub max_position_embeddings: usize,
    pub scale_embedding: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub decoder_start_token_id: u32,
    pub forced_bos_token_id: u32,
    pub forced_eos_token_id: u32,
    pub num_beams: usize,
    pub no_repeat_ngram_size: usize,
}

impl Florence2TextConfig {
    /// Learned positional embeddings carry an offset of 2 (BART hack).
    pub const POS_OFFSET: usize = 2;
    pub fn enc_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }
    pub fn dec_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }
}

/// Full Florence-2 configuration.
#[derive(Debug, Clone)]
pub struct Florence2Config {
    pub vision: Florence2VisionConfig,
    pub text: Florence2TextConfig,
    pub vocab_size: usize,
    pub projection_dim: usize,
}

impl Florence2Config {
    /// `microsoft/Florence-2-large` preset.
    pub fn large() -> Self {
        Self {
            vision: Florence2VisionConfig {
                dim_embed: vec![256, 512, 1024, 2048],
                depths: vec![1, 1, 9, 1],
                num_heads: vec![8, 16, 32, 64],
                num_groups: vec![8, 16, 32, 64],
                patch_size: vec![7, 3, 3, 3],
                patch_stride: vec![4, 2, 2, 2],
                patch_padding: vec![3, 1, 1, 1],
                patch_prenorm: vec![false, true, true, true],
                window_size: 12,
                projection_dim: 1024,
                mlp_ratio: 4.0,
                image_pos_embed_max: 50,
                visual_temporal_max: 100,
                image_feature_source: vec!["spatial_avg_pool".into(), "temporal_avg_pool".into()],
            },
            text: Florence2TextConfig {
                vocab_size: 51289,
                d_model: 1024,
                encoder_layers: 12,
                decoder_layers: 12,
                encoder_attention_heads: 16,
                decoder_attention_heads: 16,
                encoder_ffn_dim: 4096,
                decoder_ffn_dim: 4096,
                max_position_embeddings: 4096,
                scale_embedding: false,
                bos_token_id: 0,
                eos_token_id: 2,
                pad_token_id: 1,
                decoder_start_token_id: 2,
                forced_bos_token_id: 0,
                forced_eos_token_id: 2,
                num_beams: 3,
                no_repeat_ngram_size: 3,
            },
            vocab_size: 51289,
            projection_dim: 1024,
        }
    }

    /// Parse an HF `config.json`.
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read florence2 config {}", path.display()))?;
        let hf: HfConfig = serde_json::from_str(&raw)
            .with_context(|| format!("parse florence2 config {}", path.display()))?;
        Ok(hf.into_config())
    }

    /// Embedding scale (`sqrt(d_model)` when `scale_embedding`, else 1).
    pub fn embed_scale(&self) -> f32 {
        if self.text.scale_embedding {
            (self.text.d_model as f32).sqrt()
        } else {
            1.0
        }
    }
}

// ---- HF config.json deserialization ----

#[derive(Debug, Deserialize)]
struct HfConfig {
    vocab_size: usize,
    projection_dim: usize,
    text_config: HfTextConfig,
    vision_config: HfVisionConfig,
}

#[derive(Debug, Deserialize)]
struct HfTextConfig {
    vocab_size: usize,
    d_model: usize,
    encoder_layers: usize,
    decoder_layers: usize,
    encoder_attention_heads: usize,
    decoder_attention_heads: usize,
    encoder_ffn_dim: usize,
    decoder_ffn_dim: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    scale_embedding: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    decoder_start_token_id: u32,
    forced_bos_token_id: u32,
    forced_eos_token_id: u32,
    #[serde(default = "default_beams")]
    num_beams: usize,
    #[serde(default = "default_ngram")]
    no_repeat_ngram_size: usize,
}

fn default_beams() -> usize {
    3
}
fn default_ngram() -> usize {
    3
}

#[derive(Debug, Deserialize)]
struct HfVisionConfig {
    dim_embed: Vec<usize>,
    depths: Vec<usize>,
    num_heads: Vec<usize>,
    num_groups: Vec<usize>,
    patch_size: Vec<usize>,
    patch_stride: Vec<usize>,
    patch_padding: Vec<usize>,
    patch_prenorm: Vec<bool>,
    window_size: usize,
    projection_dim: usize,
    image_pos_embed: HfPosEmbed,
    visual_temporal_embedding: HfTemporalEmbed,
    image_feature_source: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HfPosEmbed {
    max_pos_embeddings: usize,
}

#[derive(Debug, Deserialize)]
struct HfTemporalEmbed {
    max_temporal_embeddings: usize,
}

impl HfConfig {
    fn into_config(self) -> Florence2Config {
        let v = self.vision_config;
        let t = self.text_config;
        Florence2Config {
            vision: Florence2VisionConfig {
                dim_embed: v.dim_embed,
                depths: v.depths,
                num_heads: v.num_heads,
                num_groups: v.num_groups,
                patch_size: v.patch_size,
                patch_stride: v.patch_stride,
                patch_padding: v.patch_padding,
                patch_prenorm: v.patch_prenorm,
                window_size: v.window_size,
                projection_dim: v.projection_dim,
                mlp_ratio: 4.0,
                image_pos_embed_max: v.image_pos_embed.max_pos_embeddings,
                visual_temporal_max: v.visual_temporal_embedding.max_temporal_embeddings,
                image_feature_source: v.image_feature_source,
            },
            text: Florence2TextConfig {
                vocab_size: t.vocab_size,
                d_model: t.d_model,
                encoder_layers: t.encoder_layers,
                decoder_layers: t.decoder_layers,
                encoder_attention_heads: t.encoder_attention_heads,
                decoder_attention_heads: t.decoder_attention_heads,
                encoder_ffn_dim: t.encoder_ffn_dim,
                decoder_ffn_dim: t.decoder_ffn_dim,
                max_position_embeddings: t.max_position_embeddings,
                scale_embedding: t.scale_embedding,
                bos_token_id: t.bos_token_id,
                eos_token_id: t.eos_token_id,
                pad_token_id: t.pad_token_id,
                decoder_start_token_id: t.decoder_start_token_id,
                forced_bos_token_id: t.forced_bos_token_id,
                forced_eos_token_id: t.forced_eos_token_id,
                num_beams: t.num_beams,
                no_repeat_ngram_size: t.no_repeat_ngram_size,
            },
            vocab_size: self.vocab_size,
            projection_dim: self.projection_dim,
        }
    }
}

/// Returns the post-processing type for a task token (e.g. `<CAPTION>` →
/// `pure_text`, `<OD>` → `description_with_bboxes`).
pub fn task_post_processing_type(task: &str) -> &'static str {
    match task {
        "<OCR>" => "pure_text",
        "<OCR_WITH_REGION>" => "ocr",
        "<CAPTION>" => "pure_text",
        "<DETAILED_CAPTION>" => "pure_text",
        "<MORE_DETAILED_CAPTION>" => "pure_text",
        "<OD>" => "description_with_bboxes",
        "<DENSE_REGION_CAPTION>" => "description_with_bboxes",
        "<CAPTION_TO_PHRASE_GROUNDING>" => "phrase_grounding",
        "<REFERRING_EXPRESSION_SEGMENTATION>" => "polygons",
        "<REGION_TO_SEGMENTATION>" => "polygons",
        "<OPEN_VOCABULARY_DETECTION>" => "description_with_bboxes_or_polygons",
        "<REGION_TO_CATEGORY>" => "pure_text",
        "<REGION_TO_DESCRIPTION>" => "pure_text",
        "<REGION_TO_OCR>" => "pure_text",
        "<REGION_PROPOSAL>" => "bboxes",
        _ => "pure_text",
    }
}

/// Expands a task token (optionally with an `input` suffix) into the prompt
/// string fed to the BART tokenizer. Mirrors `_construct_prompts`.
pub fn construct_prompt(text: &str) -> String {
    // 1. Fixed task prompts without additional inputs.
    for (token, prompt) in TASK_PROMPTS_WITHOUT_INPUTS {
        if text.contains(token) {
            // HF asserts the token is the only content.
            return prompt.to_string();
        }
    }
    // 2. Task prompts with additional inputs.
    for (token, prompt) in TASK_PROMPTS_WITH_INPUT {
        if text.contains(token) {
            let input = text.replace(token, "");
            return prompt.replace("{input}", &input);
        }
    }
    text.to_string()
}

const TASK_PROMPTS_WITHOUT_INPUTS: &[(&str, &str)] = &[
    ("<OCR>", "What is the text in the image?"),
    (
        "<OCR_WITH_REGION>",
        "What is the text in the image, with regions?",
    ),
    ("<CAPTION>", "What does the image describe?"),
    (
        "<DETAILED_CAPTION>",
        "Describe in detail what is shown in the image.",
    ),
    (
        "<MORE_DETAILED_CAPTION>",
        "Describe with a paragraph what is shown in the image.",
    ),
    (
        "<OD>",
        "Locate the objects with category name in the image.",
    ),
    (
        "<DENSE_REGION_CAPTION>",
        "Locate the objects in the image, with their descriptions.",
    ),
    (
        "<REGION_PROPOSAL>",
        "Locate the region proposals in the image.",
    ),
];

const TASK_PROMPTS_WITH_INPUT: &[(&str, &str)] = &[
    (
        "<CAPTION_TO_PHRASE_GROUNDING>",
        "Locate the phrases in the caption: {input}",
    ),
    (
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        "Locate {input} in the image with mask",
    ),
    (
        "<REGION_TO_SEGMENTATION>",
        "What is the polygon mask of region {input}",
    ),
    (
        "<OPEN_VOCABULARY_DETECTION>",
        "Locate {input} in the image.",
    ),
    ("<REGION_TO_CATEGORY>", "What is the region {input}?"),
    (
        "<REGION_TO_DESCRIPTION>",
        "What does the region {input} describe?",
    ),
    ("<REGION_TO_OCR>", "What text is in the region {input}?"),
];

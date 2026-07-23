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

//! `baidu/Unlimited-OCR` configuration — HuggingFace `config.json`.
//!
//! Architecture: SAM-ViT-B + CLIP-L/14-224 "deep encoder" vision tower, a
//! linear `2048 → 1280` projector, and a Mixture-of-Experts DeepSeek-V2-style
//! decoder (dense early layers, routed + shared experts afterwards, rolling
//! sliding-window attention).
//!
//! The real checkpoint duplicates most decoder fields at both the top level
//! and under a nested `language_config` object (it subclasses
//! `DeepseekV2Config`, whose fields get flattened into the parent at save
//! time). Vision settings live under `vision_config.width.{sam_vit_b,
//! clip-l-14-224}` and the projector under `projector_config`. We parse all
//! of that permissively — top level wins, `language_config` is the
//! fallback, then hard-coded defaults matching the published checkpoint.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::Path;

/// Vision placeholder token id spliced into the LM input ids.
pub const IMAGE_TOKEN_ID: u32 = 128_815;
pub const BOS_TOKEN_ID: u32 = 0;
pub const EOS_TOKEN_ID: u32 = 1;
pub const PAD_TOKEN_ID: u32 = 2;

/// Image-processor patch size (`processor_config.json: patch_size`).
pub const PATCH_SIZE: usize = 16;
/// Image-processor token downsample ratio (`processor_config.json: downsample_ratio`).
pub const DOWNSAMPLE_RATIO: usize = 4;

/// Number of vision-query tokens a single square view of `image_size` pixels
/// contributes, matching HF:
/// `math.ceil((image_size // patch_size) / downsample_ratio)`.
///
/// Note the floor division happens *before* the ceiling — for the checkpoint's
/// own view sizes (1024, 640, both multiples of 64) this coincides with the
/// simpler `ceil((image_size/patch_size)/downsample_ratio)` restatement.
pub fn num_queries(image_size: usize) -> usize {
    num_queries_with(image_size, PATCH_SIZE, DOWNSAMPLE_RATIO)
}

/// [`num_queries`] with explicit patch/downsample parameters.
pub fn num_queries_with(image_size: usize, patch_size: usize, downsample_ratio: usize) -> usize {
    let grid = image_size / patch_size;
    grid.div_ceil(downsample_ratio)
}

// ---------------------------------------------------------------------
// SAM-ViT-B tower.
// ---------------------------------------------------------------------

/// SAM-ViT-B tower (`vision_config.width.sam_vit_b` in HF JSON).
#[derive(Debug, Clone)]
pub struct SamTowerConfig {
    pub variant: String,
    /// Patch-embedding / transformer width (`width`).
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Input resolution the tower was trained at (pixels, square).
    pub image_size: usize,
    pub patch_size: usize,
    /// Window size for windowed-attention blocks (all blocks except
    /// `global_attn_indexes`, which use full/global attention).
    pub window_size: usize,
    /// 0-based block indices that use global (non-windowed) attention.
    pub global_attn_indexes: Vec<usize>,
    /// Neck output channels (`neck` Conv2d output width fed to `net_2`/`net_3`).
    pub out_chans: usize,
    /// `net_2`/`net_3` downsample channel widths (256 → 512 → 1024).
    pub downsample_channels: Vec<usize>,
}

impl Default for SamTowerConfig {
    fn default() -> Self {
        Self {
            variant: "sam_vit_b".into(),
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            image_size: 1024,
            patch_size: 16,
            window_size: 14,
            global_attn_indexes: vec![2, 5, 8, 11],
            out_chans: 256,
            downsample_channels: vec![512, 1024],
        }
    }
}

impl SamTowerConfig {
    fn from_raw(raw: Option<RawSamWidth>, top_image_size: Option<usize>) -> Self {
        let defaults = Self::default();
        let raw = raw.unwrap_or_default();
        Self {
            variant: "sam_vit_b".into(),
            hidden_size: raw.width.unwrap_or(defaults.hidden_size),
            num_hidden_layers: raw.layers.unwrap_or(defaults.num_hidden_layers),
            num_attention_heads: raw.heads.unwrap_or(defaults.num_attention_heads),
            image_size: raw
                .image_size
                .or(top_image_size)
                .unwrap_or(defaults.image_size),
            patch_size: raw.patch_size.unwrap_or(defaults.patch_size),
            window_size: defaults.window_size,
            global_attn_indexes: raw
                .global_attn_indexes
                .unwrap_or(defaults.global_attn_indexes),
            out_chans: defaults.out_chans,
            downsample_channels: raw
                .downsample_channels
                .unwrap_or(defaults.downsample_channels),
        }
    }
}

// ---------------------------------------------------------------------
// CLIP-L/14-224 tower.
// ---------------------------------------------------------------------

/// CLIP-L/14-224 tower (`vision_config.width."clip-l-14-224"` in HF JSON).
#[derive(Debug, Clone)]
pub struct ClipTowerConfig {
    pub variant: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    /// MLP hidden width (`fc1`/`fc2`); the published tower hard-codes 4096
    /// (`ffn_hidden_size` in `deepencoder.py`'s `vit_model_cfg`), independent
    /// of the top-level `vision_config.mlp_ratio` (which is SAM's neck ratio).
    pub intermediate_size: usize,
}

impl Default for ClipTowerConfig {
    fn default() -> Self {
        Self {
            variant: "clip-l-14-224".into(),
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            image_size: 224,
            patch_size: 14,
            intermediate_size: 4096,
        }
    }
}

impl ClipTowerConfig {
    fn from_raw(raw: Option<RawClipWidth>) -> Self {
        let defaults = Self::default();
        let raw = raw.unwrap_or_default();
        Self {
            variant: "clip-l-14-224".into(),
            hidden_size: raw.width.unwrap_or(defaults.hidden_size),
            num_hidden_layers: raw.layers.unwrap_or(defaults.num_hidden_layers),
            num_attention_heads: raw.heads.unwrap_or(defaults.num_attention_heads),
            image_size: raw.image_size.unwrap_or(defaults.image_size),
            patch_size: raw.patch_size.unwrap_or(defaults.patch_size),
            intermediate_size: defaults.intermediate_size,
        }
    }
}

/// Combined vision tower config (`vision_config` in HF JSON).
#[derive(Debug, Clone)]
pub struct UnlimitedOcrVisionConfig {
    pub sam: SamTowerConfig,
    pub clip: ClipTowerConfig,
    /// Top-level `vision_config.image_size` (== the "global view" / base size, 1024).
    pub image_size: usize,
}

impl Default for UnlimitedOcrVisionConfig {
    fn default() -> Self {
        Self {
            sam: SamTowerConfig::default(),
            clip: ClipTowerConfig::default(),
            image_size: 1024,
        }
    }
}

impl UnlimitedOcrVisionConfig {
    fn from_raw(raw: Option<RawVisionConfig>) -> Self {
        let defaults = Self::default();
        let raw = raw.unwrap_or_default();
        let image_size = raw.image_size.unwrap_or(defaults.image_size);
        let width = raw.width.unwrap_or_default();
        Self {
            sam: SamTowerConfig::from_raw(width.sam_vit_b, Some(image_size)),
            clip: ClipTowerConfig::from_raw(width.clip),
            image_size,
        }
    }
}

// ---------------------------------------------------------------------
// Projector.
// ---------------------------------------------------------------------

/// Linear projector `2048 → 1280` bridging concatenated SAM+CLIP features
/// into the LM (`projector_config` in HF JSON).
#[derive(Debug, Clone)]
pub struct ProjectorConfig {
    pub input_dim: usize,
    pub n_embed: usize,
    pub projector_type: String,
}

impl Default for ProjectorConfig {
    fn default() -> Self {
        Self {
            input_dim: 2048,
            n_embed: 1280,
            projector_type: "linear".into(),
        }
    }
}

impl ProjectorConfig {
    fn from_raw(raw: Option<RawProjectorConfig>) -> Self {
        let defaults = Self::default();
        let raw = raw.unwrap_or_default();
        Self {
            input_dim: raw.input_dim.unwrap_or(defaults.input_dim),
            n_embed: raw.n_embed.unwrap_or(defaults.n_embed),
            projector_type: raw.projector_type.unwrap_or(defaults.projector_type),
        }
    }
}

// ---------------------------------------------------------------------
// Raw (flexible) JSON shapes — every field optional, top level ∪ nested
// `language_config` ∪ hard-coded default resolved in [`UnlimitedOcrConfig::resolve`].
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
struct RawTextFields {
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    n_routed_experts: Option<usize>,
    n_shared_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    moe_intermediate_size: Option<usize>,
    intermediate_size: Option<usize>,
    first_k_dense_replace: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    sliding_window: Option<usize>,
    sliding_window_size: Option<usize>,
    use_mla: Option<bool>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    hidden_act: Option<String>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    v_head_dim: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawSamWidth {
    downsample_channels: Option<Vec<usize>>,
    global_attn_indexes: Option<Vec<usize>>,
    heads: Option<usize>,
    layers: Option<usize>,
    width: Option<usize>,
    patch_size: Option<usize>,
    image_size: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawClipWidth {
    heads: Option<usize>,
    image_size: Option<usize>,
    layers: Option<usize>,
    patch_size: Option<usize>,
    width: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawVisionWidth {
    #[serde(rename = "clip-l-14-224", default)]
    clip: Option<RawClipWidth>,
    #[serde(default)]
    sam_vit_b: Option<RawSamWidth>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawVisionConfig {
    image_size: Option<usize>,
    /// Present in the checkpoint (SAM neck ratio); not consumed — both
    /// towers' known code paths hard-code their own MLP ratios instead
    /// (see [`SamTowerConfig`]/[`ClipTowerConfig`] defaults).
    #[allow(dead_code)]
    mlp_ratio: Option<f64>,
    width: Option<RawVisionWidth>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawProjectorConfig {
    input_dim: Option<usize>,
    n_embed: Option<usize>,
    projector_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawUnlimitedOcrConfig {
    model_type: Option<String>,
    image_token_id: Option<u32>,
    #[serde(flatten)]
    text: RawTextFields,
    language_config: Option<RawTextFields>,
    vision_config: Option<RawVisionConfig>,
    projector_config: Option<RawProjectorConfig>,
}

// ---------------------------------------------------------------------
// Resolved top-level config.
// ---------------------------------------------------------------------

/// Top-level `baidu/Unlimited-OCR` checkpoint config (`config.json`), fully
/// resolved (top level ∪ `language_config` ∪ published-checkpoint defaults).
#[derive(Debug, Clone)]
pub struct UnlimitedOcrConfig {
    pub model_type: String,

    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    /// Number of routed (top-k gated) experts.
    pub n_routed_experts: usize,
    /// Number of always-on shared experts.
    pub n_shared_experts: usize,
    /// Experts activated per token (top-k routing), a.k.a. `num_experts_per_tok`.
    pub num_experts_per_tok: usize,
    /// FFN inner width for each routed/shared expert.
    pub moe_intermediate_size: usize,
    /// FFN inner width for the dense (non-MoE) leading layers.
    pub intermediate_size: usize,
    /// Leading dense (non-MoE) layers before MoE layers begin.
    pub first_k_dense_replace: usize,

    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    /// Rolling sliding-window attention span (a.k.a. `sliding_window_size` /
    /// `_ring_window` at inference — the HF `infer()` path zeroes
    /// `config.sliding_window` and instead has the ring-buffer KV cache read
    /// this value directly, so prefill isn't truncated).
    pub sliding_window: usize,
    /// Multi-head latent attention (DeepSeek-style KV compression); off for this model.
    pub use_mla: bool,

    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,

    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub image_token_id: u32,

    /// `v_head_dim` from the checkpoint, when present (should equal [`Self::head_dim`]).
    pub v_head_dim: Option<usize>,

    pub vision_config: UnlimitedOcrVisionConfig,
    pub projector: ProjectorConfig,

    /// Image-processor patch size (`processor_config.json`).
    pub patch_size: usize,
    /// Image-processor token downsample ratio (`processor_config.json`).
    pub downsample_ratio: usize,
}

fn pick<T>(top: Option<T>, lang: Option<T>, default: T) -> T {
    top.or(lang).unwrap_or(default)
}

impl UnlimitedOcrConfig {
    pub const HF_MODEL_ID: &'static str = "baidu/Unlimited-OCR";

    pub fn from_json_str(data: &str) -> Result<Self> {
        let raw: RawUnlimitedOcrConfig =
            serde_json::from_str(data).context("parse Unlimited-OCR config.json")?;
        Ok(Self::resolve(raw))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read Unlimited-OCR config {path:?}"))?;
        Self::from_json_str(&data).with_context(|| format!("parse Unlimited-OCR config {path:?}"))
    }

    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        Self::from_file(&dir.join("config.json"))
    }

    fn resolve(raw: RawUnlimitedOcrConfig) -> Self {
        let lang = raw.language_config.unwrap_or_default();
        let text = raw.text;

        let sliding_window = pick(
            text.sliding_window.or(text.sliding_window_size),
            lang.sliding_window.or(lang.sliding_window_size),
            128,
        );

        Self {
            model_type: raw.model_type.unwrap_or_else(|| "unlimited-ocr".into()),
            hidden_size: pick(text.hidden_size, lang.hidden_size, 1280),
            num_hidden_layers: pick(text.num_hidden_layers, lang.num_hidden_layers, 12),
            num_attention_heads: pick(text.num_attention_heads, lang.num_attention_heads, 10),
            num_key_value_heads: pick(text.num_key_value_heads, lang.num_key_value_heads, 10),
            n_routed_experts: pick(text.n_routed_experts, lang.n_routed_experts, 64),
            n_shared_experts: pick(text.n_shared_experts, lang.n_shared_experts, 2),
            num_experts_per_tok: pick(text.num_experts_per_tok, lang.num_experts_per_tok, 6),
            moe_intermediate_size: pick(
                text.moe_intermediate_size,
                lang.moe_intermediate_size,
                896,
            ),
            intermediate_size: pick(text.intermediate_size, lang.intermediate_size, 6848),
            first_k_dense_replace: pick(text.first_k_dense_replace, lang.first_k_dense_replace, 1),
            vocab_size: pick(text.vocab_size, lang.vocab_size, 129_280),
            max_position_embeddings: pick(
                text.max_position_embeddings,
                lang.max_position_embeddings,
                32_768,
            ),
            sliding_window,
            use_mla: pick(text.use_mla, lang.use_mla, false),
            rms_norm_eps: pick(text.rms_norm_eps, lang.rms_norm_eps, 1e-6),
            rope_theta: pick(text.rope_theta, lang.rope_theta, 10_000.0),
            hidden_act: pick(text.hidden_act, lang.hidden_act, "silu".into()),
            bos_token_id: pick(text.bos_token_id, lang.bos_token_id, BOS_TOKEN_ID),
            eos_token_id: pick(text.eos_token_id, lang.eos_token_id, EOS_TOKEN_ID),
            pad_token_id: pick(text.pad_token_id, lang.pad_token_id, PAD_TOKEN_ID),
            image_token_id: raw.image_token_id.unwrap_or(IMAGE_TOKEN_ID),
            v_head_dim: text.v_head_dim.or(lang.v_head_dim),
            vision_config: UnlimitedOcrVisionConfig::from_raw(raw.vision_config),
            projector: ProjectorConfig::from_raw(raw.projector_config),
            patch_size: PATCH_SIZE,
            downsample_ratio: DOWNSAMPLE_RATIO,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "unlimited-ocr",
            "model_type must be unlimited-ocr, got {}",
            self.model_type
        );
        ensure!(self.num_hidden_layers > 0, "num_hidden_layers");
        ensure!(
            self.first_k_dense_replace <= self.num_hidden_layers,
            "first_k_dense_replace must be <= num_hidden_layers"
        );
        ensure!(
            self.num_experts_per_tok <= self.n_routed_experts,
            "num_experts_per_tok must be <= n_routed_experts"
        );
        ensure!(!self.use_mla, "MLA is not supported by this model");
        ensure!(
            self.projector.n_embed == self.hidden_size,
            "projector n_embed must match hidden_size"
        );
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Whether `layer_idx` (0-based) uses the dense (non-MoE) FFN.
    pub fn is_dense_layer(&self, layer_idx: usize) -> bool {
        layer_idx < self.first_k_dense_replace
    }

    /// [`num_queries`] for a view of `image_size` pixels, using this
    /// checkpoint's `patch_size`/`downsample_ratio`.
    pub fn num_queries(&self, image_size: usize) -> usize {
        num_queries_with(image_size, self.patch_size, self.downsample_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim `baidu/Unlimited-OCR` `config.json` (trimmed of comments only).
    fn real_checkpoint_json() -> &'static str {
        r#"{
            "_name_or_path": "Unlimited-OCR",
            "candidate_resolutions": [[1024, 1024]],
            "global_view_pos": "head",
            "architectures": ["UnlimitedOCRForCausalLM"],
            "auto_map": {
                "AutoConfig": "modeling_unlimitedocr.UnlimitedOCRConfig",
                "AutoModel": "modeling_unlimitedocr.UnlimitedOCRForCausalLM"
            },
            "language_config": {
                "architectures": ["DeepseekOCRForCausalLM"],
                "bos_token_id": 0,
                "eos_token_id": 1,
                "first_k_dense_replace": 1,
                "hidden_size": 1280,
                "intermediate_size": 6848,
                "kv_lora_rank": null,
                "lm_head": true,
                "max_position_embeddings": 32768,
                "moe_intermediate_size": 896,
                "n_group": 1,
                "n_routed_experts": 64,
                "n_shared_experts": 2,
                "num_attention_heads": 10,
                "num_experts_per_tok": 6,
                "num_hidden_layers": 12,
                "num_key_value_heads": 10,
                "q_lora_rank": null,
                "qk_nope_head_dim": 0,
                "qk_rope_head_dim": 0,
                "rm_head": false,
                "topk_group": 1,
                "topk_method": "greedy",
                "torch_dtype": "bfloat16",
                "use_mla": false,
                "v_head_dim": 128,
                "vocab_size": 129280,
                "sliding_window_size": 128
            },
            "model_type": "unlimited-ocr",
            "projector_config": {
                "input_dim": 2048,
                "model_type": "mlp_projector",
                "n_embed": 1280,
                "projector_type": "linear"
            },
            "tile_tag": "2D",
            "torch_dtype": "bfloat16",
            "transformers_version": "4.46.3",
            "vision_config": {
                "image_size": 1024,
                "mlp_ratio": 3.7362,
                "model_name": "deeplip_b_l",
                "model_type": "vision",
                "width": {
                    "clip-l-14-224": {
                        "heads": 16,
                        "image_size": 224,
                        "layers": 24,
                        "patch_size": 14,
                        "width": 1024
                    },
                    "sam_vit_b": {
                        "downsample_channels": [512, 1024],
                        "global_attn_indexes": [2, 5, 8, 11],
                        "heads": 12,
                        "layers": 12,
                        "width": 768
                    }
                }
            },
            "bos_token_id": 0,
            "eos_token_id": 1,
            "first_k_dense_replace": 1,
            "hidden_size": 1280,
            "intermediate_size": 6848,
            "kv_lora_rank": null,
            "lm_head": true,
            "max_position_embeddings": 32768,
            "moe_intermediate_size": 896,
            "n_group": 1,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_attention_heads": 10,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 12,
            "num_key_value_heads": 10,
            "q_lora_rank": null,
            "qk_nope_head_dim": 0,
            "qk_rope_head_dim": 0,
            "rm_head": false,
            "topk_group": 1,
            "topk_method": "greedy",
            "use_mla": false,
            "v_head_dim": 128,
            "vocab_size": 129280,
            "sliding_window_size": 128,
            "sliding_window": 128
        }"#
    }

    #[test]
    fn parses_real_checkpoint_config_and_validates() {
        let cfg = UnlimitedOcrConfig::from_json_str(real_checkpoint_json()).expect("parse");
        cfg.validate().expect("validate");

        assert_eq!(cfg.hidden_size, 1280);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 10);
        assert_eq!(cfg.num_key_value_heads, 10);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.kv_group_size(), 1);
        assert_eq!(cfg.v_head_dim, Some(128));

        assert_eq!(cfg.n_routed_experts, 64);
        assert_eq!(cfg.n_shared_experts, 2);
        assert_eq!(cfg.num_experts_per_tok, 6);
        assert_eq!(cfg.moe_intermediate_size, 896);
        assert_eq!(cfg.intermediate_size, 6848);
        assert_eq!(cfg.first_k_dense_replace, 1);
        assert!(cfg.is_dense_layer(0));
        assert!(!cfg.is_dense_layer(1));

        assert_eq!(cfg.vocab_size, 129_280);
        assert_eq!(cfg.max_position_embeddings, 32_768);
        assert_eq!(cfg.sliding_window, 128);
        assert!(!cfg.use_mla);

        assert_eq!(cfg.bos_token_id, BOS_TOKEN_ID);
        assert_eq!(cfg.eos_token_id, EOS_TOKEN_ID);
        assert_eq!(cfg.pad_token_id, PAD_TOKEN_ID);
        assert_eq!(cfg.image_token_id, IMAGE_TOKEN_ID);

        assert_eq!(cfg.projector.input_dim, 2048);
        assert_eq!(cfg.projector.n_embed, 1280);
        assert_eq!(cfg.projector.projector_type, "linear");

        assert_eq!(cfg.vision_config.image_size, 1024);
        assert_eq!(cfg.vision_config.sam.variant, "sam_vit_b");
        assert_eq!(cfg.vision_config.sam.hidden_size, 768);
        assert_eq!(cfg.vision_config.sam.num_hidden_layers, 12);
        assert_eq!(cfg.vision_config.sam.num_attention_heads, 12);
        assert_eq!(cfg.vision_config.sam.patch_size, 16);
        assert_eq!(cfg.vision_config.sam.image_size, 1024);
        assert_eq!(cfg.vision_config.sam.window_size, 14);
        assert_eq!(cfg.vision_config.sam.global_attn_indexes, vec![2, 5, 8, 11]);
        assert_eq!(cfg.vision_config.sam.out_chans, 256);
        assert_eq!(cfg.vision_config.sam.downsample_channels, vec![512, 1024]);

        assert_eq!(cfg.vision_config.clip.variant, "clip-l-14-224");
        assert_eq!(cfg.vision_config.clip.hidden_size, 1024);
        assert_eq!(cfg.vision_config.clip.num_hidden_layers, 24);
        assert_eq!(cfg.vision_config.clip.num_attention_heads, 16);
        assert_eq!(cfg.vision_config.clip.image_size, 224);
        assert_eq!(cfg.vision_config.clip.patch_size, 14);
        assert_eq!(cfg.vision_config.clip.intermediate_size, 4096);
    }

    #[test]
    fn falls_back_to_hardcoded_defaults_when_absent() {
        let cfg = UnlimitedOcrConfig::from_json_str(r#"{"model_type": "unlimited-ocr"}"#)
            .expect("parse minimal");
        assert_eq!(cfg.hidden_size, 1280);
        assert_eq!(cfg.n_routed_experts, 64);
        assert_eq!(cfg.image_token_id, IMAGE_TOKEN_ID);
        assert_eq!(cfg.vision_config.sam.hidden_size, 768);
        assert_eq!(cfg.projector.n_embed, 1280);
    }

    #[test]
    fn falls_back_to_nested_language_config_when_top_level_absent() {
        let cfg = UnlimitedOcrConfig::from_json_str(
            r#"{
                "model_type": "unlimited-ocr",
                "language_config": {"hidden_size": 2048, "n_routed_experts": 32}
            }"#,
        )
        .expect("parse");
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.n_routed_experts, 32);
    }

    #[test]
    fn num_queries_matches_hf_formula() {
        assert_eq!(num_queries(1024), 16);
        assert_eq!(num_queries(640), 10);
    }
}

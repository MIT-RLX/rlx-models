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

//! HF `config.json` for [`thinkingmachines/Inkling`](https://huggingface.co/thinkingmachines/Inkling).

use anyhow::{Context, Result, anyhow, bail};
use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

pub const FAMILY: &str = "Inkling";
pub const HF_MODEL_ID: &str = "thinkingmachines/Inkling";
/// Announced but **not yet released**
/// ([Inkling-Small](https://thinkingmachines.ai/news/introducing-inkling/#inkling-small)).
/// Hub id TBD when TML publishes weights — do not assume this string exists today.
pub const HF_MODEL_ID_SMALL: &str = "thinkingmachines/Inkling-Small";
/// Unsloth GGUF repo (quantized full model; recommended for local RLX work).
pub const HF_GGUF_REPO: &str = "unsloth/inkling-GGUF";
/// Placeholder for a future Unsloth / community Small GGUF — unset until released.
pub const HF_GGUF_REPO_SMALL: &str = "unsloth/inkling-small-GGUF";
pub const MODEL_TYPE: &str = "inkling_mm_model";
pub const ARCHITECTURE: &str = "InklingForConditionalGeneration";
pub const GGUF_ARCH: &str = "inkling";

/// Which member of the Inkling family a config / checkpoint belongs to.
///
/// Small shares the same arch recipes (relative attn, shortconv, sink MoE) but
/// different width/depth/expert counts — fill in from `config.json` once TML ships it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InklingVariant {
    /// Released full model: ~975B total / ~41B active.
    Full,
    /// Preview / unreleased: ~276B total / ~12B active (public totals only).
    Small,
}

impl InklingVariant {
    pub fn name(self) -> &'static str {
        match self {
            Self::Full => "Inkling",
            Self::Small => "Inkling-Small",
        }
    }

    /// Rough parameter counts from TML's announcement (not a substitute for config dims).
    pub fn announced_params(self) -> AnnouncedParams {
        match self {
            Self::Full => AnnouncedParams {
                total_billions: 975.0,
                active_billions: 41.0,
                weights_public: true,
            },
            Self::Small => AnnouncedParams {
                total_billions: 276.0,
                active_billions: 12.0,
                weights_public: false,
            },
        }
    }

    pub fn hf_model_id(self) -> &'static str {
        match self {
            Self::Full => HF_MODEL_ID,
            Self::Small => HF_MODEL_ID_SMALL,
        }
    }

    pub fn hf_gguf_repo(self) -> &'static str {
        match self {
            Self::Full => HF_GGUF_REPO,
            Self::Small => HF_GGUF_REPO_SMALL,
        }
    }

    /// Heuristic once Small weights exist: match known full dims, else treat as Small
    /// when `config.json` parses but does not match the full preset.
    pub fn detect_from_text(text: &InklingTextConfig) -> Self {
        let full = InklingConfig::production_preset().text;
        if text.num_hidden_layers == full.num_hidden_layers
            && text.hidden_size == full.hidden_size
            && text.n_routed_experts == full.n_routed_experts
        {
            Self::Full
        } else {
            Self::Small
        }
    }
}

/// Public announcement figures only — layer/hidden/expert dims for Small are TBD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnouncedParams {
    pub total_billions: f32,
    pub active_billions: f32,
    pub weights_public: bool,
}

/// Attention pattern for one decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnLayerType {
    /// Sliding-window hybrid local attention (`hybrid_sliding`).
    Sliding,
    /// Full causal hybrid attention (`hybrid`).
    Global,
}

/// FFN pattern for one decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpLayerType {
    Dense,
    Sparse,
}

#[derive(Debug, Clone)]
pub struct InklingTextConfig {
    pub vocab_size: usize,
    pub unpadded_vocab_size: Option<usize>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub swa_num_attention_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub swa_head_dim: usize,
    pub sliding_window_size: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: Option<usize>,
    pub log_scaling_alpha: f32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub conv_kernel_size: usize,
    pub use_embed_norm: bool,
    /// Dense SwiGLU intermediate dim (HF `dense_intermediate_size`).
    pub dense_intermediate_size: usize,
    /// Per-expert / MoE intermediate dim (HF `intermediate_size` when dense is separate).
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub shared_expert_sink: bool,
    pub route_scale: f32,
    pub logits_mup_width_multiplier: f32,
    /// Layers with index `< dense_mlp_idx` use dense MLP; the rest are MoE.
    pub dense_mlp_idx: usize,
    pub local_layer_ids: Vec<usize>,
    pub layer_types: Vec<AttnLayerType>,
    pub mlp_layer_types: Vec<MlpLayerType>,
    pub num_mtp_layers: usize,
    pub mtp_local_layer_ids: Vec<usize>,
    pub eos_token_id: u32,
}

#[derive(Debug, Clone)]
pub struct InklingAudioConfig {
    pub n_mel_bins: usize,
    pub mel_vocab_size: usize,
    pub text_hidden_size: usize,
    pub dmel_min_value: f32,
    pub dmel_max_value: f32,
    pub audio_mode: String,
}

#[derive(Debug, Clone)]
pub struct InklingVisionConfig {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub n_channels: usize,
    pub n_layers: usize,
    pub text_hidden_size: usize,
    pub vision_encoder_type: String,
}

#[derive(Debug, Clone)]
pub struct InklingConfig {
    pub model_type: String,
    pub text: InklingTextConfig,
    pub audio: InklingAudioConfig,
    pub vision: InklingVisionConfig,
    pub image_token_id: u32,
    pub audio_token_id: u32,
    pub image_bos_token_id: u32,
    pub audio_bos_token_id: u32,
}

impl InklingTextConfig {
    pub fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .copied()
            .unwrap_or(AttnLayerType::Global)
            == AttnLayerType::Sliding
    }

    pub fn is_dense_mlp(&self, layer: usize) -> bool {
        self.mlp_layer_types
            .get(layer)
            .copied()
            .unwrap_or(MlpLayerType::Sparse)
            == MlpLayerType::Dense
    }

    pub fn attn_heads(&self, layer: usize) -> (usize, usize, usize) {
        if self.is_sliding(layer) {
            (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            )
        } else {
            (
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            )
        }
    }

    pub fn rel_extent_for_layer(&self, layer: usize) -> usize {
        if self.is_sliding(layer) {
            self.sliding_window_size
        } else {
            self.rel_extent
        }
    }

    /// Parse Unsloth / llama.cpp GGUF metadata (`general.architecture = inkling`).
    ///
    /// Uses [`GgufFile::header_from_path`] so the multi‑GB weight payload is
    /// never slurped — the UD-IQ1_S `00001` shard is metadata-only and is the
    /// preferred sniff target.
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow!("missing general.architecture"))?;
        if arch != GGUF_ARCH && arch != MODEL_TYPE {
            bail!("expected general.architecture=inkling, got {arch}");
        }
        let lookup = |k: &str| -> Option<&MetaValue> { raw.metadata.get(&format!("inkling.{k}")) };
        let u32k = |k: &str| -> Result<u32> {
            lookup(k)
                .and_then(MetaValue::as_u32)
                .ok_or_else(|| anyhow!("missing inkling.{k}"))
        };
        let u32k_opt = |k: &str| -> Option<u32> { lookup(k).and_then(MetaValue::as_u32) };
        let f32k = |k: &str| -> Option<f32> {
            lookup(k).and_then(|v| match v {
                MetaValue::F32(x) => Some(*x),
                _ => None,
            })
        };
        let bool_arr = |k: &str| -> Vec<bool> {
            lookup(k)
                .and_then(|v| match v {
                    MetaValue::Array(a) => Some(
                        a.iter()
                            .filter_map(|x| match x {
                                MetaValue::Bool(b) => Some(*b),
                                _ => None,
                            })
                            .collect(),
                    ),
                    _ => None,
                })
                .unwrap_or_default()
        };
        let u32_arr = |k: &str| -> Vec<usize> {
            lookup(k)
                .and_then(|v| match v {
                    MetaValue::Array(a) => Some(
                        a.iter()
                            .filter_map(|x| match x {
                                MetaValue::U32(u) => Some(*u as usize),
                                MetaValue::U64(u) => Some(*u as usize),
                                MetaValue::I32(i) => Some(*i as usize),
                                _ => None,
                            })
                            .collect(),
                    ),
                    _ => None,
                })
                .unwrap_or_default()
        };

        let layers = u32k("block_count")? as usize;
        let dense_mlp_idx = u32k_opt("dense_block_count").unwrap_or(2) as usize;
        let head_dim = u32k_opt("attention.key_length").unwrap_or(128) as usize;
        let n_heads = u32k("attention.head_count")? as usize;
        let sw_pattern = bool_arr("attention.sliding_window_pattern");
        let kv_per_layer = u32_arr("attention.head_count_kv");
        let layer_types: Vec<AttnLayerType> = (0..layers)
            .map(|i| {
                let sliding = sw_pattern
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| (i + 1) % 6 != 0);
                if sliding {
                    AttnLayerType::Sliding
                } else {
                    AttnLayerType::Global
                }
            })
            .collect();
        // SWA kv heads vs global — take mode from pattern when available.
        let swa_kv = kv_per_layer
            .iter()
            .copied()
            .zip(layer_types.iter())
            .find(|(_, t)| **t == AttnLayerType::Sliding)
            .map(|(k, _)| k)
            .unwrap_or(16);
        let global_kv = kv_per_layer
            .iter()
            .copied()
            .zip(layer_types.iter())
            .find(|(_, t)| **t == AttnLayerType::Global)
            .map(|(k, _)| k)
            .unwrap_or(8);
        let mlp_layer_types: Vec<MlpLayerType> = (0..layers)
            .map(|i| {
                if i < dense_mlp_idx {
                    MlpLayerType::Dense
                } else {
                    MlpLayerType::Sparse
                }
            })
            .collect();
        let local: Vec<usize> = layer_types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (*t == AttnLayerType::Sliding).then_some(i))
            .collect();

        Ok(Self {
            vocab_size: u32k_opt("vocab_size").unwrap_or(201_024) as usize,
            unpadded_vocab_size: u32k_opt("unpadded_vocab_size").map(|u| u as usize),
            hidden_size: u32k("embedding_length")? as usize,
            num_hidden_layers: layers,
            num_attention_heads: n_heads,
            num_key_value_heads: global_kv,
            head_dim,
            swa_num_attention_heads: n_heads,
            swa_num_key_value_heads: swa_kv,
            swa_head_dim: head_dim,
            sliding_window_size: u32k_opt("attention.sliding_window").unwrap_or(512) as usize,
            d_rel: u32k_opt("d_rel").unwrap_or(16) as usize,
            rel_extent: u32k_opt("rel_extent").unwrap_or(1024) as usize,
            log_scaling_n_floor: u32k_opt("log_scaling_n_floor").map(|u| u as usize),
            log_scaling_alpha: f32k("log_scaling_alpha").unwrap_or(0.1),
            max_position_embeddings: u32k_opt("context_length").unwrap_or(1_048_576) as usize,
            rms_norm_eps: f32k("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            conv_kernel_size: u32k_opt("shortconv_kernel").unwrap_or(4) as usize,
            use_embed_norm: true,
            dense_intermediate_size: u32k("feed_forward_length")? as usize,
            moe_intermediate_size: u32k_opt("expert_feed_forward_length").unwrap_or(3072) as usize,
            n_routed_experts: u32k("expert_count")? as usize,
            num_experts_per_tok: u32k("expert_used_count")? as usize,
            n_shared_experts: u32k_opt("expert_shared_count").unwrap_or(2) as usize,
            shared_expert_sink: true,
            route_scale: f32k("expert_weights_scale").unwrap_or(8.0),
            logits_mup_width_multiplier: f32k("logit_scale_denom").unwrap_or(24.0),
            dense_mlp_idx,
            local_layer_ids: local,
            layer_types,
            mlp_layer_types,
            num_mtp_layers: 0,
            mtp_local_layer_ids: vec![],
            eos_token_id: raw
                .metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(MetaValue::as_u32)
                .unwrap_or(200_006),
        })
    }

    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = GgufFile::header_from_path(path)
            .with_context(|| format!("rlx-inkling: GGUF header {}", path.display()))?;
        Self::from_gguf(&raw).with_context(|| format!("rlx-inkling: parse {}", path.display()))
    }
}

impl InklingConfig {
    /// [`InklingVariant::Full`] when dims match the released checkpoint; otherwise
    /// [`InklingVariant::Small`] (placeholder until Small ships and we pin its preset).
    pub fn variant(&self) -> InklingVariant {
        InklingVariant::detect_from_text(&self.text)
    }

    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("rlx-inkling: read config {}", path.display()))?;
        Self::from_json_str(&raw).with_context(|| format!("rlx-inkling: parse {}", path.display()))
    }

    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_json_path(dir.as_ref().join("config.json"))
    }

    pub fn from_json_str(s: &str) -> Result<Self> {
        let root: HfRoot = serde_json::from_str(s).context("rlx-inkling: config.json")?;
        root.into_config()
    }

    /// Production [thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling) dims.
    pub fn production_preset() -> Self {
        const LOCAL: &[usize] = &[
            0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27,
            28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54,
            55, 56, 57, 58, 60, 61, 62, 63, 64,
        ];
        let layers = 66;
        let dense_mlp_idx = 2;
        let local: Vec<usize> = LOCAL.to_vec();
        let layer_types = (0..layers)
            .map(|i| {
                if local.contains(&i) {
                    AttnLayerType::Sliding
                } else {
                    AttnLayerType::Global
                }
            })
            .collect();
        let mlp_layer_types = (0..layers)
            .map(|i| {
                if i < dense_mlp_idx {
                    MlpLayerType::Dense
                } else {
                    MlpLayerType::Sparse
                }
            })
            .collect();
        Self {
            model_type: MODEL_TYPE.into(),
            text: InklingTextConfig {
                vocab_size: 201_024,
                unpadded_vocab_size: Some(200_058),
                hidden_size: 6144,
                num_hidden_layers: layers,
                num_attention_heads: 64,
                num_key_value_heads: 8,
                head_dim: 128,
                swa_num_attention_heads: 64,
                swa_num_key_value_heads: 16,
                swa_head_dim: 128,
                sliding_window_size: 512,
                d_rel: 16,
                rel_extent: 1024,
                log_scaling_n_floor: Some(128_000),
                log_scaling_alpha: 0.1,
                max_position_embeddings: 1_048_576,
                rms_norm_eps: 1e-6,
                conv_kernel_size: 4,
                use_embed_norm: true,
                dense_intermediate_size: 24_576,
                moe_intermediate_size: 3072,
                n_routed_experts: 256,
                num_experts_per_tok: 6,
                n_shared_experts: 2,
                shared_expert_sink: true,
                route_scale: 8.0,
                logits_mup_width_multiplier: 24.0,
                dense_mlp_idx,
                local_layer_ids: local,
                layer_types,
                mlp_layer_types,
                num_mtp_layers: 8,
                mtp_local_layer_ids: vec![0, 2, 4, 5, 6, 7],
                eos_token_id: 200_006,
            },
            audio: InklingAudioConfig {
                n_mel_bins: 80,
                mel_vocab_size: 16,
                text_hidden_size: 6144,
                dmel_min_value: -7.0,
                dmel_max_value: 2.0,
                audio_mode: "dmel".into(),
            },
            vision: InklingVisionConfig {
                patch_size: 40,
                temporal_patch_size: 2,
                n_channels: 3,
                n_layers: 4,
                text_hidden_size: 6144,
                vision_encoder_type: "hmlp".into(),
            },
            image_token_id: 200_054,
            audio_token_id: 200_053,
            image_bos_token_id: 200_005,
            audio_bos_token_id: 200_020,
        }
    }
}

#[derive(Debug, Deserialize)]
struct HfRoot {
    architectures: Option<Vec<String>>,
    model_type: Option<String>,
    eos_token_id: Option<u32>,
    text_config: HfText,
    audio_config: Option<HfAudio>,
    vision_config: Option<HfVision>,
    mtp_config: Option<HfMtp>,
    image_token_id: Option<u32>,
    audio_token_id: Option<u32>,
    image_bos_token_id: Option<u32>,
    audio_bos_token_id: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct HfText {
    vocab_size: Option<usize>,
    unpadded_vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    swa_num_attention_heads: Option<usize>,
    swa_num_key_value_heads: Option<usize>,
    swa_head_dim: Option<usize>,
    sliding_window_size: Option<usize>,
    d_rel: Option<usize>,
    rel_extent: Option<usize>,
    log_scaling_n_floor: Option<usize>,
    log_scaling_alpha: Option<f32>,
    model_max_length: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f32>,
    sconv_kernel_size: Option<usize>,
    conv_kernel_size: Option<usize>,
    use_embed_norm: Option<bool>,
    dense_intermediate_size: Option<usize>,
    intermediate_size: Option<usize>,
    moe_intermediate_size: Option<usize>,
    dense_mlp_idx: Option<usize>,
    n_routed_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    n_shared_experts: Option<usize>,
    shared_expert_sink: Option<bool>,
    route_scale: Option<f32>,
    logits_mup_width_multiplier: Option<f32>,
    local_layer_ids: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct HfAudio {
    n_mel_bins: Option<usize>,
    mel_vocab_size: Option<usize>,
    decoder_dmodel: Option<usize>,
    dmel_min_value: Option<f32>,
    dmel_max_value: Option<f32>,
    audio_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HfVision {
    patch_size: Option<usize>,
    temporal_patch_size: Option<usize>,
    n_channels: Option<usize>,
    n_layers: Option<usize>,
    decoder_dmodel: Option<usize>,
    vision_encoder_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HfMtp {
    num_nextn_predict_layers: Option<usize>,
    local_layer_ids: Option<Vec<usize>>,
}

impl HfRoot {
    fn into_config(self) -> Result<InklingConfig> {
        if let Some(mt) = &self.model_type {
            if mt != MODEL_TYPE {
                bail!("rlx-inkling: expected model_type={MODEL_TYPE}, got {mt}");
            }
        }
        if let Some(archs) = &self.architectures {
            if !archs.iter().any(|a| a == ARCHITECTURE) {
                bail!(
                    "rlx-inkling: expected architectures containing {ARCHITECTURE}, got {archs:?}"
                );
            }
        }

        let t = self.text_config;
        let layers = t.num_hidden_layers.unwrap_or(66);
        let dense_mlp_idx = t.dense_mlp_idx.unwrap_or(0);
        let local = t
            .local_layer_ids
            .unwrap_or_else(|| (0..layers).filter(|i| (i + 1) % 6 != 0).collect());
        let layer_types = (0..layers)
            .map(|i| {
                if local.contains(&i) {
                    AttnLayerType::Sliding
                } else {
                    AttnLayerType::Global
                }
            })
            .collect();
        let mlp_layer_types = (0..layers)
            .map(|i| {
                if i < dense_mlp_idx {
                    MlpLayerType::Dense
                } else {
                    MlpLayerType::Sparse
                }
            })
            .collect();

        // HF stores MoE width in `intermediate_size` and dense width in
        // `dense_intermediate_size`. Transformers remaps the latter onto
        // `intermediate_size` for the dense MLP.
        let moe_intermediate_size = t
            .moe_intermediate_size
            .or(t.intermediate_size)
            .unwrap_or(3072);
        let dense_intermediate_size = t.dense_intermediate_size.unwrap_or(moe_intermediate_size);

        let hidden = t.hidden_size.unwrap_or(6144);
        let mtp = self.mtp_config.unwrap_or(HfMtp {
            num_nextn_predict_layers: None,
            local_layer_ids: None,
        });
        let audio = self.audio_config.unwrap_or(HfAudio {
            n_mel_bins: None,
            mel_vocab_size: None,
            decoder_dmodel: None,
            dmel_min_value: None,
            dmel_max_value: None,
            audio_mode: None,
        });
        let vision = self.vision_config.unwrap_or(HfVision {
            patch_size: None,
            temporal_patch_size: None,
            n_channels: None,
            n_layers: None,
            decoder_dmodel: None,
            vision_encoder_type: None,
        });

        Ok(InklingConfig {
            model_type: self.model_type.unwrap_or_else(|| MODEL_TYPE.into()),
            text: InklingTextConfig {
                vocab_size: t.vocab_size.unwrap_or(201_024),
                unpadded_vocab_size: t.unpadded_vocab_size,
                hidden_size: hidden,
                num_hidden_layers: layers,
                num_attention_heads: t.num_attention_heads.unwrap_or(64),
                num_key_value_heads: t.num_key_value_heads.unwrap_or(8),
                head_dim: t.head_dim.unwrap_or(128),
                swa_num_attention_heads: t.swa_num_attention_heads.unwrap_or(64),
                swa_num_key_value_heads: t.swa_num_key_value_heads.unwrap_or(16),
                swa_head_dim: t.swa_head_dim.unwrap_or(128),
                sliding_window_size: t.sliding_window_size.unwrap_or(512),
                d_rel: t.d_rel.unwrap_or(16),
                rel_extent: t.rel_extent.unwrap_or(1024),
                log_scaling_n_floor: t.log_scaling_n_floor,
                log_scaling_alpha: t.log_scaling_alpha.unwrap_or(0.1),
                max_position_embeddings: t
                    .max_position_embeddings
                    .or(t.model_max_length)
                    .unwrap_or(131_072),
                rms_norm_eps: t.rms_norm_eps.unwrap_or(1e-6),
                conv_kernel_size: t.conv_kernel_size.or(t.sconv_kernel_size).unwrap_or(4),
                use_embed_norm: t.use_embed_norm.unwrap_or(true),
                dense_intermediate_size,
                moe_intermediate_size,
                n_routed_experts: t.n_routed_experts.unwrap_or(256),
                num_experts_per_tok: t.num_experts_per_tok.unwrap_or(6),
                n_shared_experts: t.n_shared_experts.unwrap_or(2),
                shared_expert_sink: t.shared_expert_sink.unwrap_or(true),
                route_scale: t.route_scale.unwrap_or(8.0),
                logits_mup_width_multiplier: t.logits_mup_width_multiplier.unwrap_or(24.0),
                dense_mlp_idx,
                local_layer_ids: local,
                layer_types,
                mlp_layer_types,
                num_mtp_layers: mtp.num_nextn_predict_layers.unwrap_or(0),
                mtp_local_layer_ids: mtp.local_layer_ids.unwrap_or_default(),
                eos_token_id: self.eos_token_id.unwrap_or(200_006),
            },
            audio: InklingAudioConfig {
                n_mel_bins: audio.n_mel_bins.unwrap_or(80),
                mel_vocab_size: audio.mel_vocab_size.unwrap_or(16),
                text_hidden_size: audio.decoder_dmodel.unwrap_or(hidden),
                dmel_min_value: audio.dmel_min_value.unwrap_or(-7.0),
                dmel_max_value: audio.dmel_max_value.unwrap_or(2.0),
                audio_mode: audio.audio_mode.unwrap_or_else(|| "dmel".into()),
            },
            vision: InklingVisionConfig {
                patch_size: vision.patch_size.unwrap_or(40),
                temporal_patch_size: vision.temporal_patch_size.unwrap_or(2),
                n_channels: vision.n_channels.unwrap_or(3),
                n_layers: vision.n_layers.unwrap_or(4),
                text_hidden_size: vision.decoder_dmodel.unwrap_or(hidden),
                vision_encoder_type: vision.vision_encoder_type.unwrap_or_else(|| "hmlp".into()),
            },
            image_token_id: self.image_token_id.unwrap_or(200_054),
            audio_token_id: self.audio_token_id.unwrap_or(200_053),
            image_bos_token_id: self.image_bos_token_id.unwrap_or(200_005),
            audio_bos_token_id: self.audio_bos_token_id.unwrap_or(200_020),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_production_shaped_json() {
        let preset = InklingConfig::production_preset();
        let json = r#"{
            "architectures": ["InklingForConditionalGeneration"],
            "model_type": "inkling_mm_model",
            "eos_token_id": 200006,
            "text_config": {
                "hidden_size": 6144,
                "num_hidden_layers": 66,
                "vocab_size": 201024,
                "unpadded_vocab_size": 200058,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "d_rel": 16,
                "rel_extent": 1024,
                "model_max_length": 1048576,
                "rms_norm_eps": 1e-6,
                "use_embed_norm": true,
                "local_layer_ids": [0,1,2,3,4,6],
                "dense_mlp_idx": 2,
                "sconv_kernel_size": 4,
                "swa_head_dim": 128,
                "swa_num_attention_heads": 64,
                "swa_num_key_value_heads": 16,
                "sliding_window_size": 512,
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
                "n_shared_experts": 2,
                "shared_expert_sink": true,
                "dense_intermediate_size": 24576,
                "intermediate_size": 3072,
                "route_scale": 8.0,
                "logits_mup_width_multiplier": 24.0,
                "log_scaling_n_floor": 128000,
                "log_scaling_alpha": 0.1
            },
            "audio_config": {
                "decoder_dmodel": 6144,
                "n_mel_bins": 80,
                "mel_vocab_size": 16,
                "dmel_min_value": -7.0,
                "dmel_max_value": 2.0,
                "audio_mode": "dmel"
            },
            "vision_config": {
                "vision_encoder_type": "hmlp",
                "decoder_dmodel": 6144,
                "patch_size": 40,
                "temporal_patch_size": 2,
                "n_channels": 3,
                "n_layers": 4
            },
            "mtp_config": {
                "num_nextn_predict_layers": 8,
                "local_layer_ids": [0, 2, 4, 5, 6, 7]
            }
        }"#;
        let cfg = InklingConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.text.hidden_size, preset.text.hidden_size);
        assert_eq!(cfg.text.dense_intermediate_size, 24_576);
        assert_eq!(cfg.text.moe_intermediate_size, 3072);
        assert_eq!(cfg.text.dense_mlp_idx, 2);
        assert!(cfg.text.is_dense_mlp(0));
        assert!(cfg.text.is_dense_mlp(1));
        assert!(!cfg.text.is_dense_mlp(2));
        assert!(cfg.text.is_sliding(0));
        assert!(!cfg.text.is_sliding(5)); // 5 not in shortened local list
        assert_eq!(cfg.text.num_mtp_layers, 8);
        assert_eq!(cfg.vision.patch_size, 40);
        assert_eq!(cfg.audio.mel_vocab_size, 16);
        assert_eq!(cfg.variant(), InklingVariant::Full);
    }

    #[test]
    fn small_variant_placeholder() {
        let small = InklingVariant::Small;
        assert!(!small.announced_params().weights_public);
        assert_eq!(small.announced_params().active_billions, 12.0);
        assert_eq!(small.hf_model_id(), HF_MODEL_ID_SMALL);
        let mut text = InklingConfig::production_preset().text;
        text.hidden_size = 4096; // stand-in until real Small config.json exists
        text.num_hidden_layers = 48;
        assert_eq!(
            InklingVariant::detect_from_text(&text),
            InklingVariant::Small
        );
    }
}

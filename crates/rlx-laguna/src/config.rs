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

//! HF `config.json` + GGUF metadata for Poolside Laguna.

use anyhow::{Context, Result, anyhow, bail};
use rlx_gguf::{GgufFile, MetaValue};
use std::path::Path;

pub const FAMILY: &str = "Laguna";
pub const HF_MODEL_ID: &str = "poolside/Laguna-S-2.1";
pub const HF_MODEL_ID_XS: &str = "poolside/Laguna-XS-2.1";
pub const HF_GGUF_REPO: &str = "unsloth/Laguna-S-2.1-GGUF";
pub const HF_GGUF_REPO_XS: &str = "poolside/Laguna-XS-2.1-GGUF";
pub const MODEL_TYPE: &str = "laguna";
pub const ARCHITECTURE: &str = "LagunaForCausalLM";
pub const GGUF_ARCH: &str = "laguna";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagunaVariant {
    /// Laguna S 2.1 — 118B-A8B.
    S21,
    /// Laguna XS 2.1 — 33B-A3B (same arch recipe).
    Xs21,
}

impl LagunaVariant {
    pub fn name(self) -> &'static str {
        match self {
            Self::S21 => "Laguna-S-2.1",
            Self::Xs21 => "Laguna-XS-2.1",
        }
    }

    pub fn hf_model_id(self) -> &'static str {
        match self {
            Self::S21 => HF_MODEL_ID,
            Self::Xs21 => HF_MODEL_ID_XS,
        }
    }

    pub fn hf_gguf_repo(self) -> &'static str {
        match self {
            Self::S21 => HF_GGUF_REPO,
            Self::Xs21 => HF_GGUF_REPO_XS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnLayerType {
    Full,
    Sliding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpLayerType {
    Dense,
    Sparse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnGating {
    Off,
    /// One softplus gate scalar per head (broadcast over head_dim).
    PerHead,
    /// One gate per (head, head_dim) channel.
    PerElement,
}

impl AttnGating {
    pub fn parse(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::Bool(false) => Self::Off,
            serde_json::Value::Bool(true) => Self::PerElement,
            serde_json::Value::String(s) if s.eq_ignore_ascii_case("per-head") => Self::PerHead,
            serde_json::Value::String(s) if s.eq_ignore_ascii_case("per-element") => {
                Self::PerElement
            }
            serde_json::Value::String(s) if s.eq_ignore_ascii_case("per_head") => Self::PerHead,
            _ => Self::PerHead,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RopeLayerParams {
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    /// YaRN `factor` (full layers); unused for plain RoPE.
    pub yarn_factor: f32,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub attention_factor: f32,
}

impl Default for RopeLayerParams {
    fn default() -> Self {
        Self {
            rope_type: "default".into(),
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            yarn_factor: 1.0,
            original_max_position_embeddings: 8192,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LagunaConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub moe_routed_scaling_factor: f32,
    pub moe_router_logit_softcapping: f32,
    pub sliding_window: usize,
    pub gating: AttnGating,
    pub layer_types: Vec<AttnLayerType>,
    pub mlp_layer_types: Vec<MlpLayerType>,
    /// Per-layer query head count (full vs SWA may differ).
    pub num_attention_heads_per_layer: Vec<usize>,
    pub rope_full: RopeLayerParams,
    pub rope_sliding: RopeLayerParams,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub tie_word_embeddings: bool,
}

impl LagunaConfig {
    pub fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .copied()
            .unwrap_or(AttnLayerType::Full)
            == AttnLayerType::Sliding
    }

    pub fn is_dense_mlp(&self, layer: usize) -> bool {
        self.mlp_layer_types
            .get(layer)
            .copied()
            .unwrap_or(MlpLayerType::Sparse)
            == MlpLayerType::Dense
    }

    pub fn n_heads(&self, layer: usize) -> usize {
        self.num_attention_heads_per_layer
            .get(layer)
            .copied()
            .unwrap_or(self.num_attention_heads)
    }

    pub fn rope_for_layer(&self, layer: usize) -> &RopeLayerParams {
        if self.is_sliding(layer) {
            &self.rope_sliding
        } else {
            &self.rope_full
        }
    }

    pub fn dense_lead_count(&self) -> usize {
        self.mlp_layer_types
            .iter()
            .take_while(|t| **t == MlpLayerType::Dense)
            .count()
    }

    pub fn variant(&self) -> LagunaVariant {
        if self.num_hidden_layers >= 48 && self.hidden_size >= 3072 {
            LagunaVariant::S21
        } else {
            LagunaVariant::Xs21
        }
    }

    /// Production Laguna S 2.1 dims from Hub `config.json`.
    pub fn production_s21() -> Self {
        let layers = 48usize;
        let layer_types: Vec<AttnLayerType> = (0..layers)
            .map(|i| {
                if i % 4 == 0 {
                    AttnLayerType::Full
                } else {
                    AttnLayerType::Sliding
                }
            })
            .collect();
        let mlp_layer_types: Vec<MlpLayerType> = (0..layers)
            .map(|i| {
                if i == 0 {
                    MlpLayerType::Dense
                } else {
                    MlpLayerType::Sparse
                }
            })
            .collect();
        let num_attention_heads_per_layer: Vec<usize> = (0..layers)
            .map(|i| if i % 4 == 0 { 48 } else { 72 })
            .collect();
        Self {
            model_type: MODEL_TYPE.into(),
            vocab_size: 100_352,
            hidden_size: 3072,
            intermediate_size: 12_288,
            num_hidden_layers: layers,
            num_attention_heads: 48,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 1_048_576,
            rms_norm_eps: 1e-6,
            num_experts: 256,
            num_experts_per_tok: 10,
            moe_intermediate_size: 1024,
            shared_expert_intermediate_size: 1024,
            norm_topk_prob: true,
            moe_routed_scaling_factor: 2.5,
            moe_router_logit_softcapping: 0.0,
            sliding_window: 512,
            gating: AttnGating::PerHead,
            layer_types,
            mlp_layer_types,
            num_attention_heads_per_layer,
            rope_full: RopeLayerParams {
                rope_type: "yarn".into(),
                rope_theta: 500_000.0,
                partial_rotary_factor: 0.5,
                yarn_factor: 128.0,
                original_max_position_embeddings: 8192,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: 1.485_203,
            },
            rope_sliding: RopeLayerParams {
                rope_type: "default".into(),
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                ..RopeLayerParams::default()
            },
            bos_token_id: 2,
            eos_token_id: 2,
            pad_token_id: 9,
            tie_word_embeddings: false,
        }
    }

    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("read {}", path.as_ref().display()))?;
        Self::from_json_str(&text)
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(text).context("parse Laguna config.json")?;
        Self::from_value(&v)
    }

    pub fn from_value(v: &serde_json::Value) -> Result<Self> {
        let u = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("missing config.{k}"))
        };
        let u_opt =
            |k: &str| -> Option<usize> { v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize) };
        let f = |k: &str, default: f32| -> f32 {
            v.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(default)
        };
        let model_type = v
            .get("model_type")
            .and_then(|x| x.as_str())
            .unwrap_or(MODEL_TYPE)
            .to_string();
        if model_type != MODEL_TYPE && model_type != "Laguna" {
            // Accept anyway if architectures claim Laguna.
            let arches = v
                .get("architectures")
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str())
                        .any(|s| s.contains("Laguna"))
                })
                .unwrap_or(false);
            if !arches {
                bail!("unexpected model_type={model_type} (expected laguna)");
            }
        }

        let layers = u("num_hidden_layers")?;
        let layer_types = parse_layer_types(v.get("layer_types"), layers);
        let mlp_layer_types = parse_mlp_types(v.get("mlp_layer_types"), layers, v);
        let heads_default = u("num_attention_heads")?;
        let num_attention_heads_per_layer = match v.get("num_attention_heads_per_layer") {
            Some(serde_json::Value::Array(a)) if a.len() == layers => a
                .iter()
                .map(|x| x.as_u64().map(|u| u as usize).unwrap_or(heads_default))
                .collect(),
            _ => vec![heads_default; layers],
        };
        let gating = v
            .get("gating")
            .map(AttnGating::parse)
            .unwrap_or(AttnGating::PerHead);

        let rope_full = parse_rope(
            v.get("rope_parameters")
                .and_then(|r| r.get("full_attention")),
            RopeLayerParams {
                rope_type: "yarn".into(),
                rope_theta: 500_000.0,
                partial_rotary_factor: 0.5,
                yarn_factor: 128.0,
                original_max_position_embeddings: 8192,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: 1.0,
            },
        );
        let rope_sliding = parse_rope(
            v.get("rope_parameters")
                .and_then(|r| r.get("sliding_attention")),
            RopeLayerParams::default(),
        );

        let eos = match v.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => a
                .first()
                .and_then(|x| x.as_u64())
                .map(|u| u as u32)
                .unwrap_or(2),
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(2) as u32,
            _ => 2,
        };

        Ok(Self {
            model_type,
            vocab_size: u("vocab_size")?,
            hidden_size: u("hidden_size")?,
            intermediate_size: u("intermediate_size")?,
            num_hidden_layers: layers,
            num_attention_heads: heads_default,
            num_key_value_heads: u("num_key_value_heads")?,
            head_dim: u_opt("head_dim").unwrap_or(128),
            max_position_embeddings: u("max_position_embeddings")?,
            rms_norm_eps: f("rms_norm_eps", 1e-6),
            num_experts: u_opt("num_experts").unwrap_or(0),
            num_experts_per_tok: u_opt("num_experts_per_tok").unwrap_or(0),
            moe_intermediate_size: u_opt("moe_intermediate_size").unwrap_or(0),
            shared_expert_intermediate_size: u_opt("shared_expert_intermediate_size")
                .unwrap_or_else(|| u_opt("moe_intermediate_size").unwrap_or(0)),
            norm_topk_prob: v
                .get("norm_topk_prob")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            moe_routed_scaling_factor: f("moe_routed_scaling_factor", 1.0),
            moe_router_logit_softcapping: f("moe_router_logit_softcapping", 0.0),
            sliding_window: u_opt("sliding_window").unwrap_or(0),
            gating,
            layer_types,
            mlp_layer_types,
            num_attention_heads_per_layer,
            rope_full,
            rope_sliding,
            bos_token_id: u_opt("bos_token_id").unwrap_or(2) as u32,
            eos_token_id: eos,
            pad_token_id: u_opt("pad_token_id").unwrap_or(9) as u32,
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        })
    }

    /// Parse Unsloth / llama.cpp GGUF metadata (`general.architecture = laguna`).
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow!("missing general.architecture"))?;
        if arch != GGUF_ARCH {
            bail!("expected general.architecture=laguna, got {arch}");
        }
        let lookup = |k: &str| -> Option<&MetaValue> { raw.metadata.get(&format!("laguna.{k}")) };
        let u32k = |k: &str| -> Result<u32> {
            lookup(k)
                .and_then(MetaValue::as_u32)
                .ok_or_else(|| anyhow!("missing laguna.{k}"))
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
        let dense_lead = u32k_opt("leading_dense_block_count").unwrap_or(1) as usize;
        let swa = u32k_opt("attention.sliding_window").unwrap_or(0) as usize;
        let sw_pattern = bool_arr("attention.sliding_window_pattern");
        // llama.cpp stores SWA pattern as period (FULL first) when array absent.
        let layer_types: Vec<AttnLayerType> = (0..layers)
            .map(|i| {
                let sliding = if !sw_pattern.is_empty() {
                    sw_pattern.get(i).copied().unwrap_or(false)
                } else if swa > 0 {
                    i % 4 != 0
                } else {
                    false
                };
                if sliding {
                    AttnLayerType::Sliding
                } else {
                    AttnLayerType::Full
                }
            })
            .collect();
        let mlp_layer_types: Vec<MlpLayerType> = (0..layers)
            .map(|i| {
                if i < dense_lead {
                    MlpLayerType::Dense
                } else {
                    MlpLayerType::Sparse
                }
            })
            .collect();
        // XS/S GGUFs store per-layer head counts as an array; some fixtures use a scalar.
        let heads_per = u32_arr("attention.head_count");
        let heads = if !heads_per.is_empty() {
            heads_per
                .iter()
                .copied()
                .zip(layer_types.iter())
                .find(|(_, t)| **t == AttnLayerType::Full)
                .map(|(h, _)| h)
                .unwrap_or(heads_per[0])
        } else {
            u32k("attention.head_count")? as usize
        };
        let num_attention_heads_per_layer = if heads_per.len() == layers {
            heads_per
        } else {
            (0..layers)
                .map(|i| {
                    if layer_types[i] == AttnLayerType::Full {
                        heads
                    } else if heads == 48 {
                        // Scalar-only fallback: XS SWA uses 64 (S uses 72). Real
                        // Laguna GGUFs always ship the per-layer array.
                        64
                    } else {
                        heads
                    }
                })
                .collect()
        };

        let rope_theta = f32k("rope.freq_base").unwrap_or(500_000.0);
        let rope_theta_swa = f32k("rope.freq_base_swa").unwrap_or(10_000.0);
        let n_rot = u32k_opt("rope.dimension_count").unwrap_or(64) as usize;
        let n_rot_swa = u32k_opt("rope.dimension_count_swa").unwrap_or(128) as usize;
        let head_dim = u32k_opt("attention.key_length").unwrap_or(128) as usize;

        let norm_topk_prob = match lookup("expert_weights_norm") {
            Some(MetaValue::Bool(b)) => *b,
            Some(MetaValue::F32(x)) => *x != 0.0,
            Some(MetaValue::U32(u)) => *u != 0,
            _ => true,
        };

        Ok(Self {
            model_type: MODEL_TYPE.into(),
            vocab_size: raw
                .metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|v| match v {
                    MetaValue::Array(a) => Some(a.len()),
                    _ => None,
                })
                .or_else(|| u32k_opt("vocab_size").map(|u| u as usize))
                .unwrap_or(100_352),
            hidden_size: u32k("embedding_length")? as usize,
            intermediate_size: u32k("feed_forward_length")? as usize,
            num_hidden_layers: layers,
            num_attention_heads: heads,
            num_key_value_heads: u32k("attention.head_count_kv")? as usize,
            head_dim,
            max_position_embeddings: u32k_opt("context_length").unwrap_or(1_048_576) as usize,
            rms_norm_eps: f32k("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            num_experts: u32k("expert_count")? as usize,
            num_experts_per_tok: u32k("expert_used_count")? as usize,
            moe_intermediate_size: u32k("expert_feed_forward_length")? as usize,
            shared_expert_intermediate_size: u32k_opt("expert_shared_feed_forward_length")
                .unwrap_or_else(|| u32k_opt("expert_feed_forward_length").unwrap_or(0))
                as usize,
            norm_topk_prob,
            moe_routed_scaling_factor: f32k("expert_weights_scale").unwrap_or(1.0),
            moe_router_logit_softcapping: 0.0,
            sliding_window: swa,
            gating: AttnGating::PerHead,
            layer_types,
            mlp_layer_types,
            num_attention_heads_per_layer,
            rope_full: RopeLayerParams {
                rope_type: "yarn".into(),
                rope_theta,
                partial_rotary_factor: (n_rot as f32 / head_dim as f32).clamp(0.0, 1.0),
                yarn_factor: f32k("rope.scaling.factor").unwrap_or(128.0),
                original_max_position_embeddings: u32k_opt("rope.scaling.original_context_length")
                    .unwrap_or(8192) as usize,
                beta_fast: f32k("rope.scaling.yarn_beta_fast")
                    .or_else(|| f32k("rope.scaling.beta_fast"))
                    .unwrap_or(32.0),
                beta_slow: f32k("rope.scaling.yarn_beta_slow")
                    .or_else(|| f32k("rope.scaling.beta_slow"))
                    .unwrap_or(1.0),
                attention_factor: f32k("rope.scaling.yarn_attn_factor")
                    .or_else(|| f32k("rope.scaling.attn_factor"))
                    .unwrap_or(1.0),
            },
            rope_sliding: RopeLayerParams {
                rope_type: "default".into(),
                rope_theta: rope_theta_swa,
                partial_rotary_factor: (n_rot_swa as f32 / head_dim as f32).clamp(0.0, 1.0),
                ..RopeLayerParams::default()
            },
            bos_token_id: raw
                .metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(MetaValue::as_u32)
                .unwrap_or(2),
            eos_token_id: raw
                .metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(MetaValue::as_u32)
                .unwrap_or(2),
            pad_token_id: 9,
            tie_word_embeddings: false,
        })
    }

    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // Header-only: never slurp Q4/IQ tensor payloads into RSS.
        let raw = crate::memory::open_gguf_header_only(path)?;
        Self::from_gguf(&raw).with_context(|| format!("rlx-laguna: parse {}", path.display()))
    }
}

fn parse_layer_types(v: Option<&serde_json::Value>, layers: usize) -> Vec<AttnLayerType> {
    match v {
        Some(serde_json::Value::Array(a)) if !a.is_empty() => a
            .iter()
            .map(|x| match x.as_str().unwrap_or("") {
                "sliding_attention" | "sliding" => AttnLayerType::Sliding,
                _ => AttnLayerType::Full,
            })
            .collect(),
        _ => (0..layers)
            .map(|i| {
                if i % 4 == 0 {
                    AttnLayerType::Full
                } else {
                    AttnLayerType::Sliding
                }
            })
            .collect(),
    }
}

fn parse_mlp_types(
    v: Option<&serde_json::Value>,
    layers: usize,
    root: &serde_json::Value,
) -> Vec<MlpLayerType> {
    if let Some(serde_json::Value::Array(a)) = v {
        if !a.is_empty() {
            return a
                .iter()
                .map(|x| match x.as_str().unwrap_or("") {
                    "dense" => MlpLayerType::Dense,
                    _ => MlpLayerType::Sparse,
                })
                .collect();
        }
    }
    let lead = root
        .get("mlp_only_layers")
        .and_then(|x| x.as_array())
        .map(|a| a.len())
        .or_else(|| {
            root.get("decoder_sparse_step")
                .and_then(|x| x.as_u64())
                .map(|_| 1)
        })
        .unwrap_or(1);
    (0..layers)
        .map(|i| {
            if i < lead {
                MlpLayerType::Dense
            } else {
                MlpLayerType::Sparse
            }
        })
        .collect()
}

fn parse_rope(v: Option<&serde_json::Value>, defaults: RopeLayerParams) -> RopeLayerParams {
    let Some(v) = v else {
        return defaults;
    };
    let f = |k: &str, d: f32| -> f32 {
        v.get(k)
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(d)
    };
    let u = |k: &str, d: usize| -> usize {
        v.get(k)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d)
    };
    RopeLayerParams {
        rope_type: v
            .get("rope_type")
            .and_then(|x| x.as_str())
            .unwrap_or(&defaults.rope_type)
            .to_string(),
        rope_theta: f("rope_theta", defaults.rope_theta),
        partial_rotary_factor: f("partial_rotary_factor", defaults.partial_rotary_factor),
        yarn_factor: f("factor", defaults.yarn_factor),
        original_max_position_embeddings: u(
            "original_max_position_embeddings",
            defaults.original_max_position_embeddings,
        ),
        beta_fast: f("beta_fast", defaults.beta_fast),
        beta_slow: f("beta_slow", defaults.beta_slow),
        attention_factor: f("attention_factor", defaults.attention_factor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_s21_shape() {
        let c = LagunaConfig::production_s21();
        assert_eq!(c.num_hidden_layers, 48);
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 10);
        assert!(c.is_dense_mlp(0));
        assert!(!c.is_dense_mlp(1));
        assert!(!c.is_sliding(0));
        assert!(c.is_sliding(1));
        assert_eq!(c.n_heads(0), 48);
        assert_eq!(c.n_heads(1), 72);
        assert_eq!(c.gating, AttnGating::PerHead);
        assert_eq!(c.variant(), LagunaVariant::S21);
    }

    #[test]
    fn parse_minimal_json() {
        let j = r#"{
            "model_type": "laguna",
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 128,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "gating": "per-head",
            "sliding_window": 8,
            "layer_types": ["full_attention","sliding_attention","sliding_attention","sliding_attention"],
            "mlp_layer_types": ["dense","sparse","sparse","sparse"],
            "num_attention_heads_per_layer": [4,6,6,6],
            "moe_routed_scaling_factor": 2.5
        }"#;
        let c = LagunaConfig::from_json_str(j).unwrap();
        assert_eq!(c.n_heads(1), 6);
        assert!(c.is_dense_mlp(0));
        assert_eq!(c.moe_routed_scaling_factor, 2.5);
        assert_eq!(c.gating, AttnGating::PerHead);
    }
}

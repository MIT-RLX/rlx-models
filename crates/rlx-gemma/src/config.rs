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

//! Gemma family configuration — HF `config.json` and GGUF metadata.

use rlx_flow::blocks::{GemmaLayerStyle, gemma_strided_layer_mask, gemma2_layer_mask};
use rlx_gguf::{GgufFile, MetaValue};
use rlx_ir::op::MaskKind;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GemmaArch {
    #[default]
    Gemma,
    Gemma2,
    Gemma3,
    Gemma4,
}

impl GemmaArch {
    pub fn sliding_window_stride(self) -> usize {
        match self {
            GemmaArch::Gemma3 | GemmaArch::Gemma4 => 6,
            _ => 0,
        }
    }

    fn from_gguf_tag(tag: &str) -> Self {
        match tag {
            "gemma2" => GemmaArch::Gemma2,
            "gemma3" | "gemma3n" => GemmaArch::Gemma3,
            "gemma4" | "gemma4moe" | "gemma4_unified" | "gemma4_unified_text" => GemmaArch::Gemma4,
            _ => GemmaArch::Gemma,
        }
    }
}

/// One entry in the Gemma 4 `text_config.layer_types` array. The
/// repeating "5 sliding + 1 full" Gemma 3 pattern is just a special
/// case of this richer per-layer schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmaLayerType {
    SlidingAttention,
    FullAttention,
}

/// Nested rope_parameters block. Gemma 4 12B carries per-attention-kind
/// rope parameters: sliding layers use `theta=1e4` with full rotation,
/// full-attention layers use `theta=1e6` with `partial_rotary_factor`
/// (p-RoPE rotating only the leading slice).
#[derive(Debug, Clone, Copy, Deserialize, Default)]
pub struct GemmaRopeParameters {
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_type: Option<GemmaRopeKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GemmaRopeKind {
    #[default]
    Default,
    Proportional,
    Linear,
    Dynamic,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GemmaRopeMap {
    #[serde(default)]
    pub sliding_attention: Option<GemmaRopeParameters>,
    #[serde(default)]
    pub full_attention: Option<GemmaRopeParameters>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GemmaConfig {
    #[serde(default)]
    pub arch: GemmaArch,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    #[serde(default)]
    pub effective_num_layers: Option<usize>,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_used: usize,
    #[serde(default)]
    pub expert_ffn_size: usize,
    #[serde(default = "default_expert_weights_scale")]
    pub expert_weights_scale: f32,

    // ── Gemma 4 unified additions ──────────────────────────────────
    /// Per-layer attention kind. Empty for Gemma <=3 — fall back to
    /// the strided pattern derived from `arch.sliding_window_stride`.
    #[serde(default)]
    pub layer_types: Vec<GemmaLayerType>,
    /// Per-attention-kind rope settings. Empty for Gemma <=3.
    #[serde(default)]
    pub rope_parameters: GemmaRopeMap,
    /// Head dim for full-attention (global) layers. `None` ⇒ reuse
    /// the base `head_dim`. Gemma 4 12B sets this to 512 while the
    /// sliding `head_dim` stays at 256.
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    /// Num KV heads for full-attention layers. `None` ⇒ reuse the
    /// base `num_key_value_heads`. Gemma 4 12B sets this to 1.
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    /// When true (Gemma 4 12B), the K projection is reused as V at
    /// load time — weights only ship `.k_proj` and `.v_proj` becomes
    /// an alias.
    #[serde(default)]
    pub attention_k_eq_v: bool,
    /// When `"vision"`, media placeholder spans use bidirectional
    /// attention on sliding layers (Gemma 4 unified).
    #[serde(default)]
    pub use_bidirectional_attention: Option<String>,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_expert_weights_scale() -> f32 {
    1.0
}

impl GemmaConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        // Gemma 4 unified (e.g. `google/gemma-4-12B`) nests the LM
        // hyperparameters under `text_config` because the same file
        // also carries vision + audio configs. Pick that subtree if
        // it looks like the unified shape, otherwise stay flat.
        let value: serde_json::Value = serde_json::from_str(&data)?;
        let lm_value = match value.get("text_config") {
            Some(tc) if tc.is_object() => tc.clone(),
            _ => value.clone(),
        };
        let lm_value = normalize_hf_null_usize_fields(lm_value);
        let mut cfg: Self = serde_json::from_value(lm_value)?;
        if cfg.arch == GemmaArch::Gemma {
            cfg.arch = infer_arch_from_json(&data);
        }
        Ok(cfg)
    }

    pub fn from_gguf(raw: &GgufFile) -> anyhow::Result<Self> {
        gemma_cfg_from_gguf(raw)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    pub fn kv_proj_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    pub fn layer_style(&self) -> GemmaLayerStyle {
        match self.arch {
            GemmaArch::Gemma => GemmaLayerStyle::Gemma,
            GemmaArch::Gemma2 => GemmaLayerStyle::Gemma2,
            GemmaArch::Gemma3 => GemmaLayerStyle::Gemma3,
            GemmaArch::Gemma4 => GemmaLayerStyle::Gemma4,
        }
    }

    pub fn active_num_layers(&self) -> usize {
        self.effective_num_layers.unwrap_or(self.num_hidden_layers)
    }

    pub fn is_moe(&self) -> bool {
        self.arch == GemmaArch::Gemma4 && self.num_experts > 0
    }

    /// Gemma 4 unified: bidirectional attention inside vision/audio spans.
    pub fn use_bidirectional_vision(&self) -> bool {
        self.use_bidirectional_attention.as_deref() == Some("vision")
    }

    pub fn expert_ffn_dim(&self) -> usize {
        if self.expert_ffn_size > 0 {
            self.expert_ffn_size
        } else {
            self.intermediate_size
        }
    }

    pub fn attn_score_scale(&self) -> Option<f32> {
        match self.arch {
            GemmaArch::Gemma => None,
            GemmaArch::Gemma2 | GemmaArch::Gemma3 | GemmaArch::Gemma4 => {
                if let Some(s) = self.query_pre_attn_scalar {
                    Some(1.0 / s)
                } else {
                    Some(1.0 / (self.head_dim() as f32).sqrt())
                }
            }
        }
    }

    /// Per-layer attention options driving the prefill self-attn block:
    /// `(mask kind, softmax score scale, attention logit soft-cap)`.
    /// The mask varies across Gemma variants:
    ///
    /// - Gemma 1 / no sliding window → all-causal.
    /// - Gemma 2 → alternating sliding-window via [`gemma2_layer_mask`].
    /// - Gemma 3 / 4 → strided pattern via
    ///   [`gemma_strided_layer_mask`] (stride-6: every 6th layer is
    ///   full causal, others are sliding-window).
    pub fn layer_attn_options(&self, layer: usize) -> (MaskKind, Option<f32>, Option<f32>) {
        let scale = self.attn_score_scale();
        let softcap = self.attn_logit_softcapping;
        let mask = match (self.arch, self.sliding_window) {
            (_, None) => MaskKind::Causal,
            (GemmaArch::Gemma2, Some(w)) => gemma2_layer_mask(layer, w),
            (GemmaArch::Gemma3 | GemmaArch::Gemma4, Some(w)) => {
                gemma_strided_layer_mask(layer, w, self.arch.sliding_window_stride())
            }
            _ => MaskKind::Causal,
        };
        (mask, scale, softcap)
    }

    #[cfg(test)]
    pub(crate) fn tiny_test() -> Self {
        Self {
            arch: GemmaArch::Gemma,
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            head_dim: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            sliding_window: None,
            query_pre_attn_scalar: None,
            effective_num_layers: None,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            expert_weights_scale: 1.0,
            layer_types: Vec::new(),
            rope_parameters: GemmaRopeMap::default(),
            global_head_dim: None,
            num_global_key_value_heads: None,
            attention_k_eq_v: false,
            use_bidirectional_attention: None,
        }
    }

    // ── Per-layer dispatch (Gemma 4 unified). ──────────────────────
    //
    // For Gemma 1/2/3 the `layer_types` array is empty and these
    // helpers reduce to the existing strided pattern; for Gemma 4
    // they read the explicit array so each layer can ship its own
    // (head_dim, num_kv_heads, n_rot, rope_theta).

    /// Whether layer `i` is a full-attention (global) layer rather
    /// than a sliding-window one. Falls back to the strided pattern
    /// (every `stride`-th layer is global) when `layer_types` is
    /// unset.
    pub fn is_full_attention_layer(&self, layer: usize) -> bool {
        if !self.layer_types.is_empty() {
            return matches!(
                self.layer_types.get(layer),
                Some(GemmaLayerType::FullAttention),
            );
        }
        let stride = self.arch.sliding_window_stride();
        stride > 1 && (layer + 1).is_multiple_of(stride)
    }

    /// Per-layer head_dim. Sliding layers always use the base
    /// `head_dim`; full-attention layers use `global_head_dim` when
    /// set (Gemma 4 12B: 512 vs base 256).
    pub fn layer_head_dim(&self, layer: usize) -> usize {
        if self.is_full_attention_layer(layer) {
            self.global_head_dim.unwrap_or_else(|| self.head_dim())
        } else {
            self.head_dim()
        }
    }

    /// Per-layer KV head count. Sliding layers use
    /// `num_key_value_heads`; full-attention layers use
    /// `num_global_key_value_heads` when set (Gemma 4 12B: 1 vs 8).
    pub fn layer_num_kv_heads(&self, layer: usize) -> usize {
        if self.is_full_attention_layer(layer) {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    /// Number of leading per-head dimensions that get RoPE-rotated
    /// in layer `i`. Returns `layer_head_dim` for "default" RoPE,
    /// or `floor(partial_rotary_factor * head_dim)` for p-RoPE.
    pub fn layer_n_rot(&self, layer: usize) -> usize {
        let dh = self.layer_head_dim(layer);
        let params = self.layer_rope_parameters(layer);
        let kind = params
            .and_then(|p| p.rope_type)
            .unwrap_or(GemmaRopeKind::Default);
        let factor = params.and_then(|p| p.partial_rotary_factor);
        match (kind, factor) {
            (GemmaRopeKind::Proportional, Some(f)) if f > 0.0 && f < 1.0 => {
                ((dh as f32) * f).floor() as usize
            }
            _ => dh,
        }
    }

    /// RoPE base frequency for layer `i`. Falls back to the
    /// top-level `rope_theta` when the unified map omits the entry.
    pub fn layer_rope_theta(&self, layer: usize) -> f64 {
        self.layer_rope_parameters(layer)
            .and_then(|p| p.rope_theta)
            .map(|t| t as f64)
            .unwrap_or(self.rope_theta)
    }

    fn layer_rope_parameters(&self, layer: usize) -> Option<&GemmaRopeParameters> {
        if self.is_full_attention_layer(layer) {
            self.rope_parameters.full_attention.as_ref()
        } else {
            self.rope_parameters.sliding_attention.as_ref()
        }
    }
}

/// HF dense Gemma 4 checkpoints use JSON `null` for unused MoE keys.
fn normalize_hf_null_usize_fields(mut value: serde_json::Value) -> serde_json::Value {
    let Some(obj) = value.as_object_mut() else {
        return value;
    };
    for key in [
        "num_experts",
        "num_experts_used",
        "top_k_experts",
        "expert_ffn_size",
        "moe_intermediate_size",
        "hidden_size_per_layer_input",
    ] {
        if obj.get(key).is_some_and(|v| v.is_null()) {
            obj.insert(key.to_string(), serde_json::Value::from(0usize));
        }
    }
    value
}

fn infer_arch_from_json(raw: &str) -> GemmaArch {
    // Detect Gemma 4 first — its unified config also contains a
    // nested `gemma4_unified_text` model_type that we want to catch
    // even when the outer `model_type` is `gemma4_unified` or the
    // architecture is `Gemma4UnifiedForConditionalGeneration`.
    if raw.contains("\"gemma4_unified\"")
        || raw.contains("\"gemma4_unified_text\"")
        || raw.contains("\"gemma4\"")
        || raw.contains("\"gemma4moe\"")
        || raw.contains("Gemma4UnifiedForConditionalGeneration")
        || raw.contains("Gemma4ForCausalLM")
    {
        return GemmaArch::Gemma4;
    }
    if raw.contains("\"model_type\"") {
        if raw.contains("\"gemma2\"") {
            return GemmaArch::Gemma2;
        }
        if raw.contains("\"gemma3\"") {
            return GemmaArch::Gemma3;
        }
    }
    GemmaArch::Gemma
}

pub fn gemma_cfg_from_gguf(raw: &GgufFile) -> anyhow::Result<GemmaConfig> {
    let arch_tag = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("gemma");
    let arch_prefix = arch_tag;
    let arch = GemmaArch::from_gguf_tag(arch_tag);

    let get_meta = |k: &str| -> Option<&MetaValue> {
        raw.metadata.get(k).or_else(|| {
            let suffix = k.strip_prefix("gemma.")?;
            if arch_prefix == "gemma" {
                None
            } else {
                let arch_key = format!("{arch_prefix}.{suffix}");
                raw.metadata.get(&arch_key)
            }
        })
    };
    let get_u32 = |k: &str| -> anyhow::Result<u32> {
        get_meta(k)
            .and_then(MetaValue::as_u32)
            .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata key: {k}"))
    };
    let get_f32 = |k: &str| -> Option<f32> {
        get_meta(k).and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    };
    let get_bool = |k: &str| -> Option<bool> {
        get_meta(k).and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
    };

    let hidden_size = get_u32("gemma.embedding_length")? as usize;
    let num_attention_heads = get_u32("gemma.attention.head_count")? as usize;
    let head_dim = get_u32("gemma.attention.key_length")
        .ok()
        .or_else(|| get_u32("gemma.rope.dimension_count").ok())
        .map(|v| v as usize);

    Ok(GemmaConfig {
        arch,
        vocab_size: get_u32("gemma.vocab_size").unwrap_or(256_000) as usize,
        hidden_size,
        intermediate_size: get_u32("gemma.feed_forward_length")? as usize,
        num_hidden_layers: get_u32("gemma.block_count")? as usize,
        num_attention_heads,
        num_key_value_heads: get_u32("gemma.attention.head_count_kv")? as usize,
        max_position_embeddings: get_u32("gemma.context_length").unwrap_or(8192) as usize,
        rms_norm_eps: get_f32("gemma.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64,
        rope_theta: get_f32("gemma.rope.freq_base").unwrap_or(10_000.0) as f64,
        tie_word_embeddings: get_bool("gemma.tie_word_embeddings").unwrap_or(true),
        attention_bias: get_bool("gemma.attention.bias").unwrap_or(false),
        head_dim,
        attn_logit_softcapping: get_f32("gemma.attn_logit_softcapping"),
        final_logit_softcapping: get_f32("gemma.final_logit_softcapping"),
        sliding_window: get_u32("gemma.attention.sliding_window")
            .ok()
            .map(|v| v as usize),
        query_pre_attn_scalar: get_f32("gemma.attention.query_pre_attn_scalar"),
        effective_num_layers: get_u32("gemma.block_count_effective")
            .ok()
            .map(|v| v as usize),
        num_experts: get_u32("gemma.expert_count").unwrap_or(0) as usize,
        num_experts_used: get_u32("gemma.expert_used_count").unwrap_or(0) as usize,
        expert_ffn_size: get_u32("gemma.expert_feed_forward_length").unwrap_or(0) as usize,
        expert_weights_scale: get_f32("gemma.expert_weights_scale").unwrap_or(1.0),
        // GGUF doesn't carry the Gemma 4 unified per-layer schema
        // yet; the dense path falls back to the strided pattern and
        // uniform head dims that match every Gemma 4 GGUF currently
        // emitted by llama.cpp.
        layer_types: Vec::new(),
        rope_parameters: GemmaRopeMap::default(),
        global_head_dim: None,
        num_global_key_value_heads: None,
        attention_k_eq_v: false,
        use_bidirectional_attention: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed copy of `google/gemma-4-12B`'s `config.json` — only the
    /// fields the loader actually consumes plus the surrounding shape
    /// (top-level `model_type`, nested `text_config`) that proves we
    /// unwrap the unified layout correctly.
    const GEMMA_4_12B_CONFIG: &str = r#"{
      "architectures": ["Gemma4UnifiedForConditionalGeneration"],
      "model_type": "gemma4_unified",
      "tie_word_embeddings": true,
      "text_config": {
        "model_type": "gemma4_unified_text",
        "vocab_size": 262144,
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "attention_k_eq_v": true,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true,
        "attention_bias": false,
        "final_logit_softcapping": 30.0,
        "sliding_window": 1024,
        "layer_types": [
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
        ],
        "rope_parameters": {
          "full_attention":    { "partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional" },
          "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
        }
      }
    }"#;

    #[test]
    fn gemma_4_12b_unified_config_parses_text_subtree() {
        let dir = std::env::temp_dir();
        let path = dir.join("rlx_gemma_gemma4_12b_test_config.json");
        std::fs::write(&path, GEMMA_4_12B_CONFIG).unwrap();
        let cfg = GemmaConfig::from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(cfg.arch, GemmaArch::Gemma4);
        assert_eq!(cfg.vocab_size, 262_144);
        assert_eq!(cfg.hidden_size, 3840);
        assert_eq!(cfg.intermediate_size, 15_360);
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim(), 256);
        assert_eq!(cfg.global_head_dim, Some(512));
        assert_eq!(cfg.num_global_key_value_heads, Some(1));
        assert!(cfg.attention_k_eq_v);
        assert_eq!(cfg.sliding_window, Some(1024));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.layer_types.len(), 48);
        // Stride-6 sliding-window pattern carried over from Gemma 3.
        assert_eq!(cfg.arch.sliding_window_stride(), 6);
    }

    #[test]
    fn hf_null_moe_fields_default_to_zero() {
        let json = r#"{"num_experts": null, "top_k_experts": null}"#;
        let v = normalize_hf_null_usize_fields(serde_json::from_str(json).unwrap());
        let obj = v.as_object().unwrap();
        assert_eq!(obj["num_experts"], 0);
        assert_eq!(obj["top_k_experts"], 0);
    }

    #[test]
    fn gemma_4_12b_per_layer_dispatch() {
        let dir = std::env::temp_dir();
        let path = dir.join("rlx_gemma_gemma4_12b_dispatch_config.json");
        std::fs::write(&path, GEMMA_4_12B_CONFIG).unwrap();
        let cfg = GemmaConfig::from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // Sliding layer 0 — base shapes + full rotary on theta=1e4.
        assert!(!cfg.is_full_attention_layer(0));
        assert_eq!(cfg.layer_head_dim(0), 256);
        assert_eq!(cfg.layer_num_kv_heads(0), 8);
        assert_eq!(cfg.layer_n_rot(0), 256);
        assert!((cfg.layer_rope_theta(0) - 10_000.0).abs() < 1e-3);

        // Full-attention layer 5 (1-indexed: 6th layer) — global
        // shapes, p-RoPE (0.25 of head_dim_full=512 → 128), theta=1e6.
        assert!(cfg.is_full_attention_layer(5));
        assert_eq!(cfg.layer_head_dim(5), 512);
        assert_eq!(cfg.layer_num_kv_heads(5), 1);
        assert_eq!(cfg.layer_n_rot(5), 128);
        assert!((cfg.layer_rope_theta(5) - 1_000_000.0).abs() < 1e-3);

        // Last layer (index 47, 1-indexed 48) is also full-attention.
        assert!(cfg.is_full_attention_layer(47));
    }

    #[test]
    fn pre_gemma4_archs_keep_uniform_layer_shape() {
        // Without `layer_types` / `rope_parameters` the per-layer
        // accessors collapse to the base values so Gemma 3 / 2 / 1
        // continue to round-trip the existing flow.
        let mut cfg = GemmaConfig::tiny_test();
        cfg.arch = GemmaArch::Gemma3;
        cfg.head_dim = Some(64);
        cfg.num_key_value_heads = 2;
        cfg.rope_theta = 1_000.0;
        for i in 0..cfg.num_hidden_layers {
            assert_eq!(cfg.layer_head_dim(i), 64);
            assert_eq!(cfg.layer_num_kv_heads(i), 2);
            assert_eq!(cfg.layer_n_rot(i), 64);
            assert!((cfg.layer_rope_theta(i) - 1_000.0).abs() < 1e-3);
        }
    }

    #[test]
    fn infer_arch_picks_up_gemma4_markers() {
        assert_eq!(
            infer_arch_from_json(r#"{"model_type":"gemma4_unified"}"#),
            GemmaArch::Gemma4,
        );
        assert_eq!(
            infer_arch_from_json(r#"{"architectures":["Gemma4UnifiedForConditionalGeneration"]}"#),
            GemmaArch::Gemma4,
        );
        assert_eq!(
            infer_arch_from_json(r#"{"model_type":"gemma3"}"#),
            GemmaArch::Gemma3,
        );
    }
}
